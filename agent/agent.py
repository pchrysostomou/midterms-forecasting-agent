"""Forecast the Future 2026 - AI-only agent.

Pipeline (each stage builds on the one before; every model call goes through ftf.ask
with a role tag, every intermediate result is saved to out/run-<n>/):

  0. data       (code, no model)  market snapshots, typical moves, past-election moves
  1. research   (role=research)   one general brief + one web-search brief per question
  2. forecast   (role=forecast)   three independent models from different families, BLIND to
                                  the market price; the median is the agent's own estimate
  3. aggregate  (role=aggregate)  one cross-question consistency check (Senate/House
                                  control vs the individual races; Part B scenario moves)
  4. critique   (role=critique)   a reviewer that is not the final model compares own estimate
                                  with the market and the evidence, and sets how much weight it deserves
  5. anchor     (code, no model)  Part A: market + w * (bias-adjusted own - market), capped; w = 0
                                  unless the critique names a reason the code can check
                                  Part B: past-election baseline + kept share of the
                                  forecasters' adjustment to it, capped
  6. final      (role=final)      sees everything, answers the anchor, ends with the JSON line

Usage (from the repo root, after `git commit`):
    python agent/agent.py --run test --only A1,B4      # cheap smoke test
    python agent/agent.py --run 1                      # full run, then a dry-run submit
    python agent/agent.py --run 1 --submit             # full run and real submission
Re-running the same --run resumes: finished stages are read back from out/run-<n>/, including finals
that already have a readable answer, so only missing or unreadable questions get a new final call.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_only"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The helper reads the git commit at import and writes its run log (.ftf/) and answers/ in the folder it runs
# from, so always run from the repo root, whatever folder the command was typed in.
os.chdir(ROOT)

import data  # noqa: E402
import ftf  # noqa: E402

try:
    import ftf_tools  # live data through the data pack's own fetchers
except Exception as e:  # noqa: BLE001
    print(f"ftf_tools unavailable ({e}); using data pack files only")
    ftf_tools = None

# ----------------------------------------------------------------------------- config
M = ftf.APPROVED_MODELS
MODELS = {
    "research": M["Claude Sonnet 5"],        # web search through OpenRouter (online=True)
    "forecast_a": M["Claude Sonnet 5"],      # forecaster 1
    "forecast_b": M["GPT-6 Sol"],            # forecaster 2 (different model family)
    "forecast_c": M["Gemini 4 Argon"],       # forecaster 3; the median of three resists one outlier
    "aggregate": M["Claude Sonnet 5"],
    "critique": M["GPT-5.6 Terra"],          # reviewer is not the final model: no self-grading
    "final": M["Claude Sonnet 5"],
}
LOW_REASONING = {"reasoning": {"effort": "low"}}
# OpenRouter's model list (2 Nov) has temperature as a supported parameter only for Gemini 4 Argon among these
# models; the others set their own sampling, so it is not sent to them (ignored at best, a rejected call at worst;
# the Anthropic API rejected it in the dev runs).
TEMPERATURE_MODELS = {M["Gemini 4 Argon"]}
FORECASTERS = ("forecast_a", "forecast_b", "forecast_c")
# Evidence gate (code): research older than this, or with no dated sources, cannot move the answer much.
STALE_DAYS = 14
WEAK_EVIDENCE_MAX_WEIGHT = 0.1
WORKERS = 6
# The helper's client tries a hung call 4 times (3 retries) at up to 180 s each, so a stalled model could hold a
# stage for an hour. A forecast, consistency check or critique with no reply by SKIP_AFTER_S is skipped (the run
# goes on without it), and a model that stalls SKIP_MODEL_AFTER times is skipped for the rest of the run.
SKIP_AFTER_S = 240
SKIP_MODEL_AFTER = 2
DATA_WORKERS = 3                  # live data fetches at a time (Kalshi, Polymarket, Yahoo, Treasury rate limits)
DATA_RETRY_WAITS = (0, 3, 10, 30)  # seconds before each live-data attempt
DATA_MAX_AGE_MIN = 30             # a resumed run refetches saved data older than this
DATA_KEEP_LIVE_MIN = 120          # if a refresh fails, live data saved this recently beats the data pack's old prices
RESUME_MAX_HOURS = 3              # an event run label older than this is not resumed (use a new label)
# No new finals this close to the 22:00 GMT deadline: a run whose finals are made counts as complete, and judges take
# the last complete run, so it must also have time to submit.
FINALS_CUTOFF_UTC = datetime(2026, 11, 3, 21, 50, tzinfo=timezone.utc)
AUTO_RESUMES = 2                  # an event run that stops on an outage resumes itself this often before giving up
RESUME_WAIT_S = 30

# Part A anchor policy: markets are hard to beat, so the agent only moves part of the way.
# Against a calibrated market, a deviation of d points without new information costs about d^2 in expected
# Brier score, so moves stay small and need fresh, specific evidence.
A_MAX_WEIGHT = 0.35     # critique can give the agent's own view at most 35% weight
A_MAX_SHIFT = 4         # and the anchor never moves more than 4 points from the market
# The only reasons the critique may give for moving off the market; anything else means weight 0. Code checks the
# first three against the market data (thresholds from the 2 Nov data: the six House races trade 15-65k contracts a
# week, the Senate and control markets 0.6-7M); poll dates are in the brief, so newer_poll is the critique's call.
A_REASONS = ("stale_price", "thin_volume", "venue_gap", "newer_poll")
STALE_HOURS = 6          # no Kalshi trade for this long
THIN_VOLUME = 100_000    # Kalshi contracts in 7 days
VENUE_GAP_PTS = 3        # Kalshi vs Polymarket
# Part B: absolute-error scoring rewards the median; shrink and cap.
# Errors are capped at 5 typical moves, so a well-supported directional call has a capped downside.
B_KEEP_DEFAULT = 0.0    # keep fraction when the critique gives none: a broken critique means baseline only
B_CAP_TYPICAL = 3.0     # anchor at most 3 typical daily moves (VIX: past-election median)
B_SCENARIO_WEIGHT = 0.5  # the forecasters' adjustment to the baseline enters at half weight, times the critique's keep
# The baseline is the past-election prior: data, not research, so keep does not scale it. Its weight is how far the
# past elections agree on direction beyond a coin flip: 4 of 5 -> 0.8 - 0.5 = 0.3, 5 of 5 -> 0.5 (fewer: none).
# The forecasters see the baseline and give only the move beyond it, so the history is counted once.

NOW = datetime.now(timezone.utc)
TODAY = NOW.strftime("%A %d %B %Y, %H:%M UTC")

CONTEXT = """Context (from the event brief): US midterms on Tuesday 3 November 2026. Republicans hold the
White House, the Senate 53-47 (47 includes two independents caucusing with Democrats; the Republican VP breaks
50-50 ties, so Democrats need a net gain of four) and the House 218R-214D, one independent, two vacancies.
Ohio and Florida hold special Senate elections. Maine and Alaska use ranked-choice voting. Races resolve on
AP / major network calls."""


# ----------------------------------------------------------------------------- helpers
def last_json(text: str):
    """Last {...} object in a reply, or None."""
    if not text:
        return None
    dec, found, i = json.JSONDecoder(), None, 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found = obj  # outermost object; skip past it so nested dicts are not returned
        i = end
    return found


def final_json(reply):
    """The JSON object on a final reply's last line, read the way ftf.parse_final reads it, or None."""
    lines = [line.strip().strip("`").strip() for line in (reply or "").splitlines()]
    lines = [line for line in lines if line and line.lower() != "json"]
    try:
        obj = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def final_answer(reply):
    """The answer on a final reply's last line, or None. Strict like the helper: "55%" is not an answer there,
    so it is not one here either (the final is then asked again)."""
    value = (final_json(reply) or {}).get("answer")
    if isinstance(value, str):
        value = value.strip()
    try:
        value = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None
    return value if value is not None and math.isfinite(value) else None


def final_ok(qid, reply, strict=True, band=None):
    """A final the helper will accept and the judges will count as substantive: a plain number (a whole 1-99 for
    Part A, within the helper's limit for Part B), an own estimate and, when strict, every field the helper checks
    (the reason, and for Part A the market reference). Given a band (answer_band), the answer must also be inside
    it: code decides the answer. A final that fails is asked again."""
    answer = final_answer(reply)
    fields = (ftf.FINAL_FIELDS + (("market_reference",) if qid.startswith("A") else ())) if strict else ("own_estimate",)
    if answer is None or any((final_json(reply) or {}).get(k) in (None, "") for k in fields):
        return False
    if band and not band[0] - 1e-9 <= answer <= band[1] + 1e-9:
        return False
    if qid.startswith("A"):
        return answer.is_integer() and 1 <= answer <= 99
    return abs(answer) <= ftf.B_LIMITS[qid][0]


def saved_final_ok(store, qid, strict=True):
    """final_ok for a saved final, against the band of the anchor it was made with."""
    return final_ok(qid, store.get(f"{qid}.6_final"), strict=strict,
                    band=answer_band(qid, store.get(f"{qid}.5_anchor")))


def unusable_finals(store, strict=False):
    """Questions whose saved final has no valid answer or own estimate (strict: or lacks a field the helper checks),
    or answers outside its anchor's band: nothing is submitted while any remain. The last try submits a final that
    lacks only its reason field (its reasoning is in the reply): a finished run whose answers were not sent would
    not match the submission."""
    return [q for q in ftf.QUESTION_IDS if not saved_final_ok(store, q, strict=strict)]


NUMBER = re.compile(r"\s*([+-]?\d+(?:\.\d+)?)\s*(?:%|bps?|pts?|points?)?\s*", re.I)


def num(x, default=None):
    """A number from a model's JSON: 55, "55", "55%", "+0.5 bp". The unit is dropped (the prompt fixes it), so a
    forecaster that writes "55%" still counts. NaN and infinity (JSON's bare NaN, "nan", "inf") are no number: one
    NaN forecast would have set every Part A anchor to the 4-point cap, since min(4, nan) is 4."""
    if isinstance(x, bool):
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        m = NUMBER.fullmatch(x) if isinstance(x, str) else None
        v = float(m.group(1)) if m else default
    return v if v is None or math.isfinite(v) else default


def pct(x):
    """A forecaster's probability in %: a fraction such as 0.62 is read as 62 (the prompt asks for 1-99)."""
    p = num(x)
    return p * 100 if p is not None and 0 < p < 1 else p


def median(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


MARKET_SITES = ("kalshi.com", "polymarket.com", "predictit.org", "manifold.markets", "predictionmarket")
# A price can be quoted by name without a link ("Polymarket has him at 61%"), so lines naming a prediction market
# are dropped too. In the 2 Nov dev briefs this removed 13 more lines out of 86 briefs and changed no evidence flag.
MARKET_WORDS = (*MARKET_SITES, "kalshi", "polymarket", "predictit", "manifold markets", "prediction market",
                "prediction-market", "betting odds", "betting market")
# Research source rule: in dev runs only 5% of cited pages were original sources and many were poll aggregators,
# broker blogs or Substacks, which the mock judges marked down.
SOURCES = ("Prefer the original source: the pollster's or rater's own release (Cook, Sabato's Crystal Ball, Inside "
           "Elections), official data (FEC, BLS, Treasury, Fed, EIA), wire services (AP, Reuters) and established "
           "outlets; use aggregator, broker-marketing, blog or Substack pages only if nothing better exists, and say so.")
# Full or abbreviated month names; written dates count with or without a year ("2 Nov", "October 28, 2026").
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")


def age_days(date):
    """Days from a YYYY-MM-DD date to today, or None."""
    try:
        return (NOW.date() - datetime.strptime(str(date), "%Y-%m-%d").date()).days
    except ValueError:
        return None


def drop_market_lines(text):
    """The brief without lines that cite or name a prediction market."""
    return "\n".join(line for line in (text or "").splitlines() if not any(m in line.lower() for m in MARKET_WORDS))


def evidence_check(text: str, part_a: bool = False) -> dict:
    """Code check of a research brief: are claims dated and sourced, how fresh is it, and for Part A, how old is
    the newest poll?"""
    text = text or ""
    meta = last_json(text) or {}
    body = text[:text.rfind("{")] if meta else text  # the brief without its closing JSON line
    iso = re.findall(r"\b20\d\d-\d\d-\d\d\b", body)
    written = re.findall(rf"\b(?:\d{{1,2}} (?:{MONTHS})\b|(?:{MONTHS})\.? \d{{1,2}}\b)", body)
    # Full URLs, and bare ones the models sometimes write ("npr.org/2026/10/30/...").
    urls = {u.rstrip(".,;:") for u in re.findall(
        r"https?://[^\s)\]>\"']+|\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}/[^\s)\]>\"']*", body)}
    market_sites = sorted(u for u in urls if any(m in u.lower() for m in MARKET_SITES))
    # Freshness from the dates written in the brief (not the model's own claim); future dates are events, not sources.
    dated = sorted(d for d in iso if d <= NOW.strftime("%Y-%m-%d"))
    freshest = dated[-1] if dated else None
    age = age_days(freshest)
    n_dates = len(iso) + len(written)
    weak = (n_dates < 2 or not urls or age is None or age > STALE_DAYS or meta.get("evidence_strength") == "weak"
            or bool(market_sites))
    out = {"dated_claims": n_dates, "sources_cited": len(urls), "freshest_source_date": freshest,
           "freshest_age_days": age, "self_rated_strength": meta.get("evidence_strength"),
           "prediction_market_sources": market_sites}
    if part_a:  # a race brief without a poll from the last STALE_DAYS days is weak, however many dates it has
        poll = meta.get("freshest_poll_end_date")
        # The claimed poll date must appear in the brief and be in the past: a poll cannot end after today.
        poll = poll if poll and date_in_text(poll, body) and age_days(poll) >= 0 else None
        out.update({"freshest_poll_end_date": poll, "poll_age_days": age_days(poll)})
        weak = weak or out["poll_age_days"] is None or out["poll_age_days"] > STALE_DAYS
    return {**out, "weak": weak}


def a_bounds(market):
    """The whole-number answers allowed for a Part A market: at most A_MAX_SHIFT points away, and within 1-99."""
    return max(1, math.ceil(market - A_MAX_SHIFT)), min(99, math.floor(market + A_MAX_SHIFT))


def answer_band(qid, a):
    """The answers a final may give, (low, high), from its code anchor: the anchor itself (Part B: rounded to two
    decimals). None without an anchor. In the 3 Nov smoke test the final took a critic's misreading of the market
    (72% for the Democrat read as 72% for the Republican) for a factual error in the data and answered 29 against an
    anchor of 72; in run 1 its 2 points of leeway when the critique gave weight moved A8 and A11 a point past their
    anchors, counting the critique's weight twice. No model reading can move an answer off the anchor now."""
    if not isinstance(a, dict) or a.get("anchor") is None:
        return None
    if qid.startswith("B"):
        return a["anchor"] - 0.011, a["anchor"] + 0.011
    return a["anchor"], a["anchor"]


def logit(p):
    p = min(max(p, 1.0), 99.0) / 100
    return math.log(p / (1 - p))


def inv_logit(x):
    return 100 / (1 + math.exp(-x))


class Store:
    """Saves every stage's output so a run can resume and so the trace is inspectable."""

    def __init__(self, run, dev=False):
        self.dir = ROOT / "out" / f"{'dev' if dev else 'run'}-{run}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        p = self.dir / f"{key}.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:  # half-written by a crashed run: treat as missing, so the step is redone
            return None

    def put(self, key, value):
        # Write then rename, so a thread reading at the same moment never sees a half-written file.
        p = self.dir / f"{key}.json"
        tmp = p.with_name(f"{p.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str))
        os.replace(tmp, p)
        return value

    def cached(self, key, fn):
        hit = self.get(key)  # an empty saved reply counts as missing, so a resumed run asks again
        return hit if hit else self.put(key, fn())


def pmap(fn, items, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(items, ex.map(fn, items)))


def minutes_since(iso):
    """Minutes since an ISO timestamp, or infinity if there is none."""
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 60
    except (TypeError, ValueError):
        return math.inf


def date_in_text(date, text):
    """Is this YYYY-MM-DD date written in the text (2026-10-31, 10-31, 10/31, 31 Oct, Oct 31, Oct. 31, October 31)?"""
    try:
        day = datetime.strptime(str(date), "%Y-%m-%d")
    except ValueError:
        return False
    forms = {day.strftime("%Y-%m-%d"), day.strftime("%m-%d"), f"{day.month}/{day.day}", day.strftime("%m/%d"),
             f"{day.day} {day:%b}", f"{day:%b} {day.day}", f"{day:%b}. {day.day}", f"{day:%B} {day.day}"}
    return any(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text) for f in forms)


class Stop(Exception):
    """A stop that resuming the run can fix: stale market data, or finals that could not be read."""


def is_live(qid, d):
    """Was this data fetched live? The fallback is the data pack, days old by the event."""
    return (d.get("source") if qid.startswith("A") else (d.get("recent") or {}).get("source")) == "live"


# ----------------------------------------------------------------------------- stages
class Agent:
    def __init__(self, run, qids, dev=False):
        self.run, self.qids, self.dev = run, qids, dev
        self.qs = {q["id"]: q for q in ftf.load_questions()}
        self.store = Store(run, dev)
        self.stalls, self._stalls_lock = {}, threading.Lock()  # model -> calls skipped for no reply in SKIP_AFTER_S
        if dev:
            import dev_backend
            self._backend = dev_backend.ask
            self.dev_model = dev_backend.MODEL  # dev traces name the model that actually ran
        else:
            self._backend = ftf.ask

    def ask(self, qid, role, prompt, model, save=None, **kw):
        kw.setdefault("max_tokens", 4000)  # reasoning tokens count against this; unused room costs nothing
        if model not in TEMPERATURE_MODELS:
            kw.pop("temperature", None)
        if not self.dev:
            kw.setdefault("extra_body", LOW_REASONING)
            # The helper's OpenAI client waits up to 600 s per attempt (3 retries), so one hung call could eat the
            # 21:05-22:00 window; web-search calls get more time than the rest.
            kw.setdefault("timeout", 300 if kw.get("online") else 180)
        if save:  # keep every prompt next to its output, so the trace can be read without the logs
            self.store.put(save, prompt)
        return self._backend(qid, role, prompt, model, run=self.run, **kw)

    def ask_parsed(self, qid, role, prompt, model, ok, safe=False, again="", **kw):
        """ask(), then once more (with the note `again` added to the prompt) if the reply cannot be read or used: a
        broken reply would otherwise leave a question without an own estimate, a critique or an answer. With
        safe=True a model that keeps failing (an outage
        after the client's own retries, or no reply within SKIP_AFTER_S) gives an empty reply instead of stopping
        the run: a missing forecast leaves the median of the others, a missing critique leaves the answer on the
        market or baseline."""
        if safe:
            if self.stalls.get(model, 0) >= SKIP_MODEL_AFTER:
                print(f"WARNING: {qid} {role}: {model} skipped (no reply within {SKIP_AFTER_S} s "
                      f"{SKIP_MODEL_AFTER} times in this run); going on without it")
                return ""
            box = {}

            def call():
                try:
                    box["reply"] = self.ask_parsed(qid, role, prompt, model, ok, again=again, **kw)
                except Exception as e:  # noqa: BLE001
                    box["error"] = e
            worker = threading.Thread(target=call, daemon=True)  # a stalled call is left behind, not waited for
            worker.start()
            worker.join(SKIP_AFTER_S)
            if "reply" in box:
                return box["reply"]
            if "error" in box:
                why = f"{box['error'].__class__.__name__}: {box['error']}"
            else:
                with self._stalls_lock:
                    self.stalls[model] = self.stalls.get(model, 0) + 1
                why = f"no reply within {SKIP_AFTER_S} s"
            print(f"WARNING: {qid} {role}: {model} failed ({why}); going on without it")
            return ""
        reply = self.ask(qid, role, prompt, model, **kw)
        if not ok(reply):
            print(f"{qid} {role}: {model} reply could not be read or used; asking once more")
            reply = self.ask(qid, role, prompt + again, model, **kw)
            if not ok(reply):
                print(f"WARNING: {qid} {role}: {model} gave no usable reply twice")
        return reply

    # 0 ---------------------------------------------------------------- data (code)
    def fetch(self, qid, max_age_min):
        """Live data for one question, retried: ftf_tools silently falls back to the data pack, whose prices are
        days old and would anchor Part A to a stale market. A saved snapshot is reused only if it was live and was
        fetched less than max_age_min minutes ago (a resumed run must not anchor to old prices either). If every
        attempt fails, a live snapshot saved in the last DATA_KEEP_LIVE_MIN minutes is kept instead."""
        key = f"{qid}.0_data"
        saved = self.store.get(key)
        saved_age = minutes_since((saved or {}).get("fetched_utc"))
        if saved and is_live(qid, saved) and saved_age < max_age_min:
            return saved
        for wait in DATA_RETRY_WAITS:
            time.sleep(wait)
            try:
                hit = data.market_snapshot(qid, ftf_tools) if qid.startswith("A") else data.b_evidence(qid, ftf_tools)
            except Exception as e:  # noqa: BLE001
                print(f"{qid}: live data failed ({e})")
                hit = data.market_snapshot(qid) if qid.startswith("A") else data.b_evidence(qid)
            if is_live(qid, hit):
                break
        else:  # every attempt fell back to the data pack
            if saved and is_live(qid, saved) and saved_age < DATA_KEEP_LIVE_MIN:
                print(f"{qid}: live refresh failed; keeping the live data fetched {saved_age:.0f} min ago")
                return saved
        return self.store.put(key, {**hit, "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def fetch_all(self, qs, max_age_min):
        """Fetch a few at a time (parallel bursts got rate-limited into the stale fallback); stop on a stale market."""
        got = pmap(lambda q: self.fetch(q, max_age_min), qs, workers=DATA_WORKERS)
        markets = [q for q in qs if q.startswith("A")]
        stale = [q for q in markets if not is_live(q, got[q])]
        if markets:
            print(f"market data: {len(markets) - len(stale)} of {len(markets)} live" + (f"; stale: {', '.join(stale)}" if stale else ""))
        if stale and os.environ.get("FTF_ALLOW_STALE") != "1":
            raise Stop(f"Live market data failed for {', '.join(stale)} after retries, and the fallback is the data pack's "
                     "old prices. Rerun the same command in a minute (finished stages are kept), or set "
                     "FTF_ALLOW_STALE=1 to go ahead with stale prices.")
        return got

    def stage_data(self):
        qs = list(dict.fromkeys([*self.qids, "A14", "A15"]))  # chamber markets set the Part B scenarios, even in partial runs
        # A resumed run keeps the saved data of the questions it has already finalised, so finishing a run never
        # waits on a market feed; the rest are refetched when their saved data is older than DATA_MAX_AGE_MIN.
        kept = {q: self.store.get(f"{q}.0_data") for q in qs if final_ok(q, self.store.get(f"{q}.6_final"))}
        kept = {q: d for q, d in kept.items() if d}
        self.data = {**kept, **self.fetch_all([q for q in qs if q not in kept], DATA_MAX_AGE_MIN)}
        self.scenarios = data.scenario_probabilities(
            self.data["A14"]["market_avg_pct"], self.data["A15"]["market_avg_pct"])
        self.store.put("scenarios", self.scenarios)

    # 1 ---------------------------------------------------------------- research
    def stage_research(self):
        def general():
            return self.ask_parsed("general", "research", f"""Today is {TODAY}. {CONTEXT}

Write a dated research brief (max 400 words) on the national picture for the 2026 US midterms, using web
search: generic congressional ballot averages, presidential approval, early-vote reports, notable late
developments in the last 7 days, and what US markets expect the day after the election (futures,
positioning, scheduled economic data or Fed events on Wed 4 November 2026).
Rules: give the date (YYYY-MM-DD) and source URL for every number, copying each URL exactly as it appears in your
search results (never shortened); prefer items from the last 14 days; if you cannot find something, say "not found"
instead of estimating. {SOURCES} Do not report prediction-market prices and do not use prediction-market sites
(Kalshi, Polymarket, PredictIt) as sources.""",
                                   MODELS["research"], lambda r: bool(r and r.strip()), online=True,
                                   save="general.1_research_prompt")
        self.general = drop_market_lines(self.store.cached("general.1_research", general))

        def one(qid):
            q = self.qs[qid]
            meta = ('{"freshest_source_date": "YYYY-MM-DD", "freshest_poll_end_date": "YYYY-MM-DD (last field day '
                    'of the newest poll) or none", "evidence_strength": "strong|moderate|weak"}' if qid.startswith("A")
                    else '{"freshest_source_date": "YYYY-MM-DD", "evidence_strength": "strong|moderate|weak"}')
            if qid.startswith("A"):
                task = f"""Research this question: "{q['text']}".
Find, with web search: the candidates; the most recent polls (pollster, field dates, sample, Dem %, Rep %);
current ratings from Cook, Sabato's Crystal Ball and Inside Elections; fundraising or early-vote facts;
late news (last 7 days). For chamber-control questions, summarise the seat math and which races decide it."""
            else:
                ev = self.data[qid]
                task = f"""Research what will drive the {ev['instrument']} move from the 3 November 2026 close to the
4 November 2026 close (unit: {ev['unit']}). Find: scheduled events on 4 November (data releases, earnings,
central-bank meetings, OPEC, inventory reports), recent drivers in the last week, and how analysts expect it
to react to each US midterm outcome (Democratic sweep; Democratic House / Republican Senate; Republican sweep)."""
            # an empty brief, or one without its JSON line, is asked again
            return self.ask_parsed(qid, "research", f"""Today is {TODAY}. {CONTEXT}

{task}
Rules: max 350 words; for every number give its date (YYYY-MM-DD) and source URL, copying each URL exactly as it
appears in your search results (never shortened); prefer the last 14 days; say "not found" rather than estimate.
{SOURCES} Do not report prediction-market prices or use prediction-market sites (Kalshi,
Polymarket, PredictIt) as sources, and do not give your own probability or forecast: independent forecasters will
read this brief. End with one JSON line:
{meta}""",
                                   MODELS["research"], lambda r: last_json(r) is not None, online=True,
                                   save=f"{qid}.1_research_prompt")
        raw = pmap(lambda q: self.store.cached(f"{q}.1_research", lambda: one(q)), self.qids)
        # Briefs reach the forecasters without lines that cite prediction-market sites (the research sometimes
        # cites them for ratings despite the rule), so the blind estimates cannot pick up a market price.
        self.research = {q: drop_market_lines(raw[q]) for q in self.qids}
        self.evidence = {q: self.store.put(f"{q}.1_evidence_check", {
            **evidence_check(self.research[q], q.startswith("A")),
            "market_lines_dropped": len((raw[q] or "").splitlines()) - len(self.research[q].splitlines())})
            for q in self.qids}

    # 2 ---------------------------------------------------------------- forecast (blind)
    def stage_forecast(self):
        def prompt(qid):
            q = self.qs[qid]
            if qid.startswith("A"):
                ask = """Estimate the probability (whole %, 1-99) of this outcome from the evidence alone.
Reason in steps: base rate for this kind of race (incumbency, state lean, midterm penalty for the president's
party), then polls (weight by recency, quality and sample), then ratings and late news.
End with one JSON line: {"p": <1-99>, "confidence": "low|medium|high", "reason": "<one sentence>"}"""
            else:
                ev, h = self.data[qid], self.history_prior(qid)
                base = (f"{h['weight']:.1f} x the median past-election move ({h['prior']:+.2f}) = "
                        f"{h['baseline']:+.2f} {ev['unit']}" if h["weight"] else
                        "0 (the past elections do not agree on a direction)")
                ask = f"""Forecast the move from the 3 Nov close to the 4 Nov close in {ev['unit']}.
Hard data (computed from the event data pack, do not alter): {json.dumps({k: ev[k] for k in
('typical_daily_move', 'past_elections_day_after', 'past_midterms_moves', 'recent')})}
Prediction markets price the control outcomes as {json.dumps(self.scenarios)} (from the House and Senate
control markets), so the likely outcomes are largely priced in on 3 Nov.
The code already sets a baseline from past elections: {base}. Your numbers are added to that baseline, so
leave the usual post-election pattern out of them. For each scenario give the extra move you expect beyond the
baseline (usually small for the likely scenarios, larger for the unlikely ones); put what applies whatever the
result (scheduled data, earnings) in unconditional_drift, which is added to every scenario.
End with one JSON line: {{"D_sweep": x, "D_House_R_Senate": x, "R_sweep": x, "R_House_D_Senate": x,
"unconditional_drift": x, "reason": "<one sentence>"}}"""
            blind = " You are not shown prediction-market prices on purpose." if qid.startswith("A") else ""
            return f"""Today is {TODAY}. {CONTEXT}
Question: {q['text']}

General brief:
{self.general}

Question brief:
{self.research[qid]}

You are an independent forecaster.{blind}
Facts: use those in the briefs and the hard data. Background knowledge is fine for base rates, but mark it as
background and give no exact figures (past margins, turnout, poll numbers) that are not in the briefs.
{ask}"""

        def one(qid):
            key = "p" if qid.startswith("A") else "D_sweep"

            def run_all():
                p = self.store.put(f"{qid}.2_forecast_prompt", prompt(qid))
                with ThreadPoolExecutor(len(FORECASTERS)) as ex:
                    futs = {f: ex.submit(self.ask_parsed, qid, "forecast", p, MODELS[f],
                                         lambda r: num((last_json(r) or {}).get(key)) is not None, safe=True,
                                         temperature=0)
                            for f in FORECASTERS}
                    out = []
                    for f, fut in futs.items():
                        reply = fut.result()
                        model = MODELS[f] if not self.dev else f"{MODELS[f]} [dev run: {self.dev_model}]"
                        out.append({"model": model, "reply": reply, "json": last_json(reply)})
                return out
            hit = self.store.get(f"{qid}.2_forecast")
            # reuse unless no forecaster gave a number (an outage or unreadable replies): ask those again
            if hit and any(num((f.get("json") or {}).get(key)) is not None for f in hit):
                return hit
            return self.store.put(f"{qid}.2_forecast", run_all())
        self.forecasts = pmap(one, self.qids)
        self.own = {q: self._own(q) for q in self.qids}
        self.store.put("own_estimates", {**(self.store.get("own_estimates") or {}), **self.own})  # partial reruns add

    def _own(self, qid):
        js = [f["json"] or {} for f in self.forecasts[qid]]
        if qid.startswith("A"):
            ps = [pct(j.get("p")) for j in js]
            valid = [p for p in ps if p is not None]
            return {"p": median(valid), "by_model": dict(zip([f["model"] for f in self.forecasts[qid]], ps)),
                    "spread": (max(valid) - min(valid)) if len(valid) > 1 else None}
        keys = ("D_sweep", "D_House_R_Senate", "R_sweep", "R_House_D_Senate")
        cond = {k: median([num(j.get(k)) for j in js]) or 0.0 for k in keys}
        drift = median([num(j.get("unconditional_drift")) for j in js]) or 0.0
        adjustment = sum(self.scenarios[k] * cond[k] for k in keys) + drift
        baseline = self.history_prior(qid)["baseline"]
        return {"scenario_moves": {k: round(v, 3) for k, v in cond.items()}, "drift": round(drift, 3),
                "adjustment": round(adjustment, 3), "baseline": round(baseline, 3),
                "expected_move": round(baseline + adjustment, 3)}

    def history_prior(self, qid):
        """Part B baseline from the past elections: W x their median move, rescaled to today's volatility.
        W = share of past elections agreeing on direction minus 0.5; no prior unless 4 of 5 agree."""
        d = self.data[qid]
        scaled = [h["move_in_typical_days"] for h in d["past_elections_day_after"] if h["move_in_typical_days"] is not None]
        prior = (median(scaled) or 0.0) * d["typical_daily_move"]
        same_sign = sum(1 for v in scaled if v * prior > 0)
        weight = same_sign / len(scaled) - 0.5 if same_sign >= 4 else 0.0
        return {"prior": prior if weight else 0.0, "weight": weight, "baseline": weight * prior,
                "agreement": f"{same_sign}/{len(scaled)}"}

    # 3 ---------------------------------------------------------------- aggregate
    def stage_aggregate(self):
        def notes_ok(reply):  # notes must be an object keyed by question ID; a list or text is asked again
            return isinstance((last_json(reply) or {}).get("notes"), dict)

        def build():
            a = {q: {"own_p": self.own[q]["p"]} for q in self.qids if q.startswith("A")}
            b = {q: self.own[q] for q in self.qids if q.startswith("B")}
            return self.ask_parsed("general", "aggregate", f"""Today is {TODAY}. {CONTEXT}
You are checking a set of forecasts for internal consistency before they are finalised.

Part A (agent's own probabilities that Democrats win, made without market prices):
{json.dumps({q: {"question": self.qs[q]["text"], **a[q]} for q in a}, indent=1)}

Part B (baseline = the code's past-election prior, W x median past move; adjustment = the forecasters'
scenario-weighted moves plus drift on top of it, computed in code; expected_move = baseline + adjustment):
{json.dumps({q: {"instrument": self.data[q]["instrument"], "unit": self.data[q]["unit"], **b[q]} for q in b}, indent=1)}
Scenario probabilities used for Part B (from the House and Senate control markets, not the agent's own estimates):
{json.dumps(self.scenarios)}

Check: (1) Senate control (A15) against the individual Senate races and the seats Democrats need; (2) House
control (A14) against the House toss-ups and the national environment; (3) Part B: do related instruments
move coherently (e.g. small caps vs S&P, 2y vs 10y, VIX vs equities)?
For each question with a problem, give a short note. End with one JSON line:
{{"notes": {{"<id>": "<note>", ...}}}}""", MODELS["aggregate"], notes_ok, safe=True, temperature=0,
                                   save="general.3_aggregate_prompt")
        reply = self.store.get("general.3_aggregate")
        if not notes_ok(reply):  # a failed check (outage) is redone on resume
            reply = self.store.put("general.3_aggregate", build())
        notes = (last_json(reply) or {}).get("notes") if notes_ok(reply) else {}
        # keyed by question ID; "A15 (Senate control)" or "a15" still reaches A15
        self.consistency = {m.group(1).upper(): v for k, v in notes.items()
                            if (m := re.match(r"\s*([AB]\d{1,2})\b", str(k), re.I))}

    # 4 ---------------------------------------------------------------- critique
    def stage_critique(self):
        def one(qid):
            own, d = self.own[qid], self.data[qid]
            if qid.startswith("A"):
                body = f"""Market (average of Kalshi and Polymarket, the benchmark we are scored against). Its
probabilities are for the question exactly as written ({self.qs[qid]['text']}), never for the other side:
{json.dumps(d)}
Agent's own estimate (made blind to the market): {json.dumps(own)}

Scoring: Brier gain over the market. Moving away from the market only pays if our evidence has information
the market has not priced. Judge: is the evidence recent and specific? Do the three forecasters agree? Name the
concrete reason the market could be wrong, if there is one (the market data above has the facts):
- stale_price: no Kalshi trade for {STALE_HOURS}+ hours (hours_since_kalshi_trade)
- thin_volume: under {THIN_VOLUME:,} contracts traded in 7 days (kalshi_volume_7d)
- venue_gap: Kalshi and Polymarket differ by {VENUE_GAP_PTS}+ points (kalshi_polymarket_gap_pts)
- newer_poll: a quality poll in the brief released after stable_since_utc, since when the price has stayed within
  2 points, so it has not reacted to that poll
Code checks the first three against the market data. The evidence in the brief must also point the same way as
our estimate (own below market needs evidence for a lower probability). Priors such as a state's lean or past
polling misses are already in the price and are not a reason. If no reason applies, specific_reason is "none" and
weight_on_own is 0.
End with one JSON line: {{"specific_reason": "{'|'.join(A_REASONS)}|none", "weight_on_own": <0.0-{A_MAX_WEIGHT}>, "issues": ["..."], "reason": "<one sentence>"}}"""
            else:
                body = f"""Hard data: {json.dumps({k: d[k] for k in ('typical_daily_move', 'past_elections_day_after', 'past_midterms_moves', 'recent')})}
Agent's forecast: {json.dumps(own)}
(baseline = the code's past-election prior; adjustment = the forecasters' scenario-weighted moves plus drift on
top of it; expected_move = baseline + adjustment; all sums are computed in code, do not recompute them)
Scenario probabilities (from the House and Senate control markets): {json.dumps(self.scenarios)}

Scoring: |forecast - actual| / typical daily move, capped at 5. A "no change" forecast is the benchmark; a
nonzero forecast only pays if the direction is reliable. Judge the forecasters' adjustment: is it backed by the
brief, consistent with what is priced in, and free of invented or stale facts? keep_fraction is how much of the
adjustment to keep (0 = baseline only, 1 = all of it); the baseline comes from data and is not scaled by it.
End with one JSON line: {{"keep_fraction": <0.0-1.0>, "issues": ["..."], "reason": "<one sentence>"}}"""
            key = "weight_on_own" if qid.startswith("A") else "keep_fraction"
            return self.ask_parsed(qid, "critique", f"""Today is {TODAY}. You are a sceptical reviewer.
Question: {self.qs[qid]['text'] if qid.startswith('A') else self.data[qid]['instrument']}
Research brief:
{self.research[qid]}
Code check of the research brief: {json.dumps(self.evidence[qid])}
Consistency note: {self.consistency.get(qid, 'none')}
{body}""", MODELS["critique"], lambda r: num((last_json(r) or {}).get(key)) is not None,
                                   safe=True, temperature=0, save=f"{qid}.4_critique_prompt")
        def cached(q):  # reuse a readable critique; one that failed (outage) is asked again on resume
            hit = self.store.get(f"{q}.4_critique")
            key = "weight_on_own" if q.startswith("A") else "keep_fraction"
            return hit if num((last_json(hit) or {}).get(key)) is not None else self.store.put(f"{q}.4_critique", one(q))
        self.critiques = pmap(cached, self.qids)

    # 5 ---------------------------------------------------------------- anchor (code)
    def model_bias(self, exclude):
        """Mean logit gap between the blind estimates and the market over this run's other Part A questions
        (leave-one-out). The gaps are saved once a run has most questions, so a later partial rerun of the same
        label uses the same values."""
        saved = self.store.get("model_bias")
        if not saved:
            gaps = {q: logit(self.own[q]["p"]) - logit(self.data[q]["market_avg_pct"])
                    for q in self.qids if q.startswith("A") and self.own[q]["p"] is not None}
            if len(gaps) < 10:  # too few questions to estimate a shared lean (partial test runs)
                return 0.0
            saved = self.store.put("model_bias", {"logit_gaps": {q: round(g, 4) for q, g in gaps.items()}})
        others = [g for q, g in saved["logit_gaps"].items() if q != exclude]
        return sum(others) / len(others) if others else 0.0

    def anchor(self, qid):
        c = last_json(self.critiques[qid]) or {}
        own, d = self.own[qid], self.data[qid]
        if qid.startswith("A"):
            m = d["market_avg_pct"]
            if own["p"] is None:
                return {"anchor": int(min(a_bounds(m)[1], max(a_bounds(m)[0], round(m)))), "weight": 0.0, "market": m}
            reason, poll = c.get("specific_reason"), self.evidence[qid].get("freshest_poll_end_date")
            reason = reason.strip().lower() if isinstance(reason, str) else None  # a list or number names no reason
            backed = {"stale_price": (d.get("hours_since_kalshi_trade") or 0) >= STALE_HOURS
                                     or d.get("kalshi_volume_7d") == 0,
                      "thin_volume": (d.get("kalshi_volume_7d") or 0) < THIN_VOLUME,
                      "venue_gap": (d.get("kalshi_polymarket_gap_pts") or 0) >= VENUE_GAP_PTS,
                      # the newest poll ended after the day the price last moved 2+ points
                      "newer_poll": age_days(poll) is not None and str(poll) > str(d.get("stable_since_utc"))[:10],
                      }.get(reason, False)
            w = min(max(num(c.get("weight_on_own"), 0.0), 0.0), A_MAX_WEIGHT)
            if not backed:  # no concrete reason the market is wrong, or one the data contradicts
                w = 0.0
            if self.evidence[qid]["weak"]:
                w = min(w, WEAK_EVIDENCE_MAX_WEIGHT)
            # A gated move follows the blind estimate with the shared lean of the run's other questions removed
            # (logit scale), so the model bias that summary.json reports cannot leak through the gate. If removing
            # it puts the estimate on the other side of the market, the agent has no view the critique argued for.
            own_adj = inv_logit(logit(own["p"]) - self.model_bias(exclude=qid))
            same_side = (own_adj - m) * (own["p"] - m) > 0
            if not same_side:
                w = 0.0
            shift = max(-A_MAX_SHIFT, min(A_MAX_SHIFT, w * (own_adj - m)))
            lo, hi = a_bounds(m)  # rounding must not take the anchor past the cap (73.5 + 4 would round to 78)
            return {"anchor": int(min(hi, max(lo, round(m + shift)))), "weight": w, "market": m,
                    "own": own["p"], "own_bias_adjusted": round(own_adj, 1), "same_side_after_bias": same_side,
                    "shift": round(shift, 1), "specific_reason": reason, "reason_backed": backed,
                    "weak_evidence": self.evidence[qid]["weak"]}
        keep = min(max(num(c.get("keep_fraction"), B_KEEP_DEFAULT), 0.0), 1.0)
        if self.evidence[qid]["weak"]:
            keep = min(keep, 0.5)
        h = self.history_prior(qid)
        adjustment = B_SCENARIO_WEIGHT * keep * own["adjustment"]
        cap = B_CAP_TYPICAL * d["typical_daily_move"]
        if qid == "B4":
            cap = max(cap, abs(d["past_elections_median_move"] or 0))
        value = max(-cap, min(cap, h["baseline"] + adjustment))
        # + 0.0 turns -0.0 into 0.0: the final model read "-0.0" as a small fall and answered -0.004
        return {"anchor": round(value, 2) + 0.0, "baseline": round(h["baseline"], 3), "keep_fraction": keep,
                "adjustment_part": round(adjustment, 3), "cap": round(cap, 2), "expected_move": own["expected_move"],
                "history_prior": round(h["prior"], 3), "history_weight": round(h["weight"], 2),
                "history_agreement": h["agreement"]}

    # 6 ---------------------------------------------------------------- final
    def stage_final(self):
        def one(qid):
            if saved_final_ok(self.store, qid):  # resumed run: keep the final already made (ftf has logged it)
                return self.store.get(f"{qid}.6_final")  # and its anchor
            a = self.store.put(f"{qid}.5_anchor", self.anchor(qid))
            band = answer_band(qid, a)  # code decides the answer; the final explains it and writes the JSON line
            if qid.startswith("A"):
                own = self.own[qid]["p"]  # None only if no forecaster gave a number, even when asked again
                fmt = (f'{{"answer": {a["anchor"]}, "own_estimate": '
                       f'{own if own is not None else "<your own probability 1-99: the forecasters gave none>"}, '
                       f'"market_reference": {self.data[qid]["market_avg_pct"]}, "reason": "<one sentence: the key '
                       f'evidence on this race and why the answer sits where it does against the market>"}}')
                allowed = f"the anchor itself, {a['anchor']} (" + (
                    "the critique's weight is already in it)" if a["weight"] > 0 else
                    "the critique named no concrete reason the market is wrong)")
                rule = ("The policy anchor is market + weight x (bias-adjusted own estimate - market), capped at "
                        f"{A_MAX_SHIFT} points. Code decides the answer: answer {allowed}. Every probability in Data "
                        "is the market's probability of the question exactly as written, never of the other side.")
            else:
                allowed = f"the anchor itself, {a['anchor']}"
                fmt = (f'{{"answer": {a["anchor"]}, "own_estimate": '
                       f'{self.own[qid]["expected_move"]}, "reason": "<one sentence: the main drivers for this '
                       f'instrument on 4 Nov and why the answer sits where it does>"}}')
                rule = ("The policy anchor is the data baseline (W x the median past-election move, W = share of "
                        "past elections agreeing on direction minus 0.5) plus half the forecasters' adjustment times "
                        f"the critique's keep fraction, capped. Code decides the answer: answer {allowed}.")
            prompt = f"""Today is {TODAY}. Finalise one forecast.
Question: {self.qs[qid]['text'] if qid.startswith('A') else self.data[qid]['instrument'] + ' move, 3 Nov close to 4 Nov close'}
Data: {json.dumps(self.data[qid])}
Own estimate: {json.dumps(self.own[qid])}
Consistency note: {self.consistency.get(qid, 'none')}
Critique:
{self.critiques[qid]}
Policy anchor: {json.dumps(a)}
{rule}

Write two or three sentences of reasoning, then end with exactly one JSON line and nothing after it:
{fmt}"""
            again = (f"\n\nYour previous reply could not be used: its JSON line must be readable and complete, and "
                     f"its answer must be {allowed}. Reply again, ending with the JSON line.")
            return self.store.put(f"{qid}.6_final", self.ask_parsed(
                qid, "final", prompt, MODELS["final"], lambda r: final_ok(qid, r, band=band), again=again,
                temperature=0, max_tokens=2000, save=f"{qid}.6_final_prompt"))
        # The benchmark is the 22:00 price, so anchor to the latest one: refetch the Part A markets still to finalise.
        todo = [q for q in self.qids if not saved_final_ok(self.store, q)]
        if todo and not self.dev and datetime.now(timezone.utc) > FINALS_CUTOFF_UTC:
            sys.exit(f"\n>>> Not finalising run {self.run}: it is past {FINALS_CUTOFF_UTC:%H:%M} GMT, too close to the "
                     "22:00 deadline to finish and submit. No final was made, so the last complete run and its "
                     "submission stand. <<<\n")
        self.data.update(self.fetch_all([q for q in todo if q.startswith("A")], max_age_min=0))
        self.model_bias(exclude=None)  # computed and saved once here, before the questions run in parallel
        self.finals = pmap(one, self.qids)

    # ---------------------------------------------------------------- run
    def go(self):
        for name in ("data", "research", "forecast", "aggregate", "critique", "final"):
            print(f"\n=== stage: {name} ===")
            getattr(self, f"stage_{name}")()
        summary = {}
        for q in self.qids:
            fj = final_json(self.finals[q]) or {}  # read like ftf.parse_final, so this is what gets submitted
            a = self.store.get(f"{q}.5_anchor") or {}  # the anchor and market price this final was made with
            summary[q] = {"answer": fj.get("answer"), "anchor": a.get("anchor"), "market": a.get("market"),
                          "own": fj.get("own_estimate")}
        print("\n" + "\n".join(f"{q:4} answer={s['answer']} anchor={s['anchor']} market={s['market']} own={s['own']}"
                               for q, s in summary.items()))
        old = self.store.get("summary") or {}
        summary = {**{k: v for k, v in old.items() if k != "part_a_tilt"}, **summary}  # partial reruns add to it
        tilt = self.part_a_tilt(summary) or old.get("part_a_tilt")
        self.store.put("summary", {**summary, "part_a_tilt": tilt} if tilt else summary)

    def part_a_tilt(self, summary):
        """Part A answers that lean the same way from the market are one correlated bet, not independent views.
        Also records how the blind estimates sit against the market: a shared lean there is model bias, which is
        reported, not traded on (the critique gate needs a concrete reason before the answer moves)."""
        rows = [(num(s["answer"]) - s["market"], self.own[q]["p"] - s["market"]) for q, s in summary.items()
                if q.startswith("A") and q in self.own and num(s["answer"]) is not None and self.own[q]["p"] is not None
                and s.get("market") is not None]
        if len(rows) < 4:
            return None
        devs, gaps = [d for d, _ in rows], [g for _, g in rows]
        up, down = sum(d > 0.5 for d in devs), sum(d < -0.5 for d in devs)  # within 0.5 is rounding to a whole %
        own_up, own_down = sum(g > 2 for g in gaps), sum(g < -2 for g in gaps)
        tilt = {"questions": len(rows), "above_market": up, "below_market": down, "net_points": round(sum(devs), 1),
                "warning": max(up, down) * 2 >= len(rows),
                "own_above_market": own_up, "own_below_market": own_down,
                "mean_own_minus_market": round(sum(gaps) / len(rows), 1),
                "model_bias_warning": max(own_up, own_down) * 3 >= 2 * len(rows)}
        print(f"Part A answers vs market: {up} above, {down} below, net {tilt['net_points']} points")
        print(f"Blind estimates vs market: {own_up} above, {own_down} below, mean {tilt['mean_own_minus_market']} points")
        if tilt["warning"]:
            print(f"WARNING: {max(up, down)} of {len(rows)} Part A answers lean the same way from the market. "
                  "That is one correlated bet; check the critiques' specific_reason.")
        if tilt["model_bias_warning"]:
            print(f"NOTE: the blind estimates lean one way on {max(own_up, own_down)} of {len(rows)} questions: a "
                  "shared model bias. It is recorded here and not traded on.")
        return tilt


def newer_complete_run(label):
    """Another event run that started after this one and has a usable final for every question, or None. Submitting
    this run after it would make an older run the latest submission, while judges take the last complete run."""
    mine = (Store(label).get("run_meta") or {}).get("started_utc") or NOW.isoformat(timespec="seconds")
    for meta_file in sorted((ROOT / "out").glob("run-*/run_meta.json")):
        other = meta_file.parent.name[len("run-"):]
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            continue
        # Only this team's runs, and none dated ahead: a test folder left behind by an interrupted test run must not
        # stop a real run from submitting.
        same_team = meta.get("team") == os.environ.get("TEAM_ID")
        started = meta.get("started_utc") or ""
        ahead = started > datetime.now(timezone.utc).isoformat(timespec="seconds")
        if other != label and same_team and not ahead and started > mine and not unusable_finals(Store(other)):
            return other
    return None


def stability(run, qids, k):
    """Re-send each saved final prompt k times (dev backend) and measure how much the answer moves.
    Part A ranges are in points; Part B ranges are also given in typical daily moves (the scoring unit)."""
    import dev_backend
    store = Store(run, dev=True)

    def one(qid):
        prompt = store.get(f"{qid}.6_final_prompt")
        if prompt is None:
            print(f"{qid}: no saved final prompt in {store.dir}; run --dev --run {run} first")
            return None
        answers = [final_answer(dev_backend.ask(qid, "final", prompt, MODELS["final"], run=f"{run}-stab",
                                                temperature=0, max_tokens=2000))
                   for _ in range(k)]
        valid = [a for a in answers if a is not None]
        r = {"answers": answers, "range": round(max(valid) - min(valid), 3) if valid else None}
        if qid.startswith("B") and r["range"] is not None:
            r["range_in_typical_moves"] = round(r["range"] / store.get(f"{qid}.0_data")["typical_daily_move"], 2)
        print(f"{qid}: {answers}")
        return r

    report = {q: r for q, r in pmap(one, qids).items() if r is not None}
    store.put("stability", report)
    a = [r["range"] for q, r in report.items() if q.startswith("A") and r["range"] is not None]
    b = [r["range_in_typical_moves"] for r in report.values() if "range_in_typical_moves" in r]
    if a:
        print(f"Part A: max range {max(a)} points")
    if b:
        print(f"Part B: max range {max(b)} typical daily moves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run label, e.g. test or 1")
    ap.add_argument("--only", help="comma-separated question IDs (for tests)")
    ap.add_argument("--submit", action="store_true", help="send the real submission after the run")
    ap.add_argument("--dev", action="store_true",
                    help="DEVELOPMENT: use your own Anthropic key (agent/dev_backend.py); never submits")
    ap.add_argument("--stability", type=int, metavar="K",
                    help="with --dev: rerun the saved final prompts K times and report the spread")
    args = ap.parse_args()
    qids = [q.strip().upper() for q in args.only.split(",") if q.strip()] if args.only else list(ftf.QUESTION_IDS)
    unknown = [q for q in qids if q not in ftf.QUESTION_IDS]
    if unknown or not qids:
        sys.exit(f"Unknown question IDs in --only {args.only!r}; use A1-A15 and B1-B10, e.g. --only A1,B4")
    if args.stability:
        if not args.dev:
            sys.exit("--stability only runs with --dev, so it never adds calls to an event run.")
        return stability(args.run, qids, args.stability)
    if args.dev:
        try:
            Agent(args.run, qids, dev=True).go()
        except Stop as e:
            sys.exit(str(e))
        import dev_backend
        print(f"[dev] usage: {dev_backend.COST}. Dev mode never submits.")
        return
    needed = ("OPENROUTER_API_KEY", "TEAM_ID", *(("SUBMIT_CODE",) if args.submit and not args.only else ()))
    if missing := [name for name in needed if not os.environ.get(name)]:
        sys.exit(f"{', '.join(missing)} not set in this Terminal window: run `set -a; source .env; set +a` first.")
    if args.submit and not args.only and os.environ.get("FTF_TEST_MODE") == "1":
        sys.exit("FTF_TEST_MODE=1 would send this submission to the TEST server, where it does not count. For the real "
                 "submission put FTF_TEST_MODE=0 in front of the command (see RUNBOOK.md).")
    if args.submit and not args.only and (newer := newer_complete_run(args.run)):
        sys.exit(f"Run {newer} started after run {args.run} and is complete. Submitting run {args.run} now would make "
                 f"an older run the latest submission, while judges take the last complete run. Nothing was done.")
    if ftf.GIT_COMMIT is None or ftf.GIT_DIRTY:
        sys.exit("Commit first: every event call is tagged with the git commit, and judges see uncommitted runs.")
    store = Store(args.run)
    meta = store.get("run_meta")
    if meta and meta["git_commit"] != ftf.GIT_COMMIT:
        sys.exit(f"Run {args.run} was started on commit {meta['git_commit'][:8]}, but the code is now at "
                 f"{ftf.GIT_COMMIT[:8]}. Resuming would mix two versions in one run; use a new run label.")
    if meta and minutes_since(meta.get("started_utc")) > RESUME_MAX_HOURS * 60:
        sys.exit(f"Run {args.run} was started at {meta.get('started_utc')}, more than {RESUME_MAX_HOURS} hours ago. "
                 "Resuming it would reuse its old data and answers; use a new run label.")
    if meta and not args.only and not unusable_finals(store, strict=True):
        print(f"\n>>> Run {args.run} is already complete: nothing new is asked, it only "
              f"{'submits its answers again' if args.submit else 'checks them'}. A fresh run needs a new label. <<<\n")
    if not meta:
        store.put("run_meta", {"git_commit": ftf.GIT_COMMIT, "started_utc": NOW.isoformat(timespec="seconds"),
                               "team": os.environ.get("TEAM_ID")})
    print(f"Event run {args.run}: FTF_TEST_MODE={os.environ.get('FTF_TEST_MODE', 'unset')} "
          f"({'test server' if os.environ.get('FTF_TEST_MODE') == '1' else 'REAL server'}), "
          f"{'partial run, no submission' if args.only else 'will submit' if args.submit else 'dry run only'}")
    # An outage (a model or the market feed) or a final that could not be read stops the run; it then resumes
    # itself, as rerunning the command would, AUTO_RESUMES times. Finished stages are read back, not redone.
    for attempt in range(AUTO_RESUMES + 1):
        try:
            Agent(args.run, qids).go()
            # the first tries also wait for complete finals (with a reason); the last one never blocks on that
            complete = attempt < AUTO_RESUMES
            bad = [] if args.only else unusable_finals(store, strict=complete)
            if bad:
                raise Stop(f"Not submitting yet: no {'complete' if complete else 'usable'} final for {', '.join(bad)}.")
            break
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, Stop):
                traceback.print_exc()
            why = str(e) if isinstance(e, Stop) else f"{e.__class__.__name__}: {e}"
            if attempt == AUTO_RESUMES:
                sys.exit(f"\nRun {args.run} stopped: {why}\nRerun the same command to resume it; finished questions "
                         "are kept.")
            print(f"\n>>> Run {args.run} stopped ({why}). Resuming in {RESUME_WAIT_S} s "
                  f"({attempt + 1} of {AUTO_RESUMES}) <<<\n")
            time.sleep(RESUME_WAIT_S)
    if args.only:
        return
    answers = ftf.answers_from_finals(run=args.run)
    ftf.submit(answers, run=args.run, dry_run=True)
    if not args.submit:
        print(f"\n>>> Run {args.run} finished WITHOUT submitting (no --submit). To submit it, run the same command "
              "with --submit added: nothing is asked again. <<<\n")
    if args.submit:
        if not args.only and (newer := newer_complete_run(args.run)):  # a newer run finished while this one worked
            sys.exit(f"\n>>> NOT SUBMITTED: run {newer} started after run {args.run} and finished first, so it is the "
                     "last complete run and its submission stands. <<<\n")
        print(f"\n>>> Submitting run {args.run} to the REAL submission server <<<\n")
        receipt = ftf.submit(answers, run=args.run)
        # When the server refuses the answers or cannot be reached, the helper prints one line and returns
        # {"error", "saved"[, "status"]} instead of raising, so say it loudly and end with an error.
        if not isinstance(receipt, dict) or {"error", "saved"} <= set(receipt):
            info = receipt if isinstance(receipt, dict) else {"error": receipt, "saved": "answers/"}
            sys.exit(f"\n>>> NOT SUBMITTED: {info.get('status', '')} {info['error']!r} <<<\n"
                     f"The answers are saved in {info['saved']}. Send them again with: "
                     f"FTF_TEST_MODE=0 .venv/bin/python ai_only/ftf.py resubmit {info['saved']}\n"
                     "If the server is still down near 22:00 GMT, email that file (see RUNBOOK.md).")
    try:
        ftf.credit_left()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
