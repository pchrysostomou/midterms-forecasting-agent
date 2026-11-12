"""Offline tests of the agent's code paths that a normal run rarely exercises. No model or network calls.

Usage (from the repo root): python agent/test_offline.py
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_only"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import data  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILS.append(name)


def make_agent(run, qids):
    a = agent.Agent(run, qids, dev=True)
    shutil.rmtree(a.store.dir, ignore_errors=True)
    a.store.dir.mkdir(parents=True)
    return a


def test_numbers():
    check("numbers with units are read", [agent.num(x) for x in ("55%", "+0.5 bp", " -1.2 points", "3pts", "62")]
          == [55.0, 0.5, -1.2, 3.0, 62.0])
    check("not numbers stay unread", [agent.num(x) for x in ("about 55", "55-60", None, True, "n/a")] == [None] * 5)
    check("a final answer must be a plain number, like the helper wants",
          agent.final_answer('x\n{"answer": "55%"}') is None and agent.final_answer('x\n{"answer": "62"}') == 62)
    check("a probability written as a fraction is read in %", abs(agent.pct(0.62) - 62) < 1e-9
          and abs(agent.pct("0.62") - 62) < 1e-9 and [agent.pct(x) for x in (62, "55%", 1, 0, None)] == [62, 55, 1, 0, None])
    import json
    nan = json.loads('{"p": NaN, "q": Infinity}')  # Python's JSON parser reads these bare tokens
    check("NaN and infinity are no number, so a NaN forecast cannot push anchors to the cap (min(4, nan) is 4)",
          [agent.num(x) for x in (nan["p"], nan["q"], "nan", "inf", "-inf", "1e999")] == [None] * 6
          and agent.num("nan", 0.5) == 0.5 and agent.pct("NaN") is None
          and agent.median([agent.pct(x) for x in (45, "nan", 48)]) == 46.5
          and agent.final_answer('x\n{"answer": NaN}') is None)


def test_final_ok():
    ok = agent.final_ok
    rest = ', "market_reference": 60, "reason": "x"}'
    check("Part A final needs a whole number 1-99", ok("A1", 'x\n{"answer": 76, "own_estimate": 55' + rest)
          and not ok("A1", 'x\n{"answer": 75.5, "own_estimate": 55' + rest)
          and not ok("A1", 'x\n{"answer": 100, "own_estimate": 55' + rest))
    check("Part B final must be within the helper's limit", ok("B4", 'x\n{"answer": -1.2, "own_estimate": -1, "reason": "x"}')
          and not ok("B4", 'x\n{"answer": -31, "own_estimate": -1, "reason": "x"}'))
    check("a final without an own estimate is not usable", not ok("A1", 'x\n{"answer": 76, "own_estimate": null' + rest)
          and not ok("A1", 'x\n{"answer": 76, "own_estimate": null}', strict=False))
    check("a final without its reason or market reference is asked again, but could still be submitted",
          not ok("A1", 'x\n{"answer": 76, "own_estimate": 55, "reason": "x"}')
          and not ok("A1", 'x\n{"answer": 76, "own_estimate": 55, "market_reference": 60, "reason": ""}')
          and not ok("B4", 'x\n{"answer": 1, "own_estimate": 1}') and ok("A1", 'x\n{"answer": 76, "own_estimate": 55}', strict=False))
    band = agent.answer_band
    check("given its anchor's band, a final outside it is not usable, strict or not",
          ok("A1", 'x\n{"answer": 72, "own_estimate": 45' + rest, band=band("A1", {"anchor": 72, "market": 71.5, "weight": 0}))
          and not ok("A1", 'x\n{"answer": 29, "own_estimate": 45' + rest, band=band("A1", {"anchor": 72, "market": 71.5, "weight": 0}))
          and not ok("A1", 'x\n{"answer": 73, "own_estimate": 45}', strict=False,
                     band=band("A1", {"anchor": 72, "market": 71.5, "weight": 0})))
    check("with weight too the answer is the anchor (its weight is already in it); Part B gives the anchor",
          band("A1", {"anchor": 77, "market": 73.5, "weight": 0.35}) == (77, 77)
          and ok("B4", 'x\n{"answer": -1.25, "own_estimate": -1, "reason": "x"}', band=band("B4", {"anchor": -1.25}))
          and not ok("B4", 'x\n{"answer": -1.3, "own_estimate": -1, "reason": "x"}', band=band("B4", {"anchor": -1.25}))
          and band("A1", None) is None)


def test_submit_guard():
    s = agent.Store("t-guard", dev=True)
    for q in agent.ftf.QUESTION_IDS:
        s.put(f"{q}.6_final", 'ok\n{"answer": %s, "own_estimate": 1, "reason": "x"}' % (60 if q.startswith("A") else 0.1))
    check("nothing blocks a complete set of finals", agent.unusable_finals(s) == [])
    s.put("A2.6_final", 'ok\n{"answer": 60, "own_estimate": 55}')
    check("a final missing only its reason does not block the submission", agent.unusable_finals(s) == [])
    s.put("A7.6_final", 'ok\n{"answer": 60, "own_estimate": null, "reason": "x"}')
    (s.dir / "B3.6_final.json").unlink()
    check("a final without an own estimate, or a missing final, blocks the submission",
          agent.unusable_finals(s) == ["A7", "B3"])
    shutil.rmtree(s.dir)


def test_market_edges():
    real = data.market_snapshot("A1")
    check("data-pack snapshot has the new history facts", all(k in real for k in
          ("stable_since_utc", "hours_within_2pts", "hours_since_kalshi_trade")))
    a = make_agent("t-edges", ["A1"])
    a.data = {"A1": {"market_avg_pct": 99.6, "kalshi_volume_7d": 5_000_000, "kalshi_polymarket_gap_pts": None,
                     "hours_since_kalshi_trade": 0.0, "stable_since_utc": "2026-11-01 10:00"}}
    a.own, a.evidence = {"A1": {"p": 97.0}}, {"A1": {"weak": False, "freshest_poll_end_date": None}}
    a.critiques = {"A1": '{"specific_reason": "none", "weight_on_own": 0}'}
    check("an anchor never leaves 1-99 (market 99.6)", a.anchor("A1")["anchor"] == 99)
    a.critiques = {"A1": '{"specific_reason": "venue_gap", "weight_on_own": 0.3}'}
    check("venue_gap with one venue missing is not backed", a.anchor("A1")["weight"] == 0.0)
    shutil.rmtree(a.store.dir)


def test_resume_after_outage():
    a = make_agent("t-resume", ["A1"])
    a.store.put("A1.2_forecast", [{"model": "m", "reply": "", "json": None}] * 3)
    a.store.put("A1.4_critique", "")
    calls = []
    a._backend = lambda qid, role, *args, **kw: calls.append(role) or ('{"p": 50}' if role == "forecast" else
                                                                      '{"specific_reason": "none", "weight_on_own": 0}')
    a.qs = {"A1": {"text": "Texas"}}
    a.general, a.research, a.scenarios = "", {"A1": ""}, {}
    a.data = {"A1": {"market_avg_pct": 60.0}}
    a.evidence, a.consistency = {"A1": {"weak": True}}, {}
    a.stage_forecast()
    check("forecasts that all failed are asked again on resume", calls.count("forecast") == 3 and a.own["A1"]["p"] == 50)
    a.stage_critique()
    check("a critique that failed is asked again on resume", calls.count("critique") == 1)
    a.store.put("A1.2_forecast", [{"model": "m", "reply": "x", "json": {"confidence": "low"}}] * 3)
    a.stage_forecast()
    check("forecasts that gave no number are asked again on resume", calls.count("forecast") == 6 and a.own["A1"]["p"] == 50)
    shutil.rmtree(a.store.dir)


def test_event_timeout():
    import os
    import ftf
    seen = {}

    class Fake:
        def create(self, **kw):
            seen.update(kw)
            msg = type("M", (), {"content": '{"p": 50}', "tool_calls": None})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()],
                                  "usage": type("U", (), {"total_tokens": 1, "cost": 0.0})()})()
    real, team = ftf._openrouter, os.environ.get("TEAM_ID")
    ftf._openrouter = lambda: type("Client", (), {"chat": type("Chat", (), {"completions": Fake()})()})()
    os.environ["TEAM_ID"] = "team-offline-test"
    a = agent.Agent("t-timeout", ["A1"])
    a.ask("A1", "forecast", "prompt", agent.MODELS["forecast_a"])
    check("event calls carry a 180 s timeout and low reasoning", seen.get("timeout") == 180
          and seen.get("extra_body", {}).get("reasoning") == {"effort": "low"})
    a.ask("A1", "research", "prompt", agent.MODELS["research"], online=True)
    check("web-search calls get 300 s", seen.get("timeout") == 300)
    sent = {}
    for role in ("forecast_a", "forecast_b", "forecast_c", "critique", "final"):
        seen.clear()
        a.ask("A1", "forecast", "prompt", agent.MODELS[role], temperature=0)
        sent[role] = "temperature" in seen
    check("temperature goes only to the model that supports it (Gemini)", sent == {
        "forecast_a": False, "forecast_b": False, "forecast_c": True, "critique": False, "final": False})
    ftf._openrouter = real
    if team is None:
        os.environ.pop("TEAM_ID")
    else:
        os.environ["TEAM_ID"] = team
    shutil.rmtree(a.store.dir, ignore_errors=True)
    for p in (ROOT / ".ftf" / "runs").glob("team-offline-test-*"):
        p.unlink()


def test_parsing():
    check("final answer on the last line", agent.final_answer('text\n{"answer": 62, "reason": "x"}') == 62)
    check("final answer inside a code fence", agent.final_answer('x\n```json\n{"answer": -1.5}\n```') == -1.5)
    check("text after the JSON line is not an answer", agent.final_answer('{"answer": 3}\nmore') is None)
    check("no reply, no answer", agent.final_answer(None) is None)


def test_evidence():
    brief = ("Poll 2026-10-26 to 10/31, 52-47 https://example.com/poll (Fox, 30 Oct). Event on 2026-11-30.\n"
             '{"freshest_source_date": "2026-11-01", "freshest_poll_end_date": "2026-10-31", "evidence_strength": "moderate"}')
    e = agent.evidence_check(brief, part_a=True)
    check("freshness from the text, future dates ignored", e["freshest_source_date"] == "2026-10-26")
    check("poll date written as 10/31 is found", e["freshest_poll_end_date"] == "2026-10-31")
    e = agent.evidence_check(brief.replace("10/31", "10/30"), part_a=True)
    check("poll date not in the brief is rejected", e["freshest_poll_end_date"] is None and e["weak"])
    check("poll date written as October 31 or Oct. 31 is found", all(
        agent.evidence_check(brief.replace("10/31", f), part_a=True)["freshest_poll_end_date"] == "2026-10-31"
        for f in ("October 31", "Oct. 31")))
    future = brief.replace('"freshest_poll_end_date": "2026-10-31"', '"freshest_poll_end_date": "2026-11-30"')
    check("a poll date after today is rejected", agent.evidence_check(future, part_a=True)["freshest_poll_end_date"] is None)
    e = agent.evidence_check(brief.replace("(Fox", "(https://www.predictionmarketspicks.com/tx, Fox"), part_a=True)
    check("prediction-market source makes the brief weak", e["weak"] and e["prediction_market_sources"])
    check("empty brief is weak", agent.evidence_check("", part_a=True)["weak"])
    cleaned = agent.drop_market_lines("Cook: Toss-up (https://cookpolitical.com/x)\nRatings via "
                                      "https://predictionmarketspicks.com/tx\nKalshi news: https://news.kalshi.com/p\n"
                                      '{"freshest_source_date": "2026-11-01"}')
    check("lines citing prediction-market sites are dropped, the rest kept",
          "predictionmarket" not in cleaned and "kalshi" not in cleaned and "cookpolitical" in cleaned
          and agent.last_json(cleaned) is not None)
    named = agent.drop_market_lines("Polymarket has Talarico at 61%.\nThe prediction market favours him.\n"
                                     "Emerson: D 48, R 47 (2026-10-30)\nFed hike odds 70% (CME FedWatch)")
    check("lines naming a prediction market are dropped even without a link; other odds stay",
          named == "Emerson: D 48, R 47 (2026-10-30)\nFed hike odds 70% (CME FedWatch)")


def test_history_prior():
    a = make_agent("t-prior", ["B1"])
    a.data = {"B1": {"typical_daily_move": 0.5}}
    for moves, weight in (((3, 4, 2, -2, 6), 0.3), ((3, 4, 2, 1, 6), 0.5), ((3, -4, 2, -1, 6), 0.0), ((0, 0, 0, 1, -1), 0.0)):
        a.data["B1"]["past_elections_day_after"] = [{"move_in_typical_days": m} for m in moves]
        check(f"history weight {moves} -> {weight}", abs(a.history_prior("B1")["weight"] - weight) < 1e-9)
    shutil.rmtree(a.store.dir)


def test_anchor_a():
    a = make_agent("t-anchor", [f"A{i}" for i in range(1, 16)])
    a.data = {q: {"market_avg_pct": 60.0, "kalshi_volume_7d": 5_000_000, "kalshi_polymarket_gap_pts": 1.0,
                  "hours_since_kalshi_trade": 0.0, "stable_since_utc": "2026-11-01 10:00"} for q in a.qids}
    a.own = {q: {"p": 50.0} for q in a.qids}  # every blind estimate 10 points under the market: a shared bias
    a.evidence = {q: {"weak": False, "freshest_poll_end_date": None} for q in a.qids}
    a.own["A1"] = {"p": 30.0}  # A1 is far below even after the bias is removed
    a.critiques = {q: '{"specific_reason": "none", "weight_on_own": 0.3}' for q in a.qids}
    check("no named reason: weight 0", a.anchor("A1")["weight"] == 0.0)
    a.critiques["A1"] = '{"specific_reason": "thin_volume", "weight_on_own": 0.3}'
    check("thin_volume rejected on a 5M-contract market", a.anchor("A1")["weight"] == 0.0)
    a.data["A1"]["kalshi_volume_7d"] = 20_000
    r = a.anchor("A1")
    check("thin_volume accepted on 20k contracts, shift capped at 4", r["weight"] == 0.3 and r["anchor"] == 56)
    a.critiques["A1"] = '{"specific_reason": ["thin_volume"], "weight_on_own": 0.3}'
    check("a reason given as a list names no reason (and does not crash)", a.anchor("A1")["weight"] == 0.0)
    a.critiques["A1"] = '{"specific_reason": " Thin_Volume ", "weight_on_own": 0.3}'
    check("a reason in other case or spacing is still read", a.anchor("A1")["weight"] == 0.3)
    a.own["A2"] = {"p": 55.0}  # below the market, but above it once the shared lean is removed
    a.data["A2"]["kalshi_volume_7d"] = 20_000
    a.critiques["A2"] = '{"specific_reason": "thin_volume", "weight_on_own": 0.3}'
    r = a.anchor("A2")
    check("no move when removing the bias crosses the market", not r["same_side_after_bias"] and r["anchor"] == 60)
    a.evidence["A1"]["weak"] = True
    check("weak research caps the weight at 0.1", a.anchor("A1")["weight"] == 0.1)
    a.critiques["A3"] = '{"specific_reason": "newer_poll", "weight_on_own": 0.2}'
    a.own["A3"] = {"p": 30.0}
    a.evidence["A3"]["freshest_poll_end_date"] = "2026-10-30"
    check("newer_poll rejected for a poll before the last move", a.anchor("A3")["weight"] == 0.0)
    a.evidence["A3"]["freshest_poll_end_date"] = "2026-11-02"
    check("newer_poll accepted for a poll after the last move", a.anchor("A3")["weight"] == 0.2)
    a.critiques["A4"] = "garbled critique"
    check("unreadable critique: weight 0", a.anchor("A4")["weight"] == 0.0)
    shutil.rmtree(a.store.dir)


def test_anchor_b():
    a = make_agent("t-anchor-b", ["B4"])
    a.data = {"B4": {"typical_daily_move": 0.64, "past_elections_median_move": -4.22,
                     "past_elections_day_after": [{"move_in_typical_days": m} for m in (-6.6, -6.6, -8.4, 0.7, -5.9)]}}
    a.own = {"B4": {"adjustment": 1.0, "expected_move": -0.3}}
    a.evidence = {"B4": {"weak": False}}
    a.critiques = {"B4": "garbled critique"}
    r = a.anchor("B4")
    check("unreadable Part B critique keeps none of the adjustment",
          r["keep_fraction"] == 0.0 and abs(r["anchor"] - r["baseline"]) < 0.01)
    a.critiques = {"B4": '{"keep_fraction": 1.0}'}
    check("Part B adjustment enters at half weight", abs(a.anchor("B4")["adjustment_part"] - 0.5) < 1e-9)
    a.data["B4"]["past_elections_day_after"] = [{"move_in_typical_days": m} for m in (1, -1, 2, -2, 0)]
    a.own["B4"]["adjustment"] = -0.006
    check("a tiny negative anchor is shown as 0.0, not -0.0", str(a.anchor("B4")["anchor"]) == "0.0")
    shutil.rmtree(a.store.dir)


def test_retry():
    a = make_agent("t-retry", ["A1"])
    replies = iter(["no json here", '{"p": 55}'])
    a._backend = lambda *args, **kw: next(replies)
    r = a.ask_parsed("A1", "forecast", "prompt", "model", lambda x: agent.num((agent.last_json(x) or {}).get("p")) is not None)
    check("an unreadable reply is asked once more", r == '{"p": 55}')
    shutil.rmtree(a.store.dir)


def test_outage():
    a = make_agent("t-outage", ["A1"])

    def down(*args, **kw):
        raise ConnectionError("provider down")
    a._backend = down
    r = a.ask_parsed("A1", "forecast", "prompt", "google/gemini-4-argon", lambda x: False, safe=True)
    check("a model outage gives an empty reply instead of stopping the run", r == "")
    try:
        a.ask_parsed("A1", "final", "prompt", "model", lambda x: False)
        check("a final call that fails still stops the run", False)
    except ConnectionError:
        check("a final call that fails still stops the run", True)
    a.forecasts = {"A1": [{"model": "m1", "json": {"p": 40}}, {"model": "m2", "json": None},
                          {"model": "m3", "json": {"p": 50}}]}
    check("one missing forecast leaves the median of the other two", a._own("A1")["p"] == 45)
    shutil.rmtree(a.store.dir)


def test_stale_data():
    a = make_agent("t-stale", ["A1"])
    agent.DATA_RETRY_WAITS = (0, 0)
    real = data.market_snapshot
    data.market_snapshot = lambda q, tools=None: {**real(q), "source": "data kit files (live download failed)"}
    try:
        a.fetch_all(["A1"], 30)
        check("a stale market stops the run", False)
    except agent.Stop:
        check("a stale market stops the run", True)
    data.market_snapshot = real
    shutil.rmtree(a.store.dir)


def test_store():
    s = agent.Store("t-store", dev=True)
    (s.dir / "half.json").write_text('{"a": ')
    check("a half-written file reads as missing", s.get("half") is None)
    errors = []

    def writer():
        for i in range(200):
            s.put("x", {"i": i, "pad": "y" * 2000})

    def reader():
        for _ in range(1000):
            try:
                s.get("x")
            except Exception as e:  # noqa: BLE001
                errors.append(e)
    threads = [threading.Thread(target=f) for f in (writer, writer, reader, reader)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    check("concurrent writes and reads never see a broken file", not errors)
    shutil.rmtree(s.dir)


def test_model_bias():
    a = make_agent("t-bias", ["A1", "A2"])
    a.own = {q: {"p": 50.0} for q in a.qids}
    a.data = {q: {"market_avg_pct": 60.0} for q in a.qids}
    check("too few questions: no bias estimate", a.model_bias(exclude=None) == 0.0)
    a.store.put("model_bias", {"logit_gaps": {"A1": -0.4, "A2": -0.2, "A3": -0.6}})
    check("saved gaps are reused, leaving the question out", abs(a.model_bias(exclude="A1") - (-0.4)) < 1e-9)
    shutil.rmtree(a.store.dir)


def test_aggregate_notes():
    a = make_agent("t-agg", ["A1"])
    replies = iter(['{"notes": []}', '{"notes": "none"}'])
    a._backend = lambda *args, **kw: next(replies)
    a.qs, a.own, a.data, a.scenarios = {"A1": {"text": "Texas"}}, {"A1": {"p": 50}}, {}, {}
    a.stage_aggregate()
    check("consistency notes that are not an object are asked again, then ignored", a.consistency == {})
    shutil.rmtree(a.store.dir)


def test_data_resume():
    a = make_agent("t-data", ["A1", "B1"])
    a.store.put("A1.6_final", 'x\n{"answer": 58, "own_estimate": 55, "market_reference": 58.0, "reason": "x"}')
    a.store.put("A1.0_data", {"market_avg_pct": 58.0, "source": "data pack files"})  # even a stale one is kept
    seen = {}
    a.fetch = lambda q, max_age_min: seen.__setitem__(q, max_age_min) or (
        {"market_avg_pct": 50.0, "source": "live"} if q.startswith("A") else {"recent": {"source": "live"}})
    a.stage_data()
    age = agent.DATA_MAX_AGE_MIN
    check("a resumed run keeps the data of finished questions and refreshes the rest",
          seen == {"B1": age, "A14": age, "A15": age} and a.data["A1"]["market_avg_pct"] == 58.0)
    shutil.rmtree(a.store.dir)


def test_refresh_failure():
    a = make_agent("t-refresh", ["A1"])
    agent.DATA_RETRY_WAITS = (0,)
    real = data.market_snapshot

    def down(q, tools=None):
        if tools is not None:
            raise ConnectionError("429")
        return {**real(q), "source": "data kit files (live download failed)"}
    data.market_snapshot = down
    ago = lambda m: (datetime.now(timezone.utc) - timedelta(minutes=m)).isoformat(timespec="seconds")  # noqa: E731
    a.store.put("A1.0_data", {"market_avg_pct": 77.7, "source": "live", "fetched_utc": ago(10)})
    got = a.fetch("A1", 0)
    check("a failed refresh keeps live prices saved 10 minutes ago",
          got["market_avg_pct"] == 77.7 and a.store.get("A1.0_data")["source"] == "live")
    a.store.put("A1.0_data", {"market_avg_pct": 77.7, "source": "live", "fetched_utc": ago(180)})
    check("but not live prices saved 3 hours ago", a.fetch("A1", 0)["source"] != "live")
    data.market_snapshot = real
    shutil.rmtree(a.store.dir)


def test_research_retry():
    a = make_agent("t-research", ["B1"])
    replies = iter(["", "brief", "", 'brief\n{"freshest_source_date": "2026-11-01"}'])
    a._backend = lambda *args, **kw: next(replies)
    a.qs, a.data = {"B1": {"text": "SPY"}}, {"B1": {"instrument": "SPY", "unit": "%"}}
    a.store.put("general.1_research", "")  # saved empty by an earlier attempt
    a.stage_research()
    check("an empty research brief is asked again, also when saved empty",
          a.general == "brief" and agent.last_json(a.research["B1"]) is not None)
    shutil.rmtree(a.store.dir)


def test_stalled_model():
    a = make_agent("t-stall", ["A1"])
    calls = []

    def slow(qid, role, prompt, model, **kw):
        calls.append(model)
        time.sleep(1.0)
        return '{"p": 50}'
    a._backend = slow
    agent.SKIP_AFTER_S, limit = 0.2, agent.SKIP_AFTER_S
    ok = lambda r: agent.num((agent.last_json(r) or {}).get("p")) is not None  # noqa: E731
    start = time.monotonic()
    replies = [a.ask_parsed("A1", "forecast", "p", "m1", ok, safe=True) for _ in range(3)]
    check("a model with no reply in time is skipped, and after two stalls not called again",
          replies == ["", "", ""] and calls == ["m1", "m1"] and time.monotonic() - start < 1.0)
    a._backend = lambda *args, **kw: '{"p": 50}'
    check("other models are still asked", a.ask_parsed("A1", "forecast", "p", "m2", ok, safe=True) == '{"p": 50}')
    agent.SKIP_AFTER_S = limit
    shutil.rmtree(a.store.dir)


def run_main(argv, env=None, go=None):
    """agent.main() with the run itself stubbed out (or replaced by go) and fake settings; returns its exit message,
    or None if it ended normally."""
    ftf = agent.ftf
    names = ("FTF_TEST_MODE", "OPENROUTER_API_KEY", "TEAM_ID", "SUBMIT_CODE")
    saved = (agent.Agent.go, ftf.GIT_COMMIT, ftf.GIT_DIRTY, sys.argv, ftf.credit_left, {n: os.environ.get(n) for n in names})
    agent.Agent.go = go or (lambda self: None)
    ftf.GIT_COMMIT, ftf.GIT_DIRTY, sys.argv = "0" * 40, False, ["agent.py", *argv]
    ftf.credit_left = lambda: None
    os.environ.update({"FTF_TEST_MODE": "0", "OPENROUTER_API_KEY": "sk-or-offline-test", "TEAM_ID": "team-offline-test",
                       "SUBMIT_CODE": "offline-test", **(env or {})})
    try:
        agent.main()
        return None
    except SystemExit as e:
        return str(e.code)
    finally:
        agent.Agent.go, ftf.GIT_COMMIT, ftf.GIT_DIRTY, sys.argv, ftf.credit_left, old = saved
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_main_guards():
    msg = run_main(["--run", "t-guards", "--submit"], {"FTF_TEST_MODE": "1"})
    check("--submit with FTF_TEST_MODE=1 stops before any call", msg is not None and "TEST server" in msg
          and not agent.Store("t-guards").dir.joinpath("run_meta.json").exists())
    s = agent.Store("t-guards")
    s.put("run_meta", {"git_commit": "0" * 40, "started_utc": "2026-11-02T10:00:00+00:00"})
    msg = run_main(["--run", "t-guards", "--only", "A1"])
    check("a run label started hours ago is not resumed", msg is not None and "new run label" in msg)
    check("--only with spaces or lower case works, unknown IDs stop", "Unknown" in (run_main(["--run", "t-x", "--only", "A99"]) or "")
          and run_main(["--run", "t-y", "--only", "a1, b4"]) is None)
    for label in ("t-guards", "t-x", "t-y"):
        shutil.rmtree(agent.Store(label).dir, ignore_errors=True)


def test_final_stage():
    a = make_agent("t-final", ["A1", "A2"])
    kept = 'x\n{"answer": 58, "own_estimate": 55, "market_reference": 58.0, "reason": "kept"}'
    a.store.put("A1.6_final", kept)
    a.store.put("A1.5_anchor", {"anchor": 58, "market": 58.0, "weight": 0.0})
    fetched, prompts = [], []
    a.fetch_all = lambda qs, max_age_min: fetched.append(list(qs)) or {}
    a._backend = lambda qid, role, prompt, *args, **kw: prompts.append(prompt) or (
        'x\n{"answer": 99, "own_estimate": 97, "market_reference": 99.7, "reason": "y"}')
    a.data = {"A1": {"market_avg_pct": 61.0}, "A2": {"market_avg_pct": 99.7}}
    a.own = {"A1": {"p": 55.0}, "A2": {"p": None}}
    a.critiques, a.consistency = {"A1": "", "A2": ""}, {}
    a.evidence = {"A1": {"weak": True}, "A2": {"weak": True}}
    a.stage_final()
    check("a resumed run keeps a finished final and its anchor, and refetches only the rest",
          a.finals["A1"] == kept and a.store.get("A1.5_anchor")["market"] == 58.0 and fetched == [["A2"]])
    check("no own estimate: the final is asked for one, and the anchor stays in 1-99",
          len(prompts) == 1 and "the forecasters gave none" in prompts[0] and a.store.get("A2.5_anchor")["anchor"] == 99)
    shutil.rmtree(a.store.dir)


def test_submit_failure_is_loud():
    ftf = agent.ftf
    saved = (agent.unusable_finals, ftf.answers_from_finals, ftf.submit)
    agent.unusable_finals = lambda store, strict=False: []
    ftf.answers_from_finals = lambda run: {}
    try:
        ftf.submit = lambda answers, run, dry_run=False: {} if dry_run else {"error": "rejected (401)", "saved": "answers/x.csv"}
        msg = run_main(["--run", "t-submitfail", "--submit"]) or ""
        check("a rejected submission stops with a clear message", "NOT SUBMITTED" in msg
              and "FTF_TEST_MODE=0 .venv/bin/python ai_only/ftf.py resubmit answers/x.csv" in msg)
        ftf.submit = lambda answers, run, dry_run=False: {"submission_id": "s1", "error": None} if not dry_run else {}
        check("an accepted submission ends normally", run_main(["--run", "t-submitfail", "--submit"]) is None)
    finally:
        agent.unusable_finals, ftf.answers_from_finals, ftf.submit = saved
        shutil.rmtree(agent.Store("t-submitfail").dir, ignore_errors=True)


def test_auto_resume():
    ftf = agent.ftf
    saved = (agent.Agent.go, agent.unusable_finals, ftf.answers_from_finals, ftf.submit, agent.RESUME_WAIT_S)
    runs = []

    def go(self):
        runs.append(1)
        if len(runs) == 1:
            raise ConnectionError("provider down")
    agent.RESUME_WAIT_S = 0
    agent.unusable_finals = lambda store, strict=False: ["A3"] if len(runs) == 2 else []
    ftf.answers_from_finals, ftf.submit = (lambda run: {}), (lambda answers, run, dry_run=False: {})
    try:
        msg = run_main(["--run", "t-autoresume"], go=go)
        check("a run that stops on an outage or an unreadable final resumes itself",
              msg is None and len(runs) == 3)
        runs.clear()
        msg = run_main(["--run", "t-autoresume2"],
                       go=lambda self: runs.append(1) or (_ for _ in ()).throw(ConnectionError("still down")))
        check(f"after {agent.AUTO_RESUMES} resumes it stops and says to rerun",
              len(runs) == agent.AUTO_RESUMES + 1 and "Rerun the same command" in (msg or ""))
    finally:
        agent.Agent.go, agent.unusable_finals, ftf.answers_from_finals, ftf.submit, agent.RESUME_WAIT_S = saved
        for label in ("t-autoresume", "t-autoresume2"):
            shutil.rmtree(agent.Store(label).dir, ignore_errors=True)


def test_reason_gate():
    ftf = agent.ftf
    saved = (ftf.answers_from_finals, ftf.submit, agent.RESUME_WAIT_S)
    ftf.answers_from_finals, ftf.submit = (lambda run: {}), (lambda answers, run, dry_run=False: {})
    agent.RESUME_WAIT_S = 0
    s = agent.Store("t-reason")
    for q in ftf.QUESTION_IDS:
        s.put(f"{q}.6_final", 'ok\n{"answer": %s, "own_estimate": 1, "market_reference": 60, "reason": "x"}'
              % (60 if q.startswith("A") else 0.1))
    s.put("B1.6_final", 'ok\n{"answer": 0.1, "own_estimate": 1}')  # its reason is missing, even when asked again
    runs = []
    try:
        msg = run_main(["--run", "t-reason"], go=lambda self: runs.append(1))
        check("a final without its reason is asked again on every resume, and still submitted on the last try",
              msg is None and len(runs) == agent.AUTO_RESUMES + 1)
    finally:
        ftf.answers_from_finals, ftf.submit, agent.RESUME_WAIT_S = saved
        shutil.rmtree(s.dir, ignore_errors=True)


def test_cap_after_rounding():
    a = make_agent("t-cap", ["A1"])
    a.data = {"A1": {"market_avg_pct": 73.5, "kalshi_volume_7d": 20_000, "kalshi_polymarket_gap_pts": 1.0,
                     "hours_since_kalshi_trade": 0.0, "stable_since_utc": "2026-11-01 10:00"}}
    a.own, a.evidence, a.consistency = {"A1": {"p": 99.0}}, {"A1": {"weak": False, "freshest_poll_end_date": None}}, {}
    a.critiques = {"A1": '{"specific_reason": "thin_volume", "weight_on_own": 0.35}'}
    prompts = []
    a.fetch_all = lambda qs, max_age_min: {}
    a._backend = lambda qid, role, prompt, *args, **kw: prompts.append(prompt) or (
        'x\n{"answer": 77, "own_estimate": 99, "market_reference": 73.5, "reason": "y"}')
    a.stage_final()
    check("the anchor stays within 4 points of the market after rounding (73.5 + 4 gives 77, not 78)",
          a.store.get("A1.5_anchor")["anchor"] == 77)
    check("with weight the final is given the anchor too, and told the weight is already in it",
          '{"answer": 77, ' in prompts[0] and "the critique's weight is already in it" in prompts[0])
    shutil.rmtree(a.store.dir)


def test_final_band():
    # the 3 Nov smoke test: the critic read 71.5% for the Democrat as 71.5% for the Republican, and the final followed
    a = make_agent("t-band", ["A1"])
    a.data = {"A1": {"market_avg_pct": 71.5, "kalshi_volume_7d": 11_000_000, "kalshi_polymarket_gap_pts": 2.0,
                     "hours_since_kalshi_trade": 0.0, "stable_since_utc": "2026-11-03 15:00"}}
    a.own, a.evidence, a.consistency = {"A1": {"p": 45.0}}, {"A1": {"weak": True, "freshest_poll_end_date": None}}, {}
    a.critiques = {"A1": "Talarico (D) 28.5%: the complement of the market's 71.5% for the Republican.\n"
                         '{"specific_reason": "none", "weight_on_own": 0.0}'}
    wrong = 'x\n{"answer": 29, "own_estimate": 45.0, "market_reference": 71.5, "reason": "the complement"}'
    replies = [wrong, 'x\n{"answer": 72, "own_estimate": 45.0, "market_reference": 71.5, "reason": "the anchor"}']
    prompts = []
    a.fetch_all = lambda qs, max_age_min: {}
    a._backend = lambda qid, role, prompt, *args, **kw: prompts.append(prompt) or replies.pop(0)
    a.stage_final()
    check("weight 0: the final is given the anchor as its answer, with no factual-error way out",
          '{"answer": 72, ' in prompts[0] and "factual error" not in prompts[0]
          and "never of the other side" in prompts[0])
    check("a final outside the band (29 against an anchor of 72) is asked again with a note, and the answer is 72",
          len(prompts) == 2 and "could not be used" in prompts[1] and agent.final_answer(a.finals["A1"]) == 72)
    check("a saved final inside its anchor's band is kept", agent.saved_final_ok(a.store, "A1"))
    a.store.put("A1.6_final", wrong)
    check("a saved final outside its anchor's band blocks the submission", not agent.saved_final_ok(a.store, "A1")
          and "A1" in agent.unusable_finals(a.store, strict=False))
    shutil.rmtree(a.store.dir)


def test_critique_direction():
    a = make_agent("t-crit", ["A1"])
    a.data = {"A1": {"market_avg_pct": 71.5}}
    a.own, a.research, a.evidence, a.consistency = {"A1": {"p": 45.0}}, {"A1": "brief"}, {"A1": {"weak": True}}, {}
    prompts = []
    a._backend = lambda qid, role, prompt, *args, **kw: prompts.append(prompt) or (
        '{"specific_reason": "none", "weight_on_own": 0}')
    a.stage_critique()
    check("the critique is told the market's probabilities are for the question as written, not the other side",
          "never for the other side" in prompts[0] and "probability the Democratic candidate wins" in prompts[0])
    shutil.rmtree(a.store.dir)


def test_note_keys():
    a = make_agent("t-notes", ["A1"])
    a._backend = lambda *args, **kw: '{"notes": {"A15 (Senate control)": "x", "b4": "y", "general": "z"}}'
    a.qs, a.own, a.data, a.scenarios = {"A1": {"text": "Texas"}}, {"A1": {"p": 50}}, {}, {}
    a.stage_aggregate()
    check("consistency notes reach their question even when the key is not a bare ID",
          a.consistency == {"A15": "x", "B4": "y"})
    shutil.rmtree(a.store.dir)


TEST_TEAM = "team-offline-test"  # run_main's TEAM_ID: real runs in out/ belong to another team and never interfere


def ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def complete(store):
    for q in agent.ftf.QUESTION_IDS:
        store.put(f"{q}.6_final", 'ok\n{"answer": %s, "own_estimate": 1, "market_reference": 60, "reason": "x"}'
                  % (60 if q.startswith("A") else 0.1))


def test_older_run_not_resubmitted():
    old, new = agent.Store("t-older"), agent.Store("t-newer")
    old.put("run_meta", {"git_commit": "0" * 40, "started_utc": ago(150), "team": TEST_TEAM})
    new.put("run_meta", {"git_commit": "0" * 40, "started_utc": ago(30), "team": TEST_TEAM})
    complete(new)
    saved = os.environ.get("TEAM_ID")
    try:
        msg = run_main(["--run", "t-older", "--submit"])
        check("an older run is not submitted after a newer complete run (up-arrow once too often)",
              msg is not None and "Run t-newer started after run t-older" in msg)
        os.environ["TEAM_ID"] = TEST_TEAM
        cases = {"another team's run (a mock run)": {"started_utc": ago(30), "team": "team-mock"},
                 "a run without a team, such as a test folder left behind": {"started_utc": ago(30)},
                 "a run dated in the future": {"started_utc": ago(-24 * 60), "team": TEST_TEAM}}
        blocked = []
        for name, meta in cases.items():
            new.put("run_meta", {"git_commit": "0" * 40, **meta})
            if agent.newer_complete_run("t-older"):
                blocked.append(name)
        check("a newer run of another team, without a team or dated ahead does not block it", not blocked)
        new.put("run_meta", {"git_commit": "0" * 40, "started_utc": ago(30), "team": TEST_TEAM})
        (new.dir / "B3.6_final.json").unlink()  # the newer run is not complete: the older one is still the last complete run
        check("nor a newer run that is incomplete", agent.newer_complete_run("t-older") is None)
    finally:
        os.environ.pop("TEAM_ID", None) if saved is None else os.environ.update({"TEAM_ID": saved})
        shutil.rmtree(old.dir, ignore_errors=True)
        shutil.rmtree(new.dir, ignore_errors=True)


def test_newer_run_finishes_first():
    # run 2 is still working when run 3 starts, and run 3 finishes first: run 2 must not submit after it
    ftf = agent.ftf
    slow, fast = agent.Store("t-slow"), agent.Store("t-fast")
    for s in (slow, fast):
        shutil.rmtree(s.dir, ignore_errors=True)
        s.dir.mkdir(parents=True)
    slow.put("run_meta", {"git_commit": "0" * 40, "started_utc": ago(40), "team": TEST_TEAM})
    complete(slow)
    sent = []

    def go(self):  # while this run works, a newer one completes
        fast.put("run_meta", {"git_commit": "0" * 40, "started_utc": ago(10), "team": TEST_TEAM})
        complete(fast)
    saved = (ftf.answers_from_finals, ftf.submit)
    ftf.answers_from_finals = lambda run: {}
    ftf.submit = lambda answers, run, dry_run=False: sent.append(dry_run) or {}
    try:
        msg = run_main(["--run", "t-slow", "--submit"], go=go) or ""
        check("a run that a newer complete run overtook does not submit at the end",
              "NOT SUBMITTED" in msg and "t-fast" in msg and False not in sent)
    finally:
        ftf.answers_from_finals, ftf.submit = saved
        for s in (slow, fast):
            shutil.rmtree(s.dir, ignore_errors=True)


def test_finals_cutoff():
    a = agent.Agent("t-cutoff", ["A1"])  # event mode: the cutoff does not apply to dev runs
    shutil.rmtree(a.store.dir, ignore_errors=True)
    a.store.dir.mkdir(parents=True)
    asked = []
    a._backend = lambda *args, **kw: asked.append("final") or ""
    a.fetch_all = lambda qs, max_age_min: asked.append("fetch") or {}
    saved = agent.FINALS_CUTOFF_UTC
    agent.FINALS_CUTOFF_UTC = datetime.now(timezone.utc) - timedelta(minutes=1)
    try:
        try:
            a.stage_final()
            msg = ""
        except SystemExit as e:
            msg = str(e.code)
        check("past the cutoff no final is made, so the last complete run and its submission stand",
              "Not finalising run t-cutoff" in msg and not asked)
        a.store.put("A1.5_anchor", {"anchor": 58, "market": 58.0, "weight": 0.0})
        a.store.put("A1.6_final", 'x\n{"answer": 58, "own_estimate": 55, "market_reference": 58.0, "reason": "kept"}')
        a.data, a.own = {"A1": {"market_avg_pct": 58.0}}, {"A1": {"p": 55.0}}
        a.stage_final()  # every final already made: nothing to ask, so the run may still submit
        check("a run whose finals are all made is not stopped by the cutoff", a.finals["A1"].endswith('"kept"}')
              and "final" not in asked)
    finally:
        agent.FINALS_CUTOFF_UTC = saved
        shutil.rmtree(a.store.dir, ignore_errors=True)


def test_human_errors():
    import contextlib
    import io
    msg = run_main(["--run", "t-human", "--submit"], {"TEAM_ID": "", "OPENROUTER_API_KEY": ""})
    check("a Terminal window without the .env settings stops at once with what to type",
          msg is not None and "source .env" in msg and "TEAM_ID" in msg)
    ftf = agent.ftf
    saved = (ftf.answers_from_finals, ftf.submit)
    ftf.answers_from_finals, ftf.submit = (lambda run: {}), (lambda answers, run, dry_run=False: {})
    s = agent.Store("t-human")
    for q in ftf.QUESTION_IDS:
        s.put(f"{q}.6_final", 'ok\n{"answer": %s, "own_estimate": 1, "market_reference": 60, "reason": "x"}'
              % (60 if q.startswith("A") else 0.1))
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run_main(["--run", "t-human"])  # first time: starts the run (its finals are already there)
            run_main(["--run", "t-human"])  # again: a complete run
        text = out.getvalue()
        check("a run without --submit says loudly that nothing was submitted", "finished WITHOUT submitting" in text)
        check("rerunning a complete run says it only repeats the submission", "is already complete" in text)
    finally:
        ftf.answers_from_finals, ftf.submit = saved
        shutil.rmtree(s.dir, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{'all passed' if not FAILS else f'{len(FAILS)} failed: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
