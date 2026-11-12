"""Self-audit of a finished run, the way a judge would read it. No model calls.

Usage (from the repo root, with TEAM_ID set): python agent/check_run.py <run_label>
Reads the helper's run log (.ftf/runs/<TEAM_ID>-run-<label>.jsonl) and the agent's saved stages (out/run-<label>/),
and prints PASS / WARN / FAIL lines. Run it right after each event run, before 22:00.
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_only"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ftf  # noqa: E402
from agent import MODELS, answer_band, final_answer, last_json  # noqa: E402

label = sys.argv[1]
out = ROOT / "out" / f"run-{label}"
results = []


def _valid_one(q, value):
    try:
        ftf.validate_answers({**{k: (50 if k.startswith("A") else 0) for k in ftf.QUESTION_IDS}, q: value})
        return True
    except ValueError:
        return False


def report(level, text):
    results.append(level)
    print(f"{level:4} {text}")


def load(name):
    p = out / f"{name}.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


calls = ftf.run_calls(label)
if not calls:
    sys.exit(f"No run log for run {label} (TEAM_ID={ftf._env('TEAM_ID')}); run from the repo root.")

# --- integrity and tagging
roles = collections.Counter(c["role"] for c in calls)
qs = [q for q in ftf.QUESTION_IDS if q in {c["question"] for c in calls}]  # questions this run asked about
partial = len(qs) < len(ftf.QUESTION_IDS)
report("PASS" if len(calls) >= 152 or partial else "WARN", f"{len(calls)} calls: {dict(roles)} "
       + (f"(partial run: {', '.join(qs)})" if partial else "(a full run makes 152)"))
finals = collections.Counter(c["question"] for c in calls if c["role"] == "final")
missing = [q for q in qs if q not in finals]
report("FAIL" if missing else "PASS", "final call for every question" + (f"; missing: {missing}" if missing else ""))
repeated = {q: n for q, n in finals.items() if n > 1}
report("WARN" if repeated else "PASS", "one final call per question" + (f"; repeated: {repeated}" if repeated else ""))
researched = {c["question"] for c in calls if c["role"] == "research"}
report("PASS" if "general" in researched or set(qs) <= researched else "FAIL", "research covers every question")
bad_models = {c["model"] for c in calls} - set(ftf.APPROVED_MODELS.values())
report("FAIL" if bad_models else "PASS", "only approved models" + (f"; others: {bad_models}" if bad_models else ""))
commits = {c["git_commit"] for c in calls}
dirty = any(c["git_dirty"] for c in calls)
report("FAIL" if len(commits) != 1 or dirty else "PASS", f"one clean git commit: {sorted(str(c)[:8] for c in commits)}, dirty={dirty}")
times = sorted(c["time"] for c in calls)
mins = (datetime.fromisoformat(times[-1]) - datetime.fromisoformat(times[0])).total_seconds() / 60
report("PASS", f"calls from {times[0]} to {times[-1]} ({mins:.0f} min)")

# --- answers, exactly as submit() will see them
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    answers = ftf.answers_from_finals(label)
    try:
        ftf.validate_answers(answers)
        valid = None
    except ValueError as e:
        valid = str(e)
    warnings = ftf._run_checks(answers, label, calls)
if partial:  # a test run: check its own answers, not the 25 a submission needs
    with contextlib.redirect_stdout(buf):
        bad = [q for q in qs if not _valid_one(q, answers.get(q))]
    report("FAIL" if bad else "PASS", "answers valid for submit()" + (f"; not: {bad}" if bad else " (asked questions)"))
else:
    report("FAIL" if valid else "PASS", "answers valid for submit()" + (f": {valid}" if valid else ""))
    report("WARN" if warnings else "PASS", "submit() run checks" + (f": {warnings}" if warnings else ": no warnings"))
fields = [c["question"] for c in ftf._last_finals(calls).values()
          for k in ("answer", "own_estimate", "reason") + (("market_reference",) if c["question"].startswith("A") else ())
          if (c.get("final") or {}).get(k) in (None, "")]
report("FAIL" if fields else "PASS", "final JSON has answer, own_estimate, reason (+ market_reference)" + (f"; missing in {fields}" if fields else ""))

# --- the agent's own stages
for q in qs:
    a, ans = load(f"{q}.5_anchor"), answers.get(q)
    if a is None or ans is None:
        continue
    lo, hi = answer_band(q, a)
    if not lo - 1e-9 <= ans <= hi + 1e-9:  # the agent asks such a final again, so this should never happen
        report("FAIL", f"{q}: answer {ans} differs from the anchor {a['anchor']} (market {a.get('market')})")
markets = [m for m in (load(f"{q}.0_data") for q in ftf.QUESTION_IDS if q.startswith("A")) if m]
stale = [m.get("question_id") for m in markets if m.get("source") != "live"]
report("FAIL" if stale else "PASS", f"all {len(markets)} market snapshots live" + (f"; stale: {stale}" if stale else ""))
prices = {q: d for q in ftf.QUESTION_IDS if q.startswith("B") and (d := load(f"{q}.0_data"))}
old_prices = [q for q, d in prices.items() if (d.get("recent") or {}).get("source") != "live"]
report("WARN" if old_prices else "PASS", f"Part B recent prices live for {len(prices) - len(old_prices)} of {len(prices)}"
       + (f"; from the data pack: {old_prices}" if old_prices else ""))
ages = [m.get("fetched_utc") for m in markets if m.get("fetched_utc")]
if ages:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(min(ages))).total_seconds() / 60
    report("PASS" if age < 90 else "WARN", f"oldest market fetch {age:.0f} min ago")
parsed = collections.Counter()
total = collections.Counter()
for q in qs:
    for f in load(f"{q}.2_forecast") or []:
        name = f["model"].split(" [")[0]
        total[name] += 1
        parsed[name] += f.get("json") is not None
for name in total:
    report("PASS" if parsed[name] == total[name] else "WARN", f"forecasts readable from {name}: {parsed[name]}/{total[name]}")
crit = sum(last_json(load(f"{q}.4_critique") or "") is not None for q in qs)
report("PASS" if crit == len(qs) else "WARN", f"critiques readable: {crit}/{len(qs)} ({MODELS['critique']})")
unreadable = [q for q in qs if final_answer(load(f"{q}.6_final")) is None]
report("FAIL" if unreadable else "PASS", "saved finals readable" + (f"; not: {unreadable}" if unreadable else ""))
weak = [q for q in qs if (load(f"{q}.1_evidence_check") or {}).get("weak")]
pm = [q for q in ftf.QUESTION_IDS if (load(f"{q}.1_evidence_check") or {}).get("prediction_market_sources")]
dropped = {q: n for q in ftf.QUESTION_IDS if (n := (load(f"{q}.1_evidence_check") or {}).get("market_lines_dropped"))}
report("PASS", f"weak research briefs: {len(weak)}/{len(qs)} {weak}")
report("WARN" if pm else "PASS", "forecasters saw no prediction-market sources" + (f"; seen in {pm}" if pm else "")
       + (f" (lines dropped before them: {dropped})" if dropped else ""))
tilt = (load("summary") or {}).get("part_a_tilt") or {}
if tilt:
    report("WARN" if tilt.get("warning") else "PASS", f"Part A lean: {tilt.get('above_market')} above, "
           f"{tilt.get('below_market')} below the market, net {tilt.get('net_points')} points")
else:
    report("PASS", "Part A lean: not measured (fewer than 4 race questions in this run)")

print(f"\n{results.count('FAIL')} FAIL, {results.count('WARN')} WARN, {results.count('PASS')} PASS")
sys.exit(1 if "FAIL" in results else 0)
