"""Part B backtest of the past-election prior (code only, no model calls).

For each instrument and each of the five elections 2016-2024 in the data pack, the prior is built from the
other four (leave-one-out) and scored like the contest: |forecast - actual| / typical daily move of the 60 days
before that election, capped at 5. Forecasts: always 0, or W x the other four's median move when enough of
them agree on direction. With four elections, "4 of 5 agree" has no exact equivalent, so both 3 of 4 and 4 of 4
are shown; 4 of 4 only fires when the left-out election is the one that went the other way, its worst case.
Five elections is a small sample: this only checks that the prior does no harm.

Usage (from the repo root): python agent/backtest_b.py
"""
from __future__ import annotations

import statistics

import data

SCORE_CAP = 5.0
CAP_TYPICAL = 3.0  # as B_CAP_TYPICAL in agent.py; for VIX the cap is the past-election median move if larger
VARIANTS = [("always 0", None, 0.0), ("W=0.3, 3/4", 3, 0.3), ("W=0.5, 3/4", 3, 0.5),
            ("W=0.3, 4/4", 4, 0.3), ("W=0.5, 4/4", 4, 0.5)]


def forecast(qid, others, typ, need, w):
    """Forecast in typical moves from the other elections' (move, move in typical moves) pairs."""
    if need is None:
        return 0.0, False
    scaled = [m for _, m in others]
    prior = statistics.median(scaled)
    if sum(1 for v in scaled if v * prior > 0) < need:
        return 0.0, False
    cap = CAP_TYPICAL
    if qid == "B4" and typ:
        cap = max(cap, abs(statistics.median(mv for mv, _ in others)) / typ)
    return max(-cap, min(cap, w * prior)), True


def live_prior(hist):
    """True when the live rule (agent.py: 4 of the 5 elections agree on direction) uses a prior."""
    scaled = [h["move_in_typical_days"] for h in hist]
    median = statistics.median(scaled)
    return sum(1 for v in scaled if v * median > 0) >= 4


def main():
    totals = {name: [] for name, _, _ in VARIANTS}
    live = {name: [] for name, _, _ in VARIANTS}
    print("| Q | " + " | ".join(name for name, _, _ in VARIANTS) + " | prior used (3/4, 4/4) |")
    print("|---|" + "---|" * (len(VARIANTS) + 1))
    for qid in data.B_INSTRUMENTS:
        hist = [h for h in data.election_history(qid) if h["move_in_typical_days"] is not None]
        row, used = {}, {3: 0, 4: 0}
        for name, need, w in VARIANTS:
            scores = []
            for i, h in enumerate(hist):
                others = [(o["move"], o["move_in_typical_days"]) for j, o in enumerate(hist) if j != i]
                typ = h["move"] / h["move_in_typical_days"] if h["move_in_typical_days"] else None
                f, on = forecast(qid, others, typ, need, w)
                if on and w == 0.3:
                    used[need] += 1
                scores.append(min(abs(f - h["move_in_typical_days"]), SCORE_CAP))
            row[name] = statistics.mean(scores)
            totals[name] += scores
            if live_prior(hist):
                live[name] += scores
        tag = " (live prior)" if live_prior(hist) else ""
        print(f"| {qid} {data.B_NAMES[qid]}{tag} | " + " | ".join(f"{row[n]:.2f}" for n, _, _ in VARIANTS)
              + f" | {used[3]}/{len(hist)}, {used[4]}/{len(hist)} |")
    print("| **all 10** | " + " | ".join(f"**{statistics.mean(totals[n]):.2f}**" for n, _, _ in VARIANTS) + " | |")
    print("| **live-prior instruments** | " + " | ".join(f"**{statistics.mean(live[n]):.2f}**" for n, _, _ in VARIANTS)
          + " | |")


if __name__ == "__main__":
    main()
