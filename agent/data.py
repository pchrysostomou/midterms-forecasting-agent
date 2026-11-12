"""Deterministic data step (no model calls).

Everything here is plain code over the event data pack (data/*.csv) and the
data pack's own live fetchers (ai_only/ftf_tools.py). The numbers it produces
are passed to the models as evidence, so the models never have to recall or
invent prices, yields or market odds.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Past US federal election days in the data pack (election day close -> next day close).
ELECTION_DAYS = {
    "2016-11-08": "presidential",
    "2018-11-06": "midterm",
    "2020-11-03": "presidential",
    "2022-11-08": "midterm",
    "2024-11-05": "presidential",
}

# Part B instruments: data pack column, unit, how a move is measured.
B_INSTRUMENTS = {
    "B1": ("prices", "spy", "%"),
    "B2": ("prices", "iwm", "%"),
    "B3": ("prices", "nvda", "%"),
    "B4": ("prices", "vix", "points"),
    "B5": ("yields", "us_2y", "bp"),
    "B6": ("yields", "us_10y", "bp"),
    "B7": ("yields", "uk_10y", "bp"),
    "B8": ("prices", "gold_front_month", "%"),
    "B9": ("prices", "wti_front_month", "%"),
    "B10": ("prices", "henry_hub_front_month", "%"),
}
B_NAMES = {
    "B1": "S&P 500 (SPY)", "B2": "Russell 2000 (IWM)", "B3": "NVIDIA (NVDA)", "B4": "VIX index",
    "B5": "US 2-year Treasury yield", "B6": "US 10-year Treasury yield", "B7": "UK 10-year gilt yield",
    "B8": "COMEX gold futures (Dec 2026)", "B9": "WTI crude futures (Dec 2026)",
    "B10": "Henry Hub natural gas futures (Dec 2026)",
}


def _pct(x: float) -> float:
    """Market odds can come as 0-1 or 0-100; return percent (plain float, JSON-safe)."""
    x = float(x)
    return x * 100 if x <= 1.0 else x


def _num(x):
    """float, or None for a missing or NaN value."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _hours(start: str, end: str) -> float:
    fmt = "%Y-%m-%d %H:%M"
    return round((datetime.strptime(end[:16], fmt) - datetime.strptime(start[:16], fmt)).total_seconds() / 3600, 1)


# --------------------------------------------------------------------------- Part A
STABLE_BAND = 2.0  # points: the market has "not moved" while its average stays this close to the latest price


def market_snapshot(question_id: str, tools=None) -> dict:
    """Latest Kalshi mid and Polymarket price, their average (the scoring benchmark), 7-day trend, spread and
    volume, plus two facts the critique needs: since when the average has stayed within STABLE_BAND points of
    the latest price (so the market has not reacted to anything newer), and how long ago Kalshi last traded.
    Live hourly data through ftf_tools, data-pack files as fallback."""
    if tools is not None:
        raw = tools.get_market_odds(question_id, days=7, hourly=True)
        source = raw.get("source", "live")
        hourly = [(r["timestamp_utc"], r.get("kalshi_yes_mid"), r.get("polymarket_price"), r.get("kalshi_volume"),
                   r.get("kalshi_spread")) for r in raw.get("hourly") or []]
    else:
        k = pd.read_csv(DATA / "kalshi_odds_hourly.csv")
        p = pd.read_csv(DATA / "polymarket_odds_hourly.csv")
        k, p = k[k.question_id == question_id], p[p.question_id == question_id]
        cutoff = (pd.Timestamp(k.timestamp_utc.iloc[-1]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
        both = k[k.timestamp_utc >= cutoff][["timestamp_utc", "yes_mid", "volume", "spread"]].merge(
            p[p.timestamp_utc >= cutoff][["timestamp_utc", "yes_price"]], on="timestamp_utc", how="outer")
        hourly = [(r.timestamp_utc, r.yes_mid, r.yes_price, r.volume, r.spread)
                  for r in both.sort_values("timestamp_utc").itertuples()]
        source = "data pack files"
    rows = [(str(t), _num(km), _num(pp), _num(vol) or 0.0, _num(sp)) for t, km, pp, vol, sp in hourly]
    rows = [r for r in rows if r[1] is not None or r[2] is not None]
    if not rows:
        raise RuntimeError(f"no market price for {question_id}")

    def last(i):
        return next((r[i] for r in reversed(rows) if r[i] is not None), None)

    def first(i):
        return next((r[i] for r in rows if r[i] is not None), None)

    def avg(*vals):
        vals = [v for v in vals if v is not None]
        return _pct(sum(vals) / len(vals)) if vals else None

    kalshi, poly, spread, as_of = last(1), last(2), last(4), rows[-1][0]
    mid, week_ago = avg(kalshi, poly), avg(first(1), first(2))
    stable_since = rows[0][0]
    for t, km, pp, _, _ in reversed(rows):
        if abs(avg(km, pp) - mid) > STABLE_BAND:
            break
        stable_since = t
    trades = [t for t, _, _, vol, _ in rows if vol > 0]
    return {
        "question_id": question_id,
        "as_of_utc": as_of,
        "source": source,
        "kalshi_mid_pct": None if kalshi is None else round(_pct(kalshi), 1),
        "polymarket_pct": None if poly is None else round(_pct(poly), 1),
        "market_avg_pct": round(mid, 1),
        "kalshi_polymarket_gap_pts": None if kalshi is None or poly is None else round(abs(_pct(kalshi) - _pct(poly)), 1),
        "change_7d_pts": None if week_ago is None else round(mid - week_ago, 1),
        "kalshi_spread_pts": None if spread is None else round(_pct(spread), 1),
        "kalshi_volume_7d": int(round(sum(r[3] for r in rows))),
        "stable_since_utc": stable_since,
        "hours_within_2pts": _hours(stable_since, as_of),
        "hours_since_kalshi_trade": _hours(trades[-1], as_of) if trades else None,
    }


def scenario_probabilities(p_house: float, p_senate: float, p_house_given_senate: float = 0.95) -> dict:
    """Four combined-control scenarios from the two chamber markets (percent in, fractions out).

    Assumption (stated in the README): if Democrats win the Senate they very likely also
    win the House, P(D House | D Senate) = 0.95. The rest follows from the marginals.
    """
    h, s = p_house / 100, p_senate / 100
    d_sweep = s * p_house_given_senate
    r_house_d_senate = s - d_sweep
    d_house_r_senate = max(h - d_sweep, 0.0)
    r_sweep = max(1 - d_sweep - r_house_d_senate - d_house_r_senate, 0.0)
    total = d_sweep + r_house_d_senate + d_house_r_senate + r_sweep
    return {k: round(float(v / total), 3) for k, v in {
        "D_sweep": d_sweep, "D_House_R_Senate": d_house_r_senate,
        "R_sweep": r_sweep, "R_House_D_Senate": r_house_d_senate}.items()}


# --------------------------------------------------------------------------- Part B
def _series(question_id: str) -> tuple[pd.Series, str]:
    kind, col, unit = B_INSTRUMENTS[question_id]
    if kind == "prices":
        df = pd.read_csv(DATA / "yahoo_prices_daily.csv")
        s = df[df.instrument == col].set_index("date").close
    else:
        df = pd.read_csv(DATA / "government_bond_yields_daily.csv").set_index("date")
        s = df[col]
    return s.dropna().sort_index(), unit


def _move(prev: float, cur: float, unit: str) -> float:
    if unit == "%":
        return (cur / prev - 1) * 100
    if unit == "bp":
        return (cur - prev) * 100
    return cur - prev  # VIX points


def _daily_moves(s: pd.Series, unit: str) -> pd.Series:
    vals = s.values
    return pd.Series([_move(vals[i - 1], vals[i], unit) for i in range(1, len(vals))], index=s.index[1:])


def typical_move(question_id: str, to_date: str = "2026-10-30", days: int = 60) -> float:
    """The scoring denominator: median absolute daily change over the 60 trading days to 30 Oct."""
    s, unit = _series(question_id)
    moves = _daily_moves(s[s.index <= to_date], unit).tail(days)
    return round(float(moves.abs().median()), 3)


def election_history(question_id: str) -> list[dict]:
    """Moves from election-day close to next-day close in 2016-2024 (from the data pack)."""
    s, unit = _series(question_id)
    out = []
    for day, kind in ELECTION_DAYS.items():
        if day not in s.index:
            continue
        i = s.index.get_loc(day)
        if i + 1 >= len(s):
            continue
        typ = _daily_moves(s.iloc[max(0, i - 61): i + 1], unit).abs().median()
        mv = _move(s.iloc[i], s.iloc[i + 1], unit)
        out.append({"election": day, "type": kind, "next_day": str(s.index[i + 1]),
                    "move": round(float(mv), 2), "move_in_typical_days": round(float(mv / typ), 2) if typ else None})
    return out


def recent_levels(question_id: str, tools=None, days: int = 10) -> dict:
    """Last few closes, live when possible (ftf_tools), else the data pack."""
    kind, col, unit = B_INSTRUMENTS[question_id]
    rows, source = None, "data pack files"
    if tools is not None:
        try:
            if kind == "prices":
                raw = tools.get_prices(col, interval="1d", days=days + 5)
                rows = [(r["time_utc"], r["close"]) for r in raw["rows"] if r.get("close") is not None]
            else:
                raw = tools.get_yields(days=days + 5)
                rows = [(r["date"], r[col]) for r in raw["daily"] if r.get(col) is not None]
            source = raw.get("source", "live")
        except Exception as e:  # noqa: BLE001 - fall back to the data pack, but say so
            source = f"data pack files (live failed: {e.__class__.__name__})"
    if rows is None:
        s, _ = _series(question_id)
        rows = list(s.tail(days).items())
    rows = rows[-days:]
    return {"source": source, "unit_of_level": "%" if kind == "yields" else "price",
            "last_closes": [[d, round(float(v), 4)] for d, v in rows]}


def b_evidence(question_id: str, tools=None) -> dict:
    """Everything the Part B models get as hard numbers."""
    hist = election_history(question_id)
    moves = [h["move"] for h in hist]
    return {
        "question_id": question_id,
        "instrument": B_NAMES[question_id],
        "unit": B_INSTRUMENTS[question_id][2],
        "typical_daily_move": typical_move(question_id),
        "past_elections_day_after": hist,
        "past_elections_median_move": round(float(statistics.median(moves)), 2) if moves else None,
        "past_midterms_moves": [h["move"] for h in hist if h["type"] == "midterm"],
        "recent": recent_levels(question_id, tools),
    }
