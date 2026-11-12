"""Pre-flight check before an event run: catches setup mistakes in a minute. Makes no model calls.

Usage (from the repo root, after `set -a; source .env; set +a`): python agent/preflight.py [--submit]
Never prints secret values, only whether they are set and well formed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_only"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
results = []


def report(level, text):
    results.append(level)
    print(f"{level:4} {text}")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


submitting = "--submit" in sys.argv

# --- where and how it runs
report("PASS" if Path.cwd() == ROOT else "FAIL", f"running from the repo root ({Path.cwd()})")
report("PASS" if sys.version_info >= (3, 11) else "FAIL", f"Python {sys.version.split()[0]}")

# --- .env: set, well formed, never printed
env_file = ROOT / ".env"
for name in ("OPENROUTER_API_KEY", "TEAM_ID", "SUBMIT_CODE", "TEAM_NAME"):
    value = os.environ.get(name, "")
    report("PASS" if value else "FAIL", f"{name} is {'set' if value else 'NOT set (source .env first)'}")
    if value and (value != value.strip() or re.search(r"[\s\"'`$\\]", value) and name != "TEAM_NAME"):
        report("FAIL", f"{name} has spaces, quotes or shell characters; put it in single quotes in .env")
    if value and name != "TEAM_NAME" and not value.isascii():  # TextEdit turns ' into ’, which then stays in the value
        report("FAIL", f"{name} has non-ASCII characters (curly quotes from TextEdit?); retype it in .env with plain ' quotes")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        m = re.match(r"^([A-Z_]+)=(.*)$", line)
        if m and m.group(2)[:1] in (" ", "\t"):
            report("FAIL", f".env: {m.group(1)} has a space after '=', so `source` drops it")
key = os.environ.get("OPENROUTER_API_KEY", "")
if key:
    report("PASS" if key.startswith("sk-or-") else "WARN", "OpenRouter key looks like an OpenRouter key (sk-or-...)")
mode = os.environ.get("FTF_TEST_MODE")
if submitting:
    report("FAIL" if mode == "1" else "PASS",
           f"FTF_TEST_MODE={mode}: a --submit run goes to the {'TEST server and will NOT count' if mode == '1' else 'REAL server'}")
else:
    report("PASS", f"FTF_TEST_MODE={mode} (no submission in this run)")

# --- code state
report("PASS" if not git("status", "--porcelain") else "FAIL", f"git tree clean at {git('rev-parse', '--short', 'HEAD')}")
try:
    fetched = subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT, capture_output=True, timeout=60).returncode == 0
except subprocess.TimeoutExpired:  # a slow hotspot must not stop the other checks
    fetched = False
upstream = git("log", "--oneline", "HEAD..origin/main")
if not fetched:
    report("WARN", "could not reach GitHub to check the organisers' repo for updates (network)")
else:
    report("WARN" if upstream else "PASS", "organisers' repo has no new commits" if not upstream
           else f"organisers pushed updates, review and merge them first:\n{upstream}")
tests = subprocess.run([sys.executable, str(ROOT / "agent" / "test_offline.py")], cwd=ROOT, capture_output=True, text=True)
report("PASS" if tests.returncode == 0 else "FAIL", "offline tests: " + (tests.stdout.strip().splitlines() or ["?"])[-1])

# --- services
import ftf  # noqa: E402
import ftf_tools  # noqa: E402
import data  # noqa: E402
if key:
    try:
        left = ftf.credit_left()
        report("PASS" if left is None or left >= 10 else "WARN", f"OpenRouter credit left: {left}")
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"OpenRouter key check failed: {e.__class__.__name__}")
live = []
for q in ("A1", "A14", "A15"):
    try:
        live.append(data.market_snapshot(q, ftf_tools)["source"] == "live")
    except Exception:  # noqa: BLE001
        live.append(False)
report("PASS" if all(live) else "WARN", f"live Kalshi/Polymarket data: {sum(live)} of 3 sample markets")

# --- clock
now = datetime.now(timezone.utc)
deadline = now.replace(year=2026, month=11, day=3, hour=22, minute=0, second=0, microsecond=0)
mins = (deadline - now).total_seconds() / 60
report("PASS" if mins > 45 or mins < 0 else "WARN", f"{now:%a %d %b %H:%M} UTC, {mins:.0f} minutes to the 22:00 GMT deadline")
if submitting and now.date() == deadline.date() and now.hour < 21:
    report("WARN", "before the 21:00 GMT US close: Part B would not see the 3 Nov closing prices")

print(f"\n{results.count('FAIL')} FAIL, {results.count('WARN')} WARN, {results.count('PASS')} PASS")
sys.exit(1 if "FAIL" in results else 0)
