# AI forecasting agent - Forecast the Future 2026 (US midterms)

This is the agent I built for the AI-only track of Poros Foundry's "Forecast the Future: US Midterm Election Night
Edition" on 3 November 2026. It won **Best AI agent process**, placing 1st of 8 AI-only teams with 4.05/5.

## The event

The event had 25 questions, and answers were due before 22:00 GMT on election day.

- **Part A: 15 election questions.** Each answer is the probability that the Democrat wins, for 7 Senate races,
  6 House races, and control of the House and of the Senate.
  - Scoring: Brier points gained over the prediction markets, using the average of Kalshi and Polymarket at
    22:00 GMT.
  - Zero means matching the market, and higher is better.
- **Part B: 10 market questions.** Each answer is the move from the 3 Nov close to the 4 Nov close of SPY, IWM,
  NVIDIA, the VIX, the US 2- and 10-year yields, the UK 10-year gilt, and gold, WTI and Henry Hub futures.
  - Scoring: |error| / typical daily move, capped at 5.
  - Lower is better.
- **In the AI-only track the agent produces every answer by itself.**
  - Every model call goes through OpenRouter with the organisers' helper, which tags each call with its role and
    git commit.
  - The process prize was judged from those traces, on agent design, evidence quality, stability and
    traceability.

## Results (official)

There were 18 teams: 8 AI-only teams submitted (10 were selected), plus 10 human teams (1 Human only, 9 Human+AI).

**Best AI agent process: 1st, 4.05/5.** The AI-only average was 3.35 and the median 3.40. The agent passed the
integrity and substance checks.

| Criterion | Weight | Score |
|---|---|---|
| Agent design | 40% | 3.75 |
| Evidence quality | 30% | 3.50 |
| Stability | 15% | 5.00 |
| Traceability | 15% | 5.00 |

- **Market forecast (Part B):** 2nd of the AI-only teams, 5th overall, with a scaled error of 2.82.
- **Election forecast (Part A):** 3rd of the AI-only teams, 9th overall, with -0.29 Brier points against the
  market. Only one AI-only team beat the market.

## How it works

Markets are hard to beat. So the agent forms its own view blind to the market, and code decides how far that view
may move the answer. The validation log is in `agent/VALIDATION.md`.

**Pipeline**

`agent/agent.py` runs every step. Each model call goes through the helper's `ftf.ask()` with a role, and every
prompt and output is saved.

    data (code: live Kalshi+Polymarket average, typical daily move, 2016-2024 day-after moves)
      -> research (Claude Sonnet 5 + web: dated, URL-sourced; lines naming a prediction market are dropped)
      -> forecast (Claude Sonnet 5, GPT-6 Sol, Gemini 4 Argon, blind to the market; median)
      -> consistency check (Claude Sonnet 5) -> critique (GPT-5.6 Terra) -> anchor (code) -> final (Claude Sonnet 5)

**Design**

- **Part A: anchor = market + w x (own - market).** The weight w is at most 0.35, and the anchor stays within
  4 points of the market.
  - w = 0 unless the critique names a market failure that code confirms:
    - no Kalshi trade for 6 or more hours
    - under 100k contracts traded in a week
    - a gap of 3 or more points between Kalshi and Polymarket
    - a poll newer than the last 2-point price move
  - Weak research caps w at 0.1.
  - "own" has the models' shared lean removed, measured on each run.
- **Part B: anchor = W x median past-election move + 0.5 x keep x the forecasters' adjustment**, capped at 3
  typical moves. W = 0.3 if 4 of the 5 past elections agree on the direction, 0.5 if all 5 do, and 0 otherwise.
- **The final answers the anchor and cites the evidence, and code checks it.** The first test on the event key
  caught a critic misreading a market's direction: it answered 29 against an anchor of 72.
- **Failed calls are retried or skipped**, and an interrupted run resumes itself without repeating calls.

**Data and tools**

- the event data pack
- the kit's fetchers (Kalshi, Polymarket, Yahoo Finance, US Treasury, Bank of England)
- models and web search only via OpenRouter
- Python and pandas

The code was written with Claude Code. Development runs used every role on Claude Sonnet 5.5 with my own Anthropic
key (`--dev`). Those runs cannot submit and feed nothing into the event runs.

**Checks**

- Final reruns were 25/25 identical.
- Two full development runs agreed within 1 point (Part A) and 0.02 typical moves (Part B).
- 92-97% of cited numbers were found on the cited page.
- 96 offline tests, a chaos test with up to 60% injected faults, two independent code reviews, and an audit after
  each event run.
- The Part B prior is unproven: the backtest gave 3.05, against 3.09 for always answering 0.

**Judged run:** event run 3 on 3 November 2026, at commit `96fa458`.
- 152 model calls
- cost $4.64
- 4 min 22 s from start to receipt
- post-run audit (`agent/check_run.py`): 0 FAIL

## Running it

The agent plugs into the organisers' public kit, which has the helper (`ai_only/ftf.py`), the data fetchers and the
data pack. The data pack belongs to the organisers and is not included here.

```bash
git clone https://github.com/poros-foundry/forecast-the-future-2026
cd forecast-the-future-2026
cp -r /path/to/ftf-2026-agent-public/agent .
python -m venv .venv && source .venv/bin/activate
pip install -r /path/to/ftf-2026-agent-public/requirements.txt
python agent/test_offline.py
```

`test_offline.py` needs no keys and costs nothing.

**Development runs** use every role on Claude with your own Anthropic key, and they can never submit:

```bash
export ANTHROPIC_API_KEY=...
python agent/agent.py --dev --run d1 --only A1,B4
python agent/agent.py --dev --run d1 --stability 5
```

The second command reruns the saved final prompts 5 times and compares the answers.

**Other scripts**

- `python agent/check_run.py <run>` audits a finished run the way a judge would.
- `python agent/backtest_b.py` reruns the backtest of the Part B prior.

**Event mode** needs the organisers' OpenRouter key and team details, as described in the kit's
`ai_only/README.md`. The 2026 event is closed, so those keys and the submission server no longer work. After
election night the race markets settle, so live market data is no longer meaningful.

## What I'd improve

Most of these points come from the judges' feedback.

1. **Let the agent's own view count, once it is calibrated.**
   - In Part A the critique gave our own forecast zero weight in 11 of 15 races, and at most 0.12 in the rest. So
     the answers ended at the market, and the forecasting steps barely changed them.
   - On the night that caution paid off. The blind forecasts leaned against the Democrats on 10 of 15 races, and
     trading on them would have scored about -7.4 instead of -0.29.
   - The fix is calibration before weight: measure and remove the shared lean on many past questions, then give
     the calibrated estimate a small, fixed weight instead of an all-or-nothing gate.
2. **Feed the consistency check back.**
   - It found that the forecasters' Senate-control estimate (62%) did not match their own race probabilities
     (about 27%).
   - But it ran after the forecasts, and the anchor could only move on a market failure, so the finding changed
     nothing.
   - Better: run it before the critique and send the flagged questions back for a second forecast, or derive
     control of each chamber by simulating the races.
3. **Better evidence (3.5/5).**
   - In the judged run 10 of 25 research briefs were flagged weak, mostly because web-search replies quoted and
     dated their sources without writing the links.
   - Better: require a URL on every claim, and check cited numbers against the pages during the run, not only in
     development.
4. **A better-founded Part B prior.**
   - The past-election prior got a weight of 0.3, because five elections cannot validate it.
   - On the night the directions were right but the sizes too small. The VIX fell 4.22 points, history said 4.2,
     and we answered -1.24.
   - More comparable events, such as other scheduled high-impact days, would show what weight the prior deserves.
5. **Fresher prices for the final.** The answers used the last completed hour of market data, while the benchmark
   was taken at 22:00.

## Files

- `agent/agent.py` - the pipeline, the anchors and the command line (`--run`, `--only`, `--submit`, `--dev`,
  `--stability`)
- `agent/data.py` - the deterministic data step: market snapshots, typical daily moves, past post-election moves
- `agent/check_run.py` - an offline, judge's-eye audit of a finished run
- `agent/preflight.py` - checks before an event run (settings, a clean git tree, test or real mode)
- `agent/backtest_b.py` - the leave-one-out backtest of the Part B prior over the 2016-2024 elections
- `agent/test_offline.py` - offline checks of rarely used paths (parsing, caps, retries, resume, the submission
  guard)
- `agent/dev_backend.py` - the development backend on your own Anthropic key, which can never submit
- `agent/VALIDATION.md` - the validation log: stability, crash and resume tests, the chaos test, the blindness
  audit, the backtest

`VALIDATION.md` refers to commits in the private development history, which is not published here.

## License

MIT, see `LICENSE`. The organisers' kit and data pack are not part of this repository and have their own terms.
