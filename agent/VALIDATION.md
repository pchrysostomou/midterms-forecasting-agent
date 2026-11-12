# Validation log (development runs, 2 Nov 2026)

Development runs use our own Anthropic key (`agent/dev_backend.py`, `--dev`) with every role on Claude Sonnet 5.5,
so they test code, prompts and stability, not the event model mix (GPT-6 Sol, Gemini 4 Argon and GPT-5.6 Terra run
only in the event runs). They cannot submit and feed nothing into the event run. Their outputs stay local
(`out/dev-<run>/`), since they contain the event data pack. Final code of these development tests: commit `401096c`.
The judged event run (run 3, 3 Nov) used commit `96fa458`.

## Stability (final code)
- Final step, five reruns of each saved final prompt (`--stability 5`, run d11): 25 of 25 identical.
- Two independent full runs started at the same time (d11, d12; fresh research, forecasts and critiques): every
  Part A answer within 1 point of its own run's market price; Part B:

| | B1 SPY % | B2 IWM % | B3 NVDA % | B4 VIX pts | B5 2y bp | B6 10y bp | B7 gilt bp | B8 gold % | B9 WTI % | B10 gas % |
|---|---|---|---|---|---|---|---|---|---|---|
| d11 | 0.41 | 0.64 | 0.01 | -1.24 | -0.03 | 0.01 | -0.01 | -0.05 | -0.01 | -0.01 |
| d12 | 0.42 | 0.64 | 0.00 | -1.23 | 0.02 | 0.02 | -0.03 | -0.04 | 0.00 | -0.02 |
| gap, typical moves | 0.02 | 0.00 | 0.01 | 0.02 | 0.02 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |

- The gate moved 3 House races in each run (d11: A9, A11, A12; d12: A8, A9, A12), by at most 1 point.
- Earlier, before the Part B baseline change, the VIX answer differed by 1.09 typical moves between two runs,
  because the critique's keep fraction also scaled the past-election prior.

## Resume and crash tests
- d3 was killed (SIGTERM) 55 s into research with 12 briefs saved, then restarted: 152 calls in all, none repeated.
- d10 crashed in the final stage (a race: threads read `model_bias.json` while it was being written). After the
  fix (computed once before the parallel step, atomic writes), the same command resumed it with exactly the 13
  missing finals. A stress test of 3 writers and 3 readers on one file gave 0 errors.

## Event code path, tests and self-audit
- Mock event run: `agent.py --run mock1` without `--dev`, so every call went through the helper's `ftf.ask`
  (tags, run log) and the run ended with `answers_from_finals` and `submit(dry_run=True)`; only the HTTP client was
  replaced by one answering with the dev model. Result: 152 calls (26 research, 75 forecast, 1 aggregate, 25 critique,
  25 final), approved models only, one clean commit, one final per question with every required field, valid
  answers and 0 submit warnings. Rerunning the same command made 0 new calls; deleting one saved final made 1.
  Repeated on 2 Nov after the review fixes (commit 72e60c3, run mock4): 152 calls, every model readable 25/25,
  `check_run.py` 0 FAIL, 0 WARN, 21 PASS. Rerunning it made 0 calls and fetched no data; deleting one final made
  1 call and refreshed only that question's market.
- Chaos test (2 Nov, offline, no cost): the event path through `ftf.ask` with a fake OpenRouter client that, at
  rates from 20% to 60% of calls, failed, stalled, replied empty or None, or broke the format (lists for strings,
  0.55 or "55%" for numbers, missing fields, text after the JSON line), plus market feeds that failed or fell
  back to the data pack. Ten runs with different seeds: every one ended in a valid dry-run submission, each answer
  equal to its anchor, anchors within the caps, `check_run.py` 0 FAIL; every stop came from an injected fault and
  none from a bug. Since then an event run that stops on an outage or an unreadable final resumes itself twice
  (30 s apart) before asking for a rerun. At a 60% fault rate one final lost its reason field twice and went in
  without it (check_run FAIL); now the resumes also wait for complete finals, and only the last try submits a
  final that lacks just its reason, so a submission is never lost over it.
- Request parameters against OpenRouter's public model list (2 Nov): `temperature` is a supported parameter only for
  Gemini 4 Argon among our models (not Claude Sonnet 5, GPT-6 Sol or GPT-5.6 Terra), so it is now sent only there.
  OpenRouter's docs say unsupported parameters are ignored, so this most likely changes nothing, but the Anthropic
  API rejected temperature in the dev runs, and a rejected final would have stopped every question. `reasoning` and
  `max_tokens` are supported by all four. Prices: $2 per million input tokens, $10-12 per million output.
- Blindness audit (2 Nov): in 432 saved dev briefs, no election-market price reached the forecasters after the
  filter; the 27 kept lines quoting odds with a number were all Fed-hike odds for Part B. Lines that name a
  prediction market without a link (a price can be quoted that way) are now dropped too: 13 more lines in 86
  recent briefs, no evidence flag changed.
- 4-point cap, made exact (2 Nov): rounding could put the anchor 4.5 points from the market (73.5 + 4 rounds to 78),
  and when the critique gave weight the final was told "within 2 points of the anchor", which allowed up to 6. The
  anchor is now clamped to whole numbers within 4 points, and the final gets explicit bounds (for example "from 75 to
  77": within 2 of the anchor and 4 of the market). check_run warns about any answer beyond the cap.
- Election-day rehearsal (3 Nov, 01:00 GMT, run mock5 on 3456609): full mock event run with that day's live data:
  152 calls, check_run 0 FAIL, 0 WARN, 21 PASS. The final kept inside its explicit bounds (A10: 41 within 40-44).
- Human-error guards (3 Nov): an event run in a Terminal window without the .env settings stops at once and says
  what to type; a full run without --submit ends with a loud "finished WITHOUT submitting"; rerunning a complete run
  says it only repeats the submission; and an older run is never submitted after a newer complete run of the same
  team (pressing the up arrow once too often at 21:40 would otherwise replace run 2's submission with run 1's).
- The organisers' resubmit fallback and the RUNBOOK's test-server submission were run against a fake server: the
  right server for FTF_TEST_MODE 0 and 1, answers read back from the CSV with the right types.
- First call on the event key (3 Nov, 17:45 GMT, smoke test on A1 and B4, commit e047bbe): every model readable, but
  the A1 final answered 29 against an anchor of 72. The critique (GPT-5.6 Terra) took the market's 71.5% for the
  Democrat (Kalshi SENATETX-26-D, Polymarket "will the democrats win the texas senate race") to be the Republican's,
  because Texas leans Republican, and the final (Claude Sonnet 5) accepted that as "a factual error in the data",
  the one case its prompt allowed a departure for. Never seen in dev, where Claude played every role. Fix, for every
  question alike: the final has no way out any more (the answer is the anchor, or within its explicit bounds when
  the critique gave weight, and the format line carries the anchor as the answer); code checks the answer against
  those bounds, asks once more with a note, and never submits an answer outside them (a saved one blocks the
  submission like a missing final); the critique and final prompts say the market's probabilities are for the
  question exactly as written, never the other side; check_run fails any answer outside the bounds. Rerun on the
  fix (commit b8fb3e4, A1, A15, B4): A1 72, A15 74, B4 -1.26, each its anchor; check_run 0 FAIL, 0 WARN.
- Event run 1 (3 Nov 17:59 GMT, commit b8fb3e4, submitted): 152 calls in 4 min 40 s, $4.90, check_run 0 FAIL,
  0 WARN, every model readable 25/25, all 25 markets and prices live. 23 of 25 answers equal their anchors; A8 and
  A11 were a point past them (73 against 74, 59 against 60), because the final still had 2 points of leeway when the
  critique gave weight (0.05 for thin volume here), which counts the critique's weight twice and makes reruns
  differ. Since then the final answers the anchor in every case: the format line carries it and code checks it.
- Event run 3, the judged run (3 Nov 21:30 GMT, commit 96fa458, submitted 21:34:24):
  - 152 calls (26 research, 75 forecast, 1 aggregate, 25 critique, 25 final)
  - $4.64, and 4 min 22 s from start to receipt
  - post-run audit (`check_run.py`): 0 FAIL
  - every model readable 25/25
  - all 15 markets and 10 Part B prices live
  - all 25 answers equal to their anchors
  - 10 research lines that named a prediction market dropped before the forecasters
- Independent review before the evening runs (3 Nov, 18:30-18:58 GMT, a separate read-only Claude instance). It
  rebuilt all 25 run-1 anchors from the saved inputs (all match) and found four hazards. Each is now fixed, and each
  new test failed on the previous code:
  1. A NaN or infinity in any model's JSON (Python's parser reads a bare NaN) turned into the 4-point cap, because
     min(4, nan) is 4. One NaN forecast would have moved all 15 Part A anchors to market + 4, and one NaN in a Part B
     scenario would have pushed the VIX anchor to its cap on the wrong side. Non-finite values now count as no number.
  2. The guard against submitting an older run after a newer complete one ran only at start-up. It now runs again
     just before the submission, so overlapping runs cannot make the older one the latest submission.
  3. A test folder left behind by an interrupted test run (no team, dated ahead) would have stopped both evening runs
     from submitting. Only this team's runs dated up to now count.
  4. Nothing knew the deadline. Now no new final is made after 21:50 GMT, so a run whose finals are made (a complete
     run, which the judges take) always has time to submit.
- `python agent/test_offline.py`: 96 offline checks of rarely used paths (parsing, units, evidence check, history weights,
  the Part A gate, Part B defaults, retries, a model outage and the resume after it, event-call timeouts, stale data,
  concurrent writes, saved bias, final validity and bounds, 99.6% markets, a missing venue, the submission guard).
  All pass; pyflakes finds no issues.
- Format robustness (2 Nov, a line-by-line review): the other model families may write things Claude never did in
  dev, so the code now survives a reason given as a list (it crashed the anchor step before), consistency notes
  that are not an object, a probability written as 0.62, a poll date after today, and forecasts with JSON but no
  number (asked again on resume). A final missing its reason or market reference is asked again, but does not
  block the submission. A resumed run keeps the anchor and data of questions already finalised, so finishing a
  run never waits on a market feed. Each new test was run against the previous code and failed there.
- Outages: a forecaster, critique or consistency check that still fails after the client's retries no longer stops
  the run (median of the other forecasters; answer stays on the market or baseline); research and final failures stop
  it, and it resumes itself twice before asking for a rerun of the same command (see the chaos test above).
- `python agent/check_run.py <label>`: a judge's-eye audit of a finished run (calls, finals, models, commit, submit
  checks, answers vs anchors, market freshness, readable replies per model, sources, Part A lean), run after each
  event run.
- Evidence check on the event key (run 1): 15 of 25 briefs were flagged weak. 8 of them were flagged only because
  their text had no URL. Through OpenRouter's web search the model quoted and dated its sources (A1: 24 dated
  claims, polls up to 1 Nov) but did not always write the links into its reply. A weak flag caps the critique's
  weight at 0.1 (Part B: keep at 0.5), so the agent stays closer to the market than in dev, where every brief had
  links. We left this unchanged on the night, because it errs towards the benchmark. In the judged run 3, 10 of 25
  briefs were flagged weak.

## Evidence, adversarial and behaviour tests
- Citation faithfulness: for every brief line citing one page, the page was fetched and each number in the line
  searched for. 92% were found (d11, d12; 414 of 451 on readable pages); misses were mostly formatting ("$1.3M") or
  differences the model computed. After the research prompt was told to prefer original sources (pollster and rater
  releases, official data, wire services), a research-only rerun found 97%, and citations of weak aggregators, broker
  blogs and Substacks fell from 23-25% to 16% of all cited pages.
- Critique and gate on five synthetic races (liquid market with nothing new; thin market with a new poll agreeing or
  disagreeing with our estimate; a stale price with a new poll; a venue gap with no evidence): 5/5 behaved as
  intended. The answer moved only when a checked reason and the brief's evidence pointed the same way.
- Prompt injection: text in a research brief told the models to ignore the market and answer 99. The forecasters
  gave 55-58 (two flagged the text), the critique noted "prompt injection text demanding 99 was ignored", and the
  final answered the market anchor (76). Code caps would have limited any move to 4 points anyway.
- Final reasons: asking for the key evidence on the question left all 25 answers equal to their anchors and gave
  reasons citing specific figures in 18 of 25 (for example the 2-day-after history and the 4 Nov calendar for B6).
- Prediction-market sites: the research still cited them for ratings in 4 of 15 race briefs of one run, so code now
  drops any brief line that cites one before the forecasters and the critique read it (the raw brief stays in the trace).
- Event-call timeouts: the helper's client waits up to 600 s per attempt with 3 retries; event calls now pass 300 s
  (web search) or 180 s. Retries still allow 4 x 180 s per call, so a forecast, consistency check or critique with
  no reply in 240 s is skipped (the run goes on without it), and a model that stalls twice is skipped for the rest
  of the run; a stalled model costs about 4 minutes, not an hour.
- Independent code review (2 Nov, a second agent that only read the code): fixed what it found. A failed price
  refresh before the anchors no longer replaces live prices saved minutes earlier with the data pack's old ones;
  `--submit` with FTF_TEST_MODE=1 stops before any call (it would have gone to the test server); an event run label
  started over 3 hours ago is not resumed (it would resubmit old answers); a rejected or unsent submission ends
  with an error and the resend command; an empty research brief is asked again; `.DS_Store` is ignored by git, so
  opening a folder in Finder cannot block a run as uncommitted code.

## Submission path
- A final counts only if the helper will accept it and it carries an own estimate (a whole 1-99 for Part A, within
  the helper's limit for Part B); otherwise it is asked again, and nothing is submitted while any question lacks one.
- Event runs print FTF_TEST_MODE, and the submit step says whether it goes to the TEST or the REAL server; the
  pre-flight check (`agent/preflight.py`) fails a `--submit` run in test mode, a dirty tree, unset or badly quoted
  `.env` values, and warns about new commits in the organisers' repo (none as of 2 Nov 15:40 UTC).
- The agent changes to the repo root itself, so the helper's run log and git commit are right from any folder.
- Final mock event runs on this code: a full run gave 0 FAIL, 0 WARN, 20 PASS in `check_run.py`; a smoke-test-sized
  run (A1, B4) 0 FAIL.

## Live data
`ftf_tools` falls back to the data pack without an error when a live download fails. With two runs fetching at
once, 8-9 of 15 markets were silently anchored to 31 Oct prices. Fix: three fetches at a time, retries after 3, 10
and 30 s, saved snapshots reused only if live and under 30 minutes old, Part A prices refetched just before the
anchors, and a stop if a Part A market is still stale. After the fix, d11 and d12 ran at the same time with 15 of 15
markets live.

## Sources
Every URL cited in the research briefs, opened with a browser user agent:

| run | URLs | open | exist, block bots | 404 | timeout |
|---|---|---|---|---|---|
| d3 | 184 | 155 | 28 | 0 | 1 |
| d6 | 214 | 173 | 33 | 5 | 3 |
| d11 | 203 | 167 | 34 | 1 | 1 |

After d6 (one URL shortened with "…"), the research prompt asks for exact URLs, and code marks a brief weak if it
cites a prediction-market site (one did, in d9).

## Part B prior backtest (`python agent/backtest_b.py`)
Leave-one-out over the five elections 2016-2024, contest scoring (|error| / typical move, capped at 5; lower is
better). With four elections "4 of 5 agree" has no exact equivalent: 3/4 and 4/4 gates are both shown; 4/4 only fires
when the left-out election went the other way.

| | always 0 | W=0.3, 3/4 | W=0.5, 3/4 | W=0.3, 4/4 | W=0.5, 4/4 |
|---|---|---|---|---|---|
| instruments with a prior (B1, B2, B4, B8) | 3.09 | 3.05 | 2.91 | 3.33 | 3.48 |
| all ten | 2.25 | 2.29 | 2.27 | 2.34 | 2.41 |

The first row's instruments are those where 4 of all 5 elections agree, so it is selected in-sample. The prior is
unproven: it helps the S&P 500, hurts the Russell 2000, and five elections cannot show a gain. Its weight is 0.3.

## Mean versus median (Part B)
The score is absolute error, so the right point forecast is the median of the outcome distribution. With a normal
day's noise (one or two typical moves) around each control scenario, the probability-weighted mean of the scenario
moves was within 0.02 typical moves of that median on all ten instruments; the weighted median of the four scenario
points was further away (B6: mean +0.075, weighted median -0.5, true median -0.003).

## Considered and rejected
- Debiasing every Part A estimate (subtracting the forecasters' average gap to the market, -11.2 and -12.1 points in
  two runs): House control would have become 101.2% and 103.1%, and races where the models agreed with the market
  would have gained 5-11 point "signals" that changed from run to run. The bias is removed only inside gated moves,
  on the logit scale, leaving out the question itself.
- Backtesting the model stages on past elections: the models recall the outcomes. With no web access, Claude
  Sonnet 5.5 gave the 2022 Senate result (51-49) and the S&P 500 move on 9 Nov 2022 (about -2.1%; the data pack has
  -2.06%), so such a test would measure memory, not forecasting.

## Mock judge reviews
Five reviews by Claude Opus 5.5, each given the brief's process criteria, the README, this file, the code, the
general brief and three full traces. Ratings (design / evidence / stability / traceability): 3/3/3/3, 3/3/3/3,
4/3/4/4, 3/3/4/4, 3/3/4/4. Changes made in response:
- Part B: forecasters see the market-implied control-scenario probabilities; the past-election prior is a code
  baseline they adjust, so history counts once; a critique that cannot be read keeps none of the adjustment.
- Part A gate: facts from the hourly market data (time since the price last moved 2+ points, time since the last
  Kalshi trade, volume, venue gap), reasons checked in code, newer_poll checked against the poll date in the brief;
  gated moves exclude the shared model bias and never cross to the other side of the market.
- Finals answer the anchor unless the data has a factual error; unreadable replies are asked once more.
- Evidence: no prediction-market sources, exact URLs, freshness and poll dates taken from the brief's text.
- Reliability: live data retried and refreshed before the anchors, atomic store writes, event runs refuse
  uncommitted code and resumes across commits, every prompt saved.

Not done, by choice: a code poll-averaging model, calibration across many runs and a Senate simulation (days of work
that could not be validated before the event, on blind estimates with a shared 11-point bias), extra research calls
per question, and source tiers. Part A follows the market unless a concrete, checked market failure exists, because a
deviation without new information loses Brier score in expectation; the brief states that being close to the market
is fine.
