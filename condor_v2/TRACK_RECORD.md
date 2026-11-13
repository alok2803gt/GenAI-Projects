# Condor V2: track record and backtest evidence

This file is for anyone, person or AI, running V2 on a machine that doesn't have this laptop's logs. The full paper history and the backtests behind the rules are summarised here. **V2 is not an untested strategy.** It has been backtested on 2 years of minute data and paper-traded for 30 sessions on IBKR. Read this before deciding that a losing day or a losing week means something is broken.

Last updated 2026-11-12, after the 20-session go-live rehearsal.

## Rules being traded (as of 2026-11-12)

| | |
|---|---|
| Underlying | SPY 0DTE, 1 contract per condor |
| Entries | 11:01, 11:31, 12:01, 12:31, 13:01, 13:31, 14:01, 14:31 ET (8 per day) |
| Strikes | Shorts at spot ± 0.5u, wings $5 further out. `u = spot × ATM IV × √(minutes to 16:00 ÷ 98,280)`. Late in a low-IV day the shorts sit only 1–3 points apart. **That is the design, not a bug.** |
| VIX hold | Skip a slot while VIX's last completed 30-min bar is ≥ 5% above today's VIX open (`vix_hold_pct`, added 2026-11-12) |
| Stop | Close the whole condor the first time its loss reaches 0.5 × the credit |
| Exit | Otherwise hold to expiry. 15:57: buy back any short in the money or within $0.25 of spot (assignment protection) |
| Daily stop | No new entries once the day is at −$1,500 |
| Capital plan | $15,000 for 1 lot |

## Backtest evidence

### 1. Original validation (README.md)
- **Data:** SPY, 660 sessions.
- **Result:** +$101/day per lot, worst day −$1,255, max drawdown −$5,224.
- **Out-of-sample:** QQQ passed a pre-registered confirmation (t = 4.94).
- **Hold to expiry matters:** closing at 15:45 instead backtests at −$0.38/day, against +$8.37/day.

### 2. Replay at real costs (`replay_backtest.py`, 258 days, 2025-09-29 → 2026-10-09)
Replayed on 1-minute trade prices. Costs were measured on real paper fills: entry about $1, stop exit $2–5, commissions $3.30–5 per condor.

| Rules (1 lot, net of costs) | Net / year | Max drawdown | Worst day |
|---|---|---|---|
| **0.5× stop, entries 11:01+ (chosen 2026-10-14)** | **+$7.3k** | **−$4.5k** | **−$1.1k** |
| 1× stop, entries 11:01+ | +$7.9k | −$6.1k | −$1.8k |
| No stop, entries 11:01+ | +$14.7k | −$7.6k | −$3.1k |
| 0.5× stop, all 10 slots (10:01+) | +$4.7k | −$6.2k | −$1.4k |

- **Why the morning slots were dropped:** the 10:01 and 10:31 entries lost money in **both** half-years, and the 11:01–14:31 entries made money in both.
- **Why not run with no stop:** the 5 worst days sum to −$15.5k, more than the whole $15k plan.

### 3. VIX hold (262 days, 2024-10-18 → 2026-10-16, entries 11:01+, 0.5× stop)
Each rule uses only the VIX bar that had finished before the entry, so there's no lookahead.

| Hold new entries when… | $/day | Max drawdown | Net, 1st half / 2nd half |
|---|---|---|---|
| no hold | +$33.6 | −$4,537 | +$5,676 / +$3,119 |
| **VIX ≥ today's open +5% (chosen)** | **+$49.0** | **−$2,805** | **+$7,145 / +$5,692** |
| VIX ≥ today's open +8% | +$48.7 | −$3,366 | +$8,146 / +$4,609 |
| VIX ≥ yesterday's close +5% | +$40.1 | −$3,809 | +$5,375 / +$5,143 |
| VIX level ≥ 20 | +$33.9 | −$3,362 | +$3,280 / +$5,595 |

- **What it skips:** the condors it skipped averaged −$12.47 each.
- **Overfitting caveat:** this rule was the best of 12 variants tested on the same data. Expect the live gain to be smaller than the table shows.
- **Rehearsal check:** it would have held entries on 10/14, 10/30, 11/05 and 11/10.
- **What it doesn't cover:** calm-VIX rally days such as 11/09.

## Paper trading history (IBKR paper account, real fills)

Net is after commissions. "Real-mkt" is the gross re-priced at recorded bid/ask, an estimate of live fills. Rows marked *rebuild* were reconstructed from logs before the end-of-day tracker existed (10/23). Rows after that come from IBKR fills.

| Date | Rules | Entered | Stopped | Unwound | Held | Net $ | Real-mkt gross $ | Cumulative $ |
|---|---|---|---|---|---|---|---|---|
| 10-01 | early build | 2 | 2 | 0 | 0 | +2 | | +2 |
| 10-02 | early build | 5 | 2 | 1 | 3 | +170 | | +172 |
| 10-05 | early build | 9 | 9 | 0 | 0 | −494 | | −322 |
| 10-06 | v1 0.5× 10 slots | 10 | 2 | 0 | 8 | +232 | | −90 |
| 10-07 | v1 0.5× 10 slots | 8 | 4 | 2 | 4 | +18 | | −72 |
| 10-08 | v1 0.5× 10 slots | 10 | 7 | 0 | 3 | −352 | | −424 |
| 10-09 | v1 0.5× 10 slots | 8 | 5 | 0 | 3 | −53 | | −477 |
| 10-12 | v1 0.5× 10 slots | 10 | 2 | 0 | 5 | +203 | +276 | −274 |
| 10-13 | 1× stop, 11:01+ | 8 | 3 | 0 | 5 | +95 | +124 | −179 |
| **10-14** | **FINAL rules (session 1)** | 8 | 4 | 0 | 4 | −76 | −92 | −255 |
| 10-15 | FINAL | 7 | 4 | 1 | 3 | −60 | −64 | −315 |
| 10-16 | FINAL | 6 | 5 | 2 | 1 | −262 | −277 | −577 |
| 10-19 | FINAL | 7 | 1 | 1 | 6 | +285 | +232 | −292 |
| 10-20 | FINAL | 7 | 2 | 1 | 5 | +267 | +274 | −25 |
| 10-21 | FINAL | 8 | 6 | 0 | 2 | −268 | −269 | −293 |
| 10-22 | FINAL | 6 | 1 | 2 | 5 | +438 | +436 | +145 |
| 10-23 | FINAL | 6 | 0 | 2 | 6 | +408 | +234 | +553 |
| 10-26 | FINAL | 7 | 2 | 1 | 5 | +254 | +168 | +807 |
| 10-27 | FINAL | 8 | 0 | 0 | 8 | +664 | +461 | +1,471 |
| 10-28 | FINAL | 5 | 0 | 2 | 5 | +874 | +879 | +2,344 |
| 10-29 | FINAL | 7 | 1 | 1 | 6 | +152 | +291 | +2,497 |
| 10-30 | FINAL | 8 | 7 | 0 | 1 | −613 | −609 | +1,883 |
| 11-02 | FINAL | 5 | 4 | 0 | 1 | −244 | −404 | +1,640 |
| 11-03 | FINAL | 7 | 1 | 1 | 6 | +405 | +503 | +2,045 |
| 11-04 | FINAL | 4 | 0 | 1 | 4 | +121 | +128 | +2,167 |
| 11-05 | FINAL (paused: stray leg, no condors) | 0 | 0 | 0 | 0 | −72 | | +2,095 |
| 11-09 | FINAL | 8 | 8 | 0 | 0 | −590 | −667 | +1,505 |
| 11-10 | FINAL | 6 | 1 | 2 | 5 | −306 | −75 | +1,199 |
| 11-11 | FINAL | 7 | 3 | 1 | 4 | −2 | −29 | +1,196 |
| 11-12 | FINAL (session 20) | 8 | 4 | 0 | 4 | −301 | −305 | +895 |

(11/06 is missing because the bot didn't run that day: the pause flag set on 11/05 was still in place. 11/05 is on the tracker but not on the 20-session scorecard.)

### Go-live rehearsal summary (final rules, 10/14 → 11/12)

| | Rehearsal | Backtest expectation |
|---|---|---|
| Net | **+$1,074** over 21 trading days (+$1,146 on the 20-session scorecard) | ~+$34–101/day |
| Average day | ~+$51 | +$34 (cost replay) to +$101 (original) |
| Winning / losing days | 10 / 11 | about half, with winners bigger than losers |
| Best / worst day | +$874 / −$613 | worst −$1.1k to −$1.3k |
| Max drawdown | −$1,601 (from +$2,676 on 10/29) | −$4.5k to −$5.2k |
| Condors | 135 entered, 54 stopped (40%), 18 partial entries unwound, 81 held to expiry | stop rate ~42–49% |
| Commissions | $567 | |
| Peak margin | $7,632 (11/10) | ~$4,100 typical |

**Verdict:** everything in the rehearsal is within the backtest's range. The losing streak from 10/30 to 11/12 came on trending, low-IV days, a known weakness of narrow 0DTE condors. It is smaller than the drawdowns in the backtest.

## What is normal, and what is a real red flag

**Normal, so keep trading:**
- Losing days, including several in a row. The backtest had 4 negative months out of 12.
- A drawdown up to about $5k on 1 lot.
- Every condor stopped on a strong trend day (10/30, 11/09).
- Partial entries unwound for a few dollars.
- VIX-hold days with few or no entries.

**Real red flags, so stop and investigate:**
- A drawdown beyond −$5.5k (worse than anything in the backtest).
- Stops that cost much more than 0.5× the credit, i.e. fills far worse than the trigger.
- `PAUSED.flag` appearing.
- Shares in the account in the morning (an assignment the 15:57 pass missed).
- A `vix_unavailable` line in the log every day (the VIX hold is silently off).
- The logged-in account not matching config (the bot refuses to start).

## Incidents in paper, and the fixes now in the code

| Date | What happened | Fix |
|---|---|---|
| 10-02 | Laptop slept on battery; no stops or closes for 3 h | tasks run on battery; battery alerts; trader keeps the PC awake |
| 10-06 | IBKR liquidated near-money shorts itself before expiry; the bot re-bought some | positions reconciled against the account every cycle from 14:45 |
| 10-07, 10-27, 10-30, 11-02 | Assignments on shorts at or near the money | 15:57 pass covers in-the-money shorts and those within $0.25; retries until 15:59 |
| 10-15 | TWS sat at its login screen after the overnight restart | connection retries with HIGH alerts; IB Gateway + IBC for paper |
| 10-30 | A cancel stuck in PendingCancel crashed the bot | stuck orders tracked per contract; no crash |
| 11-05 | Partial entry couldn't be unwound; bot paused, a stray 727P was left | pause works as designed; a human clears `PAUSED.flag` |
| 11-12 | Not an incident: added the VIX hold and the check that the logged-in account matches config | — |

## Where the raw data lives (on the paper laptop only; gitignored)

- `condor_daily_tracker.csv`: the table above, one row per day from IBKR fills (`condor_eod_tracker.py`, 16:20).
- `golive_scorecard.jsonl`: the 20-session rehearsal scorecard (`monitor.py`).
- `condor_v2_log.jsonl`: every entry, stop, unwind and close.
- `iv_study.json`, `replay_cache/`: per-condor backtest rows and minute data behind the replay tables.
- `shadow_ledger.jsonl`: paper fills re-scored against recorded bid/ask.

The scripts that produced the tables are in this folder: `replay_backtest.py`, `iv_filter_backtest.py`, `condor_eod_tracker.py`. A live machine starts its own tracker and log from day 1. Compare them with this file, not with an empty history.
