# Strategy registry

This is the list of our strategies, their current stage, and the next step for each. The secretary's briefing reads this file. Update the stage whenever a strategy moves.

Stages, in order: `research` → `validated` → `paper` → `live pilot` → `live` → `retired`.

| Strategy | Underlying | Stage | Evidence | Next step | Docs |
|---|---|---|---|---|---|
| 0DTE iron condor **V2** (±0.5u, $5 wings, 10 entries/day, stop at 0.5× credit) | SPY / QQQ | **validated** | Worst day −$1,255 vs −$5,155 for V1; return per worst-day dollar 20.2 vs 7.8; QQQ pre-registered confirmation passed (t 4.94) | Trader built (`condor_v2_trader.py`; orders + quotes → IBKR; **Alpaca ruled out** (force-closes 0DTE at 15:45); needs a TWS paper session on port 7497) → run paper ≥ 40 sessions, judged on fills | [Trade plan](../../spy_0dte/TRADE_PLAN_condor.md) · [Pre-registration](../../spy_0dte/PREREG_condor_v2_risk.md) |
| 0DTE iron condor V1, held to expiry | SPY | **superseded by V2** (2026-09-22) | Passed the one-time pre-registered holdout: +$7.04 per condor per day, t 2.15, 251 sessions. Development: +$10.14, t 3.90, 403 sessions | Build the paper bot → paper trade ≥ 40 sessions; judge on fills (≤ $0.02 worse than mid) | [Trade plan](../../spy_0dte/TRADE_PLAN_condor.md) · [Pre-registration](../../spy_0dte/PREREG_condor_expiry.md) · [Research conclusion](../../spy_0dte/RESEARCH_CONCLUSION.md) |
| Inside Day Reversal + valuation-gated accumulation | S&P 500 names | **live** (signal: no edge; accumulation: validated) | **The SIGNAL has no edge** — market-adjusted, date-clustered excess +0.088pp, t 0.48; on fresh 407-ticker data `single` is **−0.010pp, t −0.11**. **The DCA STRUCTURE does** — paired on identical signals, market-adjusted, date-clustered: dca_5_10 minus single **+0.629pp (t 9.83, 112 tickers) → +0.630pp (t 16.28, 7,290 signals on a never-touched 407-ticker panel)**. Unchanged to 3 decimals out of sample = structural, not fitted: a lower average cost makes the "first close above average cost" exit reachable more often (86.7% vs 83.0%) and sooner | Sizing set from the tail: `TARGET_DOLLARS_OVERRIDE = $420`/name → −$34 at p1 (2.4% of account), −$65 worst-in-5yr (4.6%). Retimed to **09:15 ET** — at 15:30 an open Ashley 0DTE projects assignment margin and rejects tranches (Error 201). **Valuation gate is NOT backtested** (needs point-in-time fundamentals) | [Sizing study](../backend/candlestick_pattern_research/dca_sizing_study.py) · [Replication](../backend/candlestick_pattern_research/dca_replication_407.py) · [Research log](../backend/candlestick_pattern_research/RESEARCH_LOG.md) |
| Day Trader (same-day open→close, scanner-fed, long only) | S&P 500 names | **disabled 2026-09-29** (was live) | 138,057 ticker-days, market-adjusted + date-clustered: score ≥ 75 excess **+0.017pp, t 0.45**; best gate variant (atr_mult ≥ 1.8) **+0.174pp, t 1.23**; live AND gate **−0.293pp** and fires on only 4.9% of sessions; confirmation gate adds nothing (**−0.004pp, t −0.11**). Score does lift the ≥0.5% *move* hit rate 36.5%→42.7%, but not the *sign* | **Revival tested and REJECTED out of sample 2026-09-29** (pre-registered, sha256 `3071a656…`; 407 unseen tickers / 497,820 ticker-days: excess **+0.045pp, t 0.72** vs required t>2.0 and >0.47pp; in-sample +0.203pp shrank 78%). Holdout spent. **Direction search EXHAUSTED 2026-09-30**: 9 independent sources, ~57 features (daily bars, intraday microstructure, options flow, dealer gamma, NOPE, dark pool, market timing, prior-session price-level positioning, unusual options activity, plus GBM/ridge interactions). Nothing cleared the 0.473pp fee with t>2 in any source; pre-registered holdout never spent. Noise floor established: shuffled labels yield +0.184pp at t 1.39 through the same pipeline. Options flow is **coincident** (corr +0.29 with the move already made, −0.013 with the move to come). Only untested avenues are ones we cannot obtain — order-book depth, tick prints with aggressor flags, sub-second news. Trade the *move* (non-directional instrument, blocked by account size) or nothing; do not re-tune. Also blocked on account size. Live position is 10% of net liq ≈ $148; IBKR's $1.00 minimum makes a fixed $2.00 round trip = **1.35%**, ~8× the best measured edge. Revisit only when net liq supports ≥ $2,000 positions (≈ $20k at 10% sizing). Do not retune thresholds | [Research log](../backend/daytrader_research/RESEARCH_LOG.md) · [Gate study](../backend/daytrader_research/gate_threshold_study.py) · [Confirmation study](../backend/daytrader_research/confirmation_gate_study.py) |

## SPY 0DTE iron condor, held to expiry

**Rules**
- **Entries:** one new condor every 15 minutes, 10:01 to 14:31, on SPY 0DTE options (19 per day).
- **Short strikes:** SPY ± 1.0 expected move, where u = SPY × ATM IV × √(minutes to close ÷ 98,280).
- **Wings:** $3 further out on each side.
- **Exit:** held to 16:00 expiry, with no stop and no target.

**Economics per unit** (19 condors per day, holdout year, λ = 0.25, $0.65 per contract)
- **Profit:** about +$119 per day, about $30k per year.
- **Win rate:** 82% of condors.
- **Worst case:** worst day −$4,861, max drawdown −$9,960.
- **Capital:** about $25–33k.
- **Edge per condor:** about $7 against about $280 at risk.

**Replication (2026-09-22):** QQQ, run on fresh data under a pre-registered test, **passed**: +$9.42 per condor per session, t 4.22, 654 sessions. IWM did not pass (t 0.25); its $1 wings are small relative to commission.

**After-hours exercise correction (2026-09-22):** SPY options are American and physically settled. Holders can decide whether to exercise after seeing after-hours moves, until 17:30.
- **Worst case**, where every holder exercises optimally: the SPY edge drops to **+$4.83 per condor per session (t 2.17)**, and QQQ drops to +$2.99 (t 1.21).
- **Realistic edge:** about **$5–8 per condor** on SPY.
- **SPX (cash-settled, no after-hours exercise), 494 sessions:** +$3.88 per SPY-sized condor, t 1.56. The sign is positive, but it failed the pre-registered bar of t ≥ 2.
- **Best estimate of the true edge:** **about $4–6 per SPY-sized condor**, positive on all three underlyings.
- **Late-window structures rejected** (entries 14:45–15:30, held to expiry): they look strong on SPY and QQQ (t ≈ 5.9), but vanish on SPX and under after-hours exercise. That apparent premium is payment for the holder's after-hours exercise right, not a real edge.

**Open risks**
- **Real fills vs modelled:** the edge falls to t 1.82 if fills are half a width worse.
- **After-hours exercise cost on SPY/QQQ** (see above).
- **SPY physical settlement:** after-hours assignment risk.
- **Commission:** $2.60 of about $9.50 gross per condor.

**Where it came from:** four development rounds were rejected, including regime ML, the 15:45 exit, and the last hour. The key finding was that buying back at 15:45 threw the edge away.
