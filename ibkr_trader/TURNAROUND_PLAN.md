# IBKR Algo Trader — Turnaround Plan

**Account status at diagnosis:** Down ~$15,000 on paper trading account  
**Date of audit:** 2026-07-07  
**Target:** Consistent $200/day net profit  

---

## 1. Root Cause Analysis

### 1.1 What Actually Happened (Trade Journal Audit)

| Strategy | Realized P&L | Notes |
|---|---|---|
| DAY_BREAKOUT | -$74 | Structural math issue — 0.5%/7% R/R |
| STOCK_BREAKOUT | -$216 | Laggard positions held too long |
| CSP | -$459 | MRVL stop never fired (bug) |
| LEAP | -$4,676 | QCOM single-trade -$4,208 |
| **Total realized** | **-$5,425** | |
| Open losses (MRVL, QQQ, IWM LEAPs) | ~-$9,000 | Made up remaining ~$7K of -$15K |

### 1.2 Primary Failure: QCOM LEAP (-$4,208)

- **Entry:** $8,172 into a single QCOM long call (≈16% of a $50K account)
- **Outcome:** -52% loss, stopped out 2026-07-02
- **Root cause:** No single-position size limit. One trade wiped months of gains.
- **Rule violated:** Never put >5% of account into one LEAP position.

### 1.3 Secondary Failure: MRVL CSP Cascade (-$3,146 cumulative)

Two compounding bugs caused the CSP to grow from a manageable loss to a catastrophic one:

**Bug 1 — `stop_loss_mult = 0.5`** (wrong value, should be 2.0)  
The stop was configured to fire at 50% of premium received instead of 200%. A $500 premium CSP would stop at -$250 loss instead of -$1,000. On paper this sounds safer, but the problem is it fired on normal premium decay — or in this case, because of Bug 2, it *never* fired at all.

**Bug 2 — `unrealizedPNL = 0` on IBKR paper trading**  
IBKR paper accounts return `unrealizedPNL = 0` for all short options. The monitor was reading this as "flat P&L" and never triggering stops. The position grew from -$769 (where the 0.5× stop should have fired) all the way to -$2,872 before being manually closed.

**Combined effect:** Position was rolled 5× manually trying to "save" it, accumulating more premium risk each roll. Total damage: $3,146.

### 1.4 Tertiary Failure: Day Trader Math

The original day trader configuration had a broken risk/reward ratio:
- **Target:** 0.5% gain per trade
- **Stop:** 7.0% loss per trade  
- **R/R ratio:** 14:1 against
- **Break-even win rate required:** 93.3%
- **Actual win rate:** ~55%

At those parameters, even a 55% win rate loses money. The backtest confirmed all existing configs were net negative over 5 years.

---

## 2. Immediate Actions Taken (2026-07-07)

### 2.1 Bug Fixes

| Bug | Fix |
|---|---|
| `stop_loss_mult = 0.5` | Changed to `2.0` (stop fires at 2× premium received) |
| Paper `unrealizedPNL = 0` | Added mid-price fallback: `(bid + ask) / 2 × qty × 100` |

### 2.2 Positions Closed

| Position | Action | Reason |
|---|---|---|
| MRVL CSP | Closed at loss | Bleeding position, bug-driven overhang |
| QQQ LEAP | Closed at loss | LEAP paused strategy, cut exposure |
| IWM LEAP | Closed at loss | Same — reduce open risk |
| SNOW CSP | Closed at profit | Took available gain, reduce concentration |
| NVDA LEAP | Hold | 11 months runway, decision pending |

### 2.3 Stock Trader Cleanup (flagged for exit if still red next morning)

DE, C, NKE, XYZ, HD, LOW — all positions down $67–$121, candidates to exit.

---

## 3. Strategy Redesign

### 3.1 CSP-Leap Trader — New Rules (Non-Negotiable)

**Prices verified:** July 8, 2026

**Universe:** IWM, XBI, KRE only — everything else is too expensive at current prices (see below)  
**Capital allocated:** $20,000 (40% of $50k account)  
**Max positions:** 7 contracts across 3 ETFs  
**Stop loss:** 2× premium received — no exceptions, no rolls past this point  
**Warn level:** 1.5× premium received (alert sent, prepare to close)  

**Full margin map — why most ETFs are now off the table:**

IBKR Reg-T margin formula: `20% × price × 100 − OTM_amount`

| ETF | Current Price | 5% OTM Strike | OTM Amount | Reg-T Margin | Status |
|---|---|---|---|---|---|
| SPY | $751 | $713 | $3,800 | **$11,220** | ❌ Off table — 22% of account |
| QQQ | $722 | $686 | $3,600 | **$10,840** | ❌ Off table — 21% of account |
| SMH | $604 | $574 | $3,000 | **$9,080** | ❌ Off table — 18% of account |
| IWM | $295 | $280 | $1,500 | **$4,400** | ✅ Primary — 8.8% of account |
| XBI | $160 | $152 | $800 | **$2,400** | ✅ Secondary — 4.8% of account |
| KRE | $75 | $71 | $375 | **$1,125** | ✅ Fill — 2.3% of account |

**Key change since last plan:** QQQ has run from ~$530 → $722 (+36%) and SMH from ~$260 → $604 (+132%). Both are now nearly as expensive as SPY to run CSPs on. They are off the table until the account reaches $100k+.

**Recommended portfolio — $50k account, $20k budget (July 2026 prices):**

| Position | ETF | Contracts | Margin each | Total Margin | 30-DTE Premium (5% OTM) | Monthly Income |
|---|---|---|---|---|---|---|
| 1 | IWM | 2 | $4,400 | $8,800 | $2.50–4.00 | $500–800 |
| 2 | XBI | 2 | $2,400 | $4,800 | $3.00–4.00 | $600–800 |
| 3 | KRE | 3 | $1,125 | $3,375 | $0.70–1.00 | $210–300 |
| **Total** | | **7 contracts** | | **$16,975** | | **$1,310–1,900/month** |

**Buffer:** $3,025 ($20k − $16,975) — covers margin expansion if market drops 5–8%

**Why IWM + XBI + KRE:**
- **IWM** (Russell 2000): broadest small-cap exposure, high options liquidity, margin is manageable
- **XBI** (Biotech): highest IV among liquid ETFs → best premium-to-margin ratio (10–13% monthly ROI on margin vs 6–8% for IWM)
- **KRE** (Regional Banks): cheap per-contract, sector diversification from IWM/XBI, fills slots efficiently; 3× KRE = only $3,375 margin

**Monthly income ÷ 21 trading days = $62–90/day passive baseline** — much better than the prior $30–45/day estimate (larger universe, more contracts).

**Account scaling path (prices as of July 8, 2026):**

| Account Size | CSP Budget | Portfolio | Monthly Income | Daily baseline |
|---|---|---|---|---|
| **$50k (now)** | **$20k** | **2× IWM + 2× XBI + 3× KRE** | **$1,310–1,900** | **$62–90** |
| $75k | $28k | 3× IWM + 3× XBI + 4× KRE | $1,900–2,700 | $90–130 |
| $100k | $35k | 3× IWM + 3× XBI + 4× KRE + 1× QQQ | $2,500–3,600 | $120–170 |
| $150k | $50k | 1× SPY + 1× QQQ + 4× IWM + 4× XBI + 4× KRE | $4,000–5,500 | $190–260 |

> SPY re-enters the portfolio at $100k (1 contract = 11% of account); QQQ re-enters at $100k as well.
> At $150k, the combination of all four ETFs makes $200/day achievable from CSPs alone on many days.

**LEAPs:** Paused until CSP strategy is consistently profitable.  
When resumed:
- Maximum $3,000 cost basis per LEAP
- Maximum 5% of account in one position
- Only enter in low-IV environments (IV rank < 30)
- Strong directional thesis required

**Why ETFs only for CSP:**  
Single-stock CSPs carry earnings/gap risk. An ETF like SPY or QQQ cannot go to zero overnight and provides natural diversification. The MRVL cascade could not happen on SPY.

### 3.2 Day Trader — New Configuration (Backtest-Validated)

A 5-year backtest across 111 tickers and 4 configs identified only one profitable setup:

| Parameter | Old (broken) | New (backtest optimal) |
|---|---|---|
| Profit target | 0.5% | **2.0%** |
| Hard stop | 7.0% | **3.0%** |
| R/R ratio | 14:1 against | **1.5:1 in favor** |
| Position size | $2,000 | **$5,000** |
| Max positions | 5 | **10** |
| Min composite score | None | **75** (top ~25% of signals) |
| Force close | 15:45 ET | 15:45 ET |

**Backtest results (Config D, 5 years):**
- Net P&L: +$50,025
- Average daily: +$32.70
- Goal hit rate (≥$200/day): 28.8%
- With score ≥75 filter: estimated 35–40% goal hit rate

**Signal filter:** Only accept breakout signals with composite score ≥ 75. This filters roughly 60% of signals but significantly improves win rate because it restricts to the strongest momentum setups.

### 3.3 Stock Trader — Tightened Rules

- **Max hold days:** 5 (was open-ended — positions were sitting red for weeks)
- **Hard stop:** 5% loss after 3+ days holding → flagged for exit
- **Sector rotation:** Enabled — evict the weakest loser when a higher-scoring signal arrives
- **Score requirement:** ≥ 70 to enter (was no minimum)
- **Max concentration:** No single stock > 5% of account value

---

## 4. Systems Built

### 4.1 Risk Monitor Agent

Runs every 5 minutes during market hours (9:30–4:00 ET). Enforces 5 unbreakable rules:

| Rule | Trigger | Action |
|---|---|---|
| 1. CSP stop | Loss ≥ 2× premium received | **Auto-close position** + CRITICAL Telegram |
| 2. LEAP size | Cost basis > $3,000 | WARNING Telegram |
| 3. VIX spike | VIX > 25 | **Auto-disable day trader** + WARNING |
| 4. Stock loser | Down > 5% after 3+ days | WARNING Telegram |
| 5. Concentration | Position > 5% of account | WARNING Telegram |

Endpoints: `GET /risk/status` · `POST /risk/run-now` · `POST /risk/config`  
UI: Risk Monitor tab under Live Trading

**This is the guardrail that would have prevented both the MRVL cascade and the QCOM blowup.** Rule 1 would have auto-closed MRVL at ~$1,000 loss instead of $3,146. Rule 5 would have flagged QCOM LEAP before entry.

### 4.2 Breaking News Monitor Agent

Polls 11 sources every 60 seconds. Sends Telegram alerts for CRITICAL/HIGH market-moving events.

**Sources:**
- 5× Google News topic feeds (Macro/Fed, Markets, Corporate, Global/Geo, Bonds/Rates) — aggregates AP, Reuters, Bloomberg, CNBC, FT, WSJ in real time
- AP Business + AP Economy (wire speed)
- CNBC Breaking, Reuters, MarketWatch, Benzinga

**Alert tiers:**
- 🔴 CRITICAL: Fed decisions, rate changes, bankruptcies, mergers, market halts, macro reports
- 🟠 HIGH: Earnings beats/misses, FDA decisions, layoffs, analyst upgrades/downgrades
- ✅ VERIFIED: Same story confirmed on 2+ independent sources

**Max age filter:** Articles older than 120 minutes are excluded. The feed shows only genuinely fresh news.

**Why this matters for trading:**  
Breaking news is why stops exist. A sudden Fed statement or geopolitical event can move positions 5–10% in seconds. Knowing about it in real time allows:
- Manual override before auto-stop fires
- Decision to add or exit ahead of the move
- Context for why the risk monitor may be alerting

---

## 5. The $200/Day Path

### 5.1 Math

**Day Trader contribution:**
- Best case scenario per backtest: $32.70 average daily P&L at $5k × 10 positions
- On goal-hit days (28–35% of days): $200+ from day trading alone
- Remaining days: smaller gains or small losses offset by other strategies

**CSP contribution (July 2026 prices — 7 contracts across IWM/XBI/KRE):**
- 2× IWM + 2× XBI + 3× KRE at $16,975 total margin
- Monthly income: $1,310–$1,900/month
- Daily equivalent: **$62–90/day passive baseline** (non-directional, theta decay)

**Combined target:**
| Source | Conservative daily | Good day |
|---|---|---|
| Day Trader | $0–$50 | $100–$300 |
| CSP (amortized daily) | $62–90 | $62–90 |
| Stock Trader | $0–$20 | $50–$100 |
| **Total** | **$62–$160** | **$212–$490** |

### 5.2 Realistic Expectation

$200/day every day is not achievable. The math:
- Backtest shows 28.8% of days hit $200+
- Roughly 1 in 3 trading days will hit the goal
- Monthly total: 3–5 good days × $200+ = $600–$1,000/month realistic
- Account needs to grow to $150k+ before $200/day average becomes consistent

### 5.3 What Accelerates It

1. **Score filter discipline:** Only take day trader signals with score ≥ 75. Skip everything else even when volume is low.
2. **Never roll a losing CSP.** Take the 2× stop loss and redeploy. Rolling compounds risk.
3. **LEAP re-entry only in low IV.** When LEAPs resume, enter at IV rank < 30 with $3k max cost.
4. **Let winners run.** Day trader force-close at 15:45 is correct — don't exit early chasing a partial win.
5. **Trust the news monitor.** If a CRITICAL alert fires mid-position, assess before the market reacts, not after.

---

## 6. The 5 Rules That Cannot Be Broken

These are encoded in the Risk Monitor and enforced automatically. They also apply to manual decisions:

1. **No CSP loss beyond 2× premium.** Close it. No rolling, no hoping. The premium received is your max acceptable loss budget.

2. **No single position > 5% of account.** This is what turned a bad LEAP trade into a $4,208 loss. At 5% max, the same trade would have been $2,500 loss — painful but survivable.

3. **Day trader off when VIX > 25.** High VIX means intraday ranges explode. The 3% hard stop gets hit on noise, not signal. Preserve capital and wait.

4. **Exit stock positions down > 5% after 3 days.** The stock is telling you the thesis is wrong. A 5% loss now is better than a 15% loss in two weeks.

5. **No LEAP entry cost > $3,000.** Options can go to zero. Cap the maximum possible single-instrument loss at $3,000.

---

## 7. Recovery Timeline

| Phase | Target | Metric |
|---|---|---|
| **Stabilize** (Week 1–2) | Stop the bleeding | Zero new positions violating the 5 rules |
| **Rebuild** (Month 1) | Break even | Day trader win rate improving, CSP running clean |
| **Grow** (Month 2–3) | $50–$100/day average | CSP income steady, day trader hitting 30%+ goal days |
| **Scale** (Month 4–6) | $200/day average | Account size allows larger position sizing |

---

## 8. Key Metrics to Watch Daily

| Metric | Where | Target |
|---|---|---|
| Risk Monitor violations | `/risk/status` | 0 active |
| CSP live P&L vs 2× stop | CSP-Leap tab | Never cross warn level |
| Day trader win rate (rolling 20) | Decisions log | > 50% |
| Day trader avg P&L per trade | Decisions log | > $0 |
| VIX | Risk Monitor tab | < 25 for day trading |
| Breaking news | News Feed tab | Review every morning before market open |
| Account net delta | Top navbar | Keep near 0 for neutral exposure |

---

## 9. What Not to Do Again

- **Do not roll a losing CSP more than once.** If it needs a second roll, the thesis is wrong. Take the loss.
- **Do not enter a LEAP larger than $3,000 cost basis** regardless of conviction.
- **Do not ignore stop_loss_mult.** Verify config values after every restart with `/autotrader/status`.
- **Do not override the Risk Monitor auto-close.** If Rule 1 fires, the position closes. Do not re-enter the same day.
- **Do not use day trader on high-VIX days.** The backtest only works in normal volatility regimes.
- **Do not hold stock positions red for more than 3 days without re-evaluating.** NKE, HD, LOW sat red for weeks.

---

## 10. Working Principles (How Analysis Is Done)

These rules govern how Claude approaches any trade calculation or budget analysis in this project:

### 10.1 Always use live prices for budget calculations

**Rule:** Before computing any CSP/LEAP margin, position sizing, or budget allocation, fetch the current market price of the underlying via web search. Never use a price from memory, a previous session, or an estimate.

**Why this matters:** ETF prices can drift significantly between sessions. In this project, QQQ ran from ~$530 → $722 (+36%) and SMH from ~$260 → $604 (+132%) in the span of months. Using stale prices leads to margin calculations that are off by thousands of dollars per contract — budget plans that appear feasible become impossible, or vice versa.

**How to apply:**
1. When asked for a CSP budget plan, search for each ETF's current price first.
2. Label every price with the date fetched: e.g., `$295 (July 8, 2026)`.
3. If a price cannot be confirmed, state that explicitly and flag the calculation as an estimate.
4. Re-verify prices whenever more than 1 week has passed since the last quote was fetched.

---

## 11. Parking Lot — Consider Next If Plan Doesn't Work

These are enhancements that are technically feasible with IBKR but deliberately deferred. Activate them only if the current turnaround plan (Sections 3–6) is not producing results after 30–60 days of clean operation.

---

### 11.1 Options Flow Enrichment (post-breakout signal gate)

**What it does:** After the breakout scanner fires on a ticker (equity pct_b + vol_ratio threshold met), pull that ticker's OTM near-dated options and check for unusual institutional flow before passing the signal to the day/stock trader.

**The filter:**
```
Vol/OI ≥ 2.0  AND  OTM strike  AND  expiry ≤ 10 trading days  AND  last ≥ ask (aggressive lift)
```

**Why it improves signal quality:**
- High equity vol_ratio alone = unusual share volume, direction unknown
- OTM calls ≤ 10 DTE at the ask = buyer expects the move to happen *soon*, not a hedge
- Combining both filters removes a large portion of false breakout signals (dealer hedging flow, institutional rolls, noise prints)

**Why it's parked:**
- We just started collecting `vol_ratio` and `composite_score` per trade (July 2026). Need 50–100 DAY_BREAKOUT trades to backtest whether the equity vol_ratio gate alone is sufficient.
- Building the options chain pull first would add complexity before we have evidence it's needed.
- If equity Vol/OI gate improves win rate to 65%+ and avg return to > $15/trade, the options flow layer may be unnecessary.

**What IBKR provides (confirmed feasible):**
- `reqSecDefOptParams` → full chain (strikes + expiries) — free call
- `reqMktData` snapshot → bid, ask, last, volume, open interest (prior day OI) — uses ~10–20 subscription slots for ~2 seconds then releases
- Aggressor side: `last >= ask * 0.995` approximates "buyer lifted the ask" — not a true multi-exchange sweep tag but good enough for retail use

**Implementation estimate:** ~150 lines in a new `options_flow.py`, called reactively from the breakout scanner signal path. Adds ~2–3 seconds of latency before signal reaches the trader endpoints.

**Activation criteria:**
- After 50–100 DAY_BREAKOUT trades logged with `vol_ratio` and `score`
- Backtest shows equity Vol/OI gate alone is not sufficient (win rate still < 60% or avg loss > avg win)

---

### 11.2 True Multi-Exchange Sweep Detection (OPRA full feed)

**What it does:** Tags each options print as a "sweep" — the same order aggressively lifted the ask across 3+ exchanges simultaneously, indicating institutional urgency.

**Why equity-only aggressor side isn't enough:** `last >= ask` on a single exchange snapshot tells you the *last* print was aggressive, but a single large print at the ask could be a market maker crossing the spread or an automated roll. A true sweep is the same order hitting CBOE, AMEX, PHLX, and ISE within milliseconds — that's the "smart money buying before a move" signal that Unusual Whales is actually selling.

**What's needed:** OPRA (Options Price Reporting Authority) full consolidated tape. Not available via IBKR TWS API. Requires:
- Unusual Whales API (~$50/month) — provides pre-tagged sweep alerts
- Or direct OPRA feed via a market data vendor (Polygon.io, Cboe DataShop) — more expensive, requires parsing

**Why it's parked:**
- Requires paid external data subscription
- The `last >= ask` approximation in 11.1 captures the majority of the signal (aggressive lift still means directional bet)
- Only add this if 11.1 is implemented and the false-positive rate from non-sweep prints is measurably hurting results

**Activation criteria:** Options flow module (11.1) live and running for 30+ days; evidence that non-sweep aggressive lifts are generating bad signals at a rate that justifies the subscription cost.

---

### 11.3 Backend Auto-Restart Watchdog (Task Scheduler)

**What it is:** The breakout scanner already has a PowerShell watchdog (`run_scanner.ps1`) registered in Windows Task Scheduler that auto-restarts it on crash. The backend (`main.py`) has no equivalent — if it crashes it stays down until manually restarted.

**What's needed:** Register `run_backend.ps1` in Task Scheduler the same way `run_scanner.ps1` is registered (trigger: at logon, run hidden, restart on failure). The script already exists and includes a syntax-check pre-flight — it just isn't scheduled.

**Why it's parked:** Low urgency during paper trading. Backend crashes are rare and recoverable manually. Becomes more important before going live on a real account where a missed restart = missed trades = real money.

**Activation criteria:** Before switching from paper to live trading.

---

### 11.4 Kelly-Based Dynamic Position Sizing (Stock Trader + Day Trader)

**What it does:** Replaces the fixed `position_size` config with a Kelly-derived position size that automatically scales up when the strategy is working and scales down when it isn't.

**Formula:**
```
b            = avg_win_$ / avg_loss_$          (win/loss dollar ratio)
kelly_frac   = (win_rate × (b + 1) − 1) / b   (raw Kelly)
half_kelly   = kelly_frac × 0.5               (half-Kelly — standard risk management)
position_$   = half_kelly × account_capital
```

**Example at day trader backtest config (2% target / 3% stop, 60% win rate):**
- b = 2/3 = 0.67, Kelly = (0.60 × 1.67 − 1) / 0.67 ≈ 10% → half-Kelly = 5%
- Position size = 5% × $50,000 = **$2,500** (scales up to $5,000 at 65% win rate, shrinks below minimum at <55%)

**Why it's better than a fixed size:**
- Good stretch (win rate > 60%) → larger positions, capturing more of the momentum
- Bad stretch (win rate drops below break-even) → Kelly goes negative, system floors to minimum bet or pauses sizing
- No manual intervention needed — the journal win rate drives position size automatically

**Why it's parked:**
- The journal needs strategy-specific win rates: Kelly for the day trader must be computed from `DAY_BREAKOUT` trades only, not the mixed CSP/LEAP/stock history that currently pollutes `assumed_win_rate`
- We just started collecting `score` and `vol_ratio` per trade (July 2026) — need 50–100 clean `DAY_BREAKOUT` trades before the win rate signal is statistically meaningful
- Fixed sizing is safer while the strategy is still being validated

**Note:** Kelly does NOT apply to CSPs. CSP contracts have a fixed minimum margin ($1,125–$4,400) so position sizing is discrete (1 contract or 0), not continuous. The Kelly display was removed from the CSP-LEAP tab for this reason.

**Activation criteria:** 50+ `DAY_BREAKOUT` trades and 50+ `STOCK_BREAKOUT` trades in the journal with clean win/loss data; compute strategy-specific win rates before enabling.
