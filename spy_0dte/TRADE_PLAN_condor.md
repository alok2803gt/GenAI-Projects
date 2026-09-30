# Trade plan — SPY 0DTE iron condor (V2, risk-reduced)

> **V2 supersedes V1 below (2026-09-22).** Changes: short strikes at **±0.5u** (was 1.0u), wings **$5** (was $3), entries every **30 min** (10/day, was 19), and a **stop at 0.5× credit** — close the whole condor the first minute its loss reaches half the credit received.
>
> **Why:** the worst day falls from −$5,155 to **−$1,255** (−76%) on SPY, and −$4,913 to −$1,555 on QQQ, while return per dollar of worst-day risk rises from 7.8 to **20.2**. Confirmed on QQQ by a pre-registered test ([PREREG_condor_v2_risk.md](PREREG_condor_v2_risk.md)). The stop fires on 42% of condors.
>
> **Per unit:** about $101/day on SPY (about $48 under worst-case after-hours exercise), margin roughly $5,000, worst day about −$1,255.
>
> **On SPX** (cash-settled, no assignment): keep V1 instead. Wider spreads make the stop too costly there (t 1.17 vs 2.20 for V1).
>
> Everything else below — entry times, expected-move sizing, order handling, the 15:58 near-the-money close, phases — is unchanged.

# V1 (original, superseded)

**Status:** passed the pre-registered holdout test (+$7.04 per condor per day, t 2.15). Next step is **paper trading**.

Rules marked 🧪 were **not** part of the tested strategy. They are operational safeguards, and what they cost is tracked separately.

## 1. Setup (tested rules — do not change)

| Item | Rule |
|---|---|
| Instrument | SPY options expiring **today** (0DTE) |
| Entry times | 10:01, 10:16, 10:31 … 14:31. That is 19 entries per full day, one new condor each time. Half-days: entries until 5 min before the 15:45-equivalent |
| Expected move `u` | `u = SPY × ATM IV × √(minutes to 16:00 ÷ 98,280)`. ATM IV comes from the mids of the ATM call and put. Shortcut: `u ≈ ATM straddle mid ÷ 0.8` |
| Short put | Listed strike nearest **SPY − 1.0·u** |
| Long put (wing) | Short put − **$3** |
| Short call | Listed strike nearest **SPY + 1.0·u** |
| Long call (wing) | Short call + **$3** |
| Skip if | Any leg has no market, or the net credit after costs is ≤ 0 |
| Exit | **None. Hold to 16:00 expiry.** No stop loss and no profit target (none was tested) |
| Size | 1 condor per entry time (the unit) |

**Example:** at 10:01 with SPY at 660 and ATM IV at 13%, u = 660 × 0.13 × √(359/98,280) ≈ $5.2.
- Sell the 655 put, buy the 652 put.
- Sell the 665 call, buy the 668 call.
- Typical credit is about $0.40, so $40 per condor.

## 2. Order handling

- Send **one 4-leg combo order** (IBKR BAG), limit at the **mid** of the combo.
- If it isn't filled within 60 s, improve by $0.01. Maximum: **$0.02 worse than mid**. If still unfilled, **skip** and log it.
- The backtest assumed λ = 0.25, about ¼ of the modelled width worse than mid. The edge disappears at about λ 1.9. Paper trading measures where real fills land.
- Log every order: signal time, combo mid, fill price, time to fill, and skipped or filled.

## 3. Expiry and assignment (SPY is physically settled)

- 🧪 **At 15:58, close any short leg within $0.25 of SPY**, whether in or out of the money. This avoids after-hours assignment surprises: SPY options can be exercised until 17:30 on after-hours moves.
- 🧪 Legs clearly in the money at 16:00 are auto-exercised into shares. The combo's long wing caps the loss, but the account holds stock overnight. Close it at the next open, and track that cost separately.
- **Alternative:** SPX/SPXW options are cash-settled and European, so there is no assignment. This was not tested; it would need its own check.

## 4. Risk and capital (informational)

| Per unit (19 condors per day) | Value |
|---|---|
| Margin at peak (all open by 14:31) | about $5,200 |
| Worst possible day (all 19 at max loss) | about −$5,000 |
| Worst day in holdout / development | −$4,861 / −$4,657 |
| Max drawdown in holdout / development | −$9,959 / −$14,088 |
| Capital (margin + 2 × max drawdown) | about $25,000–33,000 |
| Expected profit (holdout pace) | about $119 per day, about $30k per year (not guaranteed) |

- 🧪 **Kill switch:** stop and review if the live drawdown reaches **−$15,000 per unit** (about 1.5× the worst historical drawdown).
- 🧪 **Scaling:** increase size only after the paper and pilot phases pass.

## 5. Phases

| Phase | Size | Duration | Pass to continue |
|---|---|---|---|
| **1. Paper** (**IBKR TWS paper**, port 7497 — Alpaca is ruled out, see below) | 1 unit | ≥ 40 sessions | Average fill ≤ **$0.02 worse than mid** per combo; ≥ 90% of entries filled; no operational failures |
| **2. Live pilot** | 1 condor at 5 fixed times (10:01, 11:01, 12:01, 13:01, 14:01) | ≥ 40 sessions | Fills match paper; assignment handling works |
| **3. Full** | 1 unit, then scale | ongoing | Kill switch and monthly review |

**Judge the paper phase on fills, not P&L.** Over 40 days the P&L swings about ±$5,800 from luck alone, which is bigger than the expected ≈$4,700. Execution quality is the one thing 40 days *can* measure.

## Platform constraint — Alpaca cannot run this strategy (verified live 2026-09-22)

1. **Opening near expiry is rejected:** `contract "SPY260922P00769000" expires soon, unable to open new positions`.
2. **Alpaca force-closes every 0DTE option position at 15:45 ET** (this account's own incident, 2026-09-02).

The edge is holding to the 16:00 expiry. The same condor closed at 15:45 backtests at **−$0.38/day** against **+$8.37/day** held to expiry, so Alpaca would trade the version already shown to have no edge. `--broker alpaca_paper` now refuses to start without an explicit plumbing-test flag.

**Paper trading therefore runs on IBKR TWS paper (port 7497).** Quotes stay on IBKR either way; Alpaca's free indicative feed is about 5× too wide to price or stop against.

## 6. What gets tracked daily

- Fills vs mid (the real λ) and the skip rate
- P&L against the backtest expectation of $7 per condor
- Worst day and drawdown against the limits
- Cost of the 🧪 safeguards: 15:58 closes and overnight share positions
- Commission per condor ($2.60 at $0.65 per contract; about $1.00 if the per-contract rate drops to $0.25)
