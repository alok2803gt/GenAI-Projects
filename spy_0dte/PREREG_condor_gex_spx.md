# Pre-registration — (A) dealer-gamma condition and (B) SPX implementation (v1)

This plan was written and hashed before any QQQ or IWM data was viewed, and before any SPX option data was downloaded. The QQQ and IWM downloads were running, unopened, when this was frozen.

## A. Does prior-day dealer gamma improve the condor?

- **Hypothesis (a priori):** when dealers are net long gamma, their hedging dampens intraday moves, so short-premium condors do better.
- **Condition:** net gamma = `call_gamma + put_gamma` from Unusual Whales `/api/stock/{X}/greek-exposure`, for the **prior trading day** (known before the open). A day is **HIGH** if prior-day net gamma is above the median of the trailing 252 trading days, using only data before that day. Otherwise it is **LOW**.
- **Trades:** the same condor as `PREREG_condor_replication.md` (frozen spec, per-asset wing).
- **Assets:** QQQ and IWM, the fresh assets. SPY is reported as exploratory only, because its data has already been viewed.
- **Test:** Welch t on session-mean condor P&L, HIGH minus LOW.
  - **Pass** per asset: the difference is > 0 **and t ≥ 2.24** (Bonferroni over 2 assets).
  - **Confirmed:** at least one asset passes, and neither has t ≤ −2.24.
- **Missing GEX history:** if UW has no GEX series for an asset, that asset is dropped from A and the Bonferroni bar uses the remaining count.

## B. SPX (SPXW 0DTE, cash-settled) as the vehicle

- **Purpose:** an implementation check, not independent evidence. SPX and SPY track the same index. The question is whether the condor survives SPX's own prices and spreads, since SPX removes assignment risk.
- **Specification:** frozen condor spec on SPXW 0DTE options (Polygon 1-min bars).
  - **Wing:** `round($3 × SPX/SPY ratio)` to the nearest listed $5 strike, which gives about $30.
  - **SPX spot per minute:** the median of `C − P + K` across the 3 strikes nearest ATM that printed in that minute (put-call parity; European options, r ≈ 0 intraday).
  - **Settlement:** the official SPX daily close (^GSPC).
- **Period:** 2024-02-01 to 2026-09-21.
- **Pass:** mean > 0 and t ≥ 2.00.
- **Reported, not part of the verdict:** λ 0 / 0.5 / 1.0, and credit per $ at risk vs SPY.

## Ledger

These tests are recorded in the phase-3 test ledger along with everything else run from now on.
