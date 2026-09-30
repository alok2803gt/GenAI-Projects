# Pre-registration — condor V2 (risk-reduced) confirmation on QQQ and SPX (v1)

Written and hashed **before any QQQ or SPX risk-variant data was computed**. Only SPY was searched.

## The variant (V2)

| Item | Rule |
|---|---|
| Structure | Iron condor: short strikes nearest **± 0.5u**, wings **$5** (scaled per asset: `max($1, round($5 × X_open / SPY_open))`; SPX rounds to $5 strikes) |
| Entries | Every **30 minutes**, 10:01 to 14:31 (10 per day) |
| **Stop** | Close the whole condor the first minute its mark-to-market loss reaches **0.5 × the credit received**; exit at that minute's prices, λ 0.25, with commission on all 8 legs |
| Otherwise | Hold to expiry, settled at intrinsic against the official close |
| Costs | λ 0.25, $0.65 per contract |

**Why:** searched on SPY across wing width, short-strike distance, entry cadence, stop level, a daily circuit breaker, and a trend filter. V2 gave the best return per dollar of worst-day loss (20.3 vs 7.8 for the baseline), with a worst day of −$1,255 vs −$5,155 and a higher t (4.98 vs 4.28). The circuit breaker and trend filter added nothing and are excluded.

## Confirmation test (QQQ and SPX)

Both must be measured on data never used for this variant.

1. **Edge survives:** daily P&L mean > 0 and **t ≥ 2.24** (Bonferroni over 2 assets).
2. **Risk improves:** V2's return per dollar of worst-day loss is **greater than the baseline's** (midday 1.0u/$3 condor, 15-min entries, held to expiry) on that same asset.

**Confirmed** if at least one asset passes both, and neither has a significantly negative mean (t ≤ −2.24).

## Reported, not part of the verdict

Worst-case after-hours exercise settlement for QQQ; drawdown; stop frequency; P&L at λ 0.5.
