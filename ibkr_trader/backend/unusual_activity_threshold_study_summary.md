# Unusual Activity Threshold Study — 2026-08-30

CEO request: derive empirically-grounded "unusual" volume thresholds from real
historical dark pool + options tape + equity tape data, and compare against
the flat/ad-hoc thresholds currently hardcoded in the live system.

## 1. Equity tape (`tape_prints`, local DB, 1,387,559 real rows, 50 tickers, 2026-06-26 to 2026-08-28)

Current live default: `ws_tape`'s `is_block` flag uses a flat `size >= 5000` shares.

Real pooled percentiles:
| Percentile | Shares | Dollar size |
|---|---|---|
| p50 | 20 | $6,917 |
| p90 | 198 | $59,880 |
| **p95** | **5,491** | **$1,117,154** |
| p99 | 14,033 | $5,530,460 |
| p99.9 | 69,799 | $24,127,322 |

**Finding**: the current 5,000-share default sits at only the **p93.9** percentile — 6.15% of ALL real prints already clear it. Not a rare cut.

Per-ticker p95 size (comparable mega-caps, large samples 8.6K-742K rows each):
MSFT 120 sh · ORCL 232 sh · AMZN 200 sh · META 100 sh · SPY 6,779 sh · MRVL 5,300 sh · NVDA 21,596 sh · AAPL 29,068 sh
→ ~300x spread on a flat share-count basis. A flat threshold is the wrong unit; dollar or ADV-relative sizing (like the dark-pool rule below) would generalize far better.

**Recommendation**: replace the flat share count with an ADV-relative or dollar-percentile threshold. Pooled p99 (~$5.5M) / p99.5 (~$8.6M) are reasonable starting points if a single flat dollar figure is wanted; per-ticker relative sizing is more defensible given the cross-ticker spread.

## 2. Options tape (`options_unusual_activity`, local DB, 9,788 real rows, 28 tickers, 2026-08-19 to 2026-08-28)

Current live threshold: `OPTIONS_UOA_RATIO_THRESHOLD = 2.0` (volume >= 2x open interest).

Real pooled percentiles (vol/OI ratio):
| Percentile | Ratio |
|---|---|
| p50 | 0.76x |
| p75 | 3.01x |
| p90 | 10.39x |
| **p95** | **20.66x** |
| p99 | 132.75x |
| p99.9 | 578.13x |

**Finding**: the current 2.0x threshold sits at only the **p65** percentile — **30.2% of ALL scanned rows** are currently flagged `is_unusual`. This is far too loose to mean "rare" in any real sense; it's barely above the median.

**Recommendation**: raise the bar substantially. Since vol/OI ratio is already scale-normalized (no cross-ticker size bias the way raw volume/premium has), a flat ratio threshold is the right kind of metric — it just needs to move. Candidates: ~10-20x (p90-95, "worth flagging") or 100x+ (p99, "genuinely extreme").

## 3. Dark pool (fresh Unusual Whales pull, 15-ticker representative sample spanning mega/mid/small cap, week of 2026-08-24 to 2026-08-28, 27,928 real prints)

Current live rule (`darkpool_activity_monitor.py`): `max($1,000,000 flat floor, 0.5% of ticker's own 20-day ADV)`.

Real pooled premium percentiles: p50 $699K · p90 $2.45M · **p95 $4.02M** · p99 $46.3M · p99.9 $265.7M

**Flat $1M floor alone**: sits at only **p63.7** of this pooled sample (36.3% already clear it) — same "too loose" problem as the other two, when used alone.

**But the relative 0.5%-of-ADV half of the rule is genuinely well-calibrated** — checked per-ticker against each name's own real distribution:

| Ticker | 0.5% of ADV | Real percentile on that ticker's own data |
|---|---|---|
| ROKU | $1.94M | p94.7 |
| XOM | $11.2M | p98.2 |
| C | $5.7M | p98.2 |
| PYPL | $3.0M | p98.2 |
| UAL | $1.7M | p98.5 |
| RIVN | $1.9M | p98.8 |
| LULU | $1.6M | p99.0 |
| SPY | $158.2M | p99.2 |
| NVDA | $142.6M | p99.1 |
| AAPL | $75.1M | p99.2 |
| GLD | $22.8M | p99.3 |
| CRM | $17.7M | p99.5 |
| MRVL | $28.5M | p99.5 |
| META | $46.9M | p99.7 |
| TSLA | $59.2M | p99.8 |

Across small caps (ROKU, RIVN, UAL, LULU) through mega-caps (SPY, NVDA, AAPL, META), the relative rule consistently lands in a **tight p94.7-p99.8 band** — this is the one already-live threshold that holds up empirically. No change recommended here; the flat-$1M-floor half is the redundant/weak part (the relative half already dominates it for every real name checked), not the relative sizing itself.

## Summary verdict

| Data source | Current rule | Real percentile it hits | Verdict |
|---|---|---|---|
| Equity tape | flat 5,000 shares | p93.9 | Too loose, wrong unit (flat share count) |
| Options tape | flat 2.0x vol/OI | p65 | Far too loose (30% of everything flagged) |
| Dark pool (relative half) | 0.5% of 20-day ADV | p94.7-p99.8 (all tickers checked) | **Well-calibrated** — first real validation of this design |

Caveats carried from every dark-pool tool in this codebase: none of this implies predictive power — "unusual" here means statistically rare in the real historical distribution, not a validated trading signal. The equity-tape and options-tape samples are also smaller/narrower in ticker coverage (50 and 28 tickers respectively, vs the 112-name universe) than ideal — directionally solid given the sample sizes, but worth widening before treating the exact percentile cutoffs as final.
