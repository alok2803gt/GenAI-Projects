"""THE holdout test for PREREG_daytrader_revival.md (sha256 3071a656...).

ONE test. The rule is frozen in the pre-registration; nothing here is tuned.
Runs on the 407 S&P 500 tickers never examined during the 2026-09-29 work.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gate_threshold_study import add_scores, build, clustered  # noqa: E402

HOLDOUT = HERE / "holdout_universe_5y.pkl"
FEE_PP = 0.70 / 148.0 * 100      # Tiered round trip at the live position

print("pre-registration hash on disk:")
print(" ", subprocess.run(["shasum", "-a", "256", str(HERE / "PREREG_daytrader_revival.md")],
                          capture_output=True, text=True).stdout.strip())
print("  (must equal 3071a656987d817f583a88aae7244b33e06f40fc32bf551b29abf289c7a572cc)\n")

r = add_scores(build(HOLDOUT))
print(f"HOLDOUT: {r.ticker.nunique()} tickers x {r.date.nunique()} sessions = {len(r):,} ticker-days")
print(f"         {r.date.min().date()} .. {r.date.max().date()}\n")

# ---- the frozen rule, verbatim from the pre-registration ----
rule = r[(r.composite_score >= 75)
         & (r.atr_pct >= 2.5)
         & (r.gap_pct >= -3.0) & (r.gap_pct <= 0.0)
         & (r.atr_mult >= 1.4)]

m, t, ndays, nrows = clustered(rule, "excess")
raw, _, _, _ = clustered(rule, "target")
hit = (rule.target >= 0.5).mean() * 100

print("RESULT (frozen rule: score>=75, ATR%>=2.5, gap in [-3%,0%], atr_mult>=1.4, no sigma)")
print(f"  trades            {nrows:,} on {ndays} sessions ({ndays/r.date.nunique()*100:.0f}% of days)")
print(f"  raw return        {raw:+.3f}pp")
print(f"  market-adjusted   {m:+.3f}pp")
print(f"  date-clustered t  {t:+.2f}")
print(f"  hit >= +0.5%      {hit:.1f}%\n")

c1 = (m > 0) and (t > 2.0)
c2 = m > FEE_PP
print("PASS CRITERIA (fixed before the test)")
print(f"  1. edge exists    : excess>0 and t>2.0        -> {'PASS' if c1 else 'FAIL'}"
      f"  ({m:+.3f}pp, t={t:+.2f})")
print(f"  2. beats the fee  : excess>{FEE_PP:.2f}pp            -> {'PASS' if c2 else 'FAIL'}"
      f"  ({m:+.3f}pp vs {FEE_PP:.2f}pp)")
print()
if c1 and c2:
    print("  VALIDATED -- implement the rule and size at >= $148.")
elif c1:
    be = 0.70 / (m / 100)
    print(f"  REAL BUT SUB-FEE -- break-even position ${be:,.0f}. Day Trader stays")
    print("  DISABLED until net liq supports it. Do NOT trade at $148.")
else:
    print("  REJECTED -- the revival hypothesis fails out of sample.")
    print("  Day Trader stays disabled. The holdout is now spent.")
