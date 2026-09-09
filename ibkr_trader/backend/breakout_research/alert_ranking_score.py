"""
Builds a continuous quality score per alert (not just the binary pass/fail
filter from Iteration 4) using the real, empirically-grounded feature
effects already found in faithful_extended_rows.csv (1420 gate-accurate
alerts, 2.5y). Transparent, additive scoring -- NOT a fitted regression/ML
model, deliberately: with ~1420 samples and several correlated features, a
fitted model risks overfitting; an additive score built directly from
already-validated bucket effects is easier to sanity-check and defend.

Score components (points = real avg ret_5d observed for that bucket in this
account's own data, so the score itself is interpretable as "expected %
return contribution", not an arbitrary point system):
  - signal_type: BREAKOUT vs PRE-BREAKOUT (the single largest real effect)
  - RSI: penalize >=80 (validated, monotonic, all 4 horizons)
  - pct_b dead zone: PRE-BREAKOUT-specific penalty for the 75-85 band
  - dist_52w_high: reward being near the 52-week high
  - ADX: signal-type-dependent (helps BREAKOUT, no clean effect on PRE-BO
    per Iteration 5) -- only applied to BREAKOUT alerts

Validates the ranking properly: buckets ALL alerts into quintiles by score
and checks whether real ret_5d is actually monotonic across quintiles --
the real test of a ranking (not just "top vs bottom looks different").
"""
import pandas as pd

df = pd.read_csv("faithful_extended_rows.csv")


def score_row(r) -> float:
    s = 0.0
    # Signal type -- the single biggest real effect found (Iteration 3)
    s += 1.75 if r["signal"] == "BREAKOUT" else 0.51  # real avg ret_5d per type

    # RSI extreme penalty (validated all 4 horizons, Iteration 3)
    if pd.notna(r["rsi"]):
        if r["rsi"] >= 80:
            s -= 1.7  # matches the real -1.71% avg ret_5d observed for this bucket
        elif r["rsi"] >= 70:
            s += 0.3  # mildly positive/neutral in the real data, not penalized

    # PRE-BREAKOUT dead zone (Iteration 4 headline finding) -- only meaningful for PRE-BO
    if r["signal"] == "PRE-BREAKOUT" and pd.notna(r["pct_b"]):
        if 75 <= r["pct_b"] < 85:
            s -= 0.9  # real gap between dead-zone and flanking bands
        else:
            s += 0.3

    # Distance from 52-week high -- real, modest, consistent tilt
    if pd.notna(r["dist_52w_high"]):
        if r["dist_52w_high"] > -5:
            s += 0.4
        elif r["dist_52w_high"] <= -15:
            s -= 0.2

    # ADX -- only found to matter for BREAKOUT specifically (Iteration 5), smaller sample
    if r["signal"] == "BREAKOUT" and pd.notna(r["adx"]):
        s += 0.3 if r["adx"] < 30 else -0.4

    return s


df["score"] = df.apply(score_row, axis=1)
df["quintile"] = pd.qcut(df["score"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"], duplicates="drop")

print("=== Real ret_5d by score quintile (the actual validation) ===")
for q in df["quintile"].cat.categories:
    sub = df[df["quintile"] == q]
    v = sub["ret_5d"].dropna()
    bo_pct = (sub["signal"] == "BREAKOUT").mean() * 100
    print(f"  {q:14} n={len(v):5} win_rate={(v>0).mean()*100:5.1f}% avg_ret_5d={v.mean():+.3f}% "
          f"median={v.median():+.3f}%  (%BREAKOUT: {bo_pct:.0f}%)")

print()
print("=== Same check on ret_3d ===")
for q in df["quintile"].cat.categories:
    sub = df[df["quintile"] == q]
    v = sub["ret_3d"].dropna()
    print(f"  {q:14} n={len(v):5} win_rate={(v>0).mean()*100:5.1f}% avg_ret_3d={v.mean():+.3f}%")

df.to_csv("scored_alerts.csv", index=False)
print("\nSaved scored_alerts.csv")
