"""How does AELong exit? For each position: path during the hold (max gain, max drawdown, give-back
from peak) and the state at the exit close (vs SMAs, RSI, 52w high), split winners / losers."""
import json
import statistics as st
import sys
from pathlib import Path

from analyze_entries import PX, feats, idx_before

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
POS = json.load(open(HERE / "data" / "aelong_positions_feats.json"))


def q(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(xs)
    return [round(xs[int(len(xs) * p)], 3) for p in ps]


rows = []
for p in POS:
    s = p["sym"]
    i0, i1 = idx_before(s, p["o"][:10]) + 1, idx_before(s, p["c"][:10]) + 1  # entry day, exit day bars
    bars = PX[s][i0:i1 + 1]
    if len(bars) < 2:
        continue
    entry = p["entry"]
    closes = [b["c"] / bars[0]["c"] * entry for b in bars]  # rescale adjusted closes to entry price
    peak = max(closes)
    fx = feats(s, i1 - 1)  # state at the close before the exit day
    rows.append({
        "sym": s, "ret": p["ret"] / 100, "hold": p["hold"], "max_gain": peak / entry - 1,
        "max_dd": min(closes) / entry - 1, "giveback": closes[-1] / peak - 1,
        "exit_vs_sma50": fx["vs_sma50"] if fx else None, "exit_vs_sma200": fx["vs_sma200"] if fx else None,
        "exit_rsi": fx["rsi14"] if fx else None, "exit_from_hi": fx["from_52w_hi"] if fx else None,
        "exit_mom1m": fx["mom1m"] if fx else None,
    })

win = [r for r in rows if r["ret"] > 0]
lose = [r for r in rows if r["ret"] <= 0]
print(f"positions {len(rows)}: winners {len(win)}, losers {len(lose)}\n")
print("quantiles 10/25/50/75/90")
for k in ["ret", "hold", "max_gain", "max_dd", "giveback", "exit_vs_sma50", "exit_vs_sma200", "exit_rsi",
          "exit_from_hi", "exit_mom1m"]:
    for lab, xs in (("WIN ", win), ("LOSE", lose)):
        v = [r[k] for r in xs if r[k] is not None]
        print(f"  {k:15} {lab} {q(v)}")
# stop-loss signature: losers' exit return clustered?
lr = sorted(r["ret"] for r in lose)
print("\nloser exit returns (sorted):", [round(x * 100, 1) for x in lr])
# target signature: winners' exit vs max gain
print("\nwinners exiting within 3% of their peak:", sum(1 for r in win if r["giveback"] > -0.03), "of", len(win))
print("winners exiting at RSI>65:", sum(1 for r in win if r["exit_rsi"] and r["exit_rsi"] > 65), "of", len(win))
print("winners exiting within 3% of 52w high:", sum(1 for r in win if r["exit_from_hi"] is not None and r["exit_from_hi"] > -0.03), "of", len(win))
print("losers exiting below SMA200:", sum(1 for r in lose if r["exit_vs_sma200"] is not None and r["exit_vs_sma200"] < 0), "of", len(lose))
print("losers whose max drawdown exceeded -10%:", sum(1 for r in lose if r["max_dd"] < -0.10), "of", len(lose))
