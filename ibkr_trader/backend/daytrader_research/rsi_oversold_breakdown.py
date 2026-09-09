"""
Deeper breakdown of the RSI(1min,14)<30 oversold-bounce entry (see
rsi_1min_confirm_backtest.py) -- CEO follow-up 2026-09-05: does it work
better on specific sectors / high-liquidity names, which tickers actually
drove it, and on which specific days did it fail. Same real 820-candidate
1-minute-bar dataset, same simulate_oversold_bounce mechanics (RSI<30 +
volume>=running median entry, 0.3% trailing stop exit) -- this file adds
per-ticker/per-sector/per-liquidity slicing on top, it does not re-derive
the simulation.

Real starting point already established: RSI<30 on the FULL population
(n=240) is NOT a robust edge -- removing just the top 5 of 240 winners
flips the total from +3.53% to -0.34% (see rsi_1min_confirm_results.json).
This script checks whether some real SUBSET (sector, liquidity tier)
survives that same robustness bar, or whether the fragility is universal.

Sector map reused directly from daytrader_scanner.load_universe() (the
live scanner's own cached S&P 500 sector data) -- not re-derived.
Liquidity proxy: real avg daily dollar volume from the minute bars
themselves (sum(volume*close) per day, averaged across each ticker's real
appearances in this sample) -- self-contained, no external data source.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "..")
from rsi_1min_confirm_backtest import simulate_oversold_bounce
from daytrader_scanner import load_universe

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
RSI_THRESHOLD = 30.0


def robustness_check(rows: list[dict], label: str) -> dict:
    """Same 'remove top N winners' check already applied to the full
    population -- a subset only counts as a real finding if it survives
    this, not just because its raw total looks positive."""
    rets = sorted([r["ret_pct"] for r in rows], reverse=True)
    n = len(rets)
    total = sum(rets)
    out = {"label": label, "n": n, "total_ret_pct": round(total, 2),
           "avg_ret_pct": round(total / n, 4) if n else None,
           "win_rate_pct": round(len([r for r in rets if r > 0]) / n * 100, 1) if n else None}
    for k in (1, 2, 3, 5, 10):
        if n > k:
            remaining = rets[k:]
            out[f"total_after_removing_top_{k}"] = round(sum(remaining), 2)
    return out


def main():
    with open(BARS_PATH) as f:
        bar_data = json.load(f)
    tickers_universe, sector_map = load_universe()
    print(f"Loaded sector map for {len(sector_map)} S&P 500 tickers (cached, reused from the live scanner)")

    rows = []
    for key, bars in bar_data.items():
        if not bars or len(bars) < 10:
            continue
        ticker, dt = key.split(":")
        r = simulate_oversold_bounce(bars, RSI_THRESHOLD)
        if r is None:
            continue
        daily_dollar_vol = sum(b["volume"] * b["close"] for b in bars)
        rows.append({
            "ticker": ticker, "date": dt, "ret_pct": round(r["ret_pct"], 4),
            "win": r["ret_pct"] > 0, "sector": sector_map.get(ticker, "?"),
            "daily_dollar_vol": daily_dollar_vol,
        })

    print(f"\nTotal RSI<{RSI_THRESHOLD:.0f} oversold-bounce trades: {len(rows)}\n")

    # ── Per-ticker breakdown ────────────────────────────────────────────────
    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)

    ticker_agg = []
    for tk, trs in by_ticker.items():
        total = sum(t["ret_pct"] for t in trs)
        wins = [t for t in trs if t["win"]]
        ticker_agg.append({
            "ticker": tk, "n": len(trs), "total_ret_pct": round(total, 3),
            "win_rate_pct": round(len(wins) / len(trs) * 100, 1),
            "avg_ret_pct": round(total / len(trs), 4),
            "sector": trs[0]["sector"],
        })
    ticker_agg.sort(key=lambda x: -x["total_ret_pct"])

    print("=== TOP 5 TICKERS BY TOTAL CONTRIBUTION ===")
    top5 = ticker_agg[:5]
    for t in top5:
        print(f"  {t['ticker']:6s} ({t['sector']:25s}) n={t['n']:2d}  win={t['win_rate_pct']:5.1f}%  "
              f"avg={t['avg_ret_pct']:+.4f}%  total={t['total_ret_pct']:+.3f}%")

    print("\n=== BOTTOM 5 TICKERS BY TOTAL CONTRIBUTION (worst) ===")
    for t in ticker_agg[-5:]:
        print(f"  {t['ticker']:6s} ({t['sector']:25s}) n={t['n']:2d}  win={t['win_rate_pct']:5.1f}%  "
              f"avg={t['avg_ret_pct']:+.4f}%  total={t['total_ret_pct']:+.3f}%")

    print("\n=== FULL DAY-BY-DAY DETAIL FOR THE TOP 5 TICKERS (every real occurrence, wins AND losses) ===")
    for t in top5:
        tk = t["ticker"]
        print(f"\n  {tk} ({t['sector']}) -- {t['n']} real occurrences in the sample:")
        for r in sorted(by_ticker[tk], key=lambda x: x["date"]):
            tag = "WIN " if r["win"] else "LOSS"
            print(f"    {r['date']}  {tag}  {r['ret_pct']:+.3f}%")

    # ── Sector breakdown ────────────────────────────────────────────────────
    by_sector = defaultdict(list)
    for r in rows:
        by_sector[r["sector"]].append(r)
    sector_agg = []
    for sec, trs in by_sector.items():
        total = sum(t["ret_pct"] for t in trs)
        wins = [t for t in trs if t["win"]]
        sector_agg.append({
            "sector": sec, "n": len(trs), "total_ret_pct": round(total, 3),
            "win_rate_pct": round(len(wins) / len(trs) * 100, 1),
            "avg_ret_pct": round(total / len(trs), 4),
        })
    sector_agg.sort(key=lambda x: -x["total_ret_pct"])
    print("\n=== SECTOR BREAKDOWN (sorted by total contribution) ===")
    for s in sector_agg:
        print(f"  {s['sector']:28s} n={s['n']:3d}  win={s['win_rate_pct']:5.1f}%  "
              f"avg={s['avg_ret_pct']:+.4f}%  total={s['total_ret_pct']:+.3f}%")

    # ── Liquidity tiers ─────────────────────────────────────────────────────
    rows_by_liq = sorted(rows, key=lambda r: -r["daily_dollar_vol"])
    n = len(rows_by_liq)
    top_half = rows_by_liq[: n // 2]
    bottom_half = rows_by_liq[n // 2:]
    top_quartile = rows_by_liq[: n // 4]

    print("\n=== LIQUIDITY TIERS (real avg daily $ volume that day, from the actual 1-min bars) ===")
    for label, subset in [("Top quartile by $ volume", top_quartile),
                           ("Top half by $ volume", top_half),
                           ("Bottom half by $ volume", bottom_half)]:
        total = sum(r["ret_pct"] for r in subset)
        wins = [r for r in subset if r["win"]]
        print(f"  {label:28s} n={len(subset):3d}  win={len(wins)/len(subset)*100:5.1f}%  "
              f"avg={total/len(subset):+.4f}%  total={total:+.3f}%  "
              f"$vol range=${subset[-1]['daily_dollar_vol']/1e6:.0f}M-${subset[0]['daily_dollar_vol']/1e6:.0f}M")

    # ── Robustness check on the two most promising-looking subsets ─────────
    print("\n=== ROBUSTNESS CHECK (remove top N winners) -- does any subset survive this? ===")
    checks = [
        robustness_check(rows, "FULL population (all 240)"),
        robustness_check(top_quartile, "Top-liquidity quartile"),
        robustness_check(top_half, "Top-liquidity half"),
    ]
    if top5:
        best_ticker_rows = by_ticker[top5[0]["ticker"]]
        checks.append(robustness_check(best_ticker_rows, f"Best single ticker only ({top5[0]['ticker']})"))
    for c in checks:
        print(f"\n  {c['label']}: n={c['n']}  total={c['total_ret_pct']:+.2f}%  "
              f"avg={c['avg_ret_pct']}  win={c['win_rate_pct']}%")
        for k in (1, 2, 3, 5, 10):
            key = f"total_after_removing_top_{k}"
            if key in c:
                print(f"    after removing top {k:2d} winner(s): {c[key]:+.2f}%")

    out = {
        "rsi_threshold": RSI_THRESHOLD,
        "ticker_breakdown": ticker_agg,
        "sector_breakdown": sector_agg,
        "top5_tickers_full_detail": {t["ticker"]: sorted(
            [{"date": r["date"], "ret_pct": r["ret_pct"], "win": r["win"]} for r in by_ticker[t["ticker"]]],
            key=lambda x: x["date"]) for t in top5},
        "liquidity_tiers": {
            "top_quartile": robustness_check(top_quartile, "top_quartile"),
            "top_half": robustness_check(top_half, "top_half"),
            "bottom_half": robustness_check(bottom_half, "bottom_half"),
        },
        "robustness_checks": checks,
    }
    out_path = HERE / "rsi_oversold_breakdown_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
