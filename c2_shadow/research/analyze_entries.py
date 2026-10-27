"""What does AELong select for, and when does it sell?

Entry features use the close BEFORE the entry day (what was knowable). Each feature is reported two ways:
  raw     median at entry vs median over every (symbol, day) in the same universe and period (control)
  xs-rank cross-sectional percentile of the entered stock among all universe stocks that day (0.5 = average)
A feature that drives selection shows xs-rank far from 0.5.
"""
import json
import statistics as st
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
PX = {f.stem: json.loads(f.read_text()) for f in (HERE / "data" / "prices").glob("*.json")}
POS = json.load(open(HERE / "data" / "aelong_positions.json"))
ETFS = {"VOO", "QQQ", "SPMO", "VHT", "XLP", "TAN", "IHI", "SPY"}
DATES = {s: [b["d"] for b in bars] for s, bars in PX.items()}


def feats(sym, i):
    """Features from bars[..i] inclusive (i = last known close)."""
    b = PX[sym]
    if i < 252:
        return None
    c = [x["c"] for x in b[: i + 1]]
    last = c[-1]
    sma = lambda n: sum(c[-n:]) / n
    hi52, lo52 = max(c[-252:]), min(c[-252:])
    gains = [max(c[k] - c[k - 1], 0) for k in range(len(c) - 14, len(c))]
    losses = [max(c[k - 1] - c[k], 0) for k in range(len(c) - 14, len(c))]
    rsi = 100 - 100 / (1 + (sum(gains) / (sum(losses) or 1e-9)))
    rets = [c[k] / c[k - 1] - 1 for k in range(len(c) - 60, len(c))]
    tr = [max(b[k]["h"], b[k - 1]["c"]) - min(b[k]["l"], b[k - 1]["c"]) for k in range(i - 19, i + 1)]
    return {
        "mom1m": last / c[-22] - 1, "mom3m": last / c[-64] - 1, "mom6m": last / c[-127] - 1,
        "mom12_1": c[-22] / c[-253] - 1, "ret5d": last / c[-6] - 1, "ret10d": last / c[-11] - 1,
        "vs_sma20": last / sma(20) - 1, "vs_sma50": last / sma(50) - 1, "vs_sma200": last / sma(200) - 1,
        "from_52w_hi": last / hi52 - 1, "from_52w_lo": last / lo52 - 1, "rsi14": rsi,
        "vol60": st.pstdev(rets) * (252 ** 0.5), "atr20_pct": sum(tr) / 20 / last,
    }


def idx_before(sym, day):
    """index of last bar strictly before `day` (YYYY-MM-DD)."""
    return bisect_left(DATES[sym], day) - 1


KEYS = ["mom1m", "mom3m", "mom6m", "mom12_1", "ret5d", "ret10d", "vs_sma20", "vs_sma50", "vs_sma200",
        "from_52w_hi", "from_52w_lo", "rsi14", "vol60", "atr20_pct"]
STOCKS = [s for s in PX if s not in ETFS]
_xs_cache = {}


def xs_rank(day, sym, key):
    if day not in _xs_cache:
        rows = {}
        for s in STOCKS:
            i = idx_before(s, day)
            f = feats(s, i) if i > 0 else None
            if f:
                rows[s] = f
        _xs_cache[day] = rows
    rows = _xs_cache[day]
    if sym not in rows or len(rows) < 20:
        return None
    vals = sorted(r[key] for r in rows.values())
    return bisect_left(vals, rows[sym][key]) / (len(vals) - 1)


def main():
    stock_pos = [p for p in POS if p["sym"] in STOCKS]
    entries, ranks = [], {k: [] for k in KEYS}
    for p in stock_pos:
        day = p["o"][:10]
        i = idx_before(p["sym"], day)
        f = feats(p["sym"], i)
        if not f:
            continue
        p["f"] = f
        entries.append(f)
        for k in KEYS:
            r = xs_rank(day, p["sym"], k)
            if r is not None:
                ranks[k].append(r)
    # control: every symbol, every 5th trading day across the trading period
    start, end = POS[0]["o"][:10], max(p["c"] for p in POS)[:10]
    ctrl = []
    for s in STOCKS:
        for i, d in enumerate(DATES[s]):
            if start <= d <= end and i % 5 == 0:
                f = feats(s, i)
                if f:
                    ctrl.append(f)
    print(f"stock entries analysed: {len(entries)} (of {len(stock_pos)} stock positions); control samples: {len(ctrl)}\n")
    print(f"{'feature':12} {'entry med':>10} {'ctrl med':>10} {'xs-rank mean':>13} {'xs>0.7':>7} {'xs<0.3':>7}")
    for k in KEYS:
        em, cm = st.median(e[k] for e in entries), st.median(c[k] for c in ctrl)
        rk = ranks[k]
        print(f"{k:12} {em:>10.3f} {cm:>10.3f} {st.mean(rk):>13.2f} {sum(r > .7 for r in rk) / len(rk):>7.0%} "
              f"{sum(r < .3 for r in rk) / len(rk):>7.0%}")
    # SPY regime at entry
    spy_up = [feats("SPY", idx_before("SPY", p["o"][:10]))["vs_sma200"] > 0 for p in POS]
    print(f"\nSPY above its 200-day at entry: {sum(spy_up) / len(spy_up):.0%} of entries")
    json.dump(stock_pos, open(HERE / "data" / "aelong_positions_feats.json", "w"), default=str)


if __name__ == "__main__":
    main()
