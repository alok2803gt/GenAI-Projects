"""Rule-based "AE-clone": buy pullbacks in long-term winners while SPY is in an uptrend.

Rules come from analyze_entries.py / analyze_exits.py medians, NOT from optimising this backtest:
  regime   no new entries while SPY closes below its 200-day SMA
  setup    12-1 month momentum > 0, price 8-30% below its 52-week high, below its 50-day SMA, RSI14 < 50
  rank     strongest 12-1 momentum first; fill up to MAX_POS equal-weight slots (5% of equity each)
  exits    stop: close <= entry * (1 - STOP)
           trail: once the close has been >= entry * (1 + TRAIL_ARM), exit when close <= peak * (1 - TRAIL)
           time: held MAX_HOLD trading days
  fills    signal on the close, filled at the next open, 5 bps slippage + $1 per order
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
PX = {f.stem: json.loads(f.read_text()) for f in (HERE / "data" / "prices").glob("*.json")}
ETFS = {"VOO", "QQQ", "SPMO", "VHT", "XLP", "TAN", "IHI", "SPY"}
AE_UNIV = sorted({p["sym"] for p in json.load(open(HERE / "data" / "aelong_positions.json"))} - ETFS)
SP100 = [s for s in (HERE / "sp100.txt").read_text().split() if s in PX]
START, CAPITAL, SLIP, COMM = "2022-07-01", 100_000.0, 0.0005, 1.0


def indicators(bars):
    c = [b["c"] for b in bars]
    out = []
    for i in range(len(c)):
        if i < 253:
            out.append(None)
            continue
        g = sum(max(c[k] - c[k - 1], 0) for k in range(i - 13, i + 1))
        l = sum(max(c[k - 1] - c[k], 0) for k in range(i - 13, i + 1))
        out.append({"sma50": sum(c[i - 49:i + 1]) / 50, "sma200": sum(c[i - 199:i + 1]) / 200,
                    "hi252": max(c[i - 251:i + 1]), "mom12_1": c[i - 21] / c[i - 252] - 1,
                    "rsi": 100 - 100 / (1 + g / (l or 1e-9))})
    return out


IND = {s: indicators(b) for s, b in PX.items()}
IDX = {s: {b["d"]: i for i, b in enumerate(b_)} for s, b_ in PX.items()}
DAYS = [b["d"] for b in PX["SPY"] if b["d"] >= START]


def run(universe, STOP=0.15, TRAIL=0.10, TRAIL_ARM=0.10, MAX_HOLD=180, MAX_POS=20, WEEKLY=True,
        LO=-0.30, HI=-0.08, start=START):
    cash, pos, trades, curve = CAPITAL, {}, [], []
    pending_buys, pending_sells = [], []
    days = [d for d in DAYS if d >= start]
    for di, d in enumerate(days):
        # 1) execute yesterday's signals at today's open
        for s in pending_sells:
            i = IDX[s].get(d)
            if s in pos and i is not None:
                p = pos.pop(s)
                px = PX[s][i]["o"] * (1 - SLIP)
                cash += p["qty"] * px - COMM
                trades.append({"sym": s, "ret": px / p["entry"] - 1, "hold": p["held"], "exit": d})
        equity = cash + sum(p["qty"] * PX[s][IDX[s][d]]["o"] for s, p in pos.items() if d in IDX[s])
        for s in pending_buys:
            i = IDX[s].get(d)
            if i is None or s in pos or len(pos) >= MAX_POS:
                continue
            px = PX[s][i]["o"] * (1 + SLIP)
            alloc = min(equity / MAX_POS, cash - COMM)
            qty = int(alloc // px)
            if qty <= 0:
                continue
            cash -= qty * px + COMM
            pos[s] = {"qty": qty, "entry": px, "peak": px, "held": 0}
        pending_buys, pending_sells = [], []
        # 2) mark to market at the close, update exits
        equity = cash
        for s, p in pos.items():
            i = IDX[s].get(d)
            if i is None:
                continue
            c = PX[s][i]["c"]
            p["held"] += 1
            p["peak"] = max(p["peak"], c)
            equity += p["qty"] * c
            armed = p["peak"] >= p["entry"] * (1 + TRAIL_ARM)
            if (c <= p["entry"] * (1 - STOP) or (armed and c <= p["peak"] * (1 - TRAIL))
                    or p["held"] >= MAX_HOLD):
                pending_sells.append(s)
        curve.append((d, equity))
        # 3) entry signals (on the close)
        spy = IND["SPY"][IDX["SPY"][d]]
        if spy is None or PX["SPY"][IDX["SPY"][d]]["c"] < spy["sma200"]:
            continue
        if WEEKLY and di + 1 < len(days) and (int(d[8:]) and __import__("datetime").date.fromisoformat(d).weekday() != 3):
            continue
        slots = MAX_POS - len(pos) + len(pending_sells)
        if slots <= 0:
            continue
        cands = []
        for s in universe:
            i = IDX[s].get(d)
            if i is None or s in pos:
                continue
            f = IND[s][i]
            if not f:
                continue
            c = PX[s][i]["c"]
            off_hi = c / f["hi252"] - 1
            if f["mom12_1"] > 0 and LO <= off_hi <= HI and c < f["sma50"] and f["rsi"] < 50:
                cands.append((f["mom12_1"], s))
        pending_buys = [s for _, s in sorted(cands, reverse=True)[:slots]]
    return stats(curve, trades)


def stats(curve, trades):
    eq = [e for _, e in curve]
    yrs = (len(eq) - 1) / 252
    peak, mdd = eq[0], 0
    for e in eq:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    sharpe = st.mean(rets) / (st.pstdev(rets) or 1) * 252 ** 0.5
    return {"total": eq[-1] / eq[0] - 1, "cagr": (eq[-1] / eq[0]) ** (1 / yrs) - 1, "mdd": mdd, "sharpe": sharpe,
            "trades": len(trades), "win": sum(t["ret"] > 0 for t in trades) / max(len(trades), 1),
            "hold": st.median([t["hold"] for t in trades]) if trades else 0}


def spy_bh(start=START):
    b = [x for x in PX["SPY"] if x["d"] >= start]
    return stats([(x["d"], x["c"]) for x in b], [])


def fmt(r):
    return (f"total {r['total']:+7.1%}  CAGR {r['cagr']:+6.1%}  maxDD {r['mdd']:6.1%}  Sharpe {r['sharpe']:4.2f}  "
            f"trades {r['trades']:3}  win {r['win']:.0%}  med hold {r['hold']}d")


if __name__ == "__main__":
    print(f"universes: AE-own {len(AE_UNIV)} stocks, S&P100 {len(SP100)} stocks; test {DAYS[0]} -> {DAYS[-1]}\n")
    print(f"SPY buy&hold          {fmt(spy_bh())}")
    for uname, u in (("AE-own", AE_UNIV), ("S&P100", SP100)):
        print(f"DEFAULT {uname:8}      {fmt(run(u))}")
    print("\nsame window as AELong's record (2023-05-25 ->): AELong official +112.2% total")
    print(f"SPY buy&hold          {fmt(spy_bh('2023-05-25'))}")
    for uname, u in (("AE-own", AE_UNIV), ("S&P100", SP100)):
        print(f"DEFAULT {uname:8}      {fmt(run(u, start='2023-05-25'))}")
    print("\nrobustness grid (S&P100, full period)  STOP / TRAIL / MAX_HOLD / weekly")
    for stop, trail, mh, wk in itertools.product((0.10, 0.15, 0.20), (0.08, 0.12), (120, 250), (True, False)):
        r = run(SP100, STOP=stop, TRAIL=trail, MAX_HOLD=mh, WEEKLY=wk)
        print(f"  {stop:.2f} / {trail:.2f} / {mh:3} / {'wk ' if wk else 'day'}  {fmt(r)}")
