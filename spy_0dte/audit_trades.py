"""
Data audit 2: recompute randomly sampled candidates straight from the raw
option parquet files with independent code (no DayChain, no QuoteModel) and
compare entry/exit prices and P&L with the stored values.
"""
import sys
import numpy as np
import pandas as pd

NY = "America/New_York"
LAM, COMM = 0.25, 0.65


def width(mid, T):
    return max(0.01, 0.03 * max(mid, 0.0)) * (1.5 if T <= 30 else 1.0)


def raw_price(df, sym, minute_ts, stale=2):
    s = df[df["symbol"] == sym]
    for j in range(stale + 1):
        t = minute_ts - pd.Timedelta(minutes=j)
        if t in s.index:
            r = s.loc[t]
            return float(r["vwap"] if pd.notna(r["vwap"]) else r["close"])
    return np.nan


def raw_exit(df, sym, minute_ts, fwd=10):
    p = raw_price(df, sym, minute_ts)
    if np.isfinite(p):
        return p
    s = df[df["symbol"] == sym]
    for j in range(1, fwd + 1):
        t = minute_ts + pd.Timedelta(minutes=j)
        if t in s.index:
            r = s.loc[t]
            return float(r["vwap"] if pd.notna(r["vwap"]) else r["close"])
    return np.nan


def main(n=40, seed=7):
    c = pd.read_parquet("data/optret_candidates.parquet")
    smp = c.sample(n, random_state=seed)
    bad = 0
    for _, r in smp.iterrows():
        sess = r["session"]
        df = pd.read_parquet(f"data/SPY_0dte_option_bars/{sess:%Y-%m-%d}.parquet")
        open_ = sess + pd.Timedelta(hours=9, minutes=30)
        te = open_ + pd.Timedelta(minutes=int(r["tau"]) + 1)
        T_e, T_x = r["T_entry"], r["T_exit"]
        tx = open_ + pd.Timedelta(minutes=390 - int(T_x)) if True else None
        if r["family"] == "long":
            right, k = r["strikes"][-1], float(r["strikes"][:-1])
            sym = f"SPY{sess:%y%m%d}{right}{int(round(k * 1000)):08d}"
            e, x = raw_price(df, sym, te), raw_exit(df, sym, tx)
            pnl = ((x - LAM * width(x, T_x)) - (e + LAM * width(e, T_e))) * 100 - 2 * COMM
        else:
            ks, kw = r["strikes"][:-1].split("/")
            right = r["strikes"][-1]
            sy = lambda k: f"SPY{sess:%y%m%d}{right}{int(round(float(k) * 1000)):08d}"
            es, ew = raw_price(df, sy(ks), te), raw_price(df, sy(kw), te)
            xs, xw = raw_exit(df, sy(ks), tx), raw_exit(df, sy(kw), tx)
            credit = (es - LAM * width(es, T_e)) - (ew + LAM * width(ew, T_e))
            debit = (xs + LAM * width(xs, T_x)) - max(0.0, xw - LAM * width(xw, T_x))
            pnl = (credit - debit) * 100 - 4 * COMM
        ok = np.isclose(pnl, r["pnl"], atol=0.01, equal_nan=True)
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {sess:%Y-%m-%d} tau {int(r['tau']):3d} {r['struct']:<18} {r['strikes']:<12} "
              f"stored {r['pnl']:8.2f}  independent {pnl:8.2f}")
    print(f"\n{n - bad}/{n} match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
