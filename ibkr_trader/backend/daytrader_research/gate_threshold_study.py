"""Do the Day Trader's ATR / sigma entry gates earn their place?

WHY (2026-09-29)
----------------
On 2026-09-28 the scanner surfaced EIGHT candidates above the composite-score
threshold (BE 93.6, COHR 82.4, INTC 82.2, MRNA 81.5, AKAM 80.2, GEN 79.1,
DELL 78.0, ORCL 75.5) and the agent entered none. All were stopped by
day_trader_agent.py:1345-1350, which applies

    atr_mult >= 1.8   AND   std_score >= 3.7

as a strict conjunction with an early return. Since the 2026-09-23 forming-bar
fix both are measured on the LAST COMPLETED session, so the live requirement is
"yesterday's range was >= 1.8x ATR14 AND yesterday's move was >= 3.7 sigma".

The question is NOT "how do we get more trades" -- loosening a gate because it
blocked trades is exactly the reasoning that produced the level-break and IDR
false edges. The question is whether these gates add anything OVER the
composite score that already selects the candidate.

METHOD
------
Target: the signed same-day OPEN->CLOSE return, because Day Trader buys. The
scanner's study targeted |move| >= 0.5%; a long position needs the sign.

Three disciplines this codebase has learned the hard way:
  * NO LOOK-AHEAD. Every feature at day t uses bars strictly before t, except
    gap_pct which uses open[t] (known at entry time) and nothing else.
  * MARKET-ADJUSTED. Raw same-day returns mostly measure whether the market
    went up. Each row's excess = its return minus the equal-weighted mean of
    the whole universe THAT DAY.
  * DATE-CLUSTERED. Every candidate on one morning is ONE event, not N
    independent bets. All t-stats are computed on the series of DAILY means,
    which is what collapsed the IDR result from t=7.37 to t=0.48.

CAVEAT, stated up front: this runs on the 112-ticker/5-year panel that is
actually on disk (universe_5y_ohlcv.pkl, 140,409 ticker-days), NOT the
500-ticker/609,859-ticker-day study that produced the 1.8x/3.7sigma numbers --
that script is not on this machine. Percentile ranks are therefore taken within
a 112-name universe rather than ~483, so absolute scores are not identical to
live. Pass RATES and relative lift are the outputs to trust here.

    ../../venv/bin/python daytrader_research/gate_threshold_study.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PANEL = HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl"

SCORE_MIN = 75.0        # day_trader_agent config min_composite_score
ATR_REQ = 1.8           # config atr_multiplier
SIG_REQ = 3.7           # config std_dev_threshold
MIN_ATR_PCT = 2.5       # config min_atr_pct (use_vol_filter)


def build(panel_path: Path = None) -> pd.DataFrame:
    """panel_path lets the holdout run through IDENTICAL feature code."""
    panel = pickle.load(open(panel_path or PANEL, "rb"))
    frames = []
    for tkr, df in panel.items():
        d = df[["Open", "High", "Low", "Close"]].copy()
        if len(d) < 40:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        o, h, l, c = d.Open, d.High, d.Low, d.Close

        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        # ATR14 through the last COMPLETED session -> shift(1) so day t sees only t-1 back.
        atr14 = tr.rolling(14).mean().shift(1)
        prev_range = (h - l).shift(1)
        prev_close = c.shift(1)

        ret_cc = c.pct_change()
        ret_std20 = ret_cc.rolling(20).std().shift(1)

        d["atr14"] = atr14
        d["atr_pct"] = atr14 / prev_close * 100
        d["atr_mult"] = prev_range / atr14
        d["std_score"] = ret_cc.shift(1).abs() / ret_std20
        d["gap_pct"] = (o - prev_close) / prev_close * 100
        d["prior_day_ret_pct"] = ((c - o) / o * 100).shift(1)
        d["ret5d_prior"] = (c.shift(1) / c.shift(6) - 1) * 100
        d["target"] = (c - o) / o * 100          # signed same-day open->close
        d["ticker"] = tkr
        frames.append(d.reset_index().rename(columns={"index": "date", "Date": "date"}))

    r = pd.concat(frames, ignore_index=True)
    r = r.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["atr_pct", "atr_mult", "std_score", "gap_pct",
                "prior_day_ret_pct", "ret5d_prior", "target"])
    return r


def add_scores(r: pd.DataFrame) -> pd.DataFrame:
    """compute_dt_scores(), reproduced: percentile ranks WITHIN each day."""
    g = r.groupby("date")
    pr_atr = g["atr_pct"].rank(pct=True) * 100
    pr_gap = g["gap_pct"].transform(lambda s: s.abs().rank(pct=True)) * 100
    pr_pdr = g["prior_day_ret_pct"].transform(lambda s: s.abs().rank(pct=True)) * 100
    pr_r5d = g["ret5d_prior"].transform(lambda s: s.abs().rank(pct=True)) * 100
    r["composite_score"] = (0.55 * pr_atr + 0.25 * pr_gap
                            + 0.12 * pr_pdr + 0.08 * pr_r5d).round(1)
    # Market benchmark: equal-weighted universe open->close, same day.
    r["mkt"] = g["target"].transform("mean")
    r["excess"] = r["target"] - r["mkt"]
    return r


def clustered(sub: pd.DataFrame, col: str = "excess") -> tuple:
    """Mean and t-stat on the series of DAILY means -- one event per day."""
    if sub.empty:
        return (float("nan"),) * 3 + (0,)
    daily = sub.groupby("date")[col].mean()
    n = len(daily)
    if n < 2:
        return daily.mean(), float("nan"), n, len(sub)
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(n))
    return daily.mean(), t, n, len(sub)


def line(label: str, sub: pd.DataFrame, universe_days: int) -> None:
    m_ex, t_ex, ndays, nrows = clustered(sub, "excess")
    m_raw, _, _, _ = clustered(sub, "target")
    hit = (sub["target"] >= 0.5).mean() * 100 if len(sub) else float("nan")
    print(f"  {label:<34}{nrows:>8}{ndays:>7}{ndays / universe_days * 100:>8.1f}%"
          f"{m_raw:>9.3f}{m_ex:>10.3f}{t_ex:>8.2f}{hit:>9.1f}%")


def main() -> None:
    r = add_scores(build())
    universe_days = r["date"].nunique()
    print(f"panel: {r.ticker.nunique()} tickers x {universe_days} sessions = {len(r):,} ticker-days")
    print(f"       {r.date.min().date()} .. {r.date.max().date()}\n")

    cand = r[r.composite_score >= SCORE_MIN]
    vol_ok = cand[cand.atr_pct >= MIN_ATR_PCT]
    atr_ok = vol_ok[vol_ok.atr_mult >= ATR_REQ]
    sig_ok = vol_ok[vol_ok.std_score >= SIG_REQ]
    both = vol_ok[(vol_ok.atr_mult >= ATR_REQ) & (vol_ok.std_score >= SIG_REQ)]
    either = vol_ok[(vol_ok.atr_mult >= ATR_REQ) | (vol_ok.std_score >= SIG_REQ)]

    print("Same-day OPEN->CLOSE return for a LONG entry. 'excess' = minus that day's")
    print("universe mean. t is DATE-CLUSTERED (one event per day). hit = share >= +0.5%.\n")
    print(f"  {'population':<34}{'rows':>8}{'days':>7}{'of all':>9}"
          f"{'raw%':>9}{'excess%':>10}{'t':>8}{'hit':>10}")
    line("all ticker-days (baseline)", r, universe_days)
    line(f"score >= {SCORE_MIN:.0f}", cand, universe_days)
    line(f"  + ATR% >= {MIN_ATR_PCT} (vol filter)", vol_ok, universe_days)
    line(f"    + atr_mult >= {ATR_REQ} only", atr_ok, universe_days)
    line(f"    + std_score >= {SIG_REQ} only", sig_ok, universe_days)
    line("    + BOTH  (live gate, AND)", both, universe_days)
    line("    + EITHER (proposed, OR)", either, universe_days)

    print("\n--- how often is each gate satisfiable at all? ---")
    print(f"  sessions with >=1 candidate scoring {SCORE_MIN:.0f}+ and ATR% ok : "
          f"{vol_ok.date.nunique():>5} / {universe_days}  ({vol_ok.date.nunique()/universe_days*100:.1f}%)")
    for nm, s in (("atr_mult>=1.8 only", atr_ok), ("std_score>=3.7 only", sig_ok),
                  ("BOTH (live)", both), ("EITHER (proposed)", either)):
        print(f"  sessions with >=1 passing {nm:<20}: {s.date.nunique():>5} / {universe_days}"
              f"  ({s.date.nunique()/universe_days*100:.1f}%)")

    print("\n--- does the gate BEAT the candidates it rejects? (paired, same day) ---")
    print("  If the gates add nothing, passers and rejects perform the same.")
    for nm, passed in (("BOTH (live)", both), ("EITHER (proposed)", either),
                       ("atr_mult only", atr_ok), ("std_score only", sig_ok)):
        rej = vol_ok.drop(passed.index)
        dp = passed.groupby("date")["excess"].mean()
        dr = rej.groupby("date")["excess"].mean()
        pair = pd.concat([dp.rename("pass"), dr.rename("rej")], axis=1, sort=False).dropna()
        if len(pair) < 3:
            print(f"  {nm:<20} only {len(pair)} paired days -- cannot test")
            continue
        d = pair["pass"] - pair["rej"]
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        print(f"  {nm:<20} {len(pair):>4} paired days   pass-minus-reject "
              f"{d.mean():+.3f}pp   t={t:+.2f}")

    print("\n--- what threshold would the data pick? (score>=75 & ATR% ok) ---")
    print(f"  {'atr_mult >=':<14}{'rows':>8}{'days':>7}{'excess%':>10}{'t':>8}{'hit':>9}")
    for thr in (0.0, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5):
        s = vol_ok[vol_ok.atr_mult >= thr]
        m, t, nd, nr = clustered(s)
        hit = (s["target"] >= 0.5).mean() * 100 if len(s) else float("nan")
        print(f"  {thr:<14.1f}{nr:>8}{nd:>7}{m:>10.3f}{t:>8.2f}{hit:>8.1f}%")
    print(f"\n  {'std_score >=':<14}{'rows':>8}{'days':>7}{'excess%':>10}{'t':>8}{'hit':>9}")
    for thr in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.7):
        s = vol_ok[vol_ok.std_score >= thr]
        m, t, nd, nr = clustered(s)
        hit = (s["target"] >= 0.5).mean() * 100 if len(s) else float("nan")
        print(f"  {thr:<14.1f}{nr:>8}{nd:>7}{m:>10.3f}{t:>8.2f}{hit:>8.1f}%")

    # The gate question is moot if the edge cannot clear the fee. Sizing note:
    # day_trader_agent.py:767-772 prefers position_size_pct when it is > 0, so
    # the live position is 10% of NET LIQ ($1,481 -> ~$148), NOT the $500/$5000
    # position_size default. IBKR charges $0.005/share, min $1.00, capped at 1%
    # of trade value; between ~$100 and ~$20k notional the $1.00 minimum binds,
    # so the fee is a FIXED $1.00 each way and its percentage cost rises as the
    # position shrinks -- which is exactly the wrong direction for this account.
    print("\n--- does ANY of this survive commissions? ---")
    print(f"  {'position':<12}{'round trip':>12}{'fee %':>8}   edge needed just to break even")
    for pos in (148.0, 500.0, 2000.0, 10000.0):
        rt = 2.00
        print(f"  ${pos:<11,.0f}{'$%.2f' % rt:>12}{rt / pos * 100:>7.2f}%   "
              f"{rt / pos * 100:.2f}pp")
    live_pos = 148.0                       # 10% of the 2026-09-29 net liq
    rt_pct = 2.00 / live_pos * 100
    print(f"\n  at the LIVE size (${live_pos:.0f} = 10% of net liq), round trip = {rt_pct:.2f}%")
    print(f"  {'population':<30}{'excess%':>10}{'minus fees':>12}{'verdict':>12}")
    for nm, s in (("score>=75 (no gate)", vol_ok), ("atr_mult>=1.8", atr_ok),
                  ("std_score>=3.7", sig_ok), ("BOTH (live)", both),
                  ("EITHER (proposed)", either)):
        m, _, _, _ = clustered(s)
        net = m - rt_pct
        print(f"  {nm:<30}{m:>10.3f}{net:>12.3f}{'LOSS' if net < 0 else 'profit':>12}")

    print("\nBonferroni note: 2 gate variants x 9+8 thresholds = 19 comparisons here.")
    print("A single |t|>2 among them is expected by chance; |t|>3.1 is the honest bar.")


if __name__ == "__main__":
    main()
