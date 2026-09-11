"""
Finding 2 (Hyperscalers vs Memory relative-return corr = -0.71, survives
mechanical/beta/outlier stress tests) is real structure. This script tests
CANDIDATE EXPLANATIONS for what's actually driving it -- the "what else
could it be" question -- rather than leaving it as an unexplained
descriptive fact.

Candidates tested:
  A. Risk-on/off macro factor: Memory is smaller-cap/higher-beta than
     Hyperscalers -- maybe this is just a generic risk appetite effect
     (VIX down = small/cyclical outperforms mega-cap), not anything
     AI/memory-specific. Test: correlate the Hyperscalers-Memory relative
     return against same-day VIX change and 10Y yield change.
  B. Idiosyncratic company catalysts: Memory's real earnings/guidance
     dates (MU reports quarterly) could drive its biggest relative-lead
     days independent of any "hyperscaler" story at all. Test: pull MU's
     real historical earnings dates (yfinance), check whether Memory's
     most extreme relative-return days cluster around them.
  C. Generic size-factor effect (not AI-specific at all): if a similar
     relative-rotation pattern shows up between an UNRELATED mega-cap
     basket and an unrelated small/mid-cap basket, that's evidence this
     is really "small-cap vs mega-cap risk appetite," dressed up as an
     "AI sub-sector story" by which specific names happen to be in each
     bucket. Test: same relative-correlation methodology applied to
     QQQ (mega-cap proxy) vs IWM (small-cap proxy) returns.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

HERE = Path(__file__).parent
PKL = HERE / "sector_rotation_ohlcv.pkl"

BASKETS = {
    "Hyperscalers": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "Chipmakers":   ["NVDA", "AMD", "AVGO", "MRVL", "TSM"],
    "Memory":       ["MU", "SNDK"],
    "Storage":      ["STX", "WDC"],
    "SemiCapEquip": ["AMAT", "LRCX", "KLAC"],
}


def load_returns():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    returns = {}
    for ticker, df in data.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        returns[ticker] = df["Close"].pct_change().dropna()
    return returns


def basket_returns(returns, tickers):
    aligned = pd.concat([returns[t].rename(t) for t in tickers], axis=1, join="inner")
    return aligned.mean(axis=1)


def main():
    returns = load_returns()
    basket_ret = {name: basket_returns(returns, tickers) for name, tickers in BASKETS.items()}
    all_df = pd.concat(basket_ret, axis=1, join="inner")
    daily_avg = all_df.mean(axis=1)
    rel = all_df.sub(daily_avg, axis=0)
    hyp_mem_rel = rel["Hyperscalers"] - rel["Memory"]  # the actual spread driving the -0.71
    print(f"Window: {rel.index[0].date()} -> {rel.index[-1].date()}  ({len(rel)} days)")

    # ── A. Macro/risk-on-off factor ─────────────────────────────────────
    print("\n=== A. Macro factor test (VIX, 10Y yield) ===")
    vix = yf.Ticker("^VIX").history(start=rel.index[0], end=rel.index[-1] + pd.Timedelta(days=1))["Close"]
    tnx = yf.Ticker("^TNX").history(start=rel.index[0], end=rel.index[-1] + pd.Timedelta(days=1))["Close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    tnx.index = pd.to_datetime(tnx.index).tz_localize(None)
    vix_chg = vix.diff().dropna()
    tnx_chg = tnx.diff().dropna()

    joined_vix = pd.concat([rel["Hyperscalers"].rename("hyp_rel"), rel["Memory"].rename("mem_rel"),
                             vix_chg.rename("vix_chg")], axis=1, join="inner")
    joined_tnx = pd.concat([rel["Hyperscalers"].rename("hyp_rel"), rel["Memory"].rename("mem_rel"),
                             tnx_chg.rename("tnx_chg")], axis=1, join="inner")

    c_hyp_vix = joined_vix["hyp_rel"].corr(joined_vix["vix_chg"])
    c_mem_vix = joined_vix["mem_rel"].corr(joined_vix["vix_chg"])
    c_hyp_tnx = joined_tnx["hyp_rel"].corr(joined_tnx["tnx_chg"])
    c_mem_tnx = joined_tnx["mem_rel"].corr(joined_tnx["tnx_chg"])
    print(f"Hyperscalers relative return vs VIX change: r={c_hyp_vix:+.3f}")
    print(f"Memory relative return vs VIX change:       r={c_mem_vix:+.3f}")
    print(f"Hyperscalers relative return vs 10Y yield change: r={c_hyp_tnx:+.3f}")
    print(f"Memory relative return vs 10Y yield change:       r={c_mem_tnx:+.3f}")

    # Partial: does the Hyp-Memory spread correlate with VIX/rates changes?
    c_spread_vix = joined_vix["hyp_rel"].sub(joined_vix["mem_rel"]).corr(joined_vix["vix_chg"])
    c_spread_tnx = joined_tnx["hyp_rel"].sub(joined_tnx["mem_rel"]).corr(joined_tnx["tnx_chg"])
    print(f"\n(Hyperscalers-minus-Memory) relative spread vs VIX change: r={c_spread_vix:+.3f}")
    print(f"(Hyperscalers-minus-Memory) relative spread vs 10Y yield change: r={c_spread_tnx:+.3f}")
    print("(A strong positive spread-vs-VIX corr would mean: VIX spikes -> Hyperscalers relatively "
          "outperform Memory -- i.e. this is a flight-to-quality/risk-off effect, not something "
          "AI/memory-specific.)")

    # ── B. Idiosyncratic earnings-date clustering ───────────────────────
    print("\n=== B. MU earnings-date clustering ===")
    mu = yf.Ticker("MU")
    try:
        earnings_dates = mu.get_earnings_dates(limit=20)
        real_past = [d.tz_localize(None) if d.tz is not None else d
                     for d in earnings_dates.index if d.tz_localize(None) <= rel.index[-1]
                     and d.tz_localize(None) >= rel.index[0]]
        print(f"MU real earnings dates in window: {[d.date() for d in real_past]}")
    except Exception as exc:
        print(f"Could not fetch MU earnings dates: {exc}")
        real_past = []

    mem_rel_series = rel["Memory"].dropna()
    top10_mem_lead_days = mem_rel_series.nlargest(10).index
    top10_mem_lag_days = mem_rel_series.nsmallest(10).index
    print(f"\nTop 10 Memory-relatively-LEADS days: {[d.date() for d in top10_mem_lead_days]}")
    print(f"Top 10 Memory-relatively-LAGS days:  {[d.date() for d in top10_mem_lag_days]}")

    near_earnings = 0
    for d in list(top10_mem_lead_days) + list(top10_mem_lag_days):
        if any(abs((d - e).days) <= 2 for e in real_past):
            near_earnings += 1
    print(f"\n{near_earnings}/20 of Memory's most extreme relative days fall within 2 days of a "
          f"real MU earnings date (out of {len(real_past)} earnings dates in the window, "
          f"vs {len(mem_rel_series)} total trading days -- base rate if random: "
          f"~{len(real_past)*5/len(mem_rel_series)*100:.1f}% of days are within 2 days of earnings).")

    # ── C. Generic size-factor sanity check (QQQ vs IWM) ────────────────
    print("\n=== C. Generic mega-cap vs small-cap sanity check (QQQ vs IWM) ===")
    qqq = yf.Ticker("QQQ").history(start=rel.index[0], end=rel.index[-1] + pd.Timedelta(days=1))["Close"].pct_change().dropna()
    iwm = yf.Ticker("IWM").history(start=rel.index[0], end=rel.index[-1] + pd.Timedelta(days=1))["Close"].pct_change().dropna()
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    iwm.index = pd.to_datetime(iwm.index).tz_localize(None)
    joined_size = pd.concat([qqq.rename("qqq"), iwm.rename("iwm")], axis=1, join="inner")
    raw_corr_size = joined_size["qqq"].corr(joined_size["iwm"])
    size_avg = joined_size.mean(axis=1)
    size_rel = joined_size.sub(size_avg, axis=0)
    rel_corr_size = size_rel["qqq"].corr(size_rel["iwm"])
    print(f"QQQ vs IWM raw same-day correlation: {raw_corr_size:+.3f}")
    print(f"QQQ vs IWM RELATIVE-return correlation (same demeaning method): {rel_corr_size:+.3f}")
    print(f"(Compare to Hyperscalers-vs-Memory relative corr of -0.71. Note QQQ/IWM is only 2 "
          f"series so demeaning forces PERFECT -1.0 correlation mechanically with just 2 series -- "
          f"this comparison is illustrative of general size-factor direction, not a clean apples-to-apples "
          f"statistical comparison to the 5-basket result above.)")


if __name__ == "__main__":
    main()
