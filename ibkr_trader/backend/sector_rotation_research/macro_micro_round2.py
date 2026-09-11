"""
Round 2 of "what else explains it" -- continuing from what_drives_it.py
(VIX partially explains it, MU earnings don't, generic size-factor looks
unlikely). New candidates:

MACRO:
  D. Dollar strength (UUP) -- memory/semis have heavy Asia manufacturing +
     international revenue exposure; dollar moves could differentially
     hit Memory vs mega-cap Hyperscalers (more domestic-services-weighted).
  E. Growth-vs-value factor (IWF vs IWD) -- is the Hyperscalers-Memory
     spread just a repackaged growth/value rotation, or genuinely distinct
     from that broader factor?

MICRO (industry-specific, not just this account's own tickers):
  F. Global memory-industry proxy (EWY, S.Korea ETF -- Samsung + SK Hynix
     are the other 2 major HBM/DRAM producers worldwide, both South
     Korea-listed). If Memory's relative moves correlate with EWY
     independent of the broader US market, that's evidence of a real
     GLOBAL MEMORY INDUSTRY factor (DRAM/HBM supply-demand news that hits
     the whole industry, not just MU) rather than a US-domestic or
     AI-capex-narrative-specific effect.
  G. NVDA's own earnings calendar -- mirrors the MU test, but for the
     single most important stock in the whole AI trade. Does the
     Hyperscalers-Memory (or Hyperscalers-Chipmakers) spread show extreme
     days clustering around NVDA earnings specifically?
  H. Broad semis benchmark (SOXX) vs Memory -- is Memory's rotation
     against Hyperscalers really an AI-capex story, or just "memory being
     a normal cyclical sub-segment of the broader semis cycle," which
     would show up as Memory correlating tightly with SOXX itself.
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


def fetch_ret(ticker, start, end):
    s = yf.Ticker(ticker).history(start=start, end=end + pd.Timedelta(days=1))["Close"].pct_change().dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main():
    returns = load_returns()
    basket_ret = {name: basket_returns(returns, tickers) for name, tickers in BASKETS.items()}
    all_df = pd.concat(basket_ret, axis=1, join="inner")
    daily_avg = all_df.mean(axis=1)
    rel = all_df.sub(daily_avg, axis=0)
    start, end = rel.index[0], rel.index[-1]
    print(f"Window: {start.date()} -> {end.date()}  ({len(rel)} days)")

    hyp_rel, mem_rel = rel["Hyperscalers"], rel["Memory"]
    spread = hyp_rel - mem_rel

    # ── D. Dollar strength ──────────────────────────────────────────
    print("\n=== D. Dollar strength (UUP) ===")
    uup = fetch_ret("UUP", start, end)
    j = pd.concat([hyp_rel.rename("hyp"), mem_rel.rename("mem"), spread.rename("spread"),
                   uup.rename("uup")], axis=1, join="inner")
    print(f"Hyperscalers rel vs UUP return: r={j['hyp'].corr(j['uup']):+.3f}")
    print(f"Memory rel vs UUP return:       r={j['mem'].corr(j['uup']):+.3f}")
    print(f"Spread vs UUP return:           r={j['spread'].corr(j['uup']):+.3f}")

    # ── E. Growth vs value factor ───────────────────────────────────
    print("\n=== E. Growth (IWF) vs Value (IWD) factor ===")
    iwf = fetch_ret("IWF", start, end)
    iwd = fetch_ret("IWD", start, end)
    gv_spread = (iwf - iwd).dropna()
    j2 = pd.concat([spread.rename("spread"), gv_spread.rename("gv_spread")], axis=1, join="inner")
    r_gv = j2["spread"].corr(j2["gv_spread"])
    print(f"Hyperscalers-Memory spread vs (Growth-Value) factor spread: r={r_gv:+.3f}")
    print("(If this is high, the 'AI rotation' is largely a repackaged growth/value factor. "
          "If low, it's genuinely distinct from the broad growth/value rotation.)")

    # ── F. Global memory-industry proxy (EWY) ───────────────────────
    print("\n=== F. Global memory-industry proxy (EWY, S.Korea -- Samsung + SK Hynix home market) ===")
    ewy = fetch_ret("EWY", start, end)
    j3 = pd.concat([mem_rel.rename("mem"), hyp_rel.rename("hyp"), ewy.rename("ewy")], axis=1, join="inner")
    print(f"Memory relative return vs EWY return:       r={j3['mem'].corr(j3['ewy']):+.3f}")
    print(f"Hyperscalers relative return vs EWY return: r={j3['hyp'].corr(j3['ewy']):+.3f}")
    # Does MU's OWN raw return (not relative) correlate with EWY, independent of the US market?
    mu_raw = returns["MU"]
    spy_proxy = daily_avg  # the 5-basket common factor as a market proxy
    j4 = pd.concat([mu_raw.rename("mu"), ewy.rename("ewy"), spy_proxy.rename("mkt")], axis=1, join="inner").dropna()
    slope_mkt, intercept_mkt, *_ = stats.linregress(j4["mkt"], j4["mu"])
    mu_resid = j4["mu"] - (intercept_mkt + slope_mkt * j4["mkt"])
    slope_ewy_mkt, intercept_ewy_mkt, *_ = stats.linregress(j4["mkt"], j4["ewy"])
    ewy_resid = j4["ewy"] - (intercept_ewy_mkt + slope_ewy_mkt * j4["mkt"])
    resid_corr = mu_resid.corr(ewy_resid)
    print(f"MU raw return vs EWY raw return, BOTH residualized against the common AI-basket factor: "
          f"r={resid_corr:+.3f}")
    print("(A real positive residual correlation here means MU and the Korean memory-producer market "
          "move together on days that AREN'T explained by the broader AI/market factor -- i.e. a real "
          "global memory-industry-specific co-movement, not just 'everything in tech moves together.')")

    # ── G. NVDA earnings clustering ──────────────────────────────────
    print("\n=== G. NVDA earnings-date clustering ===")
    nvda = yf.Ticker("NVDA")
    try:
        edates = nvda.get_earnings_dates(limit=20)
        real_past = [d.tz_localize(None) if d.tz is not None else d
                     for d in edates.index if start <= d.tz_localize(None) <= end]
        print(f"NVDA real earnings dates in window: {[d.date() for d in real_past]}")
    except Exception as exc:
        print(f"Could not fetch NVDA earnings dates: {exc}")
        real_past = []

    top10_lead = spread.nlargest(10).index   # Hyperscalers lead / Memory lags most
    top10_lag = spread.nsmallest(10).index   # Memory leads / Hyperscalers lags most
    near_nvda = sum(1 for d in list(top10_lead) + list(top10_lag)
                     if any(abs((d - e).days) <= 2 for e in real_past))
    print(f"{near_nvda}/20 of the Hyperscalers-Memory spread's most extreme days fall within 2 days "
          f"of a real NVDA earnings date (out of {len(real_past)} earnings dates, "
          f"base rate if random: ~{len(real_past)*5/len(spread)*100:.1f}%).")

    # ── H. Broad semis benchmark (SOXX) ──────────────────────────────
    print("\n=== H. Memory vs broad semis benchmark (SOXX) ===")
    soxx = fetch_ret("SOXX", start, end)
    j5 = pd.concat([mem_rel.rename("mem"), soxx.rename("soxx"), daily_avg.rename("mkt")], axis=1, join="inner").dropna()
    slope_soxx_mkt, int_soxx_mkt, *_ = stats.linregress(j5["mkt"], j5["soxx"])
    soxx_resid = j5["soxx"] - (int_soxx_mkt + slope_soxx_mkt * j5["mkt"])
    corr_mem_soxx_resid = j5["mem"].corr(soxx_resid)
    print(f"Memory relative return vs SOXX residual (SOXX with the AI-basket common factor removed): "
          f"r={corr_mem_soxx_resid:+.3f}")
    print("(If Memory's rotation is mostly 'broad semis cycle,' this should be strongly positive -- "
          "Memory relative gains would coincide with SOXX's own outperformance of the AI-basket factor.)")


if __name__ == "__main__":
    main()
