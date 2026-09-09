"""
Enriches the existing, already-validated 0DTE butterfly backtest
(butterfly_0dte_v2_rows.csv -- real 9:45 ET entry, real close exit, 270
rows across SPY/QQQ/IWM) with real GEX regime data (Unusual Whales
/api/stock/{ticker}/gex-levels, confirmed live to have real historical
0DTE-relevant gamma levels for this account's plan) and real
support/resistance proximity (prior-day high/low, from the same real
Polygon daily bars already used elsewhere), then checks whether EITHER
factor actually separates winning days from losing days in the data we
already have.

Deliberately does NOT build a combined multi-factor filter -- with only
~90 independent real trading days (SPY/QQQ/IWM move together on most
days), and the baseline edge already shown to be unstable across
resampling, slicing further into multiple simultaneous conditions would
mostly manufacture noise. Each factor is tested one at a time.

Output: butterfly_gex_sr_enriched.csv
"""
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import requests
import yfinance as yf
from datetime import date, timedelta

with open("scanner_config.json") as f:
    _cfg = json.load(f)
UW_API_KEY = _cfg["unusual_whales_api_key"]


def get_gex_levels(ticker: str, d: date) -> dict | None:
    headers = {"Authorization": f"Bearer {UW_API_KEY}", "Accept": "application/json"}
    try:
        r = requests.get(f"https://api.unusualwhales.com/api/stock/{ticker}/gex-levels",
                          headers=headers, params={"date": d.isoformat()}, timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    data = r.json().get("data")
    if not data:
        return None
    try:
        return {
            "call_wall": float(data["call_wall"]) if data.get("call_wall") else None,
            "put_wall": float(data["put_wall"]) if data.get("put_wall") else None,
            "gamma_flip": float(data["gamma_flip"]) if data.get("gamma_flip") else None,
            "gamma_magnet": float(data["gamma_magnet"]) if data.get("gamma_magnet") else None,
        }
    except (TypeError, ValueError):
        return None


def main():
    df = pd.read_csv("butterfly_0dte_v2_rows.csv")
    ok = df[df.status == "ok"].copy()
    print(f"Base dataset: {len(ok)} real rows")

    # ── Real GEX enrichment (fetch once per unique ticker/date, not per row) ──
    unique_td = ok[["ticker", "entry_date"]].drop_duplicates()
    print(f"Fetching real GEX levels for {len(unique_td)} unique ticker/date pairs...")
    gex_cache = {}
    for i, row in enumerate(unique_td.itertuples(), 1):
        d = date.fromisoformat(row.entry_date)
        gex_cache[(row.ticker, row.entry_date)] = get_gex_levels(row.ticker, d)
        if i % 30 == 0:
            print(f"  {i}/{len(unique_td)}")

    n_missing = sum(1 for v in gex_cache.values() if v is None)
    print(f"Real GEX data missing for {n_missing}/{len(unique_td)} ticker/date pairs")

    for field in ["call_wall", "put_wall", "gamma_flip", "gamma_magnet"]:
        ok[field] = ok.apply(lambda r: (gex_cache.get((r.ticker, r.entry_date)) or {}).get(field), axis=1)

    ok["dist_to_flip_pct"] = (ok.morning_spot - ok.gamma_flip) / ok.morning_spot
    ok["dist_to_magnet_pct"] = (ok.morning_spot - ok.gamma_magnet).abs() / ok.morning_spot
    ok["positive_gamma_regime"] = ok.morning_spot > ok.gamma_flip

    # ── Real S/R enrichment: prior trading day's high/low from real Polygon-backed yfinance history ──
    sr_cache = {}
    for ticker in ok.ticker.unique():
        hist = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        sr_cache[ticker] = hist

    def prior_day_hl(ticker, entry_date_str):
        hist = sr_cache[ticker]
        d = date.fromisoformat(entry_date_str)
        prior = hist[hist.index.date < d]
        if prior.empty:
            return None, None
        last = prior.iloc[-1]
        return float(last["High"]), float(last["Low"])

    highs, lows = [], []
    for row in ok.itertuples():
        h, l = prior_day_hl(row.ticker, row.entry_date)
        highs.append(h)
        lows.append(l)
    ok["prior_day_high"] = highs
    ok["prior_day_low"] = lows
    ok["dist_to_prior_high_pct"] = (ok.prior_day_high - ok.morning_spot) / ok.morning_spot
    ok["dist_to_prior_low_pct"] = (ok.morning_spot - ok.prior_day_low) / ok.morning_spot
    ok["near_prior_day_range_edge"] = (ok.dist_to_prior_high_pct.abs() < 0.003) | (ok.dist_to_prior_low_pct.abs() < 0.003)

    ok["payoff_dollar"] = ok.long_payoff * 100
    ok.to_csv("butterfly_gex_sr_enriched.csv", index=False)
    print(f"\nSaved enriched dataset: {len(ok)} rows")

    def report_split(name, mask):
        sub_in, sub_out = ok[mask], ok[~mask]
        print(f"\n--- {name} ---")
        print(f"  IN  (n={len(sub_in):3d}): win_rate={( sub_in.long_win).mean():.1%}  "
              f"mean=${sub_in.payoff_dollar.mean():+.2f}  median=${sub_in.payoff_dollar.median():+.2f}")
        print(f"  OUT (n={len(sub_out):3d}): win_rate={(sub_out.long_win).mean():.1%}  "
              f"mean=${sub_out.payoff_dollar.mean():+.2f}  median=${sub_out.payoff_dollar.median():+.2f}")

    print("\n" + "=" * 70)
    print("ONE FACTOR AT A TIME -- does it separate winners from losers?")
    print("=" * 70)

    valid_gex = ok.gamma_flip.notna()
    print(f"\nRows with usable real GEX data: {valid_gex.sum()}/{len(ok)}")
    if valid_gex.sum() > 20:
        report_split("Positive gamma regime (spot > gamma_flip)",
                      ok.positive_gamma_regime.fillna(False) & valid_gex)
        close_to_magnet = (ok.dist_to_magnet_pct < ok.dist_to_magnet_pct.median()) & valid_gex
        report_split("Close to gamma_magnet (below-median distance)", close_to_magnet)

    report_split("Near prior-day high/low (within 0.3% of spot)", ok.near_prior_day_range_edge.fillna(False))

    print("\nDone.")


if __name__ == "__main__":
    main()
