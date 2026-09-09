"""
v2 -- fixes a real lookahead-bias bug in butterfly_gex_sr_analysis.py.

The first version used /api/stock/{ticker}/gex-levels, which returns a
single "latest as of" snapshot per date -- confirmed its timestamp was
16:14 ET (after the close), not something knowable at the real 9:45 ET
entry decision. Its dramatic-looking 70%-vs-17.6% win-rate split by
"distance to gamma_magnet" is therefore suspect: an EOD gamma magnet can
partly reflect where price actually ended up that day, which is exactly
the circularity a real forward-test must avoid.

Fix: /api/stock/{ticker}/spot-exposures returns real PER-MINUTE gamma/
charm/vanna exposure (confirmed 535 real timestamped rows/day, starting
~6:30 ET) -- this version picks the real row closest to 9:45 ET
specifically, so every number used was genuinely knowable at the moment
this strategy would actually decide to enter. The S/R half of v1 (prior
trading day's high/low) had no lookahead issue to begin with -- carried
over unchanged.

Output: butterfly_gex_sr_enriched_v2.csv
"""
import sys
import io
import json
from datetime import date, datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import requests
import yfinance as yf

with open("scanner_config.json") as f:
    _cfg = json.load(f)
UW_API_KEY = _cfg["unusual_whales_api_key"]
ENTRY_SEARCH_WINDOW_MIN = 20


def get_gamma_at_945(ticker: str, d: date) -> dict | None:
    headers = {"Authorization": f"Bearer {UW_API_KEY}", "Accept": "application/json"}
    try:
        r = requests.get(f"https://api.unusualwhales.com/api/stock/{ticker}/spot-exposures",
                          headers=headers, params={"date": d.isoformat()}, timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    rows = r.json().get("data") or []
    if not rows:
        return None
    target = datetime(d.year, d.month, d.day, 13, 45, tzinfo=timezone.utc)  # 9:45 ET = 13:45 UTC (EDT)
    best, best_diff = None, None
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
        except Exception:
            continue
        diff = abs((ts - target).total_seconds())
        if diff <= ENTRY_SEARCH_WINDOW_MIN * 60 and (best_diff is None or diff < best_diff):
            best, best_diff = row, diff
    if best is None:
        return None
    try:
        return {
            "gamma_oi": float(best.get("gamma_per_one_percent_move_oi") or 0),
            "gamma_dir": float(best.get("gamma_per_one_percent_move_dir") or 0),
            "price_at_snapshot": float(best.get("price") or 0),
            "snapshot_time": best.get("time"),
        }
    except (TypeError, ValueError):
        return None


def main():
    df = pd.read_csv("butterfly_0dte_v2_rows.csv")
    ok = df[df.status == "ok"].copy()
    print(f"Base dataset: {len(ok)} real rows")

    unique_td = ok[["ticker", "entry_date"]].drop_duplicates()
    print(f"Fetching REAL 9:45 ET gamma exposure for {len(unique_td)} unique ticker/date pairs "
          f"(genuine intraday snapshot, no lookahead)...")
    gamma_cache = {}
    for i, row in enumerate(unique_td.itertuples(), 1):
        d = date.fromisoformat(row.entry_date)
        gamma_cache[(row.ticker, row.entry_date)] = get_gamma_at_945(row.ticker, d)
        if i % 30 == 0:
            print(f"  {i}/{len(unique_td)}")

    n_missing = sum(1 for v in gamma_cache.values() if v is None)
    print(f"Real 9:45 ET gamma data missing for {n_missing}/{len(unique_td)} ticker/date pairs")

    for field in ["gamma_oi", "gamma_dir", "snapshot_time"]:
        ok[field] = ok.apply(lambda r: (gamma_cache.get((r.ticker, r.entry_date)) or {}).get(field), axis=1)

    # ── Same S/R enrichment as v1 -- no lookahead issue (prior day is fully known) ──
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
    ok.to_csv("butterfly_gex_sr_enriched_v2.csv", index=False)
    print(f"\nSaved enriched dataset: {len(ok)} rows")

    def report_split(name, mask):
        mask = mask.fillna(False)
        sub_in, sub_out = ok[mask], ok[~mask]
        print(f"\n--- {name} ---")
        print(f"  IN  (n={len(sub_in):3d}): win_rate={(sub_in.long_win).mean():.1%}  "
              f"mean=${sub_in.payoff_dollar.mean():+.2f}  median=${sub_in.payoff_dollar.median():+.2f}")
        print(f"  OUT (n={len(sub_out):3d}): win_rate={(sub_out.long_win).mean():.1%}  "
              f"mean=${sub_out.payoff_dollar.mean():+.2f}  median=${sub_out.payoff_dollar.median():+.2f}")

    print("\n" + "=" * 70)
    print("ONE FACTOR AT A TIME -- real 9:45 ET data only, no lookahead")
    print("=" * 70)

    valid_gamma = ok.gamma_oi.notna()
    print(f"\nRows with usable real 9:45 ET gamma data: {valid_gamma.sum()}/{len(ok)}")
    if valid_gamma.sum() > 20:
        report_split("Positive real 9:45 gamma (gamma_oi > 0)", (ok.gamma_oi > 0) & valid_gamma)
        high_abs_gamma = (ok.gamma_oi.abs() > ok.gamma_oi.abs().median()) & valid_gamma
        report_split("High |gamma| magnitude at 9:45 (above median)", high_abs_gamma)

    report_split("Near prior-day high/low (within 0.3% of spot)", ok.near_prior_day_range_edge)

    print("\nDone.")


if __name__ == "__main__":
    main()
