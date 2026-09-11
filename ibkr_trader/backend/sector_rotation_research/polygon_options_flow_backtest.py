"""
Real historical-data version of the flow-positioning hypothesis that
ewy_leading_signal_test.py and the live UW check couldn't properly test --
UW's API only allows live/current-day queries (403 on any historical
`date` param, confirmed 2026-09-09), and this account's own
darkpool_activity_monitor.py log only covers 2.5 weeks with zero MU hits.

Polygon's flat-files S3 bucket (already used for real 0DTE options
backtests in fetch_real_option_bars.py) has genuine historical minute
aggregates for the WHOLE options market, any past date -- this lets us
actually test whether MU/SNDK's real historical options volume (call vs
put skew, total volume) looked different on the specific extreme
rotation days already found in rotation_analysis.py / relative_rotation.py,
vs a random baseline sample of days.

Test set: the same top-10 Memory-LEADS and top-10 Memory-LAGS days
(Hyperscalers-minus-Memory relative-return spread) identified in
macro_micro_round2.py, plus a random baseline sample for comparison.
Each daily flat file is ~24MB compressed and covers the ENTIRE US options
market -- filtered here to MU/SNDK contracts only via a strict OCC regex
(not a naive substring match, which could false-hit on unrelated tickers).
"""
import gzip
import pickle
import re
from datetime import timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from scipy import stats

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
PKL = HERE / "sector_rotation_ohlcv.pkl"

BASKETS = {
    "Hyperscalers": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "Memory":       ["MU", "SNDK"],
}
_OCC_RE = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")
N_BASELINE_DAYS = 25


def s3_client():
    import json
    with open(BACKEND_DIR / "scanner_config.json") as f:
        cfg = json.load(f)
    session = boto3.Session(
        aws_access_key_id=cfg["polygon_s3_access_key"],
        aws_secret_access_key=cfg["polygon_s3_secret_key"],
    )
    return session.client("s3", endpoint_url="https://files.massive.com",
                           config=Config(signature_version="s3v4"))


def fetch_day_volume(s3, day: str, roots: set[str]) -> dict:
    """day: 'YYYY-MM-DD'. Returns {root: {"call_vol": int, "put_vol": int, "n_contracts": int}}."""
    y, m = day[:4], day[5:7]
    key = f"us_options_opra/minute_aggs_v1/{y}/{m}/{day}.csv.gz"
    try:
        obj = s3.get_object(Bucket="flatfiles", Key=key)
    except Exception as e:
        return {"_error": str(e)}
    data = gzip.decompress(obj["Body"].read())
    lines = data.decode().splitlines()
    result = {r: {"call_vol": 0, "put_vol": 0, "n_contracts": set()} for r in roots}
    header = lines[0].split(",")
    ticker_idx = header.index("ticker")
    volume_idx = header.index("volume")
    for line in lines[1:]:
        parts = line.split(",")
        ticker = parts[ticker_idx]
        m2 = _OCC_RE.match(ticker)
        if not m2:
            continue
        root, yymmdd, right, strike = m2.groups()
        if root not in roots:
            continue
        vol = int(parts[volume_idx])
        if right == "C":
            result[root]["call_vol"] += vol
        else:
            result[root]["put_vol"] += vol
        result[root]["n_contracts"].add(ticker)
    for r in roots:
        result[r]["n_contracts"] = len(result[r]["n_contracts"])
    return result


def load_returns():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    returns = {}
    for ticker, df in data.items():
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).tz_localize(None)
        returns[ticker] = df2["Close"].pct_change().dropna()
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
    spread = (rel["Hyperscalers"] - rel["Memory"]).dropna()

    top10_hyp_leads = spread.nlargest(10).index    # Hyperscalers lead, Memory lags
    top10_mem_leads = spread.nsmallest(10).index   # Memory leads, Hyperscalers lags

    rng = np.random.default_rng(7)
    all_days = spread.index
    baseline_days = pd.DatetimeIndex(rng.choice(all_days, size=N_BASELINE_DAYS, replace=False))

    all_target_days = sorted(set(top10_hyp_leads) | set(top10_mem_leads) | set(baseline_days))
    print(f"Fetching real historical options volume for MU/SNDK across {len(all_target_days)} "
          f"real trading days (this downloads/parses one ~24MB flat file per day)...")

    s3 = s3_client()
    rows = []
    for i, day in enumerate(all_target_days):
        day_str = day.strftime("%Y-%m-%d")
        result = fetch_day_volume(s3, day_str, {"MU", "SNDK"})
        if "_error" in result:
            print(f"  [{i+1}/{len(all_target_days)}] {day_str}: ERROR {result['_error']}")
            continue
        mu, sndk = result["MU"], result["SNDK"]
        call_v = mu["call_vol"] + sndk["call_vol"]
        put_v = mu["put_vol"] + sndk["put_vol"]
        total_v = call_v + put_v
        skew = call_v / total_v if total_v > 0 else np.nan
        group = ("mem_leads" if day in top10_mem_leads else
                 "hyp_leads" if day in top10_hyp_leads else "baseline")
        rows.append({
            "date": day_str, "group": group, "mu_call_vol": mu["call_vol"], "mu_put_vol": mu["put_vol"],
            "sndk_call_vol": sndk["call_vol"], "sndk_put_vol": sndk["put_vol"],
            "total_vol": total_v, "call_skew": skew,
            "hyp_minus_mem_spread": spread.loc[day],
        })
        print(f"  [{i+1}/{len(all_target_days)}] {day_str} ({group}): total_vol={total_v:,} call_skew={skew:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "polygon_options_flow_by_day.csv", index=False)
    print(f"\nSaved {len(df)} rows to polygon_options_flow_by_day.csv")

    print("\n=== Real historical call-skew by group ===")
    for group in ["mem_leads", "hyp_leads", "baseline"]:
        sub = df[df["group"] == group]
        if len(sub) == 0:
            continue
        print(f"  {group:12s}: n={len(sub):2d}  mean_call_skew={sub['call_skew'].mean():.4f}  "
              f"mean_total_vol={sub['total_vol'].mean():,.0f}")

    mem_leads_skew = df[df["group"] == "mem_leads"]["call_skew"].dropna()
    hyp_leads_skew = df[df["group"] == "hyp_leads"]["call_skew"].dropna()
    baseline_skew = df[df["group"] == "baseline"]["call_skew"].dropna()

    if len(mem_leads_skew) >= 3 and len(baseline_skew) >= 3:
        t1, p1 = stats.ttest_ind(mem_leads_skew, baseline_skew, equal_var=False)
        print(f"\nMemory-leads-day call skew vs baseline: t={t1:.2f} p={p1:.4f}")
    if len(hyp_leads_skew) >= 3 and len(baseline_skew) >= 3:
        t2, p2 = stats.ttest_ind(hyp_leads_skew, baseline_skew, equal_var=False)
        print(f"Hyperscalers-leads-day (Memory lags) call skew vs baseline: t={t2:.2f} p={p2:.4f}")

    # Total volume: is unusual options activity itself concentrated on extreme days?
    mem_leads_vol = df[df["group"] == "mem_leads"]["total_vol"]
    baseline_vol = df[df["group"] == "baseline"]["total_vol"]
    if len(mem_leads_vol) >= 3 and len(baseline_vol) >= 3:
        t3, p3 = stats.ttest_ind(mem_leads_vol, baseline_vol, equal_var=False)
        print(f"Memory-leads-day TOTAL volume vs baseline: t={t3:.2f} p={p3:.4f}  "
              f"(mem_leads mean={mem_leads_vol.mean():,.0f} vs baseline mean={baseline_vol.mean():,.0f})")


if __name__ == "__main__":
    main()
