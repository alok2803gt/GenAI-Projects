"""
Direct test of whether polygon_options_flow_backtest.py's asymmetric
put-skew finding (Memory's options market shifts toward puts specifically
on days it LAGS the rotation, call_skew 0.507 vs 0.553 baseline,
p=0.0067) is a LEADING signal or just a same-day/contemporaneous one.

Method: for each of the 10 hyp_leads days (Memory lags, the group that
showed the real skew shift) and each of the 22 baseline days (for a
matched comparison), fetch real historical MU+SNDK options volume for the
PRIOR real trading day (T-1) and two days prior (T-2). If T-1/T-2 skew is
ALREADY depressed (more puts) before the price divergence happens on day
T, that's a genuine early-warning signal -- informed positioning ahead of
the move. If T-1/T-2 looks like a normal day and the skew shift only
shows up ON day T itself, it's contemporaneous (reactive), not leading.
"""
import gzip
import pickle
import re
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
    y, m = day[:4], day[5:7]
    key = f"us_options_opra/minute_aggs_v1/{y}/{m}/{day}.csv.gz"
    try:
        obj = s3.get_object(Bucket="flatfiles", Key=key)
    except Exception as e:
        return {"_error": str(e)}
    data = gzip.decompress(obj["Body"].read())
    lines = data.decode().splitlines()
    result = {r: {"call_vol": 0, "put_vol": 0} for r in roots}
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
    trading_days = spread.index  # real trading-day calendar for this universe

    top10_hyp_leads = list(spread.nlargest(10).index)

    # Existing 42-day dataset already has these days (T) and a baseline --
    # reuse it for T, only need to fetch NEW T-1/T-2 days here.
    existing = pd.read_csv(HERE / "polygon_options_flow_by_day.csv", parse_dates=["date"])
    existing_dates = set(existing["date"])

    def prior_trading_days(day, n_back):
        loc = trading_days.get_loc(day)
        out = []
        for k in range(1, n_back + 1):
            if loc - k >= 0:
                out.append(trading_days[loc - k])
        return out

    baseline_days = list(existing[existing["group"] == "baseline"]["date"])

    targets = {}  # date -> (offset label, group)
    for d in top10_hyp_leads:
        for k, label in [(1, "T-1"), (2, "T-2")]:
            priors = prior_trading_days(d, 2)
            if k <= len(priors):
                pd_ = priors[k - 1]
                targets[pd_] = ("hyp_leads", label, d)
    for d in baseline_days:
        for k, label in [(1, "T-1"), (2, "T-2")]:
            priors = prior_trading_days(pd.Timestamp(d), 2)
            if k <= len(priors):
                pd_ = priors[k - 1]
                targets[pd_] = ("baseline", label, d)

    to_fetch = [d for d in targets if d not in existing_dates]
    print(f"Need {len(to_fetch)} new days fetched (T-1/T-2 for {len(top10_hyp_leads)} hyp_leads "
          f"days + {len(baseline_days)} baseline days, deduped, minus days already in the existing dataset)")

    s3 = s3_client()
    rows = []
    for i, day in enumerate(sorted(to_fetch)):
        day_str = day.strftime("%Y-%m-%d")
        result = fetch_day_volume(s3, day_str, {"MU", "SNDK"})
        if "_error" in result:
            print(f"  [{i+1}/{len(to_fetch)}] {day_str}: ERROR {result['_error']}")
            continue
        mu, sndk = result["MU"], result["SNDK"]
        call_v = mu["call_vol"] + sndk["call_vol"]
        put_v = mu["put_vol"] + sndk["put_vol"]
        total_v = call_v + put_v
        skew = call_v / total_v if total_v > 0 else np.nan
        group, label, target_day = targets[day]
        rows.append({"date": day_str, "offset": label, "group": group,
                     "target_day": target_day.strftime("%Y-%m-%d") if hasattr(target_day, "strftime") else str(target_day),
                     "total_vol": total_v, "call_skew": skew})
        print(f"  [{i+1}/{len(to_fetch)}] {day_str} ({label} before a {group} day): "
              f"total_vol={total_v:,} call_skew={skew:.3f}")

    new_df = pd.DataFrame(rows)
    new_df.to_csv(HERE / "polygon_leading_signal_data.csv", index=False)
    print(f"\nSaved {len(new_df)} rows to polygon_leading_signal_data.csv")

    # Also pull T (same-day) skew from the existing dataset for reference
    existing_lookup = existing.set_index(existing["date"].dt.strftime("%Y-%m-%d"))["call_skew"].to_dict()

    print("\n=== Call skew by offset and group ===")
    for label in ["T-2", "T-1"]:
        for group in ["hyp_leads", "baseline"]:
            sub = new_df[(new_df["offset"] == label) & (new_df["group"] == group)]
            if len(sub):
                print(f"  {label} / {group:10s}: n={len(sub):2d}  mean_call_skew={sub['call_skew'].mean():.4f}")
    t_label = "T (same day, from prior dataset)"
    hyp_T = existing[existing["group"] == "hyp_leads"]["call_skew"]
    base_T = existing[existing["group"] == "baseline"]["call_skew"]
    print(f"  {t_label} / hyp_leads : n={len(hyp_T):2d}  mean_call_skew={hyp_T.mean():.4f}")
    print(f"  {t_label} / baseline  : n={len(base_T):2d}  mean_call_skew={base_T.mean():.4f}")

    print("\n=== Significance: hyp_leads vs baseline at each offset ===")
    for label in ["T-2", "T-1"]:
        hyp_sub = new_df[(new_df["offset"] == label) & (new_df["group"] == "hyp_leads")]["call_skew"].dropna()
        base_sub = new_df[(new_df["offset"] == label) & (new_df["group"] == "baseline")]["call_skew"].dropna()
        if len(hyp_sub) >= 3 and len(base_sub) >= 3:
            t, p = stats.ttest_ind(hyp_sub, base_sub, equal_var=False)
            print(f"  {label}: t={t:.2f}  p={p:.4f}  (hyp_leads mean={hyp_sub.mean():.4f} vs baseline mean={base_sub.mean():.4f})")
    t_final, p_final = stats.ttest_ind(hyp_T.dropna(), base_T.dropna(), equal_var=False)
    print(f"  T (reference, already found): t={t_final:.2f}  p={p_final:.4f}")


if __name__ == "__main__":
    main()
