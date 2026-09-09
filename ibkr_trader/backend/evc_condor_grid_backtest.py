"""
EVC (Earnings Vol Crush) iron-condor GRID historical backtest.

Extends evc_condor_backtest.py's exact methodology -- same 54-ticker universe,
same yfinance price+earnings data source, same Black-Scholes premium
ESTIMATE (realized_vol_20d * {1.3, 1.6} IV-multiplier proxy), same
IV_MULTS / MAX_MOVE_BOUNDS / MIN_MOVE sensitivity dimensions, same payoff
logic -- but generalizes the hardcoded short=1.0x-EM / wing=1.5x-EM strike
structure into a (short_mult, wing_mult) grid. Question being answered: is
there ANY strike-distance/wing-width combination where the real historical
win rate clears the real breakeven requirement with a genuine margin, or does
nothing work (meaning the earnings-condor concept should be retired outright
rather than re-tuned)?

Does NOT modify evc_condor_backtest.py (left untouched as the original
reference record), does NOT import from main.py, does NOT touch any live
file, and does NOT place trades. Read-only research script.

Efficiency note: the yfinance fetch (price history + earnings dates) per
ticker is the expensive part of this backtest and does NOT depend on
short_mult/wing_mult -- only on iv_mult (for the realized-vol-based expected
move per event). So each ticker's data and each event's
spot/expected_move/realized_vol is fetched/computed EXACTLY ONCE per
(ticker, event, iv_mult); the (short_mult, wing_mult) grid is then evaluated
cheaply -- pure Black-Scholes arithmetic at different strikes off that same
cached spot/EM, no re-fetch -- against that cached event data. This keeps
runtime roughly in line with the original single-structure backtest instead
of 20-25x slower.

Run: python evc_condor_grid_backtest.py
Outputs (in this directory):
  evc_condor_grid_summary.csv     -- grid-level aggregate, one row per
                                      (short_mult, wing_mult, iv_mult,
                                      max_move_bound) combo
  evc_condor_grid_per_ticker.csv  -- per-ticker breakdown (N>=8) for the 2-3
                                      best surviving combos only
plus printed analysis to stdout.
"""
from __future__ import annotations

import math
import time
import warnings
from datetime import date
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Universe -- IDENTICAL to evc_condor_backtest.py, unchanged.
# ----------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
    "QCOM", "AVGO", "MU", "CRM", "ADBE", "NOW", "PLTR", "UBER",
    "JPM", "GS", "MS", "AXP", "V", "MA",
    "UNH", "JNJ", "PFE", "LLY", "ISRG",
    "XOM", "CVX", "SLB",
    "DE", "CAT", "BA", "HON",
    "WMT", "COST", "TGT", "HD", "LOW", "TJX", "ROST",
    "MCD", "SBUX", "NKE", "DIS", "NFLX",
    "BABA", "JD", "SHOP", "SNAP", "COIN", "F", "GM",
]

# ----------------------------------------------------------------------------
# Config -- sensitivity dimensions IDENTICAL to evc_condor_backtest.py.
# ----------------------------------------------------------------------------
IV_MULTS = [1.3, 1.6]                 # unchanged
MAX_MOVE_BOUNDS = [0.12, 0.25]        # unchanged
MIN_MOVE = 0.03                       # unchanged
RISK_FREE = 0.045                     # unchanged
DIV_YIELD = 0.0                       # unchanged
T_YEARS = 1.0 / 252.0                 # unchanged
VOL_LOOKBACK_DAYS = 20                # unchanged

# ----------------------------------------------------------------------------
# NEW: strike-selection grid (this is the only thing being generalized).
# short strike  = spot +/- short_mult * expected_move
# wing strike   = spot +/- wing_mult  * expected_move
# (the original script hardcoded short_mult=1.0 implicitly, wing_mult=1.5)
# ----------------------------------------------------------------------------
SHORT_MULTS = [1.0, 1.2, 1.4, 1.6, 1.8]
WING_MULTS = [1.3, 1.6, 2.0, 2.5, 3.0]
VALID_COMBOS = [(s, w) for s in SHORT_MULTS for w in WING_MULTS if w > s]

MIN_MARGIN_PTS = 5.0                  # step-2 survival bar (both iv_mults)
MIN_TICKER_N = 8                      # per-ticker robustness threshold
STRATEGY_RISK_CAP_DOLLARS = 210.0     # ~5% of ~$4,200 combined net liq


def round_strike(price: float) -> float:
    """Exact replica of _evc_round_strike() in main.py (default step=0.0 branch)."""
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


def bs_call_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Black-Scholes European call & put premiums."""
    if S <= 0 or K <= 0:
        return 0.0, 0.0
    if T <= 0 or sigma <= 0:
        call = max(0.0, S - K)
        put = max(0.0, K - S)
        return call, put
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return max(call, 0.0), max(put, 0.0)


def fetch_ticker_data(ticker: str, retries: int = 3):
    """Fetch 5y unadjusted-close history + up to 20 real earnings dates."""
    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5y", interval="1d", auto_adjust=False, actions=False)
            if hist is None or hist.empty:
                raise ValueError("empty history")
            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()

            edf = t.get_earnings_dates(limit=20)
            if edf is None or edf.empty:
                raise ValueError("no earnings dates")
            return hist, edf
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: could not fetch data ({last_err})")
    return None, None


@dataclass
class Row:
    ticker: str
    earnings_date: str
    timing: str
    entry_date: str
    outcome_date: str
    iv_mult: float
    short_mult: float
    wing_mult: float
    status: str
    reason: str = ""
    spot: float = np.nan
    atm_strike: float = np.nan
    realized_vol_20d: float = np.nan
    iv_estimate: float = np.nan
    expected_move: float = np.nan
    im_pct: float = np.nan
    short_put: float = np.nan
    long_put: float = np.nan
    short_call: float = np.nan
    long_call: float = np.nan
    put_width: float = np.nan
    call_width: float = np.nan
    wing_width: float = np.nan
    net_credit: float = np.nan
    max_risk: float = np.nan
    breakeven_win_rate_trade: float = np.nan
    real_price: float = np.nan
    real_move_pct: float = np.nan
    payoff: float = np.nan
    payoff_pct_spot: float = np.nan
    payoff_pct_credit: float = np.nan
    win: object = None
    qualifies_12: bool = False
    qualifies_25: bool = False


def process_ticker(ticker: str, rows: list):
    hist, edf = fetch_ticker_data(ticker)
    if hist is None:
        return

    trading_days = list(hist.index.date)
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values
    log_ret = np.diff(np.log(closes))

    def next_trading_day(d: date):
        for td in trading_days:
            if td > d:
                return td
        return None

    def prev_trading_day(d: date):
        prev = None
        for td in trading_days:
            if td >= d:
                break
            prev = td
        return prev

    def trading_day_on_or_after(d: date):
        for td in trading_days:
            if td >= d:
                return td
        return None

    today = date.today()
    n_events = 0
    for ts, erow in edf.iterrows():
        edate = pd.Timestamp(ts)
        edate_local = edate.date()
        if edate_local >= today:
            continue
        reported_eps = erow.get("Reported EPS", np.nan)
        if pd.isna(reported_eps):
            continue

        hour = edate.hour
        timing = "BMO" if hour < 12 else "AMC"

        if timing == "BMO":
            base_day = edate_local if edate_local in td_index else trading_day_on_or_after(edate_local)
            if base_day is None:
                continue
            entry_date = prev_trading_day(base_day)
        else:
            entry_date = edate_local if edate_local in td_index else prev_trading_day(
                trading_day_on_or_after(edate_local) or edate_local
            )
        if entry_date is None or entry_date not in td_index:
            continue
        outcome_date = next_trading_day(entry_date)
        if outcome_date is None:
            continue

        n_events += 1
        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]

        base_row_kwargs = dict(
            ticker=ticker,
            earnings_date=str(edate_local),
            timing=timing,
            entry_date=str(entry_date),
            outcome_date=str(outcome_date),
        )

        def add_skip_all_combos(status: str, reason: str, iv_mult_val, extra: dict | None = None):
            """One row per iv_mult (event-level skip/reject -- no combo evaluated)."""
            kw = dict(base_row_kwargs)
            if extra:
                kw.update(extra)
            rows.append(Row(iv_mult=iv_mult_val, short_mult=np.nan, wing_mult=np.nan,
                             status=status, reason=reason, **kw))

        if entry_idx < VOL_LOOKBACK_DAYS:
            for m in IV_MULTS:
                add_skip_all_combos("skip", "insufficient_history", m)
            continue

        spot = float(closes[entry_idx])
        real_price = float(closes[outcome_idx])
        if spot <= 0 or real_price <= 0:
            for m in IV_MULTS:
                add_skip_all_combos("skip", "bad_price", m)
            continue

        window = log_ret[entry_idx - VOL_LOOKBACK_DAYS: entry_idx]
        if len(window) < VOL_LOOKBACK_DAYS or np.any(np.isnan(window)):
            for m in IV_MULTS:
                add_skip_all_combos("skip", "bad_vol_window", m)
            continue
        realized_vol = float(np.std(window, ddof=1) * math.sqrt(252))
        if realized_vol <= 0 or not math.isfinite(realized_vol):
            for m in IV_MULTS:
                add_skip_all_combos("skip", "zero_realized_vol", m)
            continue

        atm_strike = round_strike(spot)
        real_move_pct = (real_price - spot) / spot

        # ---- per (ticker, event, iv_mult): compute spot/EM/realized_vol ONCE ----
        for iv_mult in IV_MULTS:
            iv_est = realized_vol * iv_mult
            call_atm, put_atm = bs_call_put(spot, atm_strike, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            expected_move = call_atm + put_atm
            im_pct = expected_move / spot if spot > 0 else np.nan

            event_common = dict(
                spot=spot, atm_strike=atm_strike, realized_vol_20d=realized_vol,
                iv_estimate=iv_est, expected_move=expected_move, im_pct=im_pct,
                real_price=real_price, real_move_pct=real_move_pct,
            )

            if not math.isfinite(im_pct) or im_pct < MIN_MOVE:
                add_skip_all_combos("reject", "below_min_move", iv_mult, event_common)
                continue
            if im_pct > max(MAX_MOVE_BOUNDS):
                add_skip_all_combos("reject", "above_max_move_bound", iv_mult, event_common)
                continue

            qualifies_12 = im_pct <= 0.12
            qualifies_25 = im_pct <= 0.25

            # ---- cheap part: evaluate EVERY (short_mult, wing_mult) combo ----
            # against this same cached spot/expected_move -- no re-fetch, no
            # recompute of realized_vol/expected_move.
            for short_mult, wing_mult in VALID_COMBOS:
                r = Row(
                    iv_mult=iv_mult, short_mult=short_mult, wing_mult=wing_mult,
                    status="ok", reason="",
                    qualifies_12=qualifies_12, qualifies_25=qualifies_25,
                    **event_common, **base_row_kwargs,
                )

                short_put = round_strike(spot - short_mult * expected_move)
                short_call = round_strike(spot + short_mult * expected_move)
                long_put = round_strike(spot - wing_mult * expected_move)
                long_call = round_strike(spot + wing_mult * expected_move)

                if not (long_put < short_put < short_call < long_call):
                    r.status = "skip"
                    r.reason = "invalid_strikes"
                    rows.append(r)
                    continue

                sc_call, _ = bs_call_put(spot, short_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
                _, sp_put = bs_call_put(spot, short_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
                lc_call, _ = bs_call_put(spot, long_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
                _, lp_put = bs_call_put(spot, long_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)

                sc_prem, sp_prem, lc_prem, lp_prem = sc_call, sp_put, lc_call, lp_put

                net_credit = round((sp_prem + sc_prem) - (lp_prem + lc_prem), 4)
                if net_credit <= 0:
                    r.status = "skip"
                    r.reason = "non_positive_credit"
                    rows.append(r)
                    continue

                put_width = short_put - long_put
                call_width = long_call - short_call
                wing_width = max(put_width, call_width)
                max_risk = wing_width - net_credit
                if max_risk <= 0:
                    r.status = "skip"
                    r.reason = "non_positive_risk"
                    rows.append(r)
                    continue

                if short_put <= real_price <= short_call:
                    payoff = net_credit
                elif real_price < short_put:
                    breach = short_put - real_price
                    if real_price >= long_put:
                        payoff = net_credit - breach
                    else:
                        payoff = net_credit - (put_width - net_credit)
                else:  # real_price > short_call
                    breach = real_price - short_call
                    if real_price <= long_call:
                        payoff = net_credit - breach
                    else:
                        payoff = net_credit - (call_width - net_credit)

                r.short_put, r.long_put = short_put, long_put
                r.short_call, r.long_call = short_call, long_call
                r.put_width, r.call_width, r.wing_width = put_width, call_width, wing_width
                r.net_credit = net_credit
                r.max_risk = max_risk
                r.breakeven_win_rate_trade = max_risk / (max_risk + net_credit)
                r.payoff = payoff
                r.payoff_pct_spot = payoff / spot
                r.payoff_pct_credit = payoff / net_credit
                r.win = bool(payoff > 0)
                rows.append(r)

    print(f"  {ticker}: {n_events} historical earnings events processed")


def compute_grid_summary(ok: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for short_mult, wing_mult in VALID_COMBOS:
        for iv_mult in IV_MULTS:
            for bound in MAX_MOVE_BOUNDS:
                qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
                sub = ok[
                    (ok["short_mult"] == short_mult) & (ok["wing_mult"] == wing_mult)
                    & (ok["iv_mult"] == iv_mult) & (ok[qcol] == True)  # noqa: E712
                    & (ok["im_pct"] >= MIN_MOVE)
                ]
                n = len(sub)
                if n == 0:
                    summary_rows.append(dict(
                        short_mult=short_mult, wing_mult=wing_mult,
                        iv_mult=iv_mult, max_move_bound=bound, n=0,
                    ))
                    continue
                win_rate = sub["win"].mean()
                sum_risk = sub["max_risk"].sum()
                sum_credit = sub["net_credit"].sum()
                pooled_breakeven = sum_risk / (sum_risk + sum_credit)
                mean_rr = (sub["max_risk"] / sub["net_credit"]).mean()
                avg_net_credit_dollars = sub["net_credit"].mean() * 100.0
                avg_max_risk_dollars = sub["max_risk"].mean() * 100.0
                summary_rows.append(dict(
                    short_mult=short_mult, wing_mult=wing_mult,
                    iv_mult=iv_mult, max_move_bound=bound, n=n,
                    win_rate=win_rate,
                    pooled_breakeven_win_rate=pooled_breakeven,
                    margin_vs_breakeven_pts=(win_rate - pooled_breakeven) * 100.0,
                    mean_risk_reward_ratio=mean_rr,
                    avg_net_credit_dollars=avg_net_credit_dollars,
                    avg_max_risk_dollars=avg_max_risk_dollars,
                    avg_payoff_pct_spot=sub["payoff_pct_spot"].mean(),
                    avg_payoff_pct_credit=sub["payoff_pct_credit"].mean(),
                ))
    return pd.DataFrame(summary_rows)


def main():
    rows: list = []
    print(f"Processing {len(UNIVERSE)} tickers x {len(VALID_COMBOS)} valid (short_mult,wing_mult) combos...")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}")
        try:
            process_ticker(ticker, rows)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    print(f"\nTotal rows generated: {len(df)}")

    ok = df[df["status"] == "ok"].copy()

    # -------------------------------------------------------------------
    # Grid-level summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("GRID-LEVEL SUMMARY")
    print("=" * 100)
    summary_df = compute_grid_summary(ok)
    summary_df.to_csv("evc_condor_grid_summary.csv", index=False)
    print(f"Saved {len(summary_df)} grid summary rows to evc_condor_grid_summary.csv")

    # -------------------------------------------------------------------
    # Step 2: identify combos with >=+5pt margin under BOTH iv_mults
    # simultaneously, for each max_move_bound.
    # -------------------------------------------------------------------
    survivors = []
    for short_mult, wing_mult in VALID_COMBOS:
        for bound in MAX_MOVE_BOUNDS:
            rows_bw = summary_df[
                (summary_df["short_mult"] == short_mult)
                & (summary_df["wing_mult"] == wing_mult)
                & (summary_df["max_move_bound"] == bound)
            ]
            row_13 = rows_bw[rows_bw["iv_mult"] == 1.3]
            row_16 = rows_bw[rows_bw["iv_mult"] == 1.6]
            if row_13.empty or row_16.empty:
                continue
            m13 = row_13["margin_vs_breakeven_pts"].iloc[0]
            m16 = row_16["margin_vs_breakeven_pts"].iloc[0]
            n13 = row_13["n"].iloc[0]
            n16 = row_16["n"].iloc[0]
            if pd.isna(m13) or pd.isna(m16):
                continue
            if m13 >= MIN_MARGIN_PTS and m16 >= MIN_MARGIN_PTS:
                survivors.append(dict(
                    short_mult=short_mult, wing_mult=wing_mult, max_move_bound=bound,
                    margin_1p3=m13, margin_1p6=m16, n_1p3=n13, n_1p6=n16,
                    min_margin=min(m13, m16),
                ))

    survivors_df = pd.DataFrame(survivors).sort_values("min_margin", ascending=False) if survivors else pd.DataFrame()
    print(f"\nCombos clearing +{MIN_MARGIN_PTS:.0f}pt margin under BOTH iv_mult=1.3x AND iv_mult=1.6x: {len(survivors_df)}")
    if not survivors_df.empty:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(survivors_df.to_string(index=False))
    else:
        print("  NONE. No (short_mult, wing_mult) combo clears breakeven with a real margin under both IV assumptions.")

    # -------------------------------------------------------------------
    # Step 3: per-ticker breakdown for the 2-3 best surviving combos
    # -------------------------------------------------------------------
    best_combo_keys = []
    bar_met = not survivors_df.empty
    if not survivors_df.empty:
        seen = set()
        for _, srow in survivors_df.iterrows():
            key = (srow["short_mult"], srow["wing_mult"])
            if key not in seen:
                seen.add(key)
                best_combo_keys.append(key)
            if len(best_combo_keys) >= 3:
                break
    else:
        # No combo clears the +5pt bar under both IV assumptions. For honesty
        # and completeness (per task step 3), still run the per-ticker
        # breakdown on the 2-3 CLOSEST-to-breakeven combos in the whole grid
        # (ranked by min(margin_1.3, margin_1.6) at max_move_bound=0.12) so we
        # can show explicitly whether even the best-of-a-failing-grid combo
        # holds up ticker-by-ticker, or is failing everywhere uniformly.
        print(f"\n[NOTE] No combo cleared the +{MIN_MARGIN_PTS:.0f}pt bar. Falling back to the "
              f"2-3 CLOSEST-to-breakeven combos (still failing) for the per-ticker honesty check.")
        b12 = summary_df[summary_df["max_move_bound"] == 0.12]
        p13 = b12[b12["iv_mult"] == 1.3].set_index(["short_mult", "wing_mult"])["margin_vs_breakeven_pts"]
        p16 = b12[b12["iv_mult"] == 1.6].set_index(["short_mult", "wing_mult"])["margin_vs_breakeven_pts"]
        both = pd.concat([p13.rename("m13"), p16.rename("m16")], axis=1).dropna()
        both["min_margin"] = both[["m13", "m16"]].min(axis=1)
        both = both.sort_values("min_margin", ascending=False)
        print("\nClosest-to-breakeven combos (all still FAIL the bar, shown for completeness):")
        print(both.head(5).round(2).to_string())
        for key in both.head(3).index.tolist():
            best_combo_keys.append(key)

    per_ticker_records = []
    if best_combo_keys:
        print("\n" + "=" * 100)
        label = "best surviving combos" if bar_met else "closest-to-breakeven combos (NONE actually clear the bar)"
        print(f"PER-TICKER BREAKDOWN for {label}: {best_combo_keys}")
        print("=" * 100)
        for short_mult, wing_mult in best_combo_keys:
            for iv_mult in IV_MULTS:
                for bound in MAX_MOVE_BOUNDS:
                    qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
                    sub = ok[
                        (ok["short_mult"] == short_mult) & (ok["wing_mult"] == wing_mult)
                        & (ok["iv_mult"] == iv_mult) & (ok[qcol] == True)  # noqa: E712
                        & (ok["im_pct"] >= MIN_MOVE)
                    ]
                    if sub.empty:
                        continue
                    grp = sub.groupby("ticker").agg(
                        n=("win", "size"),
                        win_rate=("win", "mean"),
                        avg_net_credit_dollars=("net_credit", lambda x: x.mean() * 100.0),
                        avg_max_risk_dollars=("max_risk", lambda x: x.mean() * 100.0),
                    )
                    grp["short_mult"] = short_mult
                    grp["wing_mult"] = wing_mult
                    grp["iv_mult"] = iv_mult
                    grp["max_move_bound"] = bound
                    per_ticker_records.append(grp.reset_index())

        per_ticker_df = pd.concat(per_ticker_records, ignore_index=True)
        per_ticker_df.to_csv("evc_condor_grid_per_ticker.csv", index=False)
        print(f"Saved {len(per_ticker_df)} per-ticker rows to evc_condor_grid_per_ticker.csv")

        eligible = per_ticker_df[per_ticker_df["n"] >= MIN_TICKER_N]
        print(f"\nTickers with N>={MIN_TICKER_N} qualifying trades for a best combo:")
        if eligible.empty:
            print("  NONE. No ticker reaches the N>=8 robustness threshold for any surviving combo.")
        else:
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(eligible.sort_values(["short_mult", "wing_mult", "iv_mult", "max_move_bound", "ticker"]).to_string(index=False))
    else:
        # Still write an empty file with headers so downstream tooling doesn't break.
        pd.DataFrame(columns=[
            "ticker", "n", "win_rate", "avg_net_credit_dollars", "avg_max_risk_dollars",
            "short_mult", "wing_mult", "iv_mult", "max_move_bound",
        ]).to_csv("evc_condor_grid_per_ticker.csv", index=False)
        print("\nNo surviving combos -- wrote empty evc_condor_grid_per_ticker.csv (headers only).")

    # -------------------------------------------------------------------
    # Step 4: capital-sizing check
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"CAPITAL-SIZING CHECK (strategy risk cap ~${STRATEGY_RISK_CAP_DOLLARS:.0f}, 1 contract)")
    if not bar_met:
        print("(shown for the closest-to-breakeven combos even though none clear the +5pt bar)")
    print("=" * 100)
    if best_combo_keys:
        for short_mult, wing_mult in best_combo_keys:
            rows_bw = summary_df[(summary_df["short_mult"] == short_mult) & (summary_df["wing_mult"] == wing_mult)]
            for _, rr in rows_bw.iterrows():
                if rr["n"] == 0:
                    continue
                over = rr["avg_max_risk_dollars"] > STRATEGY_RISK_CAP_DOLLARS
                flag = "  <-- EXCEEDS CAP" if over else ""
                print(f"  short={short_mult} wing={wing_mult} iv={rr['iv_mult']} bound={rr['max_move_bound']:.0%}: "
                      f"avg_max_risk=${rr['avg_max_risk_dollars']:.0f}  avg_net_credit=${rr['avg_net_credit_dollars']:.0f}{flag}")
    else:
        print("  N/A -- no surviving combo to size.")

    print("\nDone.")


if __name__ == "__main__":
    main()
