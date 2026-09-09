"""
Historical earnings-move dispersion screen -- backtest, CEO-approved 2026-08-30.

Question: would screening EVC candidates on their OWN trailing earnings-move
history (independent of the current quarter's ATM-implied move) have flagged
or avoided the two worst real losses this session (CRWD -$849, CRM -$612)?

Finding that motivated this: CRM's implied move at entry (7.18%) almost
exactly matched its own 14-quarter historical average (7.31%) -- the market
wasn't obviously underpricing volatility on average. The real problem is
dispersion/tail risk: both CRM and CRWD have historically blown through 7%
roughly 1/3 of the time, with several 10-20%+ prints. A condor sized off the
mean (even at the current 1.3x-EM cushion) has no real protection against
that tail once the stock actually gaps 20%+ -- that's a "should this name be
in the pool at this size" problem, not a strike-selection problem.

Method: reuse cushion_symmetry_test.py's real-data event pipeline (real
Polygon option prices, no Black-Scholes, current live cushions 1.3/1.3/1.5)
to get real historical EVC trade candidates. For EACH event, independently
compute that ticker's trailing (up to 8, STRICTLY BEFORE this event's own
entry_date -- no lookahead) historical earnings-day moves via yfinance, using
the same window-based max-daily-move method used interactively to check
CRM/CRWD's real history (handles BMO/AMC ambiguity by scanning a window
around the earnings date rather than assuming timing). Partition events into
"would pass screen" / "would be excluded" under a few candidate screen
thresholds, and compare real close-based payoff between the two groups.

Skips the IBKR-minute-bar exit-timing step entirely (not needed for a
close-based screen test, and IBKR/TWS isn't running as of this test --
2026-08-30 is a Sunday). Uses each event's own real daily close (yfinance,
un-adjusted for splits via the same unadjust() factor build_candidates_real
uses for spot) as the settlement price instead.
"""
import re
import time

import pandas as pd
import yfinance as yf

from cushion_symmetry_test import build_candidates_real, payoff_at_price, make_unadjust_fn

TRAILING_QUARTERS = 8
HIST_YEARS = "5y"
EARNINGS_LOOKBACK = 24  # deep enough that even the oldest event in the ~2y
                        # window still has a full trailing-8 lookback available


def fetch_real_series(ticker: str, retries=3):
    """Real (split-unadjusted) close series + real earnings dates for `ticker`.
    Mirrors cushion_symmetry_test.fetch_ticker_data's own unadjust logic so
    the screen and the live strategy agree on what a "real" price is.
    """
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=HIST_YEARS, interval="1d", auto_adjust=False, actions=False)
            if hist is None or hist.empty:
                raise ValueError("empty history")
            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            edf = t.get_earnings_dates(limit=EARNINGS_LOOKBACK)
            if edf is None or edf.empty:
                raise ValueError("no earnings dates")
            edf.index = pd.to_datetime(edf.index).tz_localize(None)
            splits = t.splits
            if splits is not None and not splits.empty:
                splits = splits.copy()
                splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            unadjust = make_unadjust_fn(splits)
            real_close = hist["Close"] * pd.Series(
                [unadjust(d.date()) for d in hist.index], index=hist.index)
            return real_close, edf
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    print(f"  [SKIP] {ticker}: could not fetch real series")
    return None, None


def trailing_moves_before(real_close: pd.Series, edf: pd.DataFrame, cutoff) -> list[float]:
    """Real trailing earnings-day |% move|, using only earnings dates
    strictly before `cutoff` -- no lookahead. Most-recent-first, up to
    TRAILING_QUARTERS values.
    """
    moves = []
    for ts in sorted(edf.index, reverse=True):
        edate = ts.normalize().date()
        if edate >= cutoff:
            continue
        idx = real_close.index.searchsorted(pd.Timestamp(edate))
        lo, hi = max(0, idx - 2), min(len(real_close), idx + 3)
        window = real_close.iloc[lo:hi]
        if len(window) < 2:
            continue
        daily_pct = window.pct_change().dropna() * 100
        if daily_pct.empty:
            continue
        moves.append(abs(float(daily_pct.abs().max())))
        if len(moves) >= TRAILING_QUARTERS:
            break
    return moves


def real_close_on_or_after(real_close: pd.Series, d) -> float | None:
    idx = real_close.index.searchsorted(pd.Timestamp(d))
    if idx >= len(real_close):
        return None
    return float(real_close.iloc[idx])


def main():
    main_src = open("../main.py", encoding="utf-8").read()
    m = re.search(r'CANDIDATE_POOL: List\[str\] = \[(.*?)\]\n', main_src, re.DOTALL)
    all_tickers = re.findall(r'"([A-Z\.]+)"', m.group(1))
    tickers = all_tickers[::2]
    print(f"Universe: {len(all_tickers)} total, sampling {len(tickers)} (every 2nd)")

    print("Building candidate events with REAL Polygon option prices (no Black-Scholes)...")
    events, stats = build_candidates_real(tickers)
    print(f"\nPipeline funnel: {stats}")
    print(f"{len(events)} qualifying real-priced events")
    if not events:
        print("No events -- nothing to backtest.")
        return

    series_cache: dict = {}
    rows = []
    for i, ev in enumerate(events):
        if ev.ticker not in series_cache:
            series_cache[ev.ticker] = fetch_real_series(ev.ticker)
        real_close, edf = series_cache[ev.ticker]
        if real_close is None:
            continue

        outcome_close = real_close_on_or_after(real_close, ev.outcome_date)
        if outcome_close is None:
            continue
        close_payoff = payoff_at_price(
            outcome_close, ev.short_put, ev.long_put, ev.short_call, ev.long_call,
            ev.net_credit, ev.put_width, ev.call_width)

        moves = trailing_moves_before(real_close, edf, ev.entry_date)
        implied_pct = round(ev.expected_move / ev.spot * 100, 2) if ev.spot else None

        rows.append({
            "ticker": ev.ticker, "entry_date": str(ev.entry_date),
            "outcome_date": str(ev.outcome_date), "spot": round(ev.spot, 2),
            "implied_move_pct": implied_pct, "net_credit": ev.net_credit,
            "outcome_close": round(outcome_close, 2), "close_payoff": round(close_payoff, 2),
            "n_trailing_quarters": len(moves),
            "trailing_avg_abs_move_pct": round(sum(moves) / len(moves), 2) if moves else None,
            "trailing_max_abs_move_pct": round(max(moves), 2) if moves else None,
            "trailing_blowouts_gt12pct": sum(1 for x in moves if x > 12),
            "trailing_blowouts_gt15pct": sum(1 for x in moves if x > 15),
        })
        if (i + 1) % 15 == 0:
            print(f"  ... {i+1}/{len(events)} events processed")

    df = pd.DataFrame(rows)
    df = df[df["n_trailing_quarters"] >= 4]  # need a real trailing sample to screen on
    df.to_csv("historical_move_screen_events.csv", index=False)
    print(f"\n{len(df)} events with real close payoff + a real trailing-move history (>=4 quarters)")

    baseline_n = len(df)
    baseline_total = df["close_payoff"].sum() * 100
    baseline_win = (df["close_payoff"] > 0).mean() * 100
    baseline_worst = df["close_payoff"].min() * 100
    print(f"\n=== BASELINE (no screen), close-only payoff, n={baseline_n} ===")
    print(f"total P&L: ${baseline_total:.2f}   win rate: {baseline_win:.1f}%   worst single trade: ${baseline_worst:.2f}")

    screens = {
        ">=1 trailing quarter >15% move": df["trailing_blowouts_gt15pct"] >= 1,
        ">=2 trailing quarters >12% move": df["trailing_blowouts_gt12pct"] >= 2,
        "trailing max move >15%": df["trailing_max_abs_move_pct"] > 15,
        "trailing avg move > 1.3x implied": df["trailing_avg_abs_move_pct"] > df["implied_move_pct"] * 1.3,
    }

    for name, mask in screens.items():
        mask = mask.fillna(False)
        kept, excl = df[~mask], df[mask]
        print(f"\n=== Screen: exclude if {name} ===")
        tickers_excl = sorted(excl["ticker"].unique())
        print(f"  excludes {len(excl)}/{baseline_n} events, {len(tickers_excl)} tickers: "
              f"{tickers_excl[:15]}{'...' if len(tickers_excl) > 15 else ''}")
        if len(kept):
            print(f"  KEPT n={len(kept):3d}  total P&L: ${kept['close_payoff'].sum()*100:8.2f}  "
                  f"win rate: {(kept['close_payoff']>0).mean()*100:5.1f}%  worst: ${kept['close_payoff'].min()*100:8.2f}")
        if len(excl):
            print(f"  EXCL n={len(excl):3d}  total P&L: ${excl['close_payoff'].sum()*100:8.2f}  "
                  f"win rate: {(excl['close_payoff']>0).mean()*100:5.1f}%  worst: ${excl['close_payoff'].min()*100:8.2f}")

    for tk in ["CRM", "CRWD"]:
        sub = df[df["ticker"] == tk]
        if not sub.empty:
            print(f"\n{tk} rows in this backtest (sanity-check vs. the real trades):")
            print(sub.to_string(index=False))
        else:
            print(f"\n{tk}: no qualifying event found in this backtest sample.")

    print("\nSaved historical_move_screen_events.csv")


if __name__ == "__main__":
    main()
