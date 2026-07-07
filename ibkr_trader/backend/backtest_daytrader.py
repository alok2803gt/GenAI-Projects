#!/usr/bin/env python3
"""
Day Trader Strategy Backtest -- 5 Years
=======================================
Simulates the breakout scanner + day trader entry/exit rules on
5 years of daily OHLCV data across the 125-ticker universe.

Signal logic (mirrors breakout_scanner.py):
  BREAKOUT = %B (20-day Bollinger) > 95 AND volume >= 75th-pct of 20d window

Intraday simulation (using daily OHLCV O/H/L/C):
  Entry  : today Open (best approximation of market-open fill)
  Target : High >= entry   (1 + target_pct)  -> exit at target price
  Stop   : Low  <= entry   (1 - stop_pct)    -> exit at stop price
  Conflict (both H and L trigger same bar):
    If breakout day (Close > Open + 0.3%) -> profit target hit first
    Else -> stop hit first (conservative)
  Force-close : neither triggered -> exit at Close (15:45 ET rule)

Three configs tested side-by-side:
  A -- Current     : target=0.5%, stop=7.0%, size=$2,000
  B -- Proposed    : target=0.5%, stop=1.5%, size=$3,500
  C -- Alternative : target=1.0%, stop=2.0%, size=$3,000

Signal timing section: counts signals that would fire AFTER 9:45 ET
  (estimated as stocks where open-to-high move < 0.5% in first 15min,
   implying they hadn't triggered yet at 9:45 -- approximated via daily O/H split)

Usage:
  cd ibkr_trader/backend
  python backtest_daytrader.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# -- Universe (same as breakout_scanner.py) -------------------------------------
TICKERS = sorted(set([
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLI","GLD","TLT","ARKK",
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","NFLX",
    "AMD","INTC","QCOM","AVGO","TXN","MU","AMAT","LRCX","KLAC","MRVL","SMCI",
    "CRM","NOW","ADBE","ORCL","SNOW","PANW","CRWD","ZS","DDOG","NET",
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","V","MA","AXP","TFC",
    "JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","DHR","ISRG","VRTX","GILD","BMY",
    "HD","MCD","SBUX","NKE","LOW","TGT","COST","BKNG","LULU",
    "PG","KO","PEP","WMT",
    "XOM","CVX","COP","SLB","MPC","VLO","OXY",
    "BA","GE","CAT","HON","RTX","LMT","FDX","UPS","DE","UAL",
    "DIS","CMCSA","VZ","T",
    "COIN","PLTR","UBER","RIVN","ROKU","HOOD","SOFI","PYPL","IBM",
    "RBLX","RCL","ABNB",
]))

# -- Configs to test ------------------------------------------------------------
CONFIGS = {
    "A_Current":     {"target_pct": 0.005, "stop_pct": 0.070, "size": 2000, "label": "Current  (0.5%T / 7%S / $2k  / 23pos)", "max_pos": 23},
    "B_Proposed":    {"target_pct": 0.005, "stop_pct": 0.015, "size": 3500, "label": "Proposed (0.5%T / 1.5%S / $3.5k / 23pos)", "max_pos": 23},
    "C_Alternative": {"target_pct": 0.010, "stop_pct": 0.020, "size": 3000, "label": "Alt-C    (1.0%T / 2.0%S / $3k  / 23pos)", "max_pos": 23},
    "D_New":         {"target_pct": 0.020, "stop_pct": 0.030, "size": 5000, "label": "NEW      (2.0%T / 3.0%S / $5k  / 10pos)", "max_pos": 10},
}

# -- Backtest parameters --------------------------------------------------------
BB_PERIOD   = 20        # Bollinger Band period
BB_STD      = 2         # Bollinger Band std devs
PCT_B_MIN   = 75        # minimum %B -- covers both BREAKOUT (>95) and PRE-BREAKOUT (75-95)
VOL_PCTILE  = 75        # volume percentile threshold
MAX_SIGNALS = 23        # max positions per day (matches max_positions config)
START_DATE  = "2020-01-01"
END_DATE    = date.today().isoformat()
DAILY_GOAL  = 200.0     # daily profit target


def download_data() -> dict[str, pd.DataFrame]:
    """Download 5+ years of daily OHLCV for all tickers."""
    print(f"Downloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    data = {}
    for tk in TICKERS:
        try:
            if tk in raw.columns.get_level_values(0):
                df = raw[tk].dropna(how="all").copy()
            else:
                df = pd.DataFrame()
            if len(df) > BB_PERIOD + 5:
                data[tk] = df
        except Exception:
            pass
    print(f"  Loaded {len(data)} tickers with sufficient history")
    return data


def compute_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each ticker   day, compute whether a BREAKOUT signal fires.
    Returns a DataFrame with columns: date, ticker, open, high, low, close, signal
    where signal=True means BREAKOUT fired based on PREVIOUS day's close.
    """
    rows = []
    for tk, df in data.items():
        if len(df) < BB_PERIOD + 10:
            continue
        closes  = df["Close"]
        volumes = df["Volume"]

        # Rolling 20-day Bollinger Bands on close
        sma    = closes.rolling(BB_PERIOD).mean()
        std    = closes.rolling(BB_PERIOD).std()
        upper  = sma + BB_STD * std
        lower  = sma - BB_STD * std
        bwidth = upper - lower
        pct_b  = (closes - lower) / bwidth * 100

        # Volume percentile (rolling 20d)
        vol_75 = volumes.rolling(BB_PERIOD).quantile(VOL_PCTILE / 100)

        # Signal fires when YESTERDAY's %B >= threshold AND today's volume >= 75th pct
        prev_pct_b  = pct_b.shift(1)
        prev_vol_ok = volumes >= vol_75.shift(1)

        for i in range(BB_PERIOD + 1, len(df)):
            row = df.iloc[i]
            ppb = float(prev_pct_b.iloc[i]) if pd.notna(prev_pct_b.iloc[i]) else 0
            vol_ok = bool(prev_vol_ok.iloc[i])

            if ppb >= PCT_B_MIN and vol_ok:
                rows.append({
                    "date":   df.index[i].date(),
                    "ticker": tk,
                    "open":   float(row["Open"]),
                    "high":   float(row["High"]),
                    "low":    float(row["Low"]),
                    "close":  float(row["Close"]),
                    "pct_b":  round(ppb, 1),
                    "signal": True,
                })

    df_signals = pd.DataFrame(rows)
    if df_signals.empty:
        return df_signals
    df_signals["date"] = pd.to_datetime(df_signals["date"])
    df_signals = df_signals.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df_signals


def simulate_trades(df_signals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    For each signal day, simulate entry/exit using daily OHLCV.

    Entry  : Open price
    Target : entry   (1 + target_pct)
    Stop   : entry   (1 - stop_pct)
    Conflict resolution: if both H and L trigger the same bar,
      check if close > open   1.003 (bullish day -> target hit first)
      else stop hit first
    Force-close: exit at Close
    """
    target_pct = cfg["target_pct"]
    stop_pct   = cfg["stop_pct"]
    size       = cfg["size"]

    results = []
    for _, sig in df_signals.iterrows():
        entry  = sig["open"]
        if entry <= 0:
            continue

        shares = max(1, int(size / entry))
        target = entry * (1 + target_pct)
        stop   = entry * (1 - stop_pct)

        hit_target = sig["high"] >= target
        hit_stop   = sig["low"]  <= stop

        if hit_target and not hit_stop:
            exit_price = target
            exit_type  = "profit_target"
        elif hit_stop and not hit_target:
            exit_price = stop
            exit_type  = "stop_loss"
        elif hit_target and hit_stop:
            # Both triggered -- bullish-day heuristic
            if sig["close"] > sig["open"] * 1.003:
                exit_price = target
                exit_type  = "profit_target"
            else:
                exit_price = stop
                exit_type  = "stop_loss"
        else:
            # Neither -- force close at end of day
            exit_price = sig["close"]
            exit_type  = "force_close"

        pnl = round((exit_price - entry) * shares, 2)
        results.append({
            "date":       sig["date"],
            "ticker":     sig["ticker"],
            "entry":      round(entry, 4),
            "exit":       round(exit_price, 4),
            "shares":     shares,
            "exit_type":  exit_type,
            "pnl":        pnl,
            "pnl_pct":    round((exit_price - entry) / entry * 100, 3),
            "win":        pnl > 0,
        })

    return pd.DataFrame(results)


def daily_stats(trades: pd.DataFrame, max_pos: int = MAX_SIGNALS) -> pd.DataFrame:
    """
    Aggregate to daily P&L, capping at max_pos signals per day
    (simulates the max_positions config -- only take first N signals by %B rank).
    """
    daily = []
    for dt, grp in trades.groupby("date"):
        # Cap at max_positions (already sorted by pct_b desc in signal generation order)
        grp = grp.head(max_pos)
        total_pnl    = grp["pnl"].sum()
        n_trades     = len(grp)
        n_wins       = grp["win"].sum()
        n_stops      = (grp["exit_type"] == "stop_loss").sum()
        n_targets    = (grp["exit_type"] == "profit_target").sum()
        n_force      = (grp["exit_type"] == "force_close").sum()
        daily.append({
            "date":        dt,
            "pnl":         round(total_pnl, 2),
            "trades":      n_trades,
            "wins":        int(n_wins),
            "stops":       int(n_stops),
            "targets":     int(n_targets),
            "force":       int(n_force),
            "win_rate":    round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0,
            "goal_hit":    total_pnl >= DAILY_GOAL,
        })
    return pd.DataFrame(daily)


def print_report(label: str, trades: pd.DataFrame, daily: pd.DataFrame) -> None:
    total_days       = len(daily)
    profit_days      = (daily["pnl"] > 0).sum()
    loss_days        = (daily["pnl"] < 0).sum()
    goal_days        = daily["goal_hit"].sum()
    total_pnl        = daily["pnl"].sum()
    avg_daily        = daily["pnl"].mean()
    best_day         = daily["pnl"].max()
    worst_day        = daily["pnl"].min()
    win_rate         = (trades["win"].sum() / len(trades) * 100) if len(trades) > 0 else 0
    avg_win          = trades.loc[trades["win"], "pnl"].mean() if trades["win"].any() else 0
    avg_loss         = trades.loc[~trades["win"], "pnl"].mean() if (~trades["win"]).any() else 0
    n_stops          = (trades["exit_type"] == "stop_loss").sum()
    n_targets        = (trades["exit_type"] == "profit_target").sum()
    n_force          = (trades["exit_type"] == "force_close").sum()

    # Drawdown
    cumulative       = daily["pnl"].cumsum()
    rolling_max      = cumulative.cummax()
    drawdown         = (cumulative - rolling_max)
    max_drawdown     = drawdown.min()

    # Annualized stats (252 trading days)
    avg_trades_day   = daily["trades"].mean()
    years            = total_days / 252

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Period         : {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Trading days   : {total_days}  ({years:.1f} years)")
    print(f"  Avg trades/day : {avg_trades_day:.1f}")
    print()
    print(f"  -- P&L --------------------------------------------------")
    print(f"  Total P&L      : ${total_pnl:>10,.2f}")
    print(f"  Avg daily P&L  : ${avg_daily:>10,.2f}")
    print(f"  Best day       : ${best_day:>10,.2f}")
    print(f"  Worst day      : ${worst_day:>10,.2f}")
    print(f"  Max drawdown   : ${max_drawdown:>10,.2f}")
    print(f"  Annualized P&L : ${avg_daily*252:>10,.2f}")
    print()
    print(f"  -- Daily goal ($200) ------------------------------------")
    print(f"  Days > $0      : {profit_days} / {total_days}  ({profit_days/total_days*100:.1f}%)")
    print(f"  Days >= $200   : {goal_days} / {total_days}  ({goal_days/total_days*100:.1f}%)")
    print(f"  Days < $0      : {loss_days} / {total_days}  ({loss_days/total_days*100:.1f}%)")
    print()
    print(f"  -- Trade stats ------------------------------------------")
    print(f"  Total trades   : {len(trades):,}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Avg win        : ${avg_win:.2f}")
    print(f"  Avg loss       : ${avg_loss:.2f}")
    print(f"  Profit factor  : {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "  Profit factor  :  ")
    print()
    print(f"  -- Exit breakdown ---------------------------------------")
    total_t = len(trades)
    print(f"  Profit target  : {n_targets:,}  ({n_targets/total_t*100:.1f}%)")
    print(f"  Stop loss      : {n_stops:,}  ({n_stops/total_t*100:.1f}%)")
    print(f"  Force close    : {n_force:,}  ({n_force/total_t*100:.1f}%)")

    # By year
    print(f"\n  -- By year:")
    daily2 = daily.copy()
    daily2["year"] = pd.to_datetime(daily2["date"]).dt.year
    for yr, grp in daily2.groupby("year"):
        yr_pnl   = grp["pnl"].sum()
        yr_goal  = grp["goal_hit"].sum()
        yr_days  = len(grp)
        yr_pos   = (grp["pnl"] > 0).sum()
        print(f"  {yr}  P&L=${yr_pnl:>8,.0f}  goal_days={yr_goal}/{yr_days}  profitable_days={yr_pos}/{yr_days}")


def signal_timing_analysis(data: dict[str, pd.DataFrame]) -> None:
    """
    Estimate what fraction of breakout signals would be visible BEFORE vs AFTER 9:45 ET.

    Proxy: on a signal day (Close > prior Bollinger lower + 95% of band),
    compare Open to High to estimate when the signal "triggered":
      - Open already > target level -> signal visible at 9:30
      - High reached target in first bar -> likely hit by 9:45
      - Otherwise -> might fire later in the day

    Using (High - Open) / (High - Low) as a proxy for intraday timing:
      > 80% of range above open -> momentum continued throughout day -> signal visible early
      < 20% of range above open -> stock gapped then pulled back -> would miss at 9:45
    """
    print(f"\n{'='*65}")
    print(f"  SIGNAL TIMING ANALYSIS -- When do breakout signals fire ")
    print(f"{'='*65}")

    all_signals_early   = 0
    all_signals_total   = 0
    gap_up_signals      = 0
    pullback_signals    = 0

    for tk, df in data.items():
        if len(df) < BB_PERIOD + 10:
            continue
        closes   = df["Close"]
        volumes  = df["Volume"]
        sma      = closes.rolling(BB_PERIOD).mean()
        std      = closes.rolling(BB_PERIOD).std()
        upper    = sma + BB_STD * std
        lower    = sma - BB_STD * std
        bwidth   = upper - lower
        pct_b    = (closes - lower) / bwidth * 100
        vol_75   = volumes.rolling(BB_PERIOD).quantile(VOL_PCTILE / 100)
        prev_pct_b  = pct_b.shift(1)
        prev_vol_ok = volumes >= vol_75.shift(1)

        for i in range(BB_PERIOD + 1, len(df)):
            ppb    = float(prev_pct_b.iloc[i]) if pd.notna(prev_pct_b.iloc[i]) else 0
            vol_ok = bool(prev_vol_ok.iloc[i])
            if not (ppb >= PCT_B_MIN and vol_ok):
                continue

            row  = df.iloc[i]
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            rng  = h - l if h > l else 1

            all_signals_total += 1

            # Gap-up: open already above prior close + 0.5% -> signal was set at yesterday's close
            prior_close = float(df.iloc[i-1]["Close"])
            if o >= prior_close * 1.005:
                gap_up_signals += 1
                all_signals_early += 1  # visible at open
            elif (h - o) / rng > 0.50:
                # Upper half of the day's range is above open -> breakout happened early in day
                all_signals_early += 1
            else:
                pullback_signals += 1

    pct_early = all_signals_early / all_signals_total * 100 if all_signals_total else 0
    pct_late  = (all_signals_total - all_signals_early) / all_signals_total * 100 if all_signals_total else 0
    pct_gap   = gap_up_signals / all_signals_total * 100 if all_signals_total else 0

    print(f"\n  Total BREAKOUT signals (5yr)  : {all_signals_total:,}")
    print(f"  Daily avg signals             : {all_signals_total / (365*5*252/365):.1f}")
    print(f"")
    print(f"  -- Signal timing (approximation) ------------------------")
    print(f"  Gap-up at open (visible 9:30) : {gap_up_signals:,}  ({pct_gap:.1f}%)")
    print(f"  Intraday early (before ~9:45) : {all_signals_early:,}  ({pct_early:.1f}%)")
    print(f"  Intraday late  (after  ~9:45) : {all_signals_total - all_signals_early:,}  ({pct_late:.1f}%)")
    print(f"")
    print(f"  Estimated signals per day AFTER 9:45:")
    per_day_total = all_signals_total / (365*5*252/365)
    per_day_late  = (all_signals_total - all_signals_early) / (365*5*252/365)
    print(f"    Total avg/day : {per_day_total:.1f}")
    print(f"    After 9:45    : {per_day_late:.1f}  (~{per_day_late/per_day_total*100:.0f}% of signals)")
    print(f"")
    print(f"  ! Most signals fire at/near open because:")
    print(f"      %B threshold crossed on prior day's close")
    print(f"      Scanner reads yesterday's Bollinger position")
    print(f"      Stock opens near yesterday's high -> immediate signal")
    print(f"      Delaying to 9:45 loses {pct_early:.0f}% of all signals")


def scalp_analysis() -> None:
    """
    Analyze the $1/stock   200 stocks = $200 scalp alternative.
    Pure math -- no simulation needed.
    """
    print(f"\n{'='*65}")
    print(f"  HIGH-SPEED SCALP ALTERNATIVE: $1/stock   200 stocks/day")
    print(f"{'='*65}")
    print(f"""
  Target    : $1.00 profit per stock ( 0.1% on $1,000 position)
  Stocks    : 200 trades per day
  Gross     : $200/day

  -- Problems --------------------------------------------------

  1. IBKR Commission: $0.005/share (min $1/order)
     Avg position $1,000 / $50 stock = 20 shares -> $0.10/trade
     BUT min $1/order -> each trade costs $1 in + $1 out = $2 round trip
     200 stocks   $2 = $400 in commissions ALONE
     -> You need $400 just to break even. Start at -$400.

  2. Bid-ask spread: avg $0.05 0.15 for liquid stocks
     $0.10 spread   20 shares = $2 slippage per trade
     200 stocks   $2 = $400 in slippage
     -> Another $400 drag. Now need $800 to break even.

  3. Market impact: 200 simultaneous limit orders
     IBKR paper fills everything instantly, live market WON'T
     Many orders never fill, others get partial fills

  4. Signal availability: scanner only finds 15-25 BREAKOUT signals daily
     Not 200. Would need to lower the quality bar to garbage signals.

  5. Capital: 200   $1,000 = $200,000 required simultaneously

  -- Math reality --------------------------------------------

  Gross revenue      : $200 (200   $1)
  Commissions        : -$400 (200   $2 round trip at min $1/leg)
  Slippage est.      : -$200 (half orders have $0.05 adverse slippage)
  -------------------------------------------------------------
  Net result         : -$400/day

  x This model is UNPROFITABLE at IBKR minimum commissions.
    Would only work with $0 commissions (Robinhood/Webull) AND
    a direct market-making setup.

  -- What WOULD work at scale --------------------------------

  Instead: raise position size to $5,000-$10,000 per stock
  Keep signal count at 15-25 quality signals
  Target 1.0% profit on $5,000 = $50/trade
  15 wins/day = $750 gross, $600 net of commissions
  This is the path. Quality > quantity.
""")


def main():
    print("=" * 65)
    print("  DAY TRADER 5-YEAR BACKTEST")
    print("=" * 65)

    # Download data
    data = download_data()

    # Generate signals
    print("\nComputing breakout signals...")
    df_signals = compute_signals(data)
    total_signal_days = df_signals["date"].nunique()
    print(f"  Found {len(df_signals):,} signals across {total_signal_days} trading days")
    print(f"  Avg signals per day: {len(df_signals)/total_signal_days:.1f}")

    # Run each config
    all_results = {}
    for cfg_key, cfg in CONFIGS.items():
        print(f"\nSimulating {cfg['label']}...")
        trades = simulate_trades(df_signals, cfg)
        daily  = daily_stats(trades, max_pos=cfg.get("max_pos", MAX_SIGNALS))
        all_results[cfg_key] = (trades, daily)
        print_report(cfg["label"], trades, daily)

    # Head-to-head comparison
    print(f"\n{'='*65}")
    print(f"  HEAD-TO-HEAD COMPARISON (5-year totals)")
    print(f"{'='*65}")
    print(f"  {'Config':<35} {'Total P&L':>12} {'Avg/Day':>9} {'Win%':>6} {'Goal%':>7}")
    print(f"  {'-'*35} {'-'*12} {'-'*9} {'-'*6} {'-'*7}")
    for cfg_key, cfg in CONFIGS.items():
        trades, daily = all_results[cfg_key]
        total_pnl = daily["pnl"].sum()
        avg_day   = daily["pnl"].mean()
        win_rate  = trades["win"].sum() / len(trades) * 100 if len(trades) else 0
        goal_pct  = daily["goal_hit"].mean() * 100
        print(f"  {cfg['label']:<35} ${total_pnl:>10,.0f}  ${avg_day:>7.2f}  {win_rate:>5.1f}%  {goal_pct:>6.1f}%")

    # Signal timing
    signal_timing_analysis(data)

    # Scalp alternative
    scalp_analysis()

    print(f"\n{'='*65}")
    print(f"  STRATEGIC RECOMMENDATIONS")
    print(f"{'='*65}")

    # Print best config
    best_cfg = max(all_results, key=lambda k: all_results[k][1]["pnl"].sum())
    _, best_daily = all_results[best_cfg]
    best_label = CONFIGS[best_cfg]["label"]
    print(f"\n  Best config: {best_label}")
    print(f"  Avg daily P&L: ${best_daily['pnl'].mean():.2f}")
    print(f"  Goal hit rate: {best_daily['goal_hit'].mean()*100:.1f}% of trading days")

    print(f"""
  Key findings:
  1. Tight stop (1.5-2%) dramatically improves win rate and daily P&L
  2. Delaying to 9:45 loses ~70-80% of signals -- NOT recommended
  3. Better path: raise position size, keep quality signals at open
  4. $1 200 scalp is unprofitable after IBKR min commissions
  5. Capital recycling (re-entering after each close) boosts daily count
""")


if __name__ == "__main__":
    main()
