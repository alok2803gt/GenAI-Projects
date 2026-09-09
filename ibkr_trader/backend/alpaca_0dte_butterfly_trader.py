"""
Live/paper-adjacent 0DTE long call butterfly trader for SPY/QQQ/IWM.
Pricing AND execution are both on IBKR (migrated from Alpaca 2026-09-07:
Alpaca's real buying power dropped to $0, and 2026-09-07 CEO instruction to
drop all remaining Alpaca dependency, including the read-only CRO/CFO
capital-view call this script used to make -- see cro_cfo_capital_budget()'s
docstring). MLEG combo orders are known to reject on this account for
equity options (task 2026-08-12-003, the reason this strategy went to
Alpaca in the first place) -- the workaround, sequential single-leg orders
via ibkr_place_butterfly_sequential/ibkr_close_butterfly_sequential
(ibkr_0dte_common.py), works identically on IBKR.

VALIDATED PARAMS (see butterfly_0dte_v2_run.log / butterfly_full_comparison.csv
/ oversight_log.jsonl 2026-09-02 for the full backtest this comes from --
corrected version, after an earlier same-day-close-vs-close bug produced an
invalid 82% "win rate" that this build does NOT use):
  Real 9:45 ET entry (Unusual Whales real intraday tape) + real close exit,
  90 real trading days (2026-04-23 through today -- UW's plan lookback
  limit, not a 2-year window), SPY+QQQ+IWM combined:
    wing_step=2: n=135, 37.8% win rate, mean +$2.53/contract, pooled +5.3%
    wing_step=4: n=135, 49.6% win rate, mean +$10.73/contract, pooled +6.7%
  wing_step=4 confirmed the better width in ALL THREE tickers individually
  when combined with the near-S/R filter below (SPY +$20.54->+$35.62,
  QQQ +$5.00->+$25.29, IWM -$0.23->+$20.95) -- this cross-ticker
  replication is more trustworthy than the raw combined number alone.

  NEAR-S/R ENTRY GATE (added 2026-09-02, real finding): restricting entry
  to days where the real 9:45 ET spot is within 0.3% of the PRIOR trading
  day's real high or low roughly halves trade frequency (130 of 270 rows,
  ~48% of ticker-days) but flips the combined result from marginal to
  clearly positive: all-days mean +$6.63/win 43.7% -> near-S/R mean
  +$18.70/win 50.8%, and this holds under every exit rule tested (close,
  15:45, 15:30), not just one. Per-ticker, the effect is real in all three
  but strongest in SPY (near-S/R $28.08 vs not-near -$14.03, a $42 swing;
  QQQ and IWM swing ~$14). Real caveat: per-ticker/wing_step cells this
  thin (~13-26 rows each) are a real lead, not a settled result.

  REAL PLATFORM CONSTRAINT FOUND LIVE (2026-09-02, first real fire):
  Alpaca auto-closes 0DTE option positions at 15:45 ET on its own --
  discovered when this script's own attempted close at the (then) 15:55
  target failed on all 3 legs, because the position no longer existed by
  then. This means the backtest's "hold to the real 4pm close" edge is
  NOT ACHIEVABLE live on this platform -- the realistic, platform-bound
  number is the 15:45 exit column, not the close column. Recalibrated
  expectation, near-S/R + wing_step=4 (the only combination still worth
  running): n=65, mean +$7.98/contract, median +$5.00, win rate 50.8% --
  smaller than the theoretical ~$35 hold-to-close figure, but real,
  positive, and actually realizable. wing_step=2 no longer clears a real
  bar once the 15:45 constraint is priced in (median -$1.00, win 47.7%)
  and is DEPRECATED from live use -- wing_step=4 is now the only
  supported width. HARD_CLOSE_TIME is 15:47 (2 real minutes after
  Alpaca's own auto-close) -- the monitoring loop VERIFIES the auto-close
  happened rather than racing it, and only attempts a manual close as a
  fallback if the position is unexpectedly still open past that point.

THIS IS BRAND NEW, NEVER-LIVE-TESTED CODE running against a real account.
Default qty=1 -- do not raise without a fresh explicit sizing conversation,
per this account's standing rule that new/unvalidated logic gets a small
explicit-budget test, not full sizing.

Caveats carried over from the backtest, still true live:
  - Only ~90 real trading days, one single recent market regime (no
    volatile/down-trending stretch represented) -- not validated across
    multiple regimes the way EVC's wing-tightening gate or the weekly
    reverse butterfly were.
  - SPY/QQQ/IWM are effectively the SAME bet three times over (they move
    together on most days) -- 270 backtest rows is really ~90 independent
    trading days, not 270 independent bets.
  - Backtest priced entry/exit off real closing/intraday TRADE prints, not
    live bid/ask -- real fills will carry real slippage this didn't model,
    which matters more here than on a condor since the average edge is
    only a few dollars per contract.

MUST-CLOSE, NOT LET-EXPIRE: SPY/QQQ/IWM options are American-style,
physically-settled. A winning butterfly finishes ITM on some legs by
definition (that's what winning looks like) -- letting it expire would
trigger real share assignment/exercise (100+ shares/leg), which this
account cannot support. Every position is force-closed before HARD_CLOSE_TIME,
no exceptions, regardless of P&L.

Usage (2026-09-02 deployed config -- ticker-differentiated entry mode,
real finding: window-scan beats fixed-9:45 for QQQ/IWM but is NEGATIVE
for SPY, so SPY stays on its own validated fixed check):
  python alpaca_0dte_butterfly_trader.py --ticker spy --entry-mode fixed --fire
  python alpaca_0dte_butterfly_trader.py --ticker qqq --entry-mode window --fire
  python alpaca_0dte_butterfly_trader.py --ticker iwm --entry-mode window --fire

GEX-INFORMED BODY STRIKE (added 2026-09-04, CEO-requested): the body strike
was always just "nearest real strike to entry spot," with no data on where
dealer gamma is actually concentrated. Real motivating example: 2026-09-04's
live IWM butterfly landed body=$294 while the real dominant GEX wall sat at
$295 -- a real, avoidable mismatch the IWM babysitter's intraday insight
check only caught AFTER entry. price_legs() now runs a real, live GEX band
scan (reusing butterfly_babysitter_common.compute_live_gex, the same
methodology validated for the QQQ/IWM babysitters) around the naive pick
before finalizing k2, and snaps the body to the real wall's strike IF it
sits within MAX_GEX_NUDGE_STRIKES real strike positions -- bounded so this
can never turn the validated near-ATM design into something meaningfully
off-market. No historical intraday GEX data exists to backtest this idea
against, so it ships live unvalidated by backtest, same small qty=1 sizing
this whole strategy already runs at. Every entry's real GEX context (wall
strike, net_gex, whether/why a nudge did or didn't happen) is recorded in
the position's registry notes for a future backtest once enough real data
accumulates -- and any GEX-computation failure falls back silently to the
original nearest-to-spot pick, since this is an enhancement, not a gate.

MACRO CALENDAR GATE (added 2026-09-04, CEO-requested): real motivating
example was Ashley's 770C stumbling on the exact real morning of the
August jobs report (2026-09-04, payrolls +162k vs +53k expected -- a big
beat). Now BLOCKS the whole day (skips before even connecting to IBKR) on
any real FOMC/NFP/CPI/PPI date, via the same live macro_calendar.py module
SPX 0DTE's own skip_dates gate already used (extracted into its own module
this same day so this trader and SPX 0DTE share one real calendar instead
of each maintaining their own). Full-day block, not just a caution, for
consistency with SPX 0DTE's own already-live precedent on a directly
comparable same-day-expiry strategy: a position entered this morning is
still open through any scheduled afternoon volatility (FOMC decisions
publish 2:00pm ET), and CPI/NFP/PPI prints land before entry even happens,
so there's no partial way to "trade around" the event once the gate would
otherwise pass.
"""
import argparse
import sys
import time
from datetime import date, datetime

import yfinance as yf
from ib_insync import IB

from butterfly_babysitter_common import compute_live_gex, telegram
from macro_calendar import is_macro_day

from alpaca_0dte_common import (
    get_quotes_batch, etf_option, etf_spot, get_real_strikes_0dte,
    target_px, register_position, close_position, now_et, cro_cfo_capital_budget,
    load_config,
)
from ibkr_0dte_common import (
    ibkr_place_butterfly_sequential, ibkr_close_butterfly_sequential, ibkr_has_open_position,
    ibkr_place_leg, occ_symbol,
)

TWS_PORT = 7496
CLIENT_ID = {"spy": 1660, "qqq": 1661, "iwm": 1662}

WING_STEP_DEFAULT = 4
HARD_CLOSE_TIME = "15:47"   # Alpaca auto-closes 0DTE option positions at 15:45 ET on
                             # its own (real, confirmed 2026-09-02 -- not documented
                             # behavior this account knew about in advance, found only
                             # after this script's own 15:55 close attempt failed all
                             # 3 legs because the position no longer existed by then).
                             # 15:47 gives Alpaca's auto-close 2 real minutes to have
                             # already happened before this script checks -- it VERIFIES
                             # the auto-close rather than racing it, and only attempts a
                             # manual close if the position is unexpectedly still open.
NEAR_SR_THRESHOLD_PCT = 0.003   # 0.3% -- matches the backtest's definition exactly
QTY = 1
MONITOR_INTERVAL_S = 60

MAX_GEX_NUDGE_STRIKES = 2   # bounded: never move the body more than this many
                             # real strike positions away from the pure
                             # nearest-to-spot pick -- keeps the design near-ATM
GEX_SCAN_BAND = 6           # real strikes each side of spot to scan for a wall --
                             # smaller than the babysitters' 20 since this only
                             # needs to see whether a wall exists within nudge
                             # range, not build a full GEX picture; also keeps
                             # the one-time pre-entry delay reasonable (~35-40s)


def near_sr_gate(ticker: str, spot: float) -> tuple[bool, str]:
    """Real backtest finding (2026-09-02): entry only when the 9:45 ET spot
    is within 0.3% of the PRIOR real trading day's high or low. Same
    yfinance daily-bar source the backtest itself used, for an
    apples-to-apples definition -- not a live quote, just the prior day's
    already-settled high/low, so there's no lookahead risk here."""
    hist = yf.Ticker(ticker.upper()).history(period="5d", interval="1d", auto_adjust=False, actions=False)
    if hist is None or len(hist) < 2:
        return False, "insufficient real prior-day history"
    prior = hist.iloc[-2]  # most recent FULLY CLOSED prior day (today's own bar, if present, is excluded)
    prior_high, prior_low = float(prior["High"]), float(prior["Low"])
    dist_high_pct = abs(prior_high - spot) / spot
    dist_low_pct = abs(spot - prior_low) / spot
    near = dist_high_pct < NEAR_SR_THRESHOLD_PCT or dist_low_pct < NEAR_SR_THRESHOLD_PCT
    detail = (f"prior_high={prior_high:.2f} ({dist_high_pct:.2%} away), "
              f"prior_low={prior_low:.2f} ({dist_low_pct:.2%} away), threshold={NEAR_SR_THRESHOLD_PCT:.1%}")
    return near, detail


WINDOW_SCAN_END = (10, 0)      # 10:00 ET -- matches the backtest's scan window
WINDOW_POLL_INTERVAL_S = 30    # live poll cadence; backtest checked every real 1-min bar,
                                 # this is a faithful-enough live approximation, not an
                                 # exact replay of Polygon's minute-bar timestamps


def get_entry_spot(ib, ticker: str, entry_mode: str) -> tuple[float | None, str]:
    """Returns (triggered_spot, detail) if the near-S/R gate passes, or
    (None, detail) if it never does within this ticker's entry design.

    entry_mode="fixed": single real-time check, right now -- SPY's design,
    backtested as the fixed-9:45 near-S/R check (its own real edge: near-
    S/R + wing_step=4 + real 15:45 exit -- SPY specifically was the
    STRONGEST ticker under this fixed design, unlike QQQ/IWM).

    entry_mode="window": polls every WINDOW_POLL_INTERVAL_S seconds until
    WINDOW_SCAN_END (10:00 ET), entering the moment the gate first passes
    -- QQQ/IWM's design (2026-09-02 finding: beats the fixed check for
    these two specifically, real recalibrated numbers QQQ +$19.49/win
    54.1%, IWM +$39.18/win 73.5%, vs SPY where this same window design
    was actually NEGATIVE, -$7.36 -- this is why SPY stays on "fixed").
    """
    if entry_mode == "fixed":
        S0 = etf_spot(ib, ticker.upper())
        if not S0:
            return None, "no live spot quote -- market likely closed"
        near, detail = near_sr_gate(ticker, S0)
        print(f"{ticker.upper()} spot: ${S0:.2f} | Near-S/R gate: {'PASS' if near else 'SKIP'} -- {detail}")
        return (S0, detail) if near else (None, detail)

    if entry_mode == "window":
        window_end_dt = datetime.strptime(f"{date.today().isoformat()} {WINDOW_SCAN_END[0]:02d}:{WINDOW_SCAN_END[1]:02d}",
                                           "%Y-%m-%d %H:%M").replace(tzinfo=now_et().tzinfo)
        print(f"{ticker.upper()}: scanning for near-S/R trigger every {WINDOW_POLL_INTERVAL_S}s "
              f"until {WINDOW_SCAN_END[0]:02d}:{WINDOW_SCAN_END[1]:02d} ET...")
        last_detail = "no real quote obtained during scan window"
        while now_et() < window_end_dt:
            S0 = etf_spot(ib, ticker.upper())
            if S0:
                near, last_detail = near_sr_gate(ticker, S0)
                print(f"  [{now_et().strftime('%H:%M:%S')}] {ticker.upper()} spot=${S0:.2f} -- {last_detail}")
                if near:
                    print(f"TRIGGERED at {now_et().strftime('%H:%M:%S')} ET: {last_detail}")
                    return S0, last_detail
            time.sleep(WINDOW_POLL_INTERVAL_S)
        return None, last_detail

    raise ValueError(f"unknown entry_mode: {entry_mode}")


def gex_informed_body_strike(ib, ticker, S0, real_strikes, naive_k2_pos):
    """Real, live GEX check before finalizing the body strike -- see module
    docstring for the motivating example and the bounded-nudge design.
    Never raises: any failure (no data, IBKR error) falls back to the
    naive nearest-to-spot pick, since this is an enhancement, not a gate.
    Returns (chosen_k2_pos, detail_str) -- detail_str always describes what
    happened and why, for the registry notes / future backtest dataset.
    """
    try:
        gex = compute_live_gex(ib, ticker.upper(), S0, band=GEX_SCAN_BAND)
    except Exception as exc:
        return naive_k2_pos, f"GEX check failed ({exc}) -- using naive nearest-to-spot"
    if not gex or gex.get("wall_strike") is None:
        return naive_k2_pos, "GEX check returned no usable wall (no OI data) -- using naive nearest-to-spot"

    wall_strike = gex["wall_strike"]
    try:
        wall_pos = real_strikes.index(wall_strike)
    except ValueError:
        return naive_k2_pos, (f"GEX wall ${wall_strike} not found in real strike ladder -- "
                               f"using naive nearest-to-spot")

    dist = abs(wall_pos - naive_k2_pos)
    base = f"net_gex={gex['net_gex']:+,.0f} ({gex['regime']}), wall=${wall_strike}"
    if dist == 0:
        return naive_k2_pos, f"GEX wall already matches naive pick ${real_strikes[naive_k2_pos]} -- {base}"
    if dist <= MAX_GEX_NUDGE_STRIKES:
        detail = (f"GEX-nudged body from ${real_strikes[naive_k2_pos]} to wall ${wall_strike} "
                  f"({dist} real strike(s) away) -- {base}")
        return wall_pos, detail
    return naive_k2_pos, (f"GEX wall ${wall_strike} is {dist} real strikes away (beyond the "
                           f"{MAX_GEX_NUDGE_STRIKES}-strike bound) -- using naive nearest-to-spot -- {base}")


def price_legs(ib, ticker, wing_step, expiry_ibkr, S0):
    print(f"{ticker.upper()} entry spot: ${S0:.2f}")

    real_strikes = get_real_strikes_0dte(ib, ticker.upper(), expiry_ibkr)
    if len(real_strikes) < 2 * wing_step + 1:
        print(f"ERROR: only {len(real_strikes)} real strikes listed for {ticker.upper()} "
              f"{expiry_ibkr} -- not enough for wing_step={wing_step}. Aborting.")
        sys.exit(1)

    naive_k2 = min(real_strikes, key=lambda s: abs(s - S0))
    naive_k2_pos = real_strikes.index(naive_k2)
    k2_pos, gex_detail = gex_informed_body_strike(ib, ticker, S0, real_strikes, naive_k2_pos)
    k2 = real_strikes[k2_pos]
    print(f"GEX check: {gex_detail}")

    lo_pos, hi_pos = k2_pos - wing_step, k2_pos + wing_step
    if lo_pos < 0 or hi_pos >= len(real_strikes):
        print(f"ERROR: wing_step={wing_step} goes out of range of the real strike ladder "
              f"(k2 at position {k2_pos} of {len(real_strikes)}). Aborting.")
        sys.exit(1)
    k1, k3 = real_strikes[lo_pos], real_strikes[hi_pos]
    print(f"Target butterfly: {k1}C / {k2}C x2 / {k3}C  (wing_step={wing_step})")

    legs = {
        "wing_lo": etf_option(ticker.upper(), expiry_ibkr, k1, "C"),
        "body":    etf_option(ticker.upper(), expiry_ibkr, k2, "C"),
        "wing_hi": etf_option(ticker.upper(), expiry_ibkr, k3, "C"),
    }
    quotes = get_quotes_batch(ib, legs)
    for name, q in quotes.items():
        print(f"  {name}: bid={q['bid']} ask={q['ask']}")
    if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
        print("ERROR: missing live bid/ask on one or more legs -- aborting.")
        sys.exit(1)

    # Conservative (worst-fill) debit: buy wings at ASK, sell body at BID.
    conservative_debit = round(
        (quotes["wing_lo"]["ask"] + quotes["wing_hi"]["ask"]) - 2 * quotes["body"]["bid"], 2
    )
    entry_limits = {
        "wing_lo": target_px(quotes["wing_lo"], is_short=False),
        "wing_hi": target_px(quotes["wing_hi"], is_short=False),
        "body":    target_px(quotes["body"], is_short=True),
    }
    strikes = {"wing_lo": k1, "body": k2, "wing_hi": k3}
    return strikes, quotes, entry_limits, conservative_debit, gex_detail


# Real, backtested finding (2026-09-08, butterfly_intraday_stoploss_test.py +
# butterfly_intraday_takeprofit_test.py -- 90-day real underlying price paths,
# Black-Scholes repricing calibrated to each day's real entry debit): a flat
# stop-loss at -50% of debit paid, held-to-close otherwise, real-improves QQQ
# specifically (mean +$21.00 -> +$25.66/contract, full-max-loss rate 33.3% ->
# 2.6%, worst trade -$144 -> -$112) but is neutral-to-negative for SPY and
# actively hurts IWM (+$5.00 -> -$8.85) -- NOT a universal fix, QQQ-only by
# design. A take-profit was tested too (same infra) and rejected: any
# moderate profit target cuts into the convexity that IS this strategy's
# edge (a butterfly's value builds late in the day as decay concentrates
# near the body) -- confirms the original "no early profit target" design
# was right on the profit side; the new value is loss-side only.
STOP_LOSS_PCT_BY_TICKER = {"qqq": 0.50}


def _close_and_record(ib2, ticker, contracts, today_ibkr, strikes, close_limits,
                       pos_id, net_entry_debit, qty, reason_label):
    """Attempt a real close (retrying any unfilled leg), then record the
    REAL outcome in the registry -- never an optimistic live-mark guess.
    Shared by both the hard-close path and the stop-loss path. See the
    2026-09-08 root-cause fix note below for why this matters: the short
    body leg is the only one with real assignment risk, so it alone gets
    retried hard and blocks marking the position closed if it never
    confirms; an unfilled long wing (no bid) is safe to value at $0 and
    leave to expire."""
    ok, close_fills = ibkr_close_butterfly_sequential(ib2, contracts, close_limits, qty)

    # Root-cause fix (2026-09-08, found after 7/7 real butterfly closes --
    # SPY 1/1, IWM 3/3, QQQ 3/3 -- all needed manual reconciliation from raw
    # IBKR fill history): this used to call close_position() unconditionally
    # right here with an ESTIMATED live-mark P&L, even when a leg never
    # actually confirmed filled -- silently marking the registry "closed"
    # while a real leg was still open on the broker. Fixed by: (1) retrying
    # any unconfirmed leg a few more times with an aggressive/floor price,
    # (2) only ever recording a REAL confirmed fill or an explicit, reasoned
    # $0 (a long wing with no bid at all -- that's *why* it didn't fill, and
    # a long position that expires OTM has zero assignment risk, unlike the
    # short body), never a live-mark guess, and (3) refusing to mark the
    # position closed at all if the SHORT body leg -- the one leg with real
    # open assignment risk -- never confirms, so it stays visible as "open"
    # for the next check instead of getting silently buried.
    for retry_leg in ("body", "wing_lo", "wing_hi"):
        if close_fills.get(retry_leg) is not None:
            continue
        if not ibkr_has_open_position(ib2, contracts[retry_leg]):
            continue  # actually closed already -- raced the status check
        max_retries = 4 if retry_leg == "body" else 2
        for attempt in range(max_retries):
            ib2.sleep(3)
            _, mq = butterfly_mark(ib2, ticker, today_ibkr, strikes)
            q = (mq or {}).get(retry_leg) or {}
            if retry_leg == "body":
                action, leg_qty = "BUY", qty * 2
                px = round(q["ask"] * 1.10, 2) if q.get("ask") else 999.0
            else:
                action, leg_qty = "SELL", qty
                px = round(q["bid"], 2) if q.get("bid") else 0.01
            print(f"  [retry {attempt+1}/{max_retries}] {retry_leg}: {action} {leg_qty}x @ ${px}...")
            ok_leg, fill_px = ibkr_place_leg(ib2, contracts[retry_leg], action, px,
                                              f"retry {retry_leg}", leg_qty)
            if ok_leg:
                close_fills[retry_leg] = fill_px
                break
        else:
            if not ibkr_has_open_position(ib2, contracts[retry_leg]):
                continue
            if retry_leg == "body":
                print(f"  *** {retry_leg} (SHORT 2x) still open after {max_retries} retries -- "
                      f"REAL open assignment risk on {ticker}. ***")
            else:
                print(f"  {retry_leg} (long wing) left open -- no bid at any price tried, "
                      f"treating as worthless/expiring OTM (no assignment risk on a long).")

    body_open = ibkr_has_open_position(ib2, contracts["body"])
    ok = all(v is not None for v in close_fills.values())

    if ok:
        close_value = close_fills["wing_lo"] + close_fills["wing_hi"] - 2 * close_fills["body"]
        final_pnl = (close_value - net_entry_debit) * qty * 100
        close_position(pos_id, reason_label, final_pnl)
        print(f"Closed. fills={close_fills} final_pnl={final_pnl}")
        return True
    elif not body_open:
        wing_lo_val = close_fills.get("wing_lo") if close_fills.get("wing_lo") is not None else 0.0
        wing_hi_val = close_fills.get("wing_hi") if close_fills.get("wing_hi") is not None else 0.0
        close_value = wing_lo_val + wing_hi_val - 2 * close_fills["body"]
        final_pnl = (close_value - net_entry_debit) * qty * 100
        open_legs = [k for k, v in close_fills.items() if v is None]
        close_position(
            pos_id,
            f"{reason_label}_wing_left_open_worthless legs_open={open_legs} "
            f"(long wing(s), no bid at close -- safe to expire OTM, no assignment risk)",
            final_pnl,
        )
        print(f"Closed (body confirmed, wing(s) {open_legs} left to expire worthless). "
              f"fills={close_fills} final_pnl={final_pnl}")
        return True
    else:
        print(f"*** BODY (SHORT 2x) STILL OPEN on {ticker} after {reason_label} + retries -- "
              f"NOT marking position closed. Real open assignment risk. "
              f"fills so far: {close_fills}. Manual intervention required NOW. ***")
        try:
            cfg = load_config()
            telegram(
                cfg,
                f"{ticker.upper()} 0DTE butterfly: SHORT body leg (2x) still open after "
                f"{reason_label} + retries. Real assignment risk -- check IBKR NOW. "
                f"pos_id={pos_id} fills_so_far={close_fills}",
                high_priority=True,
            )
        except Exception as exc:
            print(f"  (telegram alert also failed: {exc})")
        return False


def butterfly_mark(ib, ticker, expiry_ibkr, strikes):
    """Current mark-to-market value of the LONG butterfly (what it's worth if closed now)."""
    legs = {
        "wing_lo": etf_option(ticker.upper(), expiry_ibkr, strikes["wing_lo"], "C"),
        "body":    etf_option(ticker.upper(), expiry_ibkr, strikes["body"], "C"),
        "wing_hi": etf_option(ticker.upper(), expiry_ibkr, strikes["wing_hi"], "C"),
    }
    quotes = get_quotes_batch(ib, legs)
    if any(not q["mid"] for q in quotes.values()):
        return None, quotes
    value = quotes["wing_lo"]["mid"] + quotes["wing_hi"]["mid"] - 2 * quotes["body"]["mid"]
    return value, quotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy", "qqq", "iwm"])
    ap.add_argument("--entry-mode", choices=["fixed", "window"], default="fixed",
                     help="'fixed': single near-S/R check right now (SPY's validated design). "
                          "'window': poll every 30s for a near-S/R trigger until 10:00 ET "
                          "(QQQ/IWM's validated design, 2026-09-02 -- beats 'fixed' for these "
                          "two specifically; was NEGATIVE for SPY, which is why SPY stays on 'fixed').")
    ap.add_argument("--wing-step", type=int, default=WING_STEP_DEFAULT)
    ap.add_argument("--fire", action="store_true", help="place real orders; omit for dry-run pricing only")
    ap.add_argument("--qty", type=int, default=QTY)
    ap.add_argument("--max-risk-override", type=float, default=None,
                     help="explicit per-strategy $ cap for THIS run only, overriding the default "
                          "5%%-of-combined-net-liq CRO/CFO cap. One-time CEO decision, not a config "
                          "change -- e.g. 2026-09-02: real backtested debits for wing_step=4 exceeded "
                          "the default ~$140 cap on 73%% of days, so the default cap would have "
                          "rejected most of the very trades this strategy was validated on.")
    args = ap.parse_args()
    ticker = args.ticker

    today_ibkr = date.today().strftime("%Y%m%d")
    today_alp = date.today().strftime("%Y-%m-%d")

    print(f"=== {ticker.upper()} 0DTE long butterfly -- {'FIRE' if args.fire else 'DRY RUN'} -- "
          f"expiry {today_alp} -- wing_step={args.wing_step} -- entry_mode={args.entry_mode} ===")

    # ── Macro calendar gate (added 2026-09-04, CEO-requested) ─────────────────
    # Same real, live calendar SPX 0DTE already uses (FOMC/NFP/CPI/PPI, real
    # federalreserve.gov/bls.gov dates, hardcoded fallback) -- BLOCKED here,
    # not just cautioned, for consistency with that existing precedent: a 0DTE
    # position entered this morning is still open through any scheduled
    # afternoon volatility, and CPI/NFP/PPI prints land before entry even
    # happens, so there's no way to "trade around" the event once the gate
    # would otherwise pass. Checked before connecting to IBKR at all, to avoid
    # wasting a connection + the window-scan wait on a day that's skipping.
    macro_skip, macro_reason = is_macro_day()
    if macro_skip:
        print(f"\nNO TRADE TODAY -- {today_alp} is a real {macro_reason} day. "
              f"Skipping entirely (macro calendar gate).")
        sys.exit(0)

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID[ticker], timeout=20)
    print("Connected to IBKR.")

    S0, sr_detail = get_entry_spot(ib, ticker, args.entry_mode)
    if S0 is None:
        print(f"\nNO TRADE TODAY -- near-S/R gate never passed ({args.entry_mode} mode). "
              f"Last check: {sr_detail}. This is the validated entry filter working as "
              f"designed, not an error.")
        ib.disconnect()
        sys.exit(0)
    strikes, quotes, entry_limits, conservative_debit, gex_detail = price_legs(ib, ticker, args.wing_step, today_ibkr, S0)
    max_risk = round(conservative_debit * 100, 2)
    print(f"conservative debit (worst-fill): ${conservative_debit:.2f}  (=${max_risk:.0f}/contract)")
    if conservative_debit <= 0:
        print("ERROR: non-positive conservative debit -- aborting.")
        ib.disconnect()
        sys.exit(1)

    if not args.fire:
        print("\nDRY RUN -- no orders placed. Re-run with --fire to place live orders.")
        ib.disconnect()
        return

    # ── CRO/CFO pretrade gate: real capital check before risking anything ──
    # Reuses the same connected `ib` -- migrated from a separate ib_budget
    # connection 2026-09-07 now that `ib` stays connected through order
    # placement anyway (previously disconnected right after pricing, since
    # execution went to Alpaca and didn't need this connection). IBKR-only
    # now (2026-09-07): this strategy no longer touches Alpaca at all, not
    # even read-only -- cro_cfo_capital_budget's `client` param is optional
    # for exactly this case (see its own docstring).
    budget = cro_cfo_capital_budget(ib)
    effective_cap = budget["per_strategy_cap"]
    if args.max_risk_override is not None:
        print(f"\nCEO OVERRIDE: per-strategy cap set to ${args.max_risk_override:.2f} for this run "
              f"only (default 5%% cap would have been ${budget['per_strategy_cap']:.2f}).")
        effective_cap = args.max_risk_override
    print(f"\nCRO/CFO capital budget: combined_net_liq=${budget['combined_net_liq']:.2f}  "
          f"headroom=${budget['headroom']:.2f}  effective_cap=${effective_cap:.2f}")
    trade_risk = max_risk * args.qty
    if trade_risk > effective_cap:
        print(f"REJECTED: trade risk ${trade_risk:.2f} exceeds effective cap "
              f"${effective_cap:.2f}. Aborting -- reduce qty or wing_step.")
        ib.disconnect()
        sys.exit(1)
    if trade_risk > budget["headroom"]:
        print(f"REJECTED: trade risk ${trade_risk:.2f} exceeds remaining portfolio headroom "
              f"${budget['headroom']:.2f} (this check is NOT overridden by --max-risk-override -- "
              f"headroom reflects the account's real total 50%% risk budget across every strategy, "
              f"not just this one). Aborting.")
        ib.disconnect()
        sys.exit(1)
    print("CRO/CFO gate: PASSED.")

    contracts = {
        "wing_lo": etf_option(ticker.upper(), today_ibkr, strikes["wing_lo"], "C"),
        "body":    etf_option(ticker.upper(), today_ibkr, strikes["body"], "C"),
        "wing_hi": etf_option(ticker.upper(), today_ibkr, strikes["wing_hi"], "C"),
    }
    ib.qualifyContracts(*contracts.values())
    syms = {k: occ_symbol(ticker.upper(), strikes[k], "C", date.today().strftime("%y%m%d")) for k in contracts}
    print(f"IBKR contracts: { {k: c.localSymbol for k, c in contracts.items()} }")

    ok, fills, order_state = ibkr_place_butterfly_sequential(ib, contracts, entry_limits, args.qty)
    if not ok:
        print(f"\nENTRY INCOMPLETE ({order_state}) -- check IBKR positions manually before doing anything else.")
        ib.disconnect()
        sys.exit(1)
    ib.disconnect()
    print("Disconnected from IBKR (entry done).")

    net_entry_debit = round(fills["wing_lo"] + fills["wing_hi"] - 2 * fills["body"], 2)
    entry_time = now_et()
    pos_id = f"{ticker.upper()}_0dte_butterfly_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    register_position(
        pos_id, ticker.upper(), "long_butterfly_0dte",
        [{"leg": k, "strike": strikes[k], "symbol": syms[k], "fill": fills[k],
          "qty": args.qty * (2 if k == "body" else 1)} for k in syms],
        -net_entry_debit, args.qty, max_risk * args.qty, None, HARD_CLOSE_TIME,
        entry_time.isoformat(),
        notes=f"2026-09-02 deployed config. wing_step={args.wing_step}, entry_mode={args.entry_mode} "
              f"(entry_spot=${S0:.2f}, gate: {sr_detail}), no early profit target (MVP). IBKR execution "
              f"(migrated from Alpaca 2026-09-07) has no broker-side auto-close -- this script attempts "
              f"a real close itself at {HARD_CLOSE_TIME} ET; physically-settled ETF options, cannot let expire. "
              f"GEX check (added 2026-09-04): {gex_detail}",
    )
    print(f"\nENTERED. pos_id={pos_id}  net_entry_debit=${net_entry_debit:.2f} "
          f"(${net_entry_debit*args.qty*100:.0f} total paid)")

    # ── Monitor until hard close -- MUST close, cannot let expire (assignment risk) ──
    stop_loss_pct = STOP_LOSS_PCT_BY_TICKER.get(ticker)
    stop_loss_note = f", stop-loss at -{int(stop_loss_pct*100)}% of debit (QQQ-validated only)" if stop_loss_pct else ""
    print(f"\nMonitoring every {MONITOR_INTERVAL_S}s until {HARD_CLOSE_TIME} forced close{stop_loss_note}...")
    hard_close_dt = datetime.strptime(f"{entry_time.strftime('%Y-%m-%d')} {HARD_CLOSE_TIME}", "%Y-%m-%d %H:%M")
    hard_close_dt = hard_close_dt.replace(tzinfo=entry_time.tzinfo)

    ib2 = IB()
    ib2.errorEvent += lambda reqId, code, msg, contract: None
    ib2.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID[ticker] + 100, timeout=20)
    try:
        while True:
            now = now_et()
            value, mon_quotes = butterfly_mark(ib2, ticker, today_ibkr, strikes)
            if value is None:
                print(f"[{now.strftime('%H:%M:%S')}] quote gap, will retry next cycle")
            else:
                live_pnl = (value - net_entry_debit) * args.qty * 100
                print(f"[{now.strftime('%H:%M:%S')}] butterfly_value=${value:.2f}  live_pnl=${live_pnl:+.2f}")
                loss_frac = (net_entry_debit - value) / net_entry_debit if net_entry_debit > 0 else 0.0
                if stop_loss_pct and loss_frac >= stop_loss_pct and now < hard_close_dt:
                    print(f"\nSTOP-LOSS: down {loss_frac*100:.0f}% of debit (threshold "
                          f"{stop_loss_pct*100:.0f}%) -- exiting now instead of holding to "
                          f"{HARD_CLOSE_TIME} (real backtested finding, QQQ-only, see "
                          f"STOP_LOSS_PCT_BY_TICKER comment).")
                    if mon_quotes and all(q["bid"] and q["ask"] for q in mon_quotes.values()):
                        close_limits = {
                            "wing_lo": target_px(mon_quotes["wing_lo"], is_short=True),
                            "wing_hi": target_px(mon_quotes["wing_hi"], is_short=True),
                            "body":    target_px(mon_quotes["body"], is_short=False),
                        }
                    else:
                        close_limits = {"wing_lo": 0.01, "wing_hi": 0.01, "body": 999.0}
                    _close_and_record(ib2, ticker, contracts, today_ibkr, strikes, close_limits,
                                       pos_id, net_entry_debit, args.qty, "stop_loss_50pct")
                    return
            if now >= hard_close_dt:
                print(f"\n{HARD_CLOSE_TIME} hard-close time reached -- IBKR has no broker-side "
                      f"auto-close for single-leg equity options (unlike Alpaca), so this attempts "
                      f"a real close directly.")
                still_open = any(ibkr_has_open_position(ib2, contracts[k]) for k in contracts)
                if not still_open:
                    print("Already flat (closed by some earlier action) -- no manual close needed.")
                    close_position(pos_id, "already_flat_at_hard_close", None)
                    return
                _, mon_quotes = butterfly_mark(ib2, ticker, today_ibkr, strikes)
                if mon_quotes and all(q["bid"] and q["ask"] for q in mon_quotes.values()):
                    close_limits = {
                        "wing_lo": target_px(mon_quotes["wing_lo"], is_short=True),
                        "wing_hi": target_px(mon_quotes["wing_hi"], is_short=True),
                        "body":    target_px(mon_quotes["body"], is_short=False),
                    }
                else:
                    print("WARNING: no live quotes for close -- using aggressive fallback (bid=0.01 floor).")
                    close_limits = {"wing_lo": 0.01, "wing_hi": 0.01, "body": 999.0}
                _close_and_record(ib2, ticker, contracts, today_ibkr, strikes, close_limits,
                                   pos_id, net_entry_debit, args.qty, "hard_close")
                return
            time.sleep(MONITOR_INTERVAL_S)
    finally:
        ib2.disconnect()


if __name__ == "__main__":
    main()
