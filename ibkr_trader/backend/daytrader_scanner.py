#!/usr/bin/env python3
"""
Day Trader Scanner — independent signal source for the Day Trader auto-trader.

Built entirely from a 500-ticker, 5-year, 609,859-ticker-day S&P 500 study
(sp500_daytrade_study.py, run 2026-08-06) of what actually predicts a
same-day open->close move >= 0.5% -- Day Trader's real profit-target scale
(live config: profit_target_pct=0.25%, hard_stop_pct=1.0%, force-close at
15:45 ET; see main.py's "day_trader" default config).

Deliberately NOT the breakout_scanner.py pipeline. That scanner's %B/RSI/
SMA/ADX "quality setup" (F1-F10) was built and validated for MULTI-DAY swing
continuation (5d/10d/20d forward returns) and scored near-zero-to-negative
lift on this specific same-day target (pctB>75 & RSI>60 & above SMAs:
-1.75pt vs base rate; high-ADX trending & pctB>75: -2.36pt). This scanner
instead ranks candidates on what the study actually found predictive of a
same-day mover, in descending order of RandomForest importance:

  atr_pct   (43.6% importance) -- the ticker's OWN persistent volatility
            (ATR14 as % of price, known BEFORE today's open). By far the
            dominant factor. Decile lift ranges from -7.9pt (bottom decile,
            ATR%<1.6) to +7.7pt (top decile, ATR%>4.2) vs the 35.9% base
            rate.
  gap_pct   (17.3%) -- today's opening gap vs yesterday's close. IMPORTANT:
            both directions showed positive lift on the *odds* of a >=0.5%
            day, but a big gap DOWN (<-1%) had a HIGHER average realized
            same-day return (+0.13%) than a big gap UP (>1%, avg -0.03%) --
            gap-down setups are effectively a same-day dip-buy / mean-
            reversion bet, not momentum continuation, and empirically the
            single best setup tested. This scanner surfaces BOTH and tags
            which thesis each candidate represents; it does not filter
            either direction out, since both were empirically positive.
            Dip-buy entries carry real "catching a falling knife" risk that
            a single-day backtest average cannot capture -- Day Trader's
            existing 1% hard stop is the safety net, not this scanner.
  prior_day_ret_pct (8.6%) -- magnitude of yesterday's own open->close move.
  ret5d_prior        (6.5%) -- 5-day momentum entering today.

TWO-STAGE PIPELINE (rebuilt 2026-09-05, real backtest evidence -- see
daytrader_research/scan_delay_backtest.py): a real intraday delay-
sensitivity backtest found the confirmation gate's edge decays almost 10x
between a 0min and a 7min watch-start delay, and the old design (a single
full-483-ticker scan at 9:35 ET) measured out to ~7min of real delay live
(2026-09-03 log: scan didn't finish dispatching until ~9:36:50). Now:
  1. PREMARKET_SCAN_TIME_ET (8:00 ET): scores the full universe on the
     75%-weight inputs that don't need today's session at all (atr_pct,
     prior_day_ret_pct, ret5d_prior), caches the top SHORTLIST_SIZE as
     today's shortlist. gap_pct (25% weight) is NOT estimated pre-market --
     the CEO explicitly didn't want to trust IBKR extended-hours quotes for
     thinner names, so it stays real, validated at the actual open.
  2. SCAN_TIME_ET (9:30 ET + a small real-open-print buffer): a FAST
     finalize pass -- real IBKR daily bars for just the shortlist (not all
     ~483 tickers), computes the real gap_pct + full composite_score, then
     dispatches the top 20 in PARALLEL (submit_candidates_parallel) instead
     of the old sequential one-at-a-time loop. Falls back to the original
     full-universe run_daily_scan() if the shortlist is missing/stale.

Unlike breakout_scanner's continuous intraday polling loop, none of the
features above are meaningfully re-computable mid-day -- gap_pct in
particular is only ever defined at the open -- so a single daily batch
scan matches exactly what was backtested, rather than pretending to
support an intraday re-evaluation cadence that was never validated.

Submits its top-ranked candidates to the backend's EXISTING /day-trader/
signal endpoint (same request schema breakout_scanner.py already uses:
ticker/price/alert_fired_at/composite_score) -- Day Trader's own entry gates
(use_entry_filters, use_vol_filter/min_atr_pct, min_composite_score,
capacity, signal freshness) still apply as the final safety net on entry;
this scanner is a candidate SOURCE, not a bypass of those gates.
"""
import json
import logging
import logging.handlers
import os
import socket as _socket
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# Real bug found 2026-09-01, actually inside day_trader_agent.py (fixed
# there too): its dt_log() uses raw print(), which crashed with
# UnicodeEncodeError on a sigma character once stdout was redirected to a
# file without explicit encoding -- killing all 20 of that day's candidate
# submissions with a bare, undiagnosed HTTP 500. This script uses Python's
# logging module instead, which catches emit-time encoding errors rather
# than propagating them (a lost log line, not a crash) -- so it wasn't
# actually at risk of the SAME failure mode, but this stream reconfigure is
# a cheap, harmless defensive match in case a raw print() is ever added
# here later (e.g. the emoji already used in Telegram message strings).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yfinance as yf

EARNINGS_BLACKOUT_DAYS = 2

# ── Earnings blackout cache (mirrors breakout_scanner.py's own, independently --
# this is a separate process, can't share main.py's or breakout_scanner's cache) ──
_earnings_cache: dict[str, tuple[float, "int | None"]] = {}
EARNINGS_CACHE_TTL_S = 6 * 3600


def earnings_days_out(ticker: str) -> "int | None":
    """Days until next earnings, or None if none found within 60 days.
    Never raises -- a data hiccup should never silently suppress every candidate."""
    now = time.time()
    cached = _earnings_cache.get(ticker)
    if cached and (now - cached[0]) < EARNINGS_CACHE_TTL_S:
        return cached[1]
    days: "int | None" = None
    try:
        import pandas as pd
        cal = yf.Ticker(ticker).calendar
        if cal:
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date", [])
                raw_dt = pd.to_datetime(raw[0]) if raw else None
            elif hasattr(cal, "loc"):
                raw_dt = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
            else:
                raw_dt = None
            if raw_dt is not None:
                today = datetime.now(ZoneInfo("America/New_York")).date()
                d = raw_dt.date()
                delta = (d - today).days
                days = delta if 0 <= delta <= 60 else None
    except Exception:
        days = None
    _earnings_cache[ticker] = (now, days)
    return days

import requests

# ── Paths / logging ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SCRIPT_DIR, "daytrader_scanner_state.json")
SHORTLIST_PATH = os.path.join(SCRIPT_DIR, "daytrader_premarket_shortlist.json")

_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(SCRIPT_DIR, "daytrader_scanner.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
log = logging.getLogger("daytrader_scanner")

ET = ZoneInfo("America/New_York")

# Shared runtime config (telegram creds + backend_url) -- reuses the same
# config file breakout_scanner.py reads. This is general project config, not
# breakout-scanner-exclusive (it also carries pushover/flex/anthropic keys),
# so reusing it avoids duplicating secrets into a second file.
SHARED_CONFIG_PATH = os.path.join(SCRIPT_DIR, "scanner_config.json")

# Day Trader runs as its own standalone process since 2026-08-27
# (day_trader_agent.py) -- candidates go directly here, NOT to main.py's
# backend_url, so a main.py crash/restart can never drop an entry signal
# again (real incident: main.py restarted 5x between 8:48-9:37 AM that
# morning, and CRWD -- that day's #1-ranked candidate, score 95.5, real
# +10.08% earnings gap -- hit a connection error submitting to the old
# /day-trader/signal on main.py at the exact moment it was mid-restart).
DAY_TRADER_AGENT_URL = "http://localhost:8010"

# ── Universe: full S&P 500, fetched live from Wikipedia + cached locally ────
# Extended 2026-08-06 from an initial 112-ticker curated list to the FULL
# ~503-name index, matching sp500_daytrade_study.py's actual backtested
# universe exactly (the study is the only evidence behind this scanner's
# scoring, so the live universe should match what was validated, not a
# liquidity-filtered subset of it). GICS sector comes straight from the same
# Wikipedia table (used for the sector cap below) instead of a hand-maintained
# dict, since a 500-name map would drift out of date immediately otherwise.
# Cached to disk (refreshed weekly) so a normal day's run has zero dependency
# on Wikipedia being reachable.
UNIVERSE_CACHE_PATH = os.path.join(SCRIPT_DIR, "sp500_universe_cache.json")
UNIVERSE_CACHE_MAX_AGE_DAYS = 7

# Fallback if Wikipedia is unreachable AND no cache exists yet -- the original
# 112-ticker curated list (liquid, active-options names), so the scanner can
# still run rather than hard-failing on a cold start with no network.
_FALLBACK_TICKERS: list[str] = sorted(set([
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    "DIS", "CMCSA", "VZ", "T",
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "XYZ", "IBM",
    "RBLX", "RCL", "ABNB",
]))
_FALLBACK_SECTOR_MAP: dict[str, str] = {t: "?" for t in _FALLBACK_TICKERS}


def fetch_sp500_universe() -> tuple[list[str], dict[str, str]]:
    """Scrape the current S&P 500 constituent + GICS-sector table from
    Wikipedia. Same technique as sp500_daytrade_study.py's get_sp500_tickers,
    extended to also keep the sector column."""
    import io
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                      headers=headers, timeout=20)
    r.raise_for_status()
    import pandas as pd  # local import -- only this function needs it
    tables = pd.read_html(io.StringIO(r.text))
    df = tables[0]
    tickers = [s.replace(".", "-") for s in df["Symbol"].tolist()]
    sectors = dict(zip(tickers, df["GICS Sector"].tolist()))
    return sorted(set(tickers)), sectors


def load_universe() -> tuple[list[str], dict[str, str]]:
    """Cached S&P 500 universe -- refetches from Wikipedia if the cache is
    missing or older than UNIVERSE_CACHE_MAX_AGE_DAYS, otherwise reads local
    JSON (no network dependency on a normal day's run)."""
    cached = None
    try:
        with open(UNIVERSE_CACHE_PATH) as f:
            cached = json.load(f)
        fetched = datetime.fromisoformat(cached["fetched_at"])
        age_days = (datetime.now() - fetched).days
        if age_days <= UNIVERSE_CACHE_MAX_AGE_DAYS:
            return cached["tickers"], cached["sectors"]
        log.info("Universe cache is %d days old (>%d) -- refreshing from Wikipedia.",
                  age_days, UNIVERSE_CACHE_MAX_AGE_DAYS)
    except Exception:
        log.info("No usable universe cache -- fetching S&P 500 list from Wikipedia.")

    try:
        tickers, sectors = fetch_sp500_universe()
        with open(UNIVERSE_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now().isoformat(),
                       "tickers": tickers, "sectors": sectors}, f, indent=2)
        log.info("Fetched + cached %d S&P 500 tickers from Wikipedia.", len(tickers))
        return tickers, sectors
    except Exception as exc:
        if cached:
            log.warning("Wikipedia refresh failed (%s) -- using stale cache (%d tickers).",
                        exc, len(cached["tickers"]))
            return cached["tickers"], cached["sectors"]
        log.warning("Wikipedia fetch failed (%s) and no cache exists -- "
                    "falling back to the %d-ticker curated list.", exc, len(_FALLBACK_TICKERS))
        return _FALLBACK_TICKERS, _FALLBACK_SECTOR_MAP


# ── Tunables ──────────────────────────────────────────────────────────────────
# Entry timing (9:30 vs 9:40 vs something between): the daily-bar backtest
# CANNOT distinguish these -- it only ever measured the official session
# Open, not intraday minute-level timing, so no version of "9:30 beat 9:40 by
# X%" is something the study actually supports. Split the difference at 9:35
# instead of guessing 9:30 or 9:40 outright: entering closer to 9:30 more
# faithfully replicates the true Open price gap_pct/features are computed
# against (waiting until 9:40 means entering 10 minutes of drift away from
# the price the backtest actually modeled), but 9:30:00 sharp risks slower-
# opening names not having a confirmed print yet and sits inside the worst of
# the opening-auction spread widening -- which is also why Day Trader's own
# existing limit-buffer logic already treats the WHOLE 9:30-9:45 window as
# elevated-spread and widens its buffer accordingly (see day_trader_signal's
# buf_pct). 9:35 keeps this scanner inside that same already-assumed window
# while giving virtually every name in the universe time to print a real
# open. If a precise, statistically grounded answer matters, that requires a
# separate intraday (minute-bar) backtest -- not something this daily-bar
# study can produce.
# RETIRED 2026-09-05 (CEO request, real backtest evidence): a real intraday
# (1-minute-bar) delay-sensitivity backtest -- the exact study this comment
# above said would be needed for a statistically grounded answer -- found
# the confirmation gate's edge decays almost 10x between a 0-minute and a
# 7-minute watch-start delay (win rate 52.7%->41.0%, avg return
# +0.154%->+0.017%/trade, real 820-candidate sample). The real 2026-09-03
# scanner log showed the actual live pipeline doesn't finish dispatching
# watch signals until ~9:36:50 -- a ~7min delay from the true 9:30 open,
# right in the range the backtest shows costs almost the entire edge. See
# daytrader_research/scan_delay_backtest.py + RESEARCH_LOG.md.
#
# Replaced with a two-stage pipeline: PREMARKET_SCAN_TIME_ET builds a
# shortlist overnight using only the 75%-score-weight inputs that don't
# need today's session at all (atr_pct/prior_day_ret_pct/ret5d_prior --
# see compute_partial_score), then SCAN_TIME_ET does a FAST at-open
# finalize pass -- real IBKR quotes for just the shortlist (not all ~483
# tickers), computing the real gap_pct from the REAL session open (CEO
# explicitly did not want to trust IBKR pre-market/extended-hours quotes
# for thinner names, so gap_pct is still validated live at the open, never
# estimated pre-market) -- then dispatches in PARALLEL instead of the old
# sequential loop. If the overnight shortlist is missing or stale (build
# failed, e.g. IBKR unreachable at 8am), SCAN_TIME_ET transparently falls
# back to the original full-483-ticker run_daily_scan() -- a slower day
# is fine, a silently-skipped day is not.
PREMARKET_SCAN_TIME_ET = (8, 0)    # overnight shortlist build -- no live market data needed, ATR%/
                                     # prior-day-return/5d-momentum only need data through yesterday's close
SHORTLIST_SIZE         = 80         # generous margin above what sector-cap will actually let through --
                                     # gap_pct (25% of the full score) can still reorder things once the
                                     # real open prints, so the shortlist needs real headroom
SCAN_TIME_ET       = (9, 30)   # run once daily at this ET hour:minute -- moved from 9:35 to 9:30 as part
                                 # of the same 2026-09-05 fix; the main loop adds a small real-open-print
                                 # buffer (see FINALIZE_BUFFER_S) rather than baking a 5min wait into this
                                 # constant the way the old design did
FINALIZE_BUFFER_S  = 10         # real seconds after 9:30:00 before finalize_from_shortlist actually
                                 # fires -- IBKR's daily bar for "today" needs the real opening print to
                                 # exist first; kept small and explicit rather than folded into a much
                                 # larger SCAN_TIME_ET the way 9:35 used to hide it
HIST_DAYS          = 40        # daily bars fetched per ticker (mirrors _fetch_entry_metrics)
HISTORY_TIMEOUT_S  = 240        # /market/history/bulk over ~500 tickers, paced 40/batch + 2s -- allow ample time
# Raised from 20 to effectively "no cap" 2026-09-05 (CEO request, "push all
# tickers instead of 20 in parallel"): the real ceiling on how many
# candidates can even reach this point is MAX_PER_SECTOR (3) x 11 GICS
# sectors = 33 after the sector cap below, so 100 means every sector-cap/
# earnings-cleared candidate gets submitted, not a further truncation.
# Submitting more no longer risks an IBKR pacing cascade on the RECEIVING
# side -- day_trader_agent.py now bounds its own concurrent historical-data
# fetches via MAX_CONCURRENT_HISTORY_FETCHES regardless of burst size.
# Position count is still governed by the backend's own real gates
# (max_positions, min_composite_score, capacity), same as always.
MAX_CANDIDATES_SUB = 100
MAX_PER_SECTOR     = 3          # cap concentrated sector bets in one day's submission batch (11 GICS sectors)
MIN_ATR_PCT_FLOOR  = 1.5        # hard floor -- never submit anything below this regardless of score
                                 # (well under the backend's own min_atr_pct=2.5 default DT-VOL gate,
                                 # which is the actual enforcement point; this is just a pre-filter
                                 # so obviously-dead candidates don't even get scored/submitted)


def load_shared_config() -> dict:
    try:
        with open(SHARED_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read %s: %s -- using defaults", SHARED_CONFIG_PATH, exc)
        return {}


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.ok
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


# ── State (avoid double-scan on same-day restart) ───────────────────────────

def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st: dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(st, f, indent=2)
    except Exception as exc:
        log.warning("State save failed: %s", exc)


# ── Data fetch + feature computation ─────────────────────────────────────────

DAYTRADER_SCANNER_IBKR_CLIENT_ID = 1580
DAYTRADER_SCANNER_TWS_PORT = 7496


def fetch_universe_history(tickers: list[str], backend_url: str, days: int = HIST_DAYS) -> dict[str, list]:
    """Direct IBKR daily bars, useRTH=True -- own connection, no dependency
    on main.py (2026-09-02: this was the last real coupling between this
    scanner and the main backend; day_trader_agent.py itself already had
    none. Replaces the earlier POST /market/history/bulk proxy call, which
    meant a main.py restart could stall or abort a scan -- already
    mitigated with retry-without-marking-scanned logic, but the CEO wants
    scanning/watching/trading/monitoring fully independent of main.py,
    which now just reports trades/PnL.

    `backend_url` param kept (unused) for call-site compatibility --
    removing it would touch every caller for no behavioral benefit.

    Same batching/pacing/format as main.py's old market_history_bulk:
    40-concurrent requests per batch, 2s pause between batches, daily bars,
    durationStr sized the same way, same trim-to-`days` and dict shape.
    During market hours the LAST bar is today's still-forming session (its
    Open is the real opening print, which is exactly the feature this
    scanner needs); every earlier bar is a completed prior session.
    """
    import asyncio
    from ib_insync import IB, Stock

    needed_bars = max(days * 2, 10)
    if needed_bars <= 365:
        duration_str = f"{needed_bars} D"
    else:
        # IBKR hard-rejects "D"-unit historical data requests >365 -- confirmed
        # live 2026-09-06: error 321 "durations longer than 365 days must be
        # made in years", silently swallowed to an empty bar list by this
        # function's errorEvent suppression below (which is exactly why the
        # 2026-09-01 diagnosis saw an unexplained 0-ticker failure instead of
        # this specific, actionable message). "Y" unit has no such ceiling --
        # live-tested single-ticker ("2 Y" -> 502 real daily bars, 0 errors)
        # AND at real bulk scale (40 concurrent tickers, "2 Y", 3.6s, 0
        # errors). +1 year buffer since "Y" doesn't return an exact bar count
        # the way "D" does.
        duration_str = f"{-(-needed_bars // 252) + 1} Y"

    async def _fetch_all():
        ib = IB()
        ib.errorEvent += lambda reqId, code, msg, contract: None
        await ib.connectAsync("127.0.0.1", DAYTRADER_SCANNER_TWS_PORT,
                               clientId=DAYTRADER_SCANNER_IBKR_CLIENT_ID, timeout=20)

        async def _one(ticker: str) -> tuple[str, list]:
            try:
                contract = Stock(ticker, "SMART", "USD")
                bars = await asyncio.wait_for(
                    ib.reqHistoricalDataAsync(
                        contract, endDateTime="", durationStr=duration_str,
                        barSizeSetting="1 day", whatToShow="TRADES",
                        useRTH=True, keepUpToDate=False,
                    ),
                    timeout=15,
                )
                if not bars:
                    return ticker, []
                return ticker, [
                    {"date": str(b.date), "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": b.volume}
                    for b in bars[-days:]
                ]
            except Exception:
                return ticker, []

        results: dict[str, list] = {}
        batch_size = 40
        try:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                batch_results = await asyncio.gather(*[_one(tk) for tk in batch])
                for tk, bars in batch_results:
                    if bars:
                        results[tk] = bars
                if i + batch_size < len(tickers):
                    await asyncio.sleep(2)
        finally:
            ib.disconnect()
        return results

    try:
        data = asyncio.run(_fetch_all())
    except Exception as exc:
        log.error("direct IBKR history fetch failed: %s", exc)
        return {}
    log.info("history (direct IBKR): %d/%d tickers returned", len(data), len(tickers))
    return data


def compute_dt_features(ticker: str, bars: list[dict]) -> dict | None:
    """No-look-ahead feature set matching sp500_daytrade_study.py exactly.

    bars is chronological (oldest->newest); bars[-1] is TODAY (in progress
    during market hours -- its Open is live, High/Low/Close are still
    forming), bars[-2] is YESTERDAY (complete).
    """
    if len(bars) < 22:   # need >=21 for ATR14 (needs a t-1 close) + ret5d/ret20d lookback
        return None

    closes = [b["close"] for b in bars]
    highs  = [b["high"]  for b in bars]
    lows   = [b["low"]   for b in bars]
    opens  = [b["open"]  for b in bars]

    today       = bars[-1]
    yesterday   = bars[-2]
    y_close     = yesterday["close"]
    if y_close <= 0 or today["open"] <= 0:
        return None

    # ATR14 through YESTERDAY only (excludes today -- bars[:-1])
    hist = bars[:-1]
    trs = []
    for i in range(1, len(hist)):
        h, l, pc = hist[i]["high"], hist[i]["low"], hist[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < 14:
        return None
    atr14   = sum(trs[-14:]) / 14
    atr_pct = atr14 / y_close * 100 if y_close > 0 else 0.0

    gap_pct = (today["open"] - y_close) / y_close * 100

    y_open = yesterday["open"]
    prior_day_ret_pct = (y_close - y_open) / y_open * 100 if y_open > 0 else 0.0

    # 5d momentum through yesterday (yesterday's close vs close 5 sessions earlier)
    ret5d_prior = None
    if len(hist) >= 6:
        base = hist[-6]["close"]
        if base > 0:
            ret5d_prior = (y_close - base) / base * 100

    # Setup tag -- purely descriptive, doesn't affect scoring
    if gap_pct <= -1.0:
        setup = "gap-down reversion"
    elif gap_pct >= 1.0:
        setup = "gap-up momentum"
    elif atr_pct >= 4.2:
        setup = "high-ATR (top decile)"
    else:
        setup = "neutral"

    return {
        "ticker":            ticker,
        "price":             today["open"],
        "atr_pct":           round(atr_pct, 3),
        "gap_pct":           round(gap_pct, 3),
        "prior_day_ret_pct": round(prior_day_ret_pct, 3),
        "ret5d_prior":       round(ret5d_prior, 3) if ret5d_prior is not None else None,
        "setup":             setup,
    }


def compute_dt_partial_features_premarket(ticker: str, bars: list[dict]) -> dict | None:
    """Pre-market variant of compute_dt_features -- computes ONLY the 3
    inputs that don't need today's session (atr_pct, prior_day_ret_pct,
    ret5d_prior; 75% of the full composite_score's weight combined).
    gap_pct is deliberately NOT computed here -- see PREMARKET_SCAN_TIME_ET's
    comment for why (CEO didn't want it estimated from IBKR's pre-market/
    extended-hours quotes for thinner names).

    Called before the market opens, when a daily-bar fetch's last bar is
    the most recent COMPLETE session (no "today, in progress" bar exists
    yet) -- so unlike compute_dt_features, bars[-1] here IS "yesterday"
    from compute_dt_features' perspective, not "today." Same no-lookahead
    math throughout, just re-indexed by one for this different calling
    context.
    """
    if len(bars) < 21:   # need >=20 for ATR14 (needs a t-1 close) + ret5d lookback
        return None

    y_close = bars[-1]["close"]
    y_open  = bars[-1]["open"]
    if y_close <= 0:
        return None

    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < 14:
        return None
    atr14   = sum(trs[-14:]) / 14
    atr_pct = atr14 / y_close * 100 if y_close > 0 else 0.0

    prior_day_ret_pct = (y_close - y_open) / y_open * 100 if y_open > 0 else 0.0

    ret5d_prior = None
    if len(bars) >= 6:
        base = bars[-6]["close"]
        if base > 0:
            ret5d_prior = (y_close - base) / base * 100

    return {
        "ticker":            ticker,
        "atr_pct":           round(atr_pct, 3),
        "prior_day_ret_pct": round(prior_day_ret_pct, 3),
        "ret5d_prior":       round(ret5d_prior, 3) if ret5d_prior is not None else None,
    }


def compute_partial_score(candidates: list[dict]) -> None:
    """Overnight-shortlist ranking score -- same relative weighting as
    compute_dt_scores' 3 non-gap inputs, renormalized to sum to 1.0
    (0.55/0.75, 0.12/0.75, 0.08/0.75) since gap_pct isn't available yet.
    This is ONLY used to pick which ~SHORTLIST_SIZE tickers are worth a
    real, live re-check at the open -- NOT a substitute for the real
    composite_score finalize_from_shortlist() computes once gap_pct is
    real, and never sent to day_trader_agent.py directly.
    """
    atr_pool = [c["atr_pct"] for c in candidates]
    pdr_pool = [abs(c["prior_day_ret_pct"]) for c in candidates]
    r5d_pool = [abs(c["ret5d_prior"]) for c in candidates if c["ret5d_prior"] is not None]
    for c in candidates:
        c["partial_score"] = round(
            (0.55 / 0.75) * _pct_rank(c["atr_pct"], atr_pool)
            + (0.12 / 0.75) * _pct_rank(abs(c["prior_day_ret_pct"]), pdr_pool)
            + (0.08 / 0.75) * _pct_rank(abs(c["ret5d_prior"]) if c["ret5d_prior"] is not None else None, r5d_pool),
            1,
        )


def _pct_rank(val: float | None, pool: list[float]) -> float:
    """Percentile rank of val within pool (0-100). Mirrors breakout_scanner's
    _pct_rank so composite_score stays on the same scale Day Trader's
    min_composite_score gate (default 75.0) already expects."""
    if val is None or not pool:
        return 50.0
    below = sum(1 for v in pool if v < val)
    return below / len(pool) * 100


def compute_dt_scores(candidates: list[dict]) -> None:
    """Composite score (0-100), weighted roughly by the study's RandomForest
    feature importances -- atr_pct dominates (43.6%), gap magnitude second
    (17.3%, scored on ABSOLUTE value since both directions showed positive
    lift), then prior-day-return magnitude and 5d-momentum magnitude.
    Percentile ranks are computed within today's scanned universe, same
    idiom as breakout_scanner.compute_composite_scores."""
    atr_pool  = [c["atr_pct"] for c in candidates]
    gap_pool  = [abs(c["gap_pct"]) for c in candidates]
    pdr_pool  = [abs(c["prior_day_ret_pct"]) for c in candidates]
    r5d_pool  = [abs(c["ret5d_prior"]) for c in candidates if c["ret5d_prior"] is not None]

    for c in candidates:
        score = (
            0.55 * _pct_rank(c["atr_pct"], atr_pool)
          + 0.25 * _pct_rank(abs(c["gap_pct"]), gap_pool)
          + 0.12 * _pct_rank(abs(c["prior_day_ret_pct"]), pdr_pool)
          + 0.08 * _pct_rank(abs(c["ret5d_prior"]) if c["ret5d_prior"] is not None else None, r5d_pool)
        )
        c["composite_score"] = round(score, 1)


def apply_sector_cap(candidates: list[dict], sector_map: dict[str, str],
                      max_per_sector: int = MAX_PER_SECTOR) -> list[dict]:
    """Walk the score-ranked list and keep at most max_per_sector per GICS
    sector -- same idea as safe-income-screener's MAX_PER_SECTOR, applied
    here because a same-day gap-down cluster is often one correlated sector
    move, not N independent stock-specific dips (confirmed live 2026-08-06:
    7 of the top-15 candidates were all Semis on a day SPY itself was flat)."""
    kept: list[dict] = []
    sector_count: dict[str, int] = {}
    for c in candidates:
        sector = sector_map.get(c["ticker"], "?")
        n = sector_count.get(sector, 0)
        if n >= max_per_sector:
            continue
        sector_count[sector] = n + 1
        kept.append(c)
    return kept


# ── Submission ────────────────────────────────────────────────────────────────

SUBMIT_RETRY_ATTEMPTS = 3     # real fix 2026-09-06: recovers the common case (agent
                                # mid-restart, back within its own ~15s watchdog cycle)
                                # without needing the day-level retry below to kick in
SUBMIT_RETRY_WAIT_S   = 3.0    # each candidate retries on its own thread -- worst case
                                # adds ~(attempts-1)*wait to the whole PARALLEL batch's
                                # wall-clock time, not multiplied per candidate


def _submit_one(c: dict, fired_at: str) -> dict:
    """Real POST + logging for exactly one candidate -- called concurrently
    by submit_candidates_parallel (the only real submission path as of
    2026-09-05, both the live and fallback scans) so the actual request/
    error-handling logic exists in one place.

    Retries only on a real connection-level "error" status -- added
    2026-09-06 after confirming a real gap: a submission failure (agent
    down/restarting) was previously final on the first attempt, with no
    automatic recovery even for a transient few-second outage. A normal,
    valid response (skipped/watching/ordered/at_capacity/outside_hours,
    etc.) means the agent IS reachable and gave a real answer -- retrying
    those would be pointless (same answer) or wasteful, so only "error"
    (the request itself failed) triggers a retry.
    """
    payload = {
        "ticker":          c["ticker"],
        "price":           c["price"],
        "alert_fired_at":  fired_at,
        "composite_score": c["composite_score"],
        "source":          "scanner",  # 2026-09-01: explicit tag, symmetric with dispersion_combo_scanner.py
    }
    resp = {}
    for attempt in range(1, SUBMIT_RETRY_ATTEMPTS + 1):
        try:
            # timeout raised from 25s -> 50s 2026-09-01: the agent-side retry
            # (waits up to ~40s for its own IBKR auto-reconnect before giving
            # up) needs real headroom here, or this request times out on the
            # SCANNER side first and the agent's wait accomplishes nothing.
            r = requests.post(f"{DAY_TRADER_AGENT_URL}/day-trader/signal",
                               json=payload, timeout=50)
            resp = r.json() if r.ok else {"status": "error", "http": r.status_code, "body": r.text[:200]}
        except Exception as exc:
            resp = {"status": "error", "exc": str(exc)}
        if resp.get("status") != "error":
            break
        if attempt < SUBMIT_RETRY_ATTEMPTS:
            time.sleep(SUBMIT_RETRY_WAIT_S)
    result = {**c, "submit_result": resp}
    # Real bug found 2026-09-01: this used to log only resp.get("status", ...),
    # which for an error just prints the word "error" -- the actual exception/
    # HTTP detail was computed above but never persisted anywhere, so a whole
    # scan (20/20 failed, 8/31) left zero diagnostic trail. Log the real detail.
    detail = resp.get("status", "")
    if detail == "error":
        err_detail = resp.get("exc") or f"HTTP {resp.get('http')}: {resp.get('body')}"
        detail = f"error ({err_detail}) after {SUBMIT_RETRY_ATTEMPTS} attempts"
    log.info("  %-6s score=%5.1f atr%%=%5.2f gap%%=%+6.2f setup=%-20s -> %s",
              c["ticker"], c["composite_score"], c["atr_pct"], c["gap_pct"],
              c["setup"], detail)
    return result


def submit_candidates_parallel(candidates: list[dict]) -> list[dict]:
    """POST every candidate directly to the standalone day_trader_agent.py
    process (port 8010, not main.py) -- see DAY_TRADER_AGENT_URL's comment.
    The agent still applies its own gates (use_entry_filters, use_vol_filter/
    min_atr_pct, capacity, min_composite_score, freshness) -- this is a
    candidate source, not a bypass. Fires all of them CONCURRENTLY -- real
    fix 2026-09-05, found via the real 2026-09-03 scanner log showing ~20
    sequential POSTs taking ~48 real seconds (a few seconds apart each), on
    top of the far bigger ~5-6min scan-timing delay this same fix addresses.
    Also now the fallback path's own dispatcher (run_daily_scan) -- the
    plain sequential version was retired the same day once the 15-concurrent
    semaphore on day_trader_agent.py's receiving side made parallel dispatch
    safe for both paths, not just the primary one.

    Thread-based (not asyncio) since this file is otherwise fully
    synchronous and `requests` isn't async-native -- a small
    ThreadPoolExecutor is the simplest correct way to parallelize blocking
    HTTP calls here without restructuring the whole module around asyncio
    for one function.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    fired_at = datetime.utcnow().isoformat() + "Z"
    batch = candidates[:MAX_CANDIDATES_SUB]
    if not batch:
        return []
    results = [None] * len(batch)
    with ThreadPoolExecutor(max_workers=len(batch)) as ex:
        future_to_idx = {ex.submit(_submit_one, c, fired_at): i for i, c in enumerate(batch)}
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()
    return results


# ── Daily scan orchestration ─────────────────────────────────────────────────

def _score_and_submit(cfg: dict, tickers: list[str], sector_map: dict[str, str],
                       raw: dict[str, list], submit_fn, label: str) -> bool:
    """Shared core of both scan paths -- candidate scoring, sector cap,
    earnings blackout, submission, Telegram summary. Extracted 2026-09-05
    so run_daily_scan (full-universe fallback) and finalize_from_shortlist
    (fast at-open path) share ONE real implementation instead of two copies
    that could silently drift apart. `submit_fn` is submit_candidates_parallel
    in both real call sites -- kept as a parameter rather than hardcoded so
    scoring/sector-cap/earnings logic stays decoupled from how dispatch
    happens; `label` only affects logging/Telegram text.
    """
    token   = cfg.get("telegram_token", "")
    chat_id = cfg.get("telegram_chat_id", "")

    candidates = []
    for tk in tickers:
        bars = raw.get(tk)
        if not bars:
            continue
        feat = compute_dt_features(tk, bars)
        if feat is None:
            continue
        if feat["atr_pct"] < MIN_ATR_PCT_FLOOR:
            continue
        candidates.append(feat)

    if not candidates:
        log.warning("%s: no candidates cleared the %.1f%% ATR floor -- nothing to submit today.",
                     label, MIN_ATR_PCT_FLOOR)
        return True

    compute_dt_scores(candidates)
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    log.info("%s: top 15 of %d candidates (pre-sector-cap):", label, len(candidates))
    for c in candidates[:15]:
        log.info("  %-6s sector=%-13s score=%5.1f atr%%=%5.2f gap%%=%+6.2f prior_day%%=%+6.2f ret5d%%=%s setup=%s",
                  c["ticker"], sector_map.get(c["ticker"], "?"), c["composite_score"], c["atr_pct"], c["gap_pct"],
                  c["prior_day_ret_pct"],
                  f"{c['ret5d_prior']:+.2f}" if c["ret5d_prior"] is not None else "n/a",
                  c["setup"])

    capped = apply_sector_cap(candidates, sector_map)
    n_dropped_by_cap = len(candidates) - len(capped)
    if n_dropped_by_cap:
        log.info("Sector cap (max %d/sector) dropped %d otherwise-qualifying candidates "
                  "from the submission batch (still ranked below, just concentration-limited).",
                  MAX_PER_SECTOR, n_dropped_by_cap)

    # Earnings blackout -- added 2026-08-11 after external review flagged this
    # scanner (like breakout_scanner.py before its own F10 fix) had no earnings
    # awareness at all. A same-day mover into an earnings print is a materially
    # different bet (event risk) than the technical/volatility setup this
    # scanner is built to find. Checked here on the small post-cap list, not
    # the full universe, to keep yfinance calendar lookups cheap.
    n_before_earnings = len(capped)
    capped = [c for c in capped if (earnings_days_out(c["ticker"]) or 999) > EARNINGS_BLACKOUT_DAYS]
    n_dropped_earnings = n_before_earnings - len(capped)
    if n_dropped_earnings:
        log.info("Earnings blackout (<=%dd) dropped %d otherwise-qualifying candidates.",
                  EARNINGS_BLACKOUT_DAYS, n_dropped_earnings)

    submitted = submit_fn(capped)
    n_ordered = sum(1 for s in submitted if s["submit_result"].get("status") == "ordered")
    n_errored = sum(1 for s in submitted if s["submit_result"].get("status") == "error")

    if token and chat_id:
        lines = [f"🎯 <b>{label}</b> — {date.today().strftime('%a %b %d')}",
                 f"{len(candidates)} candidates cleared ATR floor, top {len(submitted)} submitted, {n_ordered} entered."]
        for s in submitted[:10]:
            tag = "✅" if s["submit_result"].get("status") == "ordered" else \
                  "❌" if s["submit_result"].get("status") == "error" else "·"
            lines.append(f"{tag} {s['ticker']} score={s['composite_score']:.0f} "
                         f"ATR%={s['atr_pct']:.1f} gap={s['gap_pct']:+.1f}% ({s['setup']})")
        send_telegram(token, chat_id, "\n".join(lines))

        # Real gap found 2026-09-01: the summary above tags a failed submission
        # the same "·" as a routine, healthy skip (already_open/at_capacity/
        # score_below_threshold) -- 8/31 sent "20 submitted, 0 entered" with no
        # signal that anything was actually broken (an IBKR disconnect on
        # day_trader_agent's own connection 503'd every single one). A batch
        # that's ALL errors, or a meaningful fraction of one, needs its own
        # unambiguous alert -- this is the pipeline saying "I failed, look at
        # me" instead of quietly doing nothing that looks identical to a
        # normal quiet day.
        if n_errored > 0:
            frac = n_errored / len(submitted) if submitted else 0
            high = frac >= 0.5
            prefix = "🚨🚨 HIGH PRIORITY — ACTION NEEDED 🚨🚨\n" if high else "⚠️ "
            examples = "; ".join(
                f"{s['ticker']}: {s['submit_result'].get('exc') or s['submit_result'].get('body') or '?'}"
                for s in submitted if s["submit_result"].get("status") == "error"
            )[:300]
            send_telegram(token, chat_id,
                f"{prefix}{label}: {n_errored}/{len(submitted)} candidate submissions "
                f"FAILED with an error (not a normal skip) -- day_trader_agent may be unreachable "
                f"or IBKR-disconnected on its own connection. Real candidates were found and lost. "
                f"Sample: {examples}")

    log.info("=== %s complete: %d candidates, %d submitted, %d entered, %d errored ===",
              label, len(candidates), len(submitted), n_ordered, n_errored)

    # Real gap found 2026-09-06: this used to return True unconditionally, even
    # when EVERY submission failed -- the caller (_main_loop) would then mark
    # last_scan_date as done for the day, permanently losing real candidates
    # with no retry, despite the Telegram alert above literally saying "Real
    # candidates were found and lost." fetch_universe_history failures already
    # get a real retry-next-minute via this same True/False contract; a
    # submission-side failure deserves the identical treatment, not a lesser
    # one just because it happens later in the pipeline. Same 50% threshold
    # already used to decide HIGH PRIORITY vs normal Telegram severity above --
    # reused here for consistency, not a new number invented from nothing.
    if submitted and n_errored / len(submitted) >= 0.5:
        log.error("%s: %d/%d submissions failed -- NOT marking today as scanned, "
                  "will retry next minute instead of losing these candidates for the day.",
                  label, n_errored, len(submitted))
        return False
    return True


def _check_dt_agent_enabled() -> None:
    # Checked directly against the standalone agent (not proxied through
    # main.py) -- accurate even if main.py itself is down/restarting.
    try:
        r = requests.get(f"{DAY_TRADER_AGENT_URL}/day-trader/status", timeout=10)
        dt_status = r.json() if r.ok else {}
    except Exception:
        dt_status = {}
    if not dt_status.get("enabled", False):
        log.info("Day Trader agent is disabled or unreachable -- scanning anyway for logging, "
                  "but /day-trader/signal will no-op every submission.")


def build_and_cache_shortlist(cfg: dict) -> bool:
    """PREMARKET_SCAN_TIME_ET step: scores the full universe on the 75%-
    weight, non-gap inputs (atr_pct/prior_day_ret_pct/ret5d_prior -- none of
    which need today's session) and caches the top SHORTLIST_SIZE tickers
    for finalize_from_shortlist() to do a fast, real-open re-check on.
    Returns True only on a real completed build -- False means
    finalize_from_shortlist must fall back to the full run_daily_scan().
    """
    backend_url = cfg.get("backend_url", "http://localhost:8000")
    tickers, sector_map = load_universe()
    log.info("=== Premarket shortlist build starting (%d tickers) ===", len(tickers))
    raw = fetch_universe_history(tickers, backend_url, HIST_DAYS)
    if not raw:
        log.error("Premarket shortlist: no history returned -- IBKR/backend likely unavailable. "
                  "finalize_from_shortlist will fall back to the full scan.")
        return False

    candidates = []
    for tk in tickers:
        bars = raw.get(tk)
        if not bars:
            continue
        feat = compute_dt_partial_features_premarket(tk, bars)
        if feat is None:
            continue
        if feat["atr_pct"] < MIN_ATR_PCT_FLOOR:
            continue
        candidates.append(feat)

    if not candidates:
        log.warning("Premarket shortlist: no candidates cleared the %.1f%% ATR floor.", MIN_ATR_PCT_FLOOR)
        return False

    compute_partial_score(candidates)
    candidates.sort(key=lambda c: c["partial_score"], reverse=True)
    shortlist = candidates[:SHORTLIST_SIZE]

    try:
        with open(SHORTLIST_PATH, "w") as f:
            json.dump({"date": date.today().isoformat(),
                       "built_at": datetime.now(ET).isoformat(),
                       "tickers": [c["ticker"] for c in shortlist]}, f, indent=2)
    except Exception as exc:
        log.error("Premarket shortlist: failed to write %s: %s", SHORTLIST_PATH, exc)
        return False

    log.info("=== Premarket shortlist built: %d of %d candidates cleared ATR floor, "
              "top %d cached for the at-open finalize pass ===",
              len(candidates), len(tickers), len(shortlist))
    log.info("Top 10 by partial_score: %s",
              ", ".join(f"{c['ticker']}({c['partial_score']:.0f})" for c in shortlist[:10]))
    return True


def finalize_from_shortlist(cfg: dict) -> bool:
    """SCAN_TIME_ET step (the FAST at-open path): loads today's cached
    shortlist, pulls REAL daily bars (including the now-real today's open)
    for just those tickers, computes the real gap_pct + full composite_score
    the normal way (compute_dt_features -- identical math to the fallback
    path, just a far smaller ticker list), and dispatches in PARALLEL.
    Returns False if the shortlist is missing/stale/empty so the caller
    falls back to the full run_daily_scan() -- a slower day beats a
    silently-skipped one.
    """
    try:
        with open(SHORTLIST_PATH) as f:
            cached = json.load(f)
        if cached.get("date") != date.today().isoformat():
            log.warning("Premarket shortlist is stale (built %s, today is %s) -- falling back "
                        "to the full scan.", cached.get("date"), date.today().isoformat())
            return False
        tickers = cached.get("tickers") or []
        if not tickers:
            log.warning("Premarket shortlist is empty -- falling back to the full scan.")
            return False
    except Exception as exc:
        log.warning("No usable premarket shortlist (%s) -- falling back to the full scan.", exc)
        return False

    _check_dt_agent_enabled()
    _, sector_map = load_universe()
    backend_url = cfg.get("backend_url", "http://localhost:8000")
    log.info("=== Fast at-open finalize starting (%d shortlisted tickers, real gap%% now available) ===",
              len(tickers))
    raw = fetch_universe_history(tickers, backend_url, HIST_DAYS)
    if not raw:
        log.error("Fast finalize: no history returned for the shortlist -- falling back to the full scan.")
        return False

    return _score_and_submit(cfg, tickers, sector_map, raw, submit_candidates_parallel,
                              "Day Trader FAST at-open finalize")


def run_daily_scan(cfg: dict) -> bool:
    """Full-universe fallback path -- used when the premarket shortlist is
    missing/stale (IBKR unreachable at 8am, etc.), or can still be invoked
    directly (--now) for manual/diagnostic runs. Returns True only on a
    real completed scan (even if 0 candidates cleared the floor) -- False
    on any abort (no IBKR/backend data), so the caller knows NOT to mark
    today as scanned and can retry instead of silently going dark for the
    rest of the day. Real bug found 2026-08-25: the very first live run
    raced the backend's restart, aborted on a connection refusal, but
    _main_loop marked last_scan_date done anyway -- zero candidates
    submitted all day with no retry.

    Dispatches in PARALLEL (submit_candidates_parallel), not sequentially --
    changed 2026-09-05. This path already lost its old 5min deliberate wait
    when SCAN_TIME_ET moved from 9:35 to 9:30:10; sequential dispatch of the
    now-uncapped sector-cleared batch (up to ~33, vs the old 20) was the
    last real, avoidable delay left in it, and the parallel path is already
    proven safe by the same 15-concurrent semaphore on the receiving side.
    A real intraday delay backtest (scan_delay_backtest.py) shows this
    fallback's remaining ~1-2min (fetch time, which can't be shortcut here
    -- that's exactly what the premarket shortlist exists to avoid) still
    retains real positive edge (avg return +0.065-0.099%/trade in that
    range), unlike the old ~7min design (+0.017%/trade) -- so this is a
    genuine safety net, not a silent revert to the original problem.
    """
    backend_url = cfg.get("backend_url", "http://localhost:8000")
    _check_dt_agent_enabled()

    tickers, sector_map = load_universe()
    log.info("=== Day Trader daily scan starting (%d tickers) ===", len(tickers))
    raw = fetch_universe_history(tickers, backend_url, HIST_DAYS)
    if not raw:
        log.error("No history returned -- IBKR/backend likely unavailable. Aborting this scan "
                  "WITHOUT marking today as scanned -- will retry next minute.")
        return False

    return _score_and_submit(cfg, tickers, sector_map, raw, submit_candidates_parallel,
                              "Day Trader daily scan (fallback path)")


# ── Dispersion-gated combo -- small, explicitly-bounded live test ──────────
# CEO-approved 2026-09-01: VolRatio>1.2 & CCI>100 & Williams%R>-50 &
# EMA9>EMA21, fired ONLY when real-time cross-sectional dispersion (20-day
# rolling avg of same-day return std across this universe, ranked against
# its own trailing 252-day history, lagged 1 day -- no lookahead) is in the
# top quartile. Backtested in ibkr_trader/backend/daytrader_research/
# (target_combo_multiregime_backtest.py + the dispersion-gate follow-up):
# n=1,932 real trades, 56.5% win / +0.184% avg / +0.054% median return with
# the account's own 0.3% trailing stop -- vs a near-coin-flip/negative-
# median result with no dispersion gate (n=7,735, full 5-year history).
# Explicit test bounds: $150/trade (Day Trader's live size), max 2
# concurrent, review at 20 closed trades or 60 calendar days. Submits
# through day_trader_agent.py's existing /day-trader/signal endpoint --
# reuses its already-live trailing-stop exit machinery rather than a new
# parallel executor, tagged source="dispersion-combo" end to end so results
# are cleanly separable from Day Trader's normal scanner-driven trades.
#
# Real, honest caveat: the backtest used yfinance daily bars; this computes
# the same self-consistent ratios/comparisons from IBKR's own daily bars
# instead (never mixing the two providers within one calculation -- the
# exact class of bug found and fixed 2026-09-01 in the trailing-stop
# backtest). Minor cross-provider differences are possible in the exact
# dispersion percentile reading, but since it's a percentile RANK against
# the ticker universe's own trailing history rather than an absolute
# threshold, it should be robust to small data differences between vendors.

DISPERSION_COMBO_UNIVERSE = sorted(set([
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    "DIS", "CMCSA", "VZ", "T",
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "XYZ", "IBM",
    "RBLX", "RCL", "ABNB",
]))
DISPERSION_COMBO_HIST_DAYS = 320       # trading days -- 252-day rank window + 20-day MA warmup +
                                        # ~48-day buffer (same structure as the original 120-day-window
                                        # sizing: window + MA + buffer)
DISPERSION_RANK_WINDOW = 252           # restored 2026-09-06 to match what was actually backtested
                                        # (indicator_combo_backtest.py, indicator_combo_backtest_trailstop.py,
                                        # target_combo_multiregime_backtest.py all standardize on a
                                        # 252-trading-day evaluation window). Was cut to 120 on 2026-09-01
                                        # to work around fetch_universe_history's duration_str hitting
                                        # IBKR's real 365-day "D"-unit ceiling -- fetch_universe_history
                                        # now switches to "Y" units past that ceiling (live-tested,
                                        # 2026-09-06), so the full validated window no longer needs to be
                                        # shrunk to fit around it.
DISPERSION_COMBO_MAX_CONCURRENT = 2
DISPERSION_COMBO_DISP_PCTILE_MIN = 0.75
DISPERSION_COMBO_RETRY_ATTEMPTS = 3    # e.g. 9:40, 10:00, 10:20 ET -- resilience against a transient
DISPERSION_COMBO_RETRY_WAIT_S = 20 * 60  # data hiccup only, never a reason to re-check for "new" signals


def run_dispersion_combo_check(cfg: dict) -> None:
    import pandas as pd
    import numpy as np

    backend_url = cfg.get("backend_url", "http://localhost:8000")

    try:
        r = requests.get(f"{DAY_TRADER_AGENT_URL}/day-trader/status", timeout=10)
        dt_status = r.json() if r.ok else {}
    except Exception as exc:
        log.error("dispersion-combo: could not reach day_trader_agent for status check: %s -- skipping today.", exc)
        return

    current = sum(1 for p in dt_status.get("positions", {}).values() if p.get("source") == "dispersion-combo") + \
              sum(1 for w in dt_status.get("watching", {}).values() if w.get("source") == "dispersion-combo")
    slots = DISPERSION_COMBO_MAX_CONCURRENT - current
    if slots <= 0:
        log.info("dispersion-combo: at its own test cap (%d/%d concurrent) -- skipping today.",
                  current, DISPERSION_COMBO_MAX_CONCURRENT)
        return

    # Retry 2026-09-01: this signal is entirely EOD-determined (combo +
    # dispersion both use yesterday's close only) -- re-running later in the
    # day would recompute the IDENTICAL result, so retrying is only useful
    # against a real, transient DATA problem (e.g. the IBKR-disconnect class
    # of issue found earlier today), never as a way to "check for new
    # signals" later. Retries ONLY on insufficient data; a real, valid "no
    # signal today" (dispersion below threshold, 0 combo matches) is not a
    # failure and must not retry -- more waiting can't change EOD data.
    raw, closes, opens_today = None, {}, {}
    for attempt in range(1, DISPERSION_COMBO_RETRY_ATTEMPTS + 1):
        raw = fetch_universe_history(DISPERSION_COMBO_UNIVERSE, backend_url, DISPERSION_COMBO_HIST_DAYS)
        closes, opens_today = {}, {}
        if raw:
            for tk, bars in raw.items():
                if len(bars) < 280:  # covers the 252-day rank window + 20d MA + indicator lookback buffer
                    continue
                hist = bars[:-1]  # exclude today's still-forming bar -- no lookahead
                closes[tk] = pd.Series(
                    data=[b["close"] for b in hist],
                    index=pd.to_datetime([b["date"] for b in hist]),
                )
                opens_today[tk] = bars[-1]["open"]
        if len(closes) >= 20:
            break
        if attempt < DISPERSION_COMBO_RETRY_ATTEMPTS:
            log.warning("dispersion-combo: attempt %d/%d got only %d/%d tickers with enough real "
                       "data (IBKR hiccup?) -- retrying in %d min.", attempt,
                       DISPERSION_COMBO_RETRY_ATTEMPTS, len(closes), len(raw) if raw else 0,
                       DISPERSION_COMBO_RETRY_WAIT_S // 60)
            time.sleep(DISPERSION_COMBO_RETRY_WAIT_S)
    else:
        log.error("dispersion-combo: %d/%d attempts all got insufficient real data -- skipping today.",
                   DISPERSION_COMBO_RETRY_ATTEMPTS, DISPERSION_COMBO_RETRY_ATTEMPTS)
        return

    wide = pd.DataFrame(closes).sort_index()
    daily_ret = wide.pct_change()
    cross_std = daily_ret.std(axis=1, skipna=True)
    disp_ma20 = cross_std.rolling(20, min_periods=15).mean()
    disp_rank = disp_ma20.rolling(DISPERSION_RANK_WINDOW, min_periods=60).rank(pct=True)
    today_disp_pctile = disp_rank.iloc[-1]  # as of yesterday's close (hist excludes today)

    if pd.isna(today_disp_pctile):
        log.warning("dispersion-combo: not enough trailing history yet for a real dispersion "
                    "percentile -- skipping today.")
        return
    if today_disp_pctile < DISPERSION_COMBO_DISP_PCTILE_MIN:
        log.info("dispersion-combo: dispersion percentile %.2f < %.2f threshold -- no signals "
                 "today regardless of combo matches.", today_disp_pctile, DISPERSION_COMBO_DISP_PCTILE_MIN)
        return

    candidates = []
    for tk in closes:
        bars = raw[tk]
        hist = bars[:-1]
        c = pd.Series([b["close"] for b in hist])
        h = pd.Series([b["high"] for b in hist])
        l = pd.Series([b["low"] for b in hist])
        v = pd.Series([b["volume"] for b in hist])

        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        vol_avg20 = v.rolling(20).mean()
        if pd.isna(vol_avg20.iloc[-1]) or vol_avg20.iloc[-1] == 0:
            continue
        vol_ratio = v.iloc[-1] / vol_avg20.iloc[-1]

        tp = (h + l + c) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
        willr = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()

        cci_last, willr_last = cci.iloc[-1], willr.iloc[-1]
        if pd.isna(cci_last) or pd.isna(willr_last):
            continue

        if vol_ratio > 1.2 and cci_last > 100 and willr_last > -50 and ema9.iloc[-1] > ema21.iloc[-1]:
            candidates.append({"ticker": tk, "price": opens_today[tk]})

    if not candidates:
        log.info("dispersion-combo: dispersion elevated (%.2f pctile) but 0/%d tickers matched "
                 "the combo today.", today_disp_pctile, len(closes))
        return

    log.info("dispersion-combo: dispersion pctile=%.2f -- %d candidate(s) matched: %s "
             "(submitting up to %d, %d slot(s) open)",
             today_disp_pctile, len(candidates), [c["ticker"] for c in candidates], slots, slots)

    fired_at = datetime.utcnow().isoformat() + "Z"
    for c in candidates[:slots]:
        payload = {"ticker": c["ticker"], "price": c["price"], "alert_fired_at": fired_at,
                   "source": "dispersion-combo", "skip_entry_filters": True}
        try:
            resp = requests.post(f"{DAY_TRADER_AGENT_URL}/day-trader/signal", json=payload, timeout=50)
            result = resp.json() if resp.ok else {"status": "error", "http": resp.status_code, "body": resp.text[:200]}
        except Exception as exc:
            result = {"status": "error", "exc": str(exc)}
        log.info("  dispersion-combo submit %-6s -> %s", c["ticker"], result.get("status", result))


# ── Singleton lock (own port, independent of breakout_scanner's) ────────────
_SINGLETON_PORT = 47653
_singleton_sock = None


def _acquire_singleton() -> None:
    global _singleton_sock
    try:
        _singleton_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        _singleton_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 0)
        _singleton_sock.bind(("127.0.0.1", _SINGLETON_PORT))
    except OSError:
        log.error("Another daytrader_scanner instance is already running (port %d held) -- exiting.",
                   _SINGLETON_PORT)
        sys.exit(1)


def _release_singleton() -> None:
    global _singleton_sock
    if _singleton_sock is not None:
        _singleton_sock.close()
        _singleton_sock = None


def _main_loop(run_now: bool = False) -> None:
    cfg = load_shared_config()
    st  = load_state()
    tickers, _ = load_universe()
    log.info("Day Trader Scanner started. Universe=%d tickers. Premarket shortlist=%02d:%02d ET, "
              "at-open finalize=%02d:%02d:%02d ET. Last scan date: %s Last shortlist date: %s",
              len(tickers), PREMARKET_SCAN_TIME_ET[0], PREMARKET_SCAN_TIME_ET[1],
              SCAN_TIME_ET[0], SCAN_TIME_ET[1], FINALIZE_BUFFER_S,
              st.get("last_scan_date", "never"), st.get("last_shortlist_date", "never"))

    if run_now:
        ok = run_daily_scan(cfg)
        if ok:
            st["last_scan_date"] = date.today().isoformat()
            save_state(st)
        else:
            log.warning("--now scan did not complete -- last_scan_date NOT updated.")
        try:
            run_dispersion_combo_check(cfg)
        except Exception as exc:
            log.error("dispersion-combo check crashed: %s", exc, exc_info=True)
        return

    while True:
        now_et = datetime.now(ET)
        today_iso = now_et.date().isoformat()
        already_scanned   = st.get("last_scan_date") == today_iso
        already_shortlisted = st.get("last_shortlist_date") == today_iso

        premarket_dt = now_et.replace(hour=PREMARKET_SCAN_TIME_ET[0], minute=PREMARKET_SCAN_TIME_ET[1],
                                       second=0, microsecond=0)
        finalize_dt = now_et.replace(hour=SCAN_TIME_ET[0], minute=SCAN_TIME_ET[1],
                                      second=FINALIZE_BUFFER_S, microsecond=0)

        if now_et.weekday() < 5:
            # ── Stage 1: premarket shortlist (8:00 ET) -- no live market data
            # needed, safe to build well ahead of the open. Failure here is
            # NOT fatal to the day: last_shortlist_date is simply left unset,
            # and finalize_from_shortlist's own missing-shortlist check sends
            # stage 2 down the full run_daily_scan() fallback instead.
            if not already_shortlisted and now_et >= premarket_dt:
                try:
                    if build_and_cache_shortlist(load_shared_config()):
                        st["last_shortlist_date"] = today_iso
                        save_state(st)
                    else:
                        log.warning("Premarket shortlist build did not complete -- "
                                    "at-open finalize will fall back to the full scan.")
                except Exception as exc:
                    log.error("Premarket shortlist build crashed: %s", exc, exc_info=True)

            # ── Stage 2: fast at-open finalize (9:30:10 ET), falling back to
            # the full 483-ticker scan if the shortlist isn't usable ──────────
            if not already_scanned and now_et >= finalize_dt:
                ok = False
                try:
                    ok = finalize_from_shortlist(load_shared_config())
                    if not ok:
                        ok = run_daily_scan(load_shared_config())
                except Exception as exc:
                    log.error("At-open finalize/fallback crashed: %s", exc, exc_info=True)
                if ok:
                    st["last_scan_date"] = today_iso
                    save_state(st)
                    try:
                        run_dispersion_combo_check(load_shared_config())
                    except Exception as exc:
                        log.error("dispersion-combo check crashed: %s", exc, exc_info=True)
                # else: leave last_scan_date unset -- retried again next loop tick

        # Tighten polling near either trigger so real precision isn't lost to
        # coarse 60s polling -- the whole point of this pipeline is minimizing
        # delay, so re-introducing up to 59s of poll-cadence slop right at the
        # 9:30 open would undercut it. 60s the rest of the day is fine --
        # neither trigger needs sub-minute precision outside its own window.
        near_a_trigger = (
            (not already_shortlisted and abs((now_et - premarket_dt).total_seconds()) < 120)
            or (not already_scanned and abs((now_et - finalize_dt).total_seconds()) < 120)
        )
        time.sleep(2 if near_a_trigger else 60)


def main():
    _acquire_singleton()
    try:
        run_now = "--now" in sys.argv
        _main_loop(run_now=run_now)
    finally:
        _release_singleton()


if __name__ == "__main__":
    main()
