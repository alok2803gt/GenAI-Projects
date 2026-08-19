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

Runs ONCE per trading day, ~9:40 ET (a few minutes after the open so the
actual opening print has settled). Unlike breakout_scanner's continuous
intraday polling loop, none of the features above are meaningfully
re-computable mid-day -- gap_pct in particular is only ever defined at the
open -- so a single daily batch scan matches exactly what was backtested,
rather than pretending to support an intraday re-evaluation cadence that
was never validated.

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
SCAN_TIME_ET       = (9, 35)   # run once daily at this ET hour:minute
HIST_DAYS          = 40        # daily bars fetched per ticker (mirrors _fetch_entry_metrics)
HISTORY_TIMEOUT_S  = 240        # /market/history/bulk over ~500 tickers, paced 40/batch + 2s -- allow ample time
MAX_CANDIDATES_SUB = 20         # submit at most this many per day (backend enforces max_positions/quality gates too)
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

def fetch_universe_history(tickers: list[str], backend_url: str, days: int = HIST_DAYS) -> dict[str, list]:
    """POST /market/history/bulk -- IBKR daily bars, useRTH=True.

    During market hours the LAST bar is today's still-forming session (its
    Open is the real opening print, which is exactly the feature this
    scanner needs); every earlier bar is a completed prior session.
    """
    try:
        r = requests.post(
            f"{backend_url.rstrip('/')}/market/history/bulk",
            json=tickers, params={"days": days}, timeout=HISTORY_TIMEOUT_S,
        )
        if not r.ok:
            log.error("market/history/bulk returned %d: %s", r.status_code, r.text[:300])
            return {}
        data = r.json()
        log.info("history/bulk: %d/%d tickers returned", data.get("tickers_returned", 0), len(tickers))
        return data.get("data", {})
    except Exception as exc:
        log.error("market/history/bulk failed: %s", exc)
        return {}


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

def submit_candidates(candidates: list[dict], backend_url: str) -> list[dict]:
    """POST each candidate to /day-trader/signal. Backend still applies its
    own gates (use_entry_filters, use_vol_filter/min_atr_pct, capacity,
    min_composite_score, freshness) -- this is a candidate source, not a
    bypass."""
    fired_at = datetime.utcnow().isoformat() + "Z"
    results = []
    for c in candidates[:MAX_CANDIDATES_SUB]:
        payload = {
            "ticker":          c["ticker"],
            "price":           c["price"],
            "alert_fired_at":  fired_at,
            "composite_score": c["composite_score"],
        }
        try:
            r = requests.post(f"{backend_url.rstrip('/')}/day-trader/signal",
                               json=payload, timeout=25)
            resp = r.json() if r.ok else {"status": "error", "http": r.status_code, "body": r.text[:200]}
        except Exception as exc:
            resp = {"status": "error", "exc": str(exc)}
        results.append({**c, "submit_result": resp})
        log.info("  %-6s score=%5.1f atr%%=%5.2f gap%%=%+6.2f setup=%-20s -> %s",
                  c["ticker"], c["composite_score"], c["atr_pct"], c["gap_pct"],
                  c["setup"], resp.get("status", resp))
    return results


# ── Daily scan orchestration ─────────────────────────────────────────────────

def run_daily_scan(cfg: dict) -> None:
    backend_url = cfg.get("backend_url", "http://localhost:8000")
    token       = cfg.get("telegram_token", "")
    chat_id     = cfg.get("telegram_chat_id", "")

    try:
        r = requests.get(f"{backend_url.rstrip('/')}/day-trader/status", timeout=10)
        dt_status = r.json() if r.ok else {}
    except Exception:
        dt_status = {}
    if not dt_status.get("enabled", False):
        log.info("Day Trader is disabled on the backend -- scanning anyway for logging, "
                  "but /day-trader/signal will no-op every submission.")

    tickers, sector_map = load_universe()
    log.info("=== Day Trader daily scan starting (%d tickers) ===", len(tickers))
    raw = fetch_universe_history(tickers, backend_url, HIST_DAYS)
    if not raw:
        log.error("No history returned -- IBKR likely disconnected. Aborting this scan.")
        return

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
        log.warning("No candidates cleared the %.1f%% ATR floor -- nothing to submit today.", MIN_ATR_PCT_FLOOR)
        return

    compute_dt_scores(candidates)
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    log.info("Top 15 of %d candidates (pre-sector-cap):", len(candidates))
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

    submitted = submit_candidates(capped, backend_url)
    n_ordered = sum(1 for s in submitted if s["submit_result"].get("status") == "ordered")

    if token and chat_id:
        lines = [f"🎯 <b>Day Trader scan</b> — {date.today().strftime('%a %b %d')}",
                 f"{len(candidates)} candidates cleared ATR floor, top {len(submitted)} submitted, {n_ordered} entered."]
        for s in submitted[:10]:
            tag = "✅" if s["submit_result"].get("status") == "ordered" else "·"
            lines.append(f"{tag} {s['ticker']} score={s['composite_score']:.0f} "
                         f"ATR%={s['atr_pct']:.1f} gap={s['gap_pct']:+.1f}% ({s['setup']})")
        send_telegram(token, chat_id, "\n".join(lines))

    log.info("=== Scan complete: %d candidates, %d submitted, %d entered ===",
              len(candidates), len(submitted), n_ordered)


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
    log.info("Day Trader Scanner started. Universe=%d tickers. Scan time=%02d:%02d ET. "
              "Last scan date: %s", len(tickers), SCAN_TIME_ET[0], SCAN_TIME_ET[1],
              st.get("last_scan_date", "never"))

    if run_now:
        run_daily_scan(cfg)
        st["last_scan_date"] = date.today().isoformat()
        save_state(st)
        return

    while True:
        now_et = datetime.now(ET)
        today_iso = now_et.date().isoformat()
        already_scanned = st.get("last_scan_date") == today_iso

        if now_et.weekday() < 5 and not already_scanned:
            scan_dt = now_et.replace(hour=SCAN_TIME_ET[0], minute=SCAN_TIME_ET[1],
                                      second=0, microsecond=0)
            if now_et >= scan_dt:
                try:
                    run_daily_scan(load_shared_config())
                except Exception as exc:
                    log.error("Daily scan crashed: %s", exc, exc_info=True)
                st["last_scan_date"] = today_iso
                save_state(st)

        time.sleep(60)


def main():
    _acquire_singleton()
    try:
        run_now = "--now" in sys.argv
        _main_loop(run_now=run_now)
    finally:
        _release_singleton()


if __name__ == "__main__":
    main()
