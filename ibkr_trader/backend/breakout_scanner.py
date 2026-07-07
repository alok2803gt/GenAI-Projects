#!/usr/bin/env python3
"""
Breakout Scanner — Telegram Alerts (curated universe, cached history)
======================================================================

Scans a curated 125-ticker universe (high-liquidity S&P 500 names with
active options) rather than all 503 S&P 500 constituents. Historical OHLCV
is pre-cached at startup (1 yr) and refreshed with a tiny 5-day fetch before
each cycle, so the hot-path scan is pure in-memory computation.

Architecture
------------
  Startup:    load_hist_cache()    — yfinance 1yr bulk download (~20s once)
  Each cycle: refresh_hist_cache() — 5d fetch via IBKR backend or yfinance
              scan_universe()      — reads _hist_cache, no network calls
  Interval:   3 minutes (was 15 min; now feasible because scan takes ~5s)

Signal definitions
------------------
  BREAKOUT     — %B > 95 AND volume ≥ per-ticker 90th-pct ratio.
  PRE-BREAKOUT — %B 75-95, RSI > 60, above SMA-20/50, volume building.
"""

import json
import logging
import logging.handlers
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Logging — writes directly to scanner.log (5 MB × 3 backups) ──────────────
# Using RotatingFileHandler so the scanner owns its own log regardless of how
# it is launched (Task Scheduler, terminal, watchdog). No PowerShell redirection
# needed — the file is always in the backend directory.
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(SCRIPT_DIR, "scanner.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
log = logging.getLogger("breakout_scanner")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "scanner_config.json")

# ── Curated universe — 125 high-liquidity names with active options chains ────
# Selected for: tight option spreads, >500 OI on near-term ATM strikes,
# breakout history, and sector breadth.  Refresh annually or when index
# composition changes significantly.
CURATED_TICKERS: list[str] = sorted(set([
    # ETFs — largest options OI in the market
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    # Semiconductors / hardware
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    # Cloud / enterprise software
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    # Finance / banks / brokers
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    # Healthcare / pharma / biotech
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    # Consumer discretionary
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    # Consumer staples
    "PG", "KO", "PEP", "WMT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    # Industrials / defense / transport
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    # Communication services
    "DIS", "CMCSA", "VZ", "T",
    # High-beta / high-options-volume growth
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "XYZ", "IBM",
    # Miscellaneous with active options
    "RBLX", "RCL", "ABNB",
]))


# ── Config management ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "telegram_token":         "",
    "telegram_chat_id":       "",
    "scan_interval_minutes":  3,      # fast because scan is now in-memory
    "batch_size":             50,     # yfinance batch size (initial cache load only)
    "alert_on":               ["BREAKOUT", "PRE-BREAKOUT"],
    "breakout_pct_b_min":     95,
    "pre_breakout_pct_b_min": 75,
    "pre_breakout_rsi_min":   60,
    "vol_threshold_pct":      0.75,
    "backend_url":            "http://localhost:8000",
    "use_ibkr_hist":          True,   # refresh cache via backend IBKR endpoint; falls back to yfinance
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        log.info("Created scanner_config.json — fill in telegram_token and telegram_chat_id")
    with open(CONFIG_PATH) as f:
        cfg = {**DEFAULT_CONFIG, **json.load(f)}
    # env vars override file
    cfg["telegram_token"]   = os.getenv("TELEGRAM_TOKEN",   cfg["telegram_token"])
    cfg["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", cfg["telegram_chat_id"])
    return cfg


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        log.warning("Telegram not configured — message suppressed")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            # Log the actual Telegram error so misconfigured chat_id is visible
            log.warning("Telegram send failed [HTTP %d] → %s", r.status_code, r.text)
            return False
        return True
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


def fetch_tape_sentiment(backend_url: str, ticker: str) -> dict | None:
    """Fetch CVD tape sentiment for ticker from backend. Returns None on any error or stale data."""
    if not backend_url:
        return None
    try:
        r = requests.get(
            f"{backend_url.rstrip('/')}/tape/sentiment/{ticker}",
            timeout=2,
        )
        if not r.ok:
            return None
        d = r.json()
        return d if d.get("fresh") else None
    except Exception:
        return None


def fetch_regime(backend_url: str) -> dict:
    """Fetch market regime from backend (SPY SMA-200 + live VIX).

    Returns permissive defaults when backend is unreachable so alerts are never
    suppressed purely due to a backend outage.
    """
    if not backend_url:
        return {"regime_ok": True}
    try:
        r = requests.get(f"{backend_url.rstrip('/')}/market/regime", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"regime_ok": True}  # permissive — backend down ≠ bad market


def post_to_backend(backend_url: str, ind: dict, signal_type: str) -> dict:
    """POST alert to IBKR Trader backend watchlist.

    Returns the response dict with 'action' key:
      added    — new watchlist entry accepted
      refreshed — existing entry updated
      blocked  — backend gate rejected (VIX spike or bear regime)
      error    — backend returned non-2xx
      no_backend — backend unreachable
    """
    if not backend_url:
        return {"action": "no_backend"}
    try:
        r = requests.post(
            f"{backend_url.rstrip('/')}/watchlist/alert",
            json={
                "ticker":         ind["ticker"],
                "signal_type":    signal_type,
                "price_at_alert": ind.get("price", 0),
                "pct_b":          ind.get("pct_b", 0),
                "rsi":            ind.get("rsi14"),
                "vol_ratio":      ind.get("vol_ratio"),
                "timestamp_et":   datetime.now(ET).strftime("%H:%M ET %Y-%m-%d"),
            },
            timeout=3,
        )
        if r.ok:
            return r.json()
        return {"action": "error", "status": r.status_code}
    except Exception:
        return {"action": "no_backend"}


def post_to_backend_with_tape(backend_url: str, ind: dict, signal_type: str,
                               tape: dict | None) -> dict:
    """post_to_backend that also includes tape sentiment in the payload."""
    if not backend_url:
        return {"action": "no_backend"}
    try:
        payload: dict = {
            "ticker":         ind["ticker"],
            "signal_type":    signal_type,
            "price_at_alert": ind.get("price", 0),
            "pct_b":          ind.get("pct_b", 0),
            "rsi":            ind.get("rsi"),          # was incorrectly "rsi14"
            "vol_ratio":      ind.get("vol_ratio"),
            "timestamp_et":   datetime.now(ET).strftime("%H:%M ET %Y-%m-%d"),
        }
        if tape:
            payload["tape_score"] = tape.get("score")
            payload["tape_label"] = tape.get("label")
        # State lifecycle context (Stage 2 enrichment)
        if ind.get("prev_state")           is not None: payload["prev_state"]           = ind["prev_state"]
        if ind.get("mins_in_pre_breakout") is not None: payload["mins_in_pre_breakout"] = ind["mins_in_pre_breakout"]
        if ind.get("state_path")           is not None: payload["state_path"]           = ind["state_path"]
        if ind.get("sr_resistance")        is not None: payload["sr_resistance"]        = ind["sr_resistance"]
        if ind.get("sr_support")           is not None: payload["sr_support"]           = ind["sr_support"]
        r = requests.post(
            f"{backend_url.rstrip('/')}/watchlist/alert",
            json=payload,
            timeout=3,
        )
        if r.ok:
            return r.json()
        return {"action": "error", "status": r.status_code}
    except Exception:
        return {"action": "no_backend"}


def fmt_alert(ind: dict, signal: str, tape: dict | None = None) -> str:
    emoji   = "🚨" if signal == "BREAKOUT" else "⚡"
    sma_txt = "All SMAs ✓" if (ind["above_sma20"] and ind["above_sma50"] and ind["above_sma200"]) else \
              ("20/50 SMA ✓" if (ind["above_sma20"] and ind["above_sma50"]) else "partial trend")
    day_sign  = "+" if ind["day_chg_pct"] >= 0 else ""
    scale     = ind.get("vol_scale", 1.0)
    vol_str   = (f"{ind['vol_ratio']:.2f}×" if ind.get("vol_ratio") else "N/A")
    proj_note = f" (proj {scale:.1f}×)" if scale > 1.05 else ""

    tape_line = ""
    if tape:
        score = tape.get("score", 0.0)
        label = tape.get("label", "NEUTRAL")
        comp  = tape.get("components", {})
        sign  = "+" if score >= 0 else ""
        tape_emoji = (
            "🟢" if label.startswith("STRONGLY BULL") else
            "🟩" if label.startswith("BULL")          else
            "🔴" if label.startswith("STRONGLY BEAR") else
            "🟥" if label.startswith("BEAR")          else
            "⬜"
        )
        tape_line = (
            f"\n🎯 Tape: {tape_emoji} <b>{label}</b> ({sign}{score:.2f})"
            f"  cvd {'+' if comp.get('cvd',0)>=0 else ''}{comp.get('cvd',0):.2f}"
            f" · vwap_z {'+' if comp.get('vwap_z',0)>=0 else ''}{comp.get('vwap_z',0):.2f}"
        )

    # Intraday confirmation line (dual-%B signals only)
    intraday_line = ""
    if ind.get("intraday_confirmed"):
        intraday_line = (
            f"\n📡 <b>Intraday confirmed</b>: 15m %B={ind.get('intraday_pct_b', 0):.0f}"
            f"  |  daily %B={ind['pct_b']:.1f}  (wide bands — intraday move)"
        )

    # Quality context line — shows the extra filter metrics when available
    quality_parts = []
    if ind.get("pct_b_min10") is not None:
        quality_parts.append(f"Pullback %B min={ind['pct_b_min10']:.0f}")
    if ind.get("close_pos") is not None:
        quality_parts.append(f"Close top {ind['close_pos']:.0f}% of range")
    if ind.get("ret_20d") is not None:
        quality_parts.append(f"20d ret={ind['ret_20d']:+.1f}%")
    quality_line = ("\n🔍 " + "  |  ".join(quality_parts)) if quality_parts else ""

    # Composite score line
    score = ind.get("composite_score")
    adx   = ind.get("adx")
    p5ret = ind.get("prior_5d_ret")
    vt    = ind.get("vol_trend")
    score_bar = ""
    if score is not None:
        filled = int(round(score / 10))          # 0-10 blocks
        score_bar = "█" * filled + "░" * (10 - filled)
        score_emoji = "🔥" if score >= 75 else ("⚡" if score >= 55 else "·")
        adx_str   = f"ADX={adx:.0f}"   if adx   is not None else ""
        p5_str    = f"5d={p5ret:+.1f}%" if p5ret is not None else ""
        vt_str    = f"vol-trend={vt:.2f}x" if vt is not None else ""
        detail    = "  ".join(x for x in [adx_str, p5_str, vt_str] if x)
        score_line = (
            f"\n{score_emoji} <b>Signal score: {score:.0f}/100</b> [{score_bar}]"
            + (f"\n   {detail}" if detail else "")
        )
    else:
        score_line = ""

    sr_parts = []
    if ind.get("sr_resistance"):
        dist = (ind["sr_resistance"] / ind["price"] - 1) * 100
        sr_parts.append(f"R ${ind['sr_resistance']:.2f} (+{dist:.1f}%)")
    if ind.get("sr_support"):
        dist = (ind["price"] / ind["sr_support"] - 1) * 100
        sr_parts.append(f"S ${ind['sr_support']:.2f} (-{dist:.1f}%)")
    sr_line = ("\n\U0001f4d0 S/R: " + "  |  ".join(sr_parts)) if sr_parts else ""

    return (
        f"{emoji} <b>{signal}</b> — {ind['ticker']}\n"
        f"💰 Price: ${ind['price']:.2f} ({day_sign}{ind['day_chg_pct']:.1f}%)\n"
        f"📊 %B: {ind['pct_b']:.1f}  |  Vol: {vol_str}{proj_note} (90th-pct: {ind['vol_90pct']:.2f}×)\n"
        f"📈 RSI: {ind['rsi']:.0f}  |  {sma_txt}"
        f"{intraday_line}"
        f"{quality_line}"
        f"{score_line}"
        f"{sr_line}"
        f"{tape_line}"
    )


# ── Historical data cache ─────────────────────────────────────────────────────
# Populated once at startup (1yr yfinance) then refreshed each cycle (5d).
# scan_universe() reads from here — no yfinance in the hot path.
_hist_cache:    dict[str, pd.DataFrame] = {}
_ticker_states: dict[str, dict]        = {}   # state ledger — persists for the trading session
TAPE_DB = os.path.join(SCRIPT_DIR, "tape_data.db")


def _chunked(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _yf_batch_download(tickers: list[str], period: str,
                        batch_size: int = 50) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV via yfinance. Returns {ticker: DataFrame}."""
    result: dict[str, pd.DataFrame] = {}
    batches = list(_chunked(tickers, batch_size))
    for i, batch in enumerate(batches, 1):
        log.info("  yfinance batch %d/%d (%d tickers, period=%s)…", i, len(batches), len(batch), period)
        try:
            raw = yf.download(
                batch, period=period, interval="1d",
                group_by="ticker", progress=False,
                auto_adjust=True, threads=True,
            )
            if len(batch) == 1:
                tk = batch[0]
                if not raw.empty:
                    result[tk] = raw.dropna(how="all")
            else:
                for tk in batch:
                    try:
                        df = raw[tk].dropna(how="all")
                        if not df.empty:
                            result[tk] = df
                    except (KeyError, TypeError):
                        pass
        except Exception as exc:
            log.warning("  yfinance batch error: %s", exc)
        if i < len(batches):
            time.sleep(0.5)
    return result


def fetch_intraday_pct_b(tickers: list[str]) -> dict[str, float]:
    """Fetch 15-min intraday bars and compute 20-bar Bollinger %B for each ticker.

    Used by the dual-%B check in scan_universe() to catch strong intraday moves
    that are invisible on daily Bollinger Bands (wide bands, mid-range close).
    Returns {ticker: pct_b_15m} for tickers with >= 20 intraday bars available.
    """
    if not tickers:
        return {}
    try:
        # Use 2d so yesterday's bars warm up the 20-bar BB early in the session.
        # A 15-min chart has ~26 bars/day, so 2d gives ~52 bars — plenty for BB(20).
        raw = yf.download(
            tickers, period="2d", interval="15m",
            progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        log.warning("Intraday fetch failed: %s", exc)
        return {}

    result: dict[str, float] = {}
    is_multi = isinstance(raw.columns, pd.MultiIndex)
    for tk in tickers:
        try:
            if is_multi:
                df = raw.xs(tk, axis=1, level=1).dropna(how="all")
            else:
                df = raw.copy() if len(tickers) == 1 else pd.DataFrame()
            c = df["Close"].squeeze()
            if len(c) < 20:
                continue
            sma   = c.rolling(20).mean()
            std   = c.rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            bw    = upper - lower
            pct_b = (c - lower) / bw * 100
            val   = float(pct_b.iloc[-1])
            if pd.notna(val):
                result[tk] = round(val, 1)
        except Exception:
            pass
    return result


def load_hist_cache(tickers: list[str], batch_size: int = 50, backend_url: str = "") -> None:
    """Download 1 year of daily OHLCV for all tickers at startup.

    Tries the IBKR bulk endpoint first (252 trading days, ~30s); falls back to
    yfinance if IBKR returns fewer than 80% of tickers.  After this, each scan
    cycle only needs a tiny 5-day refresh.
    """
    global _hist_cache
    log.info("Loading historical cache: %d tickers (1yr daily)…", len(tickers))
    t0 = time.monotonic()

    if backend_url:
        log.info("  Trying IBKR bulk endpoint (252d)…")
        try:
            ibkr_data: dict[str, pd.DataFrame] = {}
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                r = requests.post(
                    f"{backend_url.rstrip('/')}/market/history/bulk",
                    json=batch,
                    params={"days": 252},
                    timeout=120,
                )
                if r.ok:
                    for tk, bars in r.json().get("data", {}).items():
                        if not bars:
                            continue
                        df = pd.DataFrame(bars)
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                        df.index.name = "Date"
                        df.rename(columns={"open": "Open", "high": "High",
                                           "low": "Low", "close": "Close",
                                           "volume": "Volume"}, inplace=True)
                        ibkr_data[tk] = df
                if i + batch_size < len(tickers):
                    time.sleep(2)  # IBKR pacing between batches
            if len(ibkr_data) >= 0.8 * len(tickers):
                _hist_cache = ibkr_data
                missing = [t for t in tickers if t not in ibkr_data]
                if missing:
                    log.info("  yfinance fill for %d tickers IBKR missed", len(missing))
                    _hist_cache.update(_yf_batch_download(missing, period="1y", batch_size=batch_size))
                log.info("Historical cache ready via IBKR: %d/%d tickers in %.0fs",
                         len(_hist_cache), len(tickers), time.monotonic() - t0)
                return
            log.warning("IBKR returned only %d/%d tickers — falling back to yfinance",
                        len(ibkr_data), len(tickers))
        except Exception as exc:
            log.warning("IBKR bulk startup load failed: %s — falling back to yfinance", exc)

    log.info("  Falling back to yfinance 1yr download…")
    data = _yf_batch_download(tickers, period="1y", batch_size=batch_size)
    _hist_cache = data
    log.info("Historical cache ready via yfinance: %d/%d tickers in %.0fs",
             len(_hist_cache), len(tickers), time.monotonic() - t0)


def _merge_into_cache(new_data: dict[str, pd.DataFrame]) -> None:
    """Merge fresh OHLCV rows into _hist_cache, deduplicating by date."""
    for tk, new_df in new_data.items():
        if tk in _hist_cache and not _hist_cache[tk].empty:
            combined = pd.concat([_hist_cache[tk], new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            _hist_cache[tk] = combined
        else:
            _hist_cache[tk] = new_df


def _refresh_via_ibkr(tickers: list[str], backend_url: str) -> dict[str, pd.DataFrame]:
    """Fetch last 5 trading days of OHLCV via the backend's IBKR endpoint.

    The backend proxies IBKR reqHistoricalData — faster and more reliable than
    yfinance for incremental updates (no Yahoo rate limits, local socket).
    Returns {} if the backend is unreachable or IBKR is disconnected.
    """
    if not backend_url:
        return {}
    try:
        r = requests.post(
            f"{backend_url.rstrip('/')}/market/history/bulk",
            json=tickers,
            params={"days": 5},
            timeout=45,   # allow time for IBKR pacing across 125 tickers
        )
        if not r.ok:
            log.warning("IBKR bulk history endpoint returned %d", r.status_code)
            return {}
        raw = r.json().get("data", {})
        result: dict[str, pd.DataFrame] = {}
        for tk, bars in raw.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.index.name = "Date"
            df.rename(columns={"open": "Open", "high": "High",
                                "low": "Low", "close": "Close",
                                "volume": "Volume"}, inplace=True)
            result[tk] = df
        return result
    except Exception as exc:
        log.warning("IBKR bulk history failed: %s", exc)
        return {}


def refresh_hist_cache(tickers: list[str], backend_url: str = "",
                        use_ibkr: bool = True, batch_size: int = 50) -> None:
    """Update _hist_cache with the latest 5 trading days before each scan.

    Tries the backend's IBKR endpoint first (fast, ~3s for 125 tickers).
    Falls back to yfinance period='5d' if IBKR is unavailable.
    """
    if use_ibkr and backend_url:
        ibkr_data = _refresh_via_ibkr(tickers, backend_url)
        if ibkr_data:
            _merge_into_cache(ibkr_data)
            log.info("Cache refreshed via IBKR: %d tickers updated", len(ibkr_data))
            # yfinance fallback for any ticker IBKR missed
            missing = [t for t in tickers if t not in ibkr_data]
            if missing:
                log.info("  yfinance fallback for %d tickers IBKR missed", len(missing))
                _merge_into_cache(_yf_batch_download(missing, period="5d", batch_size=batch_size))
            return

    # Full yfinance fallback (3 batches × ~1-2s = ~5s)
    log.info("Refreshing cache via yfinance (5d)…")
    _merge_into_cache(_yf_batch_download(tickers, period="5d", batch_size=batch_size))


# ── Technical indicators ──────────────────────────────────────────────────────

def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's EMA smoothing (used by RSI, ATR, ADX)."""
    out = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    start = valid[0]
    if start + period > len(arr):
        return out
    out[start + period - 1] = np.nanmean(arr[start : start + period])
    for i in range(start + period, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = out[i - 1] * (period - 1) / period + arr[i] / period
    return out


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> float:
    """Compute ADX(period) from OHLC numpy arrays. Returns NaN if insufficient data."""
    n = len(close)
    if n < period * 2 + 1:
        return float("nan")
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high  = np.roll(high,  1); prev_high[0]  = high[0]
    prev_low   = np.roll(low,   1); prev_low[0]   = low[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = minus_dm[0] = 0.0
    atr      = _wilder_smooth(tr,       period)
    plus_di  = _wilder_smooth(plus_dm,  period)
    minus_di = _wilder_smooth(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(atr > 0, 100 * plus_di  / atr, np.nan)
        mdi = np.where(atr > 0, 100 * minus_di / atr, np.nan)
        di_sum = pdi + mdi
        dx     = np.where(di_sum > 0, 100 * np.abs(pdi - mdi) / di_sum, 0.0)
    adx = _wilder_smooth(dx, period)
    last = adx[~np.isnan(adx)]
    return float(last[-1]) if len(last) else float("nan")


def _sr_quick(highs: np.ndarray, lows: np.ndarray, price: float,
              n_confirm: int = 5, lookback: int = 60, tol: float = 0.003) -> tuple:
    """Nearest swing resistance above price and swing support below price.
    Uses the lookback bars BEFORE the current bar — no lookahead.
    Returns (resistance, support) — either may be None.
    """
    h = highs[-lookback - 1: -1]   # exclude today's bar
    l = lows[-lookback - 1: -1]
    n = len(h)
    raw_res, raw_sup = [], []
    for i in range(n_confirm, n - n_confirm):
        if all(h[i] >= h[i-j] for j in range(1, n_confirm+1)) and \
           all(h[i] >= h[i+j] for j in range(1, n_confirm+1)):
            if h[i] > price: raw_res.append(float(h[i]))
        if all(l[i] <= l[i-j] for j in range(1, n_confirm+1)) and \
           all(l[i] <= l[i+j] for j in range(1, n_confirm+1)):
            if l[i] < price: raw_sup.append(float(l[i]))

    def _cl(lvls):
        if not lvls: return []
        out, grp = [], [sorted(lvls)[0]]
        for v in sorted(lvls)[1:]:
            if (v - grp[0]) / grp[0] <= tol: grp.append(v)
            else: out.append(round(sum(grp)/len(grp), 2)); grp = [v]
        out.append(round(sum(grp)/len(grp), 2))
        return out

    res = [v for v in _cl(raw_res) if v > price]
    sup = [v for v in _cl(raw_sup) if v < price]
    return (round(min(res), 2) if res else None,
            round(max(sup), 2) if sup else None)


def compute_indicators(ticker: str, hist: pd.DataFrame) -> dict | None:
    """Compute breakout indicators from one year of daily OHLCV data."""
    try:
        closes  = hist["Close"].dropna()
        volumes = hist["Volume"].dropna()
        if len(closes) < 30 or len(volumes) < 21:
            return None

        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_close
        day_chg    = (last_close / prev_close - 1) * 100

        opens_s    = hist["Open"].dropna()
        open_today = float(opens_s.iloc[-1]) if len(opens_s) >= 1 else last_close
        today_ret_pct = (last_close - open_today) / open_today * 100 if open_today > 0 else 0.0

        # Bollinger Bands (20, 2)
        sma20_s = closes.rolling(20).mean()
        std20_s = closes.rolling(20).std()
        upper   = sma20_s + 2 * std20_s
        lower   = sma20_s - 2 * std20_s
        band_w  = float(upper.iloc[-1] - lower.iloc[-1])
        pct_b   = float((last_close - lower.iloc[-1]) / band_w * 100) if band_w > 0 else 50.0
        sma20_mid = float(sma20_s.iloc[-1]) if not np.isnan(sma20_s.iloc[-1]) else last_close
        bb_bwidth = (band_w / sma20_mid * 100) if sma20_mid > 0 else 0.0

        # F3 Pullback: min %B over the 10 bars BEFORE today (excludes today so we don't
        # count the current breakout bar itself as the pullback day)
        band_denom   = (upper - lower).replace(0, float("nan"))
        pct_b_series = (closes - lower) / band_denom * 100
        pct_b_min10  = (
            float(pct_b_series.iloc[-11:-1].min())
            if len(pct_b_series) >= 11 else None
        )

        # F5 Close quality: where did today's close land within today's H-L range?
        highs_s   = hist["High"].dropna()
        lows_s    = hist["Low"].dropna()
        day_range = (
            float(highs_s.iloc[-1]) - float(lows_s.iloc[-1])
            if (len(highs_s) and len(lows_s)) else 0
        )
        close_pos = (
            (last_close - float(lows_s.iloc[-1])) / day_range * 100
            if day_range > 0 else 50.0
        )

        # F2 Relative strength: 20-day return for comparison against SPY
        ret_20d = (
            float((last_close / closes.iloc[-21] - 1) * 100)
            if len(closes) >= 21 else None
        )

        # Composite scoring fields
        prior_5d_ret = (
            float((last_close / closes.iloc[-6] - 1) * 100)
            if len(closes) >= 6 else 0.0
        )
        vol_5d_avg  = float(volumes.iloc[-5:].mean())  if len(volumes) >= 5  else float(volumes.mean())
        vol_20d_avg = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
        vol_trend   = round(vol_5d_avg / vol_20d_avg, 3) if vol_20d_avg > 0 else 1.0

        # ADX(14) — trend strength; used by F8 gate
        highs_s = hist["High"].dropna()
        lows_s  = hist["Low"].dropna()
        min_len = min(len(closes), len(highs_s), len(lows_s))
        adx_val = _compute_adx(
            highs_s.values[-min_len:].astype(float),
            lows_s.values[-min_len:].astype(float),
            closes.values[-min_len:].astype(float),
            period=14,
        ) if min_len >= 30 else float("nan")

        # RSI(14) — Wilder EMA method
        delta  = closes.diff()
        gain   = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs     = gain.iloc[-1] / max(loss.iloc[-1], 1e-9)
        rsi    = float(100 - 100 / (1 + rs))

        # SMA trend
        sma20_v  = float(sma20_s.iloc[-1]) if not np.isnan(sma20_s.iloc[-1]) else 0
        sma50_v  = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else 0
        sma200_v = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else 0

        # Volume: per-ticker 90th-percentile threshold + today ratio.
        # Today's bar is partial during market hours — project it to a full-day
        # equivalent so early-session breakouts aren't suppressed by low raw volume.
        # Formula: projected = raw_today * (390 / minutes_elapsed_since_open).
        # Capped at 3× raw so pre-open or bad timestamps can't inflate infinitely.
        roll_avg   = volumes.rolling(20).mean().shift(1)
        all_ratios = (volumes / roll_avg.replace(0, float("nan"))).dropna()
        recent     = all_ratios.iloc[-252:] if len(all_ratios) >= 252 else all_ratios
        vol_90pct  = float(recent.quantile(0.90)) if len(recent) >= 20 else 1.5
        avg_vol    = float(volumes.iloc[-21:-1].mean())
        raw_today  = float(volumes.iloc[-1])
        # Time-of-day projection
        now_et = datetime.now(ET)
        market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_elapsed = (now_et - market_open_et).total_seconds() / 60
        if 5 <= minutes_elapsed < 390:
            scale = min(3.0, 390 / minutes_elapsed)   # cap at 3× to guard bad timestamps
        else:
            scale = 1.0   # pre-open, after close, or weekend — use raw volume
        proj_today = raw_today * scale
        vol_ratio  = round(proj_today / avg_vol, 2) if avg_vol > 0 else None

        # S/R: nearest swing resistance above price and support below (60 bars, excl. today)
        sr_resistance, sr_support = None, None
        if len(highs_s) >= 72 and len(lows_s) >= 72:
            sr_resistance, sr_support = _sr_quick(
                highs_s.values.astype(float), lows_s.values.astype(float), last_close
            )

        return {
            "ticker":       ticker,
            "price":        round(last_close, 2),
            "day_chg_pct":  round(day_chg, 2),
            "pct_b":        round(pct_b, 1),
            "rsi":          round(rsi, 1),
            "vol_ratio":    vol_ratio,
            "vol_90pct":    round(vol_90pct, 2),
            "vol_scale":    round(scale, 2),
            "above_sma20":  last_close > sma20_v > 0,
            "above_sma50":  last_close > sma50_v > 0,
            "above_sma200": last_close > sma200_v > 0,
            # Quality filter fields (used by apply_quality_filters)
            "ret_20d":      round(ret_20d, 2) if ret_20d is not None else None,
            "pct_b_min10":  round(pct_b_min10, 1) if pct_b_min10 is not None else None,
            "close_pos":    round(close_pos, 1),
            "open_today":    round(open_today, 2),
            "today_ret_pct": round(today_ret_pct, 2),
            # Composite score inputs (F8 + ranking)
            "adx":          round(adx_val, 1) if not np.isnan(adx_val) else None,
            "prior_5d_ret": round(prior_5d_ret, 2),
            "vol_trend":    vol_trend,
            "bb_bwidth":    round(bb_bwidth, 2),
            # S/R levels (swing-based, 60-bar lookback, excl. today)
            "sr_resistance": sr_resistance,
            "sr_support":    sr_support,
        }
    except Exception as exc:
        log.debug("Indicator error for %s: %s", ticker, exc)
        return None


def classify_signal(ind: dict, cfg: dict) -> str | None:
    pct_b     = ind["pct_b"]
    vol_ratio = ind.get("vol_ratio") or 0
    vol_90pct = ind["vol_90pct"]
    rsi       = ind["rsi"]
    bullish   = ind["above_sma20"] and ind["above_sma50"]

    # Vol spike cap: extreme volume (news/event driven) is unpredictable, not a clean breakout
    vol_ratio_max = cfg.get("vol_ratio_max", float("inf"))
    if vol_ratio > vol_ratio_max:
        return None

    if pct_b > cfg["breakout_pct_b_min"] and vol_ratio >= vol_90pct:
        return "BREAKOUT"
    if (cfg["pre_breakout_pct_b_min"] <= pct_b <= cfg["breakout_pct_b_min"]
            and rsi >= cfg["pre_breakout_rsi_min"]
            and bullish
            and vol_ratio >= vol_90pct * cfg["vol_threshold_pct"]):
        return "PRE-BREAKOUT"
    return None


# ── State machine ─────────────────────────────────────────────────────────────

def classify_state(ind: dict) -> str:
    """Map indicator snapshot to one of 7 Bollinger Band position states."""
    pct_b = ind.get("pct_b", 50)
    if pct_b > 100:  return "EXTENDED"
    if pct_b >= 95:  return "BREAKOUT"
    if pct_b >= 75:  return "PRE-BREAKOUT"
    if pct_b >= 40:  return "NEUTRAL"
    if pct_b >= 25:  return "WEAKENING"
    if pct_b >= 0:   return "PRE-BREAKDOWN"
    return "BREAKDOWN"


def _transition_alert(prev: str, new: str) -> tuple[str, str] | None:
    """Return (emoji, label) if this state transition is worth alerting, else None."""
    if new == "BREAKOUT":
        return None   # existing alert system handles breakout entry — don't double-fire

    bullish = {"BREAKOUT", "EXTENDED", "PRE-BREAKOUT"}

    if new == "PRE-BREAKOUT" and prev not in bullish:
        return ("🌱", "SETUP FORMING")

    if new == "EXTENDED":
        return ("⚠️", "EXTENDED — above upper band")

    if prev in {"EXTENDED", "BREAKOUT"} and new in {"PRE-BREAKOUT", "NEUTRAL", "WEAKENING"}:
        return ("📉", "FADING")

    if prev == "PRE-BREAKOUT" and new in {"NEUTRAL", "WEAKENING"}:
        return ("❌", "SETUP FAILED")

    if new == "BREAKDOWN":
        return ("🔴", "BREAKDOWN")

    if new == "PRE-BREAKDOWN" and prev in {"NEUTRAL", "WEAKENING"}:
        return ("⬇️", "APPROACHING BREAKDOWN")

    if prev == "BREAKDOWN" and new in {"PRE-BREAKDOWN", "WEAKENING", "NEUTRAL"}:
        return ("🔄", "RECOVERING")

    return None   # NEUTRAL→WEAKENING and similar low-value moves — skip


def _fmt_duration(since: datetime) -> str:
    mins = int((datetime.now(ET) - since).total_seconds() / 60)
    return f"{mins}m" if mins < 60 else f"{mins // 60}h{mins % 60:02d}m"


def _persist_state_transition(session_date: str, tk: str, prev_state: str, new_state: str,
                              pct_b: float, rsi: float | None, now: datetime,
                              mins_in_prev: int, close_price: float | None = None) -> None:
    """Write one state transition row to state_transitions table (fire-and-forget)."""
    try:
        con = sqlite3.connect(TAPE_DB, check_same_thread=False)
        con.execute(
            """INSERT INTO state_transitions
               (session_date, ticker, prev_state, new_state, pct_b, rsi,
                transition_time_et, mins_in_prev_state, close_price)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_date, tk, prev_state, new_state, pct_b,
             rsi, now.strftime("%H:%M:%S"), mins_in_prev, close_price),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.debug("state_transitions insert failed: %s", exc)


def process_state_transitions(
    all_inds: list[dict],
    token: str,
    chat_id: str,
    is_first_scan: bool = False,
) -> None:
    """Compare every ticker's current %B state to its stored state.

    On the first scan of the day (is_first_scan=True) we populate the ledger
    silently — no alerts, no DB writes. Only subsequent intraday transitions
    are persisted and optionally sent to Telegram.

    State ledger per ticker tracks:
      state              — current state string
      since              — when it entered current state
      prev_state         — state before current (for Stage 2 enrichment)
      pre_breakout_since — when it entered PRE-BREAKOUT (to compute conviction time)
      path               — ordered list of states visited today
      pct_b              — latest %B value
    """
    global _ticker_states
    now          = datetime.now(ET)
    session_date = now.strftime("%Y-%m-%d")

    # Collect (emoji, label, ind, prev_state, duration_str) for alertable Telegram transitions
    transitions: list[tuple[str, str, dict, str, str]] = []

    for ind in all_inds:
        tk        = ind["ticker"]
        new_state = classify_state(ind)
        rsi_val   = ind.get("rsi")

        stored = _ticker_states.get(tk)
        if stored is None:
            # First encounter — initialise ledger without alerting or persisting
            _ticker_states[tk] = {
                "state":              new_state,
                "since":              now,
                "prev_state":         None,
                "pre_breakout_since": now if new_state == "PRE-BREAKOUT" else None,
                "path":               [new_state],
                "pct_b":              ind["pct_b"],
            }
            continue

        prev_state = stored["state"]
        if new_state == prev_state:
            stored["pct_b"] = ind["pct_b"]
            continue

        # State changed — capture duration before overwriting
        mins_in_prev = int((now - stored["since"]).total_seconds() / 60)
        duration_str = _fmt_duration(stored["since"])

        # Update path (cap at 10 steps to avoid unbounded growth)
        new_path = stored["path"] + [new_state]
        if len(new_path) > 10:
            new_path = new_path[-10:]

        # Track when PRE-BREAKOUT was entered (conviction time for Stage 2)
        pre_bo_since = stored.get("pre_breakout_since")
        if new_state == "PRE-BREAKOUT":
            pre_bo_since = now
        elif new_state not in {"PRE-BREAKOUT", "BREAKOUT", "EXTENDED"}:
            pre_bo_since = None   # left the bullish zone — reset

        _ticker_states[tk] = {
            "state":              new_state,
            "since":              now,
            "prev_state":         prev_state,
            "pre_breakout_since": pre_bo_since,
            "path":               new_path,
            "pct_b":              ind["pct_b"],
        }

        if is_first_scan:
            continue   # never write or alert on initial population pass

        # Persist every transition to DB (not just alertable ones — all data is valuable)
        _persist_state_transition(
            session_date, tk, prev_state, new_state,
            ind["pct_b"], rsi_val, now, mins_in_prev,
            close_price=ind.get("price"),
        )

        result = _transition_alert(prev_state, new_state)
        if result:
            emoji, label = result
            transitions.append((emoji, label, ind, prev_state, duration_str))
            log.info("State transition: %s %s→%s (was %s)", tk, prev_state, new_state, duration_str)

    if not transitions:
        return

    # Group by label and build one consolidated Telegram message
    groups: dict[str, list] = {}
    for emoji, label, ind, prev, dur in transitions:
        key = f"{emoji} <b>{label}</b>"
        groups.setdefault(key, []).append((ind, prev, dur))

    lines = [f"🔄 <b>STATE CHANGES</b> — {now.strftime('%H:%M ET')}\n"]
    for header, items in groups.items():
        lines.append(header)
        for ind, prev, dur in items[:6]:
            tk    = ind["ticker"]
            new_s = classify_state(ind)
            rsi_s = f"RSI={ind['rsi']:.0f}" if ind.get("rsi") else ""
            dur_s = f"  (held {dur})" if prev in {"BREAKOUT", "EXTENDED", "PRE-BREAKOUT"} else ""
            lines.append(
                f"  <b>{tk}</b>  %B={ind['pct_b']:.0f}  {rsi_s}  "
                f"{prev}→{new_s}{dur_s}"
            )
        lines.append("")

    send_telegram(token, chat_id, "\n".join(lines).rstrip())


def _get_state_context(ticker: str) -> dict:
    """Read state ledger for one ticker and return Stage 2 enrichment fields.

    Called just before firing a BREAKOUT/PRE-BREAKOUT alert so the state path
    context is stamped into alert_history at the moment of alert.

    Returns dict with keys: prev_state, mins_in_pre_breakout, state_path.
    All values may be None if the ledger has no history (e.g. first scan).
    """
    stored = _ticker_states.get(ticker)
    if not stored:
        return {"prev_state": None, "mins_in_pre_breakout": None, "state_path": None}

    prev_state   = stored.get("prev_state")
    state_path   = "→".join(stored.get("path", [stored["state"]])) or None

    mins_in_pre  = None
    pre_since    = stored.get("pre_breakout_since")
    if pre_since is not None:
        mins_in_pre = int((datetime.now(ET) - pre_since).total_seconds() / 60)

    return {
        "prev_state":           prev_state,
        "mins_in_pre_breakout": mins_in_pre,
        "state_path":           state_path,
    }


def apply_quality_filters(candidates: list[dict], regime: dict, cfg: dict) -> list[dict]:
    """F2 / F5 scanner-side quality gates applied after classify_signal.

    F2 Relative strength: stock 20d return must exceed SPY 20d return.
    F5 Close quality:     today's close must be in the top 50% of the day's range (≥ midpoint).

    NOTE: F3 (pullback — %B < 50 in prior 10 bars) is intentionally excluded from real-time
    scanning. It uses complete historical daily bars and is appropriate for end-of-day backtests,
    but kills most intraday signals from stocks in sustained uptrends. It remains in the
    backtest analysis endpoint (_backtest_breakout_filtered_one) where it operates correctly.

    F5 threshold is relaxed to 50% (vs 75% in backtests) because intraday partial bars have
    a High that was set earlier in the session — close_pos of 50-75% is still a quality bar
    closing in the upper half of range, just not necessarily at the intraday high.

    Gates are skipped (pass-through) when the required data is unavailable so cold
    starts and data gaps never suppress alerts by default.
    """
    spy_ret20 = regime.get("spy_ret20")
    adx_max   = cfg.get("adx_max", 35)
    kept      = []
    n_f2 = n_f5 = n_f6 = n_f7 = n_f8 = 0

    # F7 setup: intraday RS vs SPY — only applied after 10:00 ET (opening 30 min is too noisy
    # for a same-day relative-strength read; gaps haven't settled into a real trend yet)
    spy_today_ret: float | None = None
    spy_hist = _hist_cache.get("SPY")
    if spy_hist is not None and len(spy_hist) >= 1:
        try:
            spy_open  = float(spy_hist["Open"].dropna().iloc[-1])
            spy_close = float(spy_hist["Close"].dropna().iloc[-1])
            if spy_open > 0:
                spy_today_ret = (spy_close - spy_open) / spy_open * 100
        except Exception:
            pass
    now_et_q = datetime.now(ET)
    apply_f7 = spy_today_ret is not None and now_et_q.hour >= 10

    for ind in candidates:
        tk = ind["ticker"]

        # F2: relative strength vs SPY — stock 20d return must beat SPY 20d return
        ret_20d = ind.get("ret_20d")
        if spy_ret20 is not None and ret_20d is not None and ret_20d < spy_ret20:
            log.info("F2 relative-strength gate dropped %s: %.1f%% < SPY %.1f%%",
                     tk, ret_20d, spy_ret20)
            n_f2 += 1
            continue

        # F5: close quality — close in upper half of today's H-L range (≥ midpoint)
        # Threshold is 50% not 75% because intraday partial bars have dynamic H/L
        close_pos = ind.get("close_pos", 50.0)
        if close_pos < 50:
            log.info("F5 close-quality gate dropped %s: close_pos=%.1f%% (< 50%% of range)",
                     tk, close_pos)
            n_f5 += 1
            continue

        # F6: gap-and-reverse filter — price must hold at/above today's open.
        # BREAKOUT: strict — any fade below the open means the move already failed.
        # PRE-BREAKOUT: only enforced in the first half-hour (09:30-09:59 ET), where
        # opening-print gaps that immediately reverse are the dominant failure mode.
        # Intraday-confirmed signals skip F6: the intraday %B already confirms
        # the move is occurring within the session, not a pre-market gap artifact.
        today_ret = ind.get("today_ret_pct", 0.0)
        sig_type  = ind.get("signal", "")
        if ind.get("intraday_confirmed"):
            pass  # F6 n/a — intraday move confirmed by 15m %B, not a gap artifact
        elif sig_type == "BREAKOUT" and today_ret < 0:
            log.info("F6 gap-reverse gate dropped %s (BREAKOUT): today_ret=%.2f%%", tk, today_ret)
            n_f6 += 1
            continue
        elif sig_type == "PRE-BREAKOUT" and now_et_q.hour == 9 and today_ret < -0.5:
            log.info("F6 gap-reverse gate dropped %s (PRE-BO, opening): today_ret=%.2f%%", tk, today_ret)
            n_f6 += 1
            continue

        # F7: intraday relative strength — stock must be outperforming SPY today
        if apply_f7:
            ticker_today_ret = ind.get("today_ret_pct")
            if ticker_today_ret is not None and ticker_today_ret < spy_today_ret:
                log.info("F7 intraday-RS gate dropped %s: %.2f%% < SPY %.2f%%",
                         tk, ticker_today_ret, spy_today_ret)
                n_f7 += 1
                continue

        # F8: ADX trend-strength gate — high ADX means the stock is already deep in a
        # trend and overextended; breakouts from ranging/transitioning stocks (low ADX)
        # outperform because there is more room to run. Backtest confirmed: ADX < 25
        # avg +0.14% vs ADX top-quartile (>35) avg +0.09%.
        adx_val = ind.get("adx")
        if adx_val is not None and adx_val > adx_max:
            log.info("F8 ADX gate dropped %s: ADX=%.1f > %.1f (overextended trend)",
                     tk, adx_val, adx_max)
            n_f8 += 1
            continue

        kept.append(ind)

    if n_f2 or n_f5 or n_f6 or n_f7 or n_f8:
        log.info(
            "Quality filters: dropped %d (F2 rel-str), %d (F5 close-qual), "
            "%d (F6 gap-rev), %d (F7 intraday-RS), %d (F8 ADX)",
            n_f2, n_f5, n_f6, n_f7, n_f8,
        )
    return kept


# ── Composite scoring ────────────────────────────────────────────────────────

def _pct_rank(val: float | None, pool: list[float]) -> float:
    """Percentile rank of val within pool (0–100). Returns 50 when val is None."""
    if val is None or not pool:
        return 50.0
    below = sum(1 for v in pool if v < val)
    return below / len(pool) * 100


def compute_composite_scores(signals: list[dict], all_inds: list[dict]) -> None:
    """Compute composite signal quality score (0–100) in-place on each signal dict.

    Score = 0.30 × vol_ratio rank
          + 0.25 × prior_5d_ret rank    (momentum entering signal)
          + 0.20 × close_pos rank       (closing near day's high)
          + 0.15 × vol_trend rank       (sustained accumulation vs one-day spike)
          + 0.10 × bb_bwidth rank       (expanding bands = real move in progress)

    Ranks are percentile within the full evaluated universe (all_inds), not just
    the signals — so a score of 75 means "better than 75% of all stocks scanned today
    on this composite measure", giving context-aware signal strength.
    """
    if not signals or not all_inds:
        return

    vr_pool  = [d["vol_ratio"]    for d in all_inds if d.get("vol_ratio")    is not None]
    ret_pool = [d["prior_5d_ret"] for d in all_inds if d.get("prior_5d_ret") is not None]
    cp_pool  = [d["close_pos"]    for d in all_inds if d.get("close_pos")    is not None]
    vt_pool  = [d["vol_trend"]    for d in all_inds if d.get("vol_trend")    is not None]
    bw_pool  = [d["bb_bwidth"]    for d in all_inds if d.get("bb_bwidth")    is not None]

    for sig in signals:
        score = (
            0.30 * _pct_rank(sig.get("vol_ratio"),    vr_pool)
          + 0.25 * _pct_rank(sig.get("prior_5d_ret"), ret_pool)
          + 0.20 * _pct_rank(sig.get("close_pos"),    cp_pool)
          + 0.15 * _pct_rank(sig.get("vol_trend"),    vt_pool)
          + 0.10 * _pct_rank(sig.get("bb_bwidth"),    bw_pool)
        )
        sig["composite_score"] = round(score, 1)


# ── Scan (reads from _hist_cache — no network calls in hot path) ──────────────

def scan_universe(tickers: list[str], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Compute indicators for all tickers from _hist_cache; return (signals, all_inds).

    No yfinance/network calls here — all data comes from the pre-cached history
    updated by refresh_hist_cache() just before this is called.

    Returns both the filtered signal list AND the full universe indicator list so
    the main loop can run state-transition tracking across all 112 tickers.
    """
    found:    list[dict] = []
    all_inds: list[dict] = []
    n_skip = 0

    for tk in tickers:
        hist = _hist_cache.get(tk)
        if hist is None or len(hist) < 30:
            n_skip += 1
            continue
        ind = compute_indicators(tk, hist)
        if ind is None:
            n_skip += 1
            continue
        all_inds.append(ind)
        sig = classify_signal(ind, cfg)
        if sig:
            ind["signal"] = sig
            found.append(ind)

    # ── Dual-%B check: catch intraday momentum invisible on daily Bollinger Bands ──
    # Tickers with daily %B 50-94 (above midband, below BREAKOUT threshold), RSI >= 55,
    # and above SMA20 get their 15-min intraday %B computed. If intraday %B >= 75 the
    # ticker qualifies as PRE-BREAKOUT; if >= 95 it qualifies as BREAKOUT — even though
    # the daily BB didn't trigger. Marked intraday_confirmed=True so fmt_alert and
    # F6 handling can show/skip appropriately.
    already_signaled = {s["ticker"] for s in found}
    intra_check = [
        d["ticker"] for d in all_inds
        if 50 <= d.get("pct_b", 0) < cfg["breakout_pct_b_min"]
        and d.get("rsi", 0) >= 55
        and d.get("above_sma20", False)
        and d["ticker"] not in already_signaled
    ]
    if intra_check and is_market_open():
        log.info("Intraday dual-%%B check: %d candidate tickers", len(intra_check))
        intra_pct_b = fetch_intraday_pct_b(intra_check)
        ind_by_tk   = {d["ticker"]: d for d in all_inds}
        for tk, ipb in intra_pct_b.items():
            d = ind_by_tk.get(tk)
            if d is None:
                continue
            daily_pb = d.get("pct_b", 0)
            if ipb >= cfg["breakout_pct_b_min"] and daily_pb >= 50:
                d["signal"]             = "BREAKOUT"
                d["intraday_pct_b"]     = ipb
                d["intraday_confirmed"] = True
                found.append(d)
                log.info("  Intraday BREAKOUT: %s  daily=%.1f%%B  15m=%.1f%%B", tk, daily_pb, ipb)
            elif ipb >= cfg["pre_breakout_pct_b_min"] and daily_pb >= 50:
                d["signal"]             = "PRE-BREAKOUT"
                d["intraday_pct_b"]     = ipb
                d["intraday_confirmed"] = True
                found.append(d)
                log.info("  Intraday PRE-BREAKOUT: %s  daily=%.1f%%B  15m=%.1f%%B", tk, daily_pb, ipb)

    # Composite score: rank each signal against the full evaluated universe
    compute_composite_scores(found, all_inds)

    # Sort by composite score descending so highest-conviction signals alert first
    found.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    log.info("  Evaluated %d/%d tickers (%d skipped — no cached data)",
             len(all_inds), len(tickers), n_skip)

    if all_inds and not found:
        near = sorted(all_inds, key=lambda x: x["pct_b"], reverse=True)[:5]
        lines = [f"  {d['ticker']:6s}  %B={d['pct_b']:5.1f}  RSI={d['rsi']:4.1f}"
                 f"  vol={d['vol_ratio'] or 0:.2f}x (thr={d['vol_90pct']:.2f}x)"
                 f"  scale={d.get('vol_scale',1.0):.1f}x"
                 f"  {'▲20/50' if d['above_sma20'] and d['above_sma50'] else '—trend'}"
                 for d in near]
        log.info("Top-5 nearest to breakout:\n" + "\n".join(lines))

    return found, all_inds


# ── Market hours ──────────────────────────────────────────────────────────────
ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


def seconds_to_next_open() -> int:
    """Seconds until next weekday market open (ET 9:30 AM)."""
    now = datetime.now(ET)
    # Advance to next weekday
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= candidate:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return max(0, int((candidate - now).total_seconds()))


def run_premarket_scan(tickers: list[str], cfg: dict, token: str, chat_id: str) -> None:
    """At 9:10-9:15 ET, simulate %B using live pre-market prices over yesterday's
    Bollinger Bands and send a heads-up 15 minutes before the open — catches gap-up
    setups before the opening-print alert fires (and before any reversal happens)."""
    try:
        log.info("Pre-market scan: fetching pre-market prices for %d tickers…", len(tickers))
        raw = yf.download(
            tickers, period="1d", interval="1m",
            prepost=True, progress=False, auto_adjust=True, threads=True,
        )
        if raw.empty:
            log.info("Pre-market scan: no data yet from yfinance")
            return

        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        else:
            close_df = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw.columns else pd.DataFrame()
        if close_df.empty:
            return

        cutoff_et = datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)
        premarket_prices: dict[str, float] = {}
        for tk in tickers:
            if tk not in close_df.columns:
                continue
            try:
                col = close_df[tk].dropna()
                col.index = col.index.tz_convert(ET)
                pre = col[col.index < cutoff_et]
                if not pre.empty:
                    premarket_prices[tk] = float(pre.iloc[-1])
            except Exception:
                continue

        if not premarket_prices:
            log.info("Pre-market scan: no pre-market prints available yet")
            return

        watch_threshold = cfg.get("pre_breakout_pct_b_min", 75)
        breakout_threshold = cfg.get("breakout_pct_b_min", 95)
        watches: list[dict] = []
        for tk, pm_price in premarket_prices.items():
            hist = _hist_cache.get(tk)
            if hist is None or len(hist) < 20:
                continue
            try:
                closes = hist["Close"].dropna()
                if len(closes) < 20:
                    continue
                sim_closes = pd.concat([closes.iloc[-19:], pd.Series([pm_price])], ignore_index=True)
                sma20 = float(sim_closes.rolling(20).mean().iloc[-1])
                std20 = float(sim_closes.rolling(20).std().iloc[-1])
                band_w = (sma20 + 2 * std20) - (sma20 - 2 * std20)
                if band_w <= 0:
                    continue
                sim_pct_b = (pm_price - (sma20 - 2 * std20)) / band_w * 100
                if sim_pct_b >= watch_threshold:
                    prev_close = float(closes.iloc[-1])
                    gap_pct = (pm_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
                    watches.append({
                        "ticker": tk, "pm_price": round(pm_price, 2),
                        "sim_pct_b": round(sim_pct_b, 1), "gap_pct": round(gap_pct, 2),
                    })
            except Exception:
                continue

        if not watches:
            log.info("Pre-market scan: no tickers in watch zone (sim %%B ≥ %d%%)", watch_threshold)
            return

        watches.sort(key=lambda x: x["sim_pct_b"], reverse=True)
        log.info("Pre-market scan: %d tickers approaching breakout zone", len(watches))

        lines = ["⚡ <b>PRE-MARKET WATCH</b> — 9:15 ET\n<i>Approaching breakout zone, confirm at open:</i>\n"]
        for w in watches[:10]:
            tag = "🚨" if w["sim_pct_b"] >= breakout_threshold else "⚡"
            gap_str = f"+{w['gap_pct']:.1f}%" if w["gap_pct"] >= 0 else f"{w['gap_pct']:.1f}%"
            lines.append(f"{tag} <b>{w['ticker']}</b> ${w['pm_price']} ({gap_str} from close) — sim %B≈{w['sim_pct_b']:.0f}%")
        send_telegram(token, chat_id, "\n".join(lines))
    except Exception as exc:
        log.warning("Pre-market scan failed: %s", exc)


def run_eod_summary(
    alerted_today: dict[str, str],
    cfg: dict,
    token: str,
    chat_id: str,
    regime: dict | None = None,
) -> None:
    """Fire once at ~4:02 PM ET — backward-looking daily recap sent to Telegram.

    Shows what happened today: alerts by tier, SPY performance, market regime.
    Complements the forward-looking EOD watchlist that fires at 4:15 PM.
    """
    try:
        today_str = datetime.now(ET).strftime("%a %b %d")

        # SPY return today from hist cache
        spy_ret: str = ""
        spy_hist = _hist_cache.get("SPY")
        if spy_hist is not None and len(spy_hist) >= 2:
            closes = spy_hist["Close"].dropna()
            if len(closes) >= 2:
                ret = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
                arrow = "📈" if ret >= 0 else "📉"
                spy_ret = f"{arrow} SPY {ret:+.2f}%"

        # Regime line
        regime_label = ""
        if regime:
            r = regime.get("regime", "")
            r_emoji = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}.get(r, "⚪")
            regime_label = f"{r_emoji} Regime: {r}"

        # Split alerts by tier
        bo_tickers  = [tk for tk, sig in alerted_today.items() if sig == "BREAKOUT"]
        pre_tickers = [tk for tk, sig in alerted_today.items() if sig == "PRE-BREAKOUT"]
        total = len(alerted_today)

        lines = [f"📊 <b>EOD Summary — {today_str}</b>"]
        if spy_ret:
            lines.append(spy_ret)
        if regime_label:
            lines.append(regime_label)
        lines.append("")

        if total == 0:
            lines.append("😶 Quiet session — 0 alerts fired today")
        else:
            if bo_tickers:
                lines.append(f"🚨 <b>BREAKOUT</b> ({len(bo_tickers)}): {', '.join(bo_tickers)}")
            if pre_tickers:
                lines.append(f"⚡ <b>PRE-BREAKOUT</b> ({len(pre_tickers)}): {', '.join(pre_tickers)}")
            lines.append(f"\n<i>{total} alert{'s' if total != 1 else ''} today"
                         + (f" — {len(bo_tickers)} breakout, {len(pre_tickers)} pre-breakout" if bo_tickers and pre_tickers else "")
                         + "</i>")

        # ── #11: Same-day alert-to-close performance ─────────────────────────
        # Query alert_history for first alert price per ticker today, then
        # compare with _hist_cache EOD close to show intraday return.
        if total > 0:
            try:
                _perf_date = datetime.now(ET).strftime("%Y-%m-%d")
                _pcon = sqlite3.connect(TAPE_DB, check_same_thread=False)
                _perf_rows = _pcon.execute(
                    "SELECT ticker, price FROM alert_history "
                    "WHERE session_date = ? AND price IS NOT NULL ORDER BY fired_at ASC",
                    (_perf_date,),
                ).fetchall()
                _pcon.close()
                _alert_prices: dict[str, float] = {}
                for _ptk, _ppx in _perf_rows:
                    if _ptk not in _alert_prices and _ppx:
                        _alert_prices[_ptk] = float(_ppx)
                _rets, _wins, _losses = [], [], []
                for _ptk, _entry in _alert_prices.items():
                    _df = _hist_cache.get(_ptk)
                    if _df is None or _df.empty or _entry <= 0:
                        continue
                    _eod_px = float(_df["Close"].iloc[-1])
                    _ret = (_eod_px / _entry - 1) * 100
                    _rets.append(_ret)
                    (_wins if _ret >= 0 else _losses).append(_ptk)
                if _rets:
                    _avg = sum(_rets) / len(_rets)
                    _wr  = len(_wins) / len(_rets) * 100
                    lines.append(
                        f"\n📈 <b>Today's alert performance</b> ({len(_rets)} tracked): "
                        f"{len(_wins)}W/{len(_losses)}L | "
                        f"Win rate: {_wr:.0f}% | Avg return: {_avg:+.2f}%"
                    )
            except Exception as _pe:
                log.debug("Same-day performance calc failed: %s", _pe)

        lines.append("\n🌙 EOD watchlist incoming at 4:15 ET — setups for tomorrow.")
        send_telegram(token, chat_id, "\n".join(lines))
        log.info("EOD summary sent: %d alerts today (%d BO, %d PRE)",
                 total, len(bo_tickers), len(pre_tickers))
    except Exception as exc:
        log.warning("EOD summary failed: %s", exc)


def run_eod_watchlist_scan(tickers: list[str], cfg: dict, token: str, chat_id: str,
                           regime: dict | None = None) -> None:
    """Fire once at ~4:15 PM ET after market close.

    Uses today's complete daily close already in _hist_cache — no extra download.
    Identifies stocks approaching breakout levels for TOMORROW using relaxed thresholds
    so traders get overnight preparation time instead of the 15-minute pre-market heads-up.

    Tiers:
      BREAKOUT SETUP  pct_b >= 90  — already in breakout zone, watch for follow-through
      WATCH           70 <= pct_b < 90 — building toward the zone, worth monitoring

    Quality gates applied:
      F2  20d return must beat SPY (relative strength)
      F8  ADX < adx_max (not already overextended trend)
      EOD close quality: close_pos >= 45 (closed in upper half of today's range)
      RSI >= 50, above SMA20 and SMA50
    """
    try:
        log.info("EOD watchlist scan: evaluating %d tickers from today's close…", len(tickers))

        eod_pct_b_min   = cfg.get("eod_pct_b_min",  70.0)
        eod_rsi_min     = cfg.get("eod_rsi_min",     50.0)
        eod_close_pos   = cfg.get("eod_close_pos_min", 45.0)
        breakout_min    = cfg.get("breakout_pct_b_min", 95.0)
        setup_min       = 90.0   # pct_b >= 90 = "BREAKOUT SETUP" tier
        adx_max         = cfg.get("adx_max", 35)
        spy_ret20       = (regime or {}).get("spy_ret20")

        candidates: list[dict] = []
        all_inds:   list[dict] = []

        for tk in tickers:
            hist = _hist_cache.get(tk)
            if hist is None or len(hist) < 30:
                continue
            ind = compute_indicators(tk, hist)
            if ind is None:
                continue
            all_inds.append(ind)

            # Basic EOD entry criteria
            if ind["pct_b"] < eod_pct_b_min:
                continue
            rsi_val = ind.get("rsi") or 0
            if rsi_val < eod_rsi_min:
                continue
            if not (ind["above_sma20"] and ind["above_sma50"]):
                continue
            if ind.get("close_pos", 50) < eod_close_pos:
                continue

            # F2: 20d relative strength vs SPY
            ret_20d = ind.get("ret_20d")
            if spy_ret20 is not None and ret_20d is not None and ret_20d < spy_ret20:
                continue

            # F8: ADX — avoid already-overextended trends
            adx_val = ind.get("adx")
            if adx_val is not None and adx_val > adx_max:
                continue

            # Assign EOD tier
            if ind["pct_b"] >= breakout_min:
                ind["eod_signal"] = "BREAKOUT SETUP"
            elif ind["pct_b"] >= setup_min:
                ind["eod_signal"] = "BREAKOUT SETUP"
            else:
                ind["eod_signal"] = "WATCH"

            candidates.append(ind)

        if not candidates:
            log.info("EOD watchlist scan: no tickers passed filters — no message sent")
            return

        # Composite score ranked against full universe (all_inds)
        compute_composite_scores(candidates, all_inds)
        candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        setups = [c for c in candidates if c["eod_signal"] == "BREAKOUT SETUP"]
        watches = [c for c in candidates if c["eod_signal"] == "WATCH"]

        log.info("EOD watchlist scan: %d BREAKOUT SETUP, %d WATCH tickers",
                 len(setups), len(watches))

        today_str = datetime.now(ET).strftime("%a %b %d")
        lines = [f"🌙 <b>WATCH TOMORROW</b> — {today_str}\n"
                 f"<i>Based on today's close — {len(candidates)} setups passed filters</i>\n"]

        if setups:
            lines.append("🚨 <b>Breakout zone (act fast at open):</b>")
            for ind in setups[:6]:
                score   = ind.get("composite_score", 0)
                filled  = int(round(score / 10))
                bar     = "█" * filled + "░" * (10 - filled)
                adx_s   = f"ADX={ind['adx']:.0f}" if ind.get("adx") is not None else ""
                p5_s    = f"5d={ind['prior_5d_ret']:+.1f}%" if ind.get("prior_5d_ret") is not None else ""
                detail  = "  ".join(x for x in [adx_s, p5_s] if x)
                lines.append(
                    f"  <b>{ind['ticker']}</b>  %B={ind['pct_b']:.0f}  RSI={ind['rsi']:.0f}"
                    f"  score={score:.0f}  [{bar}]"
                    + (f"\n    {detail}" if detail else "")
                )

        if watches:
            lines.append("\n⚡ <b>Approaching zone (watch for follow-through):</b>")
            for ind in watches[:8]:
                score   = ind.get("composite_score", 0)
                adx_s   = f"ADX={ind['adx']:.0f}" if ind.get("adx") is not None else ""
                p5_s    = f"5d={ind['prior_5d_ret']:+.1f}%" if ind.get("prior_5d_ret") is not None else ""
                detail  = "  ".join(x for x in [adx_s, p5_s] if x)
                lines.append(
                    f"  <b>{ind['ticker']}</b>  %B={ind['pct_b']:.0f}  RSI={ind['rsi']:.0f}"
                    f"  score={score:.0f}"
                    + (f"  {detail}" if detail else "")
                )

        lines.append(f"\nFilters: F2 rel-str | F8 ADX max {adx_max} | close top 45% of range")
        lines.append("Pre-market update at 9:15 ET tomorrow.")

        send_telegram(token, chat_id, "\n".join(lines))
        log.info("EOD watchlist sent: %d tickers total", len(candidates))

    except Exception as exc:
        log.warning("EOD watchlist scan failed: %s", exc)


# ── Main loop ─────────────────────────────────────────────────────────────────

_SINGLETON_PORT = 47892  # Dedicated loopback port — only one scanner can bind it
_singleton_sock = None  # Kept alive for the lifetime of the process


def _acquire_singleton() -> None:
    """Bind a loopback port as a process-scoped mutex.

    The OS guarantees only one process can hold the bind at a time, with no
    race window and automatic release on crash or clean exit.
    """
    global _singleton_sock
    import socket as _socket
    try:
        _singleton_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        _singleton_sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 0)
        _singleton_sock.bind(("127.0.0.1", _SINGLETON_PORT))
    except OSError:
        log.warning(
            "Another scanner instance is already running (port %d bound). Exiting.",
            _SINGLETON_PORT,
        )
        raise SystemExit(0)


def _release_singleton() -> None:
    global _singleton_sock
    try:
        if _singleton_sock is not None:
            _singleton_sock.close()
            _singleton_sock = None
    except Exception:
        pass


def main():
    _acquire_singleton()
    try:
        _main_loop()
    finally:
        _release_singleton()


def _main_loop():
    cfg = load_config()

    token   = cfg["telegram_token"]
    chat_id = cfg["telegram_chat_id"]
    interval = cfg["scan_interval_minutes"] * 60
    alert_on = set(cfg["alert_on"])

    if not token or not chat_id:
        log.error(
            "Telegram not configured!\n"
            "  Edit %s and add:\n"
            "    telegram_token:   your bot token from @BotFather\n"
            "    telegram_chat_id: your chat/group ID\n"
            "  OR set env vars TELEGRAM_TOKEN and TELEGRAM_CHAT_ID",
            CONFIG_PATH,
        )
        return

    log.info("Breakout Scanner starting…")
    log.info("  Universe:        %d curated tickers (high-liquidity options names)", len(CURATED_TICKERS))
    log.info("  Alerts on:       %s", ", ".join(alert_on))
    log.info("  Scan interval:   %d min", cfg["scan_interval_minutes"])
    log.info("  IBKR hist:       %s", "enabled" if cfg.get("use_ibkr_hist") else "disabled (yfinance only)")
    log.info("  Config:          %s", CONFIG_PATH)
    log.info("  Volume mode:     intraday projection (scales partial-day volume to full-day "
             "equivalent so morning breakouts aren't suppressed)")

    backend_url = cfg.get("backend_url", "")
    use_ibkr    = bool(cfg.get("use_ibkr_hist", True))
    tickers     = CURATED_TICKERS

    # ── Pre-load 1 year of daily history for all tickers at startup ───────────
    # This runs once (~30s via IBKR or ~20s via yfinance).
    # Every subsequent cycle only refreshes 5 days.
    load_hist_cache(tickers, batch_size=cfg.get("batch_size", 50), backend_url=backend_url)

    send_telegram(
        token, chat_id,
        "🤖 <b>Breakout Scanner started</b>\n"
        f"Universe: {len(tickers)} curated tickers (fast in-memory scan)\n"
        f"Interval: every {cfg['scan_interval_minutes']} min | Alerts: {', '.join(alert_on)}",
    )

    alerted_today: dict[str, str] = {}   # ticker → last signal type alerted
    last_scan_day: date | None = None
    premarket_sent_day:  str = ""
    eod_summary_sent_day: str = ""
    eod_watch_sent_day:  str = ""
    state_first_scan:    bool = True     # suppress transition alerts on first scan of the day

    # ── #1: Restore alerted_today from today's alert_history ─────────────────
    # Prevents duplicate alerts and missing EOD summary if the scanner restarts
    # mid-session.  Latest signal type per ticker wins (ORDER BY fired_at ASC).
    _today_iso = date.today().isoformat()
    try:
        _con = sqlite3.connect(TAPE_DB, check_same_thread=False)
        _rows = _con.execute(
            "SELECT ticker, signal_type FROM alert_history "
            "WHERE session_date = ? ORDER BY fired_at ASC",
            (_today_iso,),
        ).fetchall()
        _con.close()
        for _tk, _sig in _rows:
            alerted_today[_tk] = _sig
        if _rows:
            log.info("#1 Restored alerted_today: %d ticker(s) from today's alert_history", len(alerted_today))
    except Exception as _exc:
        log.warning("Could not restore alerted_today from DB: %s", _exc)

    # ── #13: Restore _ticker_states from today's JSON if scanner restarted ───
    # Avoids silent first-scan suppression and preserves state machine context.
    _states_path = os.path.join(SCRIPT_DIR, f"ticker_states_{_today_iso}.json")
    if os.path.exists(_states_path):
        try:
            with open(_states_path, "r", encoding="utf-8") as _sf:
                _ticker_states.update(json.load(_sf))
            state_first_scan = False
            log.info("#13 Restored _ticker_states: %d tickers from %s", len(_ticker_states), _states_path)
        except Exception as _exc:
            log.warning("Could not restore _ticker_states: %s", _exc)

    while True:
        # Manual trigger: `touch backend/trigger_eod_watchlist` fires EOD watchlist immediately
        _trigger_path = os.path.join(SCRIPT_DIR, "trigger_eod_watchlist")
        if os.path.exists(_trigger_path):
            try:
                os.remove(_trigger_path)
            except OSError:
                pass
            log.info("Manual trigger: running EOD watchlist scan now")
            _regime = fetch_regime(cfg.get("backend_url", ""))
            run_eod_watchlist_scan(tickers, cfg, token, chat_id, regime=_regime)

        if not is_market_open():
            _now_et = datetime.now(ET)
            _today_str = _now_et.strftime("%Y-%m-%d")

            # Pre-market scan at 9:10-9:15 ET (uses live pre-market prices)
            if (_now_et.weekday() < 5
                    and _now_et.hour == 9 and 10 <= _now_et.minute <= 15
                    and premarket_sent_day != _today_str):
                premarket_sent_day = _today_str
                # ── #12: Force-refresh regime before premarket scan ───────────
                # Overnight SPY gaps won't be reflected in a stale 4-hour cache.
                if backend_url:
                    try:
                        requests.post(
                            f"{backend_url.rstrip('/')}/market/regime/refresh",
                            timeout=20,
                        )
                        log.info("#12 Regime cache refreshed before premarket scan")
                    except Exception as _re:
                        log.warning("Regime refresh failed: %s", _re)
                run_premarket_scan(tickers, cfg, token, chat_id)

            # EOD recap at 4:02-4:12 ET — backward-looking daily summary
            if (_now_et.weekday() < 5
                    and _now_et.hour == 16 and 2 <= _now_et.minute <= 12
                    and eod_summary_sent_day != _today_str):
                eod_summary_sent_day = _today_str
                _regime = fetch_regime(cfg.get("backend_url", ""))
                run_eod_summary(alerted_today, cfg, token, chat_id, regime=_regime)

            # EOD watchlist at 4:15-4:30 ET (uses today's complete close from _hist_cache)
            if (_now_et.weekday() < 5
                    and _now_et.hour == 16 and 15 <= _now_et.minute <= 30
                    and eod_watch_sent_day != _today_str):
                eod_watch_sent_day = _today_str
                _regime = fetch_regime(cfg.get("backend_url", ""))
                run_eod_watchlist_scan(tickers, cfg, token, chat_id, regime=_regime)
            secs = seconds_to_next_open()
            if secs == 0:
                # Open right now — break out of closed loop immediately
                continue
            if secs <= 90:
                # Within 90 seconds of open — poll every 10s so we catch 9:30:00 precisely
                sleep_secs = min(secs, 10)
            elif secs <= 600:
                # Within 10 minutes of open — wake every 30s to stay close to the bell
                sleep_secs = 30
            elif secs <= 3600:
                # Within 1 hour — wake every 60s
                sleep_secs = 60
            else:
                # Far from open — sleep up to 5 minutes
                sleep_secs = min(secs, 300)
            log.info("Market closed — %dm to open, sleeping %ds", secs // 60, sleep_secs)
            time.sleep(sleep_secs)
            continue

        # Reload config each cycle — picks up chat_id / token changes without restart
        cfg         = load_config()
        token       = cfg["telegram_token"]
        chat_id     = cfg["telegram_chat_id"]
        alert_on    = set(cfg["alert_on"])
        interval    = cfg["scan_interval_minutes"] * 60
        backend_url = cfg.get("backend_url", "")
        use_ibkr    = bool(cfg.get("use_ibkr_hist", True))

        today = date.today()
        if today != last_scan_day:
            alerted_today = {}
            last_scan_day = today
            state_first_scan = True
            _ticker_states.clear()   # reset intraday state ledger for the new session
            send_telegram(
                token, chat_id,
                f"🌅 <b>Market open</b> — {today.strftime('%a %b %d')}\n"
                f"Scanning {len(tickers)} tickers every {cfg['scan_interval_minutes']} min.",
            )
            log.info("New day — scanning %d tickers", len(tickers))

        scan_start = time.monotonic()

        # ── Refresh cache with today's latest bars (fast: 5d data only) ──────
        refresh_hist_cache(tickers, backend_url=backend_url,
                           use_ibkr=use_ibkr, batch_size=cfg.get("batch_size", 50))

        # ── F1 Regime gate (live VIX + SPY SMA-200 from backend) ─────────────
        regime     = fetch_regime(backend_url)
        regime_ok  = regime.get("regime_ok", True)
        if not regime_ok:
            log.info("REGIME GATE ACTIVE: %s — Telegram suppressed this cycle",
                     regime.get("reason", "bad regime"))

        log.info("Starting scan (%d tickers, in-memory)…", len(tickers))

        raw_candidates, all_inds = scan_universe(tickers, cfg)

        # ── State machine: detect and alert on %B state transitions ──────────
        if regime_ok:
            process_state_transitions(all_inds, token, chat_id,
                                      is_first_scan=state_first_scan)
        state_first_scan = False

        # ── F2 / F3 / F5 Quality filters (scanner-side) ──────────────────────
        candidates = apply_quality_filters(raw_candidates, regime, cfg)
        n_dropped  = len(raw_candidates) - len(candidates)
        if n_dropped:
            log.info("Quality filters dropped %d/%d raw signals", n_dropped, len(raw_candidates))

        breakouts = [c for c in candidates if c["signal"] == "BREAKOUT"]
        pre_bo    = [c for c in candidates if c["signal"] == "PRE-BREAKOUT"]

        refresh_elapsed = time.monotonic() - scan_start
        log.info(
            "Scan complete in %.1fs — %d BREAKOUT, %d PRE-BREAKOUT (after quality filters)",
            refresh_elapsed, len(breakouts), len(pre_bo),
        )

        # ── Combined backend sync + contingent Telegram ───────────────────────
        # BREAKOUT processed first (higher priority); PRE-BREAKOUT second.
        # For each signal:
        #   1. POST to backend (always — keeps watchlist current for refreshes)
        #   2. Backend gate may return action='blocked' for new entries → skip Telegram
        #   3. Telegram fires only when: backend accepted (action='added' or 'no_backend')
        #      AND signal is new or escalated today (dedup)
        n_alerted_this_cycle = 0
        n_f9_dropped         = 0
        for sig_type in ["BREAKOUT", "PRE-BREAKOUT"]:
            bucket = breakouts if sig_type == "BREAKOUT" else pre_bo

            for ind in sorted(bucket, key=lambda x: x.get("composite_score", 0), reverse=True):
                tk = ind["ticker"]

                # Fetch tape once per ticker (used in both POST and Telegram)
                tape = fetch_tape_sentiment(backend_url, tk)

                # Attach state context so backend can persist it in alert_history
                state_ctx = _get_state_context(tk)
                ind.update(state_ctx)

                # Skip if regime gate is active (F1) or signal type not configured
                if not regime_ok:
                    continue
                if sig_type not in alert_on:
                    continue

                # Skip after 15:45 ET — post-close signals fire on yesterday's cached
                # technicals; no actionable trade is possible.
                _now_for_alert = datetime.now(ET)
                if _now_for_alert.hour == 15 and _now_for_alert.minute >= 45:
                    log.info("Post-close suppression: skipping Telegram for %s %s at %s ET",
                             sig_type, tk, _now_for_alert.strftime("%H:%M"))
                    continue
                if _now_for_alert.hour >= 16:
                    log.info("Post-close suppression: skipping Telegram for %s %s at %s ET",
                             sig_type, tk, _now_for_alert.strftime("%H:%M"))
                    continue

                # F9: Path quality gate — BREAKOUT must approach via PRE-BREAKOUT.
                # Backtest: instant jumps (0d in PRE-BO) → 52.9% win rate, -0.03% avg 5d.
                # 2-step path (via PRE-BO)              → 62.0% win rate, +1.31% avg 5d.
                # Pass-through when prev_state is None (no intraday history yet — first scan).
                if (sig_type == "BREAKOUT"
                        and cfg.get("require_2step_breakout", True)
                        and ind.get("prev_state") is not None
                        and ind.get("prev_state") != "PRE-BREAKOUT"):
                    log.info("F9 path-gate dropped %s: BREAKOUT jumped from %s (not via PRE-BREAKOUT)",
                             tk, ind["prev_state"])
                    n_f9_dropped += 1
                    continue

                # Sync to backend watchlist only after all local filters pass (F1–F9).
                # This keeps watchlist consistent with Telegram and stock-trader triggers.
                response = post_to_backend_with_tape(backend_url, ind, sig_type, tape)
                action   = response.get("action", "no_backend")

                if action == "blocked":
                    log.info("Backend gate blocked %s %s: %s",
                             sig_type, tk, response.get("reason", "—"))
                    continue

                # Dedup: only alert on new or escalated signals
                prev = alerted_today.get(tk)
                if prev == sig_type:
                    continue  # already alerted today at this level
                if prev == "BREAKOUT" and sig_type == "PRE-BREAKOUT":
                    continue  # never downgrade BREAKOUT → PRE-BREAKOUT

                # Fire Telegram
                alerted_today[tk] = sig_type
                n_alerted_this_cycle += 1
                msg = fmt_alert(ind, sig_type, tape)
                # Monday caution: backtest shows Mon avg return -0.10% vs Fri +0.24%
                if datetime.now(ET).weekday() == 0:
                    msg += "\n⚠️ <i>Monday signal — historically weakest day (backtest avg -0.10%)</i>"
                log.info("Alerting: %s %s (tape=%s, backend=%s)",
                         sig_type, tk, tape.get("label") if tape else "no data", action)
                send_telegram(token, chat_id, msg)
                time.sleep(0.3)   # Telegram rate limit: ~30 msg/s

                # Direct stock-trader + day-trader triggers (fast path — bypasses 5-min AT poll)
                # Only fires on first alert per ticker per day, same gate as Telegram.
                if sig_type == "BREAKOUT" and backend_url:
                    _signal_payload = {
                        "ticker":          tk,
                        "price":           ind.get("price", 0),
                        "alert_fired_at":  datetime.now(ET).isoformat(),
                        "composite_score": sig.get("composite_score"),
                    }
                    for _ep in ("/stock-trader/signal", "/day-trader/signal"):
                        try:
                            requests.post(
                                f"{backend_url.rstrip('/')}{_ep}",
                                json=_signal_payload,
                                timeout=5,
                            )
                            log.info("%s direct trigger fired: %s @ %.2f",
                                     _ep, tk, ind.get("price", 0))
                        except Exception as _ste:
                            log.debug("%s direct trigger failed (non-fatal): %s", _ep, _ste)

        # If nothing found, log a quiet pulse every other cycle
        if not candidates and not (len(alerted_today) % 2):
            log.info("No signals this cycle — market quiet")

        # Scan summary (only when signals were alerted this cycle)
        if n_alerted_this_cycle > 0:
            bo_count  = sum(1 for v in alerted_today.values() if v == "BREAKOUT")
            pre_count = sum(1 for v in alerted_today.values() if v == "PRE-BREAKOUT")
            f9_note   = f" | F9 path-gate ✓ ({n_f9_dropped} dropped)" if cfg.get("require_2step_breakout") else ""
            summary = (
                f"📋 <b>Scan summary</b> {datetime.now(ET).strftime('%H:%M ET')}\n"
                f"🚨 BREAKOUT today: {bo_count}\n"
                f"⚡ PRE-BREAKOUT today: {pre_count}\n"
                f"Total alerts today: {len(alerted_today)}\n"
                f"Filters active: F2 rel-str ✓ | F5 close-qual ✓ | F6 gap-rev ✓ | F7 intraday-RS ✓ | F8 ADX ✓{f9_note}"
                + (f"\n⚠️ Regime gate active: {regime.get('reason')}" if not regime_ok else "")
            )
            send_telegram(token, chat_id, summary)

        # ── #6: Heartbeat — written after every scan so a monitor can detect hangs ──
        try:
            _hb = {"ts": datetime.utcnow().isoformat(), "tickers": len(tickers)}
            with open(os.path.join(SCRIPT_DIR, "scanner_heartbeat.json"), "w", encoding="utf-8") as _hf:
                json.dump(_hb, _hf)
        except Exception as _hb_exc:
            log.debug("Heartbeat write failed: %s", _hb_exc)

        # ── #13: Persist _ticker_states so restarts don't reset the state machine ──
        try:
            _sp = os.path.join(SCRIPT_DIR, f"ticker_states_{date.today().isoformat()}.json")
            with open(_sp, "w", encoding="utf-8") as _tsf:
                json.dump(_ticker_states, _tsf)
        except Exception as _ts_exc:
            log.debug("_ticker_states persist failed: %s", _ts_exc)

        # Sleep until next interval
        elapsed = time.monotonic() - scan_start
        sleep_for = max(0, interval - elapsed)
        log.info("Next scan in %.0fm (sleeping %.0fs)", sleep_for / 60, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
