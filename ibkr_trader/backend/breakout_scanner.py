#!/usr/bin/env python3
"""
S&P 500 Breakout Scanner — Telegram Alerts
===========================================

Runs continuously during US market hours (Mon-Fri 9:30 AM – 4:15 PM ET).
Every 30 minutes it scans the full S&P 500 (~500 tickers) via yfinance,
computes per-ticker technical indicators, and sends Telegram alerts for
BREAKOUT and PRE-BREAKOUT candidates.

FIRST-TIME SETUP
----------------
Step 1 — Create a Telegram bot:
  a. Open Telegram, search for @BotFather
  b. Send /newbot, follow prompts, copy the API token (looks like 7123456789:ABCdef...)

Step 2 — Get your chat ID:
  a. Start a conversation with your new bot (send it any message)
  b. Open this URL in a browser (replace TOKEN):
     https://api.telegram.org/bot<TOKEN>/getUpdates
  c. Find "chat":{"id":XXXXXXXXX} — that number is your chat ID
  d. For a group/channel: add the bot to the group, send a message,
     then hit getUpdates — the id will be negative (e.g., -1001234567890)

Step 3 — Configure:
  Edit scanner_config.json (auto-created on first run) with your token + chat_id,
  OR export environment variables:
    set TELEGRAM_TOKEN=7123456789:ABCdef...
    set TELEGRAM_CHAT_ID=123456789

Step 4 — Run:
  C:/Projects/GenAI-Projects/ibkr_trader/venv/Scripts/python.exe breakout_scanner.py

SIGNAL DEFINITIONS
------------------
  BREAKOUT     — Price crosses above upper Bollinger Band (%B > 95) AND today's
                 volume exceeds the per-ticker 90th-percentile of trailing 252-day
                 volume ratios (confirmed by unusual volume — not just price).

  PRE-BREAKOUT — Price in 75–95 %B zone (approaching upper BB), RSI > 60,
                 above SMA20 and SMA50 (trend intact), volume building
                 (>75% of the 90th-pct threshold). Classic coiled-spring setup.

DEDUPLICATION
-------------
  Each ticker is alerted at most once per signal type per trading day.
  If a ticker escalates from PRE-BREAKOUT → BREAKOUT, a fresh alert fires.
  The signal state resets at market open each morning.
"""

import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("breakout_scanner")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "scanner_config.json")

# ── S&P 100 fallback (used if Wikipedia fetch fails) ─────────────────────────
SP100_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JPM",
    "XOM","JNJ","V","PG","MA","HD","CVX","MRK","ABBV","PEP","KO","AVGO",
    "COST","TMO","MCD","CSCO","ACN","ABT","LLY","NEE","DIS","WMT","CMCSA",
    "DHR","VZ","TXN","ADBE","PM","BMY","ORCL","QCOM","UNP","RTX","HON",
    "AMGN","IBM","SPGI","GS","BLK","AXP","CAT","GE","MDLZ","SCHW","INTC",
    "INTU","ISRG","LMT","MMM","MO","MS","NOW","PLD","SYK","T","TGT","UPS",
    "USB","WFC","ADI","AMAT","APD","CI","CL","D","DE","DUK","ELV","EMR",
    "F","FDX","GM","ICE","KLAC","MCO","MRNA","MTD","MU","NOC","NSC","NUE",
    "PANW","PH","PSA","SHW","SO","UBER","VLO","VST","WM","WMB","ZTS",
]


# ── Config management ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "telegram_token":         "",
    "telegram_chat_id":       "",
    "scan_interval_minutes":  30,
    "batch_size":             50,
    "alert_on":               ["BREAKOUT", "PRE-BREAKOUT"],
    "breakout_pct_b_min":     95,
    "pre_breakout_pct_b_min": 75,
    "pre_breakout_rsi_min":   60,
    "vol_threshold_pct":      0.75,
    "backend_url":            "http://localhost:8000",
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

    # Quality context line — shows the extra filter metrics when available
    quality_parts = []
    if ind.get("pct_b_min10") is not None:
        quality_parts.append(f"Pullback %B min={ind['pct_b_min10']:.0f}")
    if ind.get("close_pos") is not None:
        quality_parts.append(f"Close top {ind['close_pos']:.0f}% of range")
    if ind.get("ret_20d") is not None:
        quality_parts.append(f"20d ret={ind['ret_20d']:+.1f}%")
    quality_line = ("\n🔍 " + "  |  ".join(quality_parts)) if quality_parts else ""

    return (
        f"{emoji} <b>{signal}</b> — {ind['ticker']}\n"
        f"💰 Price: ${ind['price']:.2f} ({day_sign}{ind['day_chg_pct']:.1f}%)\n"
        f"📊 %B: {ind['pct_b']:.1f}  |  Vol: {vol_str}{proj_note} (90th-pct: {ind['vol_90pct']:.2f}×)\n"
        f"📈 RSI: {ind['rsi']:.0f}  |  {sma_txt}"
        f"{quality_line}"
        f"{tape_line}"
    )


# ── S&P 500 ticker list ───────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia (BRK.B → BRK-B normalised)."""
    try:
        url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; breakout-scanner/1.0)"}
        html    = requests.get(url, headers=headers, timeout=15).text
        tables  = pd.read_html(io.StringIO(html), flavor="lxml")
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        log.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return sorted(set(tickers))
    except Exception as exc:
        log.warning("Wikipedia fetch failed (%s) — falling back to S&P 100", exc)
        return SP100_FALLBACK


# ── Technical indicators ──────────────────────────────────────────────────────

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

        # Bollinger Bands (20, 2)
        sma20_s = closes.rolling(20).mean()
        std20_s = closes.rolling(20).std()
        upper   = sma20_s + 2 * std20_s
        lower   = sma20_s - 2 * std20_s
        band_w  = float(upper.iloc[-1] - lower.iloc[-1])
        pct_b   = float((last_close - lower.iloc[-1]) / band_w * 100) if band_w > 0 else 50.0

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

    if pct_b > cfg["breakout_pct_b_min"] and vol_ratio >= vol_90pct:
        return "BREAKOUT"
    if (cfg["pre_breakout_pct_b_min"] <= pct_b <= cfg["breakout_pct_b_min"]
            and rsi >= cfg["pre_breakout_rsi_min"]
            and bullish
            and vol_ratio >= vol_90pct * cfg["vol_threshold_pct"]):
        return "PRE-BREAKOUT"
    return None


def apply_quality_filters(candidates: list[dict], regime: dict, cfg: dict) -> list[dict]:
    """F2 / F3 / F5 scanner-side quality gates applied after classify_signal.

    F2 Relative strength: stock 20d return must exceed SPY 20d return.
    F3 Pullback:          %B must have been < 50 at least once in the prior 10 bars.
    F5 Close quality:     today's close must be in the top 25% of the day's range.

    Gates are skipped (pass-through) when the required data is unavailable so cold
    starts and data gaps never suppress alerts by default.
    """
    spy_ret20 = regime.get("spy_ret20")
    kept      = []
    n_f2 = n_f3 = n_f5 = 0

    for ind in candidates:
        tk = ind["ticker"]

        # F2: relative strength vs SPY
        ret_20d = ind.get("ret_20d")
        if spy_ret20 is not None and ret_20d is not None and ret_20d < spy_ret20:
            log.info("F2 relative-strength gate dropped %s: %.1f%% < SPY %.1f%%",
                     tk, ret_20d, spy_ret20)
            n_f2 += 1
            continue

        # F3: required pullback — %B dipped below 50 at least once in prior 10 bars
        pct_b_min10 = ind.get("pct_b_min10")
        if pct_b_min10 is not None and pct_b_min10 >= 50:
            log.info("F3 pullback gate dropped %s: pct_b_min10=%.1f (no dip below 50 in 10d)",
                     tk, pct_b_min10)
            n_f3 += 1
            continue

        # F5: close quality — close must be in top 25% of today's H-L range
        close_pos = ind.get("close_pos", 50.0)
        if close_pos < 75:
            log.info("F5 close-quality gate dropped %s: close_pos=%.1f%% (< 75%%)", tk, close_pos)
            n_f5 += 1
            continue

        kept.append(ind)

    if n_f2 or n_f3 or n_f5:
        log.info("Quality filters: dropped %d (F2 rel-str), %d (F3 pullback), %d (F5 close-quality)",
                 n_f2, n_f3, n_f5)
    return kept


# ── Download + scan ───────────────────────────────────────────────────────────

def download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download 1 year of daily OHLCV for a batch; return {ticker: DataFrame}."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers,
            period="1y",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as exc:
        log.warning("Batch download error: %s", exc)
        return {}

    result = {}
    if len(tickers) == 1:
        tk = tickers[0]
        if not raw.empty:
            result[tk] = raw
    else:
        for tk in tickers:
            try:
                df = raw[tk].dropna(how="all")
                if not df.empty:
                    result[tk] = df
            except (KeyError, TypeError):
                pass
    return result


def scan_universe(tickers: list[str], cfg: dict) -> list[dict]:
    """Scan all tickers in batches; return list of indicator dicts with signal."""
    batch_size = cfg.get("batch_size", 50)
    batches    = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    found: list[dict] = []
    all_inds: list[dict] = []   # all computed indicators (for near-miss reporting)
    n_skip = 0

    for idx, batch in enumerate(batches, 1):
        log.info("  Batch %d/%d (%d tickers)…", idx, len(batches), len(batch))
        hist_map = download_batch(batch)
        for tk in batch:
            hist = hist_map.get(tk)
            if hist is None:
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
        time.sleep(1)   # brief pause between batches to avoid rate-limits

    n_computed = len(all_inds)
    log.info("  Evaluated %d/%d tickers (%d skipped — no data/history)",
             n_computed, len(tickers), n_skip)

    # Log the top-5 nearest-to-breakout tickers even when nothing triggers,
    # so you can see the market is being evaluated and how close the leaders are.
    if all_inds and not found:
        near = sorted(all_inds, key=lambda x: x["pct_b"], reverse=True)[:5]
        lines = [f"  {d['ticker']:6s}  %B={d['pct_b']:5.1f}  RSI={d['rsi']:4.1f}"
                 f"  vol={d['vol_ratio'] or 0:.2f}x (thr={d['vol_90pct']:.2f}x)"
                 f"  scale={d.get('vol_scale',1.0):.1f}x"
                 f"  {'▲20/50' if d['above_sma20'] and d['above_sma50'] else '—trend'}"
                 for d in near]
        log.info("Top-5 nearest to breakout:\n" + "\n".join(lines))

    return found


# ── Market hours ──────────────────────────────────────────────────────────────
ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=15, second=0, microsecond=0)
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
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

    log.info("S&P 500 Breakout Scanner starting…")
    log.info("  Alerts on:       %s", ", ".join(alert_on))
    log.info("  Scan interval:   %d min", cfg["scan_interval_minutes"])
    log.info("  Config:          %s", CONFIG_PATH)
    log.info("  Volume mode:     intraday projection (scales partial-day volume to full-day "
             "equivalent so morning breakouts aren't suppressed)")

    send_telegram(
        token, chat_id,
        "🤖 <b>Breakout Scanner started</b>\n"
        f"Monitoring S&P 500 every {cfg['scan_interval_minutes']} min during market hours.\n"
        f"Alerts: {', '.join(alert_on)}",
    )

    sp500         = get_sp500_tickers()
    alerted_today: dict[str, str] = {}   # ticker → last signal type alerted
    last_scan_day: date | None = None

    while True:
        if not is_market_open():
            secs = seconds_to_next_open()
            log.info("Market closed — sleeping %dm until next open", secs // 60)
            time.sleep(min(secs, 300))   # wake every 5 min max to re-check
            continue

        # Reload config each cycle — picks up chat_id / token changes without restart
        cfg      = load_config()
        token    = cfg["telegram_token"]
        chat_id  = cfg["telegram_chat_id"]
        alert_on = set(cfg["alert_on"])
        interval = cfg["scan_interval_minutes"] * 60

        today = date.today()
        if today != last_scan_day:
            # New trading day: refresh ticker list + reset dedup state
            alerted_today = {}
            sp500 = get_sp500_tickers()
            last_scan_day = today
            send_telegram(
                token, chat_id,
                f"🌅 <b>Market open</b> — {today.strftime('%a %b %d')}\n"
                f"Scanning {len(sp500)} S&P 500 tickers every {cfg['scan_interval_minutes']} min.",
            )
            log.info("New day — scanning %d tickers", len(sp500))

        scan_start  = time.monotonic()
        backend_url = cfg.get("backend_url", "")

        # ── F1 Regime gate (live VIX + SPY SMA-200 from backend) ─────────────
        regime     = fetch_regime(backend_url)
        regime_ok  = regime.get("regime_ok", True)
        if not regime_ok:
            log.info("REGIME GATE ACTIVE: %s — Telegram suppressed this cycle",
                     regime.get("reason", "bad regime"))

        log.info("Starting scan (%d tickers)…", len(sp500))

        raw_candidates = scan_universe(sp500, cfg)

        # ── F2 / F3 / F5 Quality filters (scanner-side) ──────────────────────
        candidates = apply_quality_filters(raw_candidates, regime, cfg)
        n_dropped  = len(raw_candidates) - len(candidates)
        if n_dropped:
            log.info("Quality filters dropped %d/%d raw signals", n_dropped, len(raw_candidates))

        breakouts = [c for c in candidates if c["signal"] == "BREAKOUT"]
        pre_bo    = [c for c in candidates if c["signal"] == "PRE-BREAKOUT"]

        log.info(
            "Scan complete in %.0fs — %d BREAKOUT, %d PRE-BREAKOUT (after quality filters)",
            time.monotonic() - scan_start, len(breakouts), len(pre_bo),
        )

        # ── Combined backend sync + contingent Telegram ───────────────────────
        # BREAKOUT processed first (higher priority); PRE-BREAKOUT second.
        # For each signal:
        #   1. POST to backend (always — keeps watchlist current for refreshes)
        #   2. Backend gate may return action='blocked' for new entries → skip Telegram
        #   3. Telegram fires only when: backend accepted (action='added' or 'no_backend')
        #      AND signal is new or escalated today (dedup)
        n_alerted_this_cycle = 0
        for sig_type in ["BREAKOUT", "PRE-BREAKOUT"]:
            bucket = breakouts if sig_type == "BREAKOUT" else pre_bo

            for ind in sorted(bucket, key=lambda x: x["pct_b"], reverse=True):
                tk = ind["ticker"]

                # Always sync to backend
                response = post_to_backend(backend_url, ind, sig_type)
                action   = response.get("action", "no_backend")

                if action == "blocked":
                    log.info("Backend gate blocked %s %s: %s",
                             sig_type, tk, response.get("reason", "—"))
                    continue

                # Skip Telegram if regime gate is active (F1)
                if not regime_ok:
                    continue

                # Skip if this signal type is not in alert_on config
                if sig_type not in alert_on:
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
                tape = fetch_tape_sentiment(backend_url, tk)
                msg  = fmt_alert(ind, sig_type, tape)
                log.info("Alerting: %s %s (tape=%s, backend=%s)",
                         sig_type, tk, tape.get("label") if tape else "no data", action)
                send_telegram(token, chat_id, msg)
                time.sleep(0.3)   # Telegram rate limit: ~30 msg/s

        # If nothing found, log a quiet pulse every other cycle
        if not candidates and not (len(alerted_today) % 2):
            log.info("No signals this cycle — market quiet")

        # Scan summary (only when signals were alerted this cycle)
        if n_alerted_this_cycle > 0:
            bo_count  = sum(1 for v in alerted_today.values() if v == "BREAKOUT")
            pre_count = sum(1 for v in alerted_today.values() if v == "PRE-BREAKOUT")
            summary = (
                f"📋 <b>Scan summary</b> {datetime.now(ET).strftime('%H:%M ET')}\n"
                f"🚨 BREAKOUT today: {bo_count}\n"
                f"⚡ PRE-BREAKOUT today: {pre_count}\n"
                f"Total alerts today: {len(alerted_today)}\n"
                f"Filters active: F2 rel-str ✓ | F3 pullback ✓ | F5 close-quality ✓"
                + (f"\n⚠️ Regime gate active: {regime.get('reason')}" if not regime_ok else "")
            )
            send_telegram(token, chat_id, summary)

        # Sleep until next interval
        elapsed = time.monotonic() - scan_start
        sleep_for = max(0, interval - elapsed)
        log.info("Next scan in %.0fm (sleeping %.0fs)", sleep_for / 60, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
