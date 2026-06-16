"""
IBKR Auto-Trader Watchdog
Monitors the backend, auto-restarts it on failure, and logs all events.

Usage:
    python watchdog.py

Keeps running until Ctrl-C.  Start this alongside TWS for 24/7 operation.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
VENV_PY     = ROOT / "venv" / "Scripts" / "python.exe"
BACKEND_URL = "http://localhost:8000"
LOG_FILE    = ROOT / "watchdog.log"
CHECK_SEC   = 60        # health-check interval
MAX_FAIL    = 3         # consecutive failures before restart
RECONNECT_AFTER = 5     # health checks before trying IBKR reconnect

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog")

# ── State ────────────────────────────────────────────────────────────────────
_proc:       subprocess.Popen | None = None
_failures    = 0
_ok_streak   = 0
_total_restarts = 0


def _start_backend() -> subprocess.Popen:
    log.info("Starting backend …")
    p = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "main:app",
         "--port", "8000", "--host", "0.0.0.0"],
        cwd=str(BACKEND_DIR),
    )
    log.info("Backend started  PID=%d", p.pid)
    return p


def _health() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _ibkr_connected() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BACKEND_URL}/status", timeout=5) as r:
            data = json.loads(r.read())
            return bool(data.get("connected"))
    except Exception:
        return False


def _trigger_reconnect() -> None:
    try:
        import urllib.request
        body = json.dumps({"port": 7497}).encode()
        req  = urllib.request.Request(
            f"{BACKEND_URL}/reconnect", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        log.info("Triggered IBKR reconnect")
    except Exception as exc:
        log.warning("Reconnect request failed: %s", exc)


def _portfolio_summary() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BACKEND_URL}/portfolio/delta", timeout=10) as r:
            d = json.loads(r.read())
            return (f"positions={d.get('count',0)}  "
                    f"net_delta={d.get('total_delta',0):+.1f}")
    except Exception:
        return "unavailable"


def _journal_summary() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BACKEND_URL}/journal/stats", timeout=10) as r:
            d = json.loads(r.read())
            if d.get("total_trades", 0) == 0:
                return "no trades yet"
            return (f"trades={d['total_trades']}  "
                    f"win={d.get('win_rate','?')}%  "
                    f"total_pnl=${d.get('total_pnl',0):+.0f}  "
                    f"model_v{d.get('model_version',0)}")
    except Exception:
        return "unavailable"


def _daily_report() -> None:
    log.info("=== DAILY REPORT ===")
    log.info("Portfolio : %s", _portfolio_summary())
    log.info("Journal   : %s", _journal_summary())
    log.info("Restarts  : %d", _total_restarts)
    log.info("====================")


def main() -> None:
    global _proc, _failures, _ok_streak, _total_restarts

    log.info("Watchdog starting — backend dir: %s", BACKEND_DIR)
    _proc = _start_backend()
    time.sleep(15)  # give backend time to boot

    last_report_day = -1
    ibkr_fail_streak = 0

    while True:
        try:
            healthy = _health()

            if healthy:
                _failures  = 0
                _ok_streak += 1
                ibkr_fail_streak = 0

                # Check IBKR connection every RECONNECT_AFTER ticks
                if _ok_streak % RECONNECT_AFTER == 0:
                    if not _ibkr_connected():
                        ibkr_fail_streak += 1
                        log.warning("IBKR not connected (streak=%d) — triggering reconnect",
                                    ibkr_fail_streak)
                        _trigger_reconnect()
                    else:
                        ibkr_fail_streak = 0

                # Daily report at first tick of a new calendar day
                today = datetime.now().day
                if today != last_report_day:
                    last_report_day = today
                    _daily_report()

            else:
                _failures  += 1
                _ok_streak  = 0
                log.warning("Backend unhealthy  (%d/%d)", _failures, MAX_FAIL)

                if _failures >= MAX_FAIL:
                    log.error("Restarting backend after %d consecutive failures", MAX_FAIL)
                    if _proc and _proc.poll() is None:
                        _proc.terminate()
                        try:
                            _proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            _proc.kill()
                    time.sleep(3)
                    _proc = _start_backend()
                    time.sleep(15)
                    _failures = 0
                    _total_restarts += 1
                    log.info("Restart #%d complete", _total_restarts)

        except KeyboardInterrupt:
            log.info("Watchdog stopping (Ctrl-C)")
            if _proc and _proc.poll() is None:
                _proc.terminate()
            break
        except Exception as exc:
            log.error("Watchdog error: %s", exc, exc_info=True)

        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    main()
