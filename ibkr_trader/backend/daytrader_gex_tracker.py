"""
Forward-only GEX/VEX tracking for Day Trader's REAL daily candidates.

Real, structural constraint (confirmed live 2026-09-07): IBKR's option
chain query only returns CURRENTLY valid expiries -- there is no way to
retrieve historical open interest/Greeks for a past date, so GEX cannot be
backtested against the existing historical candidate set the way RVOL,
SMA/EMA, sector-cap, etc. were. This script instead starts a REAL,
forward-accumulating log: each real trading day, compute real GEX/VEX
(reusing gex-vex-calculator's own tested calc_gex_vex.py, not a re-derived
copy) for that day's REAL Day Trader candidates, and log it alongside the
REAL trade outcome once known. After enough real days accumulate, this
becomes backtestable the normal way.

Candidate source: trade_journal.db's real DAY_BREAKOUT rows (is_paper=0)
for TODAY -- i.e., tickers Day Trader actually confirmed and entered.
Deliberately NOT the full 80-ticker premarket shortlist: calc_gex_vex.py
takes ~2min/ticker at the default band, so 80 tickers would take over
2.5 hours, impractical to run daily. The real, ENTERED candidate set is
usually well under a dozen names/day -- a bounded, realistic runtime.

Output: daytrader_gex_tracking.jsonl (append-only, one line per real
ticker/day), separate from gex_vex_cache.json (safe-income-screener's own
different curated universe -- never overwritten, just incidentally
refreshed for the overlapping tickers if any).
"""
import json
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
JOURNAL_DB_PATH = HERE / "trade_journal.db"
GEX_CACHE_PATH = Path.home() / ".claude" / "skills" / "gex-vex-calculator" / "gex_vex_cache.json"
TRACKING_LOG_PATH = HERE / "daytrader_gex_tracking.jsonl"
CALC_SCRIPT = Path.home() / ".claude" / "skills" / "gex-vex-calculator" / "calc_gex_vex.py"


def get_todays_candidates(target_date: str | None = None) -> list[str]:
    """Real tickers Day Trader confirmed+entered on target_date (default:
    today). Uses trade_journal.db's real DAY_BREAKOUT rows -- both
    opened_at (entry day) matches, live trades only."""
    d = target_date or date.today().isoformat()
    con = sqlite3.connect(JOURNAL_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT DISTINCT ticker FROM trade_journal "
        "WHERE strategy_type='DAY_BREAKOUT' AND is_paper=0 AND opened_at=?",
        (d,)
    ).fetchall()
    con.close()
    return sorted(set(r["ticker"] for r in rows))


def run_gex_for_tickers(tickers: list[str]) -> dict:
    if not tickers:
        return {}
    print(f"Running real GEX/VEX for {len(tickers)} tickers: {tickers}")
    result = subprocess.run(
        [sys.executable, str(CALC_SCRIPT), "--tickers", ",".join(tickers)],
        capture_output=True, text=True, timeout=len(tickers) * 200 + 120,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"calc_gex_vex.py real error: {result.stderr[-1000:]}")
        return {}
    with open(GEX_CACHE_PATH) as f:
        cache = json.load(f)
    return {t: cache["tickers"][t] for t in tickers if t in cache.get("tickers", {})}


def log_tracking_entries(target_date: str, gex_data: dict) -> None:
    with open(TRACKING_LOG_PATH, "a") as f:
        for ticker, data in gex_data.items():
            entry = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "trade_date": target_date,
                "ticker": ticker,
                "net_gex": data.get("net_gex"),
                "net_vex": data.get("net_vex"),
                "gex_regime": data.get("gex_regime"),
                "spot": data.get("spot"),
                "dte": data.get("dte"),
                # real trade outcome filled in later by backfill_trade_outcomes()
                # once the position closes -- None here means not yet known
                "real_pnl": None, "real_pnl_pct": None, "real_win": None,
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Logged {len(gex_data)} real GEX entries for {target_date} to {TRACKING_LOG_PATH.name}")


def backfill_trade_outcomes() -> int:
    """Fills in real_pnl/real_win for any tracking rows whose trade has
    since closed -- run this periodically (e.g., end of day) to keep the
    forward-accumulating dataset current."""
    if not TRACKING_LOG_PATH.exists():
        return 0
    lines = [json.loads(l) for l in open(TRACKING_LOG_PATH)]
    con = sqlite3.connect(JOURNAL_DB_PATH)
    con.row_factory = sqlite3.Row
    n_filled = 0
    for entry in lines:
        if entry["real_pnl"] is not None:
            continue
        row = con.execute(
            "SELECT pnl, pnl_pct, win FROM trade_journal "
            "WHERE strategy_type='DAY_BREAKOUT' AND is_paper=0 AND ticker=? AND opened_at=? "
            "ORDER BY id DESC LIMIT 1",
            (entry["ticker"], entry["trade_date"])
        ).fetchone()
        if row is not None:
            entry["real_pnl"] = row["pnl"]
            entry["real_pnl_pct"] = row["pnl_pct"]
            entry["real_win"] = bool(row["win"])
            n_filled += 1
    con.close()
    if n_filled:
        with open(TRACKING_LOG_PATH, "w") as f:
            for entry in lines:
                f.write(json.dumps(entry) + "\n")
    return n_filled


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="Target date (default: today), for testing against a real past day")
    p.add_argument("--backfill-only", action="store_true")
    args = p.parse_args()

    if args.backfill_only:
        n = backfill_trade_outcomes()
        print(f"Backfilled {n} real trade outcomes into existing tracking rows.")
        return

    target_date = args.date or date.today().isoformat()
    tickers = get_todays_candidates(target_date)
    if not tickers:
        print(f"No real DAY_BREAKOUT entries found for {target_date} -- nothing to track.")
        return
    gex_data = run_gex_for_tickers(tickers)
    log_tracking_entries(target_date, gex_data)
    n = backfill_trade_outcomes()
    print(f"Also backfilled {n} outcomes for any previously-untracked closed trades.")


if __name__ == "__main__":
    main()
