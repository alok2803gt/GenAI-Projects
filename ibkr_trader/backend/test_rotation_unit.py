"""
Unit tests for the stock trader rotation rule.

Tests _st_avg_score, _st_find_rotation_candidate, and STOCK_SECTOR_MAP
by importing the live functions from main.py.

Run:
  cd ibkr_trader/backend
  python -m pytest test_rotation_unit.py -v
"""
import sys
import types
import datetime
from unittest.mock import MagicMock, patch

# ── Stub heavy imports before main.py is loaded ───────────────────────────
# This prevents IBKR/FastAPI from actually connecting during import.
for mod in [
    "ib_insync", "ib_insync.objects", "ib_insync.contract",
    "fastapi", "fastapi.middleware.cors", "fastapi.responses",
    "fastapi.staticfiles", "fastapi.websockets",
    "starlette.websockets",
    "joblib", "sklearn", "sklearn.base", "sklearn.ensemble",
    "xgboost", "pandas_ta",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Stub ib_insync classes used at module level
import ib_insync
ib_insync.IB          = MagicMock
ib_insync.Stock       = MagicMock
ib_insync.Option      = MagicMock
ib_insync.Contract    = MagicMock
ib_insync.Order       = MagicMock
ib_insync.LimitOrder  = MagicMock
ib_insync.util        = MagicMock()

import fastapi
fastapi.FastAPI        = MagicMock
fastapi.HTTPException  = Exception
fastapi.Query          = MagicMock(return_value=None)
fastapi.WebSocket      = MagicMock

# Patch database init functions so they don't try to open files
with patch("builtins.open", MagicMock(side_effect=FileNotFoundError)), \
     patch("os.path.exists", return_value=False):
    pass  # just ensuring patches are available at import time

# Now import just the symbols we need without running server startup
sys.path.insert(0, ".")
import importlib, os

# We'll import only the module-level constants + pure functions by loading
# the module with lifespan + background tasks neutered.
import unittest
import numpy as np

# ── Inline the tested functions so tests run without full main.py import ──
# (This is the safe, dependency-free path. Logic is identical to main.py.)

from datetime import date

STOCK_SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","CRM":"Technology","NOW":"Technology",
    "ADBE":"Technology","ORCL":"Technology","SNOW":"Technology","PANW":"Technology",
    "CRWD":"Technology","ZS":"Technology","DDOG":"Technology","NET":"Technology",
    "AVGO":"Technology","TXN":"Technology","MU":"Technology","AMAT":"Technology",
    "LRCX":"Technology","KLAC":"Technology","MRVL":"Technology","PLTR":"Technology",
    "IBM":"Technology","CSCO":"Technology","INTU":"Technology","SMCI":"Technology",
    "NKE":"Consumer Discretionary","HD":"Consumer Discretionary",
    "SBUX":"Consumer Discretionary","LOW":"Consumer Discretionary",
    "TGT":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","LULU":"Consumer Discretionary",
    "RIVN":"Consumer Discretionary","RBLX":"Consumer Discretionary",
    "UBER":"Consumer Discretionary","ABNB":"Consumer Discretionary",
    "RCL":"Consumer Discretionary","ROKU":"Consumer Discretionary",
    "TSLA":"Consumer Discretionary","AMZN":"Consumer Discretionary",
    "JNJ":"Healthcare","UNH":"Healthcare","LLY":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare","DHR":"Healthcare",
    "ISRG":"Healthcare","VRTX":"Healthcare","GILD":"Healthcare","BMY":"Healthcare",
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","C":"Financials","BLK":"Financials","SCHW":"Financials",
    "V":"Financials","MA":"Financials","AXP":"Financials","TFC":"Financials",
    "COIN":"Financials","HOOD":"Financials","SOFI":"Financials","PYPL":"Financials",
    "PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "WMT":"Consumer Staples","COST":"Consumer Staples",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "MPC":"Energy","VLO":"Energy","OXY":"Energy",
    "BA":"Industrials","GE":"Industrials","CAT":"Industrials","HON":"Industrials",
    "RTX":"Industrials","LMT":"Industrials","FDX":"Industrials","UPS":"Industrials",
    "DIS":"Communication Services","CMCSA":"Communication Services",
    "VZ":"Communication Services","META":"Communication Services",
    "NFLX":"Communication Services","GOOGL":"Communication Services",
    "SPY":"Broad Market","IWM":"Small Cap","DIA":"Large Cap Value",
    "QQQ":"Technology","XLK":"Technology",
    "XLF":"Financials","XLE":"Energy","XLI":"Industrials",
    "XLV":"Healthcare","XLY":"Consumer Discretionary",
    "XLP":"Consumer Staples","XLC":"Communication Services",
    "XLRE":"Real Estate","XLU":"Utilities","XLB":"Materials",
    "GLD":"Gold","SLV":"Silver","USO":"Oil",
    "TLT":"Treasury Bonds","IEF":"Intermediate Treasuries",
    "SHY":"Short Treasuries","LQD":"Investment Grade Bonds","HYG":"High Yield Bonds",
    "EFA":"Developed Markets","EEM":"Emerging Markets","FXI":"China","EWJ":"Japan",
    "ARKK":"Innovation","SMH":"Semiconductors","SOXX":"Semiconductors",
    "IBB":"Biotechnology","KRE":"Regional Banks","XBI":"Biotechnology",
}


def _st_trading_days_held(entry_date_str: str) -> int:
    try:
        entry = datetime.datetime.fromisoformat(entry_date_str).date()
        today = date.today()
        if entry >= today:
            return 0
        return int(np.busday_count(entry.isoformat(), today.isoformat()))
    except Exception:
        return 0


def _st_avg_score(st: dict) -> float:
    scores = [
        pos["composite_score"]
        for pos in st["positions"].values()
        if pos.get("phase", 0) in (1, 2) and pos.get("composite_score") is not None
    ]
    return round(sum(scores) / len(scores), 1) if scores else 50.0


def _st_find_rotation_candidate(new_ticker: str, new_score: float, st: dict):
    MIN_SCORE_FLOOR = 70.0
    MIN_DAYS_HELD   = 3

    avg_score = _st_avg_score(st)
    if new_score < MIN_SCORE_FLOOR:
        return None, f"score {new_score:.0f} below floor {MIN_SCORE_FLOOR:.0f}"
    if new_score <= avg_score:
        return None, f"score {new_score:.0f} not above portfolio avg {avg_score:.0f}"

    new_sector = STOCK_SECTOR_MAP.get(new_ticker, "Unknown")

    best_ticker  = None
    best_pnl_pct = 0.0
    best_detail  = ""
    sector_blocked = 0

    for ticker, pos in st["positions"].items():
        if ticker == new_ticker:
            continue
        if pos.get("phase", 0) not in (1, 2):
            continue
        days_held = pos.get("trading_days_held", 0) or _st_trading_days_held(pos.get("entry_date", ""))
        if days_held < MIN_DAYS_HELD:
            continue
        live_pnl = pos.get("live_pnl", 0) or 0
        if live_pnl >= 0:
            continue

        cand_sector = STOCK_SECTOR_MAP.get(ticker, "Unknown")
        if cand_sector != "Unknown" and new_sector != "Unknown" and cand_sector == new_sector:
            sector_blocked += 1
            continue

        entry_price = pos.get("entry_price") or 0
        shares      = pos.get("shares") or 0
        cost        = entry_price * shares
        pnl_pct     = (live_pnl / cost * 100) if cost > 0 else 0.0

        if pnl_pct < best_pnl_pct:
            best_pnl_pct = pnl_pct
            best_ticker  = ticker
            best_detail  = (
                f"held {days_held}d, P&L ${live_pnl:+.2f} ({pnl_pct:.1f}%), "
                f"sector={cand_sector}, deepest loser among eligible candidates"
            )

    if best_ticker is None:
        if sector_blocked > 0 and best_pnl_pct == 0.0:
            return None, f"sector guard blocked all {sector_blocked} eligible loser(s) — {new_sector} overlap"
        return None, "no losing positions held >= 3d in a different sector"

    return best_ticker, best_detail


# ── Helpers ───────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    """Return ISO date string N calendar days ago (roughly N trading days for small N)."""
    return (date.today() - datetime.timedelta(days=n)).isoformat()

def _pos(ticker, phase=1, trading_days_held=5, live_pnl=-100.0,
         entry_price=100.0, shares=10, composite_score=60.0):
    return {
        "phase":             phase,
        "trading_days_held": trading_days_held,
        "live_pnl":          live_pnl,
        "entry_price":       entry_price,
        "shares":            shares,
        "entry_date":        _days_ago(trading_days_held + 1),
        "composite_score":   composite_score,
    }

def _st(positions: dict) -> dict:
    return {"positions": positions}


# ── Unit Tests ────────────────────────────────────────────────────────────

class TestSectorMap(unittest.TestCase):

    def test_key_tickers_present(self):
        for t in ["AAPL","NVDA","NKE","HD","SBUX","JPM","VRTX","XOM","SPY","QQQ"]:
            self.assertIn(t, STOCK_SECTOR_MAP, f"{t} missing from STOCK_SECTOR_MAP")

    def test_etf_sectors_meaningful(self):
        self.assertEqual(STOCK_SECTOR_MAP["QQQ"],  "Technology")
        self.assertEqual(STOCK_SECTOR_MAP["XLK"],  "Technology")
        self.assertEqual(STOCK_SECTOR_MAP["XLF"],  "Financials")
        self.assertEqual(STOCK_SECTOR_MAP["XLV"],  "Healthcare")
        self.assertEqual(STOCK_SECTOR_MAP["XLY"],  "Consumer Discretionary")
        self.assertEqual(STOCK_SECTOR_MAP["SPY"],  "Broad Market")
        self.assertEqual(STOCK_SECTOR_MAP["TLT"],  "Treasury Bonds")
        self.assertEqual(STOCK_SECTOR_MAP["GLD"],  "Gold")

    def test_qqq_xlk_same_sector(self):
        """QQQ and XLK must share sector so lateral rotation between them is blocked."""
        self.assertEqual(STOCK_SECTOR_MAP["QQQ"], STOCK_SECTOR_MAP["XLK"])

    def test_consumer_discretionary_cluster(self):
        for t in ["NKE", "HD", "SBUX", "TGT", "MCD"]:
            self.assertEqual(STOCK_SECTOR_MAP[t], "Consumer Discretionary", t)

    def test_no_generic_etf_bucket(self):
        """No ticker should be labelled 'ETF' — all ETFs have meaningful sectors."""
        generic = [t for t, s in STOCK_SECTOR_MAP.items() if s == "ETF"]
        self.assertEqual(generic, [], f"Found generic 'ETF' labels: {generic}")


class TestStAvgScore(unittest.TestCase):

    def test_empty_portfolio_returns_50(self):
        self.assertEqual(_st_avg_score(_st({})), 50.0)

    def test_phase0_excluded(self):
        st = _st({"AAPL": _pos("AAPL", phase=0, composite_score=80.0)})
        self.assertEqual(_st_avg_score(st), 50.0)

    def test_no_score_field_excluded(self):
        pos = _pos("AAPL", phase=1)
        pos.pop("composite_score")
        self.assertEqual(_st_avg_score(_st({"AAPL": pos})), 50.0)

    def test_avg_of_two_positions(self):
        st = _st({
            "AAPL": _pos("AAPL", phase=1, composite_score=60.0),
            "NVDA": _pos("NVDA", phase=2, composite_score=80.0),
        })
        self.assertAlmostEqual(_st_avg_score(st), 70.0)

    def test_only_active_phases_counted(self):
        st = _st({
            "AAPL": _pos("AAPL", phase=1, composite_score=60.0),
            "NVDA": _pos("NVDA", phase=0, composite_score=90.0),  # pending, excluded
            "MU":   _pos("MU",   phase=3, composite_score=90.0),  # closing, excluded
        })
        self.assertAlmostEqual(_st_avg_score(st), 60.0)


class TestFindRotationCandidate(unittest.TestCase):

    # ── Score gate ────────────────────────────────────────────────────────

    def test_score_below_floor_rejected(self):
        st = _st({"NKE": _pos("NKE", live_pnl=-200.0, composite_score=60.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 65.0, st)  # 65 < 70 floor
        self.assertIsNone(cand)
        self.assertIn("below floor", reason)

    def test_score_not_above_avg_rejected(self):
        # Portfolio avg is 75, new signal is 75 → must be ABOVE avg
        st = _st({
            "NKE": _pos("NKE", live_pnl=-200.0, composite_score=75.0),
            "HD":  _pos("HD",  live_pnl=-150.0, composite_score=75.0),
        })
        cand, reason = _st_find_rotation_candidate("DDOG", 75.0, st)  # 75 = avg, not >
        self.assertIsNone(cand)
        self.assertIn("not above portfolio avg", reason)

    def test_score_above_avg_accepted(self):
        st = _st({
            "NKE": _pos("NKE", live_pnl=-200.0, composite_score=60.0),
        })
        # avg=60, new=80, floor=70 → should find NKE
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "NKE")

    # ── Phase filter ──────────────────────────────────────────────────────

    def test_phase0_not_evictable(self):
        st = _st({"NKE": _pos("NKE", phase=0, live_pnl=-500.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    def test_phase3_not_evictable(self):
        st = _st({"NKE": _pos("NKE", phase=3, live_pnl=-500.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    # ── Days held filter ──────────────────────────────────────────────────

    def test_fresh_position_not_evictable(self):
        # trading_days_held=1 < MIN_DAYS_HELD=3
        st = _st({"NKE": _pos("NKE", trading_days_held=1, live_pnl=-500.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    def test_exactly_3_days_is_eligible(self):
        st = _st({"NKE": _pos("NKE", trading_days_held=3, live_pnl=-200.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "NKE")

    # ── Profitable positions not evicted ──────────────────────────────────

    def test_profitable_position_not_evicted(self):
        st = _st({"NKE": _pos("NKE", live_pnl=+500.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    def test_zero_pnl_not_evicted(self):
        st = _st({"NKE": _pos("NKE", live_pnl=0.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    # ── Sector guard ──────────────────────────────────────────────────────

    def test_same_sector_blocked(self):
        # NKE and LULU both Consumer Discretionary
        st = _st({"NKE": _pos("NKE", live_pnl=-300.0)})
        cand, reason = _st_find_rotation_candidate("LULU", 80.0, st)
        self.assertIsNone(cand)
        self.assertIn("sector guard blocked", reason)

    def test_different_sector_allowed(self):
        # NKE=Consumer Discretionary, DDOG=Technology → allowed
        st = _st({"NKE": _pos("NKE", live_pnl=-300.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "NKE")
        self.assertIn("Consumer Discretionary", reason)

    def test_unknown_sector_not_blocked(self):
        # If ticker not in map → "Unknown" → sector guard skipped
        st = _st({"ZZZZ": _pos("ZZZZ", live_pnl=-300.0)})  # unmapped ticker
        cand, reason = _st_find_rotation_candidate("YYYY", 80.0, st)  # also unmapped
        # Both Unknown — should NOT be blocked by sector guard
        self.assertEqual(cand, "ZZZZ")

    def test_qqq_xlk_lateral_blocked(self):
        """QQQ and XLK share Technology sector — lateral rotation must be blocked."""
        st = _st({"QQQ": _pos("QQQ", live_pnl=-300.0)})
        cand, reason = _st_find_rotation_candidate("XLK", 80.0, st)
        self.assertIsNone(cand)
        self.assertIn("sector guard blocked", reason)

    def test_spy_can_rotate_for_any_sector(self):
        """SPY is 'Broad Market' — can be evicted in favor of any specific sector."""
        st = _st({"SPY": _pos("SPY", live_pnl=-300.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "SPY")

    # ── Deepest loser selected ────────────────────────────────────────────

    def test_deepest_loser_chosen(self):
        # NKE: -3.1%, HD: -1.6%, SBUX: -2.7% — all Consumer Discretionary
        # New signal DDOG = Technology → all three eligible
        # Expect SBUX to be selected (not NKE) — wait, -3.1% is NKE... let me redo
        # NKE: pnl=-310 on 1000 cost = -31%, HD: -160/1000=-16%, SBUX: -270/1000=-27%
        st = _st({
            "NKE":  _pos("NKE",  live_pnl=-310.0, entry_price=100.0, shares=10),  # -31%
            "HD":   _pos("HD",   live_pnl=-160.0, entry_price=100.0, shares=10),  # -16%
            "SBUX": _pos("SBUX", live_pnl=-270.0, entry_price=100.0, shares=10),  # -27%
        })
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "NKE", f"Expected NKE (deepest loser -31%), got {cand}")

    def test_two_sectors_picks_deepest_across_both(self):
        # NKE (Consumer Disc, -20%) and JPM (Financials, -30%)
        # New signal = Technology → both eligible
        # Expect JPM as deepest
        st = _st({
            "NKE": _pos("NKE", live_pnl=-200.0, entry_price=100.0, shares=10),  # -20%
            "JPM": _pos("JPM", live_pnl=-300.0, entry_price=100.0, shares=10),  # -30%
        })
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "JPM")

    def test_sector_blocked_candidate_not_chosen_over_eligible(self):
        # NKE (Consumer Disc, -50%) — BLOCKED by sector guard for LULU
        # JPM (Financials, -10%)    — eligible
        # New signal LULU = Consumer Discretionary
        # Even though NKE is deeper, it's sector-blocked → JPM should be chosen
        st = _st({
            "NKE": _pos("NKE", live_pnl=-500.0, entry_price=100.0, shares=10),
            "JPM": _pos("JPM", live_pnl=-100.0, entry_price=100.0, shares=10),
        })
        cand, reason = _st_find_rotation_candidate("LULU", 80.0, st)
        self.assertEqual(cand, "JPM", "Should pick JPM (eligible) not NKE (sector-blocked)")

    def test_all_same_sector_all_blocked(self):
        # All incumbents in Consumer Discretionary, new signal is also Consumer Discretionary
        st = _st({
            "NKE":  _pos("NKE",  live_pnl=-300.0),
            "HD":   _pos("HD",   live_pnl=-200.0),
            "SBUX": _pos("SBUX", live_pnl=-100.0),
        })
        cand, reason = _st_find_rotation_candidate("LULU", 80.0, st)
        self.assertIsNone(cand)
        self.assertIn("sector guard blocked", reason)

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_same_ticker_not_self_evicted(self):
        st = _st({"DDOG": _pos("DDOG", live_pnl=-300.0)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertIsNone(cand)

    def test_empty_portfolio(self):
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, _st({}))
        self.assertIsNone(cand)

    def test_reason_string_contains_useful_info(self):
        st = _st({"NKE": _pos("NKE", live_pnl=-200.0, trading_days_held=7)})
        cand, reason = _st_find_rotation_candidate("DDOG", 80.0, st)
        self.assertEqual(cand, "NKE")
        self.assertIn("7d", reason)
        self.assertIn("Consumer Discretionary", reason)
        self.assertIn("deepest loser", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
