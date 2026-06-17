"""
Test suite — Dynamic Universe Screener (Option A)

  Phase 1  UNIT  (default)   — mocked yfinance, no server needed
  Phase 2  SIT               — requires backend on http://localhost:8000
  Phase 3  UAT               — requires backend on http://localhost:8000

  Run all:         pytest test_universe_screener.py -v
  Unit only:       pytest test_universe_screener.py -v -m "not sit and not uat"
  SIT + UAT only:  pytest test_universe_screener.py -v -m "sit or uat"
"""

import asyncio
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── safe import — no server starts until uvicorn.run() ────────────────────
sys.path.insert(0, r"C:\Projects\GenAI-Projects\ibkr_trader\backend")
import main  # noqa: E402
from main import (
    CANDIDATE_POOL,
    _DEFAULT_UNIVERSE,
    _screen_universe,
    state,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "sit: system integration test (requires running backend)")
    config.addinivalue_line("markers", "uat: user acceptance test (requires running backend)")


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

_N = 65  # matches download period in _screen_universe


def _make_df(ticker_data: dict, n: int = _N) -> pd.DataFrame:
    """
    Build a synthetic yfinance multi-ticker download DataFrame.
    ticker_data: {"TICK": {"closes": [...], "volumes": [...]}, ...}
    """
    dates = pd.bdate_range(end="2026-06-16", periods=n)
    frames = {}
    for ticker, d in ticker_data.items():
        closes  = list(d.get("closes",  [100.0] * n))
        volumes = list(d.get("volumes", [1_000_000] * n))
        if len(closes)  < n: closes  = [closes[0]]  * (n - len(closes))  + closes
        if len(volumes) < n: volumes = [volumes[0]] * (n - len(volumes)) + volumes
        frames[ticker] = pd.DataFrame(
            {"Close": closes[:n], "Volume": volumes[:n]}, index=dates
        )
    return pd.concat(frames, axis=1)


def _trend_up(n=_N, start=80.0, step=0.5) -> list:
    """Monotonically rising → last price > SMA-50 (above_sma50=True).
    Also: all daily changes positive → loss.rolling(14).mean() = 0 → rs_val = nan → RSI fallback = 50."""
    return [start + step * i for i in range(n)]


def _trend_down(n=_N, start=120.0, step=0.5) -> list:
    """Monotonically falling → last price < SMA-50 (above_sma50=False).
    Also: all daily changes negative → gain=0 → RSI = 0 → penalty -10."""
    return [start - step * i for i in range(n)]


def _rsi_neutral(n=_N, start=100.0, amp=1.0) -> list:
    """Alternating ±amp → RSI ≈ 50 (sweet spot, +25 pts).
    Last close = 100 (even index), SMA50 ≈ 100.5 → above_sma50 = False."""
    c = [start]
    for i in range(n - 1):
        c.append(c[-1] + amp * (1 if i % 2 == 0 else -1))
    return c


def _rsi_overbought(n=_N, start=100.0) -> list:
    """12 big-up + 2 small-down in last 14 → RSI ≈ 97 (> 80, penalty -10)."""
    c = [start + i * 0.5 for i in range(n - 14)]
    for _ in range(12): c.append(c[-1] + 2.0)
    for _ in range(2):  c.append(c[-1] - 0.3)
    return c


def _rsi_oversold(n=_N, start=120.0) -> list:
    """12 big-down + 2 small-up in last 14 → RSI ≈ 2 (< 30, penalty -10)."""
    c = [start - i * 0.5 for i in range(n - 14)]
    for _ in range(12): c.append(c[-1] - 2.0)
    for _ in range(2):  c.append(c[-1] + 0.3)
    return c


# ─────────────────────────────────────────────────────────────────────────────
#  Test infrastructure helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _use_pool(*tickers):
    """Temporarily replace CANDIDATE_POOL so only these tickers are screened."""
    orig = list(main.CANDIDATE_POOL)
    main.CANDIDATE_POOL.clear()
    main.CANDIDATE_POOL.extend(tickers)
    try:
        yield
    finally:
        main.CANDIDATE_POOL.clear()
        main.CANDIDATE_POOL.extend(orig)


def _run_screen(ticker_data: dict, top_n: int = 5) -> Optional[list]:
    """
    Run _screen_universe using only the tickers in ticker_data as the pool.
    This avoids the issue of synthetic names not being in CANDIDATE_POOL.
    """
    df = _make_df(ticker_data)
    with _use_pool(*ticker_data.keys()):
        with patch("main.yf.download", return_value=df):
            return asyncio.run(_screen_universe(top_n=top_n))


# Neutral filler tickers — pass price/vol filters, score ~30 each
# (RSI neutral, SMA50 borderline False → 0 SMA + 25 RSI + 5 vol = 30)
# 4 fillers so any single-subject test has 1 subject + 4 fillers = 5 ≥ minimum threshold
_FILLER = {
    "FILL1": {"closes": _rsi_neutral(), "volumes": [600_000] * _N},
    "FILL2": {"closes": _rsi_neutral(), "volumes": [600_000] * _N},
    "FILL3": {"closes": _rsi_neutral(), "volumes": [600_000] * _N},
    "FILL4": {"closes": _rsi_neutral(), "volumes": [600_000] * _N},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Restore state and CSP_UNIVERSE after every test."""
    orig_scores   = list(state["universe_scores"])
    orig_screened = state["universe_last_screened"]
    orig_iv       = {k: list(v) for k, v in state.get("iv_history", {}).items()}
    orig_universe = list(main.CSP_UNIVERSE)
    yield
    state["universe_scores"]        = orig_scores
    state["universe_last_screened"] = orig_screened
    state["iv_history"]             = orig_iv
    main.CSP_UNIVERSE.clear()
    main.CSP_UNIVERSE.extend(orig_universe)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidatePoolIntegrity:
    """Validate the static constant lists before any screening logic runs."""

    def test_no_duplicate_tickers(self):
        dupes = [t for t in set(CANDIDATE_POOL) if CANDIDATE_POOL.count(t) > 1]
        assert not dupes, f"Duplicates in CANDIDATE_POOL: {dupes}"

    def test_all_strings_and_uppercase(self):
        bad = [t for t in CANDIDATE_POOL if not isinstance(t, str) or t != t.upper()]
        assert not bad, f"Not uppercase strings: {bad}"

    def test_minimum_pool_size(self):
        assert len(CANDIDATE_POOL) >= 100, "Pool should have ≥100 tickers"

    def test_default_universe_has_original_tickers(self):
        for t in ["AAPL","MSFT","NVDA","AMZN","GOOGL","META",
                  "JPM","GS","V","MA","SPY","QQQ",
                  "AMD","MU","AVGO","COST","HD","LLY","UNH"]:
            assert t in _DEFAULT_UNIVERSE, f"{t} missing from _DEFAULT_UNIVERSE"

    def test_default_universe_size(self):
        assert len(_DEFAULT_UNIVERSE) == 19

    def test_default_universe_is_separate_object(self):
        assert _DEFAULT_UNIVERSE is not main.CSP_UNIVERSE

    def test_mutating_csp_universe_does_not_affect_default(self):
        main.CSP_UNIVERSE.append("__CANARY__")
        assert "__CANARY__" not in _DEFAULT_UNIVERSE


class TestScreenUniverseHardFilters:
    """Price and volume filters must exclude bad tickers cleanly."""

    def test_price_below_20_excluded(self):
        data = {"CHEAP": {"closes": [10.0] * _N, "volumes": [5_000_000] * _N},
                **_FILLER}
        result = _run_screen(data, top_n=5)
        assert result is None or "CHEAP" not in result

    def test_price_above_800_excluded(self):
        data = {"PRICEY": {"closes": [900.0] * _N, "volumes": [5_000_000] * _N},
                **_FILLER}
        result = _run_screen(data, top_n=5)
        assert result is None or "PRICEY" not in result

    def test_volume_below_500k_excluded(self):
        data = {"ILLIQUID": {"closes": _rsi_neutral(), "volumes": [100_000] * _N},
                **_FILLER}
        result = _run_screen(data, top_n=5)
        assert result is None or "ILLIQUID" not in result

    def test_valid_ticker_included_in_result(self):
        data = {"GOOD": {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
                **_FILLER}
        result = _run_screen(data, top_n=5)
        assert result is not None and "GOOD" in result

    def test_result_is_list_of_uppercase_strings(self):
        data = {"GOOD": {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
                **_FILLER}
        result = _run_screen(data, top_n=5)
        assert result is not None
        assert all(isinstance(t, str) and t == t.upper() for t in result)

    def test_insufficient_history_skipped_gracefully(self):
        """Ticker with only 10 bars of history must be skipped; screen still finishes."""
        n_short = 10
        dates_short = pd.bdate_range(end="2026-06-16", periods=n_short)
        short_frame = pd.DataFrame(
            {"Close": [100.0] * n_short, "Volume": [2_000_000] * n_short},
            index=dates_short,
        )
        good_df = _make_df({"GOODTICK": {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
                            **_FILLER})
        # Merge sparse ticker into the good DataFrame
        sparse_df = pd.concat({"SPARSE": short_frame}, axis=1)
        combined = pd.concat([good_df, sparse_df], axis=1)

        with _use_pool("SPARSE", "GOODTICK", "FILL1", "FILL2", "FILL3"):
            with patch("main.yf.download", return_value=combined):
                result = asyncio.run(_screen_universe(top_n=5))
        assert result is None or "SPARSE" not in result


class TestScreenUniverseTopN:

    def test_returns_exactly_top_n(self):
        data = {f"T{i:02d}": {"closes": _rsi_neutral(), "volumes": [1_000_000] * _N}
                for i in range(10)}
        result = _run_screen(data, top_n=5)
        assert result is not None and len(result) == 5

    def test_fallback_none_when_fewer_than_5_pass_filters(self):
        """If only 3 tickers survive hard filters, screen returns None (safety threshold)."""
        data = {
            "A1": {"closes": [10.0]   * _N, "volumes": [2_000_000] * _N},   # price < 20
            "A2": {"closes": [900.0]  * _N, "volumes": [2_000_000] * _N},   # price > 800
            "A3": {"closes": _rsi_neutral(), "volumes": [100_000]   * _N},   # vol < 500K
        }
        result = _run_screen(data, top_n=3)
        assert result is None

    def test_all_good_tickers_returned_when_fewer_than_top_n_pass(self):
        """5 tickers available, top_n=10 → returns all 5."""
        data = {f"T{i:02d}": {"closes": _rsi_neutral(), "volumes": [1_000_000] * _N}
                for i in range(5)}
        result = _run_screen(data, top_n=10)
        assert result is not None and len(result) == 5


class TestScreenUniverseScoring:
    """Verify score ordering — top_n=1 selects the higher-scoring ticker."""

    def _top1(self, good_data: dict, bad_data: dict) -> Optional[list]:
        """Run a 5-ticker pool where GOOD and BAD compete; top_n=1 should be GOOD."""
        data = {"GOOD": good_data, "BAD": bad_data, **_FILLER}
        return _run_screen(data, top_n=1)

    def test_above_sma50_scores_higher_than_below(self):
        """
        GOOD: _trend_up → above_sma50=True (+25), RSI=50 fallback (+25), 2M vol (+15) = 65
        BAD:  _trend_down → above_sma50=False (0), RSI=0 (-10), 2M vol (+15) = 5
        FILL: neutral → SMA borderline False (0), RSI=50 (+25), 600K (+5) = 30
        """
        result = self._top1(
            {"closes": _trend_up(),   "volumes": [2_000_000] * _N},
            {"closes": _trend_down(), "volumes": [2_000_000] * _N},
        )
        assert result == ["GOOD"], f"Expected ['GOOD'], got {result}"

    def test_rsi_sweet_spot_preferred_over_overbought(self):
        """
        GOOD: RSI≈50 (+25), SMA borderline (0), 2M vol (+15) = 40
        BAD:  RSI≈97 (-10), SMA True (+25), 2M vol (+15) = 30
        FILL: RSI≈50 (+25), SMA borderline (0), 600K (+5) = 30
        """
        result = self._top1(
            {"closes": _rsi_neutral(),    "volumes": [2_000_000] * _N},
            {"closes": _rsi_overbought(), "volumes": [2_000_000] * _N},
        )
        assert result == ["GOOD"], f"Expected ['GOOD'], got {result}"

    def test_rsi_sweet_spot_preferred_over_oversold(self):
        """
        GOOD: RSI≈50 (+25), SMA False (0), 2M vol (+15) = 40
        BAD:  RSI≈2  (-10), SMA False (0), 2M vol (+15) = 5
        """
        result = self._top1(
            {"closes": _rsi_neutral(),  "volumes": [2_000_000] * _N},
            {"closes": _rsi_oversold(), "volumes": [2_000_000] * _N},
        )
        assert result == ["GOOD"], f"Expected ['GOOD'], got {result}"

    def test_high_volume_scores_higher(self):
        """2M ADV (+15) beats 510K ADV (+5) — same RSI/SMA otherwise."""
        result = self._top1(
            {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
            {"closes": _rsi_neutral(), "volumes": [510_000]   * _N},
        )
        assert result == ["GOOD"], f"Expected ['GOOD'], got {result}"

    def test_iv_rank_bonus_from_iv_history(self):
        """Ticker with 24 iv_history entries (iv_rank≈50%) gets +12.5 bonus over one without."""
        state["iv_history"]["WITHIV"] = [
            {"date": f"2025-{m:02d}-01", "iv": 0.20 + 0.01 * i}
            for i, m in enumerate(range(1, 25))
        ]
        data = {
            "WITHIV":    {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
            "WITHOUTIV": {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
            **_FILLER,
        }
        result = _run_screen(data, top_n=1)
        assert result == ["WITHIV"], f"Expected ['WITHIV'], got {result}"

    def test_continuity_bonus_for_incumbent_ticker(self):
        """Ticker already in CSP_UNIVERSE gets +5 to break ties."""
        main.CSP_UNIVERSE.clear()
        main.CSP_UNIVERSE.append("INCUMBENT")
        data = {
            "INCUMBENT": {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
            "NEWCOMER":  {"closes": _rsi_neutral(), "volumes": [2_000_000] * _N},
            **_FILLER,
        }
        result = _run_screen(data, top_n=1)
        assert result == ["INCUMBENT"], f"Expected ['INCUMBENT'], got {result}"


class TestScreenUniverseStateUpdates:

    def _run_good(self, top_n=5):
        data = {f"T{i:02d}": {"closes": _rsi_neutral(), "volumes": [1_000_000] * _N}
                for i in range(10)}
        return _run_screen(data, top_n=top_n)

    def test_universe_scores_populated(self):
        state["universe_scores"] = []
        self._run_good()
        assert len(state["universe_scores"]) > 0

    def test_scores_have_expected_fields(self):
        self._run_good()
        required = {"ticker", "score", "price", "avg_vol_30d", "rsi14", "above_sma50"}
        for row in state["universe_scores"]:
            missing = required - set(row.keys())
            assert not missing, f"Score row missing: {missing}"

    def test_last_screened_timestamp_is_set(self):
        state["universe_last_screened"] = None
        self._run_good()
        ts = state["universe_last_screened"]
        assert ts is not None
        datetime.fromisoformat(ts)   # must be valid ISO 8601

    def test_scores_are_sorted_descending(self):
        self._run_good()
        scores = [r["score"] for r in state["universe_scores"]]
        assert scores == sorted(scores, reverse=True), "Scores must be descending"

    def test_state_unchanged_on_failure(self):
        state["universe_scores"]        = []
        state["universe_last_screened"] = "SENTINEL"
        with patch("main.yf.download", side_effect=RuntimeError("network down")):
            with _use_pool("DUMMY"):
                result = asyncio.run(_screen_universe())
        assert result is None
        assert state["universe_last_screened"] == "SENTINEL"


class TestScreenUniverseFailureModes:

    def test_returns_none_on_yfinance_exception(self):
        with _use_pool("AAPL"):
            with patch("main.yf.download", side_effect=Exception("API down")):
                result = asyncio.run(_screen_universe())
        assert result is None

    def test_returns_none_when_all_tickers_fail_price_filter(self):
        data = {"T1": {"closes": [15.0] * _N, "volumes": [1_000_000] * _N}}
        result = _run_screen(data)
        assert result is None

    def test_returns_none_on_empty_dataframe(self):
        with _use_pool("AAPL"):
            with patch("main.yf.download", return_value=pd.DataFrame()):
                result = asyncio.run(_screen_universe())
        assert result is None

    def test_single_bad_ticker_does_not_crash(self):
        """Bad data for one ticker must not abort the whole screen."""
        data = {
            f"T{i:02d}": {"closes": _rsi_neutral(), "volumes": [1_000_000] * _N}
            for i in range(10)
        }
        df = _make_df(data)
        # Corrupt one ticker's close column with NaN
        df.loc[:, ("T00", "Close")] = float("nan")
        with _use_pool(*data.keys()):
            with patch("main.yf.download", return_value=df):
                result = asyncio.run(_screen_universe(top_n=5))
        # Screen must still return a result from the remaining 9 good tickers
        assert result is not None and len(result) == 5


class TestExistingFilters:
    """CSP and LEAP recommendation filters must continue to enforce their gates."""

    # Required fields for _filter_csp_recommended
    _CSP_BASE = {
        "ticker": "AAPL", "warnings": [], "liquidity_score": 80,
        "iv_rank": 50, "score": 90, "above_sma50": True, "earnings_days_out": 30,
    }

    def test_csp_filter_blocks_warning_tickers(self):
        candidates = [
            {**self._CSP_BASE, "warnings": ["Earnings in 5 days"], "ticker": "WARN"},
            {**self._CSP_BASE, "warnings": [],                      "ticker": "CLEAN"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "WARN" not in tickers and "CLEAN" in tickers

    def test_csp_filter_requires_liq_50(self):
        candidates = [
            {**self._CSP_BASE, "liquidity_score": 49, "ticker": "LOW"},
            {**self._CSP_BASE, "liquidity_score": 50, "ticker": "OK"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "LOW" not in tickers and "OK" in tickers

    def test_csp_filter_requires_iv_rank_30(self):
        candidates = [
            {**self._CSP_BASE, "iv_rank": 29, "ticker": "LOWIV"},
            {**self._CSP_BASE, "iv_rank": 30, "ticker": "OKIV"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "LOWIV" not in tickers and "OKIV" in tickers

    def test_csp_filter_requires_score_70(self):
        candidates = [
            {**self._CSP_BASE, "score": 69, "ticker": "LOWSCORE"},
            {**self._CSP_BASE, "score": 70, "ticker": "OKSCORE"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "LOWSCORE" not in tickers and "OKSCORE" in tickers

    def test_csp_filter_blocks_downtrend_ticker(self):
        candidates = [
            {**self._CSP_BASE, "above_sma50": False, "ticker": "DOWN"},
            {**self._CSP_BASE, "above_sma50": True,  "ticker": "UP"},
            {**self._CSP_BASE, "above_sma50": None,  "ticker": "UNKNOWN"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "DOWN" not in tickers
        assert "UP" in tickers
        assert "UNKNOWN" in tickers  # None = benefit of doubt

    def test_csp_filter_blocks_upcoming_earnings(self):
        candidates = [
            {**self._CSP_BASE, "earnings_days_out": 15, "ticker": "SOON"},
            {**self._CSP_BASE, "earnings_days_out": 22, "ticker": "SAFE"},
            {**self._CSP_BASE, "earnings_days_out": None, "ticker": "NODATE"},
        ]
        result = main._filter_csp_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "SOON" not in tickers
        assert "SAFE" in tickers
        assert "NODATE" in tickers   # None = no known earnings date, allow

    def test_leap_filter_requires_liq_60(self):
        candidates = [
            {"ticker": "LOW",  "warnings": [], "liquidity_score": 59, "score": 80},
            {"ticker": "OK",   "warnings": [], "liquidity_score": 60, "score": 80},
            {"ticker": "HIGH", "warnings": [], "liquidity_score": 75, "score": 80},
        ]
        result = main._filter_leap_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "LOW" not in tickers
        assert "OK" in tickers
        assert "HIGH" in tickers

    def test_leap_filter_requires_no_warnings(self):
        candidates = [
            {"ticker": "WARN",  "warnings": ["IV too high"], "liquidity_score": 80, "score": 80},
            {"ticker": "CLEAN", "warnings": [],              "liquidity_score": 80, "score": 80},
        ]
        result = main._filter_leap_recommended(candidates)
        tickers = [r["ticker"] for r in result]
        assert "WARN" not in tickers and "CLEAN" in tickers


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — SYSTEM INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

BASE = "http://localhost:8000"


def _server_up() -> bool:
    try:
        import httpx
        httpx.get(f"{BASE}/status", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


@pytest.mark.sit
class TestSIT:

    @pytest.fixture(autouse=True)
    def require_server(self):
        if not _server_up():
            pytest.skip("Backend not running on localhost:8000")

    def test_get_universe_has_required_fields(self):
        import httpx
        r = httpx.get(f"{BASE}/csp/universe", timeout=10)
        assert r.status_code == 200
        body = r.json()
        for field in ("universe", "count", "candidate_pool", "last_screened", "scores"):
            assert field in body, f"Missing '{field}' in /csp/universe response"

    def test_universe_count_matches_list_length(self):
        import httpx
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        assert body["count"] == len(body["universe"])

    def test_candidate_pool_size_reported_correctly(self):
        import httpx
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        assert body["candidate_pool"] == len(CANDIDATE_POOL)

    def test_add_ticker_appears_in_universe(self):
        import httpx
        ticker = "TSTUNIVERSE"
        r = httpx.post(f"{BASE}/csp/universe/add", json={"ticker": ticker}, timeout=10)
        assert r.status_code == 200
        assert ticker in r.json()["universe"]

    def test_add_duplicate_ticker_is_idempotent(self):
        import httpx
        ticker = "AAPL"
        httpx.post(f"{BASE}/csp/universe/add", json={"ticker": ticker}, timeout=10)
        httpx.post(f"{BASE}/csp/universe/add", json={"ticker": ticker}, timeout=10)
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        assert body["universe"].count(ticker) == 1

    def test_refresh_returns_ok_structure(self):
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "universe"     in body
        assert "last_screened" in body
        assert "count"        in body

    def test_refresh_updates_last_screened_timestamp(self):
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        after = httpx.get(f"{BASE}/csp/universe", timeout=10).json()["last_screened"]
        assert after is not None
        datetime.fromisoformat(after)  # valid ISO 8601

    def test_refresh_honours_top_n_param(self):
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=15", timeout=120)
        body = r.json()
        if body["ok"]:
            assert len(body["universe"]) <= 15

    def test_refresh_returns_minimum_tickers(self):
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=15", timeout=120)
        body = r.json()
        if body["ok"]:
            assert len(body["universe"]) >= 5

    def test_scores_have_all_required_fields(self):
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        required = {"ticker", "score", "price", "avg_vol_30d", "rsi14", "above_sma50"}
        for row in body.get("scores", [])[:5]:   # spot-check top 5
            missing = required - set(row.keys())
            assert not missing, f"Score row missing: {missing}"

    def test_scores_are_sorted_descending(self):
        import httpx
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        scores = [r["score"] for r in body.get("scores", [])]
        if len(scores) > 1:
            assert scores == sorted(scores, reverse=True)

    def test_scan_cache_invalidated_after_refresh(self):
        """After refresh, a CSP scan forced-refresh must not crash."""
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        r = httpx.get(f"{BASE}/csp/scan?refresh=true", timeout=90)
        assert r.status_code == 200

    def test_refresh_top_n_boundary_min(self):
        """top_n=10 is the API minimum — must not 422."""
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        assert r.status_code == 200

    def test_refresh_top_n_boundary_max(self):
        """top_n=50 is the API maximum — must not 422."""
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=50", timeout=120)
        assert r.status_code == 200

    def test_refresh_top_n_below_min_is_rejected(self):
        """top_n=9 is below the minimum — API must return 422."""
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=9", timeout=10)
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — USER ACCEPTANCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.uat
class TestUAT:
    """End-to-end user journeys. Each test is a named user scenario."""

    @pytest.fixture(autouse=True)
    def require_server(self):
        if not _server_up():
            pytest.skip("Backend not running on localhost:8000")

    def test_user_opens_universe_tab_and_sees_data(self):
        """
        Scenario: User opens Universe tab for the first time.
        Expected: Tab renders; chips show active tickers, count is > 0.
        """
        import httpx
        r = httpx.get(f"{BASE}/csp/universe", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["universe"], list)
        assert body["count"] >= 1, "Must show at least 1 ticker"
        assert body["candidate_pool"] >= 100, "Pool size chip must show ≥100"
        assert "last_screened" in body     # may be None on first start

    def test_user_clicks_refresh_and_sees_updated_universe(self):
        """
        Scenario: User clicks 'Refresh Now' with default top_n=25.
        Expected: ok=True, chips refresh, last_screened timestamp appears.
        """
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=25", timeout=120)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True, f"Refresh failed: {body.get('error')}"
        assert 5 <= len(body["universe"]) <= 25
        assert body["last_screened"] is not None

        # GET confirms the new state is persisted
        after = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        assert after["last_screened"] == body["last_screened"]

    def test_user_sees_scores_table_with_readable_numbers(self):
        """
        Scenario: User reads the scores table after refresh.
        Expected: Numeric price, RSI, score; above_sma50 is boolean or null.
        """
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=15", timeout=120)
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        scores = body.get("scores", [])
        assert len(scores) > 0, "Scores table must be non-empty after refresh"
        for row in scores[:5]:
            assert isinstance(row["score"],  (int, float))
            assert isinstance(row["price"],  (int, float)) and row["price"] > 0
            assert isinstance(row["rsi14"],  (int, float))
            assert row["above_sma50"] in (True, False, None)

    def test_active_tickers_appear_in_scores_table(self):
        """
        Scenario: User sees chips at top and table below.
        Expected: Every chip ticker appears somewhere in the scores table.
        """
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=20", timeout=120)
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        active = set(body["universe"])
        scored = {r["ticker"] for r in body["scores"]}
        missing = active - scored
        assert not missing, f"Active tickers missing from scores table: {missing}"

    def test_active_tickers_are_top_scorers(self):
        """
        Scenario: User trusts that the chips show the BEST tickers.
        Expected: Active universe == top-N of the scores list.
        """
        import httpx
        top_n = 15
        httpx.post(f"{BASE}/csp/universe/refresh?top_n={top_n}", timeout=120)
        body = httpx.get(f"{BASE}/csp/universe", timeout=10).json()
        active = set(body["universe"])
        top_n_set = {r["ticker"] for r in body["scores"][:top_n]}
        not_top = active - top_n_set
        assert not not_top, f"Active tickers not in top-{top_n}: {not_top}"

    def test_user_selects_smaller_top_n_gets_fewer_tickers(self):
        """
        Scenario: User picks top_n=10 in the dropdown.
        Expected: Universe returns ≤10 tickers.
        """
        import httpx
        r = httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        body = r.json()
        if body["ok"]:
            assert len(body["universe"]) <= 10

    def test_user_manually_adds_a_ticker(self):
        """
        Scenario: User force-adds a ticker not produced by the screener.
        Expected: Appears in universe immediately without running a refresh.
        """
        import httpx
        special = "BRKA"
        r = httpx.post(f"{BASE}/csp/universe/add", json={"ticker": special}, timeout=10)
        assert r.status_code == 200
        assert special in r.json()["universe"]

    def test_csp_scan_succeeds_after_universe_refresh(self):
        """
        Scenario: User refreshes universe then immediately runs CSP scan.
        Expected: /csp/scan returns 200 (no server crash, no stale-state error).
        """
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        r = httpx.get(f"{BASE}/csp/scan?refresh=true", timeout=90)
        assert r.status_code == 200, f"/csp/scan: {r.status_code} — {r.text[:300]}"

    def test_leap_scan_succeeds_after_universe_refresh(self):
        """
        Scenario: User refreshes universe then runs LEAP scan.
        Expected: /leaps/scan returns 200.
        """
        import httpx
        httpx.post(f"{BASE}/csp/universe/refresh?top_n=10", timeout=120)
        r = httpx.get(f"{BASE}/leaps/scan?refresh=true", timeout=90)
        # Accept 200 or 503 (IBKR disconnected is OK — scan logic ran)
        assert r.status_code in (200, 503), f"/leaps/scan: {r.status_code} — {r.text[:300]}"
