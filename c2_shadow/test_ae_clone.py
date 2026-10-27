"""The live engine must reproduce the research backtest (same rules, same fills).
Replays ae_clone.step over the cached research prices for the out-of-sample window and compares the
total return with research/backtest_clone.py's +59.2% (S&P 100, 20/8/250 daily, cash in SPY)."""
import json
from pathlib import Path

import ae_clone as A

HERE = Path(__file__).resolve().parent


def replay(start="2024-07-01", end="2026-10-31"):
    px = {f.stem: json.loads(f.read_text()) for f in (HERE / "research" / "data" / "prices").glob("*.json")}
    universe = [s for s in A.UNIVERSE_FILE.read_text().split() if s in px]
    idx = {s: {b["d"]: i for i, b in enumerate(bars)} for s, bars in px.items()}
    days = [b["d"] for b in px["SPY"] if start <= b["d"] <= end]
    st = A.new_state(days[0], px["SPY"][idx["SPY"][days[0]]]["c"])
    prev = None
    for d in days:
        A.step(st, d, px, idx, universe, prev)
        prev = d
    return st


def test_matches_research_backtest():
    st = replay()
    total = st["curve"][-1][1] / A.P["CAPITAL"] - 1
    assert abs(total - 0.592) < 0.01, total
    assert len(st["closed"]) + len(st["pos"]) >= 116


def test_no_lookahead_pending_filled_next_open():
    st = replay("2025-01-01", "2025-03-31")
    for t in st["closed"]:
        assert t["closed"] > t["opened"]


def test_restart_midway_is_identical():
    """Stop halfway, save/load the state as JSON (a reboot), resume: same result as one straight run."""
    px = {f.stem: json.loads(f.read_text()) for f in (HERE / "research" / "data" / "prices").glob("*.json")}
    universe = [s for s in A.UNIVERSE_FILE.read_text().split() if s in px]
    idx = {s: {b["d"]: i for i, b in enumerate(bars)} for s, bars in px.items()}
    days = [b["d"] for b in px["SPY"] if "2025-01-01" <= b["d"] <= "2025-12-31"]
    st = A.new_state(days[0], px["SPY"][idx["SPY"][days[0]]]["c"])
    prev = None
    for k, d in enumerate(days):
        if k == len(days) // 2:
            st = json.loads(json.dumps(st))  # simulated save + restart
        A.step(st, d, px, idx, universe, prev)
        prev = d
    straight = replay("2025-01-01", "2025-12-31")
    assert st["curve"][-1][1] == straight["curve"][-1][1]
    assert st["closed"] == straight["closed"]


def test_missing_data_for_held_symbol_does_not_advance(tmp_path=None, monkeypatch=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    orig = (A.STATE, A.LOG, A.load_all, A.tg)
    try:
        A.STATE, A.LOG, A.tg = tmp / "s.json", tmp / "l.jsonl", lambda text: None
        st = {"cash": 1.0, "pos": {"AAPL": {"qty": 1, "entry": 1, "peak": 1, "held": 1, "opened": "x"}},
              "pending_buys": ["MSFT"], "pending_sells": [], "closed": [], "last_day": "2026-01-02",
              "start_day": "2026-01-02", "spy_start": 1, "curve": [["2026-01-02", 1, 1]]}
        A.STATE.write_text(json.dumps(st))
        A.load_all = lambda universe: ({"SPY": [{"d": "2026-01-05", "o": 1, "c": 1}]}, ["AAPL"])
        A.run()
        assert json.loads(A.STATE.read_text()) == st  # untouched, retried next run
    finally:
        A.STATE, A.LOG, A.load_all, A.tg = orig


if __name__ == "__main__":
    st = replay()
    print(f"replay total {st['curve'][-1][1] / A.P['CAPITAL'] - 1:+.2%}, closed {len(st['closed'])}, open {len(st['pos'])}")
