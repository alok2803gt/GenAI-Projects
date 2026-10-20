"""Paper SPX-only mode (laptop, 2026-10-19): a VIRTUAL SPY position runs every iMac exit rule, and its
close sells the SPXW twin. Run from a directory whose ashleyklieu_paper_mode.json has
"spy_trade": false and "spx_mirror": true (this test writes one into its own temp dir)."""
import asyncio
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp()
with open(os.path.join(TMP, "ashleyklieu_paper_mode.json"), "w") as f:
    json.dump({"port": 7497, "spx_mirror": True, "spx_qty": 1, "spy_trade": False}, f)
os.chdir(TMP)
sys.path.insert(0, HERE)
import ashleyklieu_trigger_executor as ax  # noqa: E402  (reads the flag file from the cwd)


class FakeIB:
    def __init__(self, bid, ask):
        self.bid, self.ask = bid, ask

    async def qualifyContractsAsync(self, *cs):
        for c in cs:
            c.conId = 1
        return cs

    def reqMktData(self, c, *a):
        return SimpleNamespace(bid=self.bid, ask=self.ask)

    def cancelMktData(self, c):
        pass


def test_mode_is_spx_only_paper():
    assert ax.MODE["paper"] and not ax.MODE["spy_trade"] and ax.MODE["spx_mirror"]
    assert ax.FORCE_CLOSE_ET == (15, 45)


def test_virtual_close_sells_twin_and_only_then_counts_as_closed():
    calls = []

    async def fake_ladder(ib, c, action, label, qty):
        calls.append((action, c.lastTradeDateOrContractMonth, c.strike))
        return fills.pop(0)

    ax.spx_ladder = fake_ladder
    exp = date.today().strftime("%Y%m%d")
    setups = [{"name": "768P", "strike": 768.0, "right": "P", "lo": 768, "hi": 768.3}]
    pos = {"virtual": True, "fill_px": 1.2, "qty": 1, "expiry": exp, "strike": 768.0, "right": "P"}

    ax.SPX_OPEN.clear()
    ax.SPX_OPEN["768P"] = {"conId": 9, "strike": 7705.0, "right": "P", "qty": 1, "fill_px": 11.5, "expiry": exp}
    fills = [(False, None), (True, 8.4)]
    ok = asyncio.run(ax.close_position(FakeIB(0.8, 0.82), "768P", pos, setups, "trailing stop"))
    assert ok is False and "768P" in ax.SPX_OPEN, "twin didn't sell -> virtual leg stays open, retried later"
    ok = asyncio.run(ax.close_position(FakeIB(0.8, 0.82), "768P", pos, setups, "trailing stop"))
    assert ok is True and "768P" not in ax.SPX_OPEN
    assert calls == [("SELL", exp, 7705.0)] * 2, calls


def test_twin_closes_against_its_own_3dte_expiry():
    seen = []

    async def fake_ladder(ib, c, action, label, qty):
        seen.append(c.lastTradeDateOrContractMonth)
        return True, 9.0

    ax.spx_ladder = fake_ladder
    exp3 = (date.today() + timedelta(days=3)).strftime("%Y%m%d")
    ax.SPX_OPEN.clear()
    ax.SPX_OPEN["770C"] = {"conId": 9, "strike": 7725.0, "right": "C", "qty": 1, "fill_px": 8.0, "expiry": exp3}
    asyncio.run(ax.spx_mirror_exit(FakeIB(1, 1), "770C", ax.SPX_OPEN, "her TP"))
    assert seen == [exp3]


def test_mirror_file_keeps_unexpired_twins_across_days():
    today = date.today().strftime("%Y%m%d")
    later = (date.today() + timedelta(days=3)).strftime("%Y%m%d")
    old = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    with open(ax.SPX_MIRROR_FILE, "w") as f:
        json.dump({"date": (date.today() - timedelta(days=1)).isoformat(),
                   "open": {"a": {"expiry": later}, "b": {"expiry": today}, "c": {"expiry": old}, "d": {}}}, f)
    assert set(ax.load_spx_mirror()) == {"a", "b"}, "carried 3DTE + today's kept; expired and legacy-0DTE dropped"


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("PASS", n)
