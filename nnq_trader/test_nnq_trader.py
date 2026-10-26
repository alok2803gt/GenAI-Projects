"""Offline tests for nnq_trader.py: signal rule, FIFO P&L, trading-hours parsing, tick rounding."""
from datetime import datetime
from zoneinfo import ZoneInfo

import nnq_trader as N


def rd(up):
    return {"up": up, "down": 100 - up}


def test_hold_until_opposite_signal():
    assert N.target_position(rd(61), 0) == 1
    assert N.target_position(rd(39), 0) == -1          # DOWN 61%
    assert N.target_position(rd(60), 0) == 1, "exactly 60 trades (>=), same as the channel rule"
    assert N.target_position(rd(40), 0) == -1, "DOWN exactly 60 shorts"
    assert N.target_position(rd(59), 0) == 0
    assert N.target_position(rd(50), 1) == 1, "50/50 never moves the position"
    assert N.THRESHOLD == N.BROADCAST_MIN, "trading and channel thresholds must stay the same"
    assert N.target_position(rd(55), 1) == 1, "a weak read keeps the long"
    assert N.target_position(rd(45), -1) == -1, "a weak read keeps the short"
    assert N.target_position(rd(30), 1) == -1, "DOWN 70% flips a long to short"
    assert N.target_position(rd(75), -1) == 1


def test_fifo_pnl_with_flip():
    fills = [{"side": "BUY", "qty": 1, "px": 29600.0}, {"side": "SELL", "qty": 2, "px": 29650.0},
             {"side": "BUY", "qty": 1, "px": 29620.0}]
    pnl, pos, avg = N.realized(fills)
    assert pos == 0 and avg is None
    assert pnl == round((50 + 30) * 0.2, 2), pnl  # +50 pts long, +30 pts short, $0.20/pt


def test_trading_hours_and_break():
    s = N.parse_hours("20261018:1700-20261019:1600;20261019:1700-20261020:1600;20261024:CLOSED", "US/Central")
    et = ZoneInfo("America/New_York")
    assert N.is_open(s, datetime(2026, 10, 19, 10, 0, tzinfo=et))
    assert not N.is_open(s, datetime(2026, 10, 19, 17, 30, tzinfo=et)), "17:00-18:00 ET daily break"
    assert N.is_open(s, datetime(2026, 10, 19, 18, 30, tzinfo=et))


def test_channel_html_escapes_then_bolds():
    out = N.to_html("**Bias:** Bullish (RSI < 30 & rising) -> **UP**")
    assert out == "<b>Bias:</b> Bullish (RSI &lt; 30 &amp; rising) -&gt; <b>UP</b>", out


def test_tick_rounding_is_marketable():
    assert N.round_tick(29640.3, True) == 29640.5 and N.round_tick(29640.3, False) == 29640.0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
