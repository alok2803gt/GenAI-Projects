"""Offline tests for vnkumar_shadow.py's message handling -- every case is a real message from the
channel (2026-10-08 .. 10-16)."""
from datetime import date

import vnkumar_shadow as V

REF = date(2026, 10, 15)


def pos(pid, ticker, legs, expiry, mentioned, his=1.0):
    return pid, {"id": pid, "ticker": ticker, "legs": legs, "expiry": expiry, "his": his, "qty": 1.0,
                 "mentioned": mentioned}


def test_entries():
    e = V.parse_entries("$SPY 773c 10/16 0.54\nSmall size lotto", REF)
    assert e == [{"ticker": "SPY", "legs": [(773.0, "C", 1)], "expiry": "2026-10-16", "his": 0.54, "kind": "single"}]
    assert V.parse_entries("$SPY 770p 10/30 $5.85\nStop loss $3", REF)[0]["his"] == 5.85
    assert V.parse_entries("Try $770p $SPY 10/09 $0.9\nLotto size", REF)[0]["legs"] == [(770.0, "P", 1)]
    s = V.parse_entries("$SPY 750/735 debit put spread 11/20 expiration for $2.65\n\nBuy 750 put \nSell 735 put", REF)[0]
    assert s["legs"] == [(750.0, "P", 1), (735.0, "P", -1)] and s["expiry"] == "2026-11-20" and s["his"] == 2.65
    a = V.parse_entries("$AMD 615/620 debit call spread 10/16 expiration for $1.35", REF)[0]
    assert a["legs"] == [(615.0, "C", 1), (620.0, "C", -1)]
    assert V.parse_entries("$BABA 108p 10/16 $0.7\nSmall size", REF)[0]["ticker"] == "BABA"


def test_recaps_and_closes_never_open():
    assert V.parse_entries("—-\nOpen positions \n\n$SPY 774c 10/16 $1.1\nSwing trade", REF) == []
    assert V.parse_entries("Closing half $750p for $3.9", REF) == []


def test_exit_matching():
    opens = dict([pos("1", "SPY", [(770.0, "P", 1)], "2026-10-30", 5),
                  pos("2", "SPY", [(778.0, "C", 1)], "2026-10-09", 7),
                  pos("3", "SPY", [(750.0, "P", 1), (735.0, "P", -1)], "2026-11-20", 3),
                  pos("4", "BABA", [(108.0, "P", 1)], "2026-10-16", 2)])
    assert V.match_exit("Closed for 0.75\n50% up", opens, REF) == (["2"], "inferred: most recently mentioned")
    assert V.match_exit("Closing 770p for $5\nWill see next week", opens, REF)[0] == ["1"]
    assert V.match_exit("Closed for $1 $BABA", opens, REF)[0] == ["4"]
    assert V.match_exit("Holding 750/735 put spread", opens, REF)[0] == ["3"]
    assert V.match_exit("Closure 770p fo $5.6", opens, REF)[0] == ["1"]
    two = dict([pos("1", "SPY", [(770.0, "P", 1)], "2026-10-16", 1), pos("2", "SPY", [(770.0, "P", 1)], "2026-10-09", 2)])
    assert V.match_exit("Please close 770p 10/16 near open", two, REF)[0] == ["1"], "expiry breaks the tie"
    assert V.match_exit("Closing 770p for $3", two, REF)[0] == [], "two 770p and no expiry: don't guess"


def test_exit_bugs_of_oct_21_to_23():
    opens = dict([pos("3", "SPCX", [(162.5, "C", 1)], "2026-10-23", 1),
                  pos("4", "SPCX", [(135.0, "P", 1)], "2026-10-23", 2),
                  pos("5", "SPY", [(762.0, "P", 1)], "2026-10-23", 3)])
    ids, how = V.match_exit("Closed $SPcX STRANGLE for $3.75", opens, REF)
    assert sorted(ids) == ["3", "4"], (ids, how)  # both SPCX legs -- never the SPY 762P
    two = dict([pos("7", "SPY", [(775.0, "C", 1)], "2026-10-23", 1),
                pos("8", "AMD", [(570.0, "C", 1), (580.0, "C", -1)], "2026-10-23", 2)])
    assert sorted(V.match_exit("Closing $SPY and $AMD near open", two, REF)[0]) == ["7", "8"]
    nb = dict([pos("9", "NBIS", [(180.0, "C", 1)], "2026-10-30", 1)])
    assert V.match_exit("Closing NVDA", nb, REF)[0] == [], "NVDA isn't held: never close NBIS"
    assert V.named_tickers("Closed $SPcX STRANGLE for $3.75") == {"SPCX"}


def test_plans_are_not_exits():
    for msg in ("Will try to hold this week\nIf it doesn’t work will close on Thursday. \nDon’t average",
                "Will close half $BABA 108p near open",
                "774c 50% up $SPY\nYou can close half here or hold for tomorrow."):
        assert V.RE_CLOSE.search(msg) and V.RE_ADVICE.search(msg), msg
    for msg in ("Closing for $1.56\nWill revisit and add the trade evening.", "Closed 774c full position",
                "Please close 782c", "Plz close it .."):
        assert not V.RE_ADVICE.search(msg), msg


def test_his_exit_price():
    p = {"his": 1.5}
    assert V.his_exit_price("Closed 771c for $0.85", p) == 0.85
    assert V.his_exit_price("Closure 770p fo $5.6", p) == 5.6
    assert V.his_exit_price("Closing 772p for break even", p) == 1.5
    assert V.his_exit_price("Closed 774c full position", p) is None


def test_prices_cross_the_spread():
    p = {"legs": [(750.0, "P", 1), (735.0, "P", -1)]}
    q = [(5.90, 5.95), (3.20, 3.25)]
    assert V.buy_cost(p, q) == 2.75 and V.sell_value(p, q) == 2.65


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
