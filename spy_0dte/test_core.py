"""Regression tests. Run: ./.venv/bin/python test_core.py  (or pytest -q test_core.py)"""
import logging

import numpy as np
import pandas as pd

import spy0dte_framework as F

NY = F.NY
logging.disable(logging.WARNING)


# --- audit item 2: spread loss stop sign ------------------------------------
def _engine_with_short_put_spread(stop_mult):
    cfg = F.EngineConfig(spread_loss_stop_mult=stop_mult, commission_per_contract=0.0,
                         quote=F.QuoteModel(lam=0.0))
    eng = F.EntryExitEngine(cfg)
    S, mins, iv = 500.0, 240, 0.15
    legs = [F.Leg("P", 497.0, -1), F.Leg("P", 494.0, +1)]
    fills = [F.bs_price(S, lg.strike, mins, iv, "P") for lg in legs]
    eng.position = F.Position("short_put_spread", legs, pd.Timestamp("2025-03-03 10:30", tz=NY), S, fills,
                              pd.Timestamp("2025-03-03", tz=NY),
                              audit=dict(signal_time=pd.Timestamp("2025-03-03 10:29", tz=NY), signal_close=S,
                                         regime=F.CHOP, p_dir=(0.2, 0.6, 0.2), p_spread=(0.2, 0.6, 0.2),
                                         iv_signal=iv, ev_usd=0.0, ev_components={}, scenario_iv=iv,
                                         model_deltas=[-0.3, -0.15], entry_mids=fills, entry_widths=[0, 0],
                                         pricer="black_scholes", lam=0.0))
    return eng, iv


def test_spread_stop_fires_on_loss():
    eng, iv = _engine_with_short_put_spread(stop_mult=1.0)
    ts = pd.Timestamp("2025-03-03 11:00", tz=NY)
    eng._manage(ts, 490, 490.5, 489.5, 490.0, 210, 209, iv)           # SPY collapses through both strikes
    assert eng.position is None and eng.trades[-1]["reason"] == "spread_loss_stop"
    assert eng.trades[-1]["pnl_usd"] < 0


def test_spread_stop_does_not_fire_on_profit():
    eng, iv = _engine_with_short_put_spread(stop_mult=0.1)            # tiny threshold: the old bug fired here
    ts = pd.Timestamp("2025-03-03 11:00", tz=NY)
    eng._manage(ts, 506, 506.5, 505.5, 506.0, 210, 209, iv)           # SPY rallies away: a winner
    assert eng.position is not None, "loss stop fired on a winning spread"


# --- audit item 1: labels anchored on the executable entry ------------------
def _one_session(opens, closes, highs, lows, iv=0.20):
    idx = pd.date_range("2025-03-03 09:30", periods=len(opens), freq="1min", tz=NY)
    raw = pd.DataFrame(dict(open=opens, high=highs, low=lows, close=closes, volume=1000, iv=iv), index=idx)
    feats = F.prepare_bars(raw)
    feats["iv_used"] = iv
    return feats


def test_label_anchored_on_next_open():
    # close[0]=100 but the fill is open[1]=101; bar 1 reaches 101.03. Barrier ~ $0.048.
    #   from close[0]: 101.03 >= 100.048 -> "up"   (the move is already in the fill)
    #   from open[1]:  101.03 <  101.048 -> "chop" (what the trade actually got)
    feats = _one_session([100.0, 101.0, 101.0, 101.0, 101.0], [100.0, 101.0, 101.0, 101.0, 101.0],
                         [100.0, 101.03, 101.0, 101.0, 101.0], [100.0, 100.99, 101.0, 101.0, 101.0])
    y = F.TripleBarrierLabeler(F.LabelConfig(horizon=1, barrier_k=0.75)).label(feats)
    assert y.iloc[0] == F.CLASS_CHOP, f"label measured from close[t] instead of open[t+1]: got {y.iloc[0]}"
    assert np.isnan(y.iloc[-1]), "last bar of the session has no fill and must be unlabeled"


# --- audit item 5: no overnight return inside rv30 --------------------------
def test_rv30_excludes_overnight_gap():
    raw = F.synthetic_spy(n_sessions=3, seed=5)
    day2 = raw.index.normalize().unique()[1]
    raw.loc[raw.index.normalize() == day2, ["open", "high", "low", "close"]] *= 1.05   # +5% overnight gap
    feats = F.FeatureBuilder().transform(F.prepare_bars(raw))
    # bars 1-28: the 30-bar window still reaches back across 09:30, so the old code
    # (C / C.shift(1)) would include the 5% overnight return here
    first30 = feats[(feats["session"] == day2) & (feats["tau"].between(1, 28))]["rv30"]
    baseline = feats[(feats["session"] == day2) & (feats["tau"].between(200, 240))]["rv30"].median()
    assert (first30 < 5 * baseline).all(), "rv30 still contains the overnight gap"


# --- audit item 19: look-ahead test, many cut points, and proof it can fail -
def test_no_lookahead_100_cuts():
    bars = F.prepare_bars(F.synthetic_spy(n_sessions=30, seed=3))
    F.assert_no_lookahead(bars, F.FeatureBuilder(), F.RegimeClassifier(), n_checks=100)


def test_no_lookahead_catches_a_planted_leak():
    class Leaky(F.FeatureBuilder):
        def transform(self, bars):
            df = super().transform(bars)
            df["f3_rsi"] = df["close"].rolling(21, center=True, min_periods=1).mean()   # uses future bars
            return df
    bars = F.prepare_bars(F.synthetic_spy(n_sessions=30, seed=3))
    try:
        F.assert_no_lookahead(bars, Leaky(), n_checks=20)
    except AssertionError:
        return
    raise AssertionError("assert_no_lookahead did not detect a centered (future) window")


# --- audit item 10: holdout is locked ----------------------------------------
def test_holdout_is_locked():
    feats = F.FeatureBuilder().transform(F.prepare_bars(F.synthetic_spy(n_sessions=160, seed=9)))
    hold = feats["session"].unique()[-30]
    cfg = F.ModelConfig(min_train_sessions=60, min_calib_per_class=5, holdout_start=str(hold.date()))
    p = F.WalkForwardModel(config=cfg).fit_predict(feats)
    assert p.loc[feats["session"] >= hold, "p_up"].isna().all(), "model produced holdout predictions while locked"
    assert p["p_up"].notna().any()


# --- audit item 15: EV responds to the model's probabilities ----------------
def test_ev_moves_with_probabilities():
    eng = F.EntryExitEngine()
    ts, sess = pd.Timestamp("2025-03-03 10:30", tz=NY), pd.Timestamp("2025-03-03", tz=NY)
    bull = eng._evaluate("long_call", ts, sess, 500.0, 330, 0.15, (0.05, 0.15, 0.80))
    bear = eng._evaluate("long_call", ts, sess, 500.0, 330, 0.15, (0.80, 0.15, 0.05))
    assert bull["ev_usd"] > bear["ev_usd"], "long-call EV must rise with P(up)"
    calm = eng._evaluate("short_put_spread", ts, sess, 500.0, 330, 0.15, (0.05, 0.90, 0.05))
    crash = eng._evaluate("short_put_spread", ts, sess, 500.0, 330, 0.15, (0.80, 0.15, 0.05))
    assert calm["ev_usd"] > crash["ev_usd"], "put-spread EV must fall as P(down breach) rises"
    width = abs(calm["legs"][0].strike - calm["legs"][1].strike)
    assert 2.0 <= width <= 5.0, f"spread width {width} outside the configured bounds"


# --- review 2, item 3: a real-price run never exits on a model price ----------
def _bars_for(symbol_prices: dict) -> dict:
    out = {}
    for sym, series in symbol_prices.items():
        idx = pd.DatetimeIndex([pd.Timestamp(t, tz=NY) for t in series])
        px = list(series.values())
        out[sym] = pd.DataFrame(dict(open=px, high=px, low=px, close=px, vwap=px, volume=10), index=idx)
    return out


def test_missing_exit_price_marks_trade_invalid():
    sess = pd.Timestamp("2025-03-03", tz=NY)
    legs = [F.Leg("P", 497.0, -1), F.Leg("P", 494.0, +1)]
    # prints only in the morning: nothing within 5 minutes of the 15:45 exit
    pricer = F.BarPricer(_bars_for({F.occ_symbol("SPY", sess, "P", 497.0): {"2025-03-03 10:30": 1.20},
                                    F.occ_symbol("SPY", sess, "P", 494.0): {"2025-03-03 10:30": 0.60}}))
    eng = F.EntryExitEngine(F.EngineConfig(commission_per_contract=0.0, quote=F.QuoteModel(lam=0.0)), pricer=pricer)
    eng.position = F.Position("short_put_spread", legs, pd.Timestamp("2025-03-03 10:30", tz=NY), 500.0, [1.20, 0.60],
                              sess, audit=dict(signal_time=pd.Timestamp("2025-03-03 10:29", tz=NY), signal_close=500.0,
                                               regime=F.CHOP, p_dir=(0.2, 0.6, 0.2), p_spread=(0.2, 0.6, 0.2),
                                               iv_signal=0.15, ev_usd=10.0, ev_components={}, scenario_iv=0.15,
                                               model_deltas=[-0.3, -0.15], entry_mids=[1.2, 0.6], entry_widths=[0, 0],
                                               pricer="option_bars", lam=0.0, entry_price_source="real",
                                               entry_staleness_s=0.0))
    eng._close(pd.Timestamp("2025-03-03 15:45", tz=NY), 499.0, 15, 0.15, "eod_1545")
    t = eng.trades[-1]
    assert t["invalid"] and np.isnan(t["pnl_usd"]), "exit without a real print must not produce a P&L"
    assert t["exit_price_source"] == "missing"
    assert np.isclose(t["pnl_usd_worst"], (0.60 - 3.0) * 100), "worst case = credit - width"


# --- review 2, item 1: EV is recomputed at the fill bar's prices --------------
def test_ev_rechecked_at_fill_prices():
    sess = pd.Timestamp("2025-03-03", tz=NY)
    sig, fill = "2025-03-03 10:29", "2025-03-03 10:30"
    probe = F.EntryExitEngine()
    legs = probe._legs_for("long_call", sess, 500.0, 330, 0.15)
    sym = F.occ_symbol("SPY", sess, "C", legs[0].strike)
    fair = F.bs_price(500.0, legs[0].strike, 330, 0.15, "C")
    # cheap at the signal minute, 3x more expensive at the fill minute
    pricer = F.BarPricer(_bars_for({sym: {sig: fair * 0.5, fill: fair * 3.0}}))
    # EV ~ +$39 at the signal price, ~ +$15 at the fill price: a $25 buffer
    # passes the first and must reject the second.
    eng = F.EntryExitEngine(F.EngineConfig(ev_buffer_usd=25.0), pricer=pricer)
    d = eng._evaluate("long_call", pd.Timestamp(sig, tz=NY), sess, 500.0, 330, 0.15, (0.05, 0.15, 0.80))
    assert d["ev_usd"] > 25.0
    base = dict(signal_time=pd.Timestamp(sig, tz=NY), signal_close=500.0, regime=F.BREAKOUT,
                p_dir=(0.05, 0.15, 0.80), p_spread=(0.05, 0.15, 0.80), iv_signal=0.15, atr=0.5, T0=330)
    eng._open(pd.Timestamp(fill, tz=NY), 500.0, 330, 0.15, sess, {**base, **d})
    assert eng.position is None and eng.stats["fill_abstain_ev"] == 1, "trade accepted on stale signal-bar EV"


# --- option-return experiment: candidate features never see the future -------
def test_option_candidates_are_causal():
    import option_returns as R
    rng = np.random.default_rng(0)
    sess = pd.Timestamp("2025-03-03", tz=NY)
    n = 390
    S = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    day = pd.DataFrame({c: rng.normal(size=n) for c in R.UNDERLYING_FEATURES})
    day["rv30"], day["close"], day["ready"], day["session_len"], day["session"] = 0.15, S, True, n, sess
    day.index.name = "tau"
    recs = []
    for m in range(n):
        for k in range(94, 107):
            for r in ("C", "P"):
                if rng.random() < 0.8:
                    px = F.bs_price(S[m], k, n - m, 0.18, r) * (1 + rng.normal(0, 0.02))
                    recs.append(dict(time=sess + pd.Timedelta(minutes=570 + m), symbol=F.occ_symbol("SPY", sess, r, k),
                                     vwap=max(px, 0.01), close=max(px, 0.01), volume=5))
    opt = pd.DataFrame(recs).set_index("time")
    q = F.QuoteModel(lam=0.25)
    tau = 60
    full = pd.DataFrame(R.build_day(R.DayChain(opt, sess, n), day, q))
    cut = opt[opt.index <= sess + pd.Timedelta(minutes=570 + tau + 1)]          # nothing after the fill minute
    trunc = pd.DataFrame(R.build_day(R.DayChain(cut, sess, n), day, q))
    a, b = full[full.tau == tau].reset_index(drop=True), trunc[trunc.tau == tau].reset_index(drop=True)
    assert len(a) > 0 and len(a) == len(b), "candidate set changed when future prints were removed"
    cols = R.FEATURES + ["strikes", "entry_mid"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)
    assert b["pnl"].isna().all() and a["pnl"].notna().any(), "exit P&L must need prints after entry"


# --- last-hour test: strikes/entries never see prints after the fill minute ---
def test_last_hour_candidates_are_causal():
    import last_hour as L
    import option_returns as R
    rng = np.random.default_rng(1)
    sess, n = pd.Timestamp("2025-03-03", tz=NY), 390
    S = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    recs = []
    for m in range(300, n):
        for k in range(90, 111):
            for r in ("C", "P"):
                px = max(F.bs_price(S[m], k, n - m, 0.18, r) * (1 + rng.normal(0, 0.02)), 0.01)
                recs.append(dict(time=sess + pd.Timedelta(minutes=570 + m), symbol=F.occ_symbol("SPY", sess, r, k),
                                 vwap=px, close=px, volume=5))
    opt = pd.DataFrame(recs).set_index("time")
    full = pd.DataFrame(L.build_day(R.DayChain(opt, sess, n), S, sess))
    for tau in L.SIGNAL_TAUS:
        cut = opt[opt.index <= sess + pd.Timedelta(minutes=570 + tau + 1)]
        trunc = pd.DataFrame(L.build_day(R.DayChain(cut, sess, n), S, sess))
        assert len(trunc) > 0, f"no candidates at tau {tau} once future prints were removed"
        a = full[full.tau == tau].reset_index(drop=True)
        b = trunc[trunc.tau == tau].reset_index(drop=True)
        assert len(a) == 4 and len(b) == 4, f"expected all 4 structures at tau {tau}"
        pd.testing.assert_frame_equal(a[["struct", "strikes", "u", "leg1_entry", "leg2_entry"]],
                                      b[["struct", "strikes", "u", "leg1_entry", "leg2_entry"]])
        assert b["leg1_exit"].isna().all(), "exit must need prints after entry"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
