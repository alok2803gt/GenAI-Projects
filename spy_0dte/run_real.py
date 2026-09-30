"""
Development-period report on real SPY history. The final 12 months are the
locked holdout: nothing here trains on, predicts, or reports them (see
run_holdout.py for the one-time final evaluation).

    ./.venv/bin/python run_real.py                  # standard report
    ./.venv/bin/python run_real.py --ablation       # + session-reset indicator comparison
    ./.venv/bin/python run_real.py --option-bars data/SPY_0dte_option_bars   # real traded option prices

Sections: 1 regime frequency, 2 regime vs realized day type, 3 model scorecards
vs base-rate and logistic baselines, 4 calibration by session phase,
5 engine at three execution-quality levels (lambda = 0 / 0.25 / 0.5).
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import spy0dte_framework as F

HOLDOUT_START = F.HOLDOUT_START   # locked: the last 12 months of the downloaded history

ap = argparse.ArgumentParser()
ap.add_argument("--bars", default="data/SPY_1min_sip.parquet")
ap.add_argument("--option-bars", default=None, help="directory of per-day option bar parquet files")
ap.add_argument("--ablation", action="store_true")
args = ap.parse_args()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
t0 = time.time()
hold = pd.Timestamp(HOLDOUT_START, tz=F.NY)

raw = pd.read_parquet(args.bars)[["open", "high", "low", "close", "volume"]]
bars = F.prepare_bars(raw)
feats = F.FeatureBuilder().transform(bars)
regimes = F.RegimeClassifier().classify(feats)
dev = feats["session"] < hold
print(f"bars {len(bars):,}, sessions {bars['session'].nunique()} ({bars.index.min().date()} .. {bars.index.max().date()})")
print(f"DEVELOPMENT period: sessions before {HOLDOUT_START} ({feats.loc[dev, 'session'].nunique()} sessions). "
      f"Holdout ({feats.loc[~dev, 'session'].nunique()} sessions) is locked.")
print("inputs: SPY OHLCV only -> realized-vol IV proxy; VRP and VIX regime conditions disabled\n")

# 1 -----------------------------------------------------------------------------
at10 = regimes.loc[(feats["tau"] == 29) & dev, "regime"]
print("1) REGIME AT 10:00 (development sessions):", at10.value_counts().to_dict())

# 2 -----------------------------------------------------------------------------
print("\n2) 10:00 CALL vs REALIZED DAY TYPE (development sessions, ex-post)")
print(F.regime_validation_report(feats[dev], regimes[dev], checkpoint=30).to_string())

# 3 -----------------------------------------------------------------------------
dmod = F.WalkForwardModel(config=F.ModelConfig(label=F.DIRECTION_LABEL, holdout_start=HOLDOUT_START))
smod = F.WalkForwardModel(config=F.ModelConfig(label=F.SPREAD_LABEL, holdout_start=HOLDOUT_START))
p_dir, p_spr = dmod.fit_predict(feats, "p_"), smod.fit_predict(feats, "s_")
assert p_dir.loc[~dev, "p_up"].isna().all() and p_spr.loc[~dev, "s_up"].isna().all(), "holdout leaked"
print(f"\n3) MODELS ({time.time() - t0:.0f}s) -- direction: {len(dmod.fold_log)} folds, spread: {len(smod.fold_log)} folds; "
      f"median trees after early stopping: direction {int(np.median([f['trees'] for f in dmod.fold_log]))}, "
      f"spread {int(np.median([f['trees'] for f in smod.fold_log]))}; "
      f"folds calibrated (>= {dmod.cfg.min_calib_per_class}/class): "
      f"{sum(f['calibrated'] for f in dmod.fold_log)}/{len(dmod.fold_log)} and "
      f"{sum(f['calibrated'] for f in smod.fold_log)}/{len(smod.fold_log)}")
for name, m in (("DIRECTION model (30-min barriers, used for long options)", dmod),
                ("SPREAD model (inside +/-0.524 implied move until 15:45, used for spreads)", smod)):
    print(f"\n   {name}  -- t < 0 means better than that baseline")
    print(F.evaluate_forecasts(feats, m.labels, m.variants, m.step()).round(4).to_string())

# 4 -----------------------------------------------------------------------------
print("\n4) CALIBRATION ERROR BY SESSION PHASE (direction model, engine variant "
      f"'{dmod.cfg.calibration}'; phase 0=open 1=AM 2=lunch 3=PM 4=close)")
print(F.calibration_by_phase(feats, dmod.labels, p_dir, dmod.step()).round(4).to_string())

# 5 -----------------------------------------------------------------------------
frame = pd.concat([feats, regimes, p_dir, p_spr], axis=1)
pricer, obars = None, None
if args.option_bars:
    parts = sorted(Path(args.option_bars).glob("*.parquet"))
    ob = pd.concat([pd.read_parquet(p) for p in parts]) if parts else pd.DataFrame()
    obars = {s: g.drop(columns="symbol") for s, g in ob.groupby("symbol")} if len(ob) else None
    pricer = F.BarPricer(obars) if obars else None
    covered = sorted({pd.Timestamp(p.stem, tz=F.NY) for p in parts})
    if covered:
        frame = frame[(frame["session"] >= covered[0]) & (frame["session"] <= covered[-1])]
    print(f"\n   option bars: {len(ob):,} rows over {len(covered)} sessions -> engine limited to those sessions")
label = ("traded option prints (1-min VWAP as mid) + MODELED bid/ask cost" if pricer
         else "THEORETICAL Black-Scholes prices -- plumbing only")
print(f"\n5) ENGINE -- EV rule, {label}")
print("   signal at bar t close picks the structure; strikes, prices and EV are redone at t+1 open")
print("   total_usd excludes trades with no real exit print (n_invalid); total_worst_usd books them at max loss")


def summarize(tr, **tags):
    out = []
    for kind, g in (tr.groupby("kind") if len(tr) else []):
        v = g[~g["invalid"]]
        out.append(dict(**tags, kind=kind, n=len(g), n_invalid=int(g["invalid"].sum()), win=(v.pnl_usd > 0).mean(),
                        avg_usd=v.pnl_usd.mean(), total_usd=v.pnl_usd.sum(), total_worst_usd=g.pnl_usd_worst.sum(),
                        ev_signal=g.ev_usd_signal.mean(), ev_fill=g.ev_usd.mean()))
    return out


rows = []
pricers = [("real_bars", pricer), ("black_scholes", None)] if pricer else [("black_scholes", None)]
fails = []
for pname, pr in pricers:
    for lam in (0.0, 0.25, 0.5):
        eng = F.EntryExitEngine(F.EngineConfig(quote=F.QuoteModel(lam=lam)), pricer=pr)
        tr, _ = eng.run(frame)
        if lam == 0.25 and len(tr) and pname == pricers[0][0]:
            tr.to_csv(f"data/trades_audit_{pname}_lambda025.csv", index=False)
            fails = F.check_engine_invariants(tr, frame, eng.cfg)
        rows += summarize(tr, pricer=pname, lam=lam)
        print(f"   {pname:<13} lambda {lam}: {eng.stats}")
if rows:
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("   engine invariants (lambda 0.25):", "ALL PASS" if not fails else fails[:5])
    print(f"   per-trade audit records: data/trades_audit_{pricers[0][0]}_lambda025.csv")

# 5b stale-print tolerance sweep (review 2, item 4) ------------------------------------
if obars:
    print("\n5b) STALE-PRINT TOLERANCE (real prints, lambda 0.25). Profit rising with tolerance = red flag.")
    srows, fnl = [], []
    for stale in (0, 1, 2, 5):
        eng = F.EntryExitEngine(F.EngineConfig(quote=F.QuoteModel(lam=0.25)),
                                pricer=F.BarPricer(obars, max_stale_minutes=stale))
        tr, _ = eng.run(frame)
        s = eng.stats
        fnl.append(dict(stale_min=stale, eligible=s["eligible"], signal_priced=s["eligible"] - s["no_quote"],
                        entries_priced=s["traded"] + s["no_credit"], trades=s["traded"],
                        exits_priced=s["traded"] - s["exit_missing"]))
        srows += summarize(tr, stale_min=stale)
    print(pd.DataFrame(fnl).to_string(index=False))
    print(pd.DataFrame(srows)[["stale_min", "kind", "n", "n_invalid", "win", "avg_usd", "total_usd",
                               "total_worst_usd"]].round(2).to_string(index=False))

# optional ablation ----------------------------------------------------------------
if args.ablation:
    feats_r = F.FeatureBuilder(F.FeatureConfig(session_reset_indicators=True)).transform(bars)
    dm_r = F.WalkForwardModel(config=F.ModelConfig(label=F.DIRECTION_LABEL, holdout_start=HOLDOUT_START))
    dm_r.fit_predict(feats_r, "p_")
    a = F.evaluate_forecasts(feats, dmod.labels, {"continuous": dmod.variants[dmod.cfg.calibration]}, dmod.step())
    b = F.evaluate_forecasts(feats_r, dm_r.labels, {"session_reset": dm_r.variants[dm_r.cfg.calibration]}, dm_r.step())
    print("\n6) ABLATION -- continuous indicators + gap vs session-reset indicators + gap (direction model)")
    print(pd.concat([a, b])[["n", "log_loss", "brier", "auc_magnitude", "auc_direction"]].round(4).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
