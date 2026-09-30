"""
ONE-TIME final evaluation on the locked holdout (the last 12 months).

Run only after the strategy, features, models and thresholds are frozen. Every
run is appended to holdout_log.jsonl with a hash of the framework source, so
repeated peeking is visible after the fact. If the result is used to change
anything, the holdout is spent: the next honest test is new (forward) data.

    ./.venv/bin/python run_holdout.py --i-have-frozen-the-model --option-bars data/SPY_0dte_option_bars
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import spy0dte_framework as F

HOLDOUT_START = F.HOLDOUT_START  # single source of truth for the split

ap = argparse.ArgumentParser()
ap.add_argument("--i-have-frozen-the-model", action="store_true", dest="frozen")
ap.add_argument("--bars", default="data/SPY_1min_sip.parquet")
ap.add_argument("--option-bars", default=None)
ap.add_argument("--note", default="")
args = ap.parse_args()
if not args.frozen:
    raise SystemExit("Refusing: pass --i-have-frozen-the-model. Every run is logged to holdout_log.jsonl.")

logging.basicConfig(level=logging.WARNING)
src_hash = hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()[:16]
log_path = Path("holdout_log.jsonl")
prior = sum(1 for _ in log_path.open()) if log_path.exists() else 0
print(f"framework source hash {src_hash}; previous holdout runs logged: {prior}")

hold = pd.Timestamp(HOLDOUT_START, tz=F.NY)
bars = F.prepare_bars(pd.read_parquet(args.bars)[["open", "high", "low", "close", "volume"]])
feats = F.FeatureBuilder().transform(bars)
regimes = F.RegimeClassifier().classify(feats)
dmod = F.WalkForwardModel(config=F.ModelConfig(label=F.DIRECTION_LABEL, holdout_start=HOLDOUT_START, allow_holdout=True))
smod = F.WalkForwardModel(config=F.ModelConfig(label=F.SPREAD_LABEL, holdout_start=HOLDOUT_START, allow_holdout=True))
frame = pd.concat([feats, regimes, dmod.fit_predict(feats, "p_"), smod.fit_predict(feats, "s_")], axis=1)
in_hold = frame["session"] >= hold
frame = frame[in_hold]

result = {}
for name, m in (("direction", dmod), ("spread", smod)):
    card = F.evaluate_forecasts(feats[in_hold], m.labels[in_hold], {k: v[in_hold] for k, v in m.variants.items()}, m.step())
    print(f"\n{name} model on the holdout:\n{card.round(4).to_string()}")
    result[name] = card.round(5).to_dict()

pricer = None
if args.option_bars:
    parts = [p for p in sorted(Path(args.option_bars).glob("*.parquet")) if pd.Timestamp(p.stem, tz=F.NY) >= hold]
    ob = pd.concat([pd.read_parquet(p) for p in parts]) if parts else pd.DataFrame()
    pricer = F.BarPricer({s: g.drop(columns="symbol") for s, g in ob.groupby("symbol")}) if len(ob) else None
for lam in (0.0, 0.25, 0.5):
    tr, _ = F.EntryExitEngine(F.EngineConfig(quote=F.QuoteModel(lam=lam)), pricer=pricer).run(frame)
    summary = tr.groupby("kind")["pnl_usd"].agg(["size", "mean", "sum"]).round(2) if len(tr) else pd.DataFrame()
    print(f"\nengine lambda {lam} ({'real option bars' if pricer else 'Black-Scholes'}):\n{summary.to_string()}")
    result[f"engine_lambda_{lam}"] = summary.to_dict()

with log_path.open("a") as f:
    f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), source_hash=src_hash,
                            holdout_start=HOLDOUT_START, pricer=pricer.name if pricer else "black_scholes",
                            note=args.note, result=result), default=str) + "\n")
print(f"\nlogged to {log_path} (run #{prior + 1})")
