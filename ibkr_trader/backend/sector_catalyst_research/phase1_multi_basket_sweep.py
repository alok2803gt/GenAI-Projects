"""
Systematic sweep: does the "2+ basket members pop together -> laggard
catches up" signal (validated on Memory/Storage, phase1_signal_test.py)
generalize to OTHER genuinely narrow, commonly-driven baskets across
different sectors -- or was Memory/Storage a one-off fluke? Same exact
signal definition and methodology, applied uniformly across 14 candidate
baskets spanning oil, airlines, homebuilders, banks, solar, uranium,
cruise lines, casinos, steel, coal, Chinese ADRs, and lithium -- chosen
because each has a plausible SHARED real-world driver (commodity price,
rate policy, a regulatory/macro theme) that could produce the same
lagged-reaction-across-peers pattern, the same structural property that
made Memory/Storage work and generic "broad semis" fail.
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

POP_RET_THRESH = 0.03
POP_VOL_MULT = 1.0
MIN_MOVERS = 2
TAIL_MOVE_THRESH = 0.04


def per_ticker_frame(df):
    df = df.copy()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df["ret"] = df["Close"].pct_change()
    df["vol_avg20"] = df["Volume"].rolling(20, min_periods=15).mean()
    df["rel_vol"] = df["Volume"] / df["vol_avg20"].shift(1)
    df["popped"] = (df["ret"] >= POP_RET_THRESH) & (df["rel_vol"] >= POP_VOL_MULT)
    df["fwd_ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1
    df["fwd_ret_2d"] = df["Close"].shift(-2) / df["Close"] - 1
    return df


def eval_basket(tickers, universe):
    frames = {t: per_ticker_frame(universe[t]) for t in tickers if t in universe}
    if len(frames) < 2:
        return None
    pop_counts = pd.DataFrame({t: f["popped"] for t, f in frames.items()}).sum(axis=1)
    hot_days = pop_counts[pop_counts >= MIN_MOVERS].index

    spill_tail, spill_base, cont_tail, cont_base, n_spill, n_cont = [], [], [], [], 0, 0
    all_cond, all_uncond = [], []
    for t, f in frames.items():
        hot = hot_days.intersection(f.index)
        if len(hot) == 0:
            continue
        was_popper = f.loc[hot, "popped"]
        spill_days = hot[~was_popper.values]
        cont_days = hot[was_popper.values]
        uncond = f.dropna(subset=["fwd_ret_1d"])
        base_tail = (uncond["fwd_ret_1d"] >= TAIL_MOVE_THRESH).mean()
        base_mean = uncond["fwd_ret_1d"].mean()

        sp = f.loc[spill_days].dropna(subset=["fwd_ret_1d"])
        if len(sp) >= 5:
            spill_tail.append((sp["fwd_ret_1d"] >= TAIL_MOVE_THRESH).mean())
            spill_base.append(base_tail)
            n_spill += len(sp)
        co = f.loc[cont_days].dropna(subset=["fwd_ret_1d"])
        if len(co) >= 5:
            cont_tail.append((co["fwd_ret_1d"] >= TAIL_MOVE_THRESH).mean())
            cont_base.append(base_tail)
            n_cont += len(co)

        hotf = f.loc[hot].dropna(subset=["fwd_ret_1d"])
        if len(hotf) >= 5:
            all_cond.append(hotf["fwd_ret_1d"].mean())
            all_uncond.append(base_mean)

    if not spill_tail and not all_cond:
        return None

    return {
        "n_tickers": len(frames),
        "n_hot_days": len(hot_days),
        "n_spill_obs": n_spill,
        "n_cont_obs": n_cont,
        "spill_tail_lift_pp": 100 * (sum(spill_tail) / len(spill_tail) - sum(spill_base) / len(spill_base)) if spill_tail else None,
        "cont_tail_lift_pp": 100 * (sum(cont_tail) / len(cont_tail) - sum(cont_base) / len(cont_base)) if cont_tail else None,
        "mean_fwd1_lift_pp": 100 * (sum(all_cond) / len(all_cond) - sum(all_uncond) / len(all_uncond)) if all_cond else None,
    }


def main():
    with open(HERE / "multi_sector_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)
    with open(HERE / "baskets.pkl", "rb") as f:
        baskets = pickle.load(f)

    rows = []
    for name, tickers in baskets.items():
        r = eval_basket(tickers, universe)
        if r is None:
            print(f"{name}: not enough hot-day observations, skipped")
            continue
        r["basket"] = name
        rows.append(r)

    df = pd.DataFrame(rows).set_index("basket")
    df = df.sort_values("spill_tail_lift_pp", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("\n=== RANKED BY SPILLOVER TAIL-RATE LIFT (laggard-catch-up edge, the tradeable insight) ===\n")
    print(df[["n_tickers", "n_hot_days", "n_spill_obs", "spill_tail_lift_pp",
              "n_cont_obs", "cont_tail_lift_pp", "mean_fwd1_lift_pp"]].to_string())
    df.to_csv(HERE / "phase1_multi_basket_results.csv")
    print(f"\nSaved -> phase1_multi_basket_results.csv")


if __name__ == "__main__":
    main()
