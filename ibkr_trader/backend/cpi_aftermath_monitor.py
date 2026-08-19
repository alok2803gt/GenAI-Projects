"""
CPI-aftermath monitor: checks whether regional banks / REITs got the outsized
rally that historically follows a cool CPI surprise, and if so, surfaces real
CSP put-credit-spread candidates on the names that moved -- selling into
fresh strength (wider cushion) with IV that's typically still elevated from
pre-print positioning, rather than picking up event risk by selling before
the print (see 2026-08-12 CRO CPI scenario analysis, oversight_log.jsonl).

Universe: 7 regional banks + 9 REITs, all already in CANDIDATE_POOL.
(CMA/Comerica dropped 2026-08-12 -- delisted on yfinance, likely acquired.)

Usage:
  python cpi_aftermath_monitor.py --baseline     # snapshot current prices as baseline
  python cpi_aftermath_monitor.py --check        # compare current vs baseline, report movers + CSP candidates
"""
import argparse
import json
from datetime import datetime, timezone

import yfinance as yf

TICKERS = [
    # Regional banks
    "FITB", "RF", "KEY", "HBAN", "ZION", "WAL", "EWBC",
    # REITs
    "AMT", "PLD", "EQIX", "SPG", "O", "CCI", "WELL", "AVB", "VICI",
]
BASELINE_FILE   = "cpi_aftermath_baseline.json"
RALLY_THRESHOLD = 1.5   # % move to flag as a candidate worth a closer look
WING_MULT       = 1.5   # protective-leg width, same as EVC's proven methodology
SHORT_OTM_PCT   = 0.12  # short put ~12% OTM


def snapshot_baseline():
    data = yf.download(TICKERS, period="1d", interval="1m", progress=False, group_by="ticker")
    prices = {}
    for t in TICKERS:
        try:
            prices[t] = round(float(data[t]["Close"].dropna().iloc[-1]), 2)
        except Exception as e:
            print(f"{t}: could not snapshot ({e})")
    out = {"snapshot_time": datetime.now(timezone.utc).isoformat(), "prices": prices}
    with open(BASELINE_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline saved: {len(prices)} tickers.")
    print(json.dumps(prices, indent=2))


def csp_candidate(ticker: str, price: float) -> dict | None:
    """Real put-credit-spread candidate: nearest 25-45 DTE expiry, ~12% OTM short,
    protective long at wing_mult x the short's OTM distance. Same shape as
    everything else built this session (AutoTrader's _at_place_csp_spread,
    tonight's EVC condor legs)."""
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        today = datetime.now().date()
        target_exp = None
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if 25 <= dte <= 45:
                target_exp = e
                break
        target_exp = target_exp or exps[min(2, len(exps) - 1)]
        puts = tk.option_chain(target_exp).puts
        short_target = price * (1 - SHORT_OTM_PCT)
        long_target  = price * (1 - SHORT_OTM_PCT * WING_MULT)
        puts["diff_short"] = (puts["strike"] - short_target).abs()
        puts["diff_long"]  = (puts["strike"] - long_target).abs()
        short_row = puts.sort_values("diff_short").iloc[0]
        long_row  = puts.sort_values("diff_long").iloc[0]
        if long_row["strike"] >= short_row["strike"]:
            return None
        s_mid = (short_row["bid"] + short_row["ask"]) / 2 if short_row["ask"] > 0 else short_row.get("lastPrice", 0)
        l_mid = (long_row["bid"] + long_row["ask"]) / 2 if long_row["ask"] > 0 else long_row.get("lastPrice", 0)
        credit = round(s_mid - l_mid, 2)
        width  = round(short_row["strike"] - long_row["strike"], 2)
        return {
            "expiry": target_exp,
            "short_strike": float(short_row["strike"]),
            "long_strike": float(long_row["strike"]),
            "width": width,
            "est_credit": credit,
            "short_iv": round(float(short_row.get("impliedVolatility", 0)) * 100, 1),
            "short_oi": int(short_row.get("openInterest", 0) or 0),
        }
    except Exception as e:
        return {"error": str(e)}


def check():
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    base_prices = baseline["prices"]
    print(f"Baseline snapshot: {baseline['snapshot_time']}")

    data = yf.download(TICKERS, period="1d", interval="1m", progress=False, group_by="ticker")
    moves = []
    for t in TICKERS:
        if t not in base_prices:
            continue
        try:
            cur = float(data[t]["Close"].dropna().iloc[-1])
            base = base_prices[t]
            pct = (cur - base) / base * 100
            moves.append({"ticker": t, "base": base, "current": round(cur, 2), "pct": round(pct, 2)})
        except Exception:
            continue

    moves.sort(key=lambda m: m["pct"], reverse=True)
    print("\n=== Move ranking (vs pre-print baseline) ===")
    for m in moves:
        flag = " <-- RALLY CANDIDATE" if m["pct"] >= RALLY_THRESHOLD else ""
        print(f"  {m['ticker']:6s} {m['base']:>8.2f} -> {m['current']:>8.2f}  {m['pct']:+6.2f}%{flag}")

    candidates = [m for m in moves if m["pct"] >= RALLY_THRESHOLD]
    if not candidates:
        print(f"\nNo names moved >= {RALLY_THRESHOLD}% -- no outsized rally detected. "
              f"Print was likely inline/hot, or the move hasn't developed yet.")
        return

    print(f"\n=== CSP candidates on the {len(candidates)} rally name(s) ===")
    results = []
    for m in candidates:
        cand = csp_candidate(m["ticker"], m["current"])
        results.append({**m, "csp": cand})
        if cand and "error" not in cand:
            print(f"\n{m['ticker']} (+{m['pct']:.2f}%, now ${m['current']:.2f}):")
            print(f"  SELL ${cand['short_strike']}P / BUY ${cand['long_strike']}P  {cand['expiry']}")
            print(f"  width=${cand['width']}  est. credit=${cand['est_credit']}  "
                  f"short IV={cand['short_iv']}%  short OI={cand['short_oi']}")
        else:
            print(f"\n{m['ticker']}: could not build a candidate ({cand.get('error') if cand else 'no data'})")

    with open("cpi_aftermath_candidates.json", "w") as f:
        json.dump({"checked_at": datetime.now(timezone.utc).isoformat(), "candidates": results}, f, indent=2)
    print("\nSaved to cpi_aftermath_candidates.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.baseline:
        snapshot_baseline()
    elif args.check:
        check()
    else:
        print("Pass --baseline or --check")
