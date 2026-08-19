"""
Compute real historical net GEX for SPY from CBOE DataShop's real OI +
CBOE-computed gamma (spy_cboe_chains.json), using the exact same dollar-
gamma convention already established in this account's gex-vex-calculator
skill (net_gex = sum(call gamma*OI) - sum(put gamma*OI), scaled to
dollars-per-1%-move: gamma * OI * 100 * spot^2 * 0.01), for consistency.

Unlike gex-vex-calculator's own single 25-45 DTE near-term-expiry
simplification (a documented tractability tradeoff for a live IBKR scan),
this uses the REAL FULL OPTION CHAIN (every expiry, every strike) since
CBOE's EOD data already gives us that for free -- a genuine improvement,
not just a rebuild.

GEX is inherently an end-of-day figure (OI is only reported once daily),
so this computes each day's GEX from its own EOD close, to be used as
NEXT trading day's regime signal -- mirrors exactly how the earlier
VIX-proxy regime filter worked (spy_0dte_regime_backtest.py: yesterday's
close VIX informs today's filter).

Usage: python compute_real_gex_spy.py
Writes spy_real_gex_history.json: {date: {spot, net_gex, regime}}
"""
import json


def compute_day_gex(day_data):
    spot = day_data["underlying_close"]
    if not spot:
        return None
    net_gex = 0.0
    for c in day_data["contracts"]:
        gamma = c.get("gamma")
        oi = c.get("open_interest")
        if gamma is None or oi is None or oi <= 0:
            continue
        dollar_gamma = gamma * oi * 100 * (spot ** 2) * 0.01
        net_gex += dollar_gamma if c["option_type"] == "C" else -dollar_gamma
    return round(net_gex, 0)


def main():
    with open("spy_cboe_chains.json") as f:
        chains = json.load(f)

    print(f"Computing real net GEX for {len(chains)} real days...\n")
    results = {}
    for date, day_data in sorted(chains.items()):
        net_gex = compute_day_gex(day_data)
        if net_gex is None:
            continue
        regime = "positive_gamma" if net_gex > 0 else "negative_gamma"
        results[date] = {
            "spot": day_data["underlying_close"],
            "net_gex": net_gex,
            "regime": regime,
            "n_contracts": len(day_data["contracts"]),
        }
        print(f"  {date}: spot=${day_data['underlying_close']:.2f}  "
              f"net_gex={net_gex:+,.0f}  regime={regime}  "
              f"(n={len(day_data['contracts'])})")

    with open("spy_real_gex_history.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} days to spy_real_gex_history.json")

    pos = sum(1 for r in results.values() if r["regime"] == "positive_gamma")
    print(f"\n{pos}/{len(results)} days were positive_gamma, {len(results)-pos}/{len(results)} negative_gamma")


if __name__ == "__main__":
    main()
