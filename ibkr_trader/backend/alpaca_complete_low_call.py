"""
Complete the LOW earnings condor's missing short-call leg (227.5C) -- the
long put/long call/short put already filled via alpaca_earnings_condor.py;
only this leg is missing. Uses the proven place_leg() from
alpaca_0dte_common.py, priced at/near the real current bid for a fast,
marketable fill given the tight entry window.
"""
from alpaca.trading.enums import OrderSide
from alpaca_0dte_common import load_config, alpaca_client, place_leg

cfg = load_config()
client = alpaca_client(cfg)
ok, fill_px = place_leg(client, "LOW260821C00227500", OrderSide.SELL, 1.00, "short call (completion)", qty=1)
print(f"ok={ok} fill_px={fill_px}")
