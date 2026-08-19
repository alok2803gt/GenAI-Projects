import yfinance as yf, warnings, datetime
warnings.filterwarnings("ignore")

t = yf.Ticker("DAL")
info = t.fast_info
spot = round(float(info.last_price), 2)
print(f"DAL spot: ${spot}")

exps = t.options
today = datetime.date.today()
tomorrows = [e for e in exps if datetime.datetime.strptime(e, "%Y-%m-%d").date() > today]
expiry = tomorrows[0]
print(f"Expiry: {expiry}")

chain_c = t.option_chain(expiry).calls
chain_p = t.option_chain(expiry).puts
strikes = sorted(chain_c["strike"].tolist())

def nearest(target):
    return min(strikes, key=lambda s: abs(s - target))

atm_strike = nearest(spot)
print(f"ATM strike: {atm_strike}")

def get_mid(df, strike):
    row = df[df["strike"] == strike]
    if row.empty: return 0.0
    b, a = float(row.iloc[0]["bid"]), float(row.iloc[0]["ask"])
    return round((b + a) / 2, 2)

call_mid = get_mid(chain_c, atm_strike)
put_mid  = get_mid(chain_p, atm_strike)
em = round(call_mid + put_mid, 2)
print(f"ATM call mid: {call_mid}  put mid: {put_mid}")
print(f"Expected Move: ${em} ({round(em/spot*100,1)}% of spot)")

wing_mult = 1.5
sp = nearest(spot - em)
sc = nearest(spot + em)
lp = nearest(spot - wing_mult * em)
lc = nearest(spot + wing_mult * em)

sp_mid = get_mid(chain_p, sp)
sc_mid = get_mid(chain_c, sc)
lp_mid = get_mid(chain_p, lp)
lc_mid = get_mid(chain_c, lc)
credit = round(sp_mid + sc_mid - lp_mid - lc_mid, 2)
width  = round(sp - lp, 2)
max_loss = round(width - credit, 2)

print()
print("=== DAL IRON CONDOR SETUP ===")
print(f"BUY  Put  {lp}  @ ${lp_mid}")
print(f"SELL Put  {sp}  @ ${sp_mid}")
print(f"SELL Call {sc}  @ ${sc_mid}")
print(f"BUY  Call {lc}  @ ${lc_mid}")
print()
print(f"Net Credit  : ${credit}/spread  (${round(credit*100):.0f}/contract)")
print(f"Spread Width: ${width}")
print(f"Max Loss    : ${max_loss}/spread  (${round(max_loss*100):.0f}/contract)")
print(f"Return/Risk : {round(credit/max_loss*100,1)}%")
print()
print(f"Breakevens  : {round(sp-credit,2)} / {round(sc+credit,2)}")
print(f"EM range    : {round(spot-em,2)} - {round(spot+em,2)}")
print(f"Short strikes vs EM: {round((sp-spot)/em*100,0):.0f}% / +{round((sc-spot)/em*100,0):.0f}%")
