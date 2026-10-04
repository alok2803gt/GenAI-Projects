# 0DTE Iron Condor V2 — IBKR trader

Self-contained. Needs only Python 3.10+ and a TWS or IB Gateway login. Nothing else from the original project is required.

## What it trades

| | |
|---|---|
| **Entries** | Every 30 minutes, 10:01 to 14:31 ET — 10 condors per session, 1 contract each |
| **Strikes** | Short legs nearest spot ± 0.5u, wings $5 further out, where `u = spot × ATM implied vol × √(minutes to 16:00 ÷ 98,280)` |
| **Stop** | Closes the whole condor the first minute its loss reaches **0.5 × the credit received** (fires on about 42% of condors) |
| **Exit** | Otherwise **held to expiry**. No profit target — none was tested |
| **15:58** | Closes only the side whose short strike is within $0.25 of spot, to limit after-hours assignment |

**Backtest per 1-contract unit:** +$101/day, worst day −$1,255, max drawdown −$5,224, peak margin about $4,100, capital about $15,000. Validated on SPY over 660 sessions; QQQ passed a pre-registered confirmation (t 4.94).

**Important:** the edge comes from holding to the 16:00 expiry. A broker that force-closes 0DTE early destroys it — closing at 15:45 backtests at −$0.38/day against +$8.37/day. Alpaca does exactly that, which is why this is IBKR-only.

## Setup on a new machine

**1. Install**
```bash
./install.sh          # macOS / Linux
install.bat           # Windows
```
Creates `.venv` beside the script and installs `ib_insync` and `requests`.

**2. Configure TWS or IB Gateway** (paper login)
- Enable the API: *Global Configuration → API → Settings* → tick **Enable ActiveX and Socket Clients**.
- Confirm the port matches `config.json`: TWS paper **7497**, TWS live 7496, Gateway paper 4002, Gateway live 4001.
- Add `127.0.0.1` to trusted IPs, or untick *Read-Only API*.
- **Market data:** the account needs a real-time US options subscription (OPRA). Without it, quotes return empty and the trader will skip every entry.

**3. Edit `config.json`**
```json
{"ticker": "SPY", "qty": 1, "tws_port": 7497, "client_id": 1580, "daily_stop_usd": -1500.0}
```
`client_id` must be unique per script connected to the same TWS. Telegram alerts are optional: fill in `telegram_token` and `telegram_chat_id`, or leave them blank.

**4. Test before trading**
```bash
./.venv/bin/python test_condor_v2.py     # 9 offline tests, no IBKR needed
./run.sh --dry-run                       # prices real condors, places NOTHING
```
The dry run only shows entries during market hours, between 10:01 and 14:31 ET.

**5. Run it**
```bash
./run.sh                                 # paper, per config.json
./run.sh --ticker QQQ --qty 1
```
Start it before 10:00 ET. It runs until 16:00 and exits on its own.

## Scheduling

**macOS (launchd):** edit the paths in `com.ibkrtrader.condorv2.plist`, then
```bash
cp com.ibkrtrader.condorv2.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ibkrtrader.condorv2.plist
```

**Linux (cron):** `55 9 * * 1-5 /path/to/condor_v2/run.sh >> /path/to/condor_v2/cron.log 2>&1`

**Windows (Task Scheduler):** daily trigger at 09:55, action `run.bat`, "Start in" set to the folder.

Note that all times are **US Eastern**; the code converts internally, but the schedule must fire at 09:55 ET in the machine's own local time.

## Safety features

- **Paper port (7497) is the default.** Live requires an explicit change, and the log records `LIVE` or `paper` at startup.
- **`PAUSED.flag`** is written on any partial fill or unexpected error, and every later run refuses to start until you delete it. Check the position by hand before clearing.
- **Daily stop** (`daily_stop_usd`, default −$1,500) halts new entries for the day. It's a safety rail, not a backtested rule.
- **Thin-credit guard** skips an entry when the conservative credit is under $0.10.
- **Restart-safe:** `state.json` records each entry's scheduled slot, so restarting mid-day doesn't duplicate entries.
- **Order placement** is sequential and long-legs-first, so no leg is ever left uncovered.

## Files

| File | Purpose |
|---|---|
| `condor_v2.py` | The trader (single file) |
| `config.json` | Settings |
| `test_condor_v2.py` | 9 offline tests |
| `state.json` | Today's positions (created at runtime) |
| `condor_v2_log.jsonl` | One JSON line per event |
| `PAUSED.flag` | Present = refuse to run |
| `build_exe.sh` | Optional single-file binary, built **on the target machine** |

## A standalone binary

PyInstaller can't cross-compile, so build on the machine that will run it:
```bash
./install.sh && ./build_exe.sh      # produces dist/condor_v2
```
Keep `config.json` beside the binary.

## Known limits

- **Fills:** the backtest assumed one 4-leg fill; this legs in one order at a time, which adds slippage. Paper trading measures it.
- **Assignment:** SPY and QQQ options are physically settled. The 15:58 rule reduces the risk but doesn't remove it. Cash-settled SPX avoids it entirely, but measured weaker (t 1.56) and needs its own contract handling (`SPXW` trading class).
- **Realistic edge:** about $4–6 per condor once after-hours exercise is priced in, not the $10 the first backtest showed.
