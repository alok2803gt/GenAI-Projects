# IBKR Trader — Daily Pre-Launch Checklist

Run these checks each morning before market open (target: by 9:15 ET).
All commands assume PowerShell from the repo root or `ibkr_trader/` directory.

---

## 1. Process & Port Inventory (30 sec)

```powershell
# Verify all 4 processes are running
Get-Process -Name python, node -ErrorAction SilentlyContinue |
  Select-Object Id, Name, CPU, @{N='MB';E={[int]($_.WorkingSet64/1MB)}} | Format-Table

# Verify key ports
@(8000, 8001, 7497, 47892) | ForEach-Object {
    $s = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue |
         Select-Object -First 1
    if ($s) { "Port $_ : $($s.State)  PID=$($s.OwningProcess)" }
    else    { "Port $_ : NOT FOUND" }
}
```

**Expected results:**
| Process | Port | Expected State |
|---------|------|----------------|
| Backend (main.py) | 8000 | Listen |
| Frontend (http-server) | 8001 | Listen |
| IBKR TWS | 7497 | Connected (from backend) |
| Scanner singleton | 47892 | Bound |

---

## 2. Backend API Health (20 sec)

```powershell
$s = Invoke-RestMethod "http://localhost:8000/status" -TimeoutSec 8
"IBKR connected : $($s.connected)"
"IBKR port      : $($s.port)"
```

**Pass criteria:** `connected = True`  
**If fail:** TWS may have disconnected overnight → restart TWS, then restart backend (Ctrl+C and re-run `uvicorn main:app`).

---

## 3. Market Regime (10 sec)

```powershell
$r = Invoke-RestMethod "http://localhost:8000/market/regime" -TimeoutSec 8
"regime_ok       : $($r.regime_ok)"
"SPY price       : $($r.spy_price)   SMA200: $($r.spy_sma200)"
"SPY above SMA200: $($r.spy_above_sma200)"
"VIX             : $($r.vix_live)"
"Reason          : $($r.reason)"
```

**Pass criteria:**  
- `regime_ok = True` → both stock traders (CSP/LEAP + Breakout) are cleared to trade  
- `spy_above_sma200 = True` AND `vix_live < 25`  

**If fail:** The regime gate will block all new entries automatically.
No action needed unless you want to override — the system handles it.

---

## 4. Stock Trader Status (10 sec)

```powershell
$st = Invoke-RestMethod "http://localhost:8000/stock-trader/status" -TimeoutSec 8
"Enabled         : $($st.enabled)"
"Open positions  : $($st.summary.open_positions)"
"Capital deployed: $($st.summary.capital_deployed)"
"Config:"
$st.config | Format-List
```

**Pass criteria:**
| Field | Expected |
|-------|----------|
| `enabled` | `True` |
| `position_size` | $3,000 |
| `max_positions` | 8 |
| `hard_stop_pct` | 7.0 |
| `trail_pct` | 5.0 |
| `max_hold_days` | 30 |
| `signal_freshness_min` | 30 |

**If positions > 0:** Review each open position for stale phase transitions.
Check `$st.positions` — any position with `phase = 0` and age > 2 days needs investigation (possible unfilled limit order).

---

## 5. Auto-Trader Status (15 sec)

```powershell
$at = Invoke-RestMethod "http://localhost:8000/autotrader/status" -TimeoutSec 30
"AT Enabled  : $($at.enabled)"
"AT Positions: $($at.positions.PSObject.Properties.Name -join ', ')"
```

**Pass criteria:** `enabled = True` (unless intentionally paused).  
**Note:** This call hits IBKR and may take 10-20s — normal.

---

## 6. Scanner Health (15 sec)

```powershell
# Heartbeat file age
$hb = Get-Content "ibkr_trader\backend\scanner_heartbeat.json" | ConvertFrom-Json
$age = (Get-Date) - [datetime]::Parse($hb.ts)
"Last scan   : $($hb.ts)"
"Tickers     : $($hb.tickers)"
"Age         : $([int]$age.TotalMinutes) min"

# Recent log lines
Get-Content "ibkr_trader\backend\scanner.log" -Tail 5
```

**Pass criteria (before 9:30 ET):**  
- Log shows `Market closed — Xm to open, sleeping 60s` → scanner is alive and waiting  
- Heartbeat `ts` from **yesterday** is normal (scanner writes heartbeat only during market hours)

**Pass criteria (after 9:30 ET):**  
- Heartbeat age ≤ 5 minutes  
- Log shows active scan cycles (`Scanning N tickers...`, `BREAKOUT:`, `PRE-BREAKOUT:`)

**If scanner is silent after 9:35 ET:** Check `scanner_watchdog.log`. Watchdog should auto-restart. If not:
```powershell
Start-Process powershell -ArgumentList "-File ibkr_trader\backend\run_scanner.ps1" -WindowStyle Normal
```

---

## 7. Watchlist State (10 sec)

```powershell
$wl = Invoke-RestMethod "http://localhost:8000/watchlist" -TimeoutSec 8
"Active entries : $($wl.count)"
if ($wl.entries) {
    $wl.entries | Sort-Object fired_at -Descending | Select-Object -First 5 |
        Format-Table ticker, signal_type, fired_at -AutoSize
}
```

**Pass criteria:**  
- Pre-market: entries from yesterday's EOD watchlist scan are visible (setups for today)  
- During market: entries refresh as scanner detects new signals

---

## 8. Frontend Accessibility (5 sec)

```powershell
(Invoke-WebRequest "http://localhost:8001" -TimeoutSec 5 -UseBasicParsing).StatusCode
```

**Pass:** `200`  
**If fail:** Restart frontend:
```powershell
npx http-server ibkr_trader/frontend -p 8001 -c-1
```

---

## 9. Singleton Verification (5 sec)

```powershell
Get-NetTCPConnection -LocalPort 47892 -ErrorAction SilentlyContinue |
    Select-Object LocalPort, State, OwningProcess
```

**Pass:** One row, state = `Bound`, OwningProcess = scanner PID  
**If two rows:** Duplicate scanner running — kill the older PID.  
**If no rows:** Scanner crashed AND watchdog hasn't restarted it yet — wait 60s and re-check.

---

## Full Checklist Summary (copy-paste version)

```powershell
# === IBKR TRADER DAILY PRE-LAUNCH CHECKS ===

Write-Host "`n[1] PROCESSES & PORTS"
@(8000, 8001, 7497, 47892) | ForEach-Object {
    $s = Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($s) { "  Port $_ : $($s.State)  PID=$($s.OwningProcess)" } else { "  Port $_ : MISSING" }
}

Write-Host "`n[2] BACKEND HEALTH"
$s = Invoke-RestMethod "http://localhost:8000/status" -TimeoutSec 8
"  connected=$($s.connected)  port=$($s.port)"

Write-Host "`n[3] MARKET REGIME"
$r = Invoke-RestMethod "http://localhost:8000/market/regime" -TimeoutSec 8
"  regime_ok=$($r.regime_ok)  SPY=$($r.spy_price)  SMA200=$($r.spy_sma200)  VIX=$($r.vix_live)"

Write-Host "`n[4] STOCK TRADER"
$st = Invoke-RestMethod "http://localhost:8000/stock-trader/status" -TimeoutSec 8
"  enabled=$($st.enabled)  positions=$($st.summary.open_positions)  deployed=`$$($st.summary.capital_deployed)"

Write-Host "`n[5] AUTO TRADER"
$at = Invoke-RestMethod "http://localhost:8000/autotrader/status" -TimeoutSec 30
"  enabled=$($at.enabled)  positions=$($at.positions.PSObject.Properties.Name.Count)"

Write-Host "`n[6] SCANNER"
$hb = Get-Content "ibkr_trader\backend\scanner_heartbeat.json" | ConvertFrom-Json
$age = [int]((Get-Date) - [datetime]::Parse($hb.ts)).TotalMinutes
"  last_scan=$($hb.ts)  tickers=$($hb.tickers)  age=${age}min"
Get-Content "ibkr_trader\backend\scanner.log" -Tail 2 | ForEach-Object { "  $_" }

Write-Host "`n[7] WATCHLIST"
$wl = Invoke-RestMethod "http://localhost:8000/watchlist" -TimeoutSec 8
"  entries=$($wl.count)"

Write-Host "`n[8] FRONTEND"
"  status=$((Invoke-WebRequest 'http://localhost:8001' -TimeoutSec 5 -UseBasicParsing).StatusCode)"

Write-Host "`n[9] SINGLETON"
Get-NetTCPConnection -LocalPort 47892 -ErrorAction SilentlyContinue |
    Select-Object @{N='check';E={"port 47892 -> State=$($_.State) PID=$($_.OwningProcess)"}}
```

---

## Pass/Fail Reference

| # | Check | PASS | FAIL Action |
|---|-------|------|-------------|
| 1 | Ports 8000, 8001, 7497, 47892 | All found | Restart missing process |
| 2 | Backend IBKR connected | `True` | Restart TWS + backend |
| 3 | Market regime | `regime_ok=True` | No action (auto-blocked) |
| 4 | Stock Trader enabled | `True` | Toggle ON in UI or `POST /stock-trader/enable?enabled=true` |
| 5 | Auto-Trader enabled | `True` | Toggle ON in UI |
| 6 | Scanner log active | Recent timestamps | Restart scanner |
| 7 | Watchlist populated | ≥ 1 entry | Run EOD watchlist scan or wait for morning scan |
| 8 | Frontend 200 | 200 | Restart http-server |
| 9 | Singleton bound | 1 row, Bound | Kill duplicate or restart scanner |

---

## Process Restart Commands

```powershell
# Backend
cd ibkr_trader\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Scanner (via watchdog)
powershell -File ibkr_trader\backend\run_scanner.ps1

# Frontend
npx http-server ibkr_trader\frontend -p 8001 -c-1
```

---

*Last updated: 2026-07-02*
