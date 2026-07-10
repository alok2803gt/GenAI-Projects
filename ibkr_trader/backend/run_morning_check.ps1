# run_morning_check.ps1 -- 9:20 AM ET pre-market startup health check
# Scheduled via Task Scheduler (Mon-Fri). Sends Telegram summary.
# Auto-fixes: kills duplicate watchdog processes.

param([string]$BackendUrl = "http://localhost:8000")

$SCRIPT_DIR  = $PSScriptRoot
$CONFIG_PATH = Join-Path $SCRIPT_DIR "scanner_config.json"
$ET_TIME     = (Get-Date).ToString("HH:mm ET")

# Load Telegram credentials from scanner_config.json
try {
    $cfg        = Get-Content $CONFIG_PATH -Raw | ConvertFrom-Json
    $TG_TOKEN   = $cfg.telegram_token
    $TG_CHAT_ID = $cfg.telegram_chat_id
} catch {
    Write-Host "FATAL: Cannot read scanner_config.json -- $($_.Exception.Message)"
    exit 1
}

function Send-Telegram([string]$text) {
    try {
        $uri  = "https://api.telegram.org/bot$TG_TOKEN/sendMessage"
        $body = @{ chat_id = "$TG_CHAT_ID"; text = $text; parse_mode = "HTML" }
        Invoke-RestMethod -Uri $uri -Method Post -Body $body -TimeoutSec 10 | Out-Null
    } catch {
        Write-Host "Telegram send failed: $($_.Exception.Message)"
    }
}

$checks = New-Object System.Collections.ArrayList
$issues = New-Object System.Collections.ArrayList

# 1. Backend health
try {
    $resp = Invoke-RestMethod -Uri "$BackendUrl/health" -Method Get -TimeoutSec 5
    if ($resp.status -eq "ok") {
        [void]$checks.Add("[OK] Backend: running")
    } else {
        [void]$checks.Add("[FAIL] Backend: unhealthy ($($resp.status))")
        [void]$issues.Add("Backend returned non-ok status -- restart backend before 9:30")
    }
} catch {
    [void]$checks.Add("[FAIL] Backend: NOT reachable")
    [void]$issues.Add("Backend not reachable at $BackendUrl -- start it before 9:30")
}

# 2. IBKR connection (critical -- disconnected causes silent day-trader failures)
try {
    $status = Invoke-RestMethod -Uri "$BackendUrl/status" -Method Get -TimeoutSec 5
    if ($status.connected -eq $true) {
        [void]$checks.Add("[OK] IBKR: connected")
    } else {
        [void]$checks.Add("[FAIL] IBKR: NOT connected")
        [void]$issues.Add("IBKR disconnected -- day-trader signals will 503 at market open")
    }
} catch {
    [void]$checks.Add("[WARN] IBKR: status check failed (backend may still be starting)")
}

# 3. Scanner running (port 47892 bound = exactly one instance active)
$scannerPort = Get-NetTCPConnection -LocalPort 47892 -ErrorAction SilentlyContinue | Where-Object { $_.State -in "Listen","Bound" }
if ($scannerPort) {
    $scannerPid = $scannerPort.OwningProcess
    [void]$checks.Add("[OK] Scanner: running (PID $scannerPid, port 47892)")
} else {
    $scannerPid = $null
    [void]$checks.Add("[FAIL] Scanner: NOT running")
    [void]$issues.Add("Breakout scanner is not running -- no BREAKOUT signals will fire at open")
}

# 4. Watchdog duplicate check (duplicate watchdogs spawn scanner chaos at open)
$allWds = @(Get-WmiObject Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "powershell*" -and $_.CommandLine -match "run_scanner" })
$wdCount = $allWds.Count

if ($wdCount -eq 0) {
    [void]$checks.Add("[WARN] Watchdog: not running (scanner has no restart guard)")
    [void]$issues.Add("Scanner watchdog not running -- scanner will not restart if it crashes")
} elseif ($wdCount -eq 1) {
    [void]$checks.Add("[OK] Watchdog: 1 instance")
} else {
    # Kill duplicates -- keep the watchdog that owns the scanner process
    $ownerWd    = $null
    $killedPids = @()
    if ($scannerPid) {
        foreach ($wd in $allWds) {
            $kids = @(Get-WmiObject Win32_Process -Filter "ParentProcessId = $($wd.ProcessId)" -ErrorAction SilentlyContinue)
            if ($kids | Where-Object { $_.ProcessId -eq [int]$scannerPid }) {
                $ownerWd = $wd.ProcessId
                break
            }
        }
    }
    foreach ($wd in $allWds) {
        if ($wd.ProcessId -ne $ownerWd) {
            Stop-Process -Id $wd.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue
            $killedPids += $wd.ProcessId
        }
    }
    [void]$checks.Add("[WARN] Watchdog: $wdCount found -- killed duplicates ($($killedPids -join ', ')), kept $ownerWd")
    [void]$issues.Add("Killed $($killedPids.Count) duplicate watchdog(s) -- would have caused scanner chaos at open")
}

# 5. Day trader enabled
try {
    $dt = Invoke-RestMethod -Uri "$BackendUrl/day-trader/status" -Method Get -TimeoutSec 5
    if ($dt.enabled -eq $true) {
        $posCount = ($dt.positions.PSObject.Properties | Measure-Object).Count
        [void]$checks.Add("[OK] Day trader: enabled ($posCount open positions)")
    } else {
        [void]$checks.Add("[WARN] Day trader: disabled")
    }
} catch {
    [void]$checks.Add("[WARN] Day trader: status check failed")
}

# 6. Stock trader enabled + tickers
try {
    $st = Invoke-RestMethod -Uri "$BackendUrl/stock-trader/status" -Method Get -TimeoutSec 5
    if ($st.enabled -eq $true) {
        $stTickers = ($st.positions.PSObject.Properties.Name) -join ", "
        $stPos     = ($st.positions.PSObject.Properties | Measure-Object).Count
        $stMax     = $st.config.max_positions
        [void]$checks.Add("[OK] Stock trader: enabled ($stPos/$stMax positions: $stTickers)")
    } else {
        [void]$checks.Add("[WARN] Stock trader: disabled")
    }
} catch {
    [void]$checks.Add("[WARN] Stock trader: status check failed")
}

# 7. SPX 0DTE
try {
    $spx = Invoke-RestMethod -Uri "$BackendUrl/spx-0dte/status" -Method Get -TimeoutSec 5
    $spxPnl     = $spx.summary.today_pnl
    $spxNext    = $spx.summary.next_target
    $spxFloor   = $spx.config.daily_floor_pnl
    $spxTiers   = ($spx.config.stop_loss_tiers -join "/") + "x"
    $spxAttempt = $spx.attempts_today
    if ($spx.enabled -eq $true) {
        [void]$checks.Add("[OK] SPX 0DTE: enabled | attempts=$spxAttempt | day_pnl=`$$spxPnl | next_target=`$$spxNext | stops=$spxTiers | floor=`$$spxFloor")
    } else {
        [void]$checks.Add("[WARN] SPX 0DTE: disabled")
        [void]$issues.Add("SPX 0DTE is disabled -- enable before market open if trading today")
    }
} catch {
    [void]$checks.Add("[WARN] SPX 0DTE: status check failed")
}

# 8. EVC (Earnings Vol Crush)
try {
    $evc = Invoke-RestMethod -Uri "$BackendUrl/earnings-vol-crush/status" -Method Get -TimeoutSec 5
    $evcPos    = ($evc.positions.PSObject.Properties | Measure-Object).Count
    $evcScan   = if ($evc.scan_date) { $evc.scan_date } else { "fresh (scans at $($evc.config.entry_start))" }
    $evcWindow = "$($evc.config.entry_start)-$($evc.config.entry_cutoff)"
    if ($evc.enabled -eq $true) {
        [void]$checks.Add("[OK] EVC: enabled | positions=$evcPos | scan=$evcScan | window=$evcWindow ET")
    } else {
        [void]$checks.Add("[WARN] EVC: disabled")
    }
} catch {
    [void]$checks.Add("[WARN] EVC: status check failed")
}

# Build and send Telegram summary
$header = if ($issues.Count -eq 0) {
    "&#x2705; <b>Morning check PASSED</b> -- $ET_TIME"
} else {
    "&#x26A0;&#xFE0F; <b>Morning check: $($issues.Count) issue(s)</b> -- $ET_TIME"
}

$issueBlock = if ($issues.Count -gt 0) {
    "`n`n" + (($issues | ForEach-Object { "&#x1F6A8; $_" }) -join "`n")
} else { "" }

$message = "$header`n`n" + ($checks -join "`n") + $issueBlock
Send-Telegram -text $message
Write-Host $message
