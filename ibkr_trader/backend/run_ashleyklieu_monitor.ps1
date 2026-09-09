# run_ashleyklieu_monitor.ps1 -- Watchdog: auto-restarts ashleyklieu_alert_monitor.py on any crash or clean exit.
# Mirrors run_scanner.ps1's pattern exactly. Continuous Discord channel
# monitor, monitor-and-alert only (no order placement) -- see the Python
# script's own docstring for the full rationale (CEO instruction 2026-08-26).

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "ashleyklieu_alert_monitor.py"
$WatchdogLog = Join-Path $BackendDir "ashleyklieu_monitor_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching ashleyklieu_alert_monitor.py..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -NoNewWindow `
            -PassThru `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Monitor exited cleanly (code=$code). Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch monitor: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
