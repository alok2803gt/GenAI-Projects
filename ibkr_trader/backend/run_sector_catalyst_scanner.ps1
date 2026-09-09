# run_sector_catalyst_scanner.ps1 — Watchdog: auto-restarts sector_catalyst_scanner.py
# on any crash or clean exit. Launched by Windows Task Scheduler at logon; runs hidden.
# Unlike breakout_scanner.py, this script has no internal file logging (just print()),
# so stdout/stderr ARE redirected here to a real log file -- the 2026-09-08 root-cause
# lesson (butterfly close scripts had zero log capture, requiring after-the-fact
# reconciliation from raw broker fills every time something went wrong) applies to any
# new unattended script, not just that one.

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "sector_catalyst_scanner.py"
$WatchdogLog = Join-Path $BackendDir "sector_catalyst_scanner_watchdog.log"
$StdOutLog   = Join-Path $BackendDir "sector_catalyst_scanner_stdout.log"
$StdErrLog   = Join-Path $BackendDir "sector_catalyst_scanner_stderr.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching sector_catalyst_scanner.py..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -NoNewWindow `
            -RedirectStandardOutput $StdOutLog `
            -RedirectStandardError $StdErrLog `
            -PassThru `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Scanner exited (code=$code). Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch scanner: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
