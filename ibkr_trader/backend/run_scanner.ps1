# run_scanner.ps1 — Watchdog: auto-restarts breakout_scanner.py on any crash or clean exit.
# Launched by Windows Task Scheduler at logon; runs hidden with no console window.
# Scanner writes its own log via RotatingFileHandler — no redirection needed here.

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "breakout_scanner.py"
$WatchdogLog = Join-Path $BackendDir "scanner_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching breakout_scanner.py..."
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
        Write-WLog "Scanner exited cleanly (code=$code). Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch scanner: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
