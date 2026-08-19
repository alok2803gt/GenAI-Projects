# run_frontend.ps1 — Watchdog: serves index.html on port 8001, auto-restarts on crash.
# Launched by Windows Task Scheduler at logon; runs hidden with no console window.

$FrontendDir = "C:\Projects\GenAI-Projects\ibkr_trader\frontend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Port        = 8001
$WatchdogLog = Join-Path $FrontendDir "frontend_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Frontend watchdog started (PID=$PID)"

while ($true) {
    # Kill any stale process already using port 8001
    $stale = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($stale) {
        $stalePid = $stale[0].OwningProcess
        Write-WLog "Port $Port already in use by PID=$stalePid — killing stale process"
        Stop-Process -Id $stalePid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    Write-WLog "Launching http.server on port $Port..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-m", "http.server", "$Port", "--directory", "`"$FrontendDir`"" `
            -WorkingDirectory $FrontendDir `
            -NoNewWindow `
            -PassThru `
            -ErrorAction Stop
        Write-WLog "Server started (PID=$($proc.Id))"
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Server exited (code=$code). Restarting in 5s..."
    } catch {
        Write-WLog "Failed to launch server: $_. Retrying in 5s..."
    }
    Start-Sleep -Seconds 5
}
