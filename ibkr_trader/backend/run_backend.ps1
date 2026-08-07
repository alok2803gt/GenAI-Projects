# run_backend.ps1 -- Watchdog: validates syntax before starting, auto-restarts on crash.
# Launched by Windows Task Scheduler at logon; runs hidden with no console window.
# Backend writes its own log via uvicorn -- no redirection needed here.

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "main.py"
$WatchdogLog = Join-Path $BackendDir "backend_watchdog.log"
$LastGoodPy  = Join-Path $BackendDir "main_last_good.py"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Backend watchdog started (PID=$PID)"

while ($true) {
    # ── Pre-flight syntax check ───────────────────────────────────────────────
    # If main.py has a syntax error, the old process would crash on startup.
    # We catch it here and refuse to start, keeping the last-known-good copy alive.
    $syntaxResult = & $Python -m py_compile $Script 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-WLog "SYNTAX ERROR in main.py -- refusing to start. Error: $syntaxResult"
        Write-WLog "Fix main.py or restore from main_last_good.py, then this watchdog will auto-retry in 60s."
        Start-Sleep -Seconds 60
        continue
    }

    # ── Snapshot the validated copy ───────────────────────────────────────────
    # After a clean syntax check, save main.py as main_last_good.py.
    # If a future edit breaks syntax, you can restore: Copy-Item main_last_good.py main.py
    try {
        Copy-Item -Path $Script -Destination $LastGoodPy -Force
    } catch {
        Write-WLog "Warning: could not write main_last_good.py: $_"
    }

    Write-WLog "Syntax OK -- launching backend..."
    $stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog  = Join-Path $BackendDir "crash_logs\backend_stdout_$stamp.log"
    $ErrLog  = Join-Path $BackendDir "crash_logs\backend_stderr_$stamp.log"
    New-Item -ItemType Directory -Force -Path (Join-Path $BackendDir "crash_logs") | Out-Null
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput $OutLog `
            -RedirectStandardError $ErrLog `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Backend exited (code=$code). Logs: $ErrLog . Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch backend: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
