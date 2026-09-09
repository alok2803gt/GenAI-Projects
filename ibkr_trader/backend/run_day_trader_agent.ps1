# run_day_trader_agent.ps1 -- Watchdog: auto-restarts day_trader_agent.py
# on any crash or clean exit. Mirrors run_ashleyklieu_trigger_executor.ps1's
# pattern exactly. Day Trader was extracted from main.py into this standalone
# process 2026-08-27 (CEO instruction, after main.py's own restart instability
# dropped a real entry signal -- see day_trader_agent.py's own docstring).

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "day_trader_agent.py"
$WatchdogLog = Join-Path $BackendDir "day_trader_agent_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching day_trader_agent.py..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -RedirectStandardOutput (Join-Path $BackendDir "day_trader_agent.log") `
            -RedirectStandardError (Join-Path $BackendDir "day_trader_agent.err.log") `
            -NoNewWindow `
            -PassThru `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Agent exited (code=$code). Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch agent: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
