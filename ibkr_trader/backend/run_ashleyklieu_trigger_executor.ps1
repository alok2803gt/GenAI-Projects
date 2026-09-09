# run_ashleyklieu_trigger_executor.ps1 -- Watchdog: auto-restarts
# ashleyklieu_trigger_executor.py on any crash or clean exit.
# Mirrors run_ashleyklieu_monitor.ps1's pattern exactly. The Python script
# itself is now a persistent, self-scheduling DAILY loop (CEO instruction
# 2026-08-27: "i think we should schedule daily during market hours") --
# this watchdog only needs to catch a genuine crash, not drive daily
# re-launch (the script handles that on its own).

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "ashleyklieu_trigger_executor.py"
$WatchdogLog = Join-Path $BackendDir "ashleyklieu_trigger_executor_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching ashleyklieu_trigger_executor.py..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -RedirectStandardOutput (Join-Path $BackendDir "ashleyklieu_trigger_executor.log") `
            -RedirectStandardError (Join-Path $BackendDir "ashleyklieu_trigger_executor.err.log") `
            -NoNewWindow `
            -PassThru `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Executor exited (code=$code) -- this should not happen since the script loops forever. Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch executor: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
