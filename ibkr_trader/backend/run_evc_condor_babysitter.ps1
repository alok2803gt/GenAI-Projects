# run_evc_condor_babysitter.ps1 -- Watchdog: auto-restarts evc_condor_babysitter.py on any crash or clean exit.
# Mirrors run_ashleyklieu_monitor.ps1's pattern exactly. Generic monitor for
# ALL currently-open EVC iron condors -- watch-and-alert only, no order
# placement (CEO instruction 2026-08-26: "lets start babysitter monitors
# for all iron condors").

$BackendDir  = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script      = Join-Path $BackendDir "evc_condor_babysitter.py"
$WatchdogLog = Join-Path $BackendDir "evc_condor_babysitter_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append
}

Write-WLog "Watchdog started (PID=$PID)"

while ($true) {
    Write-WLog "Launching evc_condor_babysitter.py..."
    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-u", "`"$Script`"" `
            -WorkingDirectory $BackendDir `
            -RedirectStandardOutput (Join-Path $BackendDir "evc_condor_babysitter.log") `
            -RedirectStandardError (Join-Path $BackendDir "evc_condor_babysitter.err.log") `
            -NoNewWindow `
            -PassThru `
            -ErrorAction Stop
        $proc.WaitForExit()
        $code = $proc.ExitCode
        Write-WLog "Babysitter exited cleanly (code=$code). Restarting in 15s..."
    } catch {
        Write-WLog "Failed to launch babysitter: $_. Retrying in 15s..."
    }
    Start-Sleep -Seconds 15
}
