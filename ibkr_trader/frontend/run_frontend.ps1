# run_frontend.ps1 -- Watchdog: serves index.html on port 8001, auto-restarts on crash.
# Launched by Windows Task Scheduler at logon; runs hidden with no console window.
#
# NOTE (2026-09-10): kept strictly ASCII -- Windows PowerShell 5.1 reads .ps1 as the
# system ANSI codepage without a BOM, and em-dash chars here previously broke the
# parser mid-string, so this watchdog silently never ran and the frontend had no
# crash-recovery.
#
# NOTE (2026-09-10, second fix): the previous version used
#   Start-Process -NoNewWindow -PassThru  +  $proc.WaitForExit()
# On this machine that handle's WaitForExit() returns IMMEDIATELY and .ExitCode
# comes back empty, so the while-loop spun -- every ~6s it saw its own child as a
# "stale" listener, killed it, and relaunched (log showed double "Server started"
# lines 2s apart). Two logon sessions each ran a copy, so two watchdogs fought over
# port 8001, each killing the other's child, until both loops died and left an
# orphaned bare server. Fix: use System.Diagnostics.Process, whose WaitForExit()
# genuinely blocks and whose .ExitCode is populated; only treat a port holder as
# stale if it is NOT the child we are currently supervising; wrap the port probe in
# try/catch so a transient Get-NetTCPConnection failure can never kill the loop.

$FrontendDir = "C:\Projects\GenAI-Projects\ibkr_trader\frontend"
$Python      = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Port        = 8001
$WatchdogLog = Join-Path $FrontendDir "frontend_watchdog.log"

function Write-WLog($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    try { "[$ts] $msg" | Out-File -FilePath $WatchdogLog -Encoding utf8 -Append } catch { }
}

Write-WLog "Frontend watchdog started (PID=$PID)"

while ($true) {
    $childPid = 0

    # Kill any OTHER process already listening on the port (never our own child).
    try {
        $stale = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        foreach ($c in @($stale)) {
            if ($c.OwningProcess -and $c.OwningProcess -ne $childPid) {
                Write-WLog "Port $Port held by PID=$($c.OwningProcess) -- killing stale listener"
                Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Seconds 1
    } catch {
        # no listener yet, or the probe cmdlet hiccuped -- either way just proceed
    }

    Write-WLog "Launching http.server on port $Port..."
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName               = $Python
        $psi.Arguments              = "-m http.server $Port --directory `"$FrontendDir`""
        $psi.WorkingDirectory       = $FrontendDir
        $psi.UseShellExecute        = $false
        $psi.CreateNoWindow         = $true
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError  = $false

        $proc = [System.Diagnostics.Process]::Start($psi)
        $childPid = $proc.Id
        Write-WLog "Server started (PID=$childPid)"

        $proc.WaitForExit()   # genuinely blocks until the child dies

        $code = try { $proc.ExitCode } catch { "unknown" }
        Write-WLog "Server exited (code=$code). Restarting in 5s..."
    } catch {
        Write-WLog ("Failed to launch server: {0}. Retrying in 5s..." -f $_)
    }
    Start-Sleep -Seconds 5
}
