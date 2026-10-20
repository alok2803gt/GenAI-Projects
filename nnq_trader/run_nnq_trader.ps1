# run_nnq_trader.ps1 -- Watchdog: keeps nnq_trader.py running around the clock (NNQ trades ~24h),
# restarting it 15 s after any exit. Started by task NNQ-Trader at logon (+ daily backup trigger).
$Dir = "C:\Projects\GenAI-Projects\nnq_trader"
$Py = "$Dir\.venv\Scripts\python.exe"
$Log = "$Dir\nnq_watchdog.log"
$Out = "$Dir\nnq_console.log"

function Write-WLog($m) { Add-Content $Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }

Write-WLog "Watchdog started (PID=$PID)"
while ($true) {
    Write-WLog "Launching nnq_trader.py..."
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo.FileName = "cmd.exe"
    $p.StartInfo.Arguments = "/c `"`"$Py`" -u `"$Dir\nnq_trader.py`" >> `"$Out`" 2>&1`""
    $p.StartInfo.WorkingDirectory = $Dir
    $p.StartInfo.UseShellExecute = $false
    $p.StartInfo.CreateNoWindow = $true
    [void]$p.Start()
    $p.WaitForExit()
    Write-WLog "Exited (code=$($p.ExitCode)). Restarting in 15s..."
    Start-Sleep -Seconds 15
}
