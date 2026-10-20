# run_vnkumar_shadow.ps1 -- Watchdog: keeps vnkumar_shadow.py running around the clock, restarting it
# 15 s after any exit (crash, network loss, Telegram disconnect). Started by task VNKumar-Shadow at logon.
$Dir = "C:\Projects\GenAI-Projects\telegram_reader"
$Py = "$Dir\.venv\Scripts\python.exe"
$Log = "$Dir\vnkumar_watchdog.log"
$Out = "$Dir\vnkumar_console.log"

function Write-WLog($m) { Add-Content $Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }

Write-WLog "Watchdog started (PID=$PID)"
while ($true) {
    Write-WLog "Launching vnkumar_shadow.py..."
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo.FileName = "cmd.exe"
    $p.StartInfo.Arguments = "/c `"`"$Py`" -u `"$Dir\vnkumar_shadow.py`" >> `"$Out`" 2>&1`""
    $p.StartInfo.WorkingDirectory = $Dir
    $p.StartInfo.UseShellExecute = $false
    $p.StartInfo.CreateNoWindow = $true
    [void]$p.Start()
    $p.WaitForExit()
    Write-WLog "Exited (code=$($p.ExitCode)). Restarting in 15s..."
    Start-Sleep -Seconds 15
}
