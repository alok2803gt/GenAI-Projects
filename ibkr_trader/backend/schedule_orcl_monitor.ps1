Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

Write-Host "ORCL Monday Monitor — Task Scheduler Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Find python
$py = $null
foreach ($candidate in @(
    (Get-Command python -ErrorAction SilentlyContinue)?.Source,
    "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe"
)) {
    if ($candidate -and (Test-Path $candidate)) { $py = $candidate; break }
}

if (-not $py) {
    Write-Host "ERROR: Cannot find python.exe" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$script = "c:\Projects\GenAI-Projects\ibkr_trader\backend\orcl_monday_monitor.py"
$wd     = "c:\Projects\GenAI-Projects\ibkr_trader\backend"

if (-not (Test-Path $script)) {
    Write-Host "ERROR: Script not found at $script" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Python:  $py"
Write-Host "Script:  $script"
Write-Host ""

# Remove old task silently
Unregister-ScheduledTask -TaskName "ORCL_Monday_Monitor" -Confirm:$false -ErrorAction SilentlyContinue

# Use schtasks.exe — no elevation needed for current user tasks
$runDate = "07/27/2026"
$runTime = "09:25"
$cmd = "`"$py`" `"$script`""

$result = & schtasks.exe /Create /F /TN "ORCL_Monday_Monitor" `
    /TR $cmd /SC ONCE /SD $runDate /ST $runTime 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Task registered!" -ForegroundColor Green
    Write-Host ""
    & schtasks.exe /Query /TN "ORCL_Monday_Monitor" /FO LIST
} else {
    Write-Host "schtasks failed: $result" -ForegroundColor Yellow
    Write-Host "Trying PowerShell Register-ScheduledTask..." -ForegroundColor Yellow

    try {
        $action   = New-ScheduledTaskAction -Execute $py -Argument "`"$script`"" -WorkingDirectory $wd
        $trigger  = New-ScheduledTaskTrigger -Once -At "2026-07-27T09:25:00"
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -StartWhenAvailable
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
        $task = Register-ScheduledTask -TaskName "ORCL_Monday_Monitor" `
            -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force

        Write-Host "SUCCESS via PowerShell!" -ForegroundColor Green
        Get-ScheduledTaskInfo -TaskName "ORCL_Monday_Monitor" | Select-Object NextRunTime | Format-List
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "MANUAL FALLBACK — paste this in any PowerShell window:" -ForegroundColor Yellow
        Write-Host "schtasks /Create /F /TN ORCL_Monday_Monitor /TR `"$py $script`" /SC ONCE /SD 07/27/2026 /ST 09:25" -ForegroundColor White
    }
}

Write-Host ""
Read-Host "Press Enter to close"
