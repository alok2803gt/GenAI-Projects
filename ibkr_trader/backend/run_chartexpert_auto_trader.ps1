# run_chartexpert_auto_trader.ps1 -- wrapper for the 9:35 AM weekday
# chartexpert shadow-signal task (chartexpert_auto_trader.py).
#
# Same stdout/stderr capture discipline as run_butterfly_entry.ps1 (built
# 2026-09-09 after a butterfly entry task silently exited 1 with no way to
# know why): Task Scheduler doesn't capture a task's console output on its
# own, and this script has several real failure paths (dead Anthropic API
# key, IBKR not connected, frontend down, no 0DTE listed today) that need
# to be visible, not silent.
#
# Uses System.Diagnostics.Process with async stream reads, NOT
# Start-Process -PassThru -RedirectStandardOutput/-RedirectStandardError --
# confirmed on this machine (2026-09-09, run_butterfly_entry.ps1) that
# combination reliably returns an EMPTY $proc.ExitCode even after
# WaitForExit(), which would make this wrapper treat every real success as
# a failure. Do not revert without re-verifying that bug is gone first.
#
# Can run long: on a signal day this script blocks tracking the shadow
# position until a trailing-stop or the 15:50 ET cutoff, so this wrapper
# does not impose its own timeout beyond what Task Scheduler's own
# execution time limit allows.
#
# Usage (Scheduled Task action): powershell -File run_chartexpert_auto_trader.ps1

$BackendDir = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python     = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script     = Join-Path $BackendDir "chartexpert_auto_trader.py"
$CrashDir   = Join-Path $BackendDir "crash_logs"
New-Item -ItemType Directory -Force -Path $CrashDir | Out-Null

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $CrashDir "chartexpert_auto_$($stamp)_stdout.log"
$ErrLog = Join-Path $CrashDir "chartexpert_auto_$($stamp)_stderr.log"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = "`"$Script`""
$psi.WorkingDirectory = $BackendDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null
$outTask = $proc.StandardOutput.ReadToEndAsync()
$errTask = $proc.StandardError.ReadToEndAsync()
$proc.WaitForExit()

$code = $proc.ExitCode
$stdoutText = $outTask.Result
$stderrText = $errTask.Result
Set-Content -Path $OutLog -Value $stdoutText -Encoding utf8
Set-Content -Path $ErrLog -Value $stderrText -Encoding utf8

if ($code -ne 0) {
    $tailLines = ($stdoutText -split "`r?`n") | Select-Object -Last 15
    $tail = $tailLines -join "`n"
    $errTailLines = ($stderrText -split "`r?`n") | Select-Object -Last 8
    $errTail = ($errTailLines -join "`n").Trim()

    try {
        $cfg = Get-Content (Join-Path $BackendDir "scanner_config.json") -Raw | ConvertFrom-Json
        $text = "&#x1F6A8; chartexpert_auto_trader FAILED (exit $code).`n`nLast stdout:`n$tail"
        if ($errTail) { $text += "`n`nstderr:`n$errTail" }
        $uri  = "https://api.telegram.org/bot$($cfg.telegram_token)/sendMessage"
        $body = @{ chat_id = "$($cfg.telegram_chat_id)"; text = $text; parse_mode = "HTML" }
        Invoke-RestMethod -Uri $uri -Method Post -Body $body -TimeoutSec 10 | Out-Null
    } catch {
        Write-Host "Telegram alert failed: $($_.Exception.Message)"
    }

    $entryJson = @{
        time      = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
        actor     = "trader"
        category  = "chartexpert_auto_task_failed"
        summary   = "chartexpert_auto_trader task exited $code. Full logs: $OutLog / $ErrLog"
        rationale = "Captured via run_chartexpert_auto_trader.ps1 wrapper (same discipline as run_butterfly_entry.ps1)."
        outcome   = "Telegram alert sent with log tail. Logs preserved in crash_logs/ for full detail."
        pnl_impact = $null
    } | ConvertTo-Json -Compress
    Add-Content -Path (Join-Path $BackendDir "oversight_log.jsonl") -Value $entryJson
}

exit $code
