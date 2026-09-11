# run_butterfly_entry.ps1 -- one-shot wrapper for the daily 0DTE butterfly
# entry tasks (SPYButterflyFixed, QQQButterflyWindow, IWMButterflyWindow).
#
# Built 2026-09-09 after IBKR-SPYButterflyFixed silently exited 1 on 9/8
# with no resulting position and no way to know why -- Task Scheduler
# doesn't capture a task's console output on its own, and
# alpaca_0dte_butterfly_trader.py has several distinct real sys.exit(1)
# paths (CRO/CFO cap rejection, non-positive debit, entry incomplete,
# near-S/R gate).
#
# NOTE on implementation: originally used Start-Process -PassThru with
# -RedirectStandardOutput/-RedirectStandardError (same pattern
# run_backend.ps1 already uses) -- but empirically confirmed on this
# machine (2026-09-09) that combination reliably returns an EMPTY
# $proc.ExitCode even after WaitForExit()/Refresh(), which would have made
# this wrapper treat every run as a failure ($null -ne 0 is $true) and
# spam a false failure alert on every single real success. Switched to the
# raw System.Diagnostics.Process API with async stream reads instead,
# which was verified to return the real exit code correctly. Do not revert
# to Start-Process -PassThru for this without re-verifying that bug is
# gone first.
#
# Usage (called from a Scheduled Task action, not interactively):
#   powershell -File run_butterfly_entry.ps1 -Ticker spy -EntryMode fixed -TaskLabel "SPY Fixed"

param(
    [Parameter(Mandatory=$true)][string]$Ticker,
    [Parameter(Mandatory=$true)][ValidateSet("fixed","window")][string]$EntryMode,
    [Parameter(Mandatory=$true)][string]$TaskLabel,
    [double]$MaxRiskOverride = 500
)

$BackendDir = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python     = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script     = Join-Path $BackendDir "alpaca_0dte_butterfly_trader.py"
$CrashDir   = Join-Path $BackendDir "crash_logs"
New-Item -ItemType Directory -Force -Path $CrashDir | Out-Null

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $CrashDir "butterfly_entry_$($Ticker)_$($stamp)_stdout.log"
$ErrLog = Join-Path $CrashDir "butterfly_entry_$($Ticker)_$($stamp)_stderr.log"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = "`"$Script`" --ticker $Ticker --entry-mode $EntryMode --fire --max-risk-override $MaxRiskOverride"
$psi.WorkingDirectory = $BackendDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

# Async whole-stream reads (not per-line events): each ReadToEndAsync()
# drains its pipe continuously in true arrival order with no per-line
# threadpool dispatch, so unlike Register-ObjectEvent + OutputDataReceived
# (tried first, confirmed 2026-09-09 to reorder lines when several arrive
# in one buffered flush) this preserves real output order while still
# reading both streams concurrently, avoiding the classic redirect-both
# -and-read-sequentially deadlock.
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
        $text = "&#x1F6A8; $TaskLabel butterfly entry FAILED (exit $code) -- no position may have been entered.`n`nLast stdout:`n$tail"
        if ($errTail) { $text += "`n`nstderr:`n$errTail" }
        $uri  = "https://api.telegram.org/bot$($cfg.telegram_token)/sendMessage"
        $body = @{ chat_id = "$($cfg.telegram_chat_id)"; text = $text; parse_mode = "HTML" }
        Invoke-RestMethod -Uri $uri -Method Post -Body $body -TimeoutSec 10 | Out-Null
    } catch {
        Write-Host "Telegram alert failed: $($_.Exception.Message)"
    }

    # Durable audit trail entry, same schema/file every other oversight
    # finding in this codebase uses.
    $entryJson = @{
        time      = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
        actor     = "trader"
        category  = "butterfly_entry_task_failed"
        summary   = "$TaskLabel butterfly entry task exited $code, no position confirmed. Full logs: $OutLog / $ErrLog"
        rationale = "Captured via run_butterfly_entry.ps1 wrapper (added 2026-09-09 after a prior silent, unexplained failure)."
        outcome   = "Telegram alert sent with log tail. Logs preserved in crash_logs/ for full detail."
        pnl_impact = $null
    } | ConvertTo-Json -Compress
    Add-Content -Path (Join-Path $BackendDir "oversight_log.jsonl") -Value $entryJson
}

exit $code
