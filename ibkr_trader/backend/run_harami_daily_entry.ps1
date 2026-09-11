# run_harami_daily_entry.ps1 -- wrapper for the 4:20pm ET weekday harami
# live-entry task (harami_daily_trader.py --mode entry). Runs AFTER
# harami_scanner.py's 4:10pm scan so today's fresh alerts are in
# oversight_log.jsonl by the time this fires.
#
# Same stdout/stderr capture discipline as run_butterfly_entry.ps1 /
# run_chartexpert_auto_trader.ps1 (System.Diagnostics.Process, NOT
# Start-Process -PassThru -- confirmed on this machine that combination
# returns an empty $proc.ExitCode). This is a REAL-MONEY trading task;
# a silent failure here is not acceptable.
#
# Usage (Scheduled Task action): powershell -File run_harami_daily_entry.ps1

$BackendDir = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$Python     = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$Script     = Join-Path $BackendDir "harami_daily_trader.py"
$CrashDir   = Join-Path $BackendDir "crash_logs"
New-Item -ItemType Directory -Force -Path $CrashDir | Out-Null

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $CrashDir "harami_entry_$($stamp)_stdout.log"
$ErrLog = Join-Path $CrashDir "harami_entry_$($stamp)_stderr.log"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = "`"$Script`" --mode entry"
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
        $text = "&#x1F6A8; harami_daily_trader --mode entry FAILED (exit $code) -- REAL MONEY task, check now.`n`nLast stdout:`n$tail"
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
        category  = "harami_live_entry_task_failed"
        summary   = "harami_daily_trader --mode entry task exited $code. Full logs: $OutLog / $ErrLog"
        rationale = "Captured via run_harami_daily_entry.ps1 wrapper."
        outcome   = "Telegram alert sent with log tail. Logs preserved in crash_logs/ for full detail."
        pnl_impact = $null
    } | ConvertTo-Json -Compress
    Add-Content -Path (Join-Path $BackendDir "oversight_log.jsonl") -Value $entryJson
}

exit $code
