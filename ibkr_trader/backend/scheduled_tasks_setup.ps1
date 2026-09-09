<#
Creates 5 Windows Scheduled Tasks so the IBKR trader account's standing
monitoring/automation keeps running even when no Claude Code session is
open. All 5 point at existing, self-contained, deterministic Python
scripts (no LLM involved at runtime, no bypassed permissions) --
IBKR-PortfolioOversight and IBKR-CRORiskCheck are new scripts built and
tested 2026-08-23; the other 3 already existed and already ran unattended
via in-session cron triggers.

Repetition note: Windows' ScheduledTasks module does not reliably support
setting -Repetition.Interval/.Duration on -Weekly/-Daily trigger objects
after creation (confirmed broken on this machine's PS/OS combo -- silent
partial failure, task still registers but never repeats). The fix used
here is the documented-reliable pattern instead: -Once + -RepetitionInterval
+ -RepetitionDuration as constructor parameters. Since that combination
can't also carry a native "weekdays only" constraint, the two hourly
checks (PortfolioOversight, CRORiskCheck) repeat around the clock at the
OS level and enforce the weekday restriction inside the Python script
itself (exits immediately on Sat/Sun) -- functionally identical result,
more robust than fighting the trigger object model.

Run this once, from a regular PowerShell window (elevation not required
for tasks running in the current user's own context):

    powershell -ExecutionPolicy Bypass -File "c:\Projects\GenAI-Projects\ibkr_trader\backend\scheduled_tasks_setup.ps1"

If you'd rather allow local scripts permanently instead of using
-ExecutionPolicy Bypass each time:

    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

then just:

    cd c:\Projects\GenAI-Projects\ibkr_trader\backend
    .\scheduled_tasks_setup.ps1

Re-running this script is safe -- Register-ScheduledTask -Force overwrites
any existing task of the same name rather than erroring or duplicating.
#>

$py = "C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
$backendDir = "C:\Projects\GenAI-Projects\ibkr_trader\backend"
$gexScript = "C:\Users\AlokD\.claude\skills\gex-vex-calculator\calc_gex_vex.py"
$weekdays = @('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')

# MultipleInstances IgnoreNew = Windows' own native duplicate-run guard --
# if a prior firing of the same task is still running, the new one is
# skipped rather than launching a second, overlapping instance. This
# matters most for SPY0DTEAutoFire, which can legitimately run for up to
# ~6 hours.
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 7)

Write-Host "Creating IBKR-PortfolioOversight (hourly around the clock; script itself skips Sat/Sun)..."
$action1 = New-ScheduledTaskAction -Execute $py -Argument "portfolio_oversight_check.py" -WorkingDirectory $backendDir
$startTime1 = (Get-Date -Hour 0 -Minute 13 -Second 0)
$trigger1 = New-ScheduledTaskTrigger -Once -At $startTime1 -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-PortfolioOversight" -Action $action1 -Trigger $trigger1 -Settings $settings `
    -Description "Unattended hourly portfolio-oversight check (deterministic script, no LLM; weekday guard inside the script)" -Force | Out-Null

Write-Host "Creating IBKR-CRORiskCheck (hourly around the clock; script itself skips Sat/Sun)..."
$action2 = New-ScheduledTaskAction -Execute $py -Argument "cro_risk_check.py" -WorkingDirectory $backendDir
$startTime2 = (Get-Date -Hour 0 -Minute 43 -Second 0)
$trigger2 = New-ScheduledTaskTrigger -Once -At $startTime2 -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-CRORiskCheck" -Action $action2 -Trigger $trigger2 -Settings $settings `
    -Description "Unattended hourly CRO risk check (deterministic script, no LLM; weekday guard inside the script)" -Force | Out-Null

Write-Host "Creating IBKR-NFLXMonitor (every 2h at :17)..."
$action3 = New-ScheduledTaskAction -Execute $py -Argument "nflx_bottom_monitor.py" -WorkingDirectory $backendDir
$startTime3 = (Get-Date -Hour 0 -Minute 17 -Second 0)
$trigger3 = New-ScheduledTaskTrigger -Once -At $startTime3 -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-NFLXMonitor" -Action $action3 -Trigger $trigger3 -Settings $settings `
    -Description "NFLX bottom-watch monitor (existing deterministic script)" -Force | Out-Null

Write-Host "Creating IBKR-GEXArchiver (daily weekdays 10:14am)..."
$action4 = New-ScheduledTaskAction -Execute $py -Argument "`"$gexScript`"" -WorkingDirectory $backendDir
$trigger4 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "10:14AM"
Register-ScheduledTask -TaskName "IBKR-GEXArchiver" -Action $action4 -Trigger $trigger4 -Settings $settings `
    -Description "Daily GEX/VEX archiver (existing deterministic script)" -Force | Out-Null

Write-Host "Creating IBKR-SPY0DTEAutoFire (daily weekdays 9:43am)..."
$action5 = New-ScheduledTaskAction -Execute $py -Argument "spy_0dte_auto.py --ticker spy" -WorkingDirectory $backendDir
$trigger5 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "9:43AM"
Register-ScheduledTask -TaskName "IBKR-SPY0DTEAutoFire" -Action $action5 -Trigger $trigger5 -Settings $settings `
    -Description "SPY 0DTE unattended auto-fire (existing deterministic script, CEO-approved 2026-08-17)" -Force | Out-Null

Write-Host "Creating IBKR-DarkPoolLevelsArchiver (daily weekdays 10:20am)..."
$darkpoolScript = "C:\Users\AlokD\.claude\skills\darkpool-levels-calculator\calc_darkpool_levels.py"
$darkpoolDir = "C:\Users\AlokD\.claude\skills\darkpool-levels-calculator"
$action6 = New-ScheduledTaskAction -Execute $py -Argument "`"$darkpoolScript`"" -WorkingDirectory $darkpoolDir
$trigger6 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "10:20AM"
Register-ScheduledTask -TaskName "IBKR-DarkPoolLevelsArchiver" -Action $action6 -Trigger $trigger6 -Settings $settings `
    -Description "Daily dark-pool price-level archiver for safe-income-screener's DarkPoolWall column (informational only, not backtest-validated)" -Force | Out-Null

Write-Host "Creating IBKR-SafeIncomeScreenerLog (daily weekdays 1:15pm)..."
$screenerScript = "C:\Users\AlokD\.claude\skills\safe-income-screener\screen.py"
$screenerDir = "C:\Users\AlokD\.claude\skills\safe-income-screener"
$action7 = New-ScheduledTaskAction -Execute $py -Argument "`"$screenerScript`" --log" -WorkingDirectory $screenerDir
$trigger7 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "1:15PM"
Register-ScheduledTask -TaskName "IBKR-SafeIncomeScreenerLog" -Action $action7 -Trigger $trigger7 -Settings $settings `
    -Description "Daily full-universe safe-income-screener scan with --log, so picks_log.jsonl accumulates real samples for track_outcomes.py to eventually validate GEX/VEX/DarkPoolWall against real outcomes. Moved from 10:30am to 1:15pm on 2026-09-02 after finding GEX/VEX was 0/120 populated in every logged pick since 8/24 -- root cause confirmed: IBKR-GEXArchiver starts 10:14am but writes gex_vex_cache.json only ONCE at the very end of its run (single json.dump after scanning the full ~113-ticker universe), and a real run observed 2026-09-02 took until 11:57am (103min for 77 tickers) -- so the 10:30am log run always saw yesterday's stale cache and screen.py's own freshness gate (computed_at_date != today) correctly rejected it, showing net_gex/net_vex as null every time. 1:15pm gives safe margin past the slowest observed real completion." -Force | Out-Null

# IBKR-PDDCondorBabysitter removed 2026-08-25 -- PDD condor confirmed closed
# for real (0 legs held on Alpaca), monitoring no longer needed. Task
# unregistered directly; pdd_condor_babysitter.py left on disk as a record
# and as the reusable pattern intu_condor_babysitter.py was built from.

# IBKR-INTUCondorBabysitter removed 2026-09-02 -- INTU position confirmed
# closed (0 legs held on either IBKR or Alpaca); the task's own "self-expires
# after 8/28" description was checked against real Get-ScheduledTaskInfo
# state and found to be wrong (it was still actively firing). Unregistered
# directly; intu_condor_babysitter.py left on disk as a record.

Write-Host "Creating IBKR-SafeIncomeTrader (twice daily, market-hours-only via in-script guard) -- PLACES REAL TRADES..."
$action9 = New-ScheduledTaskAction -Execute $py -Argument "safe_income_auto.py" -WorkingDirectory $backendDir
$trigger9a = New-ScheduledTaskTrigger -Daily -At "10:00AM"
$trigger9b = New-ScheduledTaskTrigger -Daily -At "2:00PM"
Register-ScheduledTask -TaskName "IBKR-SafeIncomeTrader" -Action $action9 -Trigger @($trigger9a, $trigger9b) -Settings $settings `
    -Description "Safe Income Trader -- automates the validated deep-OTM credit-spread strategy. Fires twice daily (10:00 AM and 2:00 PM ET, changed from hourly on 2026-08-24 since this places real live trades and hourly was unnecessarily frequent for a hold-to-expiry strategy) at the OS level; the script itself enforces weekday 9:30-16:00 ET and skips outside that window (including weekends, so the daily trigger firing on Sat/Sun is a no-op). Scans an 11-ticker curated universe (SPY/QQQ/NVDA/AAPL/GOOGL/IWM/GLD/LLY/TLT/WMT/V) -- liquidity-ranked, then filtered to exclude high-realized-vol names (>=40%) that structurally mismatch this methodology's own validated backtest (14 of the original top-25-by-liquidity list were high-vol, incl. TSLA/AMD which both cleared every other real gate), cushion lowered from screen.py's 15% default to 12% for this universe (real backtest-validated). Real trades, real capital -- max 1 new position per run by default (--max-new), CRO/CFO pre-trade review (incl. a hard high-vol reject, not just informational, plus per-trade and portfolio-headroom capital checks) before every entry, hold-to-expiry exit." -Force | Out-Null

Write-Host "Creating IBKR-GOOGCondorMonday (weekly, Monday 9:45am) -- PLACES REAL TRADES..."
$action10 = New-ScheduledTaskAction -Execute $py -Argument "goog_condor_monday_entry.py" -WorkingDirectory $backendDir
$trigger10 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "9:45AM"
Register-ScheduledTask -TaskName "IBKR-GOOGCondorMonday" -Action $action10 -Trigger $trigger10 -Settings $settings `
    -Description "GOOG weekly iron condor, Monday-only entry (matches the validated backtest cycle -- the wrapper itself refuses to run on any other day and computes that week's real Friday expiry, no hardcoded/stale date risk). Real trades -- CRO/CFO pre-trade review with a 15% GOOG-specific per-trade cap runs before every order." -Force | Out-Null

# IBKR-GOOGResidualPutMonitor removed 2026-09-02 -- GOOG position confirmed
# closed (0 legs held on either IBKR or Alpaca); the task's own "self-expires
# after 8/28" description was checked against real Get-ScheduledTaskInfo
# state and found to be wrong (it was still actively firing). Unregistered
# directly; goog_residual_put_monitor.py left on disk as a record.

Write-Host "Creating IBKR-DarkPoolActivityMonitor (every 15min, market-hours-only via in-script guard) -- alert-only..."
$action12 = New-ScheduledTaskAction -Execute $py -Argument "darkpool_activity_monitor.py" -WorkingDirectory $backendDir
$startTime12 = (Get-Date).AddMinutes(4)
$trigger12 = New-ScheduledTaskTrigger -Once -At $startTime12 -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-DarkPoolActivityMonitor" -Action $action12 -Trigger $trigger12 -Settings $settings `
    -Description "Polls Unusual Whales' real dark-pool trade tape (unusual_whales_client.py) every 15min (weekday-only, NOT market-hours-gated -- read-only alert script, no execution risk, and real extended-hours dark-pool activity confirmed live 2026-08-24) and sends ONE Telegram digest per run for new, non-canceled prints clearing a RELATIVE per-ticker threshold (0.5% of that ticker's own 20-day avg dollar volume, `$1M absolute floor -- changed from a flat `$5M 2026-08-24 after confirming empirically a flat figure is up to ~113x more/less strict depending on the ticker, e.g. `$5M = 0.0145% of MU/SPY's ADV vs 1.64% of LULU's) in the curated ~112-ticker universe, within the last 20min only (time-bounded, not just dedup). Dark pool is the master list; cross-references breakout_scanner.py's same-day ticker_states_{date}.json to flag/sort (never filter) any ticker currently BREAKOUT/PRE-BREAKOUT/EXTENDED -- two independently-real signals shown together, not a combined/backtested signal. Read-only, alert-only -- never places an order, never touches any strategy's config or position state. Purely informational: a print is size/price only, no buy/sell aggressor side, not validated as predictive (same epistemic status as darkpool-levels-calculator's GammaWall-style wall concept)." -Force | Out-Null

Write-Host "Creating IBKR-BreakoutShadowFilter (every 15min, market-hours-only via in-script guard) -- research/logging only, does NOT touch breakout_scanner.py..."
$action13 = New-ScheduledTaskAction -Execute $py -Argument "shadow_filter_monitor.py" -WorkingDirectory "$backendDir\breakout_research"
$startTime13 = (Get-Date).AddMinutes(4)
$trigger13 = New-ScheduledTaskTrigger -Once -At $startTime13 -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-BreakoutShadowFilter" -Action $action13 -Trigger $trigger13 -Settings $settings `
    -Description "Research infra (breakout_research/), NOT part of the live trading system -- read-only against breakout_scanner.py's real alert_history table, never writes to it, never touches breakout_scanner.py or its config. Tags each new real alert with a 5-level quality tier (AVOID / NEUTRAL / PREFER-LOW / PREFER-MID / PREFER-HIGH -- see RESEARCH_LOG.md Iterations 4+7) using pct_b/RSI/vol_ratio from alert_history plus ADX/distance-from-52-week-high computed in real time via yfinance (not live-scanner columns). AVOID validated against 537 real live PRE-BREAKOUT alerts (46.8%->45.7% win rate/-0.09%->-0.37% avg by +5d vs the rest). PREFER-HIGH/MID/LOW sub-ranking validated on the 2.5y gate-accurate offline backtest (monotonic on ret_3d/ret_5d, PREFER-LOW real and negative -- worse than NEUTRAL). Logs every tier to shadow_filter_log.csv for ongoing live out-of-sample validation. Sends ONE Telegram digest per run (never one per alert) for PREFER-HIGH only, per CEO request 2026-08-25 -- every message states plainly this is a research tier, not a trade signal. No trading, no gating -- runs alongside the real scanner, doesn't feed back into it." -Force | Out-Null

Write-Host "Creating IBKR-SPYButterflyFixed (daily weekdays 9:45am) -- PLACES REAL TRADES..."
$action14 = New-ScheduledTaskAction -Execute $py -Argument "alpaca_0dte_butterfly_trader.py --ticker spy --entry-mode fixed --fire --max-risk-override 500" -WorkingDirectory $backendDir
$trigger14 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "9:45AM"
Register-ScheduledTask -TaskName "IBKR-SPYButterflyFixed" -Action $action14 -Trigger $trigger14 -Settings $settings `
    -Description "SPY 0DTE long call butterfly, wing_step=4, single near-S/R check at 9:45 ET (real backtest 2026-09-02: SPY specifically is the strongest ticker under this fixed-check design -- window-scan tested NEGATIVE for SPY, -`$7.36/contract, which is why SPY stays on its own fixed check while QQQ/IWM use window-scan instead). Skips the day entirely (exit code 0, not an error) if the 9:45 spot isn't within 0.3% of the prior day's real high/low. Real trades -- CRO/CFO pre-trade review runs before every order. Alpaca auto-closes the position at 15:45 ET on its own (confirmed live 2026-09-02); the script verifies that rather than racing it, only manually closing as a fallback if the position is unexpectedly still open past 15:47." -Force | Out-Null

Write-Host "Creating IBKR-QQQButterflyWindow (daily weekdays 9:35am) -- PLACES REAL TRADES..."
$action15 = New-ScheduledTaskAction -Execute $py -Argument "alpaca_0dte_butterfly_trader.py --ticker qqq --entry-mode window --fire --max-risk-override 500" -WorkingDirectory $backendDir
$trigger15 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "9:35AM"
Register-ScheduledTask -TaskName "IBKR-QQQButterflyWindow" -Action $action15 -Trigger $trigger15 -Settings $settings `
    -Description "QQQ 0DTE long call butterfly, wing_step=4, window-scan entry -- polls every 30s from launch until 10:00 ET for a near-S/R trigger (0.3% of prior day's real high/low), entering the moment it first fires. Real backtest 2026-09-02: this design beats a fixed-9:45 check for QQQ specifically (+`$19.49/contract, 54.1% win vs the fixed check's weaker number). Skips the day (exit code 0) if the window closes with no trigger. Real trades -- CRO/CFO pre-trade review runs before every order. Alpaca auto-closes at 15:45 ET on its own; the script verifies rather than races it." -Force | Out-Null

Write-Host "Creating IBKR-IWMButterflyWindow (daily weekdays 9:35am) -- PLACES REAL TRADES..."
$action16 = New-ScheduledTaskAction -Execute $py -Argument "alpaca_0dte_butterfly_trader.py --ticker iwm --entry-mode window --fire --max-risk-override 500" -WorkingDirectory $backendDir
$trigger16 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "9:35AM"
Register-ScheduledTask -TaskName "IBKR-IWMButterflyWindow" -Action $action16 -Trigger $trigger16 -Settings $settings `
    -Description "IWM 0DTE long call butterfly, wing_step=4, window-scan entry -- same design as IBKR-QQQButterflyWindow. Real backtest 2026-09-02: IWM was the STRONGEST ticker under the window-scan design of all three (+`$39.18/contract, 73.5% win, on n=34 -- a real but still relatively thin sample). Skips the day (exit code 0) if the window closes with no trigger. Real trades -- CRO/CFO pre-trade review runs before every order. Alpaca auto-closes at 15:45 ET on its own; the script verifies rather than races it." -Force | Out-Null

Write-Host "Creating IBKR-QQQButterflyBabysitter (every 10min around the clock; script self-gates to weekday 9:30-16:00 ET)..."
$action17 = New-ScheduledTaskAction -Execute $py -Argument "qqq_butterfly_babysitter.py" -WorkingDirectory $backendDir
$startTime17 = (Get-Date -Hour 0 -Minute 27 -Second 0)
$trigger17 = New-ScheduledTaskTrigger -Once -At $startTime17 -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-QQQButterflyBabysitter" -Action $action17 -Trigger $trigger17 -Settings $settings `
    -Description "Standing babysitter for the daily QQQ 0DTE butterfly, built 2026-09-04 after a real, confirmed gap: the butterfly trader's own close-fallback records phase=closed unconditionally even when the close did not fully confirm, and its warning only prints to stdout, never Telegram/oversight_log -- exactly how QQQ's 2026-09-03 partial close (2 of 3 legs) went unnoticed for hours. Before 15:47 ET: lightweight informational check only (no alerting), PLUS (added same day) a real intraday INSIGHT check at most every 30min while the position is open -- computes live QQQ GEX for today's actual 0DTE expiry (direct IBKR modelGreeks+OI band scan, not the 25-45 DTE gex-vex-calculator skill which explicitly skips QQQ), real dark-pool prints via Unusual Whales, and live mark-to-market P&L/proximity-to-wings, then sends ONE synthesized Telegram insight recommending whether an early close looks worth considering -- this is alert-only and never places or closes an order itself, since 'cut a loss early' is a discretionary trading judgment for the CEO, not a mechanical failure to auto-remediate. From 15:47 onward: verifies all 3 real legs are actually flat on Alpaca. If yes, logs a normal confirmation (flags for manual reconciliation if close_pnl isn't already recorded -- deliberately does not auto-compute a number, since MLEG-combo reconstruction proved fragile even done by hand this week). If any leg is still really open, sends a HIGH-PRIORITY Telegram immediately and attempts a real emergency close via the same proven ladder every other close in this codebase uses -- alerts regardless of whether the auto-fix succeeds, so this can never again go unnoticed. QQQ fires fresh every weekday, so this is a standing daily check, not a one-off position monitor." -Force | Out-Null

Write-Host "Creating IBKR-IWMButterflyBabysitter (every 10min around the clock; script self-gates to weekday 9:30-16:00 ET)..."
$action18 = New-ScheduledTaskAction -Execute $py -Argument "iwm_butterfly_babysitter.py" -WorkingDirectory $backendDir
$startTime18 = (Get-Date -Hour 0 -Minute 32 -Second 0)
$trigger18 = New-ScheduledTaskTrigger -Once -At $startTime18 -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "IBKR-IWMButterflyBabysitter" -Action $action18 -Trigger $trigger18 -Settings $settings `
    -Description "Standing babysitter for the daily IWM 0DTE butterfly, built 2026-09-04 as the second instance of the same pattern already deployed for QQQ (shared logic in butterfly_babysitter_common.py), after the CEO flagged IWM chopping sideways intraday and wanted the same early-exit insight coverage. Before 15:47 ET: lightweight informational check only (no alerting), PLUS a real intraday GEX/dark-pool/momentum INSIGHT check at most every 30min while the position is open -- computes live IWM GEX for today's actual 0DTE expiry using IBKR's real listed strikes (not an assumed increment -- IWM's real strikes near the money are NOT uniformly spaced, unlike QQQ's), real dark-pool prints via Unusual Whales, and live mark-to-market P&L/proximity-to-wings using the correct asymmetric-wing math (today's real IWM butterfly is 291/294/297.5, a 3.0/3.5 split, not symmetric), then sends ONE synthesized Telegram insight -- alert-only, never places or closes an order itself. From 15:47 onward: verifies all 3 real legs are actually flat on Alpaca, same close-fallback safety net as QQQ's babysitter (the registry's close_position() can mark phase=closed even on a partial close). If any leg is still really open, sends a HIGH-PRIORITY Telegram and attempts a real emergency close via the same proven ladder every other close in this codebase uses. IWM fires fresh every weekday, so this is a standing daily check, not a one-off position monitor." -Force | Out-Null

Write-Host "Creating IBKR-HaramiScanner (weekdays, 4:10pm ET, alert-only)..."
$action19 = New-ScheduledTaskAction -Execute $py -Argument "candlestick_pattern_research\harami_scanner.py" -WorkingDirectory $backendDir
$trigger19 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "4:10PM"
Register-ScheduledTask -TaskName "IBKR-HaramiScanner" -Action $action19 -Trigger $trigger19 -Settings $settings `
    -Description "Alert-only daily scanner for the one validated finding from candlestick_pattern_research/ (2026-09-09): bullish harami + downtrend context (close below SMA20 AND SMA50), 5-trading-day hold -- real 5y/112-ticker backtest showed mean +1.073% vs a downtrend-matched baseline of +0.486% (Welch p=0.00007), positive in 76/111 tickers (binomial p=0.000062 vs the 50% pure-chance null) -- broad-based, not a few lucky names. One-shot daily run (not a poller) since the pattern only confirms once the day's candle is final; 4:10pm ET gives yfinance's daily bar a few minutes' buffer past the close. Universe loaded fresh from breakout_research's own 112-ticker pickle every run (not hand-typed, to avoid drifting from what was actually backtested). Alert-only -- flags real setups via Telegram with the backtested plan (enter next open, hold 5 trading days) for manual review, places no orders." -Force | Out-Null

Write-Host ""
Write-Host "=== Restart-safety tasks for always-on background processes (added 2026-09-03) ==="
Write-Host "These 6 processes (main.py backend, day_trader_agent.py, daytrader_scanner.py,"
Write-Host "ashleyklieu_alert_monitor.py, ashleyklieu_trigger_executor.py, breakout_scanner.py)"
Write-Host "were being started manually/out-of-band and left running with NO Task Scheduler"
Write-Host "entry at all -- confirmed live 2026-09-03: all 6 were running, but Get-ScheduledTask"
Write-Host "showed no logon/startup trigger anywhere pointing at them. Each already has its own"
Write-Host "crash-auto-restart watchdog wrapper (run_*.ps1, unchanged by this addition) -- the"
Write-Host "missing piece was just launching that watchdog automatically. This only affects"
Write-Host "future logons/reboots; it does NOT touch whatever is currently running."

# No ExecutionTimeLimit -- these are meant to run forever (the watchdog's own
# while(true) loop is the thing that never exits), unlike every other task in
# this file which either completes quickly or is capped at 7 hours.
$alwaysOnSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit ([TimeSpan]::Zero)
$currentUser = "$env:USERDOMAIN\$env:USERNAME"

$alwaysOn = @(
    @{ Name = "IBKR-BackendWatchdog";          Script = "run_backend.ps1";                     Desc = "At-logon launch of run_backend.ps1, which syntax-checks and runs main.py (port 8000, the main trading backend) with auto-restart on crash. The watchdog script itself already existed; this adds the missing Task Scheduler trigger." }
    @{ Name = "IBKR-DayTraderAgentWatchdog";   Script = "run_day_trader_agent.ps1";             Desc = "At-logon launch of run_day_trader_agent.ps1, which runs day_trader_agent.py (port 8010, independent live-trading Day Trader) with auto-restart on crash." }
    @{ Name = "IBKR-SPYWeeklyCondorAgentWatchdog"; Script = "run_spy_weekly_condor_agent.ps1";  Desc = "At-logon launch of run_spy_weekly_condor_agent.ps1, which runs spy_weekly_condor_agent.py (port 8011, independent SPY weekly condor -- extracted from main.py 2026-09-07, same pattern as Day Trader's own 2026-08-27 extraction) with auto-restart on crash." }
    @{ Name = "IBKR-DayTraderScannerWatchdog"; Script = "run_daytrader_scanner.ps1";            Desc = "At-logon launch of run_daytrader_scanner.ps1, which runs daytrader_scanner.py (Day Trader's own independent IBKR-connected scan loop) with auto-restart on crash." }
    @{ Name = "IBKR-AshleyMonitorWatchdog";    Script = "run_ashleyklieu_monitor.ps1";          Desc = "At-logon launch of run_ashleyklieu_monitor.ps1, which runs ashleyklieu_alert_monitor.py (Telegram signal monitor, alert-only) with auto-restart on crash." }
    @{ Name = "IBKR-AshleyExecutorWatchdog";   Script = "run_ashleyklieu_trigger_executor.ps1"; Desc = "At-logon launch of run_ashleyklieu_trigger_executor.ps1, which runs ashleyklieu_trigger_executor.py (price-zone-triggered entry/exit execution on Ashley's signals) with auto-restart on crash." }
    @{ Name = "IBKR-BreakoutScannerWatchdog";  Script = "run_scanner.ps1";                      Desc = "At-logon launch of run_scanner.ps1, which runs breakout_scanner.py (standalone breakout-alert scanner) with auto-restart on crash." }
    @{ Name = "IBKR-SectorCatalystScannerWatchdog"; Script = "run_sector_catalyst_scanner.ps1"; Desc = "At-logon launch of run_sector_catalyst_scanner.ps1, which runs sector_catalyst_scanner.py (Memory/Storage laggard-catchup alert scanner, validated 2026-09-08 -- alert-only, no order placement) with auto-restart on crash." }
)

foreach ($p in $alwaysOn) {
    $scriptPath = Join-Path $backendDir $p.Script
    Write-Host "Creating $($p.Name) (at logon)..."
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`"" `
        -WorkingDirectory $backendDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
    Register-ScheduledTask -TaskName $p.Name -Action $action -Trigger $trigger -Settings $alwaysOnSettings `
        -Description $p.Desc -Force | Out-Null
}

Write-Host ""
Write-Host "Done. Current state:"
Get-ScheduledTask -TaskName "IBKR-*" | Select-Object TaskName, State | Format-Table -AutoSize
