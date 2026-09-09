# query_scheduled_tasks.ps1 -- called by main.py's /independent-traders/status
# endpoint (via subprocess, run_in_executor) to report real Task Scheduler
# state for the fire-on-schedule independent traders (Ashley, Butterflies
# x3, GOOG Condor, Safe Income) that have no HTTP server of their own to
# ask directly. Takes task names as ONE comma-joined string, not a
# [string[]] -- PowerShell's -File invocation does not bind multiple
# trailing bare arguments to an array parameter the way a normal function
# call does (confirmed live 2026-09-07: only the first name after
# -TaskNames bound, the rest raised "positional parameter not found").
#
# Calls Get-ScheduledTask ONCE (unfiltered, then filtered in-memory) rather
# than once per name in a -TaskName loop -- resolving a task by name is the
# expensive part (a fresh Task Scheduler service round-trip each time), and
# doing that twice per name (once for Get-ScheduledTask, once for
# Get-ScheduledTaskInfo -TaskName) across 6 independent traders measured at
# ~14.4s end-to-end (confirmed live 2026-09-07 via the frontend's
# Independent Traders tab sitting on "Loading..." far longer than its 30s
# poll interval should allow). Get-ScheduledTaskInfo is still called once
# per matched task below, but piped the already-resolved CIM object instead
# of a bare name -- that alone drops it to ~3.5s.
#
# IMPORTANT: Get-ScheduledTaskInfo is called inside the SAME foreach as the
# task it belongs to, not batched into a separate `$tasks | Get-ScheduledTaskInfo`
# pipeline paired back up by array index afterwards -- verified live
# (2026-09-07) that the two arrays came back in DIFFERENT orders (Ashley's
# row silently got IWM's LastRunTime, GOOG Condor's row got Safe Income's,
# etc.) despite both having the same count. Index-zipping two independently-
# ordered pipeline outputs is not a safe substitute for keeping Name and
# Info paired by direct object reference.
param(
    [Parameter(Mandatory=$true)]
    [string]$TaskNames
)

$names = $TaskNames -split ","
$tasks = Get-ScheduledTask | Where-Object { $names -contains $_.TaskName }

$results = foreach ($task in $tasks) {
    $info = $task | Get-ScheduledTaskInfo
    [PSCustomObject]@{
        Name           = $task.TaskName
        State          = $task.State.ToString()
        LastRunTime    = if ($info.LastRunTime)  { $info.LastRunTime.ToString("o") }  else { $null }
        NextRunTime    = if ($info.NextRunTime)  { $info.NextRunTime.ToString("o") }  else { $null }
        LastTaskResult = $info.LastTaskResult
    }
}

$results | ConvertTo-Json
