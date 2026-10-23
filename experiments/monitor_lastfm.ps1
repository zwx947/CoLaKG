param(
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][int]$TrainingProcessId
)
$ErrorActionPreference = 'Stop'
$run = (Resolve-Path -LiteralPath $RunDirectory).Path
if ((Split-Path -Parent $run) -ne 'D:\CoLaKGExperiments' -or
    (Split-Path -Leaf $run) -notlike 'lastfm-baseline-*') {
    throw 'Only an existing formal CoLaKG run can be monitored.'
}
$training = Get-Process -Id $TrainingProcessId
if ($training.Path -ne 'D:\CondaEnvs\CoLaKG\python.exe') {
    throw 'Unexpected training executable.'
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CoLaKGExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
$log = Join-Path $run ('monitor-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.jsonl')
function Read-SharedJson([string]$path) {
    # Allow the training writer to atomically replace status while this reader is open.
    $sharing = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $stream = [System.IO.FileStream]::new($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $sharing)
    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) }
    finally { $reader.Dispose() }
}
try {
    # Process/thread-scoped idle-sleep prevention, not a power-plan change.
    if ([CoLaKGExecutionState]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Unable to request process-scoped idle-sleep prevention.'
    }
    while (-not $training.HasExited) {
        $state = Read-SharedJson (Join-Path $run 'status.json')
        if ($state.pid -ne $TrainingProcessId) { throw 'Run PID changed.' }
        $record = [ordered]@{
            at = (Get-Date -Format o)
            completed_epochs = $state.completed_epochs
            status = $state.status
            gpu_telemetry_scope = 'Whole GPU including desktop/other processes, not model-only allocation'
            gpu_csv = $null
            d_free_bytes = (Get-PSDrive D).Free
        }
        if ($smi) {
            $record.gpu_csv = (& $smi '--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw' '--format=csv,noheader,nounits' 2>&1 | Out-String).Trim()
        }
        $record | ConvertTo-Json -Compress | Add-Content -LiteralPath $log -Encoding UTF8
        Write-Output ('MONITOR epoch={0}/2000 status={1} at={2}' -f $state.completed_epochs,$state.status,$record.at)
        if ($state.status -in @('completed','failed')) { break }
        Start-Sleep -Seconds 30
        $training.Refresh()
    }
}
finally {
    [void][CoLaKGExecutionState]::SetThreadExecutionState([uint32]2147483648)
}
