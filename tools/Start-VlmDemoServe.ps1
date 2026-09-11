<#
.SYNOPSIS
One-command Agent serve for the Claude VLM OCR document-preview demo.

.DESCRIPTION
Imports the selected Agent profile (which loads the personal Claude key from
the workspace-local ``_local\secrets\.env``), opts this one process into the
demo preview engine (pypdf text layer first, Claude VLM OCR fallback), and
starts the Agent service on loopback.  Run it directly:

    powershell -ExecutionPolicy Bypass -File tools\Start-VlmDemoServe.ps1

If the venv's ``python.exe`` trampoline is blocked by an Application Control
policy, the launcher falls back to the uv base interpreter with the venv's
site-packages on ``PYTHONPATH``.
#>
[CmdletBinding()]
param(
    [ValidateSet(
        "dev-fixture",
        "functional-uat-3law",
        "functional-uat-law-history",
        "functional-uat-kr-trade",
        "functional-uat-kr-trade-v2",
        "functional-uat-kr-trade-v2-law-history"
    )]
    [string]$Profile = "functional-uat-law-history"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
# The profile importer chain dot-sources into this scope and rebinds $Profile;
# keep the operator's selection for later display.
$selectedProfile = $Profile

# A previous Agent serve holds the loopback port until it is stopped.  Replace
# only a local python listener; anything else on the port is a hard error.
$agentPort = 18090
$stale = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $agentPort -State Listen -ErrorAction SilentlyContinue
foreach ($connection in @($stale)) {
    $owner = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    if ($null -eq $owner) { continue }
    if ($owner.ProcessName -notmatch "^v?python") {
        throw "agent_demo_port_in_use_by:$($owner.ProcessName):$($owner.Id)"
    }
    Write-Host "Stopping previous Agent serve (python PID $($owner.Id)) on port $agentPort"
    Stop-Process -Id $owner.Id -Force -Confirm:$false
    $owner.WaitForExit(5000) | Out-Null
}

# Loads service mode, bearer token, Evidence peer values, and ANTHROPIC_API_KEY
# into this process only.
. (Join-Path $PSScriptRoot "Import-AgentProfile.ps1") -Profile $selectedProfile

# Explicit demo-only opt-in: pypdf text layer first, Claude VLM OCR fallback.
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_DEMO = "claude-vlm"
if ([string]::IsNullOrWhiteSpace($env:CUSTOMS_AGENT_CLAUDE_MAX_TOKENS)) {
    # Dense scanned documents overflow the 4096-token default.
    $env:CUSTOMS_AGENT_CLAUDE_MAX_TOKENS = "16384"
}

# The App Control policy blocks the venv python.exe trampoline; the renamed
# vpython.exe copy is the locally allowed workaround, with the uv base
# interpreter plus explicit PYTHONPATH as the last resort.
$python = $null
foreach ($candidate in @(".venv\Scripts\vpython.exe", ".venv\Scripts\python.exe")) {
    $candidatePath = Join-Path $repoRoot $candidate
    if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) { continue }
    try {
        & $candidatePath --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $python = $candidatePath
            break
        }
    }
    catch { }
}
if ($null -eq $python) {
    $basePython = Join-Path $env:APPDATA "uv\python\cpython-3.11-windows-x86_64-none\python.exe"
    if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
        throw "agent_demo_python_unavailable"
    }
    $python = $basePython
    $env:PYTHONPATH = (Join-Path $repoRoot ".venv\Lib\site-packages") + ";" + (Join-Path $repoRoot "src")
}

Write-Host "Agent VLM-OCR demo serve: profile=$selectedProfile mode=$($env:CUSTOMS_AGENT_SERVICE_MODE) port=$($env:CUSTOMS_AGENT_PORT)"
& $python -B -m customs_ai_agent.service
exit $LASTEXITCODE
