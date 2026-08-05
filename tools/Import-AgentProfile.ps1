<#
.SYNOPSIS
Loads only the local values needed by the Agent into the current PowerShell session.

.DESCRIPTION
Dot-source this script.  It invokes Backend's agent-scoped local importer for
the Agent bearer token and mode, then reads the Evidence bearer token generated
by Data.  It never writes or prints a secret.  The functional UAT Claude key is
intentionally not read here: the person running the UAT supplies that personal
key separately.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("dev-fixture", "functional-uat-3law")]
    [string]$Profile,

    [ValidateRange(1, 65535)]
    [int]$EvidencePort = 18083
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($MyInvocation.InvocationName -ne ".") {
    throw "agent_profile_dot_source_required"
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$localProfilesRoot = Join-Path $workspaceRoot "_local\profiles"

switch ($Profile) {
    "dev-fixture" {
        $backendProfile = "dev-fixture"
        $evidenceProfile = "dev-fixture"
        $expectedMode = "local-development"
    }
    "functional-uat-3law" {
        # Backend owns the service profile name; Data owns the three-law input profile name.
        $backendProfile = "functional-uat"
        $evidenceProfile = "functional-uat-3law"
        $expectedMode = "functional-uat"
    }
}

$evidenceTokenFile = Join-Path (Join-Path $localProfilesRoot $evidenceProfile) "evidence-service-token"
$backendImporter = Join-Path $workspaceRoot "customs-backend\tools\Import-LocalProfile.ps1"
$backendEnvironment = Join-Path (Join-Path $localProfilesRoot $backendProfile) "backend.env"

if (-not (Test-Path -LiteralPath $backendEnvironment -PathType Leaf)) {
    throw "agent_profile_input_missing:backend.env"
}
if (-not (Test-Path -LiteralPath $evidenceTokenFile -PathType Leaf)) {
    throw "agent_profile_input_missing:evidence-service-token"
}
if (-not (Test-Path -LiteralPath $backendImporter -PathType Leaf)) {
    throw "agent_profile_input_missing:Import-LocalProfile.ps1"
}

$evidenceToken = [System.IO.File]::ReadAllText($evidenceTokenFile).Trim()
if ($evidenceToken -notmatch "^[A-Za-z0-9\-._~+/]{43,504}={0,8}$") {
    throw "agent_profile_evidence_token_invalid"
}

# Scope=agent is intentional: Backend/DB/Compose credentials never enter this
# process.  Stream 6 contains Backend's benign status line and is suppressed so
# sourcing this helper never prints configuration values.
. $backendImporter -Profile $backendProfile -Scope agent 6>$null
if ($env:CUSTOMS_AGENT_SERVICE_MODE -cne $expectedMode) {
    throw "agent_profile_mode_mismatch"
}
if ($env:CUSTOMS_AGENT_SERVICE_BEARER_TOKEN -notmatch "^[\x21-\x7E]{43,512}$") {
    throw "agent_profile_service_token_invalid"
}

$env:CUSTOMS_AGENT_PORT = "8090"
$env:EVIDENCE_SERVICE_BASE_URL = "http://127.0.0.1:$EvidencePort"
$env:EVIDENCE_SERVICE_TOKEN = $evidenceToken
$env:EVIDENCE_SERVICE_TOKEN_HEADER = "Authorization"
