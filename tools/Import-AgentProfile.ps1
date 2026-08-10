<#
.SYNOPSIS
Loads only the local values needed by the Agent into the current PowerShell session.

.DESCRIPTION
Dot-source this script.  It invokes Backend's agent-scoped local importer for
the Agent bearer token and mode, then reads Data tokens for document Evidence
and, only in the composed Korean-trade profile, separate general guidance. It
never writes or prints a secret.  For functional UAT only, it also reads exactly
one personal Claude key from the ignored workspace-local ``_local\secrets\.env``
file.  The value is placed only in the current PowerShell process; the legacy
``_local\secrets\uat\agent.env`` file is never read.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet(
        "dev-fixture",
        "functional-uat-3law",
        "functional-uat-law-history",
        "functional-uat-kr-trade",
        "functional-uat-kr-trade-v2",
        "functional-uat-kr-trade-v2-law-history"
    )]
    [string]$Profile,

    [ValidateRange(1, 65535)]
    [int]$EvidencePort = 18083,

    # Zero selects the profile's immutable default: legacy V1 guidance uses
    # 18093 and the separately verified V2 guidance peer uses 18094.
    [ValidateRange(0, 65535)]
    [int]$GuidancePort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-FunctionalUatAnthropicKey {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$SecretFile
    )

    $secretItem = Get-Item -LiteralPath $SecretFile -Force -ErrorAction SilentlyContinue
    if ($null -eq $secretItem) {
        # Keep an explicit process-environment injection working for an
        # operator who deliberately chooses not to create the local file.
        $inheritedKey = $env:ANTHROPIC_API_KEY
        if (
            [string]::IsNullOrWhiteSpace($inheritedKey) -or
            $inheritedKey.Length -lt 16 -or
            $inheritedKey.Length -gt 512 -or
            $inheritedKey -match "\s"
        ) {
            throw "agent_profile_input_missing:anthropic-api-key"
        }
        return
    }

    if (
        $secretItem.PSIsContainer -or
        (($secretItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) -or
        $secretItem.Length -gt 8192
    ) {
        throw "agent_profile_anthropic_secret_invalid"
    }

    try {
        $lines = Get-Content -LiteralPath $SecretFile -Encoding utf8 -ErrorAction Stop
    }
    catch {
        throw "agent_profile_anthropic_secret_invalid"
    }

    $apiKey = $null
    foreach ($rawLine in $lines) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        if ($line -notmatch "^(?<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?<value>.*)$") {
            throw "agent_profile_anthropic_secret_invalid"
        }
        # This is a shared local secret file.  Validate and import only the
        # Claude key; never project unrelated values into the Agent process.
        if ($Matches["key"] -cne "ANTHROPIC_API_KEY") {
            continue
        }
        if ($null -ne $apiKey) {
            throw "agent_profile_anthropic_secret_invalid"
        }

        $candidate = $Matches["value"].Trim()
        if (
            [string]::IsNullOrWhiteSpace($candidate) -or
            $candidate.Length -lt 16 -or
            $candidate.Length -gt 512 -or
            $candidate -match "\s"
        ) {
            throw "agent_profile_anthropic_api_key_invalid"
        }
        $apiKey = $candidate
    }

    if ($null -eq $apiKey) {
        throw "agent_profile_anthropic_api_key_invalid"
    }

    # This script is dot-sourced, so the key exists only for this Agent shell
    # and its child process.  It is never persisted or printed.
    Set-Item -LiteralPath "Env:ANTHROPIC_API_KEY" -Value $apiKey
}

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
        $guidanceProfile = $null
        $expectedMode = "functional-uat"
    }
    "functional-uat-law-history" {
        # The 65-law temporal Evidence profile is intentionally distinct from
        # the preserved three-law profile.  Data owns collection, history and
        # the temporal response context; this importer only binds its local
        # loopback token to the Agent process.
        $backendProfile = "functional-uat"
        $evidenceProfile = "functional-uat-law-history"
        $guidanceProfile = $null
        $expectedMode = "functional-uat"
    }
    "functional-uat-kr-trade" {
        # The broader official guidance snapshot deliberately complements, not
        # replaces, the verified three-law Evidence store used by document review.
        $backendProfile = "functional-uat"
        $evidenceProfile = "functional-uat-3law"
        $guidanceProfile = "functional-uat-kr-trade"
        $guidanceRegistryVersion = "1.0"
        $defaultGuidancePort = 18093
        $expectedMode = "functional-uat"
    }
    "functional-uat-kr-trade-v2" {
        # V2 is a distinct, readiness-gated Data peer.  Keeping this profile
        # explicit prevents a scope-specific V2 request from silently using
        # the preserved V1 guidance endpoint.  Data deliberately keeps the
        # local token/state below its existing kr-trade profile; only the
        # Agent's runtime selection and peer port are new.
        $backendProfile = "functional-uat"
        $evidenceProfile = "functional-uat-3law"
        $guidanceProfile = "functional-uat-kr-trade"
        $guidanceRegistryVersion = "2.0"
        $defaultGuidancePort = 18094
        $expectedMode = "functional-uat"
    }
    "functional-uat-kr-trade-v2-law-history" {
        # This is the new composed Functional-UAT default: time-aware legal
        # Evidence plus the separately verified V2 official-guidance peer.
        # It deliberately shares only the V2 guidance token/state; the
        # Evidence token comes from the independent law-history profile.
        $backendProfile = "functional-uat"
        $evidenceProfile = "functional-uat-law-history"
        $guidanceProfile = "functional-uat-kr-trade"
        $guidanceRegistryVersion = "2.0"
        $defaultGuidancePort = 18094
        $expectedMode = "functional-uat"
    }
}
$requiresFunctionalUatKey = $Profile -ne "dev-fixture"

if ($Profile -eq "dev-fixture") {
    $guidanceProfile = $null
}
if ($null -ne $guidanceProfile -and $GuidancePort -eq 0) {
    $GuidancePort = $defaultGuidancePort
}
if ($null -ne $guidanceProfile -and $GuidancePort -eq $EvidencePort) {
    throw "agent_profile_guidance_port_conflict"
}

$evidenceTokenFile = Join-Path (Join-Path $localProfilesRoot $evidenceProfile) "evidence-service-token"
$guidanceTokenFile = if ($null -ne $guidanceProfile) {
    Join-Path (Join-Path $localProfilesRoot $guidanceProfile) "evidence-service-token"
} else {
    $null
}
$backendImporter = Join-Path $workspaceRoot "customs-backend\tools\Import-LocalProfile.ps1"
$backendEnvironment = Join-Path (Join-Path $localProfilesRoot $backendProfile) "backend.env"

if (-not (Test-Path -LiteralPath $backendEnvironment -PathType Leaf)) {
    throw "agent_profile_input_missing:backend.env"
}
if (-not (Test-Path -LiteralPath $evidenceTokenFile -PathType Leaf)) {
    throw "agent_profile_input_missing:evidence-service-token"
}
if ($null -ne $guidanceTokenFile -and -not (Test-Path -LiteralPath $guidanceTokenFile -PathType Leaf)) {
    throw "agent_profile_input_missing:guidance-service-token"
}
if (-not (Test-Path -LiteralPath $backendImporter -PathType Leaf)) {
    throw "agent_profile_input_missing:Import-LocalProfile.ps1"
}

$evidenceToken = [System.IO.File]::ReadAllText($evidenceTokenFile).Trim()
if ($evidenceToken -notmatch "^[A-Za-z0-9\-._~+/]{43,504}={0,8}$") {
    throw "agent_profile_evidence_token_invalid"
}
$guidanceToken = $null
if ($null -ne $guidanceTokenFile) {
    $guidanceToken = [System.IO.File]::ReadAllText($guidanceTokenFile).Trim()
    if ($guidanceToken -notmatch "^[A-Za-z0-9\-._~+/]{43,504}={0,8}$") {
        throw "agent_profile_guidance_token_invalid"
    }
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

if ($requiresFunctionalUatKey) {
    $anthropicKeyFile = Join-Path $workspaceRoot "_local\secrets\.env"
    Import-FunctionalUatAnthropicKey -SecretFile $anthropicKeyFile
}

$env:CUSTOMS_AGENT_PORT = "18090"
$env:EVIDENCE_SERVICE_BASE_URL = "http://127.0.0.1:$EvidencePort"
$env:EVIDENCE_SERVICE_TOKEN = $evidenceToken
$env:EVIDENCE_SERVICE_TOKEN_HEADER = "Authorization"
# Keep the Data-owned profile identity alongside its local token.  It is not a
# credential and lets the runtime require law-history temporal provenance
# without weakening nullable compatibility for the retained three-law profile.
$env:CUSTOMS_AGENT_EVIDENCE_PROFILE = $evidenceProfile
if ($null -eq $guidanceProfile) {
    # A shell can switch from the composed profile back to an existing one.
    # Remove stale overrides so its bounded guidance query falls back to the
    # same Evidence peer exactly as it did before this profile existed.
    Remove-Item -LiteralPath "Env:GUIDANCE_SERVICE_BASE_URL" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Env:GUIDANCE_SERVICE_TOKEN" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Env:GUIDANCE_SERVICE_TOKEN_HEADER" -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "Env:GUIDANCE_SERVICE_REGISTRY_VERSION" -ErrorAction SilentlyContinue
} else {
    $env:GUIDANCE_SERVICE_BASE_URL = "http://127.0.0.1:$GuidancePort"
    $env:GUIDANCE_SERVICE_TOKEN = $guidanceToken
    $env:GUIDANCE_SERVICE_TOKEN_HEADER = "Authorization"
    $env:GUIDANCE_SERVICE_REGISTRY_VERSION = $guidanceRegistryVersion
}
