from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
_SERVICE_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_ANTHROPIC_KEY_NAME = "ANTHROPIC" + "_API_KEY"
_TEST_VALUE = "fixture-value-0123456789"
_STALE_VALUE = "stale-fixture-value-9876543210"


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _anthropic_assignment(value: str) -> str:
    return f"{_ANTHROPIC_KEY_NAME}={value}\n"


def _write_workspace(
    tmp_path: Path,
    *,
    profile: str = "functional-uat-3law",
    secret_contents: str | None,
) -> Path:
    workspace = tmp_path / "workspace"
    source_root = Path(__file__).resolve().parents[1]
    agent_tools = workspace / "customs-ai-agent" / "tools"
    backend_tools = workspace / "customs-backend" / "tools"
    agent_tools.mkdir(parents=True)
    backend_tools.mkdir(parents=True)
    shutil.copy2(source_root / "tools" / "Import-AgentProfile.ps1", agent_tools)

    (backend_tools / "Import-LocalProfile.ps1").write_text(
        """param([string]$Profile, [string]$Scope)
if ($Scope -ne 'agent') { throw 'unexpected_scope' }
if ($Profile -eq 'functional-uat') {
    $env:CUSTOMS_AGENT_SERVICE_MODE = 'functional-uat'
} elseif ($Profile -eq 'dev-fixture') {
    $env:CUSTOMS_AGENT_SERVICE_MODE = 'local-development'
} else {
    throw 'unexpected_profile'
}
$env:CUSTOMS_AGENT_SERVICE_BEARER_TOKEN = 'Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK'
""",
        encoding="utf-8",
    )

    backend_profile = "functional-uat" if profile == "functional-uat-3law" else "dev-fixture"
    (workspace / "_local" / "profiles" / backend_profile).mkdir(parents=True)
    (workspace / "_local" / "profiles" / backend_profile / "backend.env").write_text(
        "placeholder\n", encoding="utf-8"
    )
    evidence_file = workspace / "_local" / "profiles" / profile / "evidence-service-token"
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_text(_SERVICE_TOKEN, encoding="utf-8")

    if secret_contents is not None:
        secret_file = workspace / "_local" / "secrets" / ".env"
        secret_file.parent.mkdir(parents=True)
        secret_file.write_text(secret_contents, encoding="utf-8")
    return workspace


def _run_importer(
    workspace: Path,
    *,
    profile: str = "functional-uat-3law",
    inherited_key: str | None = None,
    expected_key: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if _POWERSHELL is None:
        pytest.skip("PowerShell is required to test the Agent profile importer")

    importer = workspace / "customs-ai-agent" / "tools" / "Import-AgentProfile.ps1"
    expected = "KEY_ABSENT" if expected_key is None else "KEY_PRESENT"
    expected_check = (
        "[string]::IsNullOrEmpty($env:ANTHROPIC_API_KEY)"
        if expected_key is None
        else f"$env:ANTHROPIC_API_KEY -eq {_powershell_literal(expected_key)}"
    )
    command = (
        "& { $ErrorActionPreference = 'Stop'; "
        f". {_powershell_literal(str(importer))} -Profile {_powershell_literal(profile)}; "
        f"if ({expected_check}) {{ [Console]::Out.Write({_powershell_literal(expected)}) }} "
        "else { [Console]::Out.Write('UNEXPECTED_KEY_STATE') } }"
    )
    environment = os.environ.copy()
    environment.pop("ANTHROPIC_API_KEY", None)
    if inherited_key is not None:
        environment["ANTHROPIC_API_KEY"] = inherited_key
    return subprocess.run(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_functional_uat_imports_only_the_canonical_local_key(tmp_path: Path) -> None:
    workspace = _write_workspace(
        tmp_path,
        secret_contents="# personal local key\n" + _anthropic_assignment(_TEST_VALUE),
    )
    result = _run_importer(workspace, inherited_key=_STALE_VALUE, expected_key=_TEST_VALUE)

    assert result.returncode == 0
    assert result.stdout == "KEY_PRESENT"
    assert _TEST_VALUE not in result.stdout + result.stderr
    assert _STALE_VALUE not in result.stdout + result.stderr


def test_functional_uat_allows_a_valid_inherited_key_only_when_file_is_absent(
    tmp_path: Path,
) -> None:
    workspace = _write_workspace(tmp_path, secret_contents=None)
    result = _run_importer(workspace, inherited_key=_TEST_VALUE, expected_key=_TEST_VALUE)

    assert result.returncode == 0
    assert result.stdout == "KEY_PRESENT"
    assert _TEST_VALUE not in result.stdout + result.stderr


def test_functional_uat_requires_a_file_or_process_key(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, secret_contents=None)
    result = _run_importer(workspace)

    assert result.returncode != 0
    assert "agent_profile_input_missing:anthropic-api-key" in result.stdout + result.stderr


def test_dev_fixture_does_not_read_the_functional_uat_key_file(tmp_path: Path) -> None:
    workspace = _write_workspace(
        tmp_path,
        profile="dev-fixture",
        secret_contents=f"UNRELATED_KEY={_TEST_VALUE}\n",
    )
    result = _run_importer(workspace, profile="dev-fixture", expected_key=None)

    assert result.returncode == 0
    assert result.stdout == "KEY_ABSENT"
    assert _TEST_VALUE not in result.stdout + result.stderr


def test_functional_uat_rejects_a_non_regular_canonical_secret_path(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path, secret_contents=None)
    (workspace / "_local" / "secrets" / ".env").mkdir(parents=True)
    result = _run_importer(workspace, inherited_key=_TEST_VALUE)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "agent_profile_anthropic_secret_invalid" in output
    assert _TEST_VALUE not in output


@pytest.mark.parametrize(
    ("secret_contents", "expected_code"),
    [
        (f"UNRELATED_KEY={_TEST_VALUE}\n", "agent_profile_anthropic_secret_invalid"),
        (
            _anthropic_assignment(_TEST_VALUE) + _anthropic_assignment(_STALE_VALUE),
            "agent_profile_anthropic_secret_invalid",
        ),
        (_anthropic_assignment(""), "agent_profile_anthropic_api_key_invalid"),
        (_anthropic_assignment("contains a space"), "agent_profile_anthropic_api_key_invalid"),
    ],
)
def test_functional_uat_rejects_invalid_local_secret_file_without_echoing_it(
    tmp_path: Path,
    secret_contents: str,
    expected_code: str,
) -> None:
    workspace = _write_workspace(tmp_path, secret_contents=secret_contents)
    result = _run_importer(workspace)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert expected_code in output
    assert _TEST_VALUE not in output
    assert _STALE_VALUE not in output
