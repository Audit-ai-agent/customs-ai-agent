"""Fail-closed, output-sanitized wrappers for the repository Safety Gate.

The wrappers intentionally do not install tools. CI and local callers provide
the executable/environment, while this module verifies the exact top-level
version declared in ``security/toolchain.lock.toml`` before any scan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN_LOCK = REPO_ROOT / "security" / "toolchain.lock.toml"
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SecurityGateError(RuntimeError):
    """A Safety Gate prerequisite, tool invocation, or report was invalid."""


def _read_toolchain() -> dict[str, Any]:
    try:
        value = tomllib.loads(TOOLCHAIN_LOCK.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SecurityGateError("toolchain lock is unavailable or invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise SecurityGateError("toolchain lock schema is invalid")
    for section in ("uv", "gitleaks", "pip_audit", "ruff_sast"):
        if not isinstance(value.get(section), dict):
            raise SecurityGateError("toolchain lock is incomplete")
    expected_hash = value["gitleaks"].get("linux_x64_sha256")
    if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
        raise SecurityGateError("toolchain lock Gitleaks checksum is invalid")
    return value


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecurityGateError(f"{label} did not produce valid JSON") from exc


def _run(
    argv: list[str],
    *,
    cwd: Path = REPO_ROOT,
    timeout_seconds: int = 900,
) -> subprocess.CompletedProcess[bytes]:
    try:
        # The executable is either a resolved regular file or sys.executable,
        # and every argument is constructed locally without shell expansion.
        return subprocess.run(  # noqa: S603
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SecurityGateError("security tool could not complete") from exc


def _resolved_executable(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SecurityGateError(f"{label} executable is unavailable") from exc
    if not resolved.is_file():
        raise SecurityGateError(f"{label} executable is not a regular file")
    return resolved


def _verify_gitleaks_version(executable: Path, expected: str) -> None:
    completed = _run([str(executable), "version"], timeout_seconds=30)
    try:
        actual = completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SecurityGateError("Gitleaks version output is invalid") from exc
    if completed.returncode != 0 or actual != expected:
        raise SecurityGateError("Gitleaks version does not match the toolchain lock")


def _gitleaks_findings(raw_report: bytes) -> list[dict[str, Any]]:
    value = _decode_json(raw_report, label="Gitleaks")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SecurityGateError("Gitleaks report shape is invalid")
    findings = list(value)
    for finding in findings:
        # A report is never accepted if the scanner returned an unredacted
        # value. The wrapper prints only rule/path/line metadata below.
        if finding.get("Secret") not in (None, "", "REDACTED"):
            raise SecurityGateError("Gitleaks report was not fully redacted")
    return findings


def _safe_gitleaks_summary(findings: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for finding in findings:
        rule = finding.get("RuleID")
        path = finding.get("File")
        line = finding.get("StartLine")
        safe_rule = rule if isinstance(rule, str) and len(rule) <= 100 else "unknown-rule"
        safe_path = _repo_relative_display(path)
        safe_line = line if isinstance(line, int) and line >= 1 else 0
        rows.append(f"{safe_rule}: {safe_path}:{safe_line}")
    return rows


def run_secret_scan(*, gitleaks: Path, scope: str) -> None:
    toolchain = _read_toolchain()
    executable = _resolved_executable(gitleaks, label="Gitleaks")
    expected_version = str(toolchain["gitleaks"]["version"])
    _verify_gitleaks_version(executable, expected_version)
    scopes = ("history", "worktree") if scope == "all" else (scope,)
    for current_scope in scopes:
        report_path = (
            REPO_ROOT / "security" / f".gitleaks-report-{os.getpid()}-{current_scope}.json"
        )
        try:
            common = [
                "--config",
                str(GITLEAKS_CONFIG),
                "--no-banner",
                "--no-color",
                "--redact=100",
                "--log-level",
                "error",
                "--exit-code",
                "1",
                "--report-format",
                "json",
                "--report-path",
                str(report_path),
            ]
            if current_scope == "history":
                argv = [
                    str(executable),
                    "git",
                    str(REPO_ROOT),
                    "--log-opts=--all",
                    *common,
                ]
            elif current_scope == "worktree":
                argv = [str(executable), "dir", str(REPO_ROOT), *common]
            else:
                raise SecurityGateError("secret scan scope is invalid")
            completed = _run(argv)
            try:
                report = report_path.read_bytes()
            except OSError as exc:
                raise SecurityGateError("Gitleaks report is unavailable") from exc
            findings = _gitleaks_findings(report)
            if completed.returncode not in (0, 1):
                raise SecurityGateError("Gitleaks execution failed")
            if completed.returncode == 1 or findings:
                print(
                    f"Gitleaks {current_scope}: FAIL ({len(findings)} redacted finding(s))",
                    file=sys.stderr,
                )
                for row in _safe_gitleaks_summary(findings):
                    print(f"- {row}", file=sys.stderr)
                raise SecurityGateError("secret scan found a candidate")
            print(f"Gitleaks {current_scope}: PASS (version {expected_version})")
        finally:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:
                # The report is fully redacted by construction. A cleanup
                # failure still fails the Gate so stale output is never ignored.
                raise SecurityGateError("redacted Gitleaks report cleanup failed") from None


def _verify_python_module_version(
    module: str,
    expected: str,
    label: str,
    *,
    cwd: Path = REPO_ROOT,
    isolated_utf8: bool = False,
) -> None:
    argv = [sys.executable]
    if isolated_utf8:
        argv.extend(["-I", "-X", "utf8"])
    argv.extend(["-m", module, "--version"])
    completed = _run(argv, cwd=cwd, timeout_seconds=30)
    try:
        output = (completed.stdout + completed.stderr).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SecurityGateError(f"{label} version output is invalid") from exc
    if (
        completed.returncode != 0
        or re.search(rf"(?<![0-9.]){re.escape(expected)}(?![0-9.])", output) is None
    ):
        raise SecurityGateError(f"{label} version does not match the toolchain lock")


def _pip_audit_vulnerabilities(value: Any) -> list[tuple[str, str, str]]:
    dependencies: Any
    if isinstance(value, dict):
        dependencies = value.get("dependencies")
    else:
        dependencies = value
    if not isinstance(dependencies, list):
        raise SecurityGateError("pip-audit report shape is invalid")
    vulnerabilities: list[tuple[str, str, str]] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise SecurityGateError("pip-audit dependency entry is invalid")
        name = dependency.get("name")
        version = dependency.get("version")
        vulns = dependency.get("vulns")
        if not isinstance(vulns, list):
            raise SecurityGateError("pip-audit vulnerability entry is invalid")
        safe_name = name if isinstance(name, str) and len(name) <= 100 else "unknown-package"
        safe_version = (
            version if isinstance(version, str) and len(version) <= 100 else "unknown-version"
        )
        for vulnerability in vulns:
            if not isinstance(vulnerability, dict):
                raise SecurityGateError("pip-audit vulnerability entry is invalid")
            identifier = vulnerability.get("id")
            safe_identifier = (
                identifier
                if isinstance(identifier, str) and len(identifier) <= 100
                else "unknown-advisory"
            )
            vulnerabilities.append((safe_name, safe_version, safe_identifier))
    return vulnerabilities


def run_dependency_audit(*, requirements: Path) -> None:
    toolchain = _read_toolchain()
    expected_version = str(toolchain["pip_audit"]["version"])
    service = str(toolchain["pip_audit"]["vulnerability_service"])
    resolved_requirements = _resolved_executable(requirements, label="requirements export")
    # Keep pip-audit and any Python subprocess it starts away from the source
    # checkout. On Windows, a checkout path that is not representable by the
    # active legacy code page can make Python fail while importing ``site``
    # before pip-audit emits JSON. An empty private temporary directory also
    # prevents project-local files from influencing the audit environment.
    with tempfile.TemporaryDirectory(prefix="customs-ai-agent-pip-audit-") as audit_dir:
        audit_cwd = Path(audit_dir)
        _verify_python_module_version(
            "pip_audit",
            expected_version,
            "pip-audit",
            cwd=audit_cwd,
            isolated_utf8=True,
        )
        completed = _run(
            [
                sys.executable,
                "-I",
                "-X",
                "utf8",
                "-m",
                "pip_audit",
                "--strict",
                "--disable-pip",
                "--require-hashes",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--vulnerability-service",
                service,
                "--requirement",
                str(resolved_requirements),
            ],
            cwd=audit_cwd,
        )
    report = _decode_json(completed.stdout, label="pip-audit")
    vulnerabilities = _pip_audit_vulnerabilities(report)
    if completed.returncode not in (0, 1):
        # Network/advisory-service errors are infrastructure failures, never PASS.
        raise SecurityGateError("pip-audit could not complete the advisory query")
    if completed.returncode == 1 or vulnerabilities:
        print(
            f"pip-audit: FAIL ({len(vulnerabilities)} vulnerability record(s))",
            file=sys.stderr,
        )
        for package, version, identifier in vulnerabilities:
            print(f"- {package} {version}: {identifier}", file=sys.stderr)
        raise SecurityGateError("dependency audit found a vulnerability")
    print(f"pip-audit: PASS (version {expected_version}, service {service})")


def _ruff_findings(value: Any) -> list[tuple[str, int, str]]:
    if not isinstance(value, list):
        raise SecurityGateError("Ruff SAST report shape is invalid")
    findings: list[tuple[str, int, str]] = []
    for finding in value:
        if not isinstance(finding, dict):
            raise SecurityGateError("Ruff SAST finding is invalid")
        filename = finding.get("filename")
        code = finding.get("code")
        location = finding.get("location")
        row = location.get("row") if isinstance(location, dict) else None
        safe_path = _repo_relative_display(filename)
        safe_code = code if isinstance(code, str) and len(code) <= 20 else "unknown-rule"
        safe_row = row if isinstance(row, int) and row >= 1 else 0
        findings.append((safe_path, safe_row, safe_code))
    return findings


def _repo_relative_display(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        return "unknown-path"
    candidate = Path(value)
    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        normalized = Path(value.replace("\\", "/"))
        if ".." in normalized.parts:
            return "unknown-path"
        return normalized.as_posix()
    except (OSError, ValueError):
        return "unknown-path"


def run_sast() -> None:
    toolchain = _read_toolchain()
    expected_version = str(toolchain["ruff_sast"]["version"])
    _verify_python_module_version("ruff", expected_version, "Ruff")
    completed = _run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src/customs_ai_agent",
            "tools",
            "evals",
            "--select",
            str(toolchain["ruff_sast"]["rule_family"]),
            "--output-format",
            "json",
        ]
    )
    findings = _ruff_findings(_decode_json(completed.stdout, label="Ruff SAST"))
    if completed.returncode not in (0, 1):
        raise SecurityGateError("Ruff SAST could not complete")
    if completed.returncode == 1 or findings:
        print(f"Ruff SAST: FAIL ({len(findings)} finding(s))", file=sys.stderr)
        for filename, row, code in findings:
            print(f"- {code}: {filename}:{row}", file=sys.stderr)
        raise SecurityGateError("Python SAST found an unsuppressed finding")
    print(f"Ruff SAST: PASS (version {expected_version}, rule family S)")


def verify_toolchain_lock() -> None:
    toolchain = _read_toolchain()
    try:
        uv_lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SecurityGateError("uv.lock is unavailable or invalid") from exc
    packages = uv_lock.get("package")
    if not isinstance(packages, list):
        raise SecurityGateError("uv.lock package inventory is invalid")
    ruff_versions = {
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == "ruff"
    }
    if ruff_versions != {toolchain["ruff_sast"]["version"]}:
        raise SecurityGateError("Ruff SAST version does not match uv.lock")
    print("security toolchain lock: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="validate the security tool lock against uv.lock")
    secrets = subparsers.add_parser("secrets", help="run redacted Gitleaks scans")
    secrets.add_argument("--gitleaks", type=Path, required=True)
    secrets.add_argument(
        "--scope",
        choices=("history", "worktree", "all"),
        default="history",
    )
    dependencies = subparsers.add_parser(
        "dependencies",
        help="audit one hash-locked requirements export",
    )
    dependencies.add_argument("--requirements", type=Path, required=True)
    subparsers.add_parser("sast", help="run Ruff's Python security rule family")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            verify_toolchain_lock()
        elif args.command == "secrets":
            run_secret_scan(gitleaks=args.gitleaks, scope=args.scope)
        elif args.command == "dependencies":
            run_dependency_audit(requirements=args.requirements)
        elif args.command == "sast":
            run_sast()
        else:
            raise SecurityGateError("unknown security gate command")
    except SecurityGateError as exc:
        print(f"security gate failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Tool output and tracebacks can contain paths or candidate secret data.
        print("security gate failed: internal_error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
