from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evals.run_release_holdout_validation as release_holdout  # noqa: E402


SCRIPT = ROOT / "tools" / "run_security_gate.py"
SPEC = importlib.util.spec_from_file_location("run_security_gate_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError("security gate module could not be loaded")
security_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = security_gate
SPEC.loader.exec_module(security_gate)


class SecurityGateTests(unittest.TestCase):
    def test_toolchain_lock_matches_uv_lock_and_security_gate_is_release_only(self) -> None:
        toolchain = tomllib.loads(
            (ROOT / "security" / "toolchain.lock.toml").read_text(encoding="utf-8")
        )
        uv_lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        ruff_versions = {
            package["version"] for package in uv_lock["package"] if package.get("name") == "ruff"
        }
        self.assertEqual(ruff_versions, {toolchain["ruff_sast"]["version"]})
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn(toolchain["gitleaks"]["linux_x64_url"], workflow)
        self.assertNotIn(toolchain["gitleaks"]["linux_x64_sha256"], workflow)
        self.assertNotIn(f"pip-audit=={toolchain['pip_audit']['version']}", workflow)
        self.assertNotIn("run_security_gate.py", workflow)
        self.assertTrue((ROOT / "tools" / "run_security_gate.py").is_file())
        self.assertTrue((ROOT / "security" / "toolchain.lock.toml").is_file())
        self.assertNotIn("security-events: write", workflow)
        self.assertNotIn("id-token: write", workflow)
        self.assertNotIn("sarif", workflow.casefold())

    def test_gitleaks_report_rejects_unredacted_secret(self) -> None:
        with self.assertRaisesRegex(
            security_gate.SecurityGateError,
            "not fully redacted",
        ):
            security_gate._gitleaks_findings(
                json.dumps(
                    [
                        {
                            "RuleID": "generic-api-key",
                            "File": "example.py",
                            "StartLine": 1,
                            "Secret": "must-not-be-returned",
                        }
                    ]
                ).encode()
            )

    def test_gitleaks_summary_never_contains_secret_value(self) -> None:
        secret = "must-not-be-returned"
        findings = security_gate._gitleaks_findings(
            json.dumps(
                [
                    {
                        "RuleID": "generic-api-key",
                        "File": "example.py",
                        "StartLine": 9,
                        "Secret": "REDACTED",
                    }
                ]
            ).encode()
        )
        summary = "\n".join(security_gate._safe_gitleaks_summary(findings))
        self.assertNotIn(secret, summary)
        expected_summary = "{}-{}-{}: example.py:9".format("generic", "api", "key")
        self.assertEqual(summary, expected_summary)

    def test_pip_audit_summary_exposes_only_package_version_and_id(self) -> None:
        report = {
            "dependencies": [
                {
                    "name": "demo",
                    "version": "1.0",
                    "vulns": [{"id": "PYSEC-0000", "description": "not returned"}],
                }
            ]
        }
        self.assertEqual(
            security_gate._pip_audit_vulnerabilities(report),
            [("demo", "1.0", "PYSEC-0000")],
        )

    def test_dependency_audit_runs_from_private_utf8_temporary_directory(self) -> None:
        report = json.dumps(
            {"dependencies": [{"name": "demo", "version": "1.0", "vulns": []}]}
        ).encode()
        version = CompletedProcess([], 0, stdout=b"pip-audit 2.9.0\n", stderr=b"")
        completed = CompletedProcess([], 0, stdout=report, stderr=b"")
        private_audit_cwd = ROOT / ".private-pip-audit-cwd"
        with (
            patch.object(
                security_gate,
                "_read_toolchain",
                return_value={
                    "pip_audit": {
                        "version": "2.9.0",
                        "vulnerability_service": "pypi",
                    }
                },
            ),
            patch.object(
                security_gate,
                "_resolved_executable",
                return_value=ROOT / "uv.lock",
            ),
            patch.object(security_gate.tempfile, "TemporaryDirectory") as temp_directory,
            patch.object(
                security_gate,
                "_run",
                side_effect=[version, completed],
            ) as run,
        ):
            temp_directory.return_value.__enter__.return_value = str(private_audit_cwd)
            security_gate.run_dependency_audit(requirements=ROOT / "uv.lock")

        version_argv = run.call_args_list[0].args[0]
        audit_argv = run.call_args_list[1].args[0]
        self.assertEqual(
            version_argv,
            [sys.executable, "-I", "-X", "utf8", "-m", "pip_audit", "--version"],
        )
        self.assertEqual(
            audit_argv[:6],
            [sys.executable, "-I", "-X", "utf8", "-m", "pip_audit"],
        )
        self.assertEqual(run.call_args_list[0].kwargs["cwd"], private_audit_cwd)
        self.assertEqual(run.call_args_list[1].kwargs["cwd"], private_audit_cwd)
        temp_directory.assert_called_once_with(prefix="customs-ai-agent-pip-audit-")

    def test_invalid_reports_fail_closed(self) -> None:
        with self.assertRaises(security_gate.SecurityGateError):
            security_gate._gitleaks_findings(b"{}")
        with self.assertRaises(security_gate.SecurityGateError):
            security_gate._pip_audit_vulnerabilities({"dependencies": "invalid"})
        with self.assertRaises(security_gate.SecurityGateError):
            security_gate._ruff_findings({"findings": []})

    def test_reported_paths_are_repository_relative(self) -> None:
        display = security_gate._repo_relative_display(
            str(ROOT / "src" / "customs_ai_agent" / "runner.py")
        )
        self.assertEqual(display, "src/customs_ai_agent/runner.py")
        self.assertNotIn(str(ROOT), display)

    def test_holdout_git_state_uses_resolved_executable_without_shell(self) -> None:
        responses = [
            CompletedProcess([], 0, stdout=str(ROOT), stderr=""),
            CompletedProcess([], 0, stdout="a" * 40, stderr=""),
            CompletedProcess([], 0, stdout="", stderr=""),
        ]
        with (
            patch.object(release_holdout.shutil, "which", return_value=sys.executable),
            patch.object(release_holdout.subprocess, "run", side_effect=responses) as run,
        ):
            commit, clean = release_holdout._git_state(ROOT)
        self.assertEqual(commit, "a" * 40)
        self.assertTrue(clean)
        for call in run.call_args_list:
            argv = call.args[0]
            self.assertTrue(Path(argv[0]).is_absolute())
            self.assertFalse(call.kwargs["shell"])

    def test_holdout_git_state_fails_closed_when_git_is_unavailable(self) -> None:
        with (
            patch.object(release_holdout.shutil, "which", return_value=None),
            self.assertRaisesRegex(
                release_holdout.ReleaseHoldoutError,
                "exact Git state cannot be established",
            ),
        ):
            release_holdout._git_state(ROOT)

    def test_vision_release_precondition_does_not_use_runtime_assert(self) -> None:
        source = (ROOT / "evals" / "run_claude_vision.py").read_text(encoding="utf-8")
        self.assertNotIn("assert ocr_provider is not None", source)
        self.assertIn("live_gate_internal_invariant_failed", source)


if __name__ == "__main__":
    unittest.main()
