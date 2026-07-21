from __future__ import annotations

from types import SimpleNamespace

import pytest

import customs_ai_agent.service.release_admission as admission


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        release_state="approved",
        kill_switch_active=False,
        auto_completion_enabled=True,
        agent_code_snapshot_sha256="0" * 64,
        model_id="claude-sonnet-5-20260722",
        model_revision="claude-sonnet-5-20260722",
        extraction_prompt_sha256="1" * 64,
        coordinator_prompt_sha256="2" * 64,
        narrative_prompt_sha256="3" * 64,
        chat_prompt_sha256="4" * 64,
        extraction_schema_sha256="5" * 64,
        coordinator_schema_sha256="6" * 64,
        narrative_schema_sha256="7" * 64,
        chat_schema_sha256="8" * 64,
        request_schema_sha256="9" * 64,
        result_schema_sha256="a" * 64,
        policy_id="policy/v1",
        tool_registry_id="tools/v1",
        profile_sha256="b" * 64,
        review_subject_sha256="c" * 64,
        evaluation_report_sha256="d" * 64,
        reviewer_trust_store_sha256="e" * 64,
        assertion_trust_store_sha256="1" * 64,
    )


def _runtime(profile: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        agent_code_snapshot_sha256=profile.agent_code_snapshot_sha256,
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        extraction_prompt_sha256=profile.extraction_prompt_sha256,
        coordinator_prompt_sha256=profile.coordinator_prompt_sha256,
        narrative_prompt_sha256=profile.narrative_prompt_sha256,
        chat_prompt_sha256=profile.chat_prompt_sha256,
        extraction_schema_sha256=profile.extraction_schema_sha256,
        coordinator_schema_sha256=profile.coordinator_schema_sha256,
        narrative_schema_sha256=profile.narrative_schema_sha256,
        chat_schema_sha256=profile.chat_schema_sha256,
        result_schema_sha256=profile.result_schema_sha256,
        policy_id=profile.policy_id,
        tool_registry_id=profile.tool_registry_id,
        release_profile_sha256=profile.profile_sha256,
        release_profile_subject_sha256=profile.review_subject_sha256,
        release_profile_evaluation_report_sha256=profile.evaluation_report_sha256,
        reviewer_trust_store_sha256=profile.reviewer_trust_store_sha256,
        assertion_trust_store_sha256=profile.assertion_trust_store_sha256,
        assertion_trust_store_state="active",
        evidence_runtime_v3_sha256="f" * 64,
        evidence_snapshot_id="snapshot-v3",
    )


def _admitted_dependencies(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    profile = _profile()
    base = SimpleNamespace(knowledge_snapshot_id="snapshot-v3")
    v3 = SimpleNamespace(base_runtime_release=base, runtime_release_v3_sha256="f" * 64)
    monkeypatch.setattr(admission, "load_released_evidence_runtime_identity", lambda: base)
    monkeypatch.setattr(admission, "load_released_evidence_runtime_identity_v3", lambda: v3)
    monkeypatch.setattr(admission, "load_release_profile", lambda: profile)
    monkeypatch.setattr(
        admission,
        "load_assertion_trust_store",
        lambda: SimpleNamespace(
            release_state="active",
            trust_store_sha256=profile.assertion_trust_store_sha256,
        ),
    )
    monkeypatch.setattr(admission, "schema_sha256", lambda *_args: profile.request_schema_sha256)
    monkeypatch.setattr(admission, "runtime_release_identity", lambda **_kwargs: _runtime(profile))
    return profile


def test_production_release_admission_binds_every_local_release_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _admitted_dependencies(monkeypatch)
    result = admission.assert_production_release_admission(
        configured_model_id=profile.model_revision
    )
    assert result.agent_code_snapshot_sha256 == profile.agent_code_snapshot_sha256


def test_production_release_admission_rejects_configured_model_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _admitted_dependencies(monkeypatch)
    with pytest.raises(
        admission.ProductionReleaseAdmissionError,
        match="claude_configured_model_identity_mismatch",
    ):
        admission.assert_production_release_admission(
            configured_model_id="claude-sonnet-5-20260723"
        )


def test_production_release_admission_rejects_any_code_policy_or_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _admitted_dependencies(monkeypatch)
    runtime = _runtime(profile)
    runtime.policy_id = "changed-policy/v2"
    monkeypatch.setattr(admission, "runtime_release_identity", lambda **_kwargs: runtime)
    with pytest.raises(
        admission.ProductionReleaseAdmissionError,
        match="agent_runtime_release_identity_mismatch",
    ):
        admission.assert_production_release_admission(configured_model_id=profile.model_revision)


def test_default_production_runtime_stops_before_secret_loading_when_admission_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.service.app as app_module

    monkeypatch.setattr(
        app_module,
        "assert_production_release_admission",
        lambda **_kwargs: (_ for _ in ()).throw(
            admission.ProductionReleaseAdmissionError("evidence_runtime_v3_release_unreleased")
        ),
    )
    monkeypatch.setattr(
        app_module.RuntimeConfig,
        "from_environment",
        lambda **_kwargs: pytest.fail("credentials must not load before release admission"),
    )
    settings = SimpleNamespace(
        evidence_base_url="http://127.0.0.1:8010",
        evidence_service_token=None,
        evidence_allowed_origins=frozenset(),
    )
    with pytest.raises(
        app_module.RuntimeConfigurationError,
        match="evidence_runtime_v3_release_unreleased",
    ):
        app_module._production_runtime(settings)


def test_default_production_runtime_constructs_only_after_exact_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import customs_ai_agent.service.app as app_module

    exact_model = "claude-sonnet-5-20260722"
    monkeypatch.setenv("CUSTOMS_AGENT_CLAUDE_MODEL", exact_model)
    monkeypatch.setattr(
        app_module,
        "assert_production_release_admission",
        lambda **kwargs: SimpleNamespace(model_id=kwargs["configured_model_id"]),
    )
    config = SimpleNamespace(
        claude_settings=SimpleNamespace(model=exact_model),
        verification_assertion_verifier=lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        app_module.RuntimeConfig,
        "from_environment",
        lambda **_kwargs: config,
    )
    runtime = object()
    monkeypatch.setattr(
        app_module, "AgentRuntime", lambda value: runtime if value is config else None
    )
    settings = SimpleNamespace(
        evidence_base_url="http://127.0.0.1:8010",
        evidence_service_token=None,
        evidence_allowed_origins=frozenset(),
    )
    assert app_module._production_runtime(settings) is runtime
