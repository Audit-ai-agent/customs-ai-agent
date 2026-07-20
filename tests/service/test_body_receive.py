from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import read_bounded_json_body
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings
from tests.service.conftest import settings_with


def test_slow_incomplete_body_hits_receive_deadline(
    service_settings: ServiceSettings,
) -> None:
    settings = settings_with(
        service_settings,
        body_read_timeout_seconds=0.1,
    )
    receive_calls = 0

    async def receive() -> dict[str, object]:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {
                "type": "http.request",
                "body": b'{"case_id":"partial',
                "more_body": True,
            }
        await asyncio.Event().wait()
        raise AssertionError

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/internal/agent/v1/reviews",
            "headers": [(b"content-type", b"application/json")],
        },
        receive=receive,
    )

    async def exercise() -> None:
        with pytest.raises(ServiceHttpError) as captured:
            await read_bounded_json_body(request, settings)
        assert captured.value.code == "request_body_timeout"
        assert captured.value.status_code == 408
        assert captured.value.retryable is True

    asyncio.run(exercise())
    assert receive_calls == 2


def test_body_receive_timeout_setting_is_bounded(
    service_settings: ServiceSettings,
) -> None:
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_body_read_timeout_seconds_invalid",
    ):
        settings_with(service_settings, body_read_timeout_seconds=0)
