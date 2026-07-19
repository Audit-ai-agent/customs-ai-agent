from __future__ import annotations

import asyncio
import json
from typing import Any

from customs_ai_agent.service.controller import AgentServiceController
from tests.service.conftest import FakeRuntime, settings_with


def _payload(run_id: str) -> bytes:
    return json.dumps({"review_run_id": run_id}).encode()


def test_controller_single_flight_and_lifecycle(service_settings: Any) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime(delay_seconds=0.1)
        controller = AgentServiceController(
            settings=settings_with(service_settings, max_concurrency=2),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        first, second = await asyncio.gather(
            controller.submit(
                payload=_payload("run-1"),
                review_run_id="run-1",
                request_hash="a" * 64,
            ),
            controller.submit(
                payload=_payload("run-1"),
                review_run_id="run-1",
                request_hash="a" * 64,
            ),
        )
        assert first[0].status_code == second[0].status_code == 200
        assert sorted((first[1], second[1])) == [False, True]
        assert runtime.calls == 1
        await controller.shutdown()
        await controller.shutdown()
        assert runtime.close_calls == 1

    asyncio.run(scenario())


def test_controller_rejects_queue_overflow_without_extra_provider_call(
    service_settings: Any,
) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime(delay_seconds=0.2)
        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                max_concurrency=1,
                queue_wait_seconds=0.01,
            ),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        first_task = asyncio.create_task(
            controller.submit(
                payload=_payload("run-1"),
                review_run_id="run-1",
                request_hash="a" * 64,
            )
        )
        await asyncio.sleep(0.02)
        second, _ = await controller.submit(
            payload=_payload("run-2"),
            review_run_id="run-2",
            request_hash="b" * 64,
        )
        first, _ = await first_task
        assert first.status_code == 200
        assert second.status_code == 503
        assert second.error_code == "service_busy"
        assert runtime.calls == 1
        await controller.shutdown()

    asyncio.run(scenario())


def test_readiness_is_cached_and_shutdown_turns_it_off(service_settings: Any) -> None:
    async def scenario() -> None:
        calls = 0

        def ready() -> object:
            nonlocal calls
            calls += 1
            return object()

        controller = AgentServiceController(
            settings=service_settings,
            runtime=FakeRuntime(),
            readiness_probe=ready,
            logger=__import__("logging").getLogger("test-service"),
        )
        assert await controller.ready() is True
        assert await controller.ready() is True
        assert calls == 1
        await controller.shutdown()
        assert await controller.ready() is False

    asyncio.run(scenario())
