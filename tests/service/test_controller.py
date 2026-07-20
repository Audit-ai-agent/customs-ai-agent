from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from urllib.request import Request

import customs_ai_agent.adapters.retrieval as retrieval_module
from customs_ai_agent.service.controller import AgentServiceController
from tests.service.conftest import FakeResult, FakeRuntime, settings_with


def _payload(run_id: str) -> bytes:
    return json.dumps({"review_run_id": run_id}).encode()


class _SlowTransportResponse:
    def __init__(self, *, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.closed = threading.Event()

    def __enter__(self) -> _SlowTransportResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed.set()

    def read(self, _limit: int) -> bytes:
        if self.closed.wait(self.interval_seconds):
            return b""
        return b"x"


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


def test_run_timeout_keeps_capacity_consumed_until_worker_finishes(
    service_settings: Any,
) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime(delay_seconds=1.1)
        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                max_concurrency=1,
                queue_wait_seconds=0.01,
                run_timeout_seconds=1,
                shutdown_grace_seconds=2,
            ),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        timed_out, _ = await controller.submit(
            payload=_payload("run-timeout"),
            review_run_id="run-timeout",
            request_hash="c" * 64,
        )
        assert timed_out.status_code == 504
        assert timed_out.error_code == "review_timeout"
        assert controller.metrics.snapshot()["inflight"] == 1

        busy, _ = await controller.submit(
            payload=_payload("run-after-timeout"),
            review_run_id="run-after-timeout",
            request_hash="d" * 64,
        )
        assert busy.status_code == 503
        assert runtime.calls == 1
        await controller.shutdown()
        assert controller.metrics.snapshot()["inflight"] == 0
        assert runtime.close_calls == 1

    asyncio.run(scenario())


def test_absolute_evidence_deadline_releases_timed_out_review_slot(
    service_settings: Any,
    monkeypatch: Any,
) -> None:
    class EvidenceRuntime:
        def __init__(self) -> None:
            self.calls = 0
            self.close_calls = 0

        def review_json(self, _payload_bytes: bytes) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                retrieval_module._perform_http_exchange(
                    Request("http://127.0.0.1:8010/internal/rag/v2/retrieve"),
                    deadline=retrieval_module._new_deadline(1.1),
                )
            return FakeResult()

        def close(self) -> None:
            self.close_calls += 1

    slow_response = _SlowTransportResponse()
    monkeypatch.setattr(
        retrieval_module,
        "urlopen",
        lambda *_args, **_kwargs: slow_response,
    )

    async def scenario() -> None:
        runtime = EvidenceRuntime()
        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                max_concurrency=1,
                queue_wait_seconds=0.01,
                run_timeout_seconds=1,
                shutdown_grace_seconds=2,
            ),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        timed_out, _ = await controller.submit(
            payload=_payload("run-slow-evidence"),
            review_run_id="run-slow-evidence",
            request_hash="e" * 64,
        )
        assert timed_out.status_code == 504

        busy, _ = await controller.submit(
            payload=_payload("run-still-bounded"),
            review_run_id="run-still-bounded",
            request_hash="f" * 64,
        )
        assert busy.status_code == 503
        await asyncio.sleep(0.25)

        recovered, _ = await controller.submit(
            payload=_payload("run-recovered"),
            review_run_id="run-recovered",
            request_hash="1" * 64,
        )
        assert recovered.status_code == 200
        assert slow_response.closed.is_set()
        assert runtime.calls == 2
        await controller.shutdown()
        assert runtime.close_calls == 1

    asyncio.run(scenario())


def test_finite_transport_probe_does_not_poison_readiness_executor(
    service_settings: Any,
    monkeypatch: Any,
) -> None:
    def uncooperative_open(*_args: Any, **_kwargs: Any) -> _SlowTransportResponse:
        time.sleep(0.2)
        return _SlowTransportResponse()

    monkeypatch.setattr(retrieval_module, "urlopen", uncooperative_open)

    async def scenario() -> None:
        calls = 0

        def probe() -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                retrieval_module._perform_http_exchange(
                    Request("http://127.0.0.1:8010/internal/rag/v2/health/ready"),
                    deadline=retrieval_module._new_deadline(0.05),
                )
            return object()

        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                readiness_timeout_seconds=0.3,
                readiness_ttl_seconds=0.1,
            ),
            runtime=FakeRuntime(),
            readiness_probe=probe,
            logger=__import__("logging").getLogger("test-service"),
        )
        assert await controller.ready() is False
        await asyncio.sleep(0.11)
        assert await controller.ready() is True
        assert calls == 2
        await asyncio.sleep(0.15)
        await controller.shutdown()

    asyncio.run(scenario())


def test_replay_cache_evicts_completed_lru_by_serialized_byte_budget(
    service_settings: Any,
) -> None:
    class SizedResult:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

        def model_dump(self, **_options: Any) -> dict[str, Any]:
            return {
                "model_calls": [],
                "padding": "x" * 400_000,
                "review_run_id": self.run_id,
            }

    class SizedRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def review_json(self, payload: bytes) -> SizedResult:
            self.calls += 1
            return SizedResult(json.loads(payload)["review_run_id"])

        def close(self) -> None:
            return None

    async def scenario() -> None:
        runtime = SizedRuntime()
        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                replay_max_bytes=1 * 1024 * 1024,
            ),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        for run_id, request_hash in (("run-a", "a" * 64), ("run-b", "b" * 64)):
            outcome, replayed = await controller.submit(
                payload=_payload(run_id),
                review_run_id=run_id,
                request_hash=request_hash,
            )
            assert outcome.status_code == 200
            assert replayed is False

        _, replayed = await controller.submit(
            payload=_payload("run-a"),
            review_run_id="run-a",
            request_hash="a" * 64,
        )
        assert replayed is True
        await controller.submit(
            payload=_payload("run-c"),
            review_run_id="run-c",
            request_hash="c" * 64,
        )
        metrics = controller.metrics.snapshot()
        expected_result_bytes = len(
            json.dumps(
                SizedResult("run-a").model_dump(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        assert metrics["replay_cache_entries"] == 2
        assert metrics["replay_cache_result_bytes"] == 2 * expected_result_bytes
        assert metrics["replay_cache_evictions_total"] == 1

        _, replayed = await controller.submit(
            payload=_payload("run-b"),
            review_run_id="run-b",
            request_hash="b" * 64,
        )
        assert replayed is False
        assert runtime.calls == 4
        await controller.shutdown()

    asyncio.run(scenario())


def test_single_oversize_result_is_returned_but_never_cached(
    service_settings: Any,
) -> None:
    class OversizeResult:
        def model_dump(self, **_options: Any) -> dict[str, Any]:
            return {"model_calls": [], "padding": "x" * (1 * 1024 * 1024)}

    class OversizeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def review_json(self, _payload: bytes) -> OversizeResult:
            self.calls += 1
            return OversizeResult()

        def close(self) -> None:
            return None

    async def scenario() -> None:
        runtime = OversizeRuntime()
        controller = AgentServiceController(
            settings=settings_with(
                service_settings,
                replay_max_bytes=1 * 1024 * 1024,
            ),
            runtime=runtime,
            readiness_probe=lambda: object(),
            logger=__import__("logging").getLogger("test-service"),
        )
        for _ in range(2):
            outcome, replayed = await controller.submit(
                payload=_payload("run-oversize"),
                review_run_id="run-oversize",
                request_hash="d" * 64,
            )
            assert outcome.status_code == 200
            assert replayed is False
        metrics = controller.metrics.snapshot()
        assert metrics["replay_cache_entries"] == 0
        assert metrics["replay_cache_result_bytes"] == 0
        assert metrics["replay_cache_oversize_total"] == 2
        assert runtime.calls == 2
        await controller.shutdown()

    asyncio.run(scenario())
