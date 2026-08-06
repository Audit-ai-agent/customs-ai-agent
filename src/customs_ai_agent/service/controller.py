"""Small local controller for one review endpoint and bounded replay."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.runtime_errors import (
    AgentRuntimeError,
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.service.settings import ServiceSettings


class ServiceRuntime(Protocol):
    def review_v3_json(self, payload: bytes) -> Any: ...

    def assistant_answer_json(self, payload: bytes) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ServiceOutcome:
    status_code: int
    result: dict[str, Any] | None = None
    error_code: str | None = None
    retryable: bool = False
    retry_after_seconds: int | None = None


@dataclass(slots=True)
class _Replay:
    request_hash: str
    task: asyncio.Task[ServiceOutcome]
    completed_at: float | None = None


class AgentServiceController:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        runtime: ServiceRuntime,
        readiness_probe: Callable[[], object],
        logger: logging.Logger,
    ) -> None:
        self._settings = settings
        self._runtime = runtime
        self._readiness_probe = readiness_probe
        self._logger = logger
        self._lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(settings.max_concurrency)
        self._replays: OrderedDict[str, _Replay] = OrderedDict()
        self._closing = False

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [
            run_id
            for run_id, replay in self._replays.items()
            if replay.completed_at is not None
            and now - replay.completed_at >= self._settings.replay_ttl_seconds
        ]
        for run_id in expired:
            self._replays.pop(run_id, None)

    async def submit(
        self,
        *,
        payload: bytes,
        review_run_id: str,
        request_hash: str,
        runtime_method_name: str = "review_v3_json",
    ) -> tuple[ServiceOutcome, bool]:
        if runtime_method_name != "review_v3_json":
            raise ServiceHttpError("capability_unavailable", 503, retryable=True)
        async with self._lock:
            if self._closing:
                raise ServiceHttpError("service_draining", 503, retryable=True)
            self._purge()
            existing = self._replays.get(review_run_id)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise ServiceHttpError("idempotency_conflict", 409)
                self._replays.move_to_end(review_run_id)
                task = existing.task
                replayed = True
            else:
                if len(self._replays) >= self._settings.replay_max_entries:
                    completed = next(
                        (key for key, value in self._replays.items() if value.task.done()), None
                    )
                    if completed is None:
                        raise ServiceHttpError("service_busy", 503, retryable=True)
                    self._replays.pop(completed, None)
                task = asyncio.create_task(self._run_once(payload, runtime_method_name))
                replay = _Replay(request_hash=request_hash, task=task)
                self._replays[review_run_id] = replay
                replayed = False

                def _mark_done(_task: asyncio.Task[ServiceOutcome]) -> None:
                    replay.completed_at = time.monotonic()

                task.add_done_callback(_mark_done)
        return await asyncio.shield(task), replayed

    async def execute(self, *, payload: bytes, runtime_method_name: str) -> ServiceOutcome:
        """Run one non-idempotent internal capability under the same bound."""

        if runtime_method_name != "assistant_answer_json":
            raise ServiceHttpError("capability_unavailable", 503, retryable=True)
        if self._closing:
            raise ServiceHttpError("service_draining", 503, retryable=True)
        return await self._run_once(payload, runtime_method_name)

    async def _run_once(self, payload: bytes, runtime_method_name: str) -> ServiceOutcome:
        try:
            await asyncio.wait_for(
                self._capacity.acquire(), timeout=self._settings.queue_wait_seconds
            )
        except TimeoutError:
            return ServiceOutcome(status_code=503, error_code="service_busy", retryable=True)
        try:
            method = getattr(self._runtime, runtime_method_name, None)
            if not callable(method):
                raise RuntimeConfigurationError(
                    "runtime_capability_unavailable", "The Agent capability is unavailable."
                )
            result = await asyncio.wait_for(
                asyncio.to_thread(method, payload),
                timeout=self._settings.run_timeout_seconds,
            )
            public = result.model_dump(mode="json", warnings=False)
            json.dumps(public, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            return ServiceOutcome(status_code=200, result=public)
        except TimeoutError:
            return ServiceOutcome(status_code=504, error_code="review_timeout", retryable=True)
        except RuntimeInputError:
            return ServiceOutcome(status_code=422, error_code="request_invalid")
        except RuntimeConfigurationError:
            return ServiceOutcome(status_code=503, error_code="dependency_unready", retryable=True)
        except (RuntimeExecutionError, AgentRuntimeError):
            return ServiceOutcome(
                status_code=502, error_code="review_execution_failed", retryable=True
            )
        except Exception:
            return ServiceOutcome(status_code=500, error_code="internal_error")
        finally:
            self._capacity.release()

    async def ready(self) -> bool:
        if self._closing:
            return False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._readiness_probe),
                timeout=self._settings.readiness_timeout_seconds,
            )
        except Exception:
            return False
        return True

    async def shutdown(self) -> None:
        self._closing = True
        pending = [item.task for item in self._replays.values() if not item.task.done()]
        if pending:
            _, still_pending = await asyncio.wait(
                pending, timeout=self._settings.shutdown_grace_seconds
            )
            for task in still_pending:
                task.cancel()
        try:
            self._runtime.close()
        except Exception:
            self._logger.warning("runtime_close_failed")


__all__ = ["AgentServiceController", "ServiceOutcome", "ServiceRuntime"]
