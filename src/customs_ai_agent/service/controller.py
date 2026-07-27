from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from customs_ai_agent.runtime import (
    AgentRuntimeError,
    RuntimeConfigurationError,
    RuntimeConflictError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.telemetry import ServiceMetrics, safe_log


class ServiceRuntime(Protocol):
    def review_json(self, payload: bytes) -> Any: ...

    def review_v3_json(self, payload: bytes) -> Any: ...

    def revision_v3_json(self, payload: bytes) -> Any: ...

    def answer_v3_json(self, payload: bytes) -> Any: ...

    def review_experience_json(self, payload: bytes) -> Any: ...

    def answer_v2_json(self, payload: bytes) -> Any: ...

    def review_experience_stream_json(
        self,
        payload: bytes,
        stage_callback: Callable[[dict[str, Any]], None],
    ) -> Any: ...

    def answer_v2_stream_json(
        self,
        payload: bytes,
        stage_callback: Callable[[dict[str, Any]], None],
    ) -> Any: ...

    def assert_uat_capability_current(self, *, require_active: bool = False) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ServiceOutcome:
    status_code: int
    result: dict[str, Any] | None = None
    error_code: str | None = None
    retryable: bool = False
    retry_after_seconds: int | None = None
    replay_result_bytes: int = 0


@dataclass(slots=True)
class _ReplayEntry:
    request_hash: str
    task: asyncio.Task[ServiceOutcome]
    completed_at: float | None = None
    result_bytes: int = 0
    stage_events: list[dict[str, Any]] = field(default_factory=list)
    stage_callbacks: set[Callable[[dict[str, Any]], None]] = field(default_factory=set)


class AgentServiceController:
    """Own process-local admission, replay, dependency probing, and runtime lifecycle."""

    def __init__(
        self,
        *,
        settings: ServiceSettings,
        runtime: ServiceRuntime,
        readiness_probe: Callable[[], object],
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.metrics = ServiceMetrics()
        self._readiness_probe = readiness_probe
        self._logger = logger
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrency,
            thread_name_prefix="customs-agent-review",
        )
        self._probe_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="customs-agent-readiness",
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._state_lock = asyncio.Lock()
        self._readiness_lock = asyncio.Lock()
        self._replays: OrderedDict[str, _ReplayEntry] = OrderedDict()
        self._replay_result_bytes = 0
        self._background: set[asyncio.Task[None]] = set()
        self._readiness_cache: tuple[float, bool] | None = None
        self._closing = False
        self._closed = False

    @property
    def closing(self) -> bool:
        return self._closing

    def assert_uat_capability_current(self, *, require_active: bool = False) -> None:
        """Fail before readiness I/O when enhanced UAT authority is stale."""

        checker = getattr(self.runtime, "assert_uat_capability_current", None)
        if not callable(checker):
            raise RuntimeConfigurationError(
                "uat_capability_unavailable",
                "The runtime does not expose a local-UAT capability boundary.",
            )
        checker(require_active=require_active)

    def _update_replay_metrics(self) -> None:
        self.metrics.set_replay_cache(
            entries=len(self._replays),
            result_bytes=self._replay_result_bytes,
        )

    def _remove_replay(self, run_id: str) -> None:
        entry = self._replays.pop(run_id, None)
        if entry is not None:
            self._replay_result_bytes = max(
                0,
                self._replay_result_bytes - entry.result_bytes,
            )
            self._update_replay_metrics()

    def _purge_replays(self, now: float) -> None:
        expired = [
            run_id
            for run_id, entry in self._replays.items()
            if entry.completed_at is not None
            and now - entry.completed_at >= self.settings.replay_ttl_seconds
        ]
        for run_id in expired:
            self._remove_replay(run_id)
            self.metrics.increment("replay_cache_evictions_total")

    def _make_replay_room(self) -> bool:
        if len(self._replays) < self.settings.replay_max_entries:
            return True
        for run_id, entry in tuple(self._replays.items()):
            if entry.task.done():
                self._remove_replay(run_id)
                self.metrics.increment("replay_cache_evictions_total")
                return True
        return False

    def _enforce_replay_byte_cap(self, run_id: str, entry: _ReplayEntry) -> None:
        if entry.result_bytes > self.settings.replay_max_bytes:
            self._remove_replay(run_id)
            self.metrics.increment("replay_cache_oversize_total")
            return
        while self._replay_result_bytes > self.settings.replay_max_bytes:
            evicted = False
            for candidate_run_id, candidate in tuple(self._replays.items()):
                if candidate.task.done():
                    self._remove_replay(candidate_run_id)
                    self.metrics.increment("replay_cache_evictions_total")
                    evicted = True
                    break
            if not evicted:
                break

    async def submit(
        self,
        *,
        payload: bytes,
        review_run_id: str,
        request_hash: str,
        runtime_method_name: str = "review_json",
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[ServiceOutcome, bool]:
        self.metrics.increment("review_requests_total")
        async with self._state_lock:
            if self._closing:
                self.metrics.increment("review_rejected_total")
                raise ServiceHttpError("service_draining", 503, retryable=True)
            now = time.monotonic()
            self._purge_replays(now)
            entry = self._replays.get(review_run_id)
            replayed = entry is not None
            if entry is not None:
                if entry.request_hash != request_hash:
                    self.metrics.increment("review_rejected_total")
                    raise ServiceHttpError("idempotency_conflict", 409)
                self._replays.move_to_end(review_run_id)
                self.metrics.increment("review_replayed_total")
                task = entry.task
                if stage_callback is not None:
                    for event in entry.stage_events:
                        stage_callback(event)
                    entry.stage_callbacks.add(stage_callback)
            else:
                if not self._make_replay_room():
                    self.metrics.increment("review_rejected_total")
                    raise ServiceHttpError(
                        "service_busy",
                        503,
                        retryable=True,
                        retry_after_seconds=max(1, round(self.settings.queue_wait_seconds)),
                    )
                entry_holder: dict[str, _ReplayEntry] = {}
                loop = asyncio.get_running_loop()

                def record_stage_event(event: dict[str, Any]) -> None:
                    # Runtime callbacks execute in a worker thread. Move all
                    # journal/subscriber mutation back onto the owning loop.
                    def record() -> None:
                        target = entry_holder.get("entry")
                        if target is None or len(target.stage_events) >= 64:
                            return
                        safe_event = copy.deepcopy(event)
                        target.stage_events.append(safe_event)
                        for callback in tuple(target.stage_callbacks):
                            callback(copy.deepcopy(safe_event))

                    loop.call_soon_threadsafe(record)

                task = asyncio.create_task(
                    self._run_once(
                        payload,
                        runtime_method_name,
                        stage_callback=(record_stage_event if stage_callback is not None else None),
                    )
                )
                entry = _ReplayEntry(request_hash=request_hash, task=task)
                entry_holder["entry"] = entry
                if stage_callback is not None:
                    entry.stage_callbacks.add(stage_callback)
                self._replays[review_run_id] = entry
                self._update_replay_metrics()

                def mark_completed(completed_task: asyncio.Task[ServiceOutcome]) -> None:
                    entry.completed_at = time.monotonic()
                    if completed_task.cancelled():
                        self._remove_replay(review_run_id)
                        return
                    try:
                        outcome = completed_task.result()
                    except BaseException:
                        self._remove_replay(review_run_id)
                        return
                    entry.result_bytes = outcome.replay_result_bytes
                    self._replay_result_bytes += entry.result_bytes
                    if self._replays.get(review_run_id) is entry:
                        self._replays.move_to_end(review_run_id)
                    self._enforce_replay_byte_cap(review_run_id, entry)
                    self._update_replay_metrics()

                task.add_done_callback(mark_completed)
        try:
            return await asyncio.shield(task), replayed
        finally:
            if stage_callback is not None:
                async with self._state_lock:
                    entry.stage_callbacks.discard(stage_callback)

    def _track_background(self, task: asyncio.Task[None]) -> None:
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _finish_timed_out_worker(
        self,
        future: asyncio.Future[Any],
        started: float,
    ) -> None:
        try:
            await asyncio.shield(future)
        except BaseException:  # noqa: S110 - timed-out worker outcome is intentionally consumed
            pass
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            inflight = self.metrics.finish(duration_ms)
            self._semaphore.release()
            safe_log(
                self._logger,
                logging.WARNING,
                "review_worker_finished_after_timeout",
                duration_ms=duration_ms,
                inflight=inflight,
            )

    async def _run_once(
        self,
        payload: bytes,
        runtime_method_name: str = "review_json",
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ServiceOutcome:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.queue_wait_seconds,
            )
        except TimeoutError:
            self.metrics.increment("review_rejected_total")
            return ServiceOutcome(
                status_code=503,
                error_code="service_busy",
                retryable=True,
                retry_after_seconds=max(1, round(self.settings.queue_wait_seconds)),
            )
        if self._closing:
            self._semaphore.release()
            self.metrics.increment("review_rejected_total")
            return ServiceOutcome(
                status_code=503,
                error_code="service_draining",
                retryable=True,
            )

        inflight = self.metrics.begin()
        safe_log(
            self._logger,
            logging.INFO,
            "review_started",
            inflight=inflight,
        )
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        runtime_method = getattr(self.runtime, runtime_method_name, None)
        if not callable(runtime_method):
            self.metrics.finish(0)
            self._semaphore.release()
            self.metrics.increment("review_failed_total")
            return ServiceOutcome(status_code=503, error_code="capability_unavailable")
        if stage_callback is None:
            worker = loop.run_in_executor(self._executor, runtime_method, payload)
        else:
            worker = loop.run_in_executor(
                self._executor,
                runtime_method,
                payload,
                stage_callback,
            )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=self.settings.run_timeout_seconds,
            )
        except TimeoutError:
            self.metrics.increment("review_timeouts_total")
            self.metrics.increment("review_failed_total")
            background = asyncio.create_task(self._finish_timed_out_worker(worker, started))
            self._track_background(background)
            return ServiceOutcome(
                status_code=504,
                error_code="review_timeout",
                retryable=True,
            )
        except asyncio.CancelledError:
            background = asyncio.create_task(self._finish_timed_out_worker(worker, started))
            self._track_background(background)
            raise
        except RuntimeInputError:
            outcome = ServiceOutcome(status_code=422, error_code="request_invalid")
        except RuntimeConflictError:
            outcome = ServiceOutcome(status_code=409, error_code="stale_revision")
        except RuntimeConfigurationError:
            outcome = ServiceOutcome(
                status_code=503,
                error_code="dependency_unready",
                retryable=True,
            )
        except RuntimeExecutionError:
            outcome = ServiceOutcome(
                status_code=502,
                error_code="review_execution_failed",
                retryable=True,
            )
        except AgentRuntimeError:
            outcome = ServiceOutcome(
                status_code=502,
                error_code="review_execution_failed",
                retryable=True,
            )
        except Exception:
            outcome = ServiceOutcome(status_code=500, error_code="internal_error")
        else:
            try:
                public_result = result.model_dump(mode="json", warnings=False)
                serialized = json.dumps(
                    public_result,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                replay_result_bytes = len(serialized)
                del serialized
                model_calls = len(public_result.get("model_calls") or [])
                if model_calls:
                    self.metrics.increment("model_calls_total", model_calls)
                outcome = ServiceOutcome(
                    status_code=200,
                    result=public_result,
                    replay_result_bytes=replay_result_bytes,
                )
            except Exception:
                outcome = ServiceOutcome(status_code=500, error_code="internal_error")

        duration_ms = round((time.perf_counter() - started) * 1000)
        inflight = self.metrics.finish(duration_ms)
        self._semaphore.release()
        if outcome.result is None:
            self.metrics.increment("review_failed_total")
        else:
            self.metrics.increment("review_completed_total")
        safe_log(
            self._logger,
            logging.INFO if outcome.result is not None else logging.WARNING,
            "review_finished",
            duration_ms=duration_ms,
            error_code=outcome.error_code or "",
            inflight=inflight,
            retryable=outcome.retryable,
            status_code=outcome.status_code,
        )
        return outcome

    async def ready(self) -> bool:
        if self._closing or self._closed:
            return False
        now = time.monotonic()
        cached = self._readiness_cache
        if cached is not None and now < cached[0]:
            return cached[1]
        async with self._readiness_lock:
            now = time.monotonic()
            cached = self._readiness_cache
            if cached is not None and now < cached[0]:
                return cached[1]
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._probe_executor, self._readiness_probe)
            try:
                await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self.settings.readiness_timeout_seconds,
                )
            except Exception:
                ready = False
            else:
                ready = True
            self._readiness_cache = (
                time.monotonic() + self.settings.readiness_ttl_seconds,
                ready,
            )
            return ready

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closing = True
        deadline = time.monotonic() + self.settings.shutdown_grace_seconds
        active = [entry.task for entry in self._replays.values() if not entry.task.done()]
        if active:
            remaining = max(0.0, deadline - time.monotonic())
            _, pending = await asyncio.wait(active, timeout=remaining)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        background = tuple(self._background)
        background_still_running = False
        if background:
            remaining = max(0.0, deadline - time.monotonic())
            _, pending_background = await asyncio.wait(background, timeout=remaining)
            if pending_background:
                background_still_running = True
                safe_log(
                    self._logger,
                    logging.ERROR,
                    "shutdown_grace_exhausted",
                    inflight=self.metrics.snapshot()["inflight"],
                )
        if background_still_running:
            # Closing the Anthropic client while a bounded SDK call still owns it is
            # unsafe. The orchestrator termination deadline is the final boundary.
            safe_log(self._logger, logging.ERROR, "runtime_close_deferred")
        else:
            try:
                self.runtime.close()
            except Exception:
                safe_log(self._logger, logging.ERROR, "runtime_close_failed")
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._probe_executor.shutdown(wait=False, cancel_futures=True)
        self._closed = True


__all__ = [
    "AgentServiceController",
    "ServiceOutcome",
    "ServiceRuntime",
]
