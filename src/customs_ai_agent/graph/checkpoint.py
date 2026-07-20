from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver

from customs_ai_agent.nodes.redact import without_image_payloads


class SensitiveInputSafeCheckpointer(BaseCheckpointSaver[Any]):
    """Redact raw image payloads at every LangGraph persistence write."""

    def __init__(self, delegate: BaseCheckpointSaver[Any]) -> None:
        super().__init__(serde=delegate.serde)
        self._delegate = delegate

    @property
    def config_specs(self) -> list[Any]:
        return self._delegate.config_specs

    def get_tuple(self, config: Any) -> Any:
        return self._delegate.get_tuple(config)

    def list(
        self,
        config: Any | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Any | None = None,
        limit: int | None = None,
    ) -> Iterator[Any]:
        return self._delegate.list(
            config,
            filter=filter,
            before=before,
            limit=limit,
        )

    def put(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        return self._delegate.put(
            config,
            cast(Any, without_image_payloads(checkpoint)),
            cast(Any, without_image_payloads(metadata)),
            new_versions,
        )

    def put_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._delegate.put_writes(
            config,
            [(channel, without_image_payloads(value)) for channel, value in writes],
            task_id,
            task_path,
        )

    def delete_thread(self, thread_id: str) -> None:
        self._delegate.delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._delegate.delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._delegate.copy_thread(source_thread_id, target_thread_id)

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        self._delegate.prune(thread_ids, strategy=strategy)

    def get_delta_channel_history(
        self,
        *,
        config: Any,
        channels: Sequence[str],
    ) -> Mapping[str, Any]:
        return self._delegate.get_delta_channel_history(
            config=config,
            channels=channels,
        )

    async def aget_tuple(self, config: Any) -> Any:
        return await self._delegate.aget_tuple(config)

    async def alist(
        self,
        config: Any | None,
        *,
        filter: dict[str, Any] | None = None,
        before: Any | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Any]:
        async for checkpoint in self._delegate.alist(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield checkpoint

    async def aput(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        return await self._delegate.aput(
            config,
            cast(Any, without_image_payloads(checkpoint)),
            cast(Any, without_image_payloads(metadata)),
            new_versions,
        )

    async def aput_writes(
        self,
        config: Any,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self._delegate.aput_writes(
            config,
            [(channel, without_image_payloads(value)) for channel, value in writes],
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await self._delegate.adelete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self._delegate.adelete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self._delegate.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        await self._delegate.aprune(thread_ids, strategy=strategy)

    async def aget_delta_channel_history(
        self,
        *,
        config: Any,
        channels: Sequence[str],
    ) -> Mapping[str, Any]:
        return await self._delegate.aget_delta_channel_history(
            config=config,
            channels=channels,
        )

    def get_next_version(self, current: Any | None, channel: None) -> Any:
        return self._delegate.get_next_version(current, channel)

    def with_allowlist(
        self,
        extra_allowlist: Collection[tuple[str, ...]],
    ) -> BaseCheckpointSaver[Any]:
        return SensitiveInputSafeCheckpointer(self._delegate.with_allowlist(extra_allowlist))


__all__ = ["SensitiveInputSafeCheckpointer"]
