from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _discard_image_payloads_in_place(value: Any) -> None:
    """Release image payload references without copying their surrounding data."""

    if isinstance(value, dict):
        value.pop("data_base64", None)
        for item in value.values():
            _discard_image_payloads_in_place(item)
    elif isinstance(value, list):
        for item in value:
            _discard_image_payloads_in_place(item)


class SensitiveDocumentPayloadStore:
    """One-shot, execution-local ownership boundary for raw image payload copies.

    The store owns only materialized dictionaries created by the runner. It never
    owns or mutates the caller's ``ReviewRunRequest``. A document can be consumed
    once for model analysis; consumed and unconsumed payload references are
    explicitly discarded before any downstream workflow node runs.
    """

    def __init__(
        self,
        payloads: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self._payloads = dict(payloads or {})

    @property
    def is_empty(self) -> bool:
        return not self._payloads

    def take(self, document_id: str) -> dict[str, Any] | None:
        """Transfer one payload out of the store; it cannot be acquired again."""

        return self._payloads.pop(document_id, None)

    def release(self, payload: dict[str, Any] | None) -> None:
        if payload is not None:
            _discard_image_payloads_in_place(payload)

    def clear(self) -> None:
        for payload in self._payloads.values():
            _discard_image_payloads_in_place(payload)
        self._payloads.clear()


__all__ = ["SensitiveDocumentPayloadStore"]
