from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast
from uuid import uuid4

from pydantic import ValidationError

from customs_ai_agent.ocr_reading_order import (
    MAX_OCR_SPANS_PER_PAGE,
    OCR_READING_ORDER_POLICY_ID,
    OcrSpanLimitExceededError,
    preserve_provider_span_order,
)
from customs_ai_agent.schemas.models import (
    BBox,
    ImagePageInput,
    MAX_IMAGE_PAGES_PER_DOCUMENT,
    OcrLayoutLedger,
    OcrLayoutPage,
    OcrLayoutSpan,
    stable_ocr_ledger_id,
)

OcrProviderName = Literal["paddle", "docling"]

_WORKER_PROTOCOL: Final = "customs-ocr-worker/v2"
_DEFAULT_TIMEOUT_SECONDS: Final = 180.0
_MAX_WORKER_REQUEST_BYTES: Final = 32 * 1024 * 1024
_MAX_WORKER_RESPONSE_BYTES: Final = 8 * 1024 * 1024
_PROCESS_SHUTDOWN_SECONDS: Final = 2.0
_SAFE_CODE_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DOCUMENT_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_WORKER_ENV_ALLOWLIST: Final = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HF_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
    }
)
_FORBIDDEN_RESPONSE_KEYS: Final = frozenset(
    {"data_base64", "raw", "raw_bytes", "image_bytes", "document_bytes"}
)
_ROLE_ORDER: Final = {
    "heading": 0,
    "label": 1,
    "value": 2,
    "text": 3,
    "table": 4,
    "table_row": 5,
    "table_cell": 6,
}
_ALLOWED_ROLES: Final = frozenset(_ROLE_ORDER)


class OcrLayoutProviderError(RuntimeError):
    """Safe OCR provider error that never carries document content."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


class OcrLayoutProvider(Protocol):
    """Provider-neutral boundary for independent OCR/layout observations."""

    def analyze(
        self,
        *,
        document_id: str,
        pages: Sequence[ImagePageInput],
    ) -> OcrLayoutLedger: ...


class _WorkerProtocolError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class _WorkerReadFailure:
    def __init__(self, code: str) -> None:
        self.code = code


_WorkerQueueItem = bytes | _WorkerReadFailure | None


class _JsonlWorkerClient:
    """One persistent, content-opaque JSONL worker process."""

    def __init__(
        self,
        *,
        provider: OcrProviderName,
        python_executable: Path,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = provider
        self._python_executable = python_executable
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: queue.Queue[_WorkerQueueItem] = queue.Queue()
        self._reader: threading.Thread | None = None

    def request(self, payload: Mapping[str, object]) -> dict[str, Any]:
        try:
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise _WorkerProtocolError("ocr_worker_request_invalid") from exc
        if len(encoded) > _MAX_WORKER_REQUEST_BYTES:
            raise _WorkerProtocolError("ocr_worker_request_too_large")

        with self._lock:
            process = self._ensure_started()
            if process.stdin is None:
                self._stop_locked()
                raise _WorkerProtocolError("ocr_worker_pipe_unavailable")
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._stop_locked()
                raise _WorkerProtocolError(
                    "ocr_worker_unavailable",
                    retryable=True,
                ) from exc

            try:
                response = self._responses.get(timeout=self._timeout_seconds)
            except queue.Empty as exc:
                self._stop_locked()
                raise _WorkerProtocolError(
                    "ocr_worker_timeout",
                    retryable=True,
                ) from exc

            if response is None:
                self._stop_locked()
                raise _WorkerProtocolError(
                    "ocr_worker_exited",
                    retryable=True,
                )
            if isinstance(response, _WorkerReadFailure):
                self._stop_locked()
                raise _WorkerProtocolError(response.code)
            try:
                decoded = json.loads(response)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._stop_locked()
                raise _WorkerProtocolError("ocr_worker_response_malformed") from exc
            if not isinstance(decoded, dict):
                self._stop_locked()
                raise _WorkerProtocolError("ocr_worker_response_malformed")
            return cast(dict[str, Any], decoded)

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def runtime_process_id(self) -> int | None:
        """Return the live isolated worker PID without exposing request content."""

        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return None
            return process.pid

    def _ensure_started(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._stop_locked()
        responses: queue.Queue[_WorkerQueueItem] = queue.Queue()
        self._responses = responses
        try:
            # The configured interpreter is resolved/validated at provider
            # construction and the worker argv is fixed; no shell is involved.
            process = subprocess.Popen(  # noqa: S603
                [
                    str(self._python_executable),
                    "-I",
                    "-m",
                    "customs_ai_agent.ocr_worker",
                    "--provider",
                    self._provider,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Third-party diagnostics can contain local paths or untrusted
                # OCR text. They are never part of the public protocol/log.
                stderr=subprocess.DEVNULL,
                env=_worker_environment(self._provider),
                cwd=_worker_cwd(),
                bufsize=0,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
        except (OSError, ValueError) as exc:
            raise _WorkerProtocolError(
                "ocr_worker_start_failed",
                retryable=True,
            ) from exc
        if process.stdout is None or process.stdin is None:
            _terminate_process(process)
            raise _WorkerProtocolError("ocr_worker_pipe_unavailable")
        self._process = process
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(process, responses),
            name=f"customs-ocr-{self._provider}-stdout",
            daemon=True,
        )
        self._reader.start()
        return process

    def _read_stdout(
        self,
        process: subprocess.Popen[bytes],
        responses: queue.Queue[_WorkerQueueItem],
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            responses.put(_WorkerReadFailure("ocr_worker_pipe_unavailable"))
            return
        try:
            while True:
                line = stdout.readline(_MAX_WORKER_RESPONSE_BYTES + 1)
                if not line:
                    responses.put(None)
                    return
                if len(line) > _MAX_WORKER_RESPONSE_BYTES:
                    self._drain_oversized_line(stdout, line)
                    responses.put(_WorkerReadFailure("ocr_worker_response_too_large"))
                    return
                if not line.endswith(b"\n"):
                    responses.put(_WorkerReadFailure("ocr_worker_response_malformed"))
                    return
                responses.put(line)
        except (OSError, ValueError):
            responses.put(_WorkerReadFailure("ocr_worker_read_failed"))

    @staticmethod
    def _drain_oversized_line(stdout: Any, first: bytes) -> None:
        chunk = first
        try:
            while chunk and not chunk.endswith(b"\n"):
                chunk = stdout.readline(_MAX_WORKER_RESPONSE_BYTES + 1)
        except (OSError, ValueError):
            return

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        _terminate_process(process)
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=0.2)


class _SubprocessOcrLayoutProvider:
    def __init__(
        self,
        *,
        provider: OcrProviderName,
        python_executable: Path,
        worker: _JsonlWorkerClient | None = None,
    ) -> None:
        self._provider = provider
        self._worker = worker or _JsonlWorkerClient(
            provider=provider,
            python_executable=python_executable,
        )
        self._analysis_lock = threading.Lock()

    def analyze(
        self,
        *,
        document_id: str,
        pages: Sequence[ImagePageInput],
    ) -> OcrLayoutLedger:
        if not pages:
            raise OcrLayoutProviderError(
                "ocr_pages_missing",
                "OCR requires at least one validated page.",
            )
        if _DOCUMENT_ID_PATTERN.fullmatch(document_id) is None:
            raise OcrLayoutProviderError(
                "ocr_document_id_invalid",
                "OCR document_id violates the public ledger contract.",
            )
        if len(pages) > MAX_IMAGE_PAGES_PER_DOCUMENT or [
            page.page_number for page in pages
        ] != list(range(1, len(pages) + 1)):
            raise OcrLayoutProviderError(
                "ocr_pages_invalid",
                "OCR pages must be consecutive, source ordered, and within limits.",
            )
        request_id = "ocrreq_" + uuid4().hex
        request: dict[str, object] = {
            "protocol": _WORKER_PROTOCOL,
            "request_id": request_id,
            "provider": self._provider,
            "document_id": document_id,
            "pages": [
                {
                    "page_number": page.page_number,
                    "media_type": page.media_type,
                    "data_base64": page.data_base64,
                    "byte_length": page.byte_length,
                    "sha256": page.sha256,
                    "width_px": page.width_px,
                    "height_px": page.height_px,
                }
                for page in pages
            ],
        }
        with self._analysis_lock:
            try:
                response = self._worker.request(request)
            except _WorkerProtocolError as exc:
                raise OcrLayoutProviderError(
                    exc.code,
                    "The isolated OCR worker failed safely.",
                    retryable=exc.retryable,
                ) from None
        try:
            self._validate_response_envelope(
                response,
                request_id=request_id,
                pages=pages,
            )
            if response.get("status") == "error":
                error = response.get("error")
                if not isinstance(error, dict) or not set(error).issubset({"code", "retryable"}):
                    raise OcrLayoutProviderError(
                        "ocr_worker_response_malformed",
                        "The isolated OCR worker returned an invalid error envelope.",
                    )
                code = error.get("code")
                retryable = error.get("retryable", False)
                if (
                    not isinstance(code, str)
                    or _SAFE_CODE_PATTERN.fullmatch(code) is None
                    or not isinstance(retryable, bool)
                ):
                    raise OcrLayoutProviderError(
                        "ocr_worker_response_malformed",
                        "The isolated OCR worker returned an invalid error envelope.",
                    )
                raise OcrLayoutProviderError(
                    code,
                    "The OCR backend rejected the document safely.",
                    retryable=retryable,
                )

            result = response.get("result")
            if not isinstance(result, dict):
                raise OcrLayoutProviderError(
                    "ocr_worker_response_malformed",
                    "The isolated OCR worker returned an invalid result envelope.",
                )
            return _ledger_from_worker_result(
                document_id=document_id,
                provider=self._provider,
                source_pages=pages,
                result=cast(dict[str, Any], result),
            )
        except OcrLayoutProviderError:
            # Do not reuse a process after any failed or untrusted response.
            self._worker.close()
            raise

    @staticmethod
    def _validate_response_envelope(
        response: Mapping[str, Any],
        *,
        request_id: str,
        pages: Sequence[ImagePageInput],
    ) -> None:
        if (
            response.get("protocol") != _WORKER_PROTOCOL
            or response.get("request_id") != request_id
            or response.get("status") not in {"ok", "error"}
        ):
            raise OcrLayoutProviderError(
                "ocr_worker_response_malformed",
                "The isolated OCR worker returned an invalid response envelope.",
            )
        if _contains_forbidden_response_key(response):
            raise OcrLayoutProviderError(
                "ocr_worker_raw_content_rejected",
                "The OCR worker response contained forbidden raw content.",
            )
        page_payloads = {page.data_base64 for page in pages}
        if _contains_exact_string(response, page_payloads):
            raise OcrLayoutProviderError(
                "ocr_worker_raw_content_rejected",
                "The OCR worker response contained forbidden raw content.",
            )

    def close(self) -> None:
        self._worker.close()

    def runtime_process_id(self) -> int | None:
        """Return the live worker PID for release-only resource attestation."""

        return self._worker.runtime_process_id()

    def __enter__(self) -> _SubprocessOcrLayoutProvider:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            # Destructors must never mask the caller's result.
            pass


def build_ocr_layout_provider(
    provider: Literal["paddle", "docling"],
    *,
    python_executable: str | os.PathLike[str] | None = None,
) -> OcrLayoutProvider:
    """Build an isolated PaddleOCR or Docling OCR/layout provider.

    The selected Python environment must already contain this package and the
    provider dependency. Model initialization and page processing happen only
    inside a persistent subprocess; image bytes cross that boundary only in a
    base64 field written to stdin.
    """

    if provider not in {"paddle", "docling"}:
        raise OcrLayoutProviderError(
            "ocr_provider_invalid",
            "OCR provider must be either 'paddle' or 'docling'.",
        )
    executable = Path(
        os.fspath(python_executable) if python_executable is not None else sys.executable
    ).expanduser()
    try:
        executable = executable.resolve(strict=True)
    except OSError as exc:
        raise OcrLayoutProviderError(
            "ocr_python_unavailable",
            "The isolated OCR Python executable does not exist.",
        ) from exc
    if not executable.is_file():
        raise OcrLayoutProviderError(
            "ocr_python_unavailable",
            "The isolated OCR Python executable is not a file.",
        )
    return _SubprocessOcrLayoutProvider(
        provider=provider,
        python_executable=executable,
    )


def _ledger_from_worker_result(
    *,
    document_id: str,
    provider: OcrProviderName,
    source_pages: Sequence[ImagePageInput],
    result: dict[str, Any],
) -> OcrLayoutLedger:
    provider_id = result.get("provider_id")
    engine_id = result.get("engine_id")
    engine_version = result.get("engine_version")
    reading_order_policy_id = result.get("reading_order_policy_id")
    model_id = result.get("model_id")
    model_assets_lock_sha256 = result.get("model_assets_lock_sha256")
    raw_pages = result.get("pages")
    if (
        set(result)
        != {
            "provider_id",
            "engine_id",
            "engine_version",
            "reading_order_policy_id",
            "model_id",
            "model_assets_lock_sha256",
            "pages",
        }
        or provider_id != provider
        or not isinstance(engine_id, str)
        or not engine_id
        or len(engine_id) > 128
        or not isinstance(engine_version, str)
        or not engine_version
        or len(engine_version) > 128
        or reading_order_policy_id != OCR_READING_ORDER_POLICY_ID
        or not isinstance(model_id, str)
        or not model_id
        or len(model_id) > 256
        or not isinstance(model_assets_lock_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", model_assets_lock_sha256) is None
        or not isinstance(raw_pages, list)
        or len(raw_pages) != len(source_pages)
    ):
        raise OcrLayoutProviderError(
            "ocr_worker_result_invalid",
            "The OCR worker result metadata is invalid.",
        )

    page_results: dict[int, dict[str, Any]] = {}
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise OcrLayoutProviderError(
                "ocr_worker_result_invalid",
                "The OCR worker page result is invalid.",
            )
        page_number = raw_page.get("page_number")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number in page_results
        ):
            raise OcrLayoutProviderError(
                "ocr_worker_result_invalid",
                "The OCR worker page numbering is invalid.",
            )
        page_results[page_number] = cast(dict[str, Any], raw_page)

    ledger_pages: list[OcrLayoutPage] = []
    try:
        for source_page in source_pages:
            raw_page = page_results.get(source_page.page_number)
            if raw_page is None:
                raise OcrLayoutProviderError(
                    "ocr_worker_result_invalid",
                    "The OCR worker omitted a source page.",
                )
            raw_spans = raw_page.get("spans")
            if not isinstance(raw_spans, list) or not raw_spans:
                raise OcrLayoutProviderError(
                    "ocr_no_observations",
                    "OCR produced no independently grounded observations.",
                )
            spans = _normalize_worker_spans(
                document_id=document_id,
                provider_id=provider_id,
                engine_id=engine_id,
                engine_version=engine_version,
                page=source_page,
                raw_spans=raw_spans,
            )
            ledger_pages.append(
                OcrLayoutPage(
                    page_number=source_page.page_number,
                    image_sha256=source_page.sha256,
                    width_px=source_page.width_px,
                    height_px=source_page.height_px,
                    spans=spans,
                )
            )
        ledger_id = stable_ocr_ledger_id(
            document_id=document_id,
            provider_id=provider_id,
            engine_id=engine_id,
            engine_version=engine_version,
            reading_order_policy_id=reading_order_policy_id,
            model_id=model_id,
            model_assets_lock_sha256=model_assets_lock_sha256,
            pages=ledger_pages,
        )
        ledger = OcrLayoutLedger(
            ledger_id=ledger_id,
            document_id=document_id,
            provider_id=provider_id,
            engine_id=engine_id,
            engine_version=engine_version,
            reading_order_policy_id=reading_order_policy_id,
            model_id=model_id,
            model_assets_lock_sha256=model_assets_lock_sha256,
            pages=ledger_pages,
        )
    except ValidationError as exc:
        raise OcrLayoutProviderError(
            "ocr_worker_result_invalid",
            "The OCR worker result violated the provider-neutral ledger contract.",
        ) from exc
    validate_ocr_layout_ledger(
        ledger,
        document_id=document_id,
        pages=source_pages,
    )
    return ledger


def _normalize_worker_spans(
    *,
    document_id: str,
    provider_id: str,
    engine_id: str,
    engine_version: str,
    page: ImagePageInput,
    raw_spans: list[Any],
) -> list[OcrLayoutSpan]:
    if len(raw_spans) > MAX_OCR_SPANS_PER_PAGE:
        raise OcrLayoutProviderError(
            "ocr_span_limit_exceeded",
            "The OCR worker exceeded the per-page observation limit.",
        )
    normalized: list[dict[str, Any]] = []
    for source_index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, dict):
            raise OcrLayoutProviderError(
                "ocr_worker_result_invalid",
                "The OCR worker returned an invalid span.",
            )
        role = raw_span.get("role", "text")
        text = raw_span.get("text")
        raw_bbox = raw_span.get("bbox")
        confidence = raw_span.get("confidence")
        table_key = raw_span.get("table_key")
        row_index = raw_span.get("row_index")
        column_index = raw_span.get("column_index")
        if (
            role not in _ALLOWED_ROLES
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 12_000
            or not _is_bbox(raw_bbox, width=page.width_px, height=page.height_px)
            or not _is_confidence(confidence)
            or (table_key is not None and (not isinstance(table_key, str) or not table_key))
            or (
                row_index is not None
                and (not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0)
            )
            or (
                column_index is not None
                and (
                    not isinstance(column_index, int)
                    or isinstance(column_index, bool)
                    or column_index < 0
                )
            )
        ):
            raise OcrLayoutProviderError(
                "ocr_worker_result_invalid",
                "The OCR worker returned an invalid span.",
            )
        if role == "table_cell" and (
            table_key is None or row_index is None or column_index is None
        ):
            raise OcrLayoutProviderError(
                "ocr_worker_result_invalid",
                "The OCR worker returned an incomplete table cell.",
            )
        bbox = cast(list[float | int], raw_bbox)
        normalized.append(
            {
                "source_index": source_index,
                "role": role,
                "text": text.strip(),
                "bbox": [float(value) for value in bbox],
                "confidence": float(cast(float | int, confidence)),
                "table_key": table_key,
                "row_index": row_index,
                "column_index": column_index,
            }
        )

    try:
        normalized = preserve_provider_span_order(normalized)
    except OcrSpanLimitExceededError as exc:
        raise OcrLayoutProviderError(
            "ocr_span_limit_exceeded",
            "The OCR worker exceeded the per-page observation limit.",
        ) from exc
    table_ids = _stable_table_ids(
        document_id=document_id,
        provider_id=provider_id,
        engine_id=engine_id,
        engine_version=engine_version,
        page_number=page.page_number,
        spans=normalized,
    )
    spans: list[OcrLayoutSpan] = []
    for reading_order, item in enumerate(normalized):
        stable_material = {
            "document_id": document_id,
            "provider_id": provider_id,
            "engine_id": engine_id,
            "engine_version": engine_version,
            "page": page.page_number,
            "reading_order": reading_order,
            "role": item["role"],
            "text": item["text"],
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "table_key": item["table_key"],
            "row_index": item["row_index"],
            "column_index": item["column_index"],
        }
        digest = hashlib.sha256(_canonical_json(stable_material)).hexdigest()[:24]
        table_key = item["table_key"]
        spans.append(
            OcrLayoutSpan(
                span_id=f"span_p{page.page_number}_{reading_order:05d}_{digest}",
                page=page.page_number,
                reading_order=reading_order,
                role=item["role"],
                text=item["text"],
                bbox=BBox(
                    page=page.page_number,
                    x1=item["bbox"][0],
                    y1=item["bbox"][1],
                    x2=item["bbox"][2],
                    y2=item["bbox"][3],
                ),
                confidence=item["confidence"],
                table_id=table_ids.get(table_key),
                row_index=item["row_index"],
                column_index=item["column_index"],
            )
        )
    return spans


def _stable_table_ids(
    *,
    document_id: str,
    provider_id: str,
    engine_id: str,
    engine_version: str,
    page_number: int,
    spans: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    keys = sorted({cast(str, span["table_key"]) for span in spans if span["table_key"] is not None})
    return {
        key: "table_"
        + hashlib.sha256(
            _canonical_json(
                {
                    "document_id": document_id,
                    "provider_id": provider_id,
                    "engine_id": engine_id,
                    "engine_version": engine_version,
                    "page": page_number,
                    "table_key": key,
                }
            )
        ).hexdigest()[:24]
        for key in keys
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_bbox(value: object, *, width: int, height: int) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(
        isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
        for coordinate in value
    ):
        return False
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _is_confidence(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 1
    )


def _contains_forbidden_response_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_RESPONSE_KEYS:
                return True
            if _contains_forbidden_response_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_response_key(child) for child in value)
    return False


def _contains_exact_string(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, str):
        return value in forbidden
    if isinstance(value, dict):
        return any(_contains_exact_string(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(_contains_exact_string(child, forbidden) for child in value)
    return False


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_PROCESS_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=_PROCESS_SHUTDOWN_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            return
    except OSError:
        return


def _worker_environment(provider: OcrProviderName) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in _WORKER_ENV_ALLOWLIST
    }
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if provider == "paddle":
        environment["FLAGS_use_mkldnn"] = "0"
        environment["FLAGS_enable_pir_api"] = "0"
        environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        cache_root = _paddle_cache_root()
        environment["PADDLE_HOME"] = cache_root
        environment["PADDLEX_HOME"] = cache_root
        environment["PADDLE_PDX_CACHE_HOME"] = cache_root
        if os.name == "nt":
            environment["HOME"] = cache_root
            environment["USERPROFILE"] = cache_root
    return environment


def _worker_cwd() -> str:
    """Return a neutral directory that cannot shadow the attested wheel."""

    root = Path(tempfile.gettempdir()) / "customs-ai-ocr-worker"
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise _WorkerProtocolError("ocr_worker_cwd_unavailable") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise _WorkerProtocolError("ocr_worker_cwd_unavailable")
    return str(resolved)


def _paddle_cache_root() -> str:
    root = Path(tempfile.gettempdir()) / "customs-ai-ocr-cache" / "paddle"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _WorkerProtocolError("ocr_cache_path_unavailable") from exc
    if os.name != "nt":
        return str(root)
    short = _windows_short_path(root)
    if short.isascii():
        return short
    fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp"
    fallback = fallback / "customs-ai-ocr-cache" / "paddle"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _WorkerProtocolError("ocr_cache_path_unavailable") from exc
    short_fallback = _windows_short_path(fallback)
    if not short_fallback.isascii():
        raise _WorkerProtocolError("ocr_cache_path_unavailable")
    return short_fallback


def _windows_short_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    try:
        import ctypes

        get_short_path_name = ctypes.windll.kernel32.GetShortPathNameW
        required = int(get_short_path_name(str(path), None, 0))
        if required <= 0:
            return str(path)
        buffer = ctypes.create_unicode_buffer(required)
        if int(get_short_path_name(str(path), buffer, required)) <= 0:
            return str(path)
        return str(buffer.value)
    except (AttributeError, OSError, ValueError):
        return str(path)


def validate_ocr_layout_ledger(
    ledger: OcrLayoutLedger,
    *,
    document_id: str,
    pages: Sequence[ImagePageInput],
) -> None:
    """Bind a provider ledger to the exact validated images in this request."""

    if ledger.document_id != document_id:
        raise OcrLayoutProviderError(
            "ocr_ledger_document_mismatch",
            "OCR ledger is bound to a different document.",
        )
    if len(ledger.pages) != len(pages):
        raise OcrLayoutProviderError(
            "ocr_ledger_page_mismatch",
            "OCR ledger page count differs from the validated document.",
        )
    for ledger_page, image_page in zip(ledger.pages, pages, strict=True):
        if (
            ledger_page.page_number != image_page.page_number
            or ledger_page.image_sha256 != image_page.sha256
            or ledger_page.width_px != image_page.width_px
            or ledger_page.height_px != image_page.height_px
        ):
            raise OcrLayoutProviderError(
                "ocr_ledger_image_mismatch",
                "OCR ledger provenance differs from the validated source image.",
            )


def index_ocr_spans(ledger: OcrLayoutLedger) -> dict[str, OcrLayoutSpan]:
    return {span.span_id: span for page in ledger.pages for span in page.spans}


__all__ = [
    "OcrLayoutProvider",
    "OcrLayoutProviderError",
    "build_ocr_layout_provider",
    "index_ocr_spans",
    "validate_ocr_layout_ledger",
]
