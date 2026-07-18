from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from customs_ai_agent.adapters import (
    ImageInputError,
    LLMConfigurationError,
    LLMError,
    OcrLayoutProvider,
    OcrLayoutProviderError,
    PdfRasterizerError,
    RetrievalError,
    build_document_analyzer,
    ensure_llm_mode_available,
    image_page_from_file,
    pdf_pages_from_pdf_file,
    validate_ocr_layout_ledger,
)
from customs_ai_agent.adapters.images import (
    IsolatedPdfium2Rasterizer as _Pdfium2Rasterizer,
)
from customs_ai_agent.adapters.retrieval import (
    HttpRegulationRetriever,
    fetch_supported_review_as_of,
)
from customs_ai_agent.contracts import (
    ContractSchemaError,
    LocalReviewManifestV1,
    SchemaScope,
    SchemaVersion,
    get_schema,
)
from customs_ai_agent.fixtures.scenarios import list_scenarios
from customs_ai_agent.ocr_release import (
    OcrReleaseGateAttestation,
    OcrReleaseGateAttestationError,
    ReleaseLockedOcrProvider,
    load_canonical_ocr_release_gate,
)
from customs_ai_agent.runner import (
    answer_question,
    run_review,
    run_review_v2,
    to_review_run_result,
    to_review_run_result_v2,
)
from customs_ai_agent.schemas.models import (
    DocumentInput,
    DocumentInputV2,
    ImagePageInput,
    ImagePageProvenance,
    PdfPagesInput,
    MAX_IMAGE_BYTES_PER_RUN,
    MAX_IMAGE_PAGES_PER_DOCUMENT,
    MAX_IMAGE_PAGES_PER_RUN,
    OcrLayoutLedger,
    ReviewRunRequest,
    ReviewRunRequestV2,
    ReviewRunResult,
    ReviewRunResultV2,
    XaiReport,
    XaiReportV2,
)


AgentContractVersion = Literal["v1", "v2"]
AgentContractKind = Literal["request", "result"]
OcrProviderName = Literal["paddle", "docling", "ledger"]

_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_TEXT_FILE_BYTES = 800_000
_MAX_OCR_LEDGER_BYTES = 16 * 1024 * 1024
_MAX_CONTRACT_BYTES = 32 * 1024 * 1024
_DEFAULT_PDF_DPI = 180
_SAFE_CLI_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_GENERIC_CLI_ERROR_MESSAGE = "The command could not be completed safely."
_PUBLIC_RESULT_FORBIDDEN_KEYS = frozenset(
    {
        "documents",
        "content_text",
        "image_pages",
        "pdf_pages",
        "data_base64",
        "source_path",
        "manifest_path",
        "ocr_ledger_path",
        "raw_response",
        "raw_prompt",
        "filename",
    }
)


class LocalReviewInputError(ValueError):
    """A content-free error for an invalid local review manifest or source."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class ContractValidationError(ValueError):
    """A content-free error for an invalid public Agent contract payload."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class _PreparedManifest:
    request_values: dict[str, Any]
    ledgers: Mapping[str, OcrLayoutLedger]
    has_visual_documents: bool


def _cmd_list_scenarios() -> int:
    for scenario in list_scenarios():
        print(scenario)
    return 0


EVIDENCE_ENV_KEYS = frozenset(
    {
        "EVIDENCE_SERVICE_BASE_URL",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
    }
)


def _explicit_env_values(
    env_file: str | os.PathLike[str] | None,
) -> dict[str, str]:
    if env_file is None:
        return {}
    path = Path(env_file).expanduser()
    if not path.is_file():
        raise LLMConfigurationError(
            "env_file_not_found",
            "The explicitly configured environment file does not exist.",
        )
    raw = dotenv_values(dotenv_path=path, verbose=False)
    return {name: value for name in EVIDENCE_ENV_KEYS if isinstance((value := raw.get(name)), str)}


def _configured_value(
    name: str,
    file_values: dict[str, str],
    default: str | None = None,
) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return file_values.get(name, default)


def _configured_rag_url(
    cli_value: str | None,
    file_values: dict[str, str],
) -> str | None:
    if cli_value is not None:
        return cli_value
    configured = _configured_value("EVIDENCE_SERVICE_BASE_URL", file_values)
    if configured is None:
        return None
    if not configured.strip():
        raise ValueError("EVIDENCE_SERVICE_BASE_URL cannot be blank when configured")
    return configured


def _configured_service_token(file_values: dict[str, str]) -> str | None:
    configured = _configured_value("EVIDENCE_SERVICE_TOKEN", file_values)
    return configured if configured else None


def _configured_allowed_origins(file_values: dict[str, str]) -> frozenset[str]:
    configured = _configured_value(
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        file_values,
    )
    if not configured:
        return frozenset()
    return frozenset(origin.strip() for origin in configured.split(",") if origin.strip())


def _parse_review_as_of(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("review date must use YYYY-MM-DD") from exc


def _read_bounded_bytes(path: Path, *, byte_limit: int, label: str) -> bytes:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise OSError
        with resolved.open("rb") as handle:
            payload = handle.read(byte_limit + 1)
    except OSError:
        raise LocalReviewInputError(
            f"{label}_unreadable",
            f"The configured {label} file is not a readable regular file.",
        ) from None
    if not payload or len(payload) > byte_limit:
        raise LocalReviewInputError(
            f"{label}_size_invalid",
            f"The configured {label} file is empty or exceeds its safety limit.",
        )
    return payload


def _read_json_object(path: Path, *, byte_limit: int, label: str) -> dict[str, Any]:
    raw = _read_bounded_bytes(path, byte_limit=byte_limit, label=label)
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LocalReviewInputError(
            f"{label}_json_invalid",
            f"The configured {label} file is not valid UTF-8 JSON.",
        ) from None
    if not isinstance(payload, dict):
        raise LocalReviewInputError(
            f"{label}_shape_invalid",
            f"The configured {label} JSON root must be an object.",
        )
    return payload


def _resolve_manifest_file(manifest_dir: Path, relative_value: str, *, label: str) -> Path:
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or bool(relative_path.drive) or ".." in relative_path.parts:
        raise LocalReviewInputError(
            f"{label}_path_invalid",
            f"The configured {label} path must stay within the manifest directory.",
        )
    manifest_root = manifest_dir.resolve(strict=True)
    current = manifest_root
    try:
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise OSError
        resolved = (manifest_root / relative_path).resolve(strict=True)
        resolved.relative_to(manifest_root)
    except (OSError, ValueError):
        raise LocalReviewInputError(
            f"{label}_path_invalid",
            f"The configured {label} path must identify a non-symlink file within the manifest directory.",
        ) from None
    if not resolved.is_file():
        raise LocalReviewInputError(
            f"{label}_path_invalid",
            f"The configured {label} path must identify a regular file.",
        )
    return resolved


def _read_text_source(path: Path) -> str:
    raw = _read_bounded_bytes(path, byte_limit=_MAX_TEXT_FILE_BYTES, label="text_source")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise LocalReviewInputError(
            "text_source_encoding_invalid",
            "Text source files must use UTF-8 encoding.",
        ) from None
    if not text.strip() or len(text) > 200_000:
        raise LocalReviewInputError(
            "text_source_content_invalid",
            "Text source content is blank or exceeds the document character limit.",
        )
    return text


class _LedgerOcrLayoutProvider:
    """Static OCR provider for deterministic, byte-bound evaluation fixtures."""

    def __init__(self, ledgers: Mapping[str, OcrLayoutLedger]) -> None:
        self._ledgers = dict(ledgers)

    def analyze(
        self,
        *,
        document_id: str,
        pages: Sequence[ImagePageInput],
    ) -> OcrLayoutLedger:
        ledger = self._ledgers.get(document_id)
        if ledger is None:
            raise OcrLayoutProviderError(
                "ocr_ledger_missing",
                "No static OCR ledger is registered for this document.",
            )
        validate_ocr_layout_ledger(
            ledger,
            document_id=document_id,
            pages=pages,
        )
        return ledger


def _load_manifest(path: str | os.PathLike[str]) -> tuple[LocalReviewManifestV1, Path]:
    manifest_path = Path(path).expanduser()
    payload = _read_json_object(
        manifest_path,
        byte_limit=_MAX_MANIFEST_BYTES,
        label="review_manifest",
    )
    try:
        manifest = LocalReviewManifestV1.model_validate(payload)
    except ValidationError:
        raise LocalReviewInputError(
            "review_manifest_invalid",
            "The review manifest does not satisfy the local manifest contract.",
        ) from None
    return manifest, manifest_path.resolve(strict=True).parent


def _manifest_has_visual_documents(manifest: LocalReviewManifestV1) -> bool:
    return any(document.source.kind in {"image", "pdf"} for document in manifest.documents)


def _validate_visual_manifest_mode(
    manifest: LocalReviewManifestV1,
    has_visual_documents: bool,
) -> None:
    if has_visual_documents and manifest.contract_version != "v2":
        raise LocalReviewInputError(
            "v2_required_for_visual_input",
            "Image and PDF sources require the Agent V2 contract.",
        )
    if has_visual_documents and manifest.llm_mode != "llm":
        raise LocalReviewInputError(
            "llm_required_for_visual_input",
            "Image and PDF sources require llm mode.",
        )


def _attest_review_ocr_runtime(
    args: argparse.Namespace,
    *,
    has_visual_documents: bool,
) -> OcrReleaseGateAttestation | None:
    provider_name = getattr(args, "ocr_provider", None)
    ocr_python = getattr(args, "ocr_python", None)
    report_path = getattr(args, "ocr_release_gate_report", None)
    report_sha256 = getattr(args, "ocr_release_gate_report_sha256", None)
    descriptor_path = getattr(args, "ocr_release_descriptor", None)
    rendered_root = getattr(args, "ocr_release_rendered_root", None)
    expected_wheel_sha256 = getattr(args, "expected_agent_wheel_sha256", None)
    environment_lock_path = getattr(args, "ocr_environment_lock", None)
    environment_lock_sha256 = getattr(args, "ocr_environment_lock_sha256", None)
    model_assets_lock_path = getattr(args, "ocr_model_assets_lock", None)
    model_assets_lock_sha256 = getattr(args, "ocr_model_assets_lock_sha256", None)
    release_arguments = (
        report_path,
        report_sha256,
        descriptor_path,
        rendered_root,
        expected_wheel_sha256,
        environment_lock_path,
        environment_lock_sha256,
        model_assets_lock_path,
        model_assets_lock_sha256,
    )

    if not has_visual_documents:
        if ocr_python is not None:
            raise LocalReviewInputError(
                "ocr_python_without_visual_input",
                "--ocr-python is valid only when the manifest contains image or PDF sources.",
            )
        if any(value is not None for value in release_arguments):
            raise LocalReviewInputError(
                "ocr_release_gate_without_visual_input",
                "OCR release Gate arguments require image or PDF input.",
            )
        return None

    if provider_name is None:
        raise LocalReviewInputError(
            "ocr_provider_required",
            "Image and PDF sources require an explicit OCR provider.",
        )
    if provider_name == "ledger":
        if ocr_python is not None:
            raise LocalReviewInputError(
                "ocr_python_forbidden",
                "--ocr-python is valid only for paddle or docling OCR mode.",
            )
        if any(value is not None for value in release_arguments):
            raise LocalReviewInputError(
                "ocr_release_gate_forbidden",
                "Ledger mode cannot consume real-provider release Gate arguments.",
            )
        return None
    if provider_name == "paddle":
        expected_provider: Literal["paddle", "docling"] = "paddle"
    elif provider_name == "docling":
        expected_provider = "docling"
    else:
        raise LocalReviewInputError(
            "ocr_provider_invalid",
            "The requested OCR provider is not supported.",
        )
    if ocr_python is None or any(value is None for value in release_arguments):
        raise LocalReviewInputError(
            "ocr_release_gate_arguments_required",
            "Real OCR providers require the complete canonical release attestation.",
        )
    if (
        not isinstance(ocr_python, (str, os.PathLike))
        or not isinstance(report_path, (str, os.PathLike))
        or not isinstance(report_sha256, str)
        or not isinstance(descriptor_path, (str, os.PathLike))
        or not isinstance(rendered_root, (str, os.PathLike))
        or not isinstance(expected_wheel_sha256, str)
        or not isinstance(environment_lock_path, (str, os.PathLike))
        or not isinstance(environment_lock_sha256, str)
        or not isinstance(model_assets_lock_path, (str, os.PathLike))
        or not isinstance(model_assets_lock_sha256, str)
    ):
        raise LocalReviewInputError(
            "ocr_release_gate_arguments_invalid",
            "OCR release Gate argument types are invalid.",
        )

    try:
        return load_canonical_ocr_release_gate(
            report_path=Path(report_path),
            expected_report_sha256=report_sha256,
            expected_provider=expected_provider,
            descriptor_path=Path(descriptor_path),
            rendered_root=Path(rendered_root),
            expected_agent_wheel_sha256=expected_wheel_sha256,
            environment_lock_path=Path(environment_lock_path),
            expected_environment_lock_sha256=environment_lock_sha256,
            model_assets_lock_path=Path(model_assets_lock_path),
            expected_model_assets_lock_sha256=model_assets_lock_sha256,
            ocr_python=Path(ocr_python),
        )
    except OcrReleaseGateAttestationError as exc:
        raise LocalReviewInputError(
            exc.code,
            "OCR release Gate attestation failed.",
        ) from None


def _prepare_manifest(
    manifest: LocalReviewManifestV1,
    manifest_dir: Path,
    *,
    ocr_provider: OcrProviderName | None,
    review_as_of: date | None,
) -> _PreparedManifest:
    has_visual_documents = _manifest_has_visual_documents(manifest)
    _validate_visual_manifest_mode(manifest, has_visual_documents)
    if has_visual_documents and ocr_provider is None:
        raise LocalReviewInputError(
            "ocr_provider_required",
            "Image and PDF sources require an explicit OCR provider.",
        )

    documents: list[dict[str, Any]] = []
    ledgers: dict[str, OcrLayoutLedger] = {}
    rasterizer: _Pdfium2Rasterizer | None = None
    visual_page_count = 0
    visual_byte_count = 0
    for manifest_document in manifest.documents:
        source = manifest_document.source
        source_path = _resolve_manifest_file(
            manifest_dir,
            source.path,
            label=f"{source.kind}_source",
        )
        document_values: dict[str, Any] = {
            "document_id": manifest_document.document_id,
            "filename": manifest_document.filename,
            "declared_type": manifest_document.declared_type,
            "requirement_document_codes": manifest_document.requirement_document_codes,
        }
        if source.kind == "text":
            if source.ocr_ledger is not None:
                raise LocalReviewInputError(
                    "text_ledger_forbidden",
                    "Text sources cannot reference an OCR ledger.",
                )
            document_values["content_text"] = _read_text_source(source_path)
        else:
            if source.kind == "image":
                image_pages = [image_page_from_file(source_path)]
                document_values["image_pages"] = image_pages
            else:
                rasterizer = rasterizer or _Pdfium2Rasterizer()
                pdf_pages = pdf_pages_from_pdf_file(
                    source_path,
                    rasterizer,
                    dpi=_DEFAULT_PDF_DPI,
                    max_pages=MAX_IMAGE_PAGES_PER_DOCUMENT,
                )
                image_pages = pdf_pages.pages
                document_values["pdf_pages"] = pdf_pages
            visual_page_count += len(image_pages)
            visual_byte_count += sum(page.byte_length for page in image_pages)
            if (
                visual_page_count > MAX_IMAGE_PAGES_PER_RUN
                or visual_byte_count > MAX_IMAGE_BYTES_PER_RUN
            ):
                raise LocalReviewInputError(
                    "visual_run_limit_exceeded",
                    "Visual documents exceed the per-run page or byte safety limit.",
                )
            if ocr_provider == "ledger":
                if source.ocr_ledger is None:
                    raise LocalReviewInputError(
                        "ocr_ledger_required",
                        "Ledger OCR mode requires one ledger for every visual document.",
                    )
                ledger_path = _resolve_manifest_file(
                    manifest_dir,
                    source.ocr_ledger,
                    label="ocr_ledger",
                )
                raw_ledger = _read_json_object(
                    ledger_path,
                    byte_limit=_MAX_OCR_LEDGER_BYTES,
                    label="ocr_ledger",
                )
                try:
                    ledger = OcrLayoutLedger.model_validate(raw_ledger)
                    validate_ocr_layout_ledger(
                        ledger,
                        document_id=manifest_document.document_id,
                        pages=image_pages,
                    )
                except (ValidationError, OcrLayoutProviderError):
                    raise LocalReviewInputError(
                        "ocr_ledger_invalid",
                        "An OCR ledger does not match its validated visual document.",
                    ) from None
                ledgers[manifest_document.document_id] = ledger
            elif source.ocr_ledger is not None:
                raise LocalReviewInputError(
                    "ocr_ledger_forbidden",
                    "Precomputed ledgers are accepted only in explicit ledger OCR mode.",
                )
        documents.append(document_values)

    request_values: dict[str, Any] = {
        "documents": documents,
        "llm_mode": manifest.llm_mode,
        "trade_type": manifest.trade_type,
        "jurisdiction": manifest.jurisdiction,
        "uploaded_requirement_document_codes": (manifest.uploaded_requirement_document_codes),
    }
    if manifest.case_id is not None:
        request_values["case_id"] = manifest.case_id
    if manifest.review_run_id is not None:
        request_values["review_run_id"] = manifest.review_run_id
    effective_review_as_of = review_as_of or manifest.review_as_of
    if effective_review_as_of is not None:
        request_values["review_as_of"] = effective_review_as_of
    return _PreparedManifest(
        request_values=request_values,
        ledgers=ledgers,
        has_visual_documents=has_visual_documents,
    )


def _build_live_retriever(
    *,
    llm_mode: Literal["mock", "llm"],
    env_file: str | os.PathLike[str] | None,
    rag_url: str | None,
    review_as_of: date | None,
) -> tuple[Any, date | None]:
    file_values = _explicit_env_values(env_file)
    configured_url = _configured_rag_url(rag_url, file_values)
    if llm_mode == "llm" and configured_url is None:
        raise LLMConfigurationError(
            "explicit_retriever_required",
            "llm mode requires --rag-url or EVIDENCE_SERVICE_BASE_URL; implicit Mock RAG is forbidden.",
        )
    if configured_url is None:
        return None, review_as_of
    service_token = _configured_service_token(file_values)
    allowed_origins = _configured_allowed_origins(file_values)
    token_header = _configured_value(
        "EVIDENCE_SERVICE_TOKEN_HEADER",
        file_values,
        "X-Demo-Token",
    )
    if not token_header or not token_header.strip():
        raise ValueError("EVIDENCE_SERVICE_TOKEN_HEADER cannot be blank")
    retriever_options: dict[str, Any] = {
        "service_token": service_token,
        "token_header": token_header.strip(),
    }
    if allowed_origins:
        retriever_options["allowed_origins"] = allowed_origins
    retriever = HttpRegulationRetriever(configured_url, **retriever_options)
    if review_as_of is None:
        review_as_of = fetch_supported_review_as_of(
            configured_url,
            **retriever_options,
        )
    return retriever, review_as_of


def _build_visual_analyzer(
    *,
    provider_name: OcrProviderName,
    ledgers: Mapping[str, OcrLayoutLedger],
    env_file: str | os.PathLike[str] | None,
    python_executable: str | os.PathLike[str] | None,
    release_attestation: OcrReleaseGateAttestation | None,
) -> Any:
    if provider_name == "ledger":
        if release_attestation is not None:
            raise LocalReviewInputError(
                "ocr_release_gate_forbidden",
                "Ledger mode cannot consume a real-provider release attestation.",
            )
        if python_executable is not None:
            raise LocalReviewInputError(
                "ocr_python_forbidden",
                "--ocr-python is valid only for paddle or docling OCR mode.",
            )
        provider: OcrLayoutProvider = _LedgerOcrLayoutProvider(ledgers)
    else:
        if release_attestation is None:
            raise LocalReviewInputError(
                "ocr_release_gate_required",
                "A passing OCR release Gate is required for a real provider.",
            )
        import customs_ai_agent.adapters as adapters

        factory = getattr(adapters, "build_ocr_layout_provider", None)
        if not callable(factory):
            raise LocalReviewInputError(
                "ocr_provider_unavailable",
                "The selected OCR provider adapter is not available in this build.",
            )
        provider = factory(
            provider_name,
            python_executable=python_executable,
        )
        provider = ReleaseLockedOcrProvider(provider, release_attestation)
    return build_document_analyzer(
        env_file,
        ocr_layout_provider=provider,
    )


def _assert_public_result_safe(payload: Any) -> None:
    if isinstance(payload, dict):
        forbidden = _PUBLIC_RESULT_FORBIDDEN_KEYS.intersection(payload)
        if forbidden:
            raise RuntimeError(
                "public_result_safety_violation: internal input material reached the public result"
            )
        for value in payload.values():
            _assert_public_result_safe(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_public_result_safe(value)


def _agent_contract_schema(
    version: AgentContractVersion,
    kind: AgentContractKind,
) -> dict[str, Any]:
    schema_kind: Literal["review-request", "review-result"] = (
        "review-request" if kind == "request" else "review-result"
    )
    return get_schema("backend", version, schema_kind)


def _assert_known_object_fields(
    payload: Mapping[str, Any],
    model: type[BaseModel],
    *,
    label: str,
) -> None:
    unknown = set(payload).difference(model.model_fields)
    if unknown:
        raise ContractValidationError(
            "contract_unknown_field",
            f"The {label} contains fields outside the selected contract.",
        )


def _validate_contract_payload(
    version: AgentContractVersion,
    kind: AgentContractKind,
    payload: dict[str, Any],
) -> BaseModel:
    try:
        if kind == "request":
            request_model = ReviewRunRequest if version == "v1" else ReviewRunRequestV2
            document_model = DocumentInput if version == "v1" else DocumentInputV2
            _assert_known_object_fields(payload, request_model, label="Agent request")
            raw_documents = payload.get("documents")
            if isinstance(raw_documents, list):
                for raw_document in raw_documents:
                    if not isinstance(raw_document, dict):
                        raise ContractValidationError(
                            "contract_shape_invalid",
                            "Each Agent request document must be an object.",
                        )
                    if version == "v1" and "image_pages" in raw_document:
                        raise ContractValidationError(
                            "v1_visual_input_forbidden",
                            "The V1 Agent request contract accepts text documents only.",
                        )
                    _assert_known_object_fields(
                        raw_document,
                        document_model,
                        label="Agent request document",
                    )
                    raw_pages = raw_document.get("image_pages") or []
                    raw_pdf_pages = raw_document.get("pdf_pages")
                    if isinstance(raw_pdf_pages, dict):
                        _assert_known_object_fields(
                            raw_pdf_pages,
                            PdfPagesInput,
                            label="Agent PDF page envelope",
                        )
                        raw_pages = raw_pdf_pages.get("pages") or []
                    if isinstance(raw_pages, list):
                        for raw_page in raw_pages:
                            if not isinstance(raw_page, dict):
                                raise ContractValidationError(
                                    "contract_shape_invalid",
                                    "Each image page must be an object.",
                                )
                            _assert_known_object_fields(
                                raw_page,
                                ImagePageInput,
                                label="Agent image page",
                            )
                            raw_provenance = raw_page.get("provenance")
                            if isinstance(raw_provenance, dict):
                                _assert_known_object_fields(
                                    raw_provenance,
                                    ImagePageProvenance,
                                    label="Agent image provenance",
                                )
            return request_model.model_validate(payload)

        result_model = ReviewRunResultV2 if version == "v2" else ReviewRunResult
        report_model = XaiReportV2 if version == "v2" else XaiReport
        _assert_known_object_fields(payload, result_model, label="Agent result")
        raw_report = payload.get("report")
        if isinstance(raw_report, dict):
            _assert_known_object_fields(
                raw_report,
                report_model,
                label="Agent report",
            )
        expected_version = "1.0" if version == "v1" else "2.0"
        if payload.get("schema_version", expected_version) != expected_version:
            raise ContractValidationError(
                "contract_version_mismatch",
                "The payload schema_version does not match the selected Agent contract.",
            )
        result = result_model.model_validate(payload)
        _assert_public_result_safe(payload)
        return result
    except ContractValidationError:
        raise
    except ValidationError:
        raise ContractValidationError(
            "contract_validation_failed",
            "The payload does not satisfy the selected Agent contract.",
        ) from None


def _serialize_public_result(
    state: dict[str, Any],
    *,
    contract_version: AgentContractVersion,
) -> dict[str, Any]:
    # Validate the pre-model report as well. Pydantic intentionally ignores
    # unknown fields on legacy report models, so checking only the serialized
    # model could silently discard (and therefore conceal) a filename leak
    # introduced upstream instead of failing the CLI safety gate.
    _assert_public_result_safe(state.get("report_json"))
    public_result = (
        to_review_run_result_v2(state) if contract_version == "v2" else to_review_run_result(state)
    )
    payload = public_result.model_dump(mode="json")
    _assert_public_result_safe(payload)
    _validate_contract_payload(contract_version, "result", payload)
    return payload


def _write_json_payload(path_value: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path_value).expanduser()
    if path.exists() and path.is_symlink():
        raise LocalReviewInputError(
            "output_path_invalid",
            "The output path cannot be a symbolic link.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _cmd_review(args: argparse.Namespace) -> int:
    manifest, manifest_dir = _load_manifest(args.manifest)
    has_visual_documents = _manifest_has_visual_documents(manifest)
    _validate_visual_manifest_mode(manifest, has_visual_documents)
    release_attestation = _attest_review_ocr_runtime(
        args,
        has_visual_documents=has_visual_documents,
    )
    ensure_llm_mode_available(manifest.llm_mode, getattr(args, "env_file", None))
    prepared = _prepare_manifest(
        manifest,
        manifest_dir,
        ocr_provider=args.ocr_provider,
        review_as_of=getattr(args, "review_as_of", None),
    )
    analyzer = None
    if prepared.has_visual_documents:
        analyzer = _build_visual_analyzer(
            provider_name=args.ocr_provider,
            ledgers=prepared.ledgers,
            env_file=getattr(args, "env_file", None),
            python_executable=getattr(args, "ocr_python", None),
            release_attestation=release_attestation,
        )

    requested_date = prepared.request_values.get("review_as_of")
    retriever, effective_date = _build_live_retriever(
        llm_mode=manifest.llm_mode,
        env_file=getattr(args, "env_file", None),
        rag_url=getattr(args, "rag_url", None),
        review_as_of=requested_date,
    )
    if effective_date is not None:
        prepared.request_values["review_as_of"] = effective_date
    request_model = ReviewRunRequestV2 if manifest.contract_version == "v2" else ReviewRunRequest
    try:
        request = request_model.model_validate(prepared.request_values)
    except ValidationError:
        raise LocalReviewInputError(
            "review_request_invalid",
            "The prepared request does not satisfy the selected Agent contract.",
        ) from None

    run_options: dict[str, Any] = {"retriever": retriever}
    if analyzer is not None:
        run_options["analyzer"] = analyzer
    if getattr(args, "env_file", None) is not None:
        run_options["env_file"] = args.env_file
    state = (
        run_review_v2(request, **run_options)
        if isinstance(request, ReviewRunRequestV2)
        else run_review(request, **run_options)
    )
    payload = _serialize_public_result(
        state,
        contract_version=manifest.contract_version,
    )
    _write_json_payload(args.out, payload)

    report = state.get("report_json") or {}
    summary = report.get("summary") or {}
    print(f"case_id: {state['case_id']}")
    print(f"status: {report.get('overall_status', state.get('final_status'))}")
    print(f"issues: {summary.get('total_issues', 0)}")
    print(f"requires_human_review: {state.get('requires_human_review', False)}")
    print("result_written: true")
    return 0


def _cmd_export_schema(args: argparse.Namespace) -> int:
    scope: SchemaScope = args.scope
    version: SchemaVersion = args.version
    raw_kind = str(args.kind)
    if scope == "backend" and raw_kind in {"request", "result"}:
        raw_kind = f"review-{raw_kind}"
    try:
        schema = get_schema(scope, version, raw_kind)  # type: ignore[arg-type]
    except ContractSchemaError:
        raise ContractValidationError(
            "contract_schema_selection_invalid",
            "The selected scope, version, and kind do not identify a released schema.",
        ) from None
    encoded = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        _write_json_payload(args.out, schema)
        print("schema_written: true")
    else:
        print(encoded, end="")
    return 0


def _cmd_validate_contract(args: argparse.Namespace) -> int:
    payload = _read_json_object(
        Path(args.input),
        byte_limit=_MAX_CONTRACT_BYTES,
        label="contract_payload",
    )
    _validate_contract_payload(args.version, args.kind, payload)
    print(f"valid: agent {args.version} {args.kind}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    # Validate provider configuration before any Evidence Service network call.
    ensure_llm_mode_available(args.llm_mode, getattr(args, "env_file", None))
    file_values = _explicit_env_values(getattr(args, "env_file", None))
    rag_url = _configured_rag_url(getattr(args, "rag_url", None), file_values)
    if args.llm_mode == "llm" and rag_url is None:
        raise LLMConfigurationError(
            "explicit_retriever_required",
            "llm mode requires --rag-url or EVIDENCE_SERVICE_BASE_URL; implicit Mock RAG is forbidden.",
        )
    review_as_of = getattr(args, "review_as_of", None)
    retriever = None
    if rag_url is not None:
        service_token = _configured_service_token(file_values)
        allowed_origins = _configured_allowed_origins(file_values)
        token_header = _configured_value(
            "EVIDENCE_SERVICE_TOKEN_HEADER",
            file_values,
            "X-Demo-Token",
        )
        if not token_header or not token_header.strip():
            raise ValueError("EVIDENCE_SERVICE_TOKEN_HEADER cannot be blank")
        retriever_options: dict[str, Any] = {
            "service_token": service_token,
            "token_header": token_header.strip(),
        }
        if allowed_origins:
            retriever_options["allowed_origins"] = allowed_origins
        retriever = HttpRegulationRetriever(rag_url, **retriever_options)
        if review_as_of is None:
            readiness_options = dict(retriever_options)
            review_as_of = fetch_supported_review_as_of(
                rag_url,
                **readiness_options,
            )
    request_values = {
        "scenario_id": args.scenario,
        "llm_mode": args.llm_mode,
    }
    if review_as_of is not None:
        request_values["review_as_of"] = review_as_of
    request = ReviewRunRequest.model_validate(request_values)
    if getattr(args, "env_file", None) is not None:
        state = run_review(request, retriever=retriever, env_file=args.env_file)
    else:
        state = run_review(request, retriever=retriever)
    public_result = to_review_run_result(state)
    payload = json.dumps(
        public_result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    report = state.get("report_json") or {}
    summary = report.get("summary") or {}
    print(f"case_id: {state['case_id']}")
    print(f"status: {report.get('overall_status', state.get('final_status'))}")
    print(f"issues: {summary.get('total_issues', 0)}")
    print(f"requires_human_review: {state.get('requires_human_review', False)}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"wrote: {out_path}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = payload.get("report_json", payload)
    answer = answer_question(report, args.question)
    print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="customs-ai-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-scenarios")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--out")
    run_parser.add_argument("--llm-mode", choices=["mock", "llm"], default="mock")
    run_parser.add_argument(
        "--env-file",
        help=(
            "Explicit local environment file for llm mode. Only ANTHROPIC_API_KEY "
            "and allowlisted Claude/Evidence settings are read without mutating "
            "the process environment. Existing process values win; the path is not logged."
        ),
    )
    run_parser.add_argument(
        "--review-as-of",
        type=_parse_review_as_of,
        help=(
            "Review date in YYYY-MM-DD. For Live RAG, omission uses the date "
            "advertised by the Evidence Service readiness endpoint."
        ),
    )
    run_parser.add_argument(
        "--rag-url",
        help=(
            "Evidence Service base URL; calls POST /internal/rag/v1/retrieve. "
            "Overrides EVIDENCE_SERVICE_BASE_URL. If neither is configured, uses "
            "the deterministic mock retriever. A failed live call never falls back."
        ),
    )

    review_parser = subparsers.add_parser(
        "review",
        help=(
            "Review local text/image/PDF sources described by a bounded JSON manifest. "
            "Only the redacted public result is written."
        ),
    )
    review_parser.add_argument("--manifest", required=True)
    review_parser.add_argument("--out", required=True)
    review_parser.add_argument(
        "--env-file",
        help=(
            "Explicit shared local environment file. Use the workspace "
            "_secrets/.env; the path and secret values are never logged."
        ),
    )
    review_parser.add_argument(
        "--review-as-of",
        type=_parse_review_as_of,
        help=(
            "Override the manifest review date. If both are omitted in Live RAG, "
            "use the Evidence Service readiness date."
        ),
    )
    review_parser.add_argument(
        "--rag-url",
        help=(
            "Evidence Service base URL. llm mode requires this value or "
            "EVIDENCE_SERVICE_BASE_URL; there is no Mock RAG fallback."
        ),
    )
    review_parser.add_argument(
        "--ocr-provider",
        choices=["paddle", "docling", "ledger"],
        help=(
            "Required for image/PDF input; there is no implicit provider. paddle and "
            "docling require the canonical release Gate, while ledger consumes pinned "
            "offline observations."
        ),
    )
    review_parser.add_argument(
        "--ocr-python",
        type=Path,
        help=(
            "Required Python executable for the isolated paddle/docling OCR provider. "
            "Its installed Agent wheel must match the release Gate. It is forbidden "
            "in ledger mode."
        ),
    )
    review_parser.add_argument(
        "--ocr-release-gate-report",
        type=Path,
        help="Exact canonical 60+20 combined PASS report for a real OCR provider.",
    )
    review_parser.add_argument(
        "--ocr-release-gate-report-sha256",
        help="Caller-pinned SHA-256 of the canonical combined OCR release report.",
    )
    review_parser.add_argument(
        "--ocr-release-descriptor",
        type=Path,
        help="Canonical OCR benchmark descriptor used by the combined report.",
    )
    review_parser.add_argument(
        "--ocr-release-rendered-root",
        type=Path,
        help="Canonical rendered dataset root whose manifest is pinned by the report.",
    )
    review_parser.add_argument(
        "--expected-agent-wheel-sha256",
        help=(
            "Caller-pinned exact Agent wheel SHA-256; it must match both the report "
            "and the wheel installed in --ocr-python."
        ),
    )
    review_parser.add_argument(
        "--ocr-environment-lock",
        type=Path,
        help=(
            "Selected provider's hash-locked Python requirements file. The installed "
            "OCR environment must match it exactly."
        ),
    )
    review_parser.add_argument(
        "--ocr-environment-lock-sha256",
        help="Caller-pinned SHA-256 of --ocr-environment-lock.",
    )
    review_parser.add_argument(
        "--ocr-model-assets-lock",
        type=Path,
        help=(
            "Model asset revision/tree lock for the selected OCR provider. The live "
            "provider cache must match it exactly."
        ),
    )
    review_parser.add_argument(
        "--ocr-model-assets-lock-sha256",
        help="Caller-pinned SHA-256 of --ocr-model-assets-lock.",
    )

    export_schema_parser = subparsers.add_parser(
        "export-schema",
        help=("Export one immutable Agent-owned Backend, internal, or local JSON Schema."),
    )
    export_schema_parser.add_argument(
        "--scope",
        choices=["backend", "internal", "local"],
        default="backend",
    )
    export_schema_parser.add_argument("--version", choices=["v1", "v2"], required=True)
    export_schema_parser.add_argument(
        "--kind",
        choices=[
            "request",
            "result",
            "review-request",
            "review-result",
            "ocr-layout-ledger",
            "claude-text-candidate",
            "claude-vision-candidate",
            "claude-crop-read-candidate",
            "review-manifest",
        ],
        required=True,
    )
    export_schema_parser.add_argument("--out")

    validate_contract_parser = subparsers.add_parser(
        "validate-contract",
        help="Validate one JSON payload against the selected Agent wire contract.",
    )
    validate_contract_parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        required=True,
    )
    validate_contract_parser.add_argument(
        "--kind",
        choices=["request", "result"],
        required=True,
    )
    validate_contract_parser.add_argument("--input", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--report", required=True)
    chat_parser.add_argument("--question", required=True)

    return parser


def _safe_cli_error_code(exc: Exception) -> str:
    coded_errors = (
        LocalReviewInputError,
        ContractValidationError,
        LLMError,
        OcrLayoutProviderError,
        OcrReleaseGateAttestationError,
    )
    if isinstance(exc, coded_errors):
        code = getattr(exc, "code", None)
        return (
            code
            if isinstance(code, str) and _SAFE_CLI_ERROR_CODE.fullmatch(code) is not None
            else "internal_error"
        )
    if isinstance(exc, RetrievalError):
        return "retrieval_error"
    if isinstance(exc, ImageInputError):
        return "image_input_error"
    if isinstance(exc, PdfRasterizerError):
        return "pdf_rasterizer_error"
    if isinstance(exc, ContractSchemaError):
        return "contract_schema_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(exc, ValidationError):
        return "validation_error"
    if isinstance(exc, OSError):
        return "io_error"
    if isinstance(exc, ValueError):
        return "invalid_input"
    return "internal_error"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-scenarios":
            return _cmd_list_scenarios()
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "review":
            return _cmd_review(args)
        if args.command == "export-schema":
            return _cmd_export_schema(args)
        if args.command == "validate-contract":
            return _cmd_validate_contract(args)
        if args.command == "chat":
            return _cmd_chat(args)
        parser.error("unknown command")
    except Exception as exc:
        code = _safe_cli_error_code(exc)
        payload = {
            "error_code": code,
            "message": _GENERIC_CLI_ERROR_MESSAGE,
            "status": "error",
        }
        print(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
            file=sys.stderr,
        )
        return 1
    return 2
