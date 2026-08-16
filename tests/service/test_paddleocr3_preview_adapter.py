from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import customs_ai_agent.evaluation.synthetic_pdf_candidate_runner as candidate_runner_module
import customs_ai_agent.service.document_preview_runtime as runtime_module
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import (
    DocumentPreviewExecutionError,
    DocumentPreviewUnavailable,
)
from tools.build_document_preview_runtime_inventory import build_runtime_inventory


_P3_RUNTIME_API = "paddleocr3-paddlex-3.2"
_P3_EXTRACTOR_REVISION = "paddleocr-3.2.0-ppocrv5-mobile-en-raster216-v1"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _p3_options() -> dict[str, object]:
    return {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_directory": "models/paddleocr3/det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "models/paddleocr3/rec",
        "raster_scale": 3,
        "device": "cpu",
        "cpu_threads": 1,
        "enable_hpi": False,
        "enable_mkldnn": False,
        "use_tensorrt": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }


def _p3_configuration(*, options: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "paddleocr",
        "runtime_api": _P3_RUNTIME_API,
        "extractor_revision": _P3_EXTRACTOR_REVISION,
        "options": _p3_options() if options is None else options,
    }


def _p3_inventory(*, omitted: str | None = None) -> dict[str, object]:
    packages = []
    for name in sorted(runtime_module._required_packages("paddleocr", _P3_RUNTIME_API)):
        if name == omitted:
            continue
        packages.append(
            {
                "name": name,
                "version": runtime_module._PADDLEOCR3_PINNED_PACKAGE_VERSIONS[name],
                "license_id": "test-license",
                "source_uri": "https://example.test/paddleocr3/",
            }
        )
    return {
        "schema_version": "1.0",
        "engine_kind": "paddleocr",
        "runtime_api": _P3_RUNTIME_API,
        "packages": packages,
        "artifacts": [],
    }


def _preview_input() -> PreviewInputV1:
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-paddleocr3",
            extraction_request_id="request-paddleocr3",
            source_sha256="a" * 64,
            media_type="application/pdf",
        ),
        byte_size=3,
        content=b"pdf",
    )


def _adapter(tmp_path: Path) -> runtime_module.PaddleOcr3PdfPreviewAdapter:
    for relative_directory in (
        "models/paddleocr3/det",
        "models/paddleocr3/rec",
    ):
        (tmp_path / relative_directory).mkdir(parents=True, exist_ok=True)
    return runtime_module.PaddleOcr3PdfPreviewAdapter(
        extractor_revision=_P3_EXTRACTOR_REVISION,
        model_root=tmp_path,
        options=_p3_options(),
    )


def _install_one_page_pdfium(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object] | None = None,
) -> None:
    class FakeRenderedImage:
        size = (100, 100)

        def convert(self, mode: str) -> "FakeRenderedImage":
            assert mode == "RGB"
            return self

    class FakePdfDocument:
        def __init__(self, _content: bytes) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> SimpleNamespace:
            def render(*, scale: float) -> SimpleNamespace:
                if captured is not None:
                    captured["render_scale"] = scale
                return SimpleNamespace(to_pil=FakeRenderedImage)

            return SimpleNamespace(render=render)

    pdfium = ModuleType("pypdfium2")
    pdfium.PdfDocument = FakePdfDocument  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)


def _install_fake_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("numpy")
    module.uint8 = "uint8"  # type: ignore[attr-defined]
    module.zeros = lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype)  # type: ignore[attr-defined]
    module.asarray = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", module)


def test_paddleocr3_configuration_requires_its_explicit_api_and_offline_profile() -> None:
    parsed = runtime_module.parse_runtime_configuration(_canonical(_p3_configuration()))

    assert parsed.runtime_api == _P3_RUNTIME_API
    assert parsed.extractor_revision == _P3_EXTRACTOR_REVISION
    assert parsed.options["device"] == "cpu"
    assert parsed.options["cpu_threads"] == 1
    assert parsed.options["use_doc_orientation_classify"] is False
    assert parsed.options["use_doc_unwarping"] is False
    assert parsed.options["use_textline_orientation"] is False

    missing_api = _p3_configuration()
    missing_api.pop("runtime_api")
    wrong_preprocessing = _p3_options()
    wrong_preprocessing["use_doc_unwarping"] = True
    wrong_raster_scale = _p3_options()
    wrong_raster_scale["raster_scale"] = 1
    wrong_revision = _p3_configuration()
    wrong_revision["extractor_revision"] = "paddleocr-3.2.0-unreviewed-raster216-v1"
    non_paddle_api = {
        "schema_version": "1.0",
        "engine_kind": "docling",
        "runtime_api": _P3_RUNTIME_API,
        "extractor_revision": "docling-test-v1",
        "options": {},
    }
    for value in (
        missing_api,
        _p3_configuration(options=wrong_preprocessing),
        _p3_configuration(options=wrong_raster_scale),
        wrong_revision,
        non_paddle_api,
    ):
        with pytest.raises(
            runtime_module.DocumentPreviewRuntimeError,
            match="document_preview_configuration_invalid",
        ):
            runtime_module.parse_runtime_configuration(_canonical(value))


def test_paddleocr3_revision_whitelist_does_not_rewrite_legacy_paddleocr2() -> None:
    legacy = runtime_module.parse_runtime_configuration(
        _canonical(
            {
                "schema_version": "1.0",
                "engine_kind": "paddleocr",
                "extractor_revision": "paddleocr2-existing-profile-v1",
                "options": {
                    "det_model_directory": "models/paddle/det",
                    "rec_model_directory": "models/paddle/rec",
                    "cls_model_directory": "models/paddle/cls",
                    "ocr_version": "PP-OCRv4",
                    "use_gpu": False,
                    "cpu_threads": 1,
                },
            }
        )
    )

    assert legacy.runtime_api == "paddleocr2-2.9"
    assert legacy.extractor_revision == "paddleocr2-existing-profile-v1"


def test_paddleocr3_inventory_requires_its_own_paddlex_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module, "_recorded_cv2_tree_matches_designated_provider", lambda: None
    )
    inventory = runtime_module.parse_model_inventory(_canonical(_p3_inventory()))
    versions = {item.name: item.version for item in inventory.packages}

    runtime_module.verify_inventory_packages(
        inventory,
        package_version_reader=lambda name: versions[name],
    )
    assert set(versions) == set(runtime_module._PADDLEOCR3_PINNED_PACKAGE_VERSIONS)
    assert len(versions) == 56
    assert {
        "aistudio-sdk",
        "bce-python-sdk",
        "httpcore",
        "modelscope-hub",
        "pydantic-core",
        "pypdfium2",
    }.issubset(versions)
    assert runtime_module._required_packages("paddleocr", _P3_RUNTIME_API) != (
        runtime_module._required_packages("paddleocr")
    )

    missing_transitive = runtime_module.parse_model_inventory(
        _canonical(_p3_inventory(omitted="modelscope-hub"))
    )
    with pytest.raises(
        runtime_module.DocumentPreviewRuntimeError,
        match="document_preview_inventory_packages_invalid",
    ):
        runtime_module.verify_inventory_packages(
            missing_transitive,
            package_version_reader=lambda _name: "test-v1",
        )

    version_drift = _p3_inventory()
    drift_packages = version_drift["packages"]
    assert isinstance(drift_packages, list)
    for item in drift_packages:
        assert isinstance(item, dict)
        if item["name"] == "httpx":
            item["version"] = "0.0.0"
            break
    else:
        pytest.fail("P3 closure must retain the httpx transitive dependency")
    with pytest.raises(
        runtime_module.DocumentPreviewRuntimeError,
        match="document_preview_inventory_packages_invalid",
    ):
        runtime_module.verify_inventory_packages(
            runtime_module.parse_model_inventory(_canonical(version_drift)),
            package_version_reader=lambda name: runtime_module._PADDLEOCR3_PINNED_PACKAGE_VERSIONS[
                name
            ],
        )

    extra_package = _p3_inventory()
    packages = extra_package["packages"]
    assert isinstance(packages, list)
    packages.append(
        {
            "name": "unreviewed-p3-package",
            "version": "1.0.0",
            "license_id": "test-license",
            "source_uri": "https://example.test/paddleocr3/",
        }
    )
    with pytest.raises(
        runtime_module.DocumentPreviewRuntimeError,
        match="document_preview_inventory_packages_invalid",
    ):
        runtime_module.verify_inventory_packages(
            runtime_module.parse_model_inventory(_canonical(extra_package)),
            package_version_reader=lambda _name: "test-v1",
        )


def test_paddleocr3_candidate_factory_cannot_select_the_legacy_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = runtime_module.parse_runtime_configuration(_canonical(_p3_configuration()))
    inventory = runtime_module.parse_model_inventory(_canonical(_p3_inventory()))
    captured: dict[str, object] = {}

    class FakePaddleOcr3:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def verify_native_runtime(self) -> None:
            captured["preflight"] = True

    monkeypatch.setattr(candidate_runner_module, "PaddleOcr3PdfPreviewAdapter", FakePaddleOcr3)
    profile = candidate_runner_module.EvaluationRuntimeProfileV1(
        document_preview_root=tmp_path,
        engine_kind="paddleocr",
        extractor_revision=configuration.extractor_revision,
        configuration_sha256="a" * 64,
        model_inventory_sha256="b" * 64,
        configuration=configuration,
        inventory=inventory,
    )

    adapter = candidate_runner_module.build_default_synthetic_pdf_candidate_adapter(profile)
    adapter.preflight_native_runtime()  # type: ignore[attr-defined]

    assert captured["extractor_revision"] == configuration.extractor_revision
    assert captured["model_root"] == tmp_path
    assert captured["options"] == configuration.options
    assert captured["preflight"] is True


def test_paddleocr3_inventory_builder_binds_the_paddlex_profile_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_path = root / "staging" / "configuration-paddleocr3.json"
    provenance_path = root / "staging" / "provenance-paddleocr3.json"
    configuration_path.parent.mkdir(parents=True)
    for directory in ("det", "rec"):
        path = root / "models" / "paddleocr3" / directory / "model.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(directory.encode("ascii"))
    configuration_path.write_bytes(_canonical(_p3_configuration()))
    provenance_path.write_bytes(
        _canonical(
            {
                "schema_version": "1.0",
                "engine_kind": "paddleocr",
                "runtime_api": _P3_RUNTIME_API,
                "packages": [
                    {
                        "name": name,
                        "license_id": "test-license",
                        "source_uri": "https://example.test/paddleocr3/",
                    }
                    for name in sorted(
                        runtime_module._required_packages("paddleocr", _P3_RUNTIME_API)
                    )
                ],
                "artifact_directories": [
                    {
                        "relative_directory": f"models/paddleocr3/{directory}",
                        "license_id": "test-license",
                        "source_uri": "https://example.test/paddleocr3/",
                    }
                    for directory in ("det", "rec")
                ],
            }
        )
    )
    monkeypatch.setattr(
        runtime_module, "_recorded_cv2_tree_matches_designated_provider", lambda: None
    )

    inventory, _configuration_sha256 = build_runtime_inventory(
        document_preview_root=root,
        configuration_path=configuration_path,
        provenance_path=provenance_path,
        package_version_reader=lambda name: runtime_module._PADDLEOCR3_PINNED_PACKAGE_VERSIONS[
            name
        ],
    )

    assert inventory["runtime_api"] == _P3_RUNTIME_API
    packages = {item["name"] for item in inventory["packages"]}
    assert packages == set(runtime_module._PADDLEOCR3_PINNED_PACKAGE_VERSIONS)


def test_paddleocr3_windows_uses_only_ascii_short_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    short_path_inputs: list[Path] = []

    class FakePaddleOcr:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    def _short_path(path: Path) -> str:
        short_path_inputs.append(path)
        return rf"C:\P3MODELS\{path.name}"

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "_windows_short_path", _short_path)

    _adapter(tmp_path)._paddle_engine()

    assert short_path_inputs == [
        (tmp_path / "models" / "paddleocr3" / "det").resolve(),
        (tmp_path / "models" / "paddleocr3" / "rec").resolve(),
    ]
    assert captured["text_detection_model_dir"] == r"C:\P3MODELS\det"
    assert captured["text_recognition_model_dir"] == r"C:\P3MODELS\rec"


def test_paddleocr3_windows_rejects_a_missing_ascii_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    class FakePaddleOcr:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal called
            called = True

    def _missing_short_path(_path: Path) -> str:
        raise DocumentPreviewUnavailable("document_preview_paddleocr3_native_runtime_unavailable")

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module, "_windows_short_path", _missing_short_path)

    with pytest.raises(
        DocumentPreviewUnavailable,
        match="document_preview_paddleocr3_native_runtime_unavailable",
    ):
        _adapter(tmp_path)._paddle_engine()

    assert not called


def test_paddleocr3_non_windows_uses_resolved_local_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOcr:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    def _windows_only(_path: Path) -> str:
        pytest.fail("non-Windows P3 profile must not request a short path")

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    monkeypatch.setattr(runtime_module, "_windows_short_path", _windows_only)

    _adapter(tmp_path)._paddle_engine()

    assert captured["text_detection_model_dir"] == str(
        (tmp_path / "models" / "paddleocr3" / "det").resolve()
    )
    assert captured["text_recognition_model_dir"] == str(
        (tmp_path / "models" / "paddleocr3" / "rec").resolve()
    )


def test_paddleocr3_constructor_cannot_open_a_network_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NetworkAttemptingPaddleOcr:
        def __init__(self, **_kwargs: object) -> None:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)

    module = ModuleType("paddleocr")
    module.PaddleOCR = NetworkAttemptingPaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    with pytest.raises(
        DocumentPreviewUnavailable,
        match="document_preview_paddleocr3_native_runtime_unavailable",
    ):
        _adapter(tmp_path)._paddle_engine()


def test_paddleocr3_blank_native_preflight_cannot_open_a_network_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NetworkAttemptingPaddleOcr:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def predict(self, _image: object, **_kwargs: object) -> object:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
            raise AssertionError("network guard must stop predict first")

    module = ModuleType("paddleocr")
    module.PaddleOCR = NetworkAttemptingPaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    _install_fake_numpy(monkeypatch)

    with pytest.raises(
        DocumentPreviewUnavailable,
        match="document_preview_paddleocr3_native_runtime_unavailable",
    ):
        _adapter(tmp_path).verify_native_runtime()


def test_paddleocr3_predict_uses_blank_native_preflight_and_normalizes_rec_polys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {"predictions": []}

    class FakePaddleOcr:
        def __init__(self, **kwargs: object) -> None:
            captured["constructor"] = kwargs

        def predict(self, image: object, **kwargs: object) -> list[dict[str, object]]:
            predictions = captured["predictions"]
            assert isinstance(predictions, list)
            predictions.append((image, kwargs))
            if len(predictions) == 1:
                return [{"rec_texts": [], "rec_polys": []}]
            return [
                {
                    "rec_texts": ["  invoice-17  "],
                    "rec_polys": [[[10, 20], [30, 20], [30, 40], [10, 40]]],
                }
            ]

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")

    _install_fake_numpy(monkeypatch)
    _install_one_page_pdfium(monkeypatch, captured)

    candidate = _adapter(tmp_path).extract_raster_or_image(_preview_input())

    constructor = captured["constructor"]
    assert constructor == {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_dir": str(
            (tmp_path / "models" / "paddleocr3" / "det").resolve()
        ),
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_dir": str(
            (tmp_path / "models" / "paddleocr3" / "rec").resolve()
        ),
        "device": "cpu",
        "cpu_threads": 1,
        "enable_hpi": False,
        "enable_mkldnn": False,
        "use_tensorrt": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    predictions = captured["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert captured["render_scale"] == 3
    assert getattr(predictions[0][0], "shape") == (64, 64, 3)
    assert predictions[0][1] == {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    span = candidate.pages[0].spans[0]
    assert span.text == "invoice-17"
    assert span.bbox is not None
    assert round(span.bbox.x, 8) == 0.1
    assert round(span.bbox.y, 8) == 0.2
    assert round(span.bbox.width, 8) == 0.2
    assert round(span.bbox.height, 8) == 0.2


def test_paddleocr3_malformed_predict_result_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePaddleOcr:
        def __init__(self, **_kwargs: object) -> None:
            self.calls = 0

        def predict(self, _image: object, **_kwargs: object) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [{"rec_texts": [], "rec_polys": []}]
            return [{"rec_texts": ["only-text"], "rec_polys": []}]

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    _install_fake_numpy(monkeypatch)
    _install_one_page_pdfium(monkeypatch)

    with pytest.raises(DocumentPreviewExecutionError, match="PaddleOCR 3 result is invalid"):
        _adapter(tmp_path).extract_raster_or_image(_preview_input())
