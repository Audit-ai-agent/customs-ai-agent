"""Offline direct PaddleOCR sidecar for the text-only raster profile.

The module is deliberately self-contained inside the isolated sidecar bundle.
It receives decoded RGB pages, never a PDF, and validates package/model
inventories before PaddleOCR import. A private diagnostic becomes only a
non-zero exit at the parent boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
from typing import BinaryIO, Iterator, Mapping, Sequence, cast


_PROTOCOL = "customs-raster-text-sidecar-v1"
_SESSION_PROTOCOL = "customs-raster-text-sidecar-v1-session"
# The evaluation protocol remains deliberately distinct from the long-lived
# service process.  A customer-side parent must never accidentally turn an
# evaluation child into a serving engine merely because both use the same
# request/result payload below.
_SERVICE_PROTOCOL = "customs-raster-text-sidecar-v1-service"
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_INPUT_BYTES = 128 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MODEL_HASH_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class RasterTextSidecarError(RuntimeError):
    """Private failure that is never serialized to the parent."""


@dataclass(frozen=True, slots=True)
class _Page:
    page_number: int
    width: int
    height: int
    pixels: bytes


@dataclass(frozen=True, slots=True)
class _Profile:
    raster_text_profile_revision: str
    extractor_revision: str
    sidecar_required_packages: tuple[str, ...]
    text_detection_model_name: str
    text_detection_model_directory: str
    text_recognition_model_name: str
    text_recognition_model_directory: str
    cpu_threads: int
    text_detection_limit_side_len: int
    text_detection_limit_type: str
    text_detection_threshold: float
    text_detection_box_threshold: float
    text_detection_unclip_ratio: float


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RasterTextSidecarError("raster_text_sidecar_environment_invalid")
    return value


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise RasterTextSidecarError("raster_text_sidecar_material_invalid")
        value = path.read_bytes()
    except RasterTextSidecarError:
        raise
    except OSError:
        raise RasterTextSidecarError("raster_text_sidecar_material_invalid") from None
    if not value or len(value) > limit:
        raise RasterTextSidecarError("raster_text_sidecar_material_invalid")
    return value


def _strict_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    return value


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    return value


def _profile_from_configuration(raw: bytes) -> _Profile:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid") from None
    if not isinstance(value, dict):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    required = {
        "schema_version",
        "engine_kind",
        "runtime_api",
        "extractor_revision",
        "raster_text_profile_revision",
        "text_detection_model_name",
        "text_detection_model_directory",
        "text_recognition_model_name",
        "text_recognition_model_directory",
        "cpu_threads",
        "text_detection_limit_side_len",
        "text_detection_limit_type",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
        "sidecar_required_packages",
    }
    if not required <= set(value):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    packages = value["sidecar_required_packages"]
    if (
        not isinstance(packages, list)
        or tuple(packages) != ("paddleocr", "paddlex", "paddlepaddle")
        or any(not isinstance(item, str) for item in packages)
    ):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    numeric = {
        "cpu_threads": value["cpu_threads"],
        "text_detection_limit_side_len": value["text_detection_limit_side_len"],
        "text_detection_threshold": value["text_detection_threshold"],
        "text_detection_box_threshold": value["text_detection_box_threshold"],
        "text_detection_unclip_ratio": value["text_detection_unclip_ratio"],
    }
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in numeric.values()):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    if (
        value.get("schema_version") != "1.0"
        or value.get("engine_kind") != "paddleocr3_direct_text"
        or value.get("runtime_api") != "paddleocr3-paddlex-3.2"
        or value.get("text_detection_model_name") != "PP-OCRv5_server_det"
        or value.get("text_recognition_model_name") != "en_PP-OCRv5_mobile_rec"
        or int(numeric["cpu_threads"]) not in {1, 2, 4, 6}
        or int(numeric["text_detection_limit_side_len"]) not in {1536, 2048, 2526}
        or value.get("text_detection_limit_type") != "max"
        or float(numeric["text_detection_threshold"]) != 0.3
        or float(numeric["text_detection_box_threshold"]) not in {0.4, 0.6}
        or float(numeric["text_detection_unclip_ratio"]) not in {1.5, 2.0}
    ):
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    revision = _required_string(value["raster_text_profile_revision"])
    if not revision.startswith("synthetic-raster-text-v1:") or len(revision) != 49:
        raise RasterTextSidecarError("raster_text_sidecar_configuration_invalid")
    return _Profile(
        raster_text_profile_revision=revision,
        extractor_revision=_required_string(value["extractor_revision"]),
        sidecar_required_packages=tuple(packages),
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_directory=_required_string(value["text_detection_model_directory"]),
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_directory=_required_string(value["text_recognition_model_directory"]),
        cpu_threads=int(numeric["cpu_threads"]),
        text_detection_limit_side_len=int(numeric["text_detection_limit_side_len"]),
        text_detection_limit_type="max",
        text_detection_threshold=float(numeric["text_detection_threshold"]),
        text_detection_box_threshold=float(numeric["text_detection_box_threshold"]),
        text_detection_unclip_ratio=float(numeric["text_detection_unclip_ratio"]),
    )


def _verify_package_inventory(raw: bytes, *, expected_sha256: str, required: Sequence[str]) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RasterTextSidecarError("raster_text_sidecar_package_drift")
    try:
        packages = json.loads(raw)["packages"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid") from None
    if not isinstance(packages, list):
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid")
    observed: dict[str, str] = {}
    for item in packages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not item["name"]
            or not item["version"]
            or item["name"] in observed
        ):
            raise RasterTextSidecarError("raster_text_sidecar_package_invalid")
        observed[item["name"]] = item["version"]
    for package in required:
        try:
            if importlib.metadata.version(package) != observed.get(package):
                raise RasterTextSidecarError("raster_text_sidecar_package_drift")
        except importlib.metadata.PackageNotFoundError:
            raise RasterTextSidecarError("raster_text_sidecar_package_drift") from None


def _sha256_model_artifact(path: Path) -> str:
    """Hash one declared model file without copying a whole weight into RAM."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_MODEL_HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        raise RasterTextSidecarError("raster_text_sidecar_model_drift") from None


def _verify_model_inventory(raw: bytes, *, expected_sha256: str, model_root: Path) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RasterTextSidecarError("raster_text_sidecar_model_drift")
    try:
        artifacts = json.loads(raw)["artifacts"]
        root = model_root.resolve(strict=True)
    except (KeyError, TypeError, OSError, json.JSONDecodeError):
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None
    if model_root.is_symlink() or not root.is_dir() or not isinstance(artifacts, list) or not artifacts:
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid")
    expected: set[str] = set()
    for item in artifacts:
        relative = item.get("relative_path") if isinstance(item, dict) else None
        digest = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not isinstance(digest, str) or relative in expected:
            raise RasterTextSidecarError("raster_text_sidecar_model_invalid")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError:
            raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or _sha256_model_artifact(candidate) != digest
        ):
            raise RasterTextSidecarError("raster_text_sidecar_model_drift")
        expected.add(relative.replace("\\", "/"))
    actual: set[str] = set()
    try:
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise RasterTextSidecarError("raster_text_sidecar_model_invalid")
            if candidate.is_file():
                actual.add(candidate.relative_to(root).as_posix())
            elif not candidate.is_dir():
                raise RasterTextSidecarError("raster_text_sidecar_model_invalid")
    except (OSError, ValueError):
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None
    if actual != expected:
        raise RasterTextSidecarError("raster_text_sidecar_model_drift")


def configuration_from_environment() -> tuple[_Profile, Path]:
    """Verify child-owned package/model material before Paddle imports."""

    configuration_path = Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_CONFIG_PATH"))
    configuration_sha256 = _strict_sha256(
        _required_environment("CUSTOMS_AGENT_RASTER_TEXT_CONFIG_SHA256")
    )
    configuration_raw = _read_regular_file(configuration_path, limit=_MAX_HEADER_BYTES)
    if hashlib.sha256(configuration_raw).hexdigest() != configuration_sha256:
        raise RasterTextSidecarError("raster_text_sidecar_configuration_drift")
    profile = _profile_from_configuration(configuration_raw)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _strict_sha256(
        _required_environment("CUSTOMS_AGENT_RASTER_TEXT_SIDECAR_MODULE_SHA256")
    ):
        raise RasterTextSidecarError("raster_text_sidecar_module_drift")
    try:
        required = json.loads(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_REQUIRED_PACKAGES"))
    except json.JSONDecodeError:
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid") from None
    if not isinstance(required, list) or tuple(required) != profile.sidecar_required_packages:
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid")
    _verify_package_inventory(
        _read_regular_file(
            Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_PATH")),
            limit=_MAX_HEADER_BYTES,
        ),
        expected_sha256=_strict_sha256(
            _required_environment("CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_SHA256")
        ),
        required=profile.sidecar_required_packages,
    )
    model_root = Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_ROOT"))
    _verify_model_inventory(
        _read_regular_file(
            Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_PATH")),
            limit=_MAX_HEADER_BYTES,
        ),
        expected_sha256=_strict_sha256(
            _required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_SHA256")
        ),
        model_root=model_root,
    )
    return profile, model_root


@contextmanager
def deny_network_connections() -> Iterator[None]:
    """Reject network access before importing Paddle/PaddleX."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise RasterTextSidecarError("raster_text_sidecar_network_forbidden")

    # ``socket.socket`` must remain a type while Paddle imports its HTTP/TLS
    # stack: replacing it with a function makes ordinary ``isinstance`` checks
    # fail before any connection is attempted.  Keep construction compatible,
    # but deny every operation that can emit a packet or establish a peer.
    class _BlockedSocket(original_socket):  # type: ignore[misc,valid-type]
        def connect(self, *_args: object, **_kwargs: object) -> None:
            _blocked()

        def connect_ex(self, *_args: object, **_kwargs: object) -> int:
            _blocked()
            raise AssertionError("unreachable")

        def sendto(self, *_args: object, **_kwargs: object) -> int:
            _blocked()
            raise AssertionError("unreachable")

    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]
    socket.gethostbyname_ex = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment,misc]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = original_gethostbyname  # type: ignore[assignment]
        socket.gethostbyname_ex = original_gethostbyname_ex  # type: ignore[assignment]


def _read_request(stream: BinaryIO) -> tuple[str, str, tuple[_Page, ...]]:
    header = stream.readline(_MAX_HEADER_BYTES + 1)
    if not header or len(header) > _MAX_HEADER_BYTES or not header.endswith(b"\n"):
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    try:
        value = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid") from None
    if not isinstance(value, dict) or value.get("protocol") != _PROTOCOL:
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    source_sha256 = value.get("source_sha256")
    revision = value.get("raster_text_profile_revision")
    page_values = value.get("pages")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in _HEX for character in source_sha256)
        or not isinstance(revision, str)
        or not isinstance(page_values, list)
        or not 1 <= len(page_values) <= 20
    ):
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    dimensions: list[tuple[int, int, int]] = []
    for index, item in enumerate(page_values, start=1):
        if not isinstance(item, dict):
            raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
        page_number, width, height = item.get("page_number"), item.get("width"), item.get("height")
        if (
            page_number != index
            or isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width < 1
            or height < 1
            or width * height > 40_000_000
        ):
            raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
        dimensions.append((page_number, width, height))
    size = sum(width * height * 3 for _page, width, height in dimensions)
    if size > _MAX_INPUT_BYTES:
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    pixels = stream.read(size)
    if len(pixels) != size or stream.read(1):
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    offset = 0
    pages: list[_Page] = []
    for page_number, width, height in dimensions:
        length = width * height * 3
        pages.append(_Page(page_number, width, height, pixels[offset : offset + length]))
        offset += length
    return source_sha256, revision, tuple(pages)


def _native_model_directory(*, model_root: Path, relative_directory: str) -> str:
    try:
        root = model_root.resolve(strict=True)
        directory = (root / relative_directory).resolve(strict=True)
        directory.relative_to(root)
        if directory.is_symlink() or not directory.is_dir():
            raise OSError
    except (OSError, ValueError):
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None
    if sys.platform != "win32":
        return str(directory)
    try:
        import ctypes

        required = ctypes.windll.kernel32.GetShortPathNameW(str(directory), None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = ctypes.windll.kernel32.GetShortPathNameW(str(directory), buffer, len(buffer))
        if not isinstance(written, int) or written < 1 or written >= len(buffer):
            raise OSError
        if not buffer.value or not buffer.value.isascii():
            raise OSError
        return buffer.value
    except Exception:
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None


def _engine(profile: _Profile, model_root: Path) -> object:
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]

        return PaddleOCR(
            text_detection_model_name=profile.text_detection_model_name,
            text_detection_model_dir=_native_model_directory(
                model_root=model_root, relative_directory=profile.text_detection_model_directory
            ),
            text_recognition_model_name=profile.text_recognition_model_name,
            text_recognition_model_dir=_native_model_directory(
                model_root=model_root, relative_directory=profile.text_recognition_model_directory
            ),
            device="cpu",
            cpu_threads=profile.cpu_threads,
            enable_hpi=False,
            enable_mkldnn=False,
            use_tensorrt=False,
            text_det_limit_side_len=profile.text_detection_limit_side_len,
            text_det_limit_type=profile.text_detection_limit_type,
            text_det_thresh=profile.text_detection_threshold,
            text_det_box_thresh=profile.text_detection_box_threshold,
            text_det_unclip_ratio=profile.text_detection_unclip_ratio,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception:
        raise RasterTextSidecarError("raster_text_sidecar_runtime_unavailable") from None


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    try:
        converted = tolist()
    except Exception:
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid") from None
    if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes, bytearray)):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    return list(converted)


def _polygon_geometry(value: object) -> tuple[float, float, float] | None:
    """Return visual centre-y, left-x and height from one OCR polygon."""

    try:
        points = _sequence(value)
        coordinates: list[tuple[float, float]] = []
        for point in points:
            pair = _sequence(point)
            if len(pair) != 2 or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in pair
            ):
                return None
            x, y = float(cast(int | float, pair[0])), float(cast(int | float, pair[1]))
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            coordinates.append((x, y))
        if len(coordinates) < 3:
            return None
        top, bottom = min(y for _x, y in coordinates), max(y for _x, y in coordinates)
        height = bottom - top
        if height <= 0:
            return None
        return ((top + bottom) / 2, min(x for x, _y in coordinates), height)
    except RasterTextSidecarError:
        return None


def _prediction_lines(prediction: object) -> list[tuple[str, float]]:
    if isinstance(prediction, Mapping) or isinstance(prediction, (str, bytes, bytearray)):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    try:
        iterator = iter(prediction)  # type: ignore[call-overload]
        result = next(iterator)
    except (TypeError, StopIteration):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid") from None
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    if not isinstance(result, Mapping):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    texts = _sequence(result.get("rec_texts"))
    scores = _sequence(result.get("rec_scores"))
    if len(texts) != len(scores):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    polygons = _sequence(result.get("rec_polys")) if result.get("rec_polys") is not None else []
    if polygons and len(polygons) != len(texts):
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    # Paddle's raw sequence can reflect detector batching rather than the
    # document's visual order.  Mirror the source-PDFium Gold's row banding:
    # nearby vertical centres form a row (0.6×maximum line height), then the
    # row reads left-to-right.  Provider ordinal resolves exact geometric
    # ties; geometry-free output follows positioned text in provider order.
    lines: list[tuple[str, float, float | None, float | None, float | None, int]] = []
    for ordinal, (text_value, score_value) in enumerate(zip(texts, scores, strict=True)):
        if (
            not isinstance(text_value, str)
            or isinstance(score_value, bool)
            or not isinstance(score_value, (int, float))
            or not math.isfinite(float(score_value))
            or not 0 <= float(score_value) <= 1
        ):
            raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
        text = text_value.strip()
        if text:
            geometry = _polygon_geometry(polygons[ordinal]) if polygons else None
            lines.append(
                (
                    text[:8_000],
                    float(score_value),
                    None if geometry is None else geometry[0],
                    None if geometry is None else geometry[1],
                    None if geometry is None else geometry[2],
                    ordinal,
                )
            )
    positioned: list[tuple[str, float, float, float, float, int]] = []
    unpositioned: list[tuple[str, float, float | None, float | None, float | None, int]] = []
    for item in lines:
        text, score, center, x, height, ordinal = item
        if center is None or x is None or height is None:
            unpositioned.append(item)
        else:
            positioned.append((text, score, center, x, height, ordinal))
    positioned.sort(key=lambda item: (item[2], item[3], item[5]))
    rows: list[list[tuple[str, float, float, float, float, int]]] = []
    row_center: list[float] = []
    row_height: list[float] = []
    for item in positioned:
        center, height = item[2], item[4]
        if rows and abs(center - row_center[-1]) <= 0.6 * max(height, row_height[-1]):
            rows[-1].append(item)
            row_center[-1] = sum(entry[2] for entry in rows[-1]) / len(rows[-1])
            row_height[-1] = sum(entry[4] for entry in rows[-1]) / len(rows[-1])
        else:
            rows.append([item])
            row_center.append(center)
            row_height.append(height)
    ordered = [
        (text, score)
        for row in rows
        for text, score, _center, _x, _height, _ordinal in sorted(
            row, key=lambda item: (item[3], item[2], item[5])
        )
    ]
    ordered.extend((text, score) for text, score, *_rest in sorted(unpositioned, key=lambda item: item[5]))
    return ordered


def _predict_pages(engine: object, pages: Sequence[_Page]) -> list[dict[str, object]]:
    try:
        import numpy as np  # type: ignore[import-not-found]

        output: list[dict[str, object]] = []
        for page in pages:
            image = np.frombuffer(page.pixels, dtype=np.uint8).reshape((page.height, page.width, 3))
            prediction = engine.predict(  # type: ignore[attr-defined]
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            lines = _prediction_lines(prediction)
            if lines:
                output.append(
                    {
                        "page_number": page.page_number,
                        "text": "\n".join(line for line, _score in lines),
                        # Conservative deterministic page confidence: a weak
                        # identifier makes the entire editable page reviewable.
                        "confidence": min(score for _line, score in lines),
                    }
                )
        return output
    except RasterTextSidecarError:
        raise
    except Exception:
        raise RasterTextSidecarError("raster_text_sidecar_execution_failed") from None


def run_once_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    profile, model_root = configuration_from_environment()
    source_sha256, revision, pages = _read_request(stream)
    if revision != profile.raster_text_profile_revision:
        raise RasterTextSidecarError("raster_text_sidecar_binding_invalid")
    with deny_network_connections():
        engine = _engine(profile, model_root)
        text_pages = _predict_pages(engine, pages)
    encoded = _response_bytes(profile=profile, source_sha256=source_sha256, revision=revision, text_pages=text_pages)
    output.write(encoded)


def _response_bytes(*, profile: _Profile, source_sha256: str, revision: str, text_pages: list[dict[str, object]]) -> bytes:
    """Serialize one bounded response for both one-shot and persistent modes."""

    encoded = json.dumps(
        {
            "protocol": _PROTOCOL,
            "source_sha256": source_sha256,
            "raster_text_profile_revision": revision,
            "provider_revision": profile.extractor_revision,
            "pages": text_pages,
            "warnings": [],
            "needs_review_reasons": [],
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    return encoded


def _service_ready_bytes(*, profile: _Profile) -> bytes:
    """Return the bounded, content-free service-start acknowledgement.

    It is emitted only after the child has independently verified its local
    material and constructed the direct OCR engine.  No source identifier,
    document bytes, OCR result, or private diagnostic belongs in this frame.
    """

    encoded = json.dumps(
        {
            "protocol": _SERVICE_PROTOCOL,
            "status": "ready",
            "raster_text_profile_revision": profile.raster_text_profile_revision,
            "provider_revision": profile.extractor_revision,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    return encoded


def _evaluation_session_ready_bytes(*, profile: _Profile) -> bytes:
    """Return the distinct, content-free evaluation-session ready frame.

    Candidate evaluation must not charge Paddle import/model construction to
    the first sealed asset.  This acknowledgement is emitted only after the
    same offline configuration/package/model checks and engine load complete,
    but it deliberately cannot be confused with the customer service
    protocol.
    """

    encoded = json.dumps(
        {
            "protocol": _SESSION_PROTOCOL,
            "status": "ready",
            "raster_text_profile_revision": profile.raster_text_profile_revision,
            "provider_revision": profile.extractor_revision,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise RasterTextSidecarError("raster_text_sidecar_output_invalid")
    return encoded


def _read_framed_request(stream: BinaryIO) -> bytes | None:
    """Read exactly one bounded request frame, or clean EOF between frames."""

    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    length = struct.unpack(">Q", header)[0]
    if not 1 <= length <= _MAX_INPUT_BYTES:
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    payload = stream.read(length)
    if len(payload) != length:
        raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
    return payload


def _verify_service_materials(*, profile: _Profile, model_root: Path) -> None:
    """Reject a changed local binding before or after an in-memory prediction.

    A warm process keeps Paddle objects in memory, so it cannot rely on a
    process start to discover altered configuration, module, package, or
    model bytes.  The serving security contract deliberately trades bounded
    local I/O for fail-closed drift detection: every request re-hashes each
    declared model file and rejects undeclared extras both before and after
    prediction.  The parent owns the 30-second request deadline, so a slow
    full inventory check becomes unavailable rather than allowing a stale
    in-memory model to produce a result.  This function performs no import,
    network operation, or model fallback.
    """

    configuration_path = Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_CONFIG_PATH"))
    configuration_sha256 = _strict_sha256(
        _required_environment("CUSTOMS_AGENT_RASTER_TEXT_CONFIG_SHA256")
    )
    configuration_raw = _read_regular_file(configuration_path, limit=_MAX_HEADER_BYTES)
    if hashlib.sha256(configuration_raw).hexdigest() != configuration_sha256:
        raise RasterTextSidecarError("raster_text_sidecar_configuration_drift")
    current_profile = _profile_from_configuration(configuration_raw)
    if current_profile != profile:
        raise RasterTextSidecarError("raster_text_sidecar_binding_invalid")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _strict_sha256(
        _required_environment("CUSTOMS_AGENT_RASTER_TEXT_SIDECAR_MODULE_SHA256")
    ):
        raise RasterTextSidecarError("raster_text_sidecar_module_drift")
    try:
        required = json.loads(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_REQUIRED_PACKAGES"))
    except json.JSONDecodeError:
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid") from None
    if not isinstance(required, list) or tuple(required) != profile.sidecar_required_packages:
        raise RasterTextSidecarError("raster_text_sidecar_package_invalid")
    _verify_package_inventory(
        _read_regular_file(
            Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_PATH")),
            limit=_MAX_HEADER_BYTES,
        ),
        expected_sha256=_strict_sha256(
            _required_environment("CUSTOMS_AGENT_RASTER_TEXT_PACKAGE_INVENTORY_SHA256")
        ),
        required=profile.sidecar_required_packages,
    )
    current_model_root = Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_ROOT"))
    try:
        current_root = current_model_root.resolve(strict=True)
        expected_root = model_root.resolve(strict=True)
        if (
            current_model_root.is_symlink()
            or model_root.is_symlink()
            or not current_root.is_dir()
            or current_root != expected_root
        ):
            raise OSError
        for relative_directory in (
            profile.text_detection_model_directory,
            profile.text_recognition_model_directory,
        ):
            candidate = (current_root / relative_directory).resolve(strict=True)
            candidate.relative_to(current_root)
            if candidate.is_symlink() or not candidate.is_dir():
                raise OSError
    except (OSError, ValueError):
        raise RasterTextSidecarError("raster_text_sidecar_model_invalid") from None
    model_inventory = _read_regular_file(
        Path(_required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_PATH")),
        limit=_MAX_HEADER_BYTES,
    )
    _verify_model_inventory(
        raw=model_inventory,
        expected_sha256=_strict_sha256(
            _required_environment("CUSTOMS_AGENT_RASTER_TEXT_MODEL_INVENTORY_SHA256")
        ),
        # Use the just-validated non-symlink path, not the startup argument,
        # so this complete tree walk binds the same live root checked above.
        model_root=current_model_root,
    )


def run_session_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Serve framed local evaluation requests after exactly one model load.

    This mode is only used by the synthetic activation runner.  It shares the
    exact decoder/configuration/model path with customer one-shot OCR, but
    avoids 3,600 cold Paddle imports during a sealed evaluation.  Its ready
    frame separates bounded setup from the first candidate request.  Frames
    are strictly bounded and no child diagnostics are serialized.
    """

    profile, model_root = configuration_from_environment()
    with deny_network_connections():
        engine = _engine(profile, model_root)
        # Engine construction can take long enough for local material to
        # change after the initial startup check.  Revalidate the complete
        # config/package/model binding before publishing a ready frame, so a
        # parent never routes a request to a cold-loaded stale engine.
        _verify_service_materials(profile=profile, model_root=model_root)
        ready = _evaluation_session_ready_bytes(profile=profile)
        output.write(struct.pack(">Q", len(ready)))
        output.write(ready)
        output.flush()
        while True:
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
            length = struct.unpack(">Q", header)[0]
            if not 1 <= length <= _MAX_INPUT_BYTES:
                raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
            payload = stream.read(length)
            if len(payload) != length:
                raise RasterTextSidecarError("raster_text_sidecar_input_invalid")
            # The persistent evaluator retains native model objects, so the
            # child independently validates config/package/direct-model
            # material around every candidate.  A parent lightweight check
            # complements this without rehashing its interpreter tree for
            # every asset.
            _verify_service_materials(profile=profile, model_root=model_root)
            source_sha256, revision, pages = _read_request(io.BytesIO(payload))
            if revision != profile.raster_text_profile_revision:
                raise RasterTextSidecarError("raster_text_sidecar_binding_invalid")
            text_pages = _predict_pages(engine, pages)
            _verify_service_materials(profile=profile, model_root=model_root)
            encoded = _response_bytes(
                profile=profile,
                source_sha256=source_sha256,
                revision=revision,
                text_pages=text_pages,
            )
            output.write(struct.pack(">Q", len(encoded)))
            output.write(encoded)
            output.flush()


def run_service_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Serve one sequential, profile-bound direct OCR service child.

    Unlike the evaluation-only session, this protocol first publishes a small
    ready frame after its complete offline validation and model load.  Each
    later frame is independently revalidated before and after prediction, so
    a changed local bundle can never yield a stale OCR response.  Errors are
    intentionally represented by process failure only; the parent owns the
    content-free public error mapping.
    """

    profile, model_root = configuration_from_environment()
    with deny_network_connections():
        engine = _engine(profile, model_root)
        # See evaluation-session startup above.  A ready acknowledgement is
        # valid only after the post-engine full material verification too.
        _verify_service_materials(profile=profile, model_root=model_root)
        ready = _service_ready_bytes(profile=profile)
        output.write(struct.pack(">Q", len(ready)))
        output.write(ready)
        output.flush()
        while True:
            payload = _read_framed_request(stream)
            if payload is None:
                return
            # The profile/model verification deliberately happens around the
            # native call.  A write racing either edge makes the child exit
            # without serializing a possibly stale document result.
            _verify_service_materials(profile=profile, model_root=model_root)
            source_sha256, revision, pages = _read_request(io.BytesIO(payload))
            if revision != profile.raster_text_profile_revision:
                raise RasterTextSidecarError("raster_text_sidecar_binding_invalid")
            text_pages = _predict_pages(engine, pages)
            _verify_service_materials(profile=profile, model_root=model_root)
            encoded = _response_bytes(
                profile=profile,
                source_sha256=source_sha256,
                revision=revision,
                text_pages=text_pages,
            )
            output.write(struct.pack(">Q", len(encoded)))
            output.write(encoded)
            output.flush()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == [f"--{_PROTOCOL}"]:
            run_once_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_SESSION_PROTOCOL}"]:
            run_session_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_SERVICE_PROTOCOL}"]:
            run_service_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        else:
            return 2
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RasterTextSidecarError",
    "configuration_from_environment",
    "deny_network_connections",
    "main",
    "run_service_from_stdio",
    "run_session_from_stdio",
]
