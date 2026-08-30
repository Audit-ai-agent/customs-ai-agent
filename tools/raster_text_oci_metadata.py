"""Build and validate deterministic OCR runtime component metadata offline.

The module is intentionally standard-library-only.  Component identities come
from the already hash-bound runtime profile and package/model inventories.
License data comes only from installed distribution metadata, an exact CPython
license file, an attested public-model inventory, or a checked-in hash-bound
fine-tuned-model policy.  Missing or ambiguous evidence is never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import importlib.metadata
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,255}$")
_DNS_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_LICENSE_REF = re.compile(r"^LicenseRef-[A-Za-z0-9][A-Za-z0-9.-]*$")
_KNOWN_SPDX_IDS = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "GPL-3.0-or-later",
        "HPND",
        "MIT",
        "MIT-CMU",
        "MPL-2.0",
        "PSF-2.0",
    }
)
_KNOWN_SPDX_EXCEPTIONS = frozenset({"GCC-exception-3.1"})
_POLICY_RELATIVE_PATH = PurePosixPath("_deployment/runtime-license-policy.json")
_PARENT_EVIDENCE_RELATIVE_PATH = PurePosixPath("_deployment/parent-package-evidence.json")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_LICENSE_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_LICENSE_BYTES = 16 * 1024 * 1024
_MAX_LICENSE_FILES = 64
_MODEL_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_MODEL_SCOPE = "model"
_PACKAGE_SCOPE = "python-package"
_RUNTIME_SCOPE = "runtime"


class RuntimeMetadataError(RuntimeError):
    """A content-free runtime metadata failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ActualComponent:
    name: str
    version: str
    scope: str
    checksum_sha256: str | None

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.scope, _normalized_name(self.name), self.version)


@dataclass(frozen=True, slots=True)
class ExtractedLicense:
    license_id: str
    extracted_text: str


@dataclass(frozen=True, slots=True)
class LicensedComponent:
    actual: ActualComponent
    license_id: str
    source_uri: str
    extracted_licenses: tuple[ExtractedLicense, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeMetadataArtifacts:
    license_manifest: bytes
    sbom: bytes
    source_evidence_paths: tuple[PurePosixPath, ...]
    generated_evidence: tuple[tuple[PurePosixPath, bytes], ...]
    policy: bytes


@dataclass(frozen=True, slots=True)
class DistributionEvidence:
    name: str
    version: str
    metadata_sha256: str
    license_expressions: tuple[str, ...]
    legacy_licenses: tuple[str, ...]
    project_urls: tuple[str, ...]
    home_pages: tuple[str, ...]
    sbom_license_evidence: tuple[tuple[str, str, str], ...]
    license_files: tuple[tuple[str, str], ...]


DistributionFinder = Callable[[str], Sequence[importlib.metadata.Distribution]]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _safe_relative(value: object, *, code: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise RuntimeMetadataError(code)
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative == PurePosixPath(".")
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise RuntimeMetadataError(code)
    return relative


def _safe_file(root: Path, relative: PurePosixPath, *, code: str) -> Path:
    candidate = root
    try:
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise OSError
        return resolved
    except OSError:
        raise RuntimeMetadataError(code) from None


def _safe_directory(root: Path, relative: PurePosixPath, *, code: str) -> Path:
    candidate = root
    try:
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not resolved.is_dir():
            raise OSError
        return resolved
    except OSError:
        raise RuntimeMetadataError(code) from None


def _read_bytes(path: Path, *, maximum: int, code: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise OSError
        return path.read_bytes()
    except OSError:
        raise RuntimeMetadataError(code) from None


def _read_json(
    root: Path,
    relative: PurePosixPath,
    *,
    expected_sha256: object | None = None,
    code: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_bytes(_safe_file(root, relative, code=code), maximum=_MAX_JSON_BYTES, code=code)
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeMetadataError(code) from None
    if (
        not isinstance(value, dict)
        or raw not in {canonical_json_bytes(value), canonical_json_bytes(value) + b"\n"}
        or (
            expected_sha256 is not None
            and (
                not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
                or hashlib.sha256(raw).hexdigest() != expected_sha256
            )
        )
    ):
        raise RuntimeMetadataError(code)
    return value, raw


def _valid_source_uri(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or len(value) > 2048
        or "\\" in value
        or any(ord(character) <= 0x20 or ord(character) >= 0x7F for character in value)
        or re.fullmatch(r"https://[A-Za-z0-9._~:/?@!$&'()*+,;=%-]+", value) is None
        or re.search(r"%(?![0-9A-F]{2})", value) is not None
    ):
        raise RuntimeMetadataError("runtime_component_source_uri_invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname.casefold() if parsed.hostname else ""
        parsed.port
    except ValueError:
        raise RuntimeMetadataError("runtime_component_source_uri_invalid") from None
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise RuntimeMetadataError("runtime_component_source_uri_invalid")
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.geturl() != value
        or parsed.netloc != hostname
        or _DNS_HOSTNAME.fullmatch(hostname) is None
        or parsed.username is not None
        or parsed.password is not None
        or hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith((".example", ".invalid", ".local", ".localhost", ".test"))
        or parsed.fragment
    ):
        raise RuntimeMetadataError("runtime_component_source_uri_invalid")
    return value


def _valid_license_expression(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeMetadataError("runtime_component_license_invalid")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > 512:
        raise RuntimeMetadataError("runtime_component_license_invalid")
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*|[()]", normalized)
    if "".join(tokens) != re.sub(r"\s+", "", normalized):
        raise RuntimeMetadataError("runtime_component_license_invalid")
    state = "operand"
    depth = 0
    can_have_exception = False
    for token in tokens:
        if state == "operand":
            if token == "(":
                depth += 1
                can_have_exception = False
            elif token in _KNOWN_SPDX_IDS or _LICENSE_REF.fullmatch(token) is not None:
                state = "operator"
                can_have_exception = True
            else:
                raise RuntimeMetadataError("runtime_component_license_invalid")
        elif state == "exception":
            if token not in _KNOWN_SPDX_EXCEPTIONS:
                raise RuntimeMetadataError("runtime_component_license_invalid")
            state = "operator"
            can_have_exception = False
        elif token == "WITH" and can_have_exception:
            state = "exception"
        elif token in {"AND", "OR"}:
            state = "operand"
            can_have_exception = False
        elif token == ")" and depth:
            depth -= 1
            can_have_exception = False
        else:
            raise RuntimeMetadataError("runtime_component_license_invalid")
    if state != "operator" or depth:
        raise RuntimeMetadataError("runtime_component_license_invalid")
    return normalized


def _valid_license_declaration(value: object, *, allow_noassertion: bool) -> str:
    if allow_noassertion and value == "NOASSERTION":
        return "NOASSERTION"
    return _valid_license_expression(value)


def _valid_component_source(value: object, *, allow_noassertion: bool) -> str:
    if allow_noassertion and value == "NOASSERTION":
        return "NOASSERTION"
    return _valid_source_uri(value)


def _parse_package_inventory(
    root: Path, *, relative_path: object, expected_sha256: object
) -> tuple[tuple[str, str], ...]:
    relative = _safe_relative(relative_path, code="runtime_package_inventory_invalid")
    value, _raw = _read_json(
        root,
        relative,
        expected_sha256=expected_sha256,
        code="runtime_package_inventory_invalid",
    )
    packages = value.get("packages")
    if (
        not set(value).issubset({"inventory_kind", "packages", "schema_version"})
        or ("schema_version" in value and value["schema_version"] != "1.0")
        or ("inventory_kind" in value and not isinstance(value["inventory_kind"], str))
        or not isinstance(packages, list)
        or not packages
    ):
        raise RuntimeMetadataError("runtime_package_inventory_invalid")
    result: list[tuple[str, str]] = []
    observed: set[tuple[str, str]] = set()
    for package in packages:
        if (
            not isinstance(package, dict)
            or set(package) != {"name", "version"}
            or not isinstance(package.get("name"), str)
            or not isinstance(package.get("version"), str)
            or _SAFE_NAME.fullmatch(package["name"]) is None
            or not package["version"].strip()
        ):
            raise RuntimeMetadataError("runtime_package_inventory_invalid")
        identity = (_normalized_name(package["name"]), package["version"])
        if identity in observed:
            raise RuntimeMetadataError("runtime_package_inventory_invalid")
        observed.add(identity)
        result.append((package["name"], package["version"]))
    return tuple(result)


def _model_components(
    root: Path, profile: Mapping[str, Any]
) -> tuple[tuple[ActualComponent, ActualComponent], PurePosixPath]:
    model_root = _safe_relative(
        profile.get("mobile_model_bundle_root_relative_path"),
        code="runtime_model_inventory_invalid",
    )
    manifest, manifest_raw = _read_json(
        root,
        model_root / "manifest.json",
        expected_sha256=profile.get("mobile_model_bundle_manifest_sha256"),
        code="runtime_model_inventory_invalid",
    )
    inventory_relative = _safe_relative(
        manifest.get("model_inventory_relative_path"), code="runtime_model_inventory_invalid"
    )
    profile_inventory = _safe_relative(
        profile.get("model_inventory_path"), code="runtime_model_inventory_invalid"
    )
    if profile_inventory != model_root / inventory_relative:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    inventory, _raw = _read_json(
        root,
        profile_inventory,
        expected_sha256=profile.get("model_inventory_sha256"),
        code="runtime_model_inventory_invalid",
    )
    detector_name = inventory.get("text_detection_model_name")
    recognizer_name = inventory.get("text_recognition_model_name")
    detector_snapshot = manifest.get("source_detector_model_snapshot_sha256")
    recognizer_snapshot = manifest.get("source_recognizer_model_snapshot_sha256")
    artifacts = inventory.get("artifacts")
    if (
        not isinstance(detector_name, str)
        or not isinstance(recognizer_name, str)
        or _SAFE_NAME.fullmatch(detector_name) is None
        or _SAFE_NAME.fullmatch(recognizer_name) is None
        or not isinstance(detector_snapshot, str)
        or _SHA256.fullmatch(detector_snapshot) is None
        or not isinstance(recognizer_snapshot, str)
        or _SHA256.fullmatch(recognizer_snapshot) is None
        or not isinstance(artifacts, list)
    ):
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    records: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"byte_size", "relative_path", "sha256"}
            or not isinstance(artifact.get("relative_path"), str)
            or not isinstance(artifact.get("sha256"), str)
            or _SHA256.fullmatch(artifact["sha256"]) is None
            or isinstance(artifact.get("byte_size"), bool)
            or not isinstance(artifact.get("byte_size"), int)
            or artifact["byte_size"] < 1
        ):
            raise RuntimeMetadataError("runtime_model_inventory_invalid")
        path = _safe_relative(artifact["relative_path"], code="runtime_model_inventory_invalid")
        if path.as_posix() in records or path.parts[0] not in {"text-det", "text-rec"}:
            raise RuntimeMetadataError("runtime_model_inventory_invalid")
        records[path.as_posix()] = artifact
    expected_paths = {
        f"{prefix}/{name}" for prefix in ("text-det", "text-rec") for name in _MODEL_FILENAMES
    }
    if set(records) != expected_paths:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")

    def snapshot(prefix: str) -> str:
        rows: list[dict[str, object]] = []
        for name in _MODEL_FILENAMES:
            relative = f"{prefix}/{name}"
            record = records[relative]
            rows.append(
                {
                    "byte_size": record["byte_size"],
                    "relative_path": (f"models/{relative}" if prefix == "text-det" else relative),
                    "sha256": record["sha256"],
                }
            )
        return hashlib.sha256(canonical_json_bytes({"artifacts": rows})).hexdigest()

    if snapshot("text-det") != detector_snapshot or snapshot("text-rec") != recognizer_snapshot:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    components = (
        ActualComponent(
            detector_name, f"sha256:{detector_snapshot}", _MODEL_SCOPE, detector_snapshot
        ),
        ActualComponent(
            recognizer_name,
            f"sha256:{recognizer_snapshot}",
            _MODEL_SCOPE,
            recognizer_snapshot,
        ),
    )
    if hashlib.sha256(manifest_raw).hexdigest() != profile.get(
        "mobile_model_bundle_manifest_sha256"
    ):
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    return components, model_root


def _python_component(root: Path, profile: Mapping[str, Any]) -> tuple[ActualComponent, Path, str]:
    interpreter_root_relative = _safe_relative(
        profile.get("sidecar_interpreter_root"), code="runtime_cpython_inventory_invalid"
    )
    interpreter_root = _safe_directory(
        root, interpreter_root_relative, code="runtime_cpython_inventory_invalid"
    )
    try:
        headers = [
            item
            for item in (
                interpreter_root / "include" / "patchlevel.h",
                interpreter_root / "include" / "python3.11" / "patchlevel.h",
            )
            if item.is_file() and not item.is_symlink()
        ]
        license_files = [
            item
            for item in (
                interpreter_root / "LICENSE",
                interpreter_root / "LICENSE.txt",
                interpreter_root / "LICENSE.rst",
                interpreter_root / "lib" / "python3.11" / "LICENSE.txt",
            )
            if item.is_file() and not item.is_symlink()
        ]
    except OSError:
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid") from None
    if len(headers) != 1 or len(license_files) != 1:
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid")
    header_raw = _read_bytes(
        headers[0], maximum=_MAX_METADATA_BYTES, code="runtime_cpython_inventory_invalid"
    )
    try:
        header = header_raw.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid") from None
    values: dict[str, str] = {}
    for name in ("PY_MAJOR_VERSION", "PY_MINOR_VERSION", "PY_MICRO_VERSION"):
        matched = re.findall(rf"^#define\s+{name}\s+([0-9]+)\s*$", header, flags=re.MULTILINE)
        if len(matched) != 1:
            raise RuntimeMetadataError("runtime_cpython_inventory_invalid")
        values[name] = matched[0]
    version = ".".join(
        values[name] for name in ("PY_MAJOR_VERSION", "PY_MINOR_VERSION", "PY_MICRO_VERSION")
    )
    if not version.startswith("3.11."):
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid")
    expected_tree = profile.get("sidecar_interpreter_root_tree_sha256")
    if not isinstance(expected_tree, str) or _SHA256.fullmatch(expected_tree) is None:
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid")
    license_raw = _read_bytes(
        license_files[0], maximum=_MAX_LICENSE_BYTES, code="runtime_cpython_license_invalid"
    )
    try:
        license_text = license_raw.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeMetadataError("runtime_cpython_license_invalid") from None
    if "https://www.python.org/psf/" not in license_text:
        raise RuntimeMetadataError("runtime_cpython_license_invalid")
    return (
        ActualComponent("CPython", version, _RUNTIME_SCOPE, expected_tree),
        license_files[0],
        license_text,
    )


def actual_runtime_components(
    root: Path, profile: Mapping[str, Any]
) -> tuple[ActualComponent, ...]:
    package_rows = [
        *_parse_package_inventory(
            root,
            relative_path=profile.get("parent_renderer_package_inventory_path"),
            expected_sha256=profile.get("parent_renderer_package_inventory_sha256"),
        ),
        *_parse_package_inventory(
            root,
            relative_path=profile.get("sidecar_package_inventory_path"),
            expected_sha256=profile.get("sidecar_package_inventory_sha256"),
        ),
    ]
    packages: dict[tuple[str, str], ActualComponent] = {}
    for name, version in package_rows:
        identity = (_normalized_name(name), version)
        if identity in packages:
            raise RuntimeMetadataError("runtime_component_identity_duplicate")
        packages[identity] = ActualComponent(
            name=_normalized_name(name),
            version=version,
            scope=_PACKAGE_SCOPE,
            # The inventories identify an installed distribution by exact
            # normalized name/version but do not attest a wheel digest.
            # Do not mislabel a synthetic identity hash as an artifact
            # checksum; the exhaustive OCI file manifest binds its bytes.
            checksum_sha256=None,
        )
    models, _model_root = _model_components(root, profile)
    python, _license_path, _license_text = _python_component(root, profile)
    result = [*packages.values(), *models, python]
    identities = [item.identity for item in result]
    if len(identities) != len(set(identities)):
        raise RuntimeMetadataError("runtime_component_identity_duplicate")
    return tuple(sorted(result, key=lambda item: item.identity))


def _metadata_source_uri(evidence: DistributionEvidence) -> str:
    priorities = (
        {"source", "source code", "repository", "github", "code"},
        {"homepage", "home"},
    )

    def candidate(value: str) -> str | None:
        if value == "UNKNOWN":
            return None
        # Some older wheel metadata still publishes a plain-HTTP homepage.
        # Validate its complete shape before ignoring it; the deterministic
        # registry fallback below is the only value that is emitted.
        if value.startswith("http://"):
            _valid_source_uri(f"https://{value.removeprefix('http://')}")
            return None
        return _valid_source_uri(value)

    for labels in priorities:
        candidates: set[str] = set()
        for item in evidence.project_urls:
            label, separator, value = str(item).partition(",")
            if separator and label.strip().casefold() in labels:
                selected = candidate(value.strip())
                if selected is not None:
                    candidates.add(selected)
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            raise RuntimeMetadataError("runtime_package_source_ambiguous")
    homepages = {selected for item in evidence.home_pages if (selected := candidate(item))}
    if len(homepages) == 1:
        return next(iter(homepages))
    if len(homepages) > 1:
        raise RuntimeMetadataError("runtime_package_source_ambiguous")
    # The exact normalized distribution identity is already bound by the
    # hash-verified inventory and METADATA.  PyPI's canonical project/version
    # page is therefore a deterministic, offline-computable registry source;
    # it does not make any unverified repository claim.
    registry_source = (
        "https://pypi.org/project/"
        f"{quote(_normalized_name(evidence.name), safe='')}/"
        f"{quote(evidence.version, safe='')}/"
    )
    return _valid_source_uri(registry_source)


def _distribution_layout(
    distribution: importlib.metadata.Distribution, *, code: str
) -> tuple[Path, str, tuple[importlib.metadata.PackagePath, ...]]:
    try:
        located_root = Path(str(distribution.locate_file("")))
        metadata_path_value = getattr(distribution, "_path", None)
        if metadata_path_value is None:
            raise OSError
        metadata_path = Path(metadata_path_value)
        if (
            located_root.is_symlink()
            or not located_root.is_dir()
            or metadata_path.is_symlink()
            or not metadata_path.is_dir()
        ):
            raise OSError
        distribution_root = located_root.resolve(strict=True)
        resolved_metadata = metadata_path.resolve(strict=True)
        if (
            resolved_metadata.parent != distribution_root
            or not resolved_metadata.name.casefold().endswith(".dist-info")
        ):
            raise OSError
        owner = resolved_metadata.name
        distribution_files = tuple(distribution.files or ())
        owner_metadata_rows = [
            relative
            for relative in distribution_files
            if PurePosixPath(str(relative).replace("\\", "/")).parts == (owner, "METADATA")
        ]
        if len(owner_metadata_rows) != 1:
            raise OSError
    except Exception:
        raise RuntimeMetadataError(code) from None
    return distribution_root, owner, distribution_files


def _license_file_texts(
    distribution: importlib.metadata.Distribution,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    total_bytes = 0
    try:
        distribution_root, metadata_owner, distribution_files = _distribution_layout(
            distribution, code="runtime_package_license_invalid"
        )
        for relative in distribution_files:
            parts = PurePosixPath(str(relative).replace("\\", "/")).parts
            if not parts or not parts[0].endswith(".dist-info"):
                continue
            is_license_evidence = any(
                marker in parts[-1].casefold()
                for marker in ("license", "licence", "copying", "notice")
            ) or any(
                part.casefold() in {"license", "licenses", "licence", "licences"}
                for part in parts[1:-1]
            )
            if not is_license_evidence:
                continue
            if parts[0] != metadata_owner:
                raise RuntimeMetadataError("runtime_package_license_invalid")
            safe_relative = _safe_relative("/".join(parts), code="runtime_package_license_invalid")
            path = distribution_root.joinpath(*safe_relative.parts)
            candidate = distribution_root
            for part in safe_relative.parts:
                candidate /= part
                if candidate.is_symlink():
                    raise RuntimeMetadataError("runtime_package_license_invalid")
            # RECORD can name a wheel license file that the installed runtime
            # did not materialize.  It is not part of the actual hash-bound
            # runtime tree, so do not claim or synthesize its contents.
            if not path.is_file():
                continue
            path = _safe_file(
                distribution_root,
                safe_relative,
                code="runtime_package_license_invalid",
            )
            raw = _read_bytes(
                path, maximum=_MAX_LICENSE_BYTES, code="runtime_package_license_invalid"
            )
            total_bytes += len(raw)
            if len(result) >= _MAX_LICENSE_FILES or total_bytes > _MAX_TOTAL_LICENSE_BYTES:
                raise RuntimeMetadataError("runtime_package_license_invalid")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise RuntimeMetadataError("runtime_package_license_invalid") from None
            if not text.strip():
                raise RuntimeMetadataError("runtime_package_license_invalid")
            evidence_name = "/".join(parts[1:])
            if any(name == evidence_name for name, _text in result):
                raise RuntimeMetadataError("runtime_package_license_invalid")
            result.append((evidence_name, text))
    except RuntimeMetadataError:
        raise
    except Exception:
        raise RuntimeMetadataError("runtime_package_license_invalid") from None
    return tuple(sorted(result))


def _sbom_license_evidence(
    distribution: importlib.metadata.Distribution,
    *,
    name: str,
    version: str,
) -> tuple[tuple[str, str, str], ...]:
    result: list[tuple[str, str, str]] = []
    try:
        distribution_root, metadata_owner, distribution_files = _distribution_layout(
            distribution, code="runtime_package_sbom_invalid"
        )
        candidates: list[tuple[PurePosixPath, Path]] = []
        for distribution_file in distribution_files:
            untrusted_relative = str(distribution_file).replace("\\", "/")
            parts = PurePosixPath(untrusted_relative).parts
            if (
                len(parts) == 3
                and parts[0] == metadata_owner
                and parts[1] == "sboms"
                and PurePosixPath(parts[2]).suffix.casefold() == ".json"
            ):
                safe_relative = _safe_relative(
                    untrusted_relative,
                    code="runtime_package_sbom_invalid",
                )
                candidates.append(
                    (
                        safe_relative,
                        _safe_file(
                            distribution_root,
                            safe_relative,
                            code="runtime_package_sbom_invalid",
                        ),
                    )
                )
        if len(candidates) > 8:
            raise RuntimeMetadataError("runtime_package_sbom_invalid")
        for sbom_relative, path in sorted(candidates):
            raw = _read_bytes(
                path,
                maximum=_MAX_JSON_BYTES,
                code="runtime_package_sbom_invalid",
            )
            try:
                value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                raise RuntimeMetadataError("runtime_package_sbom_invalid") from None
            if not isinstance(value, dict) or value.get("bomFormat") != "CycloneDX":
                raise RuntimeMetadataError("runtime_package_sbom_invalid")
            components = value.get("components")
            if not isinstance(components, list):
                raise RuntimeMetadataError("runtime_package_sbom_invalid")
            matches = [
                component
                for component in components
                if isinstance(component, dict)
                and isinstance(component.get("name"), str)
                and _normalized_name(component["name"]) == _normalized_name(name)
                and component.get("version") == version
            ]
            expressions: list[str] = []
            for match in matches:
                licenses = match.get("licenses")
                # Wheel SBOMs may list the wheel itself without a license and
                # separately list licensed native components.  Absence is not
                # evidence and must not override stronger METADATA/files.
                if licenses is None or licenses == []:
                    continue
                if (
                    not isinstance(licenses, list)
                    or len(licenses) != 1
                    or not isinstance(licenses[0], dict)
                ):
                    raise RuntimeMetadataError("runtime_package_sbom_invalid")
                declaration = licenses[0]
                if set(declaration) == {"expression"}:
                    value = declaration.get("expression")
                elif (
                    set(declaration) == {"license"}
                    and isinstance(declaration.get("license"), dict)
                    and set(declaration["license"]) == {"id"}
                ):
                    value = declaration["license"].get("id")
                else:
                    raise RuntimeMetadataError("runtime_package_sbom_invalid")
                expressions.append(_valid_license_expression(value))
            if not expressions:
                continue
            if len(set(expressions)) != 1:
                raise RuntimeMetadataError("runtime_package_sbom_invalid")
            expression = expressions[0]
            result.append(
                (
                    "/".join(sbom_relative.parts[1:]),
                    hashlib.sha256(raw).hexdigest(),
                    expression,
                )
            )
    except RuntimeMetadataError:
        raise
    except Exception:
        raise RuntimeMetadataError("runtime_package_sbom_invalid") from None
    return tuple(result)


def _distribution_evidence(
    distribution: importlib.metadata.Distribution,
) -> DistributionEvidence:
    try:
        distribution_root, metadata_owner, _distribution_files = _distribution_layout(
            distribution, code="runtime_package_metadata_invalid"
        )
        metadata_raw = _read_bytes(
            _safe_file(
                distribution_root,
                PurePosixPath(metadata_owner, "METADATA"),
                code="runtime_package_metadata_invalid",
            ),
            maximum=_MAX_METADATA_BYTES,
            code="runtime_package_metadata_invalid",
        )
        metadata_text = metadata_raw.decode("utf-8")
    except (RuntimeMetadataError, UnicodeDecodeError):
        raise RuntimeMetadataError("runtime_package_metadata_invalid") from None
    if not metadata_text or len(metadata_raw) > _MAX_METADATA_BYTES:
        raise RuntimeMetadataError("runtime_package_metadata_invalid")
    parsed = BytesParser(policy=compat32).parsebytes(
        metadata_text.encode("utf-8"), headersonly=True
    )
    names = parsed.get_all("Name") or []
    versions = parsed.get_all("Version") or []
    if (
        len(names) != 1
        or len(versions) != 1
        or not isinstance(names[0], str)
        or not isinstance(versions[0], str)
        or _SAFE_NAME.fullmatch(names[0]) is None
        or not versions[0].strip()
    ):
        raise RuntimeMetadataError("runtime_package_metadata_invalid")
    return DistributionEvidence(
        name=names[0],
        version=versions[0],
        metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
        license_expressions=tuple(
            str(value) for value in (parsed.get_all("License-Expression") or [])
        ),
        legacy_licenses=tuple(str(value) for value in (parsed.get_all("License") or [])),
        project_urls=tuple(str(value) for value in (parsed.get_all("Project-URL") or [])),
        home_pages=tuple(str(value) for value in (parsed.get_all("Home-page") or [])),
        sbom_license_evidence=_sbom_license_evidence(
            distribution,
            name=names[0],
            version=versions[0],
        ),
        license_files=_license_file_texts(distribution),
    )


def _distribution_license(
    evidence: DistributionEvidence,
) -> tuple[str, tuple[ExtractedLicense, ...]]:
    if len(evidence.license_expressions) > 1 or len(evidence.legacy_licenses) > 1:
        raise RuntimeMetadataError("runtime_package_license_invalid")
    expression_value = evidence.license_expressions[0] if evidence.license_expressions else None
    legacy_value = evidence.legacy_licenses[0] if evidence.legacy_licenses else None
    for declared in (expression_value, legacy_value):
        if isinstance(declared, str) and len(declared.encode("utf-8")) > _MAX_LICENSE_BYTES:
            raise RuntimeMetadataError("runtime_package_license_invalid")
    normalized = " ".join(expression_value.split()) if isinstance(expression_value, str) else ""
    try:
        expression = _valid_license_expression(normalized)
    except RuntimeMetadataError:
        expression = ""
    license_files = evidence.license_files
    if expression:
        references = sorted(set(re.findall(r"LicenseRef-[A-Za-z0-9.+-]+", expression)))
        extracted: list[ExtractedLicense] = []
        for reference in references:
            matches = [
                text for name, text in license_files if reference.casefold() in name.casefold()
            ]
            if len(matches) != 1:
                raise RuntimeMetadataError("runtime_package_license_invalid")
            extracted.append(ExtractedLicense(reference, matches[0]))
        return expression, tuple(extracted)
    sbom_expressions = {
        expression for _relative, _sha256, expression in evidence.sbom_license_evidence
    }
    if sbom_expressions:
        if len(sbom_expressions) != 1:
            raise RuntimeMetadataError("runtime_package_license_invalid")
        return next(iter(sbom_expressions)), ()
    evidence_items: list[tuple[str, str]] = list(license_files)
    if not evidence_items:
        raise RuntimeMetadataError("runtime_package_license_missing")
    if isinstance(expression_value, str) and expression_value.strip():
        evidence_items.insert(0, ("METADATA:License-Expression", expression_value.strip()))
    if isinstance(legacy_value, str) and legacy_value.strip():
        evidence_items.insert(0, ("METADATA:License", legacy_value.strip()))
    extracted_text = "\n\n".join(f"===== {name} =====\n{text}" for name, text in evidence_items)
    if len(extracted_text.encode("utf-8")) > _MAX_TOTAL_LICENSE_BYTES:
        raise RuntimeMetadataError("runtime_package_license_invalid")
    digest = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
    reference = f"LicenseRef-PackageEvidence-{digest}"
    return reference, (ExtractedLicense(reference, extracted_text),)


def _default_distribution_finder(name: str) -> Sequence[importlib.metadata.Distribution]:
    return tuple(importlib.metadata.distributions(name=name))


def _distribution_name(distribution: importlib.metadata.Distribution, *, code: str) -> str:
    try:
        name = distribution.metadata["Name"]
    except Exception:
        raise RuntimeMetadataError(code) from None
    if not isinstance(name, str) or not name:
        raise RuntimeMetadataError(code)
    return name


def _sidecar_distributions(
    root: Path, profile: Mapping[str, Any]
) -> dict[tuple[str, str], importlib.metadata.Distribution]:
    module_root_relative = _safe_relative(
        profile.get("sidecar_module_root"), code="runtime_package_metadata_invalid"
    )
    module_root = _safe_directory(
        root, module_root_relative, code="runtime_package_metadata_invalid"
    )
    discovered: dict[tuple[str, str], importlib.metadata.Distribution] = {}
    try:
        sidecar_distributions = tuple(importlib.metadata.distributions(path=[str(module_root)]))
    except Exception:
        raise RuntimeMetadataError("runtime_package_metadata_invalid") from None
    for distribution in sidecar_distributions:
        name = _distribution_name(distribution, code="runtime_package_metadata_invalid")
        version = distribution.version
        if not isinstance(name, str) or not version:
            raise RuntimeMetadataError("runtime_package_metadata_invalid")
        identity = (_normalized_name(name), version)
        if identity in discovered:
            raise RuntimeMetadataError("runtime_package_metadata_duplicate")
        discovered[identity] = distribution
    sidecar_inventory = {
        (_normalized_name(name), version)
        for name, version in _parse_package_inventory(
            root,
            relative_path=profile.get("sidecar_package_inventory_path"),
            expected_sha256=profile.get("sidecar_package_inventory_sha256"),
        )
    }
    if set(discovered) != sidecar_inventory:
        raise RuntimeMetadataError("runtime_package_metadata_set_mismatch")
    return discovered


def _package_licenses(
    root: Path,
    profile: Mapping[str, Any],
    actual: Sequence[ActualComponent],
    *,
    distribution_finder: DistributionFinder,
) -> tuple[
    dict[tuple[str, str, str], LicensedComponent],
    tuple[DistributionEvidence, ...],
]:
    discovered = _sidecar_distributions(root, profile)
    expected_packages = {item.identity for item in actual if item.scope == _PACKAGE_SCOPE}
    result: dict[tuple[str, str, str], LicensedComponent] = {}
    parent_evidence: list[DistributionEvidence] = []
    for actual_component in actual:
        if actual_component.scope != _PACKAGE_SCOPE:
            continue
        key = (_normalized_name(actual_component.name), actual_component.version)
        distribution = discovered.get(key)
        is_parent_only = distribution is None
        if distribution is None:
            matches = [
                item
                for item in distribution_finder(actual_component.name)
                if _normalized_name(
                    _distribution_name(item, code="runtime_parent_package_metadata_missing")
                )
                == key[0]
                and item.version == key[1]
            ]
            if len(matches) != 1:
                raise RuntimeMetadataError("runtime_parent_package_metadata_missing")
            distribution = matches[0]
        evidence = _distribution_evidence(distribution)
        if _normalized_name(evidence.name) != key[0] or evidence.version != key[1]:
            raise RuntimeMetadataError("runtime_package_metadata_invalid")
        license_id, extracted = _distribution_license(evidence)
        licensed = LicensedComponent(
            actual=actual_component,
            license_id=license_id,
            source_uri=_metadata_source_uri(evidence),
            extracted_licenses=extracted,
        )
        result[licensed.actual.identity] = licensed
        if is_parent_only:
            parent_evidence.append(evidence)
    if set(result) != expected_packages:
        raise RuntimeMetadataError("runtime_package_metadata_set_mismatch")
    return result, tuple(
        sorted(
            parent_evidence,
            key=lambda item: (_normalized_name(item.name), item.version),
        )
    )


def _parent_evidence_payload(evidence: Sequence[DistributionEvidence]) -> bytes:
    return canonical_json_bytes(
        {
            "distributions": [
                {
                    "license_files": [
                        {
                            "relative_path": relative,
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "text": text,
                        }
                        for relative, text in item.license_files
                    ],
                    "home_pages": list(item.home_pages),
                    "legacy_licenses": list(item.legacy_licenses),
                    "license_expressions": list(item.license_expressions),
                    "metadata_sha256": item.metadata_sha256,
                    "name": item.name,
                    "project_urls": list(item.project_urls),
                    "sbom_license_evidence": [
                        {
                            "expression": expression,
                            "relative_path": relative,
                            "sha256": sha256,
                        }
                        for relative, sha256, expression in item.sbom_license_evidence
                    ],
                    "version": item.version,
                }
                for item in evidence
            ],
            "schema_version": "1.0",
        }
    )


def _read_parent_evidence(root: Path) -> tuple[DistributionEvidence, ...]:
    value, _raw = _read_json(
        root,
        _PARENT_EVIDENCE_RELATIVE_PATH,
        code="runtime_parent_package_evidence_invalid",
    )
    rows = value.get("distributions")
    if (
        set(value) != {"distributions", "schema_version"}
        or value.get("schema_version") != "1.0"
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
    result: list[DistributionEvidence] = []
    observed: set[tuple[str, str]] = set()
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "home_pages",
            "legacy_licenses",
            "license_expressions",
            "license_files",
            "metadata_sha256",
            "name",
            "project_urls",
            "sbom_license_evidence",
            "version",
        }:
            raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
        name = row.get("name")
        version = row.get("version")
        license_rows = row.get("license_files")
        sbom_rows = row.get("sbom_license_evidence")
        header_fields = (
            "home_pages",
            "legacy_licenses",
            "license_expressions",
            "project_urls",
        )
        if (
            not isinstance(name, str)
            or _SAFE_NAME.fullmatch(name) is None
            or not isinstance(version, str)
            or not version.strip()
            or not isinstance(row.get("metadata_sha256"), str)
            or _SHA256.fullmatch(str(row["metadata_sha256"])) is None
            or not isinstance(license_rows, list)
            or len(license_rows) > _MAX_LICENSE_FILES
            or not isinstance(sbom_rows, list)
            or len(sbom_rows) > 8
            or any(not isinstance(row.get(field), list) for field in header_fields)
        ):
            raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
        headers: dict[str, tuple[str, ...]] = {}
        for field in header_fields:
            values = row[field]
            if len(values) > _MAX_LICENSE_FILES or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            encoded_size = sum(len(value.encode("utf-8")) for value in values)
            total_bytes += encoded_size
            if encoded_size > _MAX_METADATA_BYTES:
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            headers[field] = tuple(values)
        license_files: list[tuple[str, str]] = []
        for license_row in license_rows:
            if not isinstance(license_row, dict) or set(license_row) != {
                "relative_path",
                "sha256",
                "text",
            }:
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            relative = _safe_relative(
                license_row.get("relative_path"),
                code="runtime_parent_package_evidence_invalid",
            ).as_posix()
            text = license_row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            raw = text.encode("utf-8")
            total_bytes += len(raw)
            if (
                len(raw) > _MAX_LICENSE_BYTES
                or license_row.get("sha256") != hashlib.sha256(raw).hexdigest()
                or any(existing == relative for existing, _text in license_files)
            ):
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            license_files.append((relative, text))
        sbom_evidence: list[tuple[str, str, str]] = []
        for sbom_row in sbom_rows:
            if not isinstance(sbom_row, dict) or set(sbom_row) != {
                "expression",
                "relative_path",
                "sha256",
            }:
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            relative = _safe_relative(
                sbom_row.get("relative_path"),
                code="runtime_parent_package_evidence_invalid",
            ).as_posix()
            digest = sbom_row.get("sha256")
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or any(item[0] == relative for item in sbom_evidence)
            ):
                raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
            sbom_evidence.append(
                (
                    relative,
                    digest,
                    _valid_license_expression(sbom_row.get("expression")),
                )
            )
        if total_bytes > _MAX_TOTAL_LICENSE_BYTES:
            raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
        evidence = DistributionEvidence(
            name=name,
            version=version,
            metadata_sha256=str(row["metadata_sha256"]),
            license_expressions=headers["license_expressions"],
            legacy_licenses=headers["legacy_licenses"],
            project_urls=headers["project_urls"],
            home_pages=headers["home_pages"],
            sbom_license_evidence=tuple(sbom_evidence),
            license_files=tuple(license_files),
        )
        identity = (_normalized_name(name), version)
        if identity in observed:
            raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
        observed.add(identity)
        result.append(evidence)
    if [(_normalized_name(item.name), item.version) for item in result] != sorted(observed):
        raise RuntimeMetadataError("runtime_parent_package_evidence_invalid")
    return tuple(result)


def _detector_license(
    root: Path,
    profile: Mapping[str, Any],
    detector: ActualComponent,
) -> tuple[LicensedComponent, PurePosixPath]:
    _models, model_root = _model_components(root, profile)
    combined, _raw = _read_json(
        root,
        model_root / "manifest.json",
        expected_sha256=profile.get("mobile_model_bundle_manifest_sha256"),
        code="runtime_model_license_evidence_invalid",
    )
    mobile_relative = _safe_relative(
        combined.get("mobile_manifest_provenance_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    mobile = _model_record_json(root, model_root, combined, mobile_relative)
    source_relative = _safe_relative(
        mobile.get("source_model_inventory_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    source, _source_raw = _read_json(
        root,
        source_relative,
        expected_sha256=mobile.get("source_model_inventory_sha256"),
        code="runtime_model_license_evidence_invalid",
    )
    rows = source.get("artifacts")
    if not isinstance(rows, list):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    combined_inventory_relative = _safe_relative(
        combined.get("model_inventory_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    combined_inventory, _combined_inventory_raw = _read_json(
        root,
        model_root / combined_inventory_relative,
        expected_sha256=profile.get("model_inventory_sha256"),
        code="runtime_model_license_evidence_invalid",
    )
    combined_rows = combined_inventory.get("artifacts")
    if not isinstance(combined_rows, list):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    combined_detector_hashes = {
        str(row.get("sha256"))
        for row in combined_rows
        if isinstance(row, dict)
        and isinstance(row.get("relative_path"), str)
        and str(row["relative_path"]).startswith("text-det/")
        and isinstance(row.get("sha256"), str)
        and _SHA256.fullmatch(str(row["sha256"])) is not None
    }
    evidence: set[tuple[str, str]] = set()
    hashes: set[str] = set()
    detector_row_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        path = row.get("relative_path")
        if not isinstance(path, str):
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        relative = _safe_relative(path, code="runtime_model_license_evidence_invalid")
        if relative.parts[:3] != ("models", "paddleocr3", "det"):
            continue
        detector_row_count += 1
        digest = row.get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        evidence.add(
            (
                _valid_license_expression(row.get("license_id")),
                _valid_source_uri(row.get("source_uri")),
            )
        )
        hashes.add(digest)
    if (
        len(evidence) != 1
        or len(hashes) != 4
        or detector_row_count != 4
        or hashes != combined_detector_hashes
    ):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    license_id, source_uri = next(iter(evidence))
    return LicensedComponent(detector, license_id, source_uri), source_relative


def _read_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_bytes(path, maximum=_MAX_JSON_BYTES, code="runtime_license_policy_invalid")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeMetadataError("runtime_license_policy_invalid") from None
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value) + b"\n"
        or set(value)
        != {
            "cpython",
            "distribution_scope",
            "fine_tuned_models",
            "schema_version",
        }
        or value.get("schema_version") != "1.0"
        or value.get("distribution_scope") != "private-ghcr-only"
        or not isinstance(value.get("cpython"), list)
        or not isinstance(value.get("fine_tuned_models"), list)
    ):
        raise RuntimeMetadataError("runtime_license_policy_invalid")
    return value, raw


def _python_license(
    python: ActualComponent,
    license_text: str,
    *,
    policy: Mapping[str, Any],
) -> LicensedComponent:
    license_digest = hashlib.sha256(license_text.encode("utf-8")).hexdigest()
    matches: list[Mapping[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    for item in policy["cpython"]:
        if not isinstance(item, dict) or set(item) != {
            "license_sha256",
            "source_uri",
            "version",
        }:
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        version = item.get("version")
        digest = item.get("license_sha256")
        if (
            not isinstance(version, str)
            or not version.strip()
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        _valid_source_uri(item.get("source_uri"))
        identity = (version, digest)
        if identity in observed:
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        observed.add(identity)
        if version == python.version and digest == license_digest:
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeMetadataError("runtime_cpython_license_policy_missing")
    license_id = f"LicenseRef-CPython-{license_digest}"
    return LicensedComponent(
        python,
        license_id,
        _valid_source_uri(matches[0]["source_uri"]),
        (ExtractedLicense(license_id, license_text),),
    )


def _model_record_json(
    root: Path,
    model_root: PurePosixPath,
    manifest: Mapping[str, Any],
    relative: PurePosixPath,
) -> dict[str, Any]:
    files = manifest.get("files")
    matches = (
        [
            item
            for item in files
            if isinstance(item, dict) and item.get("path") == relative.as_posix()
        ]
        if isinstance(files, list)
        else []
    )
    if len(matches) != 1:
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    record = matches[0]
    digest = record.get("sha256")
    size = record.get("byte_size")
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    raw = _read_bytes(
        _safe_file(
            root,
            model_root / relative,
            code="runtime_model_license_evidence_invalid",
        ),
        maximum=_MAX_JSON_BYTES,
        code="runtime_model_license_evidence_invalid",
    )
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid") from None
    if not isinstance(value, dict):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    return value


def _verify_recognizer_upstream_evidence(
    root: Path,
    profile: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> None:
    _models, model_root = _model_components(root, profile)
    combined, _combined_raw = _read_json(
        root,
        model_root / "manifest.json",
        expected_sha256=profile.get("mobile_model_bundle_manifest_sha256"),
        code="runtime_model_license_evidence_invalid",
    )
    receipt_relative = _safe_relative(
        combined.get("recognizer_receipt_provenance_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    receipt = _model_record_json(root, model_root, combined, receipt_relative)
    receipt_digest = receipt.get("receipt_sha256")
    receipt_candidate = dict(receipt)
    receipt_candidate["receipt_sha256"] = ""
    if (
        not isinstance(receipt_digest, str)
        or _SHA256.fullmatch(receipt_digest) is None
        or hashlib.sha256(canonical_json_bytes(receipt_candidate)).hexdigest() != receipt_digest
        or receipt_digest != combined.get("source_recognizer_receipt_sha256")
        or receipt_digest != selected.get("upstream_training_receipt_sha256")
        or receipt.get("initialization") != "pinned_public_inference_graph_warm_start"
        or receipt.get("tuning_scope") != "head"
    ):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")

    mobile_relative = _safe_relative(
        combined.get("mobile_manifest_provenance_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    mobile, _mobile_raw = _read_json(
        root,
        model_root / mobile_relative,
        code="runtime_model_license_evidence_invalid",
    )
    source_relative = _safe_relative(
        mobile.get("source_model_inventory_relative_path"),
        code="runtime_model_license_evidence_invalid",
    )
    source_inventory_sha256 = selected.get("upstream_base_source_inventory_sha256")
    if (
        not isinstance(source_inventory_sha256, str)
        or _SHA256.fullmatch(source_inventory_sha256) is None
        or source_inventory_sha256 != mobile.get("source_model_inventory_sha256")
        or source_inventory_sha256 != receipt.get("public_mobile_source_inventory_sha256")
        or mobile.get("text_recognition_model_name") != selected.get("upstream_base_model_name")
    ):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    source, _source_raw = _read_json(
        root,
        source_relative,
        expected_sha256=source_inventory_sha256,
        code="runtime_model_license_evidence_invalid",
    )
    receipt_artifacts = receipt.get("public_recognizer_artifact_sha256s")
    expected_receipt_paths = {f"models/text-rec/{name}" for name in _MODEL_FILENAMES}
    if (
        not isinstance(receipt_artifacts, dict)
        or set(receipt_artifacts) != expected_receipt_paths
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in receipt_artifacts.values()
        )
        or hashlib.sha256(canonical_json_bytes(receipt_artifacts)).hexdigest()
        != selected.get("upstream_base_artifacts_sha256")
    ):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    rows = source.get("artifacts")
    if not isinstance(rows, list):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    upstream_license = selected.get("upstream_base_license_id")
    upstream_source = selected.get("upstream_base_source_uri")
    if upstream_license != "Apache-2.0":
        raise RuntimeMetadataError("runtime_license_policy_invalid")
    _valid_source_uri(upstream_source)
    recognizer_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        relative_value = row.get("relative_path")
        if not isinstance(relative_value, str):
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        relative = _safe_relative(relative_value, code="runtime_model_license_evidence_invalid")
        if relative.parts[:3] != ("models", "paddleocr3", "rec"):
            continue
        if len(relative.parts) != 4 or relative.parts[3] in recognizer_rows:
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
        recognizer_rows[relative.parts[3]] = row
    if set(recognizer_rows) != set(_MODEL_FILENAMES):
        raise RuntimeMetadataError("runtime_model_license_evidence_invalid")
    for name, row in recognizer_rows.items():
        if (
            row.get("sha256") != receipt_artifacts[f"models/text-rec/{name}"]
            or row.get("license_id") != upstream_license
            or row.get("source_uri") != upstream_source
        ):
            raise RuntimeMetadataError("runtime_model_license_evidence_invalid")


def _recognizer_license(
    root: Path,
    profile: Mapping[str, Any],
    recognizer: ActualComponent,
    *,
    policy: Mapping[str, Any],
) -> LicensedComponent:
    matches: list[Mapping[str, Any]] = []
    observed: set[tuple[str, str, str]] = set()
    for item in policy["fine_tuned_models"]:
        if not isinstance(item, dict) or set(item) != {
            "combined_model_manifest_sha256",
            "distribution_scope",
            "license_id",
            "model_name",
            "model_snapshot_sha256",
            "source_uri",
            "upstream_base_artifacts_sha256",
            "upstream_base_license_id",
            "upstream_base_model_name",
            "upstream_base_source_inventory_sha256",
            "upstream_base_source_uri",
            "upstream_training_receipt_sha256",
        }:
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        for field in (
            "combined_model_manifest_sha256",
            "model_snapshot_sha256",
            "upstream_base_artifacts_sha256",
            "upstream_base_source_inventory_sha256",
            "upstream_training_receipt_sha256",
        ):
            if not isinstance(item.get(field), str) or _SHA256.fullmatch(str(item[field])) is None:
                raise RuntimeMetadataError("runtime_license_policy_invalid")
        model_name = item.get("model_name")
        if not isinstance(model_name, str) or _SAFE_NAME.fullmatch(model_name) is None:
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        identity = (
            str(item["combined_model_manifest_sha256"]),
            model_name,
            str(item["model_snapshot_sha256"]),
        )
        if identity in observed:
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        observed.add(identity)
        if (
            item.get("distribution_scope") != "private-ghcr-only"
            or item.get("license_id") != "NOASSERTION"
            or item.get("source_uri") != "NOASSERTION"
            or item.get("upstream_base_license_id") != "Apache-2.0"
            or not isinstance(item.get("upstream_base_model_name"), str)
            or _SAFE_NAME.fullmatch(str(item["upstream_base_model_name"])) is None
        ):
            raise RuntimeMetadataError("runtime_license_policy_invalid")
        _valid_source_uri(item.get("upstream_base_source_uri"))
        if (
            item.get("combined_model_manifest_sha256")
            == profile.get("mobile_model_bundle_manifest_sha256")
            and item.get("model_name") == recognizer.name
            and f"sha256:{item.get('model_snapshot_sha256')}" == recognizer.version
        ):
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeMetadataError("runtime_finetuned_model_license_policy_missing")
    selected = matches[0]
    _verify_recognizer_upstream_evidence(root, profile, selected)
    return LicensedComponent(
        recognizer,
        "NOASSERTION",
        "NOASSERTION",
    )


def licensed_runtime_components(
    root: Path,
    profile: Mapping[str, Any],
    *,
    policy_path: Path,
    distribution_finder: DistributionFinder = _default_distribution_finder,
) -> tuple[
    tuple[LicensedComponent, ...],
    tuple[PurePosixPath, ...],
    tuple[tuple[PurePosixPath, bytes], ...],
    bytes,
]:
    actual = actual_runtime_components(root, profile)
    policy, policy_raw = _read_policy(policy_path)
    licensed, parent_evidence = _package_licenses(
        root,
        profile,
        actual,
        distribution_finder=distribution_finder,
    )
    models = [item for item in actual if item.scope == _MODEL_SCOPE]
    if len(models) != 2:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    detector = next((item for item in models if item.name == "PP-OCRv5_mobile_det"), None)
    recognizer = next(
        (item for item in models if item.name == "source_attested_en_PP-OCRv5_mobile_rec_head_v2"),
        None,
    )
    if detector is None or recognizer is None:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    detector_license, detector_evidence = _detector_license(root, profile, detector)
    licensed[detector.identity] = detector_license
    licensed[recognizer.identity] = _recognizer_license(root, profile, recognizer, policy=policy)
    python = next((item for item in actual if item.scope == _RUNTIME_SCOPE), None)
    if python is None:
        raise RuntimeMetadataError("runtime_cpython_inventory_invalid")
    _python_actual, _license_path, license_text = _python_component(root, profile)
    licensed[python.identity] = _python_license(
        python,
        license_text,
        policy=policy,
    )
    if set(licensed) != {item.identity for item in actual}:
        raise RuntimeMetadataError("runtime_license_component_set_mismatch")
    generated_evidence = (
        ((_PARENT_EVIDENCE_RELATIVE_PATH, _parent_evidence_payload(parent_evidence)),)
        if parent_evidence
        else ()
    )
    return (
        tuple(sorted(licensed.values(), key=lambda item: item.actual.identity)),
        (detector_evidence,),
        generated_evidence,
        policy_raw,
    )


def _license_manifest(components: Sequence[LicensedComponent]) -> dict[str, object]:
    return {
        "components": [
            {
                "checksum_sha256": item.actual.checksum_sha256,
                "license_id": item.license_id,
                "name": item.actual.name,
                "scope": item.actual.scope,
                "source_uri": item.source_uri,
                "version": item.actual.version,
            }
            for item in components
        ],
        "distribution_scope": "private-ghcr-only",
        "schema_version": "1.0",
    }


def _spdx_id(component: LicensedComponent) -> str:
    digest = hashlib.sha256(canonical_json_bytes(component.actual.identity)).hexdigest()
    return f"SPDXRef-Package-{digest}"


def _spdx(components: Sequence[LicensedComponent], *, profile_revision: str) -> dict[str, object]:
    component_payload = [
        {
            "checksum_sha256": item.actual.checksum_sha256,
            "license_id": item.license_id,
            "name": item.actual.name,
            "scope": item.actual.scope,
            "source_uri": item.source_uri,
            "version": item.actual.version,
        }
        for item in components
    ]
    namespace_sha = hashlib.sha256(
        canonical_json_bytes(
            {"components": component_payload, "profile_revision": profile_revision}
        )
    ).hexdigest()
    extracted: dict[str, str] = {}
    for component in components:
        for extracted_item in component.extracted_licenses:
            previous = extracted.setdefault(
                extracted_item.license_id, extracted_item.extracted_text
            )
            if previous != extracted_item.extracted_text:
                raise RuntimeMetadataError("runtime_extracted_license_conflict")
    packages: list[dict[str, object]] = []
    for licensed_component in components:
        package: dict[str, object] = {
            "SPDXID": _spdx_id(licensed_component),
            "copyrightText": "NOASSERTION",
            "downloadLocation": licensed_component.source_uri,
            "filesAnalyzed": False,
            "licenseConcluded": licensed_component.license_id,
            "licenseDeclared": licensed_component.license_id,
            "name": licensed_component.actual.name,
            "primaryPackagePurpose": (
                "LIBRARY" if licensed_component.actual.scope == _PACKAGE_SCOPE else "APPLICATION"
            ),
            "summary": f"customs-runtime-scope:{licensed_component.actual.scope}",
            "versionInfo": licensed_component.actual.version,
        }
        if licensed_component.actual.checksum_sha256 is not None:
            package["checksums"] = [
                {
                    "algorithm": "SHA256",
                    "checksumValue": licensed_component.actual.checksum_sha256,
                }
            ]
        packages.append(package)
    result: dict[str, object] = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: customs-raster-text-oci-metadata/1"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://spdx.org/spdxdocs/customs-raster-text-{namespace_sha}",
        "name": f"customs-raster-text-runtime-{profile_revision}",
        "packages": packages,
        "relationships": [
            {
                "relatedSpdxElement": _spdx_id(item),
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            }
            for item in components
        ],
        "spdxVersion": "SPDX-2.3",
    }
    if extracted:
        result["hasExtractedLicensingInfos"] = [
            {"extractedText": text, "licenseId": license_id}
            for license_id, text in sorted(extracted.items())
        ]
    return result


def build_runtime_metadata(
    root: Path,
    profile: Mapping[str, Any],
    *,
    policy_path: Path,
    distribution_finder: DistributionFinder = _default_distribution_finder,
) -> RuntimeMetadataArtifacts:
    components, evidence, generated_evidence, policy = licensed_runtime_components(
        root,
        profile,
        policy_path=policy_path,
        distribution_finder=distribution_finder,
    )
    revision = profile.get("raster_text_profile_revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeMetadataError("runtime_profile_invalid")
    return RuntimeMetadataArtifacts(
        license_manifest=canonical_json_bytes(_license_manifest(components)),
        sbom=canonical_json_bytes(_spdx(components, profile_revision=revision)),
        source_evidence_paths=tuple(sorted(evidence, key=lambda item: item.as_posix())),
        generated_evidence=generated_evidence,
        policy=policy,
    )


def _manifest_component_identities(value: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    if (
        set(value) != {"components", "distribution_scope", "schema_version"}
        or value.get("schema_version") != "1.0"
        or value.get("distribution_scope") != "private-ghcr-only"
    ):
        raise RuntimeMetadataError("license_manifest_invalid")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise RuntimeMetadataError("license_manifest_invalid")
    result: set[tuple[str, str, str]] = set()
    order: list[tuple[str, str, str]] = []
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "checksum_sha256",
            "license_id",
            "name",
            "scope",
            "source_uri",
            "version",
        }:
            raise RuntimeMetadataError("license_manifest_invalid")
        name = component.get("name")
        version = component.get("version")
        scope = component.get("scope")
        checksum = component.get("checksum_sha256")
        if (
            not isinstance(name, str)
            or _SAFE_NAME.fullmatch(name) is None
            or not isinstance(version, str)
            or not version.strip()
            or scope not in {_MODEL_SCOPE, _PACKAGE_SCOPE, _RUNTIME_SCOPE}
            or (
                checksum is not None
                and (not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None)
            )
            or (scope == _PACKAGE_SCOPE and checksum is not None)
            or (scope != _PACKAGE_SCOPE and checksum is None)
        ):
            raise RuntimeMetadataError("license_manifest_invalid")
        _valid_license_declaration(
            component.get("license_id"), allow_noassertion=scope == _MODEL_SCOPE
        )
        _valid_component_source(
            component.get("source_uri"), allow_noassertion=scope == _MODEL_SCOPE
        )
        identity = (str(scope), _normalized_name(name), version)
        if identity in result:
            raise RuntimeMetadataError("license_manifest_invalid")
        result.add(identity)
        order.append(identity)
    if order != sorted(order):
        raise RuntimeMetadataError("license_manifest_invalid")
    return result


def validate_metadata_component_sets(
    root: Path,
    profile: Mapping[str, Any],
    *,
    license_manifest_path: Path,
    sbom_path: Path,
    distribution_finder: DistributionFinder = _default_distribution_finder,
) -> frozenset[PurePosixPath]:
    license_raw = _read_bytes(
        license_manifest_path, maximum=_MAX_JSON_BYTES, code="license_manifest_invalid"
    )
    sbom_raw = _read_bytes(sbom_path, maximum=_MAX_JSON_BYTES, code="runtime_sbom_invalid")
    try:
        licenses = json.loads(license_raw.decode("utf-8"), parse_constant=_reject_json_constant)
        sbom = json.loads(sbom_raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeMetadataError("runtime_metadata_invalid") from None
    if (
        not isinstance(licenses, dict)
        or license_raw != canonical_json_bytes(licenses)
        or not isinstance(sbom, dict)
        or sbom_raw != canonical_json_bytes(sbom)
    ):
        raise RuntimeMetadataError("runtime_metadata_not_canonical")
    license_identities = _manifest_component_identities(licenses)
    actual_components = actual_runtime_components(root, profile)
    actual_by_identity = {item.identity: item for item in actual_components}
    expected = set(actual_by_identity)
    if license_identities != expected:
        raise RuntimeMetadataError("runtime_license_component_set_mismatch")
    license_by_identity = {
        (str(item["scope"]), _normalized_name(str(item["name"])), str(item["version"])): item
        for item in licenses["components"]
    }

    # Re-derive every legal claim from raw evidence carried inside this tree.
    evidence_components: list[LicensedComponent] = []
    sidecar_distributions = _sidecar_distributions(root, profile)
    for (name, version), distribution in sidecar_distributions.items():
        identity = (_PACKAGE_SCOPE, name, version)
        actual = actual_by_identity.get(identity)
        if actual is None:
            raise RuntimeMetadataError("runtime_package_metadata_set_mismatch")
        distribution_evidence = _distribution_evidence(distribution)
        license_id, extracted = _distribution_license(distribution_evidence)
        evidence_components.append(
            LicensedComponent(
                actual,
                license_id,
                _metadata_source_uri(distribution_evidence),
                extracted,
            )
        )
    parent_only_identities = {
        identity
        for identity in expected
        if identity[0] == _PACKAGE_SCOPE and (identity[1], identity[2]) not in sidecar_distributions
    }
    parent_evidence_path = root.joinpath(*_PARENT_EVIDENCE_RELATIVE_PATH.parts)
    if parent_only_identities:
        parent_evidence = _read_parent_evidence(root)
        observed_parent = {
            (_PACKAGE_SCOPE, _normalized_name(item.name), item.version) for item in parent_evidence
        }
        if observed_parent != parent_only_identities:
            raise RuntimeMetadataError("runtime_parent_package_evidence_set_mismatch")
        for item in parent_evidence:
            identity = (_PACKAGE_SCOPE, _normalized_name(item.name), item.version)
            actual = actual_by_identity[identity]
            matches = [
                distribution
                for distribution in distribution_finder(item.name)
                if _normalized_name(
                    _distribution_name(
                        distribution,
                        code="runtime_parent_package_metadata_missing",
                    )
                )
                == identity[1]
                and distribution.version == identity[2]
            ]
            if len(matches) != 1:
                raise RuntimeMetadataError("runtime_parent_package_metadata_missing")
            if _distribution_evidence(matches[0]) != item:
                raise RuntimeMetadataError("runtime_parent_package_evidence_mismatch")
            license_id, extracted = _distribution_license(item)
            evidence_components.append(
                LicensedComponent(
                    actual,
                    license_id,
                    _metadata_source_uri(item),
                    extracted,
                )
            )
    elif parent_evidence_path.exists() or parent_evidence_path.is_symlink():
        raise RuntimeMetadataError("runtime_parent_package_evidence_set_mismatch")
    models = [item for item in actual_components if item.scope == _MODEL_SCOPE]
    detector = next((item for item in models if item.name == "PP-OCRv5_mobile_det"), None)
    recognizer = next(
        (item for item in models if item.name == "source_attested_en_PP-OCRv5_mobile_rec_head_v2"),
        None,
    )
    if detector is None or recognizer is None or len(models) != 2:
        raise RuntimeMetadataError("runtime_model_inventory_invalid")
    detector_license, detector_evidence = _detector_license(root, profile, detector)
    evidence_components.append(detector_license)
    policy_path = _safe_file(root, _POLICY_RELATIVE_PATH, code="runtime_license_policy_invalid")
    policy, _policy_raw = _read_policy(policy_path)
    evidence_components.append(_recognizer_license(root, profile, recognizer, policy=policy))
    python_actual, _python_license_path, python_license_text = _python_component(root, profile)
    evidence_components.append(
        _python_license(
            python_actual,
            python_license_text,
            policy=policy,
        )
    )
    expected_extracted: dict[str, str] = {}
    for component in evidence_components:
        declared = license_by_identity.get(component.actual.identity)
        if (
            declared is None
            or declared.get("license_id") != component.license_id
            or declared.get("source_uri") != component.source_uri
            or declared.get("checksum_sha256") != component.actual.checksum_sha256
        ):
            raise RuntimeMetadataError("runtime_license_evidence_mismatch")
        for extracted_license in component.extracted_licenses:
            previous = expected_extracted.setdefault(
                extracted_license.license_id, extracted_license.extracted_text
            )
            if previous != extracted_license.extracted_text:
                raise RuntimeMetadataError("runtime_extracted_license_conflict")

    packages = sbom.get("packages")
    relationships = sbom.get("relationships")
    revision = profile.get("raster_text_profile_revision")
    component_payload = [
        {
            "checksum_sha256": item["checksum_sha256"],
            "license_id": item["license_id"],
            "name": item["name"],
            "scope": item["scope"],
            "source_uri": item["source_uri"],
            "version": item["version"],
        }
        for item in licenses["components"]
    ]
    namespace_sha = hashlib.sha256(
        canonical_json_bytes({"components": component_payload, "profile_revision": revision})
    ).hexdigest()
    expected_top_level = {
        "SPDXID",
        "creationInfo",
        "dataLicense",
        "documentNamespace",
        "name",
        "packages",
        "relationships",
        "spdxVersion",
    }
    if "hasExtractedLicensingInfos" in sbom:
        expected_top_level.add("hasExtractedLicensingInfos")
    if (
        set(sbom) != expected_top_level
        or sbom.get("spdxVersion") != "SPDX-2.3"
        or sbom.get("dataLicense") != "CC0-1.0"
        or sbom.get("SPDXID") != "SPDXRef-DOCUMENT"
        or sbom.get("creationInfo")
        != {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: customs-raster-text-oci-metadata/1"],
        }
        or sbom.get("name") != f"customs-raster-text-runtime-{revision}"
        or sbom.get("documentNamespace")
        != f"https://spdx.org/spdxdocs/customs-raster-text-{namespace_sha}"
        or not isinstance(packages, list)
        or not isinstance(relationships, list)
        or len(packages) != len(expected)
        or len(relationships) != len(expected)
    ):
        raise RuntimeMetadataError("runtime_sbom_invalid")
    declared_references: set[str] = set()
    for item in licenses["components"]:
        declared_references.update(
            re.findall(r"LicenseRef-[A-Za-z0-9.+-]+", str(item["license_id"]))
        )
    extracted_infos = sbom.get("hasExtractedLicensingInfos", [])
    if not isinstance(extracted_infos, list):
        raise RuntimeMetadataError("runtime_sbom_invalid")
    extracted_by_id: dict[str, str] = {}
    extracted_order: list[str] = []
    for item in extracted_infos:
        if (
            not isinstance(item, dict)
            or set(item) != {"extractedText", "licenseId"}
            or not isinstance(item.get("licenseId"), str)
            or not isinstance(item.get("extractedText"), str)
            or not item["extractedText"].strip()
            or item["licenseId"] in extracted_by_id
        ):
            raise RuntimeMetadataError("runtime_sbom_invalid")
        license_id = item["licenseId"]
        text = item["extractedText"]
        if license_id.startswith("LicenseRef-PackageEvidence-") and hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest() != license_id.removeprefix("LicenseRef-PackageEvidence-"):
            raise RuntimeMetadataError("runtime_extracted_license_mismatch")
        extracted_by_id[license_id] = text
        extracted_order.append(license_id)
    if (
        set(extracted_by_id) != declared_references
        or extracted_order != sorted(extracted_order)
        or bool(declared_references) != ("hasExtractedLicensingInfos" in sbom)
    ):
        raise RuntimeMetadataError("runtime_extracted_license_mismatch")
    for license_id, text in expected_extracted.items():
        if extracted_by_id.get(license_id) != text:
            raise RuntimeMetadataError("runtime_extracted_license_mismatch")

    spdx_identities: set[tuple[str, str, str]] = set()
    spdx_ids: set[str] = set()
    package_order: list[tuple[str, str, str]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeMetadataError("runtime_sbom_invalid")
        summary = package.get("summary")
        if not isinstance(summary, str) or not summary.startswith("customs-runtime-scope:"):
            raise RuntimeMetadataError("runtime_sbom_invalid")
        scope = summary.removeprefix("customs-runtime-scope:")
        expected_package_fields = {
            "SPDXID",
            "copyrightText",
            "downloadLocation",
            "filesAnalyzed",
            "licenseConcluded",
            "licenseDeclared",
            "name",
            "primaryPackagePurpose",
            "summary",
            "versionInfo",
        }
        if scope != _PACKAGE_SCOPE:
            expected_package_fields.add("checksums")
        if set(package) != expected_package_fields:
            raise RuntimeMetadataError("runtime_sbom_invalid")
        package_name = package.get("name")
        package_version = package.get("versionInfo")
        spdx_id = package.get("SPDXID")
        if (
            scope not in {_MODEL_SCOPE, _PACKAGE_SCOPE, _RUNTIME_SCOPE}
            or not isinstance(package_name, str)
            or not isinstance(package_version, str)
            or not isinstance(spdx_id, str)
            or not spdx_id.startswith("SPDXRef-Package-")
            or spdx_id in spdx_ids
            or package.get("filesAnalyzed") is not False
            or package.get("copyrightText") != "NOASSERTION"
        ):
            raise RuntimeMetadataError("runtime_sbom_invalid")
        identity = (scope, _normalized_name(package_name), package_version)
        licensed = license_by_identity.get(identity)
        expected_spdx_id = (
            f"SPDXRef-Package-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()}"
        )
        if (
            licensed is None
            or package.get("name") != licensed["name"]
            or package.get("licenseDeclared") != licensed["license_id"]
            or package.get("licenseConcluded") != licensed["license_id"]
            or package.get("downloadLocation") != licensed["source_uri"]
            or spdx_id != expected_spdx_id
            or package.get("summary") != f"customs-runtime-scope:{scope}"
            or package.get("primaryPackagePurpose")
            != ("LIBRARY" if scope == _PACKAGE_SCOPE else "APPLICATION")
            or (
                scope != _PACKAGE_SCOPE
                and package.get("checksums")
                != [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": licensed["checksum_sha256"],
                    }
                ]
            )
            or identity in spdx_identities
        ):
            raise RuntimeMetadataError("runtime_sbom_component_mismatch")
        spdx_identities.add(identity)
        spdx_ids.add(spdx_id)
        package_order.append(identity)
    if spdx_identities != expected:
        raise RuntimeMetadataError("runtime_sbom_component_mismatch")
    if package_order != sorted(package_order):
        raise RuntimeMetadataError("runtime_sbom_component_mismatch")
    expected_relationships = [
        {
            "relatedSpdxElement": package["SPDXID"],
            "relationshipType": "DESCRIBES",
            "spdxElementId": "SPDXRef-DOCUMENT",
        }
        for package in packages
    ]
    if relationships != expected_relationships or len(spdx_ids) != len(relationships):
        raise RuntimeMetadataError("runtime_sbom_component_mismatch")
    expected_evidence = {_POLICY_RELATIVE_PATH, detector_evidence}
    if parent_only_identities:
        expected_evidence.add(_PARENT_EVIDENCE_RELATIVE_PATH)
    return frozenset(expected_evidence)


__all__ = [
    "ActualComponent",
    "LicensedComponent",
    "RuntimeMetadataArtifacts",
    "RuntimeMetadataError",
    "_POLICY_RELATIVE_PATH",
    "actual_runtime_components",
    "build_runtime_metadata",
    "canonical_json_bytes",
    "validate_metadata_component_sets",
]
