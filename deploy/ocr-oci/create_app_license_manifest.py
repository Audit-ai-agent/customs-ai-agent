"""Create a deterministic license inventory from the installed application venv."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

_CLASSIFIER_LICENSES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
_FIRST_PARTY = "customs-ai-agent"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LicenseInventoryError(RuntimeError):
    pass


def _license(metadata: importlib.metadata.PackageMetadata) -> str:
    expression = metadata.get("License-Expression") or metadata.get("License")
    if isinstance(expression, str) and expression.strip():
        return " ".join(expression.split())
    classifiers = metadata.get_all("Classifier") or []
    matched = [_CLASSIFIER_LICENSES[item] for item in classifiers if item in _CLASSIFIER_LICENSES]
    if len(set(matched)) == 1:
        return matched[0]
    raise LicenseInventoryError("app_package_license_missing")


def _source_uri(metadata: importlib.metadata.PackageMetadata) -> str:
    for item in metadata.get_all("Project-URL") or []:
        label, separator, value = item.partition(",")
        if separator and label.strip().casefold() in {
            "homepage",
            "repository",
            "source",
            "source code",
        }:
            if value.strip().startswith("https://"):
                return value.strip()
    homepage = metadata.get("Home-page")
    if isinstance(homepage, str) and homepage.startswith("https://"):
        return homepage
    raise LicenseInventoryError("app_package_source_missing")


def build_inventory() -> dict[str, object]:
    packages: list[dict[str, str]] = []
    observed: set[str] = set()
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = metadata.get("Name")
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise LicenseInventoryError("app_package_name_invalid")
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if normalized in observed:
            raise LicenseInventoryError("app_package_duplicate")
        observed.add(normalized)
        version = distribution.version
        if normalized == _FIRST_PARTY:
            packages.append(
                {
                    "license_id": "LicenseRef-Project-Internal",
                    "name": normalized,
                    "scope": "first-party",
                    "source_uri": "https://github.com/Audit-ai-agent/customs-ai-agent",
                    "version": version,
                }
            )
            continue
        packages.append(
            {
                "license_id": _license(metadata),
                "name": normalized,
                "scope": "python-package",
                "source_uri": _source_uri(metadata),
                "version": version,
            }
        )
    packages.sort(key=lambda item: (item["name"], item["version"]))
    if not packages or not any(item["name"] == _FIRST_PARTY for item in packages):
        raise LicenseInventoryError("first_party_package_missing")
    return {"components": packages, "schema_version": "1.0"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        value = build_inventory()
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        args.output.write_bytes(payload)
    except (LicenseInventoryError, OSError) as exc:
        code = (
            str(exc) if isinstance(exc, LicenseInventoryError) else "license_inventory_write_failed"
        )
        print(code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
