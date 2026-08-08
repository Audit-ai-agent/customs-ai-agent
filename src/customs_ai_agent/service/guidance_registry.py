"""Fail-closed consumer view of Data's official-guidance registry manifest.

The Data repository owns the source registry.  This package ships an exact
consumer copy of its safe manifest so a Data response cannot introduce an
unregistered source, topic, direction, title, or URI into the Agent boundary.
The manifest contains source metadata only; it never carries document text or
official-source bodies.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal, Mapping, cast
from urllib.parse import unquote, urlsplit


GuidanceProfileMode = Literal["local-development", "functional-uat"]
GuidanceDirection = Literal["import", "export", "both"]
GuidanceSourceKind = Literal[
    "official_law",
    "official_law_catalog",
    "official_agency_guidance",
    "official_requirement_reference",
    "official_public_guidance",
]

_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_TOPICS = frozenset(
    {
        "customs_clearance",
        "customs_declaration",
        "customs_duty",
        "hs_classification",
        "origin_verification",
        "trade_documents",
        "import_requirements",
        "export_requirements",
        "transport_documents",
        "fta_procedure",
        "trade_terms",
    }
)
_SOURCE_KINDS = frozenset(
    {
        "official_law",
        "official_law_catalog",
        "official_agency_guidance",
        "official_requirement_reference",
        "official_public_guidance",
    }
)
_AVAILABILITY = frozenset({"available_in_legacy_3law", "requires_kr_trade_snapshot"})


class GuidanceRegistryError(ValueError):
    """A content-free registry or provenance validation failure."""


@dataclass(frozen=True, slots=True)
class GuidanceRegistrySource:
    source_id: str
    source_kind: GuidanceSourceKind
    title: str
    official_uri: str
    directions: frozenset[Literal["import", "export"]]
    topics: frozenset[str]
    law_id: str | None

    def supports(self, *, topic: str, direction: GuidanceDirection) -> bool:
        if topic not in self.topics:
            return False
        return direction == "both" or direction in self.directions


@dataclass(frozen=True, slots=True)
class GuidanceRegistry:
    registry_sha256: str
    manifest_sha256: str
    sources_by_id: Mapping[str, GuidanceRegistrySource]
    topic_source_ids: Mapping[str, frozenset[str]]
    topic_directions: Mapping[str, frozenset[Literal["import", "export"]]]
    topic_titles: Mapping[str, str]


def _manifest_bytes() -> bytes:
    try:
        return (
            resources.files("customs_ai_agent.contracts")
            .joinpath("evidence/v3/guidance_registry_v1.manifest.json")
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise GuidanceRegistryError("guidance_registry_manifest_missing") from exc


def _https_uri(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_000:
        raise GuidanceRegistryError("guidance_registry_manifest_invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise GuidanceRegistryError("guidance_registry_manifest_invalid")
    return value


def _string_set(
    value: object,
    *,
    allowed: frozenset[str],
    code: str,
) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or item not in allowed for item in value)
        or len(value) != len(set(value))
    ):
        raise GuidanceRegistryError(code)
    return frozenset(value)


def _load_registry() -> GuidanceRegistry:
    raw = _manifest_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuidanceRegistryError("guidance_registry_manifest_invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "coverage_profiles",
        "jurisdiction",
        "registry_sha256",
        "schema_version",
        "sources",
        "topics",
    }:
        raise GuidanceRegistryError("guidance_registry_manifest_invalid")
    registry_sha256 = payload.get("registry_sha256")
    sources = payload.get("sources")
    topics = payload.get("topics")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("jurisdiction") != "KR"
        or not isinstance(registry_sha256, str)
        or _SHA256.fullmatch(registry_sha256) is None
        or not isinstance(sources, list)
        or not sources
        or not isinstance(topics, list)
        or not topics
    ):
        raise GuidanceRegistryError("guidance_registry_manifest_invalid")

    sources_by_id: dict[str, GuidanceRegistrySource] = {}
    for item in sources:
        if not isinstance(item, dict):
            raise GuidanceRegistryError("guidance_registry_manifest_invalid")
        source_id = item.get("source_id")
        source_kind = item.get("source_kind")
        title = item.get("title")
        law_id = item.get("law_id")
        directions = _string_set(
            item.get("directions"),
            allowed=frozenset({"import", "export"}),
            code="guidance_registry_manifest_invalid",
        )
        source_topics = _string_set(
            item.get("topics"),
            allowed=_TOPICS,
            code="guidance_registry_manifest_invalid",
        )
        required = {
            "authority",
            "availability",
            "directions",
            "jurisdiction",
            "official_uri",
            "source_id",
            "source_kind",
            "title",
            "topics",
        }
        if source_kind == "official_law":
            required.add("law_id")
        if (
            set(item) != required
            or not isinstance(source_id, str)
            or _SOURCE_ID.fullmatch(source_id) is None
            or source_id in sources_by_id
            or source_kind not in _SOURCE_KINDS
            or not isinstance(title, str)
            or not title
            or not isinstance(item.get("authority"), str)
            or not item["authority"]
            or item.get("availability") not in _AVAILABILITY
            or item.get("jurisdiction") != "KR"
            or source_kind == "official_law"
            and (not isinstance(law_id, str) or not law_id)
            or source_kind != "official_law"
            and law_id is not None
        ):
            raise GuidanceRegistryError("guidance_registry_manifest_invalid")
        official_uri = _https_uri(item.get("official_uri"))
        sources_by_id[source_id] = GuidanceRegistrySource(
            source_id=source_id,
            source_kind=source_kind,  # type: ignore[arg-type]
            title=title,
            official_uri=official_uri,
            directions=directions,  # type: ignore[arg-type]
            topics=source_topics,
            law_id=law_id if isinstance(law_id, str) else None,
        )

    topic_source_ids: dict[str, frozenset[str]] = {}
    topic_directions: dict[str, frozenset[Literal["import", "export"]]] = {}
    topic_titles: dict[str, str] = {}
    for item in topics:
        if not isinstance(item, dict) or set(item) != {
            "directions",
            "source_ids",
            "title",
            "topic",
        }:
            raise GuidanceRegistryError("guidance_registry_manifest_invalid")
        topic = item.get("topic")
        source_ids = item.get("source_ids")
        directions = _string_set(
            item.get("directions"),
            allowed=frozenset({"import", "export"}),
            code="guidance_registry_manifest_invalid",
        )
        if (
            not isinstance(topic, str)
            or topic not in _TOPICS
            or topic in topic_source_ids
            or not isinstance(item.get("title"), str)
            or not item["title"]
            or not isinstance(source_ids, list)
            or not source_ids
            or any(
                not isinstance(source_id, str) or source_id not in sources_by_id
                for source_id in source_ids
            )
            or len(source_ids) != len(set(source_ids))
        ):
            raise GuidanceRegistryError("guidance_registry_manifest_invalid")
        source_id_set = frozenset(source_ids)
        safe_directions = cast(frozenset[Literal["import", "export"]], directions)
        if not any(
            sources_by_id[source_id].supports(topic=topic, direction=direction)
            for source_id in source_id_set
            for direction in safe_directions
        ):
            raise GuidanceRegistryError("guidance_registry_manifest_invalid")
        topic_source_ids[topic] = source_id_set
        topic_directions[topic] = safe_directions
        topic_titles[topic] = item["title"]
    if frozenset(topic_source_ids) != _TOPICS:
        raise GuidanceRegistryError("guidance_registry_manifest_invalid")
    return GuidanceRegistry(
        registry_sha256=registry_sha256,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        sources_by_id=sources_by_id,
        topic_source_ids=topic_source_ids,
        topic_directions=topic_directions,
        topic_titles=topic_titles,
    )


@lru_cache(maxsize=1)
def guidance_registry() -> GuidanceRegistry:
    """Return the one strict, packaged provider-manifest view."""

    return _load_registry()


def _law_uri_matches(uri: str, source: GuidanceRegistrySource, article_no: str) -> bool:
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.hostname.casefold().removeprefix("www.") != "law.go.kr"
        # An explicit port is a distinct URL spelling.  Match the sibling
        # ingress canonicalisation boundary: neither a numeric port (even
        # ``:443``) nor the parser-normalised empty-port spelling is allowed.
        or port is not None
        or ":" in parsed.netloc.rsplit("@", 1)[-1]
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    path = unquote(parsed.path)
    segments = [segment for segment in path.split("/") if segment]
    return len(segments) == 3 and segments == ["법령", source.title, article_no]


def validate_guidance_source_provenance(
    *,
    profile_mode: GuidanceProfileMode,
    registry_sha256: str,
    registry_source_id: str | None,
    source_kind: str,
    title: str,
    uri: str,
    topic: str,
    direction: str,
    law_id: str | None,
    law_name: str | None,
    article_no: str | None,
) -> None:
    """Fail closed unless one Data source exactly fits the copied registry.

    Data must additionally bind a response to the source-registry digest and
    a source record to one registry source ID.  Official law articles use a
    canonical public ``law.go.kr/법령/<law>/<article>`` URI, while all other
    official sources must exactly preserve the registered title and URI.
    ``development_fixture`` remains a clearly synthetic local-development
    exception and is never accepted from Functional-UAT.
    """

    registry = guidance_registry()
    if registry_sha256 != registry.registry_sha256:
        raise GuidanceRegistryError("guidance_registry_sha256_mismatch")
    if direction not in {"import", "export", "both"}:
        raise GuidanceRegistryError("guidance_registry_source_invalid")
    safe_direction = cast(GuidanceDirection, direction)
    topic_directions = registry.topic_directions.get(topic)
    if (
        topic_directions is None
        or (safe_direction == "both" and topic_directions != frozenset({"import", "export"}))
        or (safe_direction != "both" and safe_direction not in topic_directions)
    ):
        raise GuidanceRegistryError("guidance_registry_direction_invalid")
    if source_kind == "development_fixture":
        if profile_mode != "local-development" or not isinstance(registry_source_id, str):
            raise GuidanceRegistryError("guidance_fixture_provenance_invalid")
        source = registry.sources_by_id.get(registry_source_id)
        if (
            topic not in _TOPICS
            or source is None
            or registry_source_id not in registry.topic_source_ids[topic]
            or not source.supports(topic=topic, direction=safe_direction)
        ):
            raise GuidanceRegistryError("guidance_fixture_provenance_invalid")
        expected_uri = f"https://customs.local.invalid/dev-fixture/guidance/{topic}"
        if uri != expected_uri or title != registry.topic_titles[topic]:
            raise GuidanceRegistryError("guidance_fixture_provenance_invalid")
        return
    if profile_mode not in {"local-development", "functional-uat"}:
        raise GuidanceRegistryError("guidance_profile_mode_invalid")
    if not isinstance(registry_source_id, str):
        raise GuidanceRegistryError("guidance_registry_source_missing")
    source = registry.sources_by_id.get(registry_source_id)
    if (
        source is None
        or source.source_kind != source_kind
        or topic not in registry.topic_source_ids
        or registry_source_id not in registry.topic_source_ids[topic]
        or not source.supports(topic=topic, direction=safe_direction)
    ):
        raise GuidanceRegistryError("guidance_registry_source_invalid")
    if source.source_kind == "official_law":
        if (
            law_id != source.law_id
            or law_name != source.title
            or not isinstance(article_no, str)
            or not article_no
            or not title.startswith(f"{source.title} {article_no}")
            or not _law_uri_matches(uri, source, article_no)
        ):
            raise GuidanceRegistryError("guidance_law_provenance_invalid")
        return
    if (
        law_id is not None
        or law_name is not None
        or article_no is not None
        or title != source.title
        or uri != source.official_uri
    ):
        raise GuidanceRegistryError("guidance_official_provenance_invalid")


__all__ = [
    "GuidanceRegistry",
    "GuidanceRegistryError",
    "GuidanceRegistrySource",
    "guidance_registry",
    "validate_guidance_source_provenance",
]
