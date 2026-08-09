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
from dataclasses import dataclass, field
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
_V2_SCOPES = frozenset(
    {
        "import_clearance_flow",
        "export_clearance_flow",
        "import_declaration",
        "export_declaration",
        "customs_value_basics",
        "duty_payment_basics",
        "classification_principles",
        "advance_ruling_or_confirmation",
        "origin_evidence",
        "origin_verification",
        "commercial_and_declaration_documents",
        "agency_confirmation_and_supporting_documents",
        "bill_of_lading",
        "air_waybill",
        "origin_certificate",
        "preference_procedure",
        "cost_risk_delivery_terms",
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
    scope_bindings: Mapping[tuple[str, str], frozenset[Literal["import", "export"]]] = field(
        default_factory=dict
    )

    def supports(
        self,
        *,
        topic: str,
        direction: GuidanceDirection,
        scope: str | None = None,
    ) -> bool:
        if topic not in self.topics:
            return False
        if scope is not None:
            binding_directions = self.scope_bindings.get((topic, scope))
            if binding_directions is None:
                return False
            return (
                frozenset({"import", "export"}) <= binding_directions
                if direction == "both"
                else direction in binding_directions
            )
        return (
            frozenset({"import", "export"}) <= self.directions
            if direction == "both"
            else direction in self.directions
        )


@dataclass(frozen=True, slots=True)
class GuidanceRegistry:
    registry_version: Literal["1.0", "2.0"]
    registry_sha256: str
    manifest_sha256: str
    sources_by_id: Mapping[str, GuidanceRegistrySource]
    topic_source_ids: Mapping[str, frozenset[str]]
    topic_directions: Mapping[str, frozenset[Literal["import", "export"]]]
    topic_titles: Mapping[str, str]
    scope_directions: Mapping[tuple[str, str], frozenset[Literal["import", "export"]]] = field(
        default_factory=dict
    )
    scope_titles: Mapping[tuple[str, str], str] = field(default_factory=dict)
    scope_required_source_ids: Mapping[tuple[str, str], frozenset[str]] = field(
        default_factory=dict
    )


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
        registry_version="1.0",
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


def _v2_manifest_bytes() -> bytes:
    try:
        return (
            resources.files("customs_ai_agent.contracts")
            .joinpath("evidence/v3/guidance_registry_v2.manifest.json")
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise GuidanceRegistryError("guidance_registry_v2_manifest_missing") from exc


def _load_registry_v2() -> GuidanceRegistry:
    """Load Data's V2 page-level consumer manifest without source bodies."""

    raw = _v2_manifest_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid") from exc
    required_root = {
        "coverage_profiles",
        "jurisdiction",
        "registry_sha256",
        "schema_version",
        "sources",
        "topics",
    }
    if not isinstance(payload, dict) or set(payload) != required_root:
        raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
    registry_sha256 = payload.get("registry_sha256")
    sources = payload.get("sources")
    topics = payload.get("topics")
    if (
        payload.get("schema_version") != "2.0"
        or payload.get("jurisdiction") != "KR"
        or not isinstance(registry_sha256, str)
        or _SHA256.fullmatch(registry_sha256) is None
        or not isinstance(sources, list)
        or not sources
        or not isinstance(topics, list)
        or not topics
    ):
        raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")

    sources_by_id: dict[str, GuidanceRegistrySource] = {}
    for item in sources:
        required_source = {
            "authority",
            "availability",
            "directions",
            "jurisdiction",
            "official_uri",
            "scopes",
            "source_id",
            "source_kind",
            "title",
            "topics",
        }
        if not isinstance(item, dict) or set(item) != required_source:
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        source_id = item.get("source_id")
        source_kind = item.get("source_kind")
        title = item.get("title")
        if (
            not isinstance(source_id, str)
            or _SOURCE_ID.fullmatch(source_id) is None
            or not source_id.startswith("v2_")
            or source_id in sources_by_id
            or source_kind not in _SOURCE_KINDS
            or not isinstance(title, str)
            or not title
            or not isinstance(item.get("authority"), str)
            or not item["authority"]
            or item.get("availability") != "requires_kr_trade_snapshot"
            or item.get("jurisdiction") != "KR"
        ):
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        directions = _string_set(
            item.get("directions"),
            allowed=frozenset({"import", "export"}),
            code="guidance_registry_v2_manifest_invalid",
        )
        source_topics = _string_set(
            item.get("topics"),
            allowed=_TOPICS,
            code="guidance_registry_v2_manifest_invalid",
        )
        raw_bindings = item.get("scopes")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        bindings: dict[tuple[str, str], frozenset[Literal["import", "export"]]] = {}
        for binding in raw_bindings:
            if not isinstance(binding, dict) or set(binding) != {"topic", "scope", "directions"}:
                raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
            topic = binding.get("topic")
            scope = binding.get("scope")
            binding_directions = _string_set(
                binding.get("directions"),
                allowed=frozenset({"import", "export"}),
                code="guidance_registry_v2_manifest_invalid",
            )
            if (
                not isinstance(topic, str)
                or topic not in source_topics
                or not isinstance(scope, str)
                or scope not in _V2_SCOPES
                or (topic, scope) in bindings
                or not binding_directions <= directions
            ):
                raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
            bindings[(topic, scope)] = cast(
                frozenset[Literal["import", "export"]], binding_directions
            )
        if source_topics != frozenset(topic for topic, _scope in bindings):
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        sources_by_id[source_id] = GuidanceRegistrySource(
            source_id=source_id,
            source_kind=cast(GuidanceSourceKind, source_kind),
            title=title,
            official_uri=_https_uri(item.get("official_uri")),
            directions=cast(frozenset[Literal["import", "export"]], directions),
            topics=source_topics,
            law_id=None,
            scope_bindings=bindings,
        )

    topic_source_ids: dict[str, frozenset[str]] = {}
    topic_directions: dict[str, frozenset[Literal["import", "export"]]] = {}
    topic_titles: dict[str, str] = {}
    scope_directions: dict[tuple[str, str], frozenset[Literal["import", "export"]]] = {}
    scope_titles: dict[tuple[str, str], str] = {}
    scope_required_source_ids: dict[tuple[str, str], frozenset[str]] = {}
    for item in topics:
        if not isinstance(item, dict) or set(item) != {"topic", "title", "scopes"}:
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        topic = item.get("topic")
        title = item.get("title")
        raw_scopes = item.get("scopes")
        if (
            not isinstance(topic, str)
            or topic not in _TOPICS
            or topic in topic_titles
            or not isinstance(title, str)
            or not title
            or not isinstance(raw_scopes, list)
            or not raw_scopes
        ):
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        topic_titles[topic] = title
        topic_source_ids[topic] = frozenset()
        topic_direction_values: set[Literal["import", "export"]] = set()
        for scope_item in raw_scopes:
            if not isinstance(scope_item, dict) or set(scope_item) != {
                "directions",
                "required_source_ids",
                "scope",
                "title",
            }:
                raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
            scope = scope_item.get("scope")
            scope_title = scope_item.get("title")
            key = (topic, scope) if isinstance(scope, str) else None
            directions = _string_set(
                scope_item.get("directions"),
                allowed=frozenset({"import", "export"}),
                code="guidance_registry_v2_manifest_invalid",
            )
            required_ids = scope_item.get("required_source_ids")
            if (
                not isinstance(scope, str)
                or scope not in _V2_SCOPES
                or key is None
                or key in scope_directions
                or not isinstance(scope_title, str)
                or not scope_title
                or not isinstance(required_ids, list)
                or not required_ids
                or len(required_ids) != len(set(required_ids))
                or any(
                    not isinstance(source_id, str) or source_id not in sources_by_id
                    for source_id in required_ids
                )
            ):
                raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
            required = frozenset(cast(str, source_id) for source_id in required_ids)
            safe_directions = cast(frozenset[Literal["import", "export"]], directions)
            if any(
                not sources_by_id[source_id].supports(topic=topic, scope=scope, direction=direction)
                for source_id in required
                for direction in safe_directions
            ):
                raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
            scope_directions[key] = safe_directions
            scope_titles[key] = scope_title
            scope_required_source_ids[key] = required
            topic_source_ids[topic] = frozenset(set(topic_source_ids[topic]) | set(required))
            topic_direction_values.update(safe_directions)
        if not topic_direction_values:
            raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
        topic_directions[topic] = frozenset(topic_direction_values)
    if frozenset(topic_titles) != _TOPICS:
        raise GuidanceRegistryError("guidance_registry_v2_manifest_invalid")
    return GuidanceRegistry(
        registry_version="2.0",
        registry_sha256=registry_sha256,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        sources_by_id=sources_by_id,
        topic_source_ids=topic_source_ids,
        topic_directions=topic_directions,
        topic_titles=topic_titles,
        scope_directions=scope_directions,
        scope_titles=scope_titles,
        scope_required_source_ids=scope_required_source_ids,
    )


@lru_cache(maxsize=1)
def guidance_registry_v2() -> GuidanceRegistry:
    return _load_registry_v2()


def guidance_registry_for_version(version: str) -> GuidanceRegistry:
    if version == "1.0":
        return guidance_registry()
    if version == "2.0":
        return guidance_registry_v2()
    raise GuidanceRegistryError("guidance_registry_version_invalid")


def guidance_scope_directions(
    *, registry_version: str, topic: str, scope: str
) -> frozenset[Literal["import", "export"]]:
    """Resolve only a manifest-declared scope and its Korean applicability."""

    registry = guidance_registry_for_version(registry_version)
    if registry_version == "1.0":
        if scope != "legacy_topic" or topic not in registry.topic_directions:
            raise GuidanceRegistryError("guidance_registry_scope_invalid")
        return registry.topic_directions[topic]
    directions = registry.scope_directions.get((topic, scope))
    if directions is None:
        raise GuidanceRegistryError("guidance_registry_scope_invalid")
    return directions


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
    registry_version: str = "1.0",
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
    guidance_scope: str | None = None,
) -> None:
    """Fail closed unless one Data source exactly fits the copied registry.

    Data must additionally bind a response to the source-registry digest and
    a source record to one registry source ID.  Official law articles use a
    canonical public ``law.go.kr/법령/<law>/<article>`` URI, while all other
    official sources must exactly preserve the registered title and URI.
    ``development_fixture`` remains a clearly synthetic local-development
    exception and is never accepted from Functional-UAT.
    """

    registry = guidance_registry_for_version(registry_version)
    if registry_sha256 != registry.registry_sha256:
        raise GuidanceRegistryError("guidance_registry_sha256_mismatch")
    if direction not in {"import", "export", "both"}:
        raise GuidanceRegistryError("guidance_registry_source_invalid")
    safe_direction = cast(GuidanceDirection, direction)
    scope = "legacy_topic" if registry_version == "1.0" else guidance_scope
    if not isinstance(scope, str):
        raise GuidanceRegistryError("guidance_registry_scope_invalid")
    try:
        scope_directions = guidance_scope_directions(
            registry_version=registry_version,
            topic=topic,
            scope=scope,
        )
    except GuidanceRegistryError:
        raise GuidanceRegistryError("guidance_registry_direction_invalid") from None
    if (safe_direction == "both" and scope_directions != frozenset({"import", "export"})) or (
        safe_direction != "both" and safe_direction not in scope_directions
    ):
        raise GuidanceRegistryError("guidance_registry_direction_invalid")
    if source_kind == "development_fixture":
        if profile_mode != "local-development" or not isinstance(registry_source_id, str):
            raise GuidanceRegistryError("guidance_fixture_provenance_invalid")
        source = registry.sources_by_id.get(registry_source_id)
        if (
            topic not in _TOPICS
            or source is None
            or (
                registry_version == "1.0"
                and registry_source_id not in registry.topic_source_ids[topic]
            )
            or (
                registry_version == "2.0"
                and registry_source_id
                not in registry.scope_required_source_ids.get((topic, scope), frozenset())
            )
            or not source.supports(
                topic=topic,
                scope=None if registry_version == "1.0" else scope,
                direction=safe_direction,
            )
        ):
            raise GuidanceRegistryError("guidance_fixture_provenance_invalid")
        expected_uri = (
            f"https://customs.local.invalid/dev-fixture/guidance/{topic}"
            if registry_version == "1.0"
            else f"https://customs.local.invalid/dev-fixture/guidance/{topic}/{scope}"
        )
        expected_title = (
            registry.topic_titles[topic]
            if registry_version == "1.0"
            else registry.scope_titles[(topic, scope)]
        )
        if uri != expected_uri or title != expected_title:
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
        or (
            registry_version == "1.0" and registry_source_id not in registry.topic_source_ids[topic]
        )
        or not source.supports(
            topic=topic,
            scope=None if registry_version == "1.0" else scope,
            direction=safe_direction,
        )
    ):
        raise GuidanceRegistryError("guidance_registry_source_invalid")
    if source.source_kind == "official_law":
        if registry_version == "2.0":
            law_values = (law_id, law_name, article_no)
            if (
                (
                    any(value is not None for value in law_values)
                    and not all(value is not None for value in law_values)
                )
                or title != source.title
                or uri != source.official_uri
            ):
                raise GuidanceRegistryError("guidance_law_provenance_invalid")
            return
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
    "guidance_registry_for_version",
    "guidance_registry_v2",
    "guidance_scope_directions",
    "validate_guidance_source_provenance",
]
