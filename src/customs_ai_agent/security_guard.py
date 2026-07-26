from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


PROMPT_INJECTION_REASON_CODE = "untrusted_document_prompt_injection_pattern_detected"

# This is deliberately a narrow production-candidate/UAT release guard, not a
# general prompt-injection classifier. It recognizes high-confidence English
# and Korean instruction-override/exfiltration patterns in text documents.
# Claude never participates in this decision and the guard never creates or
# edits customs Issues, Findings, Evidence, or Tool names.
_MAX_SCAN_CHARACTERS = 500_000
_SEPARATOR = r"[\W_]{0,3}"
_CONFUSABLES = str.maketrans(
    {
        # Small, explicit set needed to prevent common mixed-script spelling
        # evasions of the English control words below.
        "а": "a",
        "е": "e",
        "і": "i",
        "ј": "j",
        "о": "o",
        "р": "p",
        "с": "c",
        "ѕ": "s",
        "х": "x",
        "у": "y",
        "ο": "o",
        "ρ": "p",
        "χ": "x",
    }
)


def _flex_word(value: str) -> str:
    return r"(?<!\w)" + _SEPARATOR.join(re.escape(character) for character in value) + r"(?!\w)"


def _words(*values: str) -> str:
    return "(?:" + "|".join(_flex_word(value) for value in values) + ")"


_OVERRIDE_VERB = _words(
    "ignore",
    "disregard",
    "forget",
    "override",
    "bypass",
)
_PRIOR_SCOPE = _words(
    "all",
    "every",
    "previous",
    "prior",
    "above",
    "earlier",
)
_INSTRUCTION_TARGET = _words(
    "instruction",
    "instructions",
    "rule",
    "rules",
    "directive",
    "directives",
    "policy",
    "policies",
    "prompt",
    "prompts",
)
_PRIVILEGED_INSTRUCTION_TARGET = _words(
    "system",
    "developer",
    "administrator",
    "security",
)
_EXFILTRATION_VERB = _words(
    "reveal",
    "disclose",
    "expose",
    "leak",
    "print",
    "show",
)
_SECRET_TARGET = _words(
    "credential",
    "credentials",
    "secret",
    "secrets",
    "token",
    "tokens",
)
_SYSTEM_PROMPT = rf"{_words('system', 'developer')}.{{0,24}}{_words('prompt', 'message')}"
_API_KEY = rf"{_words('api')}.{{0,16}}{_words('key', 'keys')}"
_ROLE_DIRECTIVE = (
    rf"(?:{_words('you')}.{{0,16}}{_words('are')}.{{0,16}}{_words('now')}"
    rf"|{_words('act')}.{{0,16}}{_words('as')})"
)
_PRIVILEGED_ROLE = _words(
    "system",
    "developer",
    "administrator",
    "admin",
)
_TOOL_VERB = _words("call", "invoke", "execute", "run", "use")
_TOOL_SCOPE = _words("any", "available", "hidden", "internal", "unauthorized")
_TOOL_TARGET = _words("tool", "tools", "function", "functions")
_OUTCOME_VERB = _words(
    "mark",
    "set",
    "return",
    "change",
    "invent",
    "fabricate",
    "declare",
)
_OUTCOME_TARGET = _words(
    "approved",
    "completed",
    "outcome",
    "finding",
    "findings",
    "status",
)

_STRONG_PATTERNS = tuple(
    re.compile(pattern, re.DOTALL)
    for pattern in (
        # Explicitly replacing a prior/system instruction is by itself a
        # high-confidence document-to-model instruction override.
        rf"{_OVERRIDE_VERB}.{{0,80}}{_PRIOR_SCOPE}.{{0,80}}{_INSTRUCTION_TARGET}",
        rf"{_OVERRIDE_VERB}.{{0,80}}{_PRIVILEGED_INSTRUCTION_TARGET}"
        rf".{{0,48}}{_INSTRUCTION_TARGET}",
        # Requests to expose model/runtime secrets have no legitimate customs
        # document semantics in this bounded text-only profile.
        rf"{_EXFILTRATION_VERB}.{{0,96}}(?:{_SYSTEM_PROMPT}|{_API_KEY}|{_SECRET_TARGET})",
        # Direct role replacement.
        rf"{_ROLE_DIRECTIVE}.{{0,48}}{_PRIVILEGED_ROLE}",
        # Korean high-confidence override/exfiltration forms.
        r"(?:이전|앞선|위의|모든).{0,40}"
        r"(?:지시|명령|규칙|프롬프트|정책).{0,40}"
        r"(?:무시|우회|덮어쓰)",
        r"(?:무시|우회|덮어쓰).{0,40}"
        r"(?:이전|앞선|위의|모든).{0,40}"
        r"(?:지시|명령|규칙|프롬프트|정책)",
        r"(?:(?:시스템|개발자).{0,16}프롬프트|API.{0,8}키|비밀|토큰)"
        r".{0,48}(?:공개|노출|출력|유출)",
        r"(?:공개|노출|출력|유출).{0,48}"
        r"(?:(?:시스템|개발자).{0,16}프롬프트|API.{0,8}키|비밀|토큰)",
    )
)
_TOOL_MANIPULATION_PATTERN = re.compile(
    rf"{_TOOL_VERB}.{{0,64}}(?:{_TOOL_SCOPE}.{{0,32}})?{_TOOL_TARGET}",
    re.DOTALL,
)
_OUTCOME_MANIPULATION_PATTERN = re.compile(
    rf"{_OUTCOME_VERB}.{{0,80}}{_OUTCOME_TARGET}",
    re.DOTALL,
)


def _normalized_document_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = unicodedata.normalize("NFKD", normalized).casefold()
    normalized = unicodedata.normalize(
        "NFC",
        "".join(
            character
            for character in normalized
            if unicodedata.category(character) not in {"Cf", "Mn"}
        ),
    )
    return normalized.translate(_CONFUSABLES)


def document_contains_prompt_injection_pattern(value: str) -> bool:
    """Return a bounded high-confidence instruction-override signal.

    This intentionally does not assign legal meaning, inspect model output, or
    claim broad adversarial coverage. A match only adds a human-review release
    guard to the existing deterministic customs review.
    """

    normalized = _normalized_document_text(value[:_MAX_SCAN_CHARACTERS])
    if any(pattern.search(normalized) is not None for pattern in _STRONG_PATTERNS):
        return True
    return (
        _TOOL_MANIPULATION_PATTERN.search(normalized) is not None
        and _OUTCOME_MANIPULATION_PATTERN.search(normalized) is not None
    )


def prompt_injection_unmet_reason_codes(
    document_texts: Iterable[str],
) -> list[str]:
    remaining = _MAX_SCAN_CHARACTERS
    for value in document_texts:
        if remaining <= 0:
            break
        bounded = value[:remaining]
        remaining -= len(bounded)
        if document_contains_prompt_injection_pattern(bounded):
            return [PROMPT_INJECTION_REASON_CODE]
    return []


__all__ = [
    "PROMPT_INJECTION_REASON_CODE",
    "document_contains_prompt_injection_pattern",
    "prompt_injection_unmet_reason_codes",
]
