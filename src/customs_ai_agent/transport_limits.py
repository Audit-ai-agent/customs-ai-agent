"""Versioned text-first transport limits shared by Schema and service admission.

These are contract limits, not tuning defaults.  Production may not override
them without publishing a new request contract.  Non-production service
instances may choose stricter limits and advertise the effective values in
metadata.
"""

TEXT_FIRST_MAX_BODY_BYTES = 2 * 1024 * 1024
TEXT_FIRST_MAX_DOCUMENTS = 6
TEXT_FIRST_MAX_TEXT_CHARACTERS = 500_000


__all__ = [
    "TEXT_FIRST_MAX_BODY_BYTES",
    "TEXT_FIRST_MAX_DOCUMENTS",
    "TEXT_FIRST_MAX_TEXT_CHARACTERS",
]
