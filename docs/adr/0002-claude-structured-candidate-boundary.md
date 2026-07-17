# ADR 0002: Claude structured candidate boundary

- Status: Accepted
- Date: 2026-07-17

## Context

The Mock-only Agent used keyword and label rules for classification and extraction. Replacing the
whole workflow with an LLM would allow a non-deterministic model to invent legal evidence, severity,
or completion status. Merely placing a Claude call before the old rules would also leave schema,
grounding, provenance, fallback, and mixed Mock/Live execution undefined.

The current request contract contains text, not PDF/image bytes or OCR/layout blocks.

## Decision

Claude Sonnet 5 is used once per text document to produce a `ClaudeDocumentCandidate` through
Messages API structured output. The candidate contains only:

- document type and a classification source quote;
- allowlisted document fields with verbatim values and quotes;
- allowlisted line-item fields in source order with verbatim values and quotes.

Every object forbids extra properties. Local validation enforces document-type allowlists,
uniqueness, size bounds, case-sensitive exact quote spans in the supplied text, and exact bounded
value spans in the quote. Document-field quotes must be unique; line-item occurrences must map in
source order without exceeding the available occurrence count. Approximate NFKC/case/whitespace
matching is rejected. The code, not the model, creates bounded stable item IDs and exact text
offsets. Text-only evidence has no bbox. The user input is a JSON envelope whose metadata and
document text are explicitly untrusted values.

Claude cannot produce Evidence IDs, legal requirements, findings, severity, or final status.
Normalization, arithmetic, cross-document validation, Evidence API verification, Evidence Gate,
severity, and outcome remain deterministic. Provider/schema/grounding failure never falls back to
the rule extractor. LLM mode also requires an explicit retriever to prevent silent Mock RAG.

The local canonical configuration remains outside product repositories in one `_secrets/.env`.
An explicit env file supplies only allowlisted Claude/Evidence values to settings objects without
changing `os.environ`; an already-set process value takes precedence. The evaluated runtime is
pinned to `claude-sonnet-5` and `https://api.anthropic.com`. Anthropic endpoint/header/auth-token
overrides and provider debug logging are rejected.

Exact quote grounding does not prove semantic field binding. Claude values therefore use
uncalibrated confidence 0.85 and force `needs_review`. Even a critical deterministic issue derived
from those values cannot autonomously produce `blocked` until a separate human approval contract
exists. Provider-wide failure opens a per-run circuit so remaining documents are not repeatedly
sent.

## Consequences

- Real Claude text classification/extraction is available and independently testable.
- Default CI remains keyless, deterministic, and network-free through transport/analyzer injection.
- Model call audit is limited to safe model/prompt/request/message/usage/latency metadata; failures
  keep safe codes and bounded provider metadata, never exception bodies.
- Package version is raised to `0.2.0` so Backend cannot confuse this wheel with Mock-only `0.1.0`.
- PDF/image OCR, layout provenance, real bbox, Backend pinning, and full system release remain
  separate work and must not be inferred from the synthetic text evaluation.
