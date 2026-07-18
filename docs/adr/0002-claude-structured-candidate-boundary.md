# ADR 0002: Claude structured candidate boundary

- Status: Accepted
- Date: 2026-07-17
- Scope: Historical `0.2.x` text-candidate baseline; `0.3.x` adds a stricter vision/OCR boundary

## Context

The Mock-only Agent used keyword and label rules for classification and extraction. Replacing the
whole workflow with an LLM would allow a non-deterministic model to invent legal evidence, severity,
or completion status. Merely placing a Claude call before the old rules would also leave schema,
grounding, provenance, fallback, and mixed Mock/Live execution undefined.

At the time of this decision, the current request contract contained text, not PDF/image bytes or
OCR/layout blocks.

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
- For the `0.2.x` baseline, PDF/image OCR, layout provenance, real bbox, Backend pinning, and full
  system release remained separate work and could not be inferred from the synthetic text
  evaluation.

## Successor implementation status

TASK-0009 targets a `0.3.0` release candidate. It preserves the V1 text-only request/result wire
contract and this decision's authority boundary while adding a separate V2 image/PDF contract,
independent OCR/layout ledgers, Claude Vision candidates and bounded crop reconciliation. Claude
still cannot create legal Evidence, findings, severity or final outcome.

This historical ADR does not claim that the new image path has passed release evaluation. A
provider can be selected only by an exact-commit-wheel run of the canonical locked 60-document
CI/PL/B/L and 20-document C/O benchmark plus the hardened combiner. Pre-freeze test counts, wheel
hashes and smoke reports are diagnostic history, not final release evidence.

The user-approved c28 text Claude Live run is retained as historical evidence: it scored 64/64
against normal Gold SHA-256
`7cfe2cf8e2b88e55f80398425e6866d52bfa85e0886ec8a4b5862f736e01c7e5` and 21/21
against adversarial Gold
`b4416cbad6df9fba62f2f636c4267a348bf5f11a1d3b78771aee8b16786db661`. It is not the
result for the ISO-6346-corrected current normal Gold
`1e300058f6d01330e82002d04e1a3238f42aaceeb9dc5716653a2aa1c67634e0`, which has not
yet been rerun with the post-c28 source.

The public c28 `AgentRuntime -> Claude -> Evidence HTTP` execution also predates mandatory runtime
contract identity. The current Agent requires `evidence_contract` in readiness, while the current
Data Provider does not return it and has only 3 of 58 expected law records. Current execution is
therefore blocked before retrieval POST and there is no successful current Agent–Data Live E2E.
The historical image diagnostic also used Gold that was later invalidated; the current evaluator
stops before OCR, Claude and Data until a human-reviewed replacement is pinned. A one-document OCR
smoke selected no provider because both Paddle and Docling failed the configured thresholds.
TASK-0009 does not change Backend, Frontend or Data Platform source. Backend framework selection
remains deferred, and legacy files in `project/_engineering/archive` remain `reference-only`
unless separately promoted and revalidated.

The successor OCR runtime uses internal ledger schema 2.0 and
`reading_order_policy_id=worker-emitted-sequence/v1`. The worker emits that policy, the edge adapter
requires the exact value, and the canonical ledger hash binds both the policy and the ordered
spans. Internal/v1 remains a historical compatibility resource only; a pre-policy v1 ledger or
checkpoint is not current runtime provenance. OCR Gold text uses
`renderer-bbox-yx/v1`: renderer word bbox top-left y/x with source index as tie-break. Prediction
text uses ledger `reading_order`; geometry must not reorder provider predictions.

A non-release, unsealed CI/PL/B/L/C/O one-document-per-type ordering audit produced 0% CER for worker/source order
against Gold bbox-yx order on all four documents. Counterfactual renderer-array CER was
23.31%, 19.30%, 29.58%, and 21.19% (24% micro), while OCR-line geometry resort was 36.55% micro.
Table row/cell F1 remained zero on all four documents. No durable Artifact Run was produced, so
this is a development ordering-contract diagnostic, not an authoritative or production OCR
benchmark. Historical v1 CER 22.17% and separate rendered-v2 CER observations 17.07%/0% use
different input hashes and must not be presented as a before/after improvement.
