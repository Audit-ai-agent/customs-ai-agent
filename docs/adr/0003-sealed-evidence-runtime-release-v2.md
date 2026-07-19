# ADR 0003: Sealed Evidence Runtime Release v2

## Status

Accepted for implementation; release lock remains unreleased.

## Context

The retrieval v1 body contract binds requests and responses, but provider code identity alone
does not prove which law corpus, embedding/reranker models, indexes, or evaluation report served
a result. A rolling deployment can also expose readiness from build/generation A and retrieval
from B.

## Decision

Production `AgentRuntime` uses only:

- `GET /internal/rag/v2/health/ready`
- `POST /internal/rag/v2/retrieve`
- the unchanged provider-owned retrieval body schema v1
- the exact provider-owned readiness v2 schema bytes

The packaged production lock pins the complete `EvidenceRuntimeReleaseIdentity`, including the
provider build/source manifest, sealed knowledge generation and attestations, model/index
manifests, locked evaluation report, snapshot, supported date, and their canonical release hash.
Every POST sends eight release-binding headers and validates all eight response headers before
reading the body. Request ID and snapshot are then validated from the v1 body.

The Agent release builder consumes only Data's exact
`evidence-runtime-release-provider-lock` 1.0 document. It independently verifies both schema
byte hashes, the source-manifest hash projection, and the canonical provider-build hash. The
obsolete seven-field provider-build document is not a release input.

One monotonic deadline covers readiness GET, retrieval POST, retries, headers, and bounded body
chunks. Each exchange runs behind a caller-visible finite daemon boundary and a process-wide
bounded transport semaphore. Deadline expiry closes the response asynchronously and releases the
Agent review/readiness caller even if an underlying third-party socket implementation does not
cooperate.

The existing v1 HTTP retriever remains only an explicit CLI/evaluation/test compatibility
adapter. It is not selectable in `RuntimeConfig`.

Until Data has a verified sealed generation and evaluation report, the package lock is
`release_state=unreleased`. Production runtime construction, service readiness, and review
admission fail closed without constructing/calling Claude. Hashes are never synthesized to make
the Gate pass.

## Consequences

- A Data rollout requires a new verified lock and Agent artifact.
- An incomplete 3/58 corpus cannot appear release-ready.
- Liveness remains observable while readiness and reviews return unavailable.
- The lock builder accepts only a schema-byte-identical Data readiness document and Data's exact
  canonical `provider-runtime-lock.json`.
- Slow-header and slow-body behavior cannot extend the configured Evidence operation deadline or
  permanently retain a service review/readiness caller.
