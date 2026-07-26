# LOCAL USER UAT Agent Service

This is a loopback-only, text-only diagnostic composition. It is under
`tools/`, is absent from the production wheel, and does not change Production
release admission or packaged locks.

It requires:

- an explicit manifest with a full Evidence V3 identity;
- `LOCAL_UAT_DIAGNOSTIC_NOT_PRODUCTION` acknowledgement in that manifest;
- the explicit `--acknowledge-local-uat` CLI flag;
- real `ANTHROPIC_API_KEY`, internal service token, and Evidence token in the
  explicitly selected `.env` file;
- a literal loopback Evidence URL;
- one request at a time.

The harness creates the manifest from the Data provider's generated identity:

```json
{
  "schema_version": "customs-agent-local-uat-manifest/v2",
  "acknowledgement": "LOCAL_UAT_DIAGNOSTIC_NOT_PRODUCTION",
  "diagnostic_only": true,
  "text_only": true,
  "mock_fallback_allowed": false,
  "auto_completion_allowed": false,
  "evidence_mode": "synthetic-v3",
  "evidence_base_url": "http://127.0.0.1:18013",
  "evidence_runtime_identity": {},
  "assertion_fixture_mode": "DISABLED",
  "assertion_trust_store_path": null,
  "assertion_trust_store_sha256": null
}
```

`evidence_runtime_identity` must be the complete, hash-valid V3 identity from
the running diagnostic provider. The exact identity must come from the
harness/Data build context; the Agent does not derive it from an environment
variable or patch its packaged Production lock.

The `preserved-3law-v3-scenario-eval` manifest must instead set
`assertion_fixture_mode=LOOPBACK_TEST_ONLY_HUMAN_APPROVER` and bind an absolute
`_secrets` path plus SHA-256 for its one-key, public-only Agent trust store.
The other profiles must keep all assertion fixture fields disabled. Private
signing key material is never part of the Agent manifest or metadata.

The UAT endpoint is `/health/uat-ready`. Production `/health/ready` always
returns 503. Review results always have `assurance_level=diagnostic`, can never
return `auto_review_complete`, and never trust a test issuer. Image/PDF input is
rejected before Claude is called.

Run from the repository root:

```powershell
uv run python -m tools.local_uat_agent_service `
  --manifest C:\path\to\agent-uat-manifest.json `
  --env-file C:\path\to\.env `
  --acknowledge-local-uat
```

The authenticated product-facing contract remains:

- `GET /internal/agent/v2/metadata`
- `POST /internal/agent/v2/reviews`
- `POST /internal/agent/v2/revisions`
- `POST /internal/agent/v2/answers`

`evidence_mode` has three explicit diagnostic profiles:

- `synthetic-v3` verifies the deterministic Evidence V3 contract control.
- `preserved-3law-v3-diagnostic` connects the Agent to the separately
  corroborated, read-only preserved three-law Data diagnostic through the same
  exact-identity-bound Evidence V3 HTTP boundary.
- `preserved-3law-v3-scenario-eval` uses that same preserved, read-only
  three-law Knowledge snapshot for all nine user scenarios, while requiring the
  separately signed test-only CCCT human-review policy identity.

The actual-store profiles do not give the Agent direct database access. The Agent
does not infer PostgreSQL, Qdrant, Neo4j, corpus counts, or RAG-quality claims
from an Evidence response. The harness and Backend must corroborate those Data
facts independently before showing them. ReviewResult V3 already carries the
safe provider-returned provenance used by the UI: snapshot ID, source record
and URI, version hash, retrieval methods, normalized score, and raw method
scores.

All profiles are diagnostic-only. Production `/health/ready` remains 503,
automatic completion remains disabled, and none of the profiles is a Claude
calibration, legal-coverage, RAG-quality, or Production release result.

The safe UAT metadata exposes the fixed policy mode for each profile:

- `synthetic-v3` → `SYNTHETIC_FIXTURE`
- `preserved-3law-v3-diagnostic` → `ACTUAL_CODES_HUMAN_ONLY`
- `preserved-3law-v3-scenario-eval` →
  `SEALED_TEST_ONLY_ACTUAL_CCCT_HUMAN_ONLY`
