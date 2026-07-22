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
  "schema_version": "customs-agent-local-uat-manifest/v1",
  "acknowledgement": "LOCAL_UAT_DIAGNOSTIC_NOT_PRODUCTION",
  "diagnostic_only": true,
  "text_only": true,
  "mock_fallback_allowed": false,
  "auto_completion_allowed": false,
  "evidence_mode": "synthetic-v3",
  "evidence_base_url": "http://127.0.0.1:18013",
  "evidence_runtime_identity": {}
}
```

`evidence_runtime_identity` must be the complete, hash-valid V3 identity from
the running diagnostic provider. This task accepts only `synthetic-v3`.
The preserved 3-law corpus currently has a V2-only diagnostic path; it must not
be presented as full Workbench V3 UAT until a separate read-only V3 bridge is
implemented and verified.

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

This UAT proves integration behavior only. It is not a Claude calibration,
legal-coverage, RAG-quality, or Production release result.
