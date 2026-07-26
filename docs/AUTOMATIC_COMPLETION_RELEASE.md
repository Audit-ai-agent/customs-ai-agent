# Automatic-completion Release Gate

## Current state

The packaged `release-profile-v1.lock.json` and reviewer trust store are intentionally
`unreleased`. The kill switch is active, automatic completion is disabled, and active
adaptive Tool execution is disabled. Consequently a clean diagnostic result remains
`human_review_required`; changing a confidence number cannot open this gate.

## Trust root and signing subject

Production approval requires an immutable reviewer trust store packaged in the wheel.
The store must be `active`, contain at least two distinct active reviewer identities,
have a correct canonical SHA-256, and contain only Ed25519 public keys. Private keys and
test keys must never be included in source, wheels, artifacts, logs, or environment files.

Each reviewer signs the following domain-separated message:

```text
customs-ai-agent/release-review/v1\0 || canonical_json(attestation_without_signature)
```

Every attestation binds the same exact release-subject SHA-256. The release subject binds:

- configured model ID and an immutable Claude model revision ending in an eight-digit
  snapshot date; an alias such as `latest` is not eligible;
- extraction, coordinator, narrative, and Case-answer prompt hashes;
- request and result Schema hashes;
- Gold set and immutable evaluation-report hashes;
- approved HSK10, language, and document-type scope;
- the bounded-adaptive active flag;
- approval/expiry dates and the exact reviewer trust-store hash;
- the complete machine-checkable evaluation summary.

At least two different reviewer identities, keys, and review artifacts are required.
Unknown, revoked, expired, duplicate-reviewer, invalid-signature, subject-mismatch, digest-
mismatch, or expired-profile input fails closed while the profile is loaded.

## Evaluation thresholds

An approved profile is rejected unless all of these signed values meet the threshold:

- each material field has at least 300 independently reviewed occurrences;
- critical identifier exact match is `1.0`;
- required-field exact match is at least `0.99`;
- hallucinated required fields are `0`;
- source grounding is `1.0`;
- false automatic completions are `0`;
- the locked holdout contains at least 500 Cases;
- the clean Production control runs at least 30 times;
- those 30 runs have material-result consistency `1.0`.

These values must come from the hashed evaluation report. They must not be synthesized from
the local Inspector or from unit-test fixtures. Unit tests generate ephemeral Ed25519 keys in
memory solely to prove verification and failure behavior.

## Runtime enforcement

Every V3 result reports the observed model revision, release profile digest and subject,
evaluation-report digest, trust-store digest, prompt/Schema hashes, policy/tool registry, and
Evidence runtime identity. A provider alias or unknown model revision keeps assurance at
Production-candidate and requires human review.

Bounded adaptive execution remains shadow-only unless the signed profile explicitly enables
it and the packaged policy ID matches. The active executor accepts only injected typed span-
reread and pinned Evidence-retry handlers. It has at most two cycles, six calls, and one reread
per field; no URL, SQL, file, shell, dynamic Tool, Outcome, or severity authority is available.
Only the deterministic affected-stage reevaluator may resolve a Gap.

The packaged FastAPI service registers the typed span-reread and exact pinned Evidence-retry
handlers through its runtime-owned bounded executor. Metadata therefore reports
`active_tool_handlers_registered=true`; execution nevertheless remains `shadow` until the whole
signed Production admission (exact code/model/prompts/Schemas/policy/tool registry plus released
Evidence V2/V3) succeeds. `active_execution_authorized_by_release` and
`active_tool_handlers_registered` are separate fields: registered handlers never authorize
themselves, and the activation and execution boundaries both re-check the signed profile and kill
switch.

Supporting-document assertions are not trusted from request JSON. The service now composes the
runtime-owned `SupportingDocumentAssertionVerifier`, which verifies an Ed25519 detached receipt
against the packaged `assertion-trust-store-v1.lock.json`. The receipt binds the Case, every
material supporting-document claim, Data policy source/release, authority/key/method, signature
and expiry, plus exactly one Backend decision-record hash or connector-result hash. Any source,
Case, HSK, item, document-code, policy, key status/window, receipt window, or signature drift is
fail-closed; `test_issuer` is always rejected by this Gate.

The packaged trust root is intentionally `unreleased` and empty until Backend/connector public
keys are approved. Therefore the concrete implementation exists, but Production admission and
readiness remain closed with `assertion_trust_store_unreleased`. Activating it requires publishing
an immutable active public-key store and binding that store SHA-256 into the independently signed
Claude release profile. A service bearer, authority prefix, or stable object ID alone is never a
verification receipt.

## Rotation and rollback

Do not edit an approved profile or move a tag. Rotate keys by publishing a new trust store,
new signed profile, new wheel, and new exact release identity. Revocation, identity drift,
evaluation regression, Data partial/failed state, or an incident activates the kill switch and
routes every Case to human review. Roll back with a new reviewed release or a revert; retain the
old artifacts for audit.
