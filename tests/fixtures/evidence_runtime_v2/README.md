# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@568c241cd1b8f614f27b863ebec97a3d00ec8e4e`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `320216de8685493e529f389b4bd3a1e832bd3c5df05b0463c887b677aa25e821`
- provider revision:
  `568c241cd1b8f614f27b863ebec97a3d00ec8e4e`
- provider source manifest SHA-256:
  `ab817a8f1876a8f0f983b763c79df3114a970d77066a03a95a21857d4eaf86b2`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
