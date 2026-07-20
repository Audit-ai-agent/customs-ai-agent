# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@8012f8ae5f530a4917b6d2790ee89a7e01fb5ee6`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `5b391374f897b308d6e9091a36cd6568f7227bd5e8ec9770d4c7ecc22b1550be`
- provider revision:
  `8012f8ae5f530a4917b6d2790ee89a7e01fb5ee6`
- provider source manifest SHA-256:
  `d88e2b1432558d12ce98cdaaf346374089e591161eb88595084e7362772c6360`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
