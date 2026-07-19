# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@facf9e051812523914bbd599be9391fc58cfa05e`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `e96f5802db02b67ba83b58001780e2cacab4233a5a452c587c17d8d8373b1de5`
- provider revision:
  `facf9e051812523914bbd599be9391fc58cfa05e`
- provider source manifest SHA-256:
  `c9b5beff86a06298cfebd1e2fca8e7605bd259a0383dd3b665ded764e5d685d4`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
