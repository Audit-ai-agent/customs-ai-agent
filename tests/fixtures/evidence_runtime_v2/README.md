# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@c9214d17343cec21a575ffde979f045169c7a341`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `e3e16ef56e2e2ce7106b7a5dce2eb3b3fb2618e7658f97742c76bbf47de99a5e`
- provider revision:
  `c9214d17343cec21a575ffde979f045169c7a341`
- provider source manifest SHA-256:
  `0c99c24ff4bc51d3f458bbb093666952c36a2942f4967141e0e35bcb06ad5158`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
