# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@d8a0c4876ecf29a62371b8e4e388814e61701907`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `71a0ddf4b9e58f1d598221e980fd90230ed1bff560ad467c2cfa0692116e35eb`
- provider revision:
  `d8a0c4876ecf29a62371b8e4e388814e61701907`
- provider source manifest SHA-256:
  `8791829ca706719e9d5201b6c4f1ca499b3c2765c0968f3dfc203891808c7d1a`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
