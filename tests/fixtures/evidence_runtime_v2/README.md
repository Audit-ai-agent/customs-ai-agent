# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@0f6fbe031ddf531b44bc8f338bd2ec1d39e47039`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `fe36dac90a034cb3b506e5a9071ded53865f8d3594dd557c3d9f42c58175ad86`
- provider revision:
  `0f6fbe031ddf531b44bc8f338bd2ec1d39e47039`
- provider source manifest SHA-256:
  `dada3f2e5406e95c1c092250e0229af5967428a39856b6535808984f7154037f`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
