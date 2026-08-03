# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@aacf248e9f48dc997abdbcac7e5d7b228e235561`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `0c4188af0460e8b0f94eb92a9b3669185178ffda48d46f5d559e3c13c9e8e254`
- provider revision:
  `aacf248e9f48dc997abdbcac7e5d7b228e235561`
- provider source manifest SHA-256:
  `3d677c39ae9544566f1fd94b166b943df88a4d814dc0055abdc2be544064a3f9`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
