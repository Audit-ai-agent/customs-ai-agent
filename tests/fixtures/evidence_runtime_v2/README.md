# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@aae07d01508dbe8eb299663f3cc1d9ee24f9d29d`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `7427a38d0efb4075952274a85ef32ee65959fd4a638911d5d2819c737f52b64f`
- provider revision:
  `aae07d01508dbe8eb299663f3cc1d9ee24f9d29d`
- provider source manifest SHA-256:
  `224b794fe4312514f461d6ee269211e0ab0f152cdaa670079db5f82d6d42a2b1`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
