# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@036ef6cec5422c22228e865ea1689b2d319f0b65`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `1a05a40c6d6b4256d317fd67db6c90b12aaa7bda1b54ea5ddbd1ce4f4604ddae`
- provider revision:
  `036ef6cec5422c22228e865ea1689b2d319f0b65`
- provider source manifest SHA-256:
  `f11e46290dc6de240a27e6c610d1a8696877e09cfceed4269f326d8f0fe39e4a`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
