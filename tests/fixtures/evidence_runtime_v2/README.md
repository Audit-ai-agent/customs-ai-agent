# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@14cde864425eee615a2e3f6799f6f2719448aab9`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `d87f1b86af4812f3de7a2de070f23453c354c0be1534725950f154befffb557f`
- provider revision:
  `14cde864425eee615a2e3f6799f6f2719448aab9`
- provider source manifest SHA-256:
  `dd8ceb68c2da83d1fdb82f3d00f4f1137a559fcb577efdf86b6f87e52d84db1c`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
