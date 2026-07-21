# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@a0db83a7c69f792f0801dc7bd63ef74692a30d03`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `6932b6323b123ba77624a39793dbbf6386a95aabb8cb90a7169125ffb98d2e17`
- provider revision:
  `a0db83a7c69f792f0801dc7bd63ef74692a30d03`
- provider source manifest SHA-256:
  `318e7af0e233a92261dbf95b40ee87ab811ab8d7a9d3a25c0dc1da037691a482`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
