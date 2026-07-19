# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@27bc2b43392262a93113fd1b6e06840b0c816c17`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `6845f05dd154cd78583fe618fddbf681e092314eb3fc0a611dbb81788aa8f6da`
- provider revision:
  `27bc2b43392262a93113fd1b6e06840b0c816c17`
- provider source manifest SHA-256:
  `8472f7c9e249de059348e9b47e6d9c5ed7be6a9992aabece90235fc990e3ae4e`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
