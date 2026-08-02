# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@7df26860e3d31eef4545fae8851b6f16d17a331d`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `71c0ff5db125267fbeca3ab448bdd9f830c8046e00d5e82898c2f4231bb98da5`
- provider revision:
  `7df26860e3d31eef4545fae8851b6f16d17a331d`
- provider source manifest SHA-256:
  `3c085dba66bb3b546cc117124c224b36ccfa86b622123c74578d29fb86325c1c`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
