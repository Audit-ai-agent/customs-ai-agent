# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@b13e8bf7238a61eb185efcf2e9cd768ad5a908e2`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `3b8f5955422060dc1dc732b79d4cd8222ce3adbca2f32961a0e69b90156e921f`
- provider revision:
  `b13e8bf7238a61eb185efcf2e9cd768ad5a908e2`
- provider source manifest SHA-256:
  `ddf271e0e0f8b3628917146d6f1a83bedb8e17d9dc98de30f50e12e0c8b1d8e0`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
