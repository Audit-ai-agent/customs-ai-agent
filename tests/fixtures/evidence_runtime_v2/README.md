# Evidence runtime v2 cross-repository fixture

`provider-runtime-lock.json` is the byte-for-byte canonical handoff generated
from `customs-data-platform@89fe32ed43686fe7e1eb07bb8b8841c6ff393b05`.
The lock format itself was introduced at `78e2392` by the canonical
`tools.provider_build.create_runtime_provider_lock_document` helper.

Artifact identity:

- lock file SHA-256:
  `1fcb0fbe554a5bafff908c6963b3b5f6ceb6b499dae1ef380e4e63ef8b1590f8`
- provider revision:
  `89fe32ed43686fe7e1eb07bb8b8841c6ff393b05`
- provider source manifest SHA-256:
  `fd4b7f6fd0d6b01917480ba8285ac0092525da41ddcc77ff91505cf82a513094`
- retrieval/readiness schemas: byte-identical Agent/Data schema snapshots

The fixture is committed so Agent clean-clone tests do not import a sibling
Data checkout. The Data source manifest is not copied into the Agent
repository; its identity remains bound by the provider lock. Tests
independently recompute the canonical provider-build digest, validate both
local schema byte hashes, and reject the obsolete seven-field provider-build
document.
