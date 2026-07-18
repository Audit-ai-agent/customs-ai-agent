# OCR provider environments v1

The OCR release comparison runs PaddleOCR and Docling in separate Python 3.11
environments. The two `*.requirements.in` files declare the deliberately small
set of provider roots. The generated `*.requirements.lock.txt` files pin the
complete Windows dependency graph and include distribution hashes.
`model-assets.lock.json` pins every model repository revision and the canonical
content-tree hash used by the corresponding provider configuration. It contains
no machine path and no model weights.

Regenerate the locks only as an intentional evaluation-environment change:

```powershell
$uv = "<workspace>\_artifacts\tools\uv\v0.11.28\uv.exe"
& $uv pip compile --python-version 3.11 --python-platform windows `
  --generate-hashes --no-strip-markers `
  --output-file evals/ocr_environments/v1/paddle.requirements.lock.txt `
  evals/ocr_environments/v1/paddle.requirements.in
& $uv pip compile --python-version 3.11 --python-platform windows `
  --generate-hashes --no-strip-markers `
  --output-file evals/ocr_environments/v1/docling.requirements.lock.txt `
  evals/ocr_environments/v1/docling.requirements.in
```

Create each provider environment from its lock, then install the exact
release-candidate Agent wheel separately. `--require-hashes` prevents an
unhashed dependency from entering the provider environment.

```powershell
& $uv venv --python 3.11 "<artifact-root>\ocr-envs\paddle"
& $uv pip sync --python "<artifact-root>\ocr-envs\paddle\Scripts\python.exe" `
  --require-hashes evals/ocr_environments/v1/paddle.requirements.lock.txt
& $uv pip install --python "<artifact-root>\ocr-envs\paddle\Scripts\python.exe" `
  --no-deps "<package-root>\customs_ai_agent-0.3.0-py3-none-any.whl"
& $uv pip check --python "<artifact-root>\ocr-envs\paddle\Scripts\python.exe"
```

Repeat the last three commands with the Docling lock and environment. The OCR
Gate must verify the selected dependency inventory against its lock, verify the
selected model cache against `model-assets.lock.json`, and record both lock
SHA-256 values, the exact Agent wheel SHA-256, installed Agent `RECORD`
integrity, provider engine/version, and benchmark descriptor hashes. A lock
refresh, Python/platform change, provider package change, or model-asset change
requires a new benchmark run; an old PASS cannot be reused.

The v2 benchmark runner also emits only redacted runtime numbers. For each
provider it records per-document end-to-end latency, the isolated worker
process peak working set, and the sum of installed distribution files plus
attested model-asset bytes. The release combiner uses those values only when
both providers pass quality/stability and the locked quality tuple is exactly
tied. It recomputes p95 from the document measurements and compares
`p95 latency -> peak memory -> disk footprint`. Missing or malformed metrics,
identity drift, or an exact runtime tie cannot select a provider.

Provider model assets remain in the isolated runtime cache. Paddle verification
uses each model directory under the PaddleX official-model cache and excludes
its `.cache` metadata subtree. Docling verification uses the exact Hugging Face
snapshot directory named by the locked revision. The tree hash follows the
algorithm declared in the lock and hashes regular file content after following
snapshot links. Paddle release attestation also requires the service-owned
official-model and repository-cache inventories to contain exactly the declared
models. Docling rejects undeclared repositories in every namespace declared by
its lock while allowing unrelated namespaces in a shared Hugging Face hub.
The Agent forces `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_HUB_DISABLE_TELEMETRY=1` after its worker environment allowlist, so Docling
cannot refresh verified refs or download a replacement snapshot. This policy is
bound to `DocumentConverter:default-image-pipeline-offline-v2`; a missing cache
fails initialization.

No OCR provider is a product default until the locked 60-document benchmark
(20 cases × CI/PL/B/L) and 20-document C/O benchmark jointly select it.
