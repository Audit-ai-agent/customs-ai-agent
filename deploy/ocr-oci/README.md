# Raster Text OCR OCI 배포

이 경로는 모델을 Git 소스에 커밋하지 않고도 검증된 Raster Text OCR을 다른
Linux 서버에서 동일하게 실행하기 위한 배포 경계다. 최종 결과는
`linux/amd64` 전용 private GHCR 이미지이며, 배포에는 tag가 아니라 반드시
`@sha256:...` digest를 사용한다.

## 무엇이 자동화되는가

배포물은 다음 두 개의 OCI 이미지로 나뉜다.

1. runtime context 이미지: Paddle sidecar, 모델, schema-11 pointer, runtime
   profile, 실제 재시작에 필요한 compact evidence, runtime SPDX SBOM과 license
   manifest만 보관한다.
2. Agent release 이미지: 고정 Python base와 작은 Agent application layer를
   runtime context 위에 결합한다. 코드만 바뀌면 큰 runtime layer는 registry와
   배포 노드의 cache에서 재사용된다.

일반 PR에는 모델이나 registry secret이 필요하지 않다. CI는 Dockerfile과
workflow의 fail-closed 구조를 검사하고 `app-structure` target까지만 확인한다.
최종 publish는 다음 경우에만 가능하다.

- `main`에서 명시적으로 `workflow_dispatch` 실행
- `ocr-oci-v*` tag push

두 경우 모두 이미 private GHCR에 올라간 runtime의 digest가 필요하다. tag나
`latest`는 입력으로 거부한다.

## 1. Linux runtime closure 내보내기

이 작업은 최종 Linux 배포 환경과 같은 `linux/amd64`, CPython 3.11
builder에서 수행한다. Windows의 `python.exe` bundle을 Linux 이미지로 복사할
수 없으며 exporter도 이를 `linux_amd64_builder_required` 또는
`linux_cp311_sidecar_required`로 거부한다.

입력 profile은 다음을 이미 만족해야 한다.

- `approvals/raster-text-active.json` schema 11, `activated`
- Linux sidecar interpreter 경로 `sidecar/interpreter/bin/python3.11`
- 전체 activation 검증과 현재 runner hash 일치
- sidecar/model/package inventory 및 모델 byte 검증
- 외부 다운로드나 fallback 없이 runtime compact 검증 가능

runtime SBOM은 SPDX JSON(`SPDX-2.x`, `CC0-1.0`)이어야 한다. license manifest는
다음 형태로 모든 Python package, runtime, model의 출처와 license를 명시한다.

```json
{"components":[{"license_id":"Apache-2.0","name":"example","scope":"python-package","source_uri":"https://example.invalid/source","version":"1.0"}],"schema_version":"1.0"}
```

내보내기는 기존 디렉터리를 덮어쓰지 않는다. 전체 activation을 먼저 검증한
후 compact runtime이 실제로 읽은 파일과 sidecar/model release root만 staging에
복사하고, 최종 manifest 검증이 끝났을 때만 새 디렉터리로 원자적으로 바꾼다.

```bash
uv run --locked --no-sync python tools/export_raster_text_oci_runtime.py \
  --document-preview-root /srv/customs/_local/data/document-preview \
  --runtime-sbom /srv/customs/release-input/runtime.spdx.json \
  --runtime-license-manifest /srv/customs/release-input/runtime-licenses.json \
  --destination /srv/customs/release-output/raster-runtime-context

python tools/verify_raster_text_oci_runtime.py \
  --document-preview-root \
  /srv/customs/release-output/raster-runtime-context/document-preview
```

Exporter와 verifier는 다음을 거부한다.

- symlink, special file, 누락·추가·변경된 file
- Windows interpreter 또는 Linux/amd64가 아닌 bundle
- 원본 PDF/JPG/PNG, raster/Gold corpus, 2,400건 평가 결과 전체
- cache, secret 디렉터리, group/other writable file
- 128개 또는 64MiB를 넘는 compact activation evidence
- SBOM/license manifest가 없거나 pointer/profile hash와 맞지 않는 bundle

## 2. runtime context를 private GHCR에 게시

먼저 제한된 `packages:write` 권한으로 `docker login ghcr.io`를 수행한다. Token은
스크립트 인수나 출력으로 전달하지 않는다. 그 다음 exporter 결과를 data-only
OCI image로 게시한다.

```bash
bash deploy/ocr-oci/publish_runtime_context.sh \
  /srv/customs/release-output/raster-runtime-context \
  ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime
```

스크립트는 runtime manifest를 다시 검증하고, OCI provenance/SBOM attestation을
붙여 push한 다음 아래처럼 변경 불가능한 참조만 출력한다.

```text
runtime_image=ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime@sha256:<digest>
```

이 값을 저장소 Actions variable `CUSTOMS_AGENT_RASTER_RUNTIME_IMAGE`에 넣는다.
변수는 secret이 아니지만 반드시 같은 GitHub owner의 private GHCR digest여야 한다.
수동 workflow 실행에서는 `runtime_image` 입력으로 같은 값을 직접 줄 수도 있다.

## 3. 최종 Agent+OCR 이미지 게시

GitHub의 **Raster OCR OCI publish** workflow를 `main`에서 수동 실행하거나,
검증된 release에 `ocr-oci-v*` tag를 만든다. Workflow는 다음 순서를 지킨다.

1. runtime 입력이 `ghcr.io/<현재 owner>/...@sha256:<64 hex>`인지 확인한다.
2. private runtime digest가 실제로 pull 가능한지 확인한다.
3. `deploy/ocr-oci/Dockerfile`의 `release` target을 `linux/amd64`로 build한다.
4. runtime manifest의 모든 file hash, pointer, profile, SBOM, license를 재검증한다.
5. non-root final image를 GHCR에 push하고 OCI provenance/SBOM을 첨부한다.
6. workflow summary에 최종 `image@sha256:digest`를 남긴다.

로컬에서 같은 결합 build를 확인하려면 다음처럼 정확한 runtime digest와 현재
40자리 Git commit을 전달한다.

```bash
docker buildx build \
  --file deploy/ocr-oci/Dockerfile \
  --target release \
  --platform linux/amd64 \
  --build-arg \
  RASTER_RUNTIME_IMAGE=ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime@sha256:<digest> \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --load \
  --tag customs-ai-agent-raster-ocr:local \
  .
```

runtime 인수 없이 `release` target을 build하면 `scratch`에 runtime 파일이 없어
반드시 실패한다. PR 구조 검사는 asset 없는 아래 명령만 사용한다.

```bash
python deploy/ocr-oci/verify_deployment_sources.py --repository-root .
docker buildx build --file deploy/ocr-oci/Dockerfile \
  --platform linux/amd64 --target app-structure --check .
```

## 4. digest 고정 실행과 OCR readiness

현재 Agent 설정은 보안상 `127.0.0.1` bind만 허용한다. 따라서 이 이미지는 포트를
bridge network에 공개하지 않고 다음 중 하나로 실행한다.

- Linux host network에서 Backend도 같은 host의 loopback으로 호출
- Backend와 Agent가 같은 Pod/network namespace를 공유

예제 Compose는 host network, read-only root filesystem, non-root UID, capability
제거와 `/tmp` 전용 tmpfs를 강제한다. 실제 secret 값은 Git 밖의 제한된 배포
환경에서 주입한다.

```bash
export CUSTOMS_AGENT_IMAGE='ghcr.io/audit-ai-agent/customs-ai-agent-raster-ocr@sha256:<digest>'
docker compose --file deploy/ocr-oci/compose.example.yaml up -d
```

Docker `healthy`는 일반 `/health/ready`가 아니다. 이미지의 healthcheck가 인증된
`GET /internal/agent/v3/raster-text-preview-status`를 호출하고 다음을 모두 확인한다.

- schema `1.0`
- availability `available`
- 응답의 `raster_text_profile_revision`이 이미지 속 schema-11 pointer와 정확히 일치

서비스 시작 때 2,400건 평가를 다시 수행하지 않는다. compact byte 검증 후 OCR
child를 warm-up하며, 완료 전에는 health가 `starting`/`unhealthy`이고 Raster OCR
요청은 fail-closed 된다. 배포 오케스트레이터는 컨테이너 실행 여부가 아니라 이
OCR health가 성공한 뒤에만 Backend 트래픽을 열어야 한다.

현재 서비스 mode는 `local-development`와 `functional-uat`뿐이다. 이 OCI 경로는
검증된 배포 후보를 재현하는 경계이며, 별도의 Production Release·KMS·외부 서명
정책을 새로 주장하지 않는다.
