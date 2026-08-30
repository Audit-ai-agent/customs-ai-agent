# Raster Text OCR OCI 배포

이 경로는 모델을 Git 소스에 커밋하지 않고도 검증된 Raster Text OCR을 다른
Linux 서버에서 동일하게 실행하기 위한 배포 경계다. 최종 결과는
`linux/amd64` 전용 private GHCR 이미지이며, 배포에는 tag가 아니라 반드시
`@sha256:...` digest를 사용한다.

## 무엇이 자동화되는가

배포물은 다음 두 개의 OCI 이미지로 나뉜다.

1. runtime context 이미지: Paddle sidecar, 모델, current-native schema-12
   pointer, sealed V9, runtime profile, 실제 재시작에 필요한 compact evidence,
   runtime SPDX SBOM과 license manifest만 보관한다.
2. Agent release 이미지: 고정 Python base와 작은 Agent application layer를
   runtime context 위에 결합한다. 코드만 바뀌면 큰 runtime layer는 registry와
   배포 노드의 cache에서 재사용된다.

Paddle와 OpenCV가 slim Debian에서 실제 import되려면 각각 `libgomp.so.1`,
`libGL.so.1`, `libgthread-2.0.so.0`이 필요하다. 공통 runtime layer에는 확인된
Debian Bookworm 패키지 `libgomp1`, `libgl1`, `libglib2.0-0`만 버전 고정하여
설치하고, 빌드 중 세 shared library를 동적 로딩해 누락을 즉시 차단한다. 별도
X11 패키지는 직접 추가하지 않는다. 이 OS 패키지는 최종 OCI SBOM/provenance와
이미지 안의 Debian copyright 자료에 포함된다.

현재 평가·고정된 sidecar가 `opencv-contrib-python`의 non-headless build를 쓰기
때문에 `libgl1`의 전이 의존성까지 합치면 실제 증가는 43개 패키지, 약 221MB다.
이를 줄이기 위해 임의로 headless wheel로 바꾸면 검증된 package/model closure가
달라지므로, 별도 재평가 전에는 이 세 직접 의존성보다 더 줄였다고 주장하지 않는다.

일반 PR에는 모델이나 registry secret이 필요하지 않다. CI는 Dockerfile과
workflow의 fail-closed 구조를 검사하고 `app-structure` target까지만 확인한다.
최종 publish는 다음 경우에만 가능하다.

- `main`의 현재 SHA에서 `workflow_dispatch` `operation=publish`를 명시적으로
  실행

Tag나 branch push로는 publish가 시작되지 않는다. Publish 실행에는 이미
private GHCR에 올라간 runtime의 `@sha256:...` digest가 필요하며,
tag나 `latest`는 runtime 입력으로 거부한다.

## 1. Linux runtime closure 내보내기

이 작업은 최종 Linux 배포 환경과 같은 `linux/amd64`, CPython 3.11
builder에서 수행한다. Windows의 `python.exe` bundle을 Linux 이미지로 복사할
수 없으며 exporter도 이를 `linux_amd64_builder_required` 또는
`linux_cp311_sidecar_required`로 거부한다.

입력 profile은 다음을 이미 만족해야 한다.

- `approvals/raster-text-active.json` schema 12, `activated`
- sealed evaluation schema 9 및 `current-runner-native-full-gate-v1`
- Linux sidecar interpreter 경로 `sidecar/interpreter/bin/python3.11`
- 전체 activation 검증과 현재 runner hash 일치
- sidecar/model/package inventory 및 모델 byte 검증
- 외부 다운로드나 fallback 없이 runtime compact 검증 가능

Linux export는 이 V12/V9 조합만 허용한다. 기존 Windows schema-11 pointer는 로컬
서비스 호환을 위해 Agent에 남아 있지만 Linux OCI 입력으로는 명시적으로 거부된다.

runtime SBOM 입력은 SPDX JSON(`SPDX-2.x`, `CC0-1.0`)이어야 한다. license manifest는
다음 형태로 Python package, runtime, model 세 scope를 모두 선언해야 한다. 데이터
전용 검사는 package inventory의 이름·버전이 license 선언에 포함됐는지 확인한다.
모델/runtime 선언의 법적 완전성을 자동 판정한다고 주장하지 않으며, 최종 이미지의
실제 OS·application 구성은 BuildKit의 OCI SBOM/provenance가 별도로 기록한다.
이 두 metadata 파일은 현재 활성 pointer만으로 완전하게 재생성할 수 없으므로 Linux
export 이전에 release input으로 준비해야 하는 명시적 선행조건이다. 누락되거나 scope,
package identity가 불완전하면 exporter가 대신 추정하지 않고 실패한다.

```json
{"components":[{"license_id":"Apache-2.0","name":"example","scope":"python-package","source_uri":"https://example.invalid/source","version":"1.0"}],"schema_version":"1.0"}
```

내보내기는 기존 디렉터리를 덮어쓰지 않는다. 전체 activation을 먼저 검증한
후 compact runtime이 실제로 읽은 파일과 sidecar/model release root만 staging에
복사하고, 최종 manifest 검증이 끝났을 때만 새 디렉터리로 원자적으로 바꾼다.
sealed V9의 7개 직접 evidence(stage-1 판단 receipt 포함)와 calibration의 중첩
closure를 포함하되, 2,400건 평가 corpus나 개별 결과는 포함하지 않는다. 배포 전용
공통 검증기가 7개 직접 evidence의 SHA를 확인하고, 권위 있는 Agent verifier가
선언된 activation-evidence를 전부 실제로 읽었는지 감사한다.

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

데이터 전용 verifier는 파일 운송 경계만 담당하고, 이어지는 Agent verifier는 실제
V19 profile fingerprint와 V9 receipt 의미를 권위 있는 서비스 코드로 검증한다. 게시
스크립트와 최종 Docker 빌드는 둘 중 하나라도 실패하면 중단한다. Exporter와 이
결합 검증은 다음을 거부한다.

- symlink, special file, 누락·추가·변경된 file
- Windows interpreter 또는 Linux/amd64가 아닌 bundle
- 원본 PDF/JPG/PNG, raster/Gold corpus, 2,400건 평가 결과 전체. Python package의
  CSV/PNG resource는 content-addressed sidecar manifest의 정확한 path·size·SHA에
  이미 포함된 경우만 허용한다.
- cache, secret 디렉터리, group/other writable file 또는 directory
- 128개 또는 64MiB를 넘는 compact activation evidence
- trust-root deployment manifest까지 합쳐 25,000개를 넘거나 총 2GiB를 넘는 전체
  runtime closure
- SBOM/license manifest가 없거나 pointer/profile hash와 맞지 않는 bundle

직접 Paddle Linux bundle은 패키지의 작은 파일이 많아 20,000개에 가까울 수 있다.
현재 상한 25,000개는 그 실측 예상치에 25%의 제한된 여유만 둔 값이다. Export와
publish 결과의 `file_count`를 기록해 실제 closure가 확인되면 코드를 검토해 더
좁힐 수 있지만, 실행 인수로 임의 완화할 수는 없고 2GiB 상한도 유지된다.

## 2. runtime context를 private GHCR에 게시

최초 한 번은 GitHub의 **Raster OCR OCI publish** workflow를 `main`에서
`operation=bootstrap`, 빈 `runtime_image`로 실행한다. Workflow는 `GITHUB_TOKEN`과 정확한
`org.opencontainers.image.source` label로 runtime과 최종 Agent package에
filesystem layer가 없는 `FROM scratch` placeholder만 게시한다. 그 뒤 두
package가 모두 `private`인지 API로 확인한다. 이 단계에는 모델이나
runtime byte가 없으며, workflow 저장소가 package에 자동 연결되고
쓰기 권한도 실제 placeholder push로 입증된다.

게시 job은 `raster-ocr-production` environment에 고정된다. 이 environment는
최초 실행 전에 required reviewer, self-review 금지, `main`만 허용하는 custom
deployment branch policy 하나를 갖춰야 한다. Workflow가 이 설정을 API로
다시 검증하므로 미설정 environment가 자동 생성되어도 registry 쓰기 전에
실패한다. Private 저장소에서 required reviewer를 지원하지 않는 GitHub
plan은 외부 배포 승인 선행조건을 만족하지 못하므로 fail-closed된다.

먼저 제한된 `packages:write` 권한으로 `docker login ghcr.io`를 수행하고, API 조회가
가능한 token을 `GH_TOKEN` 환경변수로만 제공한다. Token은 스크립트 인수나 출력으로
전달하지 않는다. 위 bootstrap이 만든 package가 존재하고 `private`일 때만
아래 스크립트가 runtime byte를 push한다. Public, API 404, 권한 오류는 모두
최초 모델 byte 게시 전에 실패한다.
이미 public이 된 container package는 private으로 되돌릴 수 없으므로 해당 고정
package 이름이 public이면 자동 복구하지 않고 새 검토된 이름으로 전환해야 한다.

```bash
bash deploy/ocr-oci/publish_runtime_context.sh \
  /srv/customs/release-output/raster-runtime-context \
  ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime
```

스크립트는 runtime manifest를 다시 검증한 뒤 권위 있는 Agent loader로 bundled
interpreter를 실행해 package/model 검사, `cv2`·Paddle import와 sidecar warm ready를
확인한다. Push 직전과 직후 GitHub Packages API에서 같은 target package가 계속
`visibility=private`인지 확인한다. 그 다음 OCI provenance/SBOM attestation을 붙인
아래의 변경 불가능한 참조만 출력한다.

```text
runtime_image=ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime@sha256:<digest>
```

이 값을 두 번째 수동 workflow 실행의 `operation=publish` 및
`runtime_image` 입력으로 정확히 전달한다. `bootstrap`과 `publish`는 명시적으로
분리되며 저장소 variable에서 runtime digest를 암묵적으로 fallback하지 않는다.

## 3. 최종 Agent+OCR 이미지 게시

GitHub의 **Raster OCR OCI publish** workflow를 `main`에서만 수동 실행한다.
Tag나 branch push는 배포를 시작하지 않는다. Workflow는 다음 순서를 지킨다.

1. dispatch ref·type·checkout SHA가 모두 현재 `main` tip과 일치하는지 확인한다.
2. 독립 reviewer가 승인한 environment와 self-review 금지·`main` 정책을 API로 재검증한다.
3. `bootstrap`은 고정된 asset-free placeholder를 runtime/output 두 package에 게시하고,
   `publish`는 output package에 같은 placeholder를 다시 push해 쓰기 권한을 입증한다.
   대상 package는 모두 `visibility=private`인지 즉시 확인한다.
4. `publish` 모드에서 runtime 입력이 정확한
   `ghcr.io/<owner>/<repo>-raster-runtime@sha256:<64 hex>`이고 현재 token으로
   private pull 가능한지 확인한다. `bootstrap` 모드는 자산 build 없이 요약만
   남기고 종료한다.
5. `deploy/ocr-oci/Dockerfile`의 `release` target을 `linux/amd64`로 build한다.
6. 네트워크가 차단된 build stage에서 모든 file hash·SBOM·license 운송 경계와 실제
   Agent V12/V19/V9 binding, bundled interpreter의 native import/warm ready를 재검증한다.
7. 최종과 같은 UID 10001로 warm 검증한 non-root final image를 GHCR에 push하고 OCI
   provenance/SBOM을 첨부한다.
8. push 뒤 output package가 여전히 private인지 다시 확인한다.
9. workflow summary에 최종 `image@sha256:digest`를 남긴다.

이 publish build는 private OCR 모델이 중간 layer에 포함되므로 GitHub Actions
cache를 import하거나 export하지 않는다. 재사용은 private GHCR runtime
digest와 배포 노드의 로컬 Docker layer cache 경계 안에서만 허용한다.

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
- 응답의 `raster_text_profile_revision`이 이미지 속 schema-12 pointer와 정확히 일치

서비스 시작 때 2,400건 평가를 다시 수행하지 않는다. compact byte 검증 후 OCR
child를 warm-up하며, 완료 전에는 health가 `starting`/`unhealthy`이고 Raster OCR
요청은 fail-closed 된다. 배포 오케스트레이터는 컨테이너 실행 여부가 아니라 이
OCR health가 성공한 뒤에만 Backend 트래픽을 열어야 한다.

현재 서비스 mode는 `local-development`와 `functional-uat`뿐이다. 이 OCI 경로는
검증된 배포 후보를 재현하는 경계이며, 별도의 Production Release·KMS·외부 서명
정책을 새로 주장하지 않는다.
