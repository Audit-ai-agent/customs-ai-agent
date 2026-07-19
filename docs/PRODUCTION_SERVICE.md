# AI Agent Production Service

## 서비스 경계

이 프로세스는 OCR을 활성화하지 않은 text-only AI Agent다. Backend와 Frontend를 대신하지
않으며 Case, 사용자, 권한, Job, HITL, 결과 영속화 및 복제본 간 durable idempotency를 소유하지
않는다.

```text
Backend or trusted service
  -> POST /internal/agent/v1/reviews
AI Agent
  -> Claude text structured extraction
  -> Evidence readiness/retrieval HTTP
  -> redacted ReviewResult V2
```

Agent의 유일한 제품 실행 엔진은 `AgentRuntime.review_json(raw_bytes)`다. ASGI 계층은 인증,
streaming body 상한, text-only preflight, admission, 단일 프로세스 single-flight/replay 및
HTTP 오류 변환만 담당한다. 기존 workflow, Claude grounding, Evidence Gate와 outcome 정책은
변경하지 않는다.

## API

| Endpoint | 인증 | 의미 |
|---|---|---|
| `GET /health/live` | 없음 | 프로세스 liveness. downstream을 호출하지 않는다. |
| `GET /health/ready` | 없음 | exact Evidence identity와 provider readiness를 확인한다. Claude 과금 호출은 하지 않는다. |
| `GET /internal/agent/v1/metadata` | Bearer | Agent build/wheel, text capability, model/prompt/schema/Data pin |
| `GET /internal/agent/v1/metrics` | Bearer | ID label이 없는 process-local 집계 |
| `POST /internal/agent/v1/reviews` | Bearer | text-only ReviewResult V2 실행 |

OpenAPI endpoint를 런타임에 공개하지 않는다. canonical OpenAPI는
`contracts/agent/service/v1/openapi.lock.json`의 hash·크기·route로 고정하고
`tools/verify_service_openapi.py`로 drift를 검사한다.

## 입력 불변식

- `Content-Type`은 `application/json`과 선택적 UTF-8 charset만 허용한다.
- 압축 body와 중복 Content-Type/Length/Encoding/Authorization header를 거부한다.
- body는 buffering 중 `CUSTOMS_AGENT_MAX_BODY_BYTES`를 넘는 즉시 413으로 종료한다.
- strict UTF-8, duplicate JSON key, NaN/Infinity를 거부한다.
- `review_run_id`와 `review_as_of`는 service profile에서 반드시 명시한다.
- `image_pages` 또는 `pdf_pages` key는 값이 `null`이어도 nested image decode, OCR, Claude,
  Evidence 호출 전에 422 `visual_input_not_supported`로 거부한다.
- Service document/text 상한은 core 상한보다 같거나 작아야 한다.
- Mock/scenario 실행은 service route에 존재하지 않는다.

## 인증과 네트워크

`CUSTOMS_AGENT_SERVICE_BEARER_TOKEN`은 최소 256-bit CSPRNG 결과를 사용한다.

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Agent는 token을 SHA-256 고정 길이 digest로 바꾼 뒤 `hmac.compare_digest`로 비교한다. 이
credential은 end-user 인증이 아니라 Backend 또는 승인된 내부 호출자의 service-to-service
credential이다.

- 외부에 직접 공개하지 않는다.
- TLS/mTLS gateway 또는 private service mesh 뒤에 둔다.
- gateway log가 Authorization과 body를 기록하지 않게 한다.
- 인증 token, Anthropic key와 Evidence token은 image, compose 파일, 명령 인자에 넣지 않는다.
- 외부 Evidence는 HTTPS exact allowlist만 허용한다.
- 운영 Evidence 인증은 `Authorization: Bearer <EVIDENCE_SERVICE_TOKEN>`만 허용한다.
  `X-Demo-Token` 같은 사용자 지정 header는 production 설정에서 거부한다.

## 동시성, timeout과 replay

하나의 container는 Uvicorn worker 한 개만 실행한다. `CUSTOMS_AGENT_MAX_CONCURRENCY`가 실제
Claude workflow thread 수를 제한하고 queue wait를 초과한 요청은 provider 호출 없이
`503 service_busy`와 `Retry-After`를 받는다.

`CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS`를 넘으면 호출자에게 504를 반환한다. Python의 동기 SDK
thread를 강제로 종료하면 client state를 손상할 수 있으므로 이미 시작된 provider 호출은 SDK
timeout까지 background에서 끝나며 그동안 semaphore와 in-flight 계수를 계속 점유한다. 따라서
timeout 뒤의 새 요청으로 concurrency 상한을 우회할 수 없다.

`review_run_id`는 process-local replay key다.

- 같은 ID와 canonical request hash: 진행 중 task를 공유하거나 TTL 안의 결과를 replay한다.
- 같은 ID와 다른 hash: 409 `idempotency_conflict`
- 결과/error는 bounded TTL과 bounded entry 수만큼 메모리에 유지한다.

이 보장은 container 재시작 또는 여러 replica 사이에서는 유지되지 않는다. durable Case/Job
idempotency는 Backend가 소유해야 한다. Backend 구현 전 여러 Agent replica를 active-active로
사용해 exactly-once를 주장해서는 안 된다.

## Readiness 의미

`/health/ready` 200은 다음만 뜻한다.

- service 설정과 wheel-local contract/prompt metadata가 로드됐다.
- Data readiness가 Agent wheel에 고정된 schema/release/revision/build와 일치한다.
- Data가 자신의 corpus/store readiness Gate를 통과했다.

Claude API에 과금 probe를 보내지 않는다. 현재 팀 Data가 3/58이면 503이 정확한 결과다.
conformance Provider의 200을 실제 법령 RAG 품질이나 전체 서비스 release로 표현하지 않는다.

## 결과와 관측성

로그는 JSON 한 줄이며 event, 생성된 request ID, safe error code, HTTP status, retryability,
duration, in-flight만 허용한다. body, case ID, review_run_id, 문서 text, Evidence text, prompt,
provider raw response와 secret은 로그에 넣지 않는다.

metrics는 label 없는 count/gauge만 제공한다. caller 또는 문서 식별자를 label로 추가하지 않는다.
ReviewResult의 model audit는 기존 공개 계약의 safe provider request ID, token 및 latency만
사용한다.

## Container

Dockerfile은 CI에서 이미 생성·검증한 exact wheel만 입력받는다.

```powershell
$Uv = "uv"
$Wheel = "dist/wheel/customs_ai_agent-0.3.0-py3-none-any.whl"
$Uv export --locked --no-dev --no-emit-project --no-emit-local --no-sources `
  --format requirements.txt --output-file dist/runtime-requirements.txt
$WheelSha = (Get-FileHash $Wheel -Algorithm SHA256).Hash.ToLowerInvariant()
$BuildSha = git rev-parse HEAD
docker build `
  --build-arg "AGENT_WHEEL=$Wheel" `
  --build-arg "BUILD_SHA=$BuildSha" `
  --build-arg "WHEEL_SHA256=$WheelSha" `
  -t "customs-ai-agent:$BuildSha" .
```

image는 wheel hash를 build 안에서 다시 확인한다. dependency는 `uv.lock`에서 export한
hash-pinned runtime requirements를 `pip --require-hashes`로 설치한 뒤 `pip check`한다. 이후
UID/GID 65532로 실행한다. filesystem write는 필요하지 않으며 compose 예제는 read-only root,
`/tmp` tmpfs, all capability drop, no-new-privileges와 resource limit을 적용한다.

Base image tag도 실제 release pipeline에서 digest로 고정해야 한다. Dockerfile 기본 tag는
개발 편의를 위한 것이며 digest 없는 build를 최종 release attestation으로 사용하지 않는다.

## 명시적 비범위

- OCR/image/PDF
- Frontend, 제품 Backend, 사용자 인증, HITL
- Operational DB와 Case/Job 저장
- 58법령 수집 및 실제 RAG 품질
- 자동 세관 승인 또는 법적 최종 판단

이 service 구현이 통과해도 실제 Data Release Gate가 막혀 있으면 배포 상태는
`implemented, dependency-unready`다.
