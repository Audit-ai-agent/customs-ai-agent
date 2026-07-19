# AI Agent Service Runbook

## 1. 배포 전 확인

1. 배포할 Git SHA가 review된 branch head와 정확히 같은지 확인한다.
2. `uv lock --check`, 전체 pytest, Ruff, mypy, Safety Gate와 OpenAPI drift를 통과시킨다.
3. sdist에서 wheel을 빌드하고 clean environment install/record 검증을 수행한다.
4. wheel SHA-256과 Git SHA를 release manifest에 기록한다.
5. Data provider release/revision/build/schema pin과 배포 대상 readiness를 확인한다.
6. Anthropic으로 전송할 문서의 권리, retention/ZDR와 개인정보 처리 승인을 확인한다.
7. service bearer를 CSPRNG로 새로 만들고 secret manager에 저장한다.

다음 값은 배포 필수다.

```text
ANTHROPIC_API_KEY
CUSTOMS_AGENT_SERVICE_BEARER_TOKEN
CUSTOMS_AGENT_ENVIRONMENT=production
CUSTOMS_AGENT_BUILD_SHA=<40 lowercase hex>
CUSTOMS_AGENT_WHEEL_SHA256=<64 lowercase hex>
EVIDENCE_SERVICE_BASE_URL=<exact URL>
EVIDENCE_SERVICE_ALLOWED_ORIGINS=<exact HTTPS origin when non-loopback>
EVIDENCE_SERVICE_TOKEN=<high-entropy Data service bearer>
EVIDENCE_SERVICE_TOKEN_HEADER=Authorization
CUSTOMS_AGENT_BODY_READ_TIMEOUT_SECONDS=15
```

secret 값 자체를 확인 명령이나 ticket에 붙이지 않는다. 존재 여부, 길이 정책 통과 여부와 secret
version만 기록한다.

## 2. 기동 확인

배포 직후 순서:

1. `/health/live`가 200인지 확인한다.
2. `/health/ready`를 확인한다.
3. 503이면 deploy를 성공으로 승격하지 않고 아래 dependency 진단으로 이동한다.
4. 인증된 `/internal/agent/v1/metadata`가 manifest의 Git/wheel/model/prompt/schema/Data pin과
   같은지 비교한다.
5. 실제 문서가 아닌 승인된 synthetic text로 한 번만 smoke한다.
6. result가 `needs_review`여도 정책상 정상일 수 있다. `completed`로 강제 변경하지 않는다.

## 3. Readiness 503

가능한 범주:

- Data service network/TLS/auth failure
- Evidence readiness 503
- provider schema/release/revision/build mismatch
- Data corpus/store projection 미완료
- Agent가 draining 중

조치:

1. Agent를 Mock 또는 optional identity로 바꾸지 않는다.
2. Data target law count를 낮추거나 빈 law row를 추가하지 않는다.
3. Agent metadata의 pin과 Data 배포 manifest를 값이 아닌 hash/identity로 비교한다.
4. Data 복구 후 readiness TTL이 지난 뒤 재확인한다.
5. retrieval POST를 강제로 보내 readiness를 우회하지 않는다.

현재 팀 Data 3/58은 readiness 503, Agent retrieval POST 0이 정상이다.

## 4. 401/403

- 401: Authorization Bearer가 없음
- 403: scheme, header cardinality 또는 credential이 다름

gateway가 Authorization을 제거하거나 중복 추가하지 않는지 확인한다. token 회전은 새 secret을
배포한 뒤 caller를 전환하는 rolling 절차로 수행한다. application log에 token을 출력하지 않는다.

## 5. 409 idempotency_conflict

같은 `review_run_id`가 다른 canonical body에 사용됐다. retry로 해결하지 않는다.

1. Backend Job과 payload hash 결박을 확인한다.
2. 새 업무 실행이면 새 review_run_id를 만든다.
3. 같은 업무 실행이면 변경된 body를 조사하고 기존 run을 보존한다.

process-local replay는 재시작/replica를 넘지 않는다. 여러 replica의 durable 중복 방지는 Backend
책임이다.

## 6. 503 service_busy

`Retry-After` 뒤 bounded retry한다. 즉시 replica/concurrency를 올리기 전에 다음을 확인한다.

- in-flight와 duration
- Claude rate limit/latency
- Evidence latency
- timeout 뒤 background worker 수
- CPU/memory 및 외부 API quota

한 process의 concurrency를 크게 올리는 것보다 동일 설정의 검증된 replica와 Backend durable
idempotency를 먼저 준비한다.

## 7. 504 review_timeout

호출자는 같은 review_run_id로 즉시 반복 호출하지 않는다. 기존 provider thread는 SDK timeout까지
in-flight로 남고 같은 ID는 504 outcome을 TTL 동안 replay한다.

1. 문서 수와 text 크기가 service limit 안인지 확인한다.
2. model_calls/error audit와 집계 latency를 확인한다.
3. Claude와 Evidence 장애를 구분한다.
4. timeout 값을 임의로 늘리기 전에 SLO, gateway timeout과 shutdown grace를 함께 검토한다.

## 8. 종료와 배포 교체

SIGTERM을 받은 process는 새 admission을 차단하고 진행 task를
`CUSTOMS_AGENT_SHUTDOWN_GRACE_SECONDS` 동안 drain한 뒤 Anthropic client를 닫는다. orchestrator
termination grace는 이 값보다 길어야 한다. 강제 종료가 발생하면 해당 review_run_id의 durable
상태를 Backend에서 확인하고 새 ID를 만들지 않은 채 재조정한다.

## 9. Rollback

1. 이전 image digest, Agent wheel SHA, Git SHA와 Data pin을 release manifest에서 선택한다.
2. 기존 tag를 이동하지 않는다.
3. Data provider와 호환되는 Agent pin인지 확인한다.
4. rollback image의 `/metadata`, `/ready`, synthetic smoke를 다시 검증한다.
5. 실패 release는 Incident/RCA로 남긴다.

## 10. 증거 보존

Artifact Run에는 secret/body 없이 다음을 기록한다.

- Agent Git SHA, wheel/image digest
- OpenAPI lock SHA
- model/prompt/schema/Data contract identity
- Gate command, exit code, duration
- aggregate metrics와 safe error code
- readiness 결과와 미검증 범위

원문, case/review ID, bearer/API key, raw Claude/Data response는 저장하지 않는다.
