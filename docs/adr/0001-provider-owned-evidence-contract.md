# ADR-0001: Provider-owned Evidence 계약을 HTTP로 소비한다

- 상태: Accepted
- 일자: 2026-07-14

## 상황

Agent는 법령·HSK·CCCT 근거가 필요하지만 해당 데이터와 PostgreSQL/Qdrant/Neo4j 운영 책임은
Data Platform에 있다. 소스 복제나 DB 직접 연결은 데이터 snapshot과 권한 경계를 깨고 독립 배포를
어렵게 만든다.

## 결정

`customs-data-platform`이 Evidence Retrieval schema의 canonical owner다. AI Agent는
`POST /internal/rag/v1/retrieve`만 호출하고, provider schema version과 SHA-256을 lock manifest에
고정한다. 이 저장소의 schema 파일은 consumer regression용 pinned snapshot이다. breaking change는
기존 v1을 덮어쓰지 않고 새 API/schema 버전으로 제공한다.

Mock retriever는 결정론적 테스트에서만 허용한다. Live/운영 실행은 HTTP retriever를 명시적으로
주입하며 실패 시 Mock으로 fallback하지 않는다.

## 결과

- 장점: 소유권, 배포, 접근제어, snapshot 추적과 계약 검증이 분리된다.
- 비용: provider/consumer release 조율과 계약 pin 갱신이 필요하다.
- 검증: lock hash, JSON Schema wire example, fail-closed retrieval 회귀 테스트를 CI에서 실행한다.

## 2026-07-18 Runtime identity 보강

Checkout의 lock 검사만으로는 실제 HTTP 서버가 그 release를 실행 중인지 증명할 수 없다. 따라서
서비스용 `AgentRuntime`은 첫 retrieval 전에 readiness를 호출하고 다음 응답을 요구한다.

```json
{
  "status": "ready",
  "knowledge_snapshot_id": "<non-empty snapshot id>",
  "supported_review_as_of": "YYYY-MM-DD",
  "evidence_contract": {
    "contract": "evidence-retrieval",
    "schema_version": "1.0",
    "provider_release": "evidence-api-v1.0.0",
    "provider_revision": "<40 lowercase hex>",
    "provider_build_sha256": "<64 lowercase hex>",
    "schema_sha256": "<64 lowercase hex>"
  }
}
```

Agent wheel은 `customs_ai_agent.contracts/evidence/v1.lock.json`과 그 SHA-256에 대응하는
`evidence/v1.schema.json`을 함께 배포한다. 시작 시 lock/schema 결합을 확인하고 모든 HTTP
request/response 원문을 packaged provider schema로 검증한다. readiness의
release, revision, schema SHA가 이 lock과 모두 같아야 retrieval POST가 허용된다. readiness가
광고한 `supported_review_as_of`와 요청 날짜도 같아야 한다. HTTP retriever 자체의 기본값이
`required`이므로 CLI와 evaluation runner도 별도 설정 없이 같은 검증을 거친다. POST 응답의
`knowledge_snapshot_id`는 직전 readiness snapshot과 같아야 하며 rollout 중 값이 바뀌면 응답을
버린다. identity 누락, 형식 오류, 값 불일치, 지원하지 않는 날짜와 snapshot 불일치는 모두
실패로 바뀌며 최종 결과는 `needs_review`가 된다.

TASK-0010 Data candidate는 `evidence_contract`를 Provider-owned build artifact에서 생성해
readiness로 반환한다. Agent는 이 값을 추정하거나 대신 채우지 않는다. 실제 3/58 corpus는
readiness 503으로 차단되고, sealed test-only conformance Provider만 명시적인 비릴리스
상호운용성 검증에 사용한다. 서비스 `AgentRuntime`은 optional을 구성 단계에서 거절한다.
`disabled`는 HTTP transport 단위 테스트에서만 사용하며 CLI, evaluation 또는 서비스 실행
설정으로 노출하지 않는다.

## Provider와 합의해야 하는 잔여 release 조건

현재 lock은 Data candidate revision
`7fa46fb487c08c4230312ffe539978f7141fa56f`와 clean source manifest에서 계산한
`provider_build_sha256`
`97c7c99886f333547b84b1f93584dcc7729ce021f2f365b4714f04d8ec75f0df`를 함께 고정한다.
이는 검증된 local candidate identity이며 annotated tag나 production 배포 증거는 아니다.
두 consumer lock과 evaluation literal은 byte-identical identity를 사용하고
Cross-repository Gate에서 다시 검증한다.

readiness GET과 retrieval POST는 서로 다른 load-balancer target에서 처리될 수 있다. Agent는
readiness에서 검증한 identity를 다음 네 POST header로 보낸다.

- `X-Evidence-Schema-SHA256`
- `X-Evidence-Provider-Release`
- `X-Evidence-Provider-Revision`
- `X-Evidence-Provider-Build-SHA256`

Data는 처리한 build의 exact identity를 같은 네 response header로 echo해야 한다. Agent는 body
byte를 읽기 전에 네 값을 exact match로 검증하고 누락·불일치 시 body를 폐기한다. Data가 expected
identity와 다른 build라면 처리 전에 HTTP 412를 반환한다. 이 transport 결박은 frozen v1 body를
변경하지 않는다. 실제 GET A/POST A conformance Gate가 통과하기 전에는 “원자적 build
attestation 완료”로 표시하지 않으며, 그 통과도 실제 58법령 RAG 품질을 뜻하지 않는다.
