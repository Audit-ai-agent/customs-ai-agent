# Evidence Provider 연동 handoff

## 현재 판정

Agent의 제품 Evidence consumer는 provider-owned retrieval v1 JSON Schema와 readiness v2
JSON Schema snapshot, 그리고 runtime release lock을 wheel에 포함한다. 제품 경계는
`GET /internal/rag/v2/health/ready`와 `POST /internal/rag/v2/retrieve`이며 retrieval body만
기존 v1을 유지한다. readiness의 provider build뿐 아니라 sealed knowledge generation,
model/index manifest, evaluation report, snapshot과 지원 날짜를 하나의 canonical
runtime-release identity로 fail-closed 검증한다.

별도의 sealed test-only conformance Provider에서는 실제 HTTP GET/POST 계약을 검증한다.
실제 팀 Data에서는 Evidence POST 전에 의도대로 차단돼야 하며 Agent가 Mock으로 fallback하지
않는다. Agent에서 `required`를 `optional`로 낮추거나 provider 신원을 추정해 우회하지 않는다.

## Data Provider candidate가 제공하는 것

기존 `evidence-api-v1.0.0`, `v1.1.0` 계약과 frozen v1 retrieval body를 수정하지 않는다.
새 `evidence-api-v1.2.0` 경계는 typed readiness의 `evidence_runtime_release`로 다음을
제공해야 한다.

- exact provider release/revision/build 및 provider source manifest SHA-256
- retrieval schema version/SHA-256
- sealed generation ID와 generation/source/seal attestation SHA-256
- model/index config manifest SHA-256
- 실제 locked Gold 평가 report SHA-256
- knowledge snapshot, supported review date
- 앞선 모든 필드의 canonical JSON SHA-256인 runtime release SHA-256

`.git` checkout이나 사람이 입력한 환경변수만 보고 실행 artifact 신원을 추정해서는 안 된다.
identity가 누락·오염됐거나 bundled schema hash와 다르면 readiness 503이어야 한다.

## GET readiness와 POST retrieval의 원자적 결박

readiness를 처리한 build와 retrieval을 처리한 build가 다른 rollout 상황도 검출해야 한다.
frozen v1 response body는 `additionalProperties: false`이므로 임의 identity 필드를 추가하지
않는다. provider-owned 새 계약에서 다음 중 하나를 확정한다.

1. POST request/response header에 expected/actual build revision, schema hash, artifact
   digest를 교환하고 불일치 시 처리 전 412로 거절한다.
2. readiness가 identity·snapshot·지원 날짜에 결박된 서명 attestation을 발급하고 POST가 이를
   검증·echo한다.
3. response body를 바꿔야 한다면 v1을 수정하지 않고 v2 endpoint/schema를 추가한다.

제품 Agent consumer는 다음 exact header로 GET에서 관측한 release를 POST에 결박한다.

| Header | 값 |
|---|---|
| `X-Evidence-Schema-SHA256` | readiness의 `retrieval_schema_sha256` |
| `X-Evidence-Provider-Release` | readiness의 `provider_release` |
| `X-Evidence-Provider-Revision` | readiness의 `provider_revision` |
| `X-Evidence-Provider-Build-SHA256` | readiness의 `provider_build_sha256` |
| `X-Evidence-Knowledge-Generation` | readiness의 `knowledge_generation_id` |
| `X-Evidence-Generation-Attestation-SHA256` | readiness의 `knowledge_generation_attestation_sha256` |
| `X-Evidence-Evaluation-SHA256` | readiness의 `evaluation_report_sha256` |
| `X-Evidence-Runtime-Release-SHA256` | readiness의 `runtime_release_sha256` |

Agent는 각 POST에 8개 expected header를 보내고, 각 response의 8개 actual header를 body byte보다 먼저
exact match로 검증한다. 누락·불일치 시 body를 읽지 않고 실패한다. Data는 expected identity와
실행 build가 다르면 body 처리 전에 HTTP 412를 반환해야 한다.

현재 Agent lock은 `release_state=unreleased`다. 실제 sealed generation과 locked Gold
evaluation report가 아직 없으므로 generation/evaluation hash를 임의 생성하거나 기존 3/58
snapshot을 released로 표시하지 않는다.

이 candidate에 맞춘 conformance provider 검증은 runtime evaluator의 명시적
`--provider-conformance`에서만 수행한다. 성공해도 report scope는
`PROVIDER_CONFORMANCE_NOT_RAG_QUALITY`, `release_ready=false`다. 실제 Data release 모드는
별도의 검증된 RAG 품질 attestation이 없으면
`BLOCKED_RAG_RELEASE_ATTESTATION`으로 종료한다. 따라서 동일한 conformance provider를
대상으로 플래그를 생략하는 것만으로 release-ready를 만들 수 없다. 최종 provider digest,
58/58 corpus와 RAG 품질 Gate를 별도로 요구한다.

## Data readiness 완료 조건

- target law corpus 58/58
- PostgreSQL/Qdrant/Neo4j 기반 law index 1007/1007/1007
- readiness HTTP 200
- supported review date와 immutable knowledge snapshot 제공
- runtime identity와 bundled schema hash 일치
- Data provider contract/OpenAPI drift 0
- 예상치 못한 Live test skip 0

expected count를 3으로 줄여 readiness를 통과시키는 것은 corpus scope를 공식 변경하는 별도
Data ADR과 평가셋 변경이 없는 한 허용하지 않는다.

## Cross-repository Live Gate

새 Data release 이후에만 다음 순서로 검증한다.

1. Data canonical schema와 Agent packaged schema의 byte/hash 일치
2. candidate build commit·artifact digest를 Agent lock에 고정하고, 발행 시 annotated tag를 검증
3. Agent → readiness → retrieval → ReviewResult 한 Case 실제 실행
4. request ID, item ID, review date, snapshot, citation metadata 일치
5. `partial`, `failed`, `ambiguous` Evidence의 자동 완료 0
6. header/attestation 누락·GET A/POST B rollout에서 Evidence 결과 폐기
7. secret, attestation token, raw Evidence 본문이 결과·로그에 포함되지 않음

Data 팀 변경 전 Agent 저장소에서 성공 상태를 조작하지 않는다.
