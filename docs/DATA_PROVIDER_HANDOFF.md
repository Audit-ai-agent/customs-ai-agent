# Evidence Provider 연동 handoff

## 현재 판정

Agent의 Evidence HTTP consumer는 provider-owned v1 JSON Schema와 lock을 wheel에 포함하고,
readiness identity·지원 날짜·snapshot·wire schema를 fail-closed로 검증한다. TASK-0010
Data candidate `7fa46fb487c08c4230312ffe539978f7141fa56f`는 runtime identity와
GET/POST build binding을 구현했다. 다만 실제 로컬 law corpus는 여전히 58개 중 3개만
적재돼 production readiness 대상이 아니다.

별도의 sealed test-only conformance Provider에서는 실제 HTTP GET/POST 계약을 검증한다.
실제 팀 Data에서는 Evidence POST 전에 의도대로 차단돼야 하며 Agent가 Mock으로 fallback하지
않는다. Agent에서 `required`를 `optional`로 낮추거나 provider 신원을 추정해 우회하지 않는다.

## Data Provider candidate가 제공하는 것

기존 `evidence-api-v1.0.0` tag와 frozen v1 response body를 수정하지 않았다. 새
`evidence-api-v1.1.0` candidate는 다음 runtime identity를 build artifact에 주입하고 typed
readiness response의 `evidence_contract`로 제공한다.

- contract name과 schema version
- packaged canonical schema bytes에서 다시 계산한 SHA-256
- immutable provider release
- 실제 실행 artifact를 만든 exact build revision
- OCI image 등 실제 provider artifact digest
- 필요하면 contract revision과 build revision의 분리

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

TASK-0010 Agent consumer는 1번 방식을 다음 exact header로 구현한다.

| Header | 값 |
|---|---|
| `X-Evidence-Schema-SHA256` | readiness의 `schema_sha256` |
| `X-Evidence-Provider-Release` | readiness의 `provider_release` |
| `X-Evidence-Provider-Revision` | readiness의 `provider_revision` |
| `X-Evidence-Provider-Build-SHA256` | readiness의 `provider_build_sha256` |

Agent는 각 POST에 네 expected header를 보내고, 각 response의 네 actual header를 body byte보다 먼저
exact match로 검증한다. 누락·불일치 시 body를 읽지 않고 실패한다. Data는 expected identity와
실행 build가 다르면 body 처리 전에 HTTP 412를 반환해야 한다.

현재 Agent lock은 Data candidate revision
`7fa46fb487c08c4230312ffe539978f7141fa56f`와 provider build SHA-256
`97c7c99886f333547b84b1f93584dcc7729ce021f2f365b4714f04d8ec75f0df`를 고정한다.
이 값은 clean commit에서 생성·검증한 local candidate identity이며, 원격 merge와 annotated
tag 발행을 의미하지 않는다.

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
