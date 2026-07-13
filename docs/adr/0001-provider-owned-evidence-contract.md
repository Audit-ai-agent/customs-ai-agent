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
