# Architecture

## 목적

이 Agent는 여러 통관 문서의 필드를 비교하고 규정 근거를 item 단위로 결합해 검토 결과를 만든다.
법률·규정 사실을 모델의 기억이나 자유 생성으로 만들지 않고, provider가 관리하는 Evidence Service의
버전된 응답에 결박하는 것이 핵심 불변식이다.

## 현재 데이터 흐름

```text
ReviewRunRequest
  -> fixture load
  -> classify -> extract -> bounded self-correction
  -> normalize -> pair -> cross-document validation
  -> RegulationRetriever
       mock: deterministic fixture (test only)
       live: POST /internal/rag/v1/retrieve (HTTP only)
  -> deterministic compliance + Evidence Gate
  -> canonical XAI JSON
  -> completed / needs_review / blocked
```

LangGraph는 순서·분기·재시도 상한을 관리한다. 정규화, 금액/수량/중량 비교, Evidence Gate와 최종
route는 결정론적으로 유지한다. 현재 분류·추출도 규칙 기반이며 실제 OCR/VLM adapter는 후속 구현이다.

## 경계와 소유권

| 경계 | 제공자 | 소비자 | 통합 방식 |
|---|---|---|---|
| Evidence Retrieval v1 | `customs-data-platform` | AI Agent | versioned JSON/HTTP |
| Agent Request/Result | AI Agent | Backend | versioned Python package/schema |
| 실행 상태·HITL | Backend | Frontend | Backend OpenAPI |

Agent가 Data DB에 직접 연결하면 snapshot, 권한, 배포와 schema 소유권이 섞이므로 금지한다.
consumer schema snapshot은 회귀 검증 편의를 위한 복사본이며 canonical 원본이 아니다.

## 실패 정책

- Evidence 응답은 schema와 request ID가 일치해야 한다.
- service unavailable, invalid response, partial/unknown coverage, ambiguous HS는 자동 완료를 막는다.
- Live RAG의 시점을 생략하면 provider readiness가 공개한
  `supported_review_as_of`를 사용하며 readiness 실패 시 Mock으로 전환하지 않는다.
- Finding과 Evidence의 `item_id`, context date, trade type, HS context를 검증한다.
- HTTP token은 헤더로만 주입하고 URL·오류 메시지에 노출하지 않는다.
- Mock은 명시적인 테스트 dependency이며 운영 fallback이 아니다.
- rerun에서 실제 retriever가 없으면 Mock으로 조용히 교체하지 않는다.

## 실제 AI 전환 방향

파일 검사·OCR 결과를 입력으로 받고, 문서 분류/추출과 제한된 설명에만 VLM/LLM adapter를 주입한다.
모델 응답은 strict structured candidate로 받고 page/bbox/raw text 근거를 검증한다. 실패 시 제한된 crop
reread 후 HITL로 전환하며, 모델이 최종 상태나 severity를 직접 바꾸지 못하게 한다.
