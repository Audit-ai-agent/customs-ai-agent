# Architecture

## 목적

이 Agent는 여러 통관 문서의 필드를 비교하고 규정 근거를 item 단위로 결합해 검토 결과를 만든다.
법률·규정 사실을 모델의 기억이나 자유 생성으로 만들지 않고, provider가 관리하는 Evidence Service의
버전된 응답에 결박하는 것이 핵심 불변식이다.

## 현재 데이터 흐름

```text
ReviewRunRequest
  -> text input / fixture load
  -> document analysis
       mock: deterministic rules
       llm: Claude structured candidate -> schema/allowlist/exact-span grounding
  -> extract -> bounded deterministic OCR correction
  -> normalize -> pair -> cross-document validation
  -> RegulationRetriever
       mock: deterministic fixture (test only)
       live: POST /internal/rag/v1/retrieve (HTTP only)
  -> deterministic compliance + Evidence Gate
  -> canonical XAI JSON
  -> completed / needs_review / blocked
```

LangGraph는 순서·분기·재시도 상한을 관리한다. Claude는 문서 유형과 verbatim field candidate만
반환한다. 정규화, 금액/수량/중량 비교, Evidence Gate, severity와 최종 route는 결정론적으로
유지한다. 모델 오류는 규칙 extractor로 대체하지 않는다.

Claude 호출은 문서당 한 번이며 분류와 추출을 같은 candidate로 결합한다. 응답은 strict Pydantic
schema와 document-type allowlist를 통과하고 모든 source quote/value를 원문과 대조한 뒤에만
downstream shape로 변환한다. 모델의 confidence는 사용하지 않는다. 현재 text-only provenance는
`page=1`, `bbox=None`, 실제 source quote이며 실제 OCR/layout bbox인 것처럼 표시하지 않는다.
입력은 JSON envelope로 직렬화하여 filename/newline/delimiter-like 문서 내용이 envelope 구조를
깨지 못하게 한다. quote는 원문의 case-sensitive exact slice, 값은 quote 안의 exact bounded
span이어야 하며 정규화 근사 일치는 이 경로에서 허용하지 않는다. document field quote는
원문에서 유일해야 하고 line item은 원문 occurrence 순서·개수대로 mapping되어 실제
`text_start/text_end`와 source slice를 저장한다.

exact grounding은 “문자열이 있었다”는 사실만 증명하고 Claude가 그 문자열을 올바른 업무 필드에
연결했다는 의미는 아니다. 따라서 0.2.x Claude field confidence는 0.85
`source_quote_grounded_uncalibrated`이고, 모델 호출·grounding·Evidence 검색이 모두 성공해도
최종 상태는 `needs_review`다. Claude 유래 critical issue도 사람 승인 계약이 생기기 전에는
자동 `blocked`로 승격하지 않는다.

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
- loopback 이외의 Evidence HTTPS origin은 명시적 exact-origin allowlist가 없으면 호출하지
  않는다.
- Evidence client는 redirect를 거부하고 응답 body를 2 MiB로 제한한다.
- 외부에서 받은 128자 document/review run ID는 suffix로 계약 길이를 넘지 않도록
  prefix+SHA-256 기반의 결정론적 bounded item/request ID로 변환한다.
- Mock은 명시적인 테스트 dependency이며 운영 fallback이 아니다.
- `llm_mode=llm`은 명시적 retriever 없이는 시작하지 않는다.
- Claude provider/schema/grounding 실패는 safe error code와 HITL로 닫히며 규칙 추출로
  fallback하지 않는다.
- 인증·권한·rate limit·provider outage처럼 run 전체에 적용되는 오류가 한 번 발생하면
  circuit을 열고 남은 문서는 호출하지 않은 safe error로 기록한다.
- rerun에서 실제 retriever가 없으면 Mock으로 조용히 교체하지 않는다.
- 성공 모델 audit는 provider/model/prompt/request/message/usage/latency를 기록한다. 실패는
  provider/model/prompt/latency와 SDK가 제공한 안전한 request ID/status만 기록하며 key, 원문,
  exception body, prompt와 raw response는 기록하지 않는다.

## 남은 OCR/VLM 전환 방향

현재 Claude adapter는 `content_text`에 대한 실제 분류·추출이다. 다음 단계에서 파일 검사,
digital PDF text/layout 추출, scan OCR, page/block/table 좌표 계약을 추가한다. 그때도 VLM/LLM
응답은 strict candidate로 받고 page/bbox/raw text를 OCR/layout 원장과 검증해야 한다. 제한된 crop
reread 후에도 근거가 없으면 HITL로 전환하며 모델이 최종 상태나 severity를 바꾸지 못하게 한다.
