# Architecture

## 목적과 신뢰 경계

이 Agent의 목적은 문서 내용을 “그럴듯하게 요약”하는 것이 아니라 다음 질문에 검증 가능한
결과를 만드는 것이다.

1. 입력된 문서 묶음은 어떤 유형인가?
2. 각 업무 필드는 원문의 어느 text span 또는 image bbox에 근거하는가?
3. invoice, packing list, B/L, C/O 사이의 값은 정합한가?
4. 판단에 사용한 법령·HSK·요건 근거는 Data Platform의 어느 versioned record인가?
5. 자동 판정할 수 없는 실패·모호성은 사람에게 전달됐는가?

따라서 Claude, OCR, RAG 결과를 모두 신뢰하지 않는 관측으로 취급한다. 각 관측은 schema,
allowlist, source grounding, provenance, Evidence Gate를 통과해야 한다.

## 전체 데이터 흐름

```text
Local review manifest / Backend Agent V1·V2 request
  |
  +-- text ---------------------------------------------+
  |     UTF-8 + length validation                       |
  |                                                     v
  +-- image --> decode/hash/dimension/provenance --> document analyzer
  |                                                     |
  +-- PDF --> bounded PDFium raster pages --------------+
                                                        |
                         +------------------------------+------------------+
                         |                                                 |
                 deterministic/mock                                  llm mode
                                                                           |
                                               independent OCR/layout provider
                                                 paddle | docling | ledger
                                                                           |
                                                  immutable OCR layout ledger
                                                                           |
                                              Claude text/vision candidate
                                                                           |
                                          schema/allowlist/span+bbox grounding
                                                                           |
                                    extract -> normalize -> pair -> validate
                                                                           |
                                        Evidence Service HTTP retrieval
                                                                           |
                                 compliance -> Evidence Gate -> XAI result
                                                                           |
                                    completed | needs_review | blocked
```

LangGraph는 단계 순서, 분기, bounded retry를 관리한다. 모델은 단계 자체를 재작성하거나 임의 tool,
DB, network를 선택하지 않는다.

## 입력 계층

### V1 텍스트

V1은 기존 `DocumentInput.content_text` 계약을 보존한다. Claude는 문서당 한 번의 structured
candidate로 분류와 추출을 수행한다. 모든 non-empty value는 case-sensitive exact source quote와
offset에 결박된다. 텍스트에는 실제 image 좌표가 없으므로 bbox를 합성하지 않는다.

### V2 이미지

V2 `image_pages`는 arbitrary file path/URL이 아니라 검증된 wire object다.

- 지원 encoding: JPEG, PNG, WebP
- page별 decoded byte length와 SHA-256
- 실제 decoded width/height
- source media type/hash/page
- EXIF transpose 등 적용 transform
- 문서별·run별 page/byte 상한

파일 경로는 CLI edge에서만 사용하고 `ReviewRunRequest`에는 넣지 않는다. Backend가 V2 request를
만들 때도 object storage URL을 Agent에 전달하는 대신 승인된 업로드 계층에서 byte를 읽어 동일한
`ImagePageInput`을 만들어야 한다.

### PDF

현재 PDF는 텍스트를 신뢰해 직접 추출하지 않고 page image로 rasterize한다. parent adapter는
PDF bytes를 `python -I` worker에 한 번 전달한다. worker는 neutral working directory와
allowlist 환경, 한 줄 bounded JSON protocol에서 PDFium을 실행하며 stderr는 공개 결과에
전달하지 않는다.

worker는 signature/trailer, polyglot, encryption, page 수, MediaBox, 예상 pixel dimension과
전체 pixel budget을 render 전에 검증한다. parent는 응답 protocol/request ID, PDF hash,
page 순서, PNG hash·signature·dimension·byte budget을 다시 확인한다. rasterizer
ID/version/DPI, PDF source hash와 source page는 provenance에 남기며 페이지 초과 PDF를 조용히
잘라내지 않고 실패한다.

현재 제한은 process/protocol isolation이다. Windows Job Object, container/seccomp,
CPU/memory quota를 적용한 OS-level sandbox는 아니며 운영 전 별도 보안 Gate가 필요하다.

이는 digital PDF text layer 활용 최적화가 아니라, 스캔과 digital PDF를 같은 vision grounding
경계에서 정확히 검증하기 위한 현재 선택이다. 추후 native text/layout extractor를 추가할 경우
별도 관측 원장과 source hash를 갖는 새로운 입력 경계로 설계해야 한다.

## OCR/layout과 Claude vision의 역할 분리

OCR/layout provider는 Claude와 독립적으로 다음 immutable ledger를 만든다.

```text
ledger_id
document_id
provider_id / engine_id / engine_version
page_number / image_sha256 / width / height
span_id / reading_order / role / text / bbox / confidence
```

`ledger_id`는 canonical ledger 내용의 SHA-256으로 계산한다. Agent는 ledger의 document ID,
page 순서, page image hash와 크기를 현재 입력 이미지와 다시 대조한다.

Claude vision은 업무 필드 후보와 그 후보가 참조하는 OCR span/bbox를 반환한다. 다음 조건 중 하나가
깨지면 후보를 사용하지 않는다.

- structured schema와 field allowlist
- document type별 허용 field
- 참조한 OCR span 존재
- value/source text grounding
- bbox가 원장과 page bounds에 일치
- line item occurrence/reading order의 결정론적 mapping

제한된 crop reread는 이미 검증된 page의 bounded crop에만 사용할 수 있다. crop은 새로운
source-of-truth가 아니며 전체 run budget을 넘기지 않는다. 끝까지 ground할 수 없는 값은
HITL 대상으로 남긴다.

`paddle`과 `docling`은 격리된 subprocess provider다. `ledger` provider는 사전에 고정한 관측을
사용하는 재현 평가용이다. ledger PASS는 OCR 엔진 품질 PASS가 아니다.

활성 OCR provider는 코드 존재나 smoke가 아니라 canonical descriptor의 locked 60문서와
C/O 20문서가 모두 고정 threshold를 통과한 뒤에만 선택한다. 과거 4문서 lightweight benchmark와
pre-freeze 1문서 smoke는 구현 진단 이력이며 provider 선택 증거가 아니다. exact-commit wheel로
60+20 평가와 hardened 결합 Gate를 끝내기 전에는 선택된 provider가 없다. 둘 다 실패하면 Claude
bbox를 승격하지 않고 image release path를 `BLOCKED`로 유지한다.

Claude Vision Live evaluator와 제품 `review` CLI는 향후 외부에서 공급된 evaluation-v3
sealed/unseen bundle과 그 계약으로 생성된 canonical descriptor/rendered manifest를 검증하고,
`ocr-release-gate-report/v2`의 exact hash, 80문서 coverage, PASS selection,
provider/engine/wheel identity, 선택 provider의 dependency lock, model-assets lock과 실제
runtime inventory가 모두 일치해야 OCR provider나 Anthropic client를 생성한다. CLI에는 암묵적
provider가 없고 locked-only report나 report 경로만으로 이 경계를 통과할 수 없다.
현재 repository-visible evaluation-v2는 공개 seed로 membership이 도출되어 영구 무효다.
release runner, descriptor builder와 공개 OCR combiner는 파일 접근 전에
`release_holdout_v2_publicly_derivable`로 중단한다. 구 결합 규칙은 별도 schema와
`NONRELEASE_DIAGNOSTIC_*`, `release_eligible=false`만 생성한다. v3 sealed runner도 아직
없으므로 code-owned descriptor hash는 `NOT_FROZEN`이고 provider·Claude 생성 전에 중단한다.

향후 v3 descriptor freeze는 C0 authorization commit과 C1 literal-only pin commit으로
분리한다. descriptor는 `release_context.authorization_commit_sha`로 C0만 참조하며 C1 SHA를
source나 descriptor에 넣지 않는다. C1은 C0의 single-parent child이고 descriptor SHA와 C0 SHA
두 literal 외의 변경을 허용하지 않는다. benchmark는 provider 생성 전에 clean C1, parent,
exact diff, descriptor·receipt identity를 검증한다. C1 뒤 행위 코드가 바뀌면 기존 freeze와
wheel-bound report는 모두 무효다. 전체 절차는 `docs/OCR_RELEASE_FREEZE.md`를 따른다.

두 provider가 품질·crash/all-documents Gate를 모두 통과하고 locked 품질 tuple도 같을 때만
identity-bound runtime 지표를 사용한다. 순서는 locked sample의 end-to-end p95 latency,
isolated worker peak memory, 설치 distribution과 attested model asset의 disk footprint다.
benchmark report의 문서별 latency에서 p95를 결합기가 다시 계산하며, runtime 지표 누락·비정상,
wheel/environment/model/engine identity drift, 동일 provider의 sample 간 disk identity drift,
세 지표 전체 동률은 모두 자동 선택 없이 fail-closed한다.

비용 제한 Live subset은 성공해도 `DIAGNOSTIC_PASS`로만 기록한다. 3,000장 계열은 two-reviewer
Gate가 고정한 12 Case/36문서 전체, rendered 계열은 locked 60+C/O regression 20 전체를 exact
ID·count·hash로 평가한 경우에만 extraction release `PASS`가 가능하다.

wheel은 결합 report의 hash, caller가 명시한 expected hash, OCR Python의 PEP 610 metadata에서
읽은 installed-wheel hash의 3-way 일치와 설치 `RECORD`/console executable 무결성을 요구한다.
dependency lock은 Python 3.11/Windows 환경의 전체 installed distribution inventory와 비교한다.
model-assets lock은 Paddle official-model tree 또는 Docling exact Hugging Face snapshot의
revision/file count/byte count/content-tree hash와 report engine ID를 결박한다.
provider별 dependency lock hash, 공통 model-assets lock hash, provider engine ID는 코드가
소유한 canonical 상수에도 결박되어 임의 lock/report/caller 조합으로 대체할 수 없다.

선택된 runtime provider는 attested wrapper로 감싸며, 각 실제 OCR ledger의 `provider_id`,
`engine_id`, `engine_version`, canonical model-set ID, model-assets lock SHA-256을 다시 대조한
뒤에만 Claude transport로 전달한다. 설치 뒤 엔진이 바뀌거나 provider/model identity가 다른
ledger를 반환하면 외부 호출 전에 중단한다.

## 결정론적 Agent workflow

모델 이후 단계의 소유권은 다음과 같다.

| 단계 | 책임 | 모델 권한 |
|---|---|---|
| extract | grounded candidate를 내부 field shape로 변환 | candidate만 제공 |
| normalize | 날짜·금액·수량·중량·코드 표준화 | 없음 |
| pair | 문서/line item 연결 | 없음 |
| validate | 교차문서 불일치와 누락 검출 | 없음 |
| retrieve | item별 Evidence HTTP 요청 | query 의미를 코드가 구성 |
| compliance | requirement/evidence 연결 | 법률 자유 생성 금지 |
| report | canonical XAI JSON | 자유 서술로 판정 변경 금지 |
| route | completed/needs_review/blocked | 없음 |

Claude의 exact grounding은 “해당 문자열/좌표가 존재했다”는 증거다. 그 문자열을 올바른 업무
필드에 연결한 정확도까지 운영 수준으로 보정됐다는 증거는 아니다. 현재 정책이 성공 LLM run도
사람 검토 대상으로 보내는 이유다.

controlled-defect 결정론적 평가는 `trade-document-evaluation-assets/v3` manifest와
`deterministic-validation-evaluation-report/v4` report를 사용한다. evaluator는
dataset-lock, opaque evaluation-v2 provenance bytes, controlled-defect와 manifest의 canonical
hash만 확인한다. split JSON을 파싱하거나 membership/holdout seed를 재계산하지 않는다.
development 20개 Case와 operator assignment는 이미 hash-locked controlled manifest에서만
읽는다. report는 항상 `release_eligible=false`,
`release_holdout_status=release_holdout_v2_publicly_derivable`를 기록한다.

## Data Platform 경계

```text
AI Agent
  -> GET  /internal/rag/v1/health/ready
  -> POST /internal/rag/v1/retrieve
Data Platform
```

Agent가 PostgreSQL/Qdrant/Neo4j에 직접 연결하거나 Data repo 모듈을 import하는 것은 금지한다.
Evidence response는 request ID, item ID, review date, trade type, HS context, coverage, snapshot,
citation metadata를 검증한다.

다음 상태는 Mock으로 fallback하지 않고 자동 완료를 막는다.

- service/readiness/network 실패
- schema 또는 request/item ID 불일치
- partial/unknown coverage
- ambiguous/invalid/missing HS resolution
- finding과 evidence 연결 실패
- citation metadata 부족

외부 Evidence origin은 HTTPS와 exact allowlist가 필요하다. credential/query/fragment/path prefix,
redirect, 2 MiB 초과 response는 거부한다. token은 header에만 둔다.

## 공개 결과와 민감 데이터 수명

원본 image base64는 graph 실행에 필요한 execution-local payload다. orchestration layer는 실행
후 참조를 해제하고, graph state/report/result에는 원본 payload를 남기지 않는다.

공개 `ReviewRunResult`는 업무 결과, Evidence, 검증 이슈, 모델의 비민감 audit metadata만 담는다.
직렬화 직전 recursive safety gate가 다음 key를 탐지하면 결과 쓰기를 중단한다.

- `documents`, `content_text`, `image_pages`, `data_base64`
- source/manifest/OCR-ledger path
- raw prompt/response

모델 audit에는 provider/model/prompt version, request/message ID, token, latency, safe error code만
허용한다. key, 원문, prompt, provider response body와 SDK exception body는 기록하지 않는다.

## 계약과 Backend 독립성

```text
Data Platform -- Evidence API v1 --> AI Agent
AI Agent      -- Agent V1/V2 -----> Backend (future)
Backend       -- Backend OpenAPI --> Frontend (future)
```

Agent V1은 text request/result 1.0을 보존한다. V2는 image pages와 result 2.0을 추가한다.
Backend/service용 공개 package 실행 경계는 `AgentRuntime.review()` 하나이며 현재 V2 결과만
반환한다. `customs_ai_agent.runner`의 V1/V2 helper는 mock scenario와 임의 analyzer/retriever를
주입하는 CLI·test·evaluation 내부 경계라서 Backend에서 호출하지 않는다. Backend는 exact
Agent wheel commit과 package version, `backend` schema scope/version/kind/hash를 함께 고정해야
한다. 내부 V1 helper의 1.0 결과를 V2로 재표시해서도 안 된다.

wheel에는 서로 섞을 수 없는 세 계약 scope가 포함된다.

- `backend`: Backend가 소비하는 V1/V2 review request/result
- `local`: CLI가 로컬 text/image/PDF를 읽기 위한 path-bearing manifest
- `internal`: OCR ledger와 Claude text/vision/crop candidate

로컬 path, OCR 원문과 model candidate를 Backend 계약에 추가해 편의상 전달하지 않는다. 설치된
consumer는 `customs_ai_agent.contracts.get_schema()`와 `schema_sha256()`으로 선택한 JSON
Schema와 exact package resource hash를 읽는다.

Backend의 Spring/Java 또는 FastAPI/Python 선택은 이 아키텍처 범위 밖이며 아직 결정하지 않았다.
선택 기준은 Backend 구현 시점에 별도 ADR로 확정한다. 어떤 스택을 선택해도 형제 repo import나
Agent DB 직접 접근 대신 versioned package/schema boundary를 지켜야 한다.

TASK-0009는 이 AI Agent 저장소만 제품 코드 범위로 한다. Backend·Frontend와 Data Platform
source는 변경하지 않으며, Data Platform은 pinned Evidence HTTP 계약을 통해서만 관찰한다.
`project/_engineering/archive`의 legacy 자료는 `reference-only`이고 이 graph, prompt,
evaluator의 자동 context가 아니다. 재사용하려면 active Investigation/ADR로 승격해 현재
불변식과 다시 대조해야 한다.

## 실패 정책

- 입력 파일·manifest·ledger 오류 메시지는 원문과 경로를 포함하지 않는다.
- llm mode는 명시적 Evidence retriever 없이는 시작하지 않는다.
- OCR provider 미설치·timeout·invalid output은 Claude 호출 전 또는 해당 문서에서 fail-closed한다.
- Claude provider/schema/grounding 실패는 규칙 extractor fallback 없이 safe model error가 된다.
- 인증·rate limit·provider outage처럼 run 전체에 해당하는 오류는 circuit을 열어 후속 과금 호출을
  막는다.
- rerun은 실제 retriever를 Mock으로 바꾸지 않는다.
- evidence 없는 finding과 high/critical finding–evidence 단절은 허용하지 않는다.

## 테스트로 증명해야 하는 것

각 기능의 “작동”은 다음 증거를 분리해 판정한다.

1. Fast: unit, lint, type, schema
2. Contract: Agent V1/V2와 Evidence consumer snapshot
3. OCR provider: 설치 preflight, timeout, invalid output, ledger binding
4. Claude live: normal/adversarial synthetic text/vision gold
5. Data live: 실제 Evidence HTTP와 snapshot/citation
6. System: Agent → Evidence API → 공개 result와 fail-closed 상태

Mock, ledger, Claude-only, RAG-only 결과를 전체 서비스 E2E로 승격해 보고하지 않는다.
사용자 승인 뒤 수행한 exhaustive Claude text normal/adversarial 평가와 공개 AgentRuntime의
Claude 4문서 실행은 통과했다. 그러나 같은 runtime의 actual Evidence HTTP 검색은 Data
readiness 불완전으로 `partial`, 최종 `needs_review`와 `BLOCKED_DATA_READINESS`다. 따라서
Agent의 실제 Claude 실행 성공은 확인했지만 Data live 또는 전체 서비스 Release PASS는 아니다.

pre-freeze offline 테스트 수, pre-commit wheel hash와 smoke report는 구현 이력일 뿐 최종
release 증거가 아니다. exact commit을 정한 뒤 생성한 JUnit, clean wheel install, 60+20 OCR
report/결합 판정과 safety 결과를 `_artifacts/runs/<RUN-ID>/`에 봉인하고 `_engineering`에서
commit과 함께 승인한 기록만 최종 판정의 권위 있는 근거로 사용한다.
