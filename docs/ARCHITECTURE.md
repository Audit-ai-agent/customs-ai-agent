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
- 문서별 최대 5 page, run 전체 최대 10 page·20 MiB decoded image bytes
- run 전체 text 최대 500,000 characters와 base64/declared-byte 합계 사전 검사

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
schema_version=2.0
ledger_id
document_id
provider_id / engine_id / engine_version
reading_order_policy_id=worker-emitted-sequence/v1
page_number / image_sha256 / width / height
span_id / reading_order / role / text / bbox / confidence
```

OCR worker는 native provider sequence를 성공 envelope에 담고
`reading_order_policy_id=worker-emitted-sequence/v1`을 명시한다. edge adapter는 이 exact
policy와 연속된 span order를 검증할 뿐 geometry로 순서를 재구성하지 않는다. `ledger_id`는
policy ID와 ordered spans를 포함한 canonical ledger 내용의 SHA-256으로 계산한다. Agent는
ledger의 document ID, page 순서, page image hash와 크기를 현재 입력 이미지와 다시 대조한다.

현재 runtime 계약은 `internal/v2/ocr-layout-ledger`, `schema_version=2.0`이다.
`internal/v1/ocr-layout-ledger`는 package 소비자의 역사 호환과 감사를 위해 남긴 immutable
resource이며 current runtime fallback이 아니다. reading-order policy가 없던 v1 ledger 또는
그 ledger를 저장한 checkpoint는 현재 provenance로 재사용할 수 없다.

Claude vision은 업무 필드 후보와 그 후보가 참조하는 OCR span/bbox를 반환한다. 다음 조건 중 하나가
깨지면 후보를 사용하지 않는다.

- structured schema와 field allowlist
- document type별 허용 field
- 참조한 OCR span 존재
- value/source text grounding
- bbox가 원장과 page bounds에 일치
- line item occurrence/reading order의 결정론적 mapping

Claude Sonnet 5 transport는 structured output 요청에 `thinking: disabled`를 명시한다. Sonnet 5는
해당 값을 생략하면 adaptive thinking을 기본 활성화하고 thinking과 JSON이 같은 `max_tokens`
예산을 사용한다. 이 단계는 법률 추론이 아니라 OCR 원장에 대한 제한된 전사·필드 매핑이므로,
불필요한 thinking으로 JSON이 잘리는 것을 방지하고 출력 예산을 재현 가능하게 유지한다.
sampling parameter는 설정하지 않는다.

제한된 crop reread는 이미 검증된 page의 bounded crop에만 사용할 수 있다. crop은 새로운
source-of-truth가 아니며 전체 run budget을 넘기지 않는다. 끝까지 ground할 수 없는 값은
HITL 대상으로 남긴다.

분류 근거가 독립 OCR과 결속되지 않으면 문서 전체를 실패시킨다. 반면 개별 필드나 line-item row의
결속만 실패하면 해당 값 또는 row만 제거하고, 원문 대신 candidate/OCR SHA-256과 OCR span ID를
conflict audit에 남긴다. 이 경우 다른 grounded 값은 보존하지만 run은 반드시 `needs_review`로
전환된다.

`paddle`과 `docling`은 격리된 subprocess provider다. `ledger` provider는 사전에 고정한 관측을
사용하는 재현 평가용이다. ledger PASS는 OCR 엔진 품질 PASS가 아니다.

Paddle engine behavior `customs-paddle-layout-table-v4`는 전체 페이지 결과에 실제 table
object가 없으면 page pixel과 동일 OCR 실행의 token bbox로 generic ruled-grid observation을
만든다. nominal/strict threshold에서 topology가 같아야 하고 연속 수평·수직선, 상하 border
coverage, 네 모서리, 실제 공유 변 connected component, 행별 column signature와 최소
`2행×2열`을 검증한다. 각 OCR token은 면적 `0.7` 초과로 정확히 한 cell에만 귀속돼야 하고
모든 cell이 text/confidence를 가져야 한다. layout image 후보가 있으면 table 면적의 `0.5`
초과 겹침을 corroboration으로 요구하되, 후보가 마지막 행을 자르면 page pixel의 전체 구조를
보존한다. 후보가 전혀 없을 때도 같은 강한 구조·token 조건은 허용하지만 문서 유형이나
line-items 의미는 추론하지 않는다.

page 구조가 승인되지 않으면 유한값·page bounds·최소 크기·면적·중복을 검증한 최대 4개 image
crop에서 같은 ruled 검증을 수행하고, 그 다음에만 fixed PP table pipeline의 실제 object를
엄격하게 검증한다. Gold/template 좌표, header keyword, 문서 유형은 사용하지 않는다. empty
cell, 여러 cell에 걸친 token, 구조 불안정, 복수 crop 성공은 원문 없는 safe error로 닫힌다.
수용한 table key는 algorithm revision, page, 분석 bbox와 RGB pixel SHA-256에 결박된다.
이는 Claude field crop reread와 별개의 OCR provider 내부 경계이며, 새 engine ID와
model-assets lock을 사용하는 만큼 canonical 60+20 benchmark를 다시 통과하기 전에는 release
provider로 승인되지 않는다.

Paddle의 text detection bbox는 실제 glyph보다 넓을 수 있다. v4는
`PP-OCRv5_mobile_det`의 `text_det_unclip_ratio=0.75`를 engine 설정으로 고정하고 그
provider-native bbox를 ledger에 그대로 기록한다. page raster의 corner·edge·foreground 색을
사후 해석하여 좌표를 축소하지 않는다. 이 원칙은 bbox 점수보다 provenance containment를
우선하며, 좁은 glyph나 밝은 글자와 배경색이 겹치는 반례에서 OCR text는 존재하지만 ledger
bbox가 원문 glyph를 제외하는 false grounding을 차단한다. table cell에 token을 귀속할 때도
동일한 provider-native token bbox를 사용한다. 단, generic ruled-grid의 table/row/cell
geometry 자체는 앞 절에 명시한 worker-derived pixel observation이며 detector-native
text bbox라고 표시하지 않는다.

모든 text geometry 좌표는 finite number이고 원본 page bounds 안에 완전히 들어와야 한다.
`NaN`, `Infinity`, 음수, page를 넘는 좌표를 clamp하여 정상 bbox로 바꾸지 않는다. nonblank
provider text에 유효한 bbox가 없으면 해당 span만 조용히 버리지 않고
`ocr_text_geometry_invalid`로 page를 fail-closed한다.

worker 출력 뒤 provider-neutral adapter는 table graph를 다시 검증한다. 각 table key에는
정확히 하나의 table root와 0부터 연속인 row anchor가 있어야 하고, row/cell bbox는 table
bounds 안에 있어야 한다. row index가 증가하면 물리 row band도 위에서 아래로 진행하며
서로 면적으로 겹치지 않아야 한다. logical row/column span은 범위를 벗어나거나 겹칠 수 없고,
rowspan cell은 첫 주장 row의 위 경계부터 마지막 주장 row의 아래 경계까지 실제로 덮어야 한다.
physical cell bbox도 면적 중첩 없이 완전한 직사각형 grid를 구성해야 한다. Docling은 개별
cell provenance가 없을 때 table bbox를 cell bbox로 대체하지 않는다. 이 조건을 만족하지
못한 provider 결과는 ledger ID를 만들기 전에 safe error로 폐기한다. Docling이 table item을
관찰했지만 cell 목록이 없거나 비어 있는 경우도 표를 조용히 삭제하지 않고 같은 safe error로
닫힌다.

OCR benchmark의 text contract도 provenance 계약과 분리하지 않는다. Gold 문자열은
`renderer-bbox-yx/v1`에 따라 renderer word bbox의 top-left y/x와 source-index tie-break로
시각 순서를 만들고, prediction 문자열은 ledger의 worker-attested `reading_order`를 사용한다.
bbox는 Gold 시각 순서와 bbox/layout metric에 권한이 있지만 prediction 순서를 다시 만들지
않는다.

현재 v4 적용 전 CI/PL/B/L/C/O 각 1문서의 unsealed non-release ordering
감사에서 worker/source order와 Gold
bbox-yx order의 CER는 문서별 모두 `0%`였다. renderer array order를 Gold로 쓴
counterfactual은 `23.31%`, `19.30%`, `29.58%`, `21.19%`와 micro `24%`, prediction을
OCR line geometry로 다시 정렬한 counterfactual은 micro `36.55%`였다. 표 row/cell F1은
모든 문서 `0/0`이다. 이는 durable Artifact Run으로 봉인되지 않은 순서 계약 개발 감사이지
전체 OCR production benchmark가 아니다.
과거 pre-policy v1 smoke의 `22.17%`와 별도 rendered-v2에서 관찰한 `17.07%`/`0%` CER도
입력 image hash가 달라 전후 개선 수치로 비교하지 않는다.

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

비용 제한 Live subset은 성공해도 `DIAGNOSTIC_PASS`로만 기록한다. 과거 two-reviewer
12 Case/36문서 Gold는 후속 pixel 감사에서 라벨 오류가 확인돼 exact hash와 semantic
fingerprint 모두 무효화됐다. 따라서 현재 3,000장 계열에는 scoreable Gold가 없고, 독립 사람
검수 replacement의 exact hash가 pin되기 전에는 extraction release `PASS`가 불가능하다.
rendered 계열은 이와 별도로 locked 60+C/O regression 20 전체를 exact ID·count·hash로 평가해야
한다.

wheel은 결합 report의 hash, caller가 명시한 expected hash, OCR Python의 PEP 610 metadata에서
읽은 installed-wheel hash의 3-way 일치와 설치 `RECORD`/console executable 무결성을 요구한다.
dependency lock은 Python 3.11/Windows 환경의 전체 installed distribution inventory와 비교한다.
model-assets lock은 Paddle official-model tree 또는 Docling exact Hugging Face snapshot의
revision/file count/byte count/content-tree hash와 report engine ID를 결박한다.
provider별 dependency lock hash, 공통 model-assets lock hash, provider engine ID는 코드가
소유한 canonical 상수에도 결박되어 임의 lock/report/caller 조합으로 대체할 수 없다.
Docling release preflight는 hub ref와 snapshot tree를 검증하고 isolated worker 환경에서
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 강제한다. 따라서 기본 Docling converter가
`main` 또는 tag를 해석하더라도 검증 뒤 네트워크에서 다른 revision을 가져올 수 없으며, cache가
불완전하면 `default-image-pipeline-offline-v2` 초기화가 실패한다.

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

컨테이너 번호는 B/L과 Packing List가 서로 같은지만 비교하지 않는다. ISO 6346의 lexical
form과 check digit도 결정론적으로 검증하며, 같은 잘못된 번호가 두 문서에 반복돼도
`CONTAINER_NUMBER_FORMAT_INVALID` 또는 `CONTAINER_CHECK_DIGIT_INVALID`로 검토가 필요하다.
반면 컨테이너별 실제 적재 용적, 복합 결제조건, Packing List 추가 품목처럼 업무 계약·작성
의도를 알아야 하는 항목은 임의 임계값으로 위반을 확정하지 않고 `ambiguous` 또는
`needs_review`로 유지한다.

역사적 evaluation-v2 controlled-defect 결정론적 평가는
`trade-document-evaluation-assets/v3` manifest와
`deterministic-validation-evaluation-report/v4` report를 사용한다. evaluator는
dataset-lock, opaque evaluation-v2 provenance bytes, controlled-defect와 manifest의 canonical
hash만 확인한다. split JSON을 파싱하거나 membership/holdout seed를 재계산하지 않는다.
development 20개 Case와 operator assignment는 이미 hash-locked controlled manifest에서만
읽는다. report는 항상 `release_eligible=false`,
`release_holdout_status=release_holdout_v2_publicly_derivable`를 기록한다.

현재 규칙 커버리지 진단은 별도 `trade-document-rule-coverage-assets/v1` manifest와
`deterministic-validation-evaluation-report/v5`를 사용한다. 공개 등록 합성 Case에서 29개
positive operator를 relation-aware finding 단위로 점수화하고, PL-only 추가 품목 1건이
finding을 추가하거나 제거하지 않으면서 unmatched 목록에 정확히 하나만 증가하는지도 검사한다.
release split membership은 읽지 않지만 공개 selector와 seed로 재현 가능하므로 이 결과도
development diagnostic이며 `release_eligible=false`다.

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

모든 HTTP retrieval은 readiness의 `evidence_contract`가 packaged consumer lock의 provider
release, provider revision, provider build SHA-256, schema SHA-256과 모두 같은지 먼저 확인한다. HTTP retriever의
기본 모드는 `required`이므로 서비스 Runtime, CLI와 evaluation runner에 동일하게 적용되며
identity가 없으면 POST를 보내지 않는다. 전환기 `optional`은 명시적인 비릴리스 진단 전용이고,
서비스용 `AgentRuntime`에서는 거절된다. identity가 광고된 경우에는 optional 진단도 똑같이
exact match를 요구한다. 요청 `review_as_of`도 readiness의
`supported_review_as_of`와 같아야 한다. 검증한 네 identity 값은
`X-Evidence-Schema-SHA256`, `X-Evidence-Provider-Release`,
`X-Evidence-Provider-Revision`, `X-Evidence-Provider-Build-SHA256` POST header로 보낸다.
각 response의 네 header는 body byte를 읽기 전에 exact match여야 한다. 누락·GET A/POST B
불일치와 provider HTTP 412는 안전 오류로 종료하며 불일치 body를 사용하지 않는다. POST 응답의
snapshot도 바로 앞 readiness snapshot과 같아야 하며 rollout 불일치는 결과를 폐기한다. 현재
Data candidate는 identity를 광고하고 Agent lock은 exact candidate build를 고정한다. sealed
test-only Provider의 conformance와 달리 실제 3/58 corpus는 readiness 503으로 fail-closed하며,
58/58 적재와 실제 RAG 품질 Gate 전에는 HTTP release-ready로 판정하지 않는다.

현재 provider의 정확한 handoff 조건과 GET–POST build header 계약은
[`DATA_PROVIDER_HANDOFF.md`](DATA_PROVIDER_HANDOFF.md)에 기록한다. frozen v1 response body를
Agent가 독자적으로 확장하거나 Data build identity를 추정하지 않는다.

Agent wheel은 consumer lock과 그 lock의 SHA-256에 정확히 대응하는 provider JSON Schema
snapshot을 함께 포함한다. 모든 HTTP request/response 원문은 Pydantic 변환과 별도로 이
packaged schema를 통과해야 한다. 내부 모델이 provider 제약보다 느슨해도 실제 HTTP 경계에서는
provider 계약 밖의 값을 수락하지 않는다. network JSON decoder는 표준 JSON에 없는
`NaN`·`Infinity`와 의미가 모호한 duplicate object key도 schema/Pydantic 이전에 거부한다.

다음 상태는 Mock으로 fallback하지 않고 자동 완료를 막는다.

- service/readiness/network 실패
- Evidence provider release/revision/build/schema SHA 누락 또는 불일치
- readiness가 지원하지 않는 review_as_of
- readiness와 retrieval response의 knowledge snapshot 불일치
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
Backend V2 request는 필수 `case_id`, 비어 있지 않은 `documents`, 명시적 `llm_mode=llm`만
허용하며 scenario/mock은 포함하지 않는다. Backend/service용 공개 package 실행 경계는
untrusted UTF-8 JSON bytes용 `AgentRuntime.review_json()`과 신뢰된 in-process object용
`AgentRuntime.review()`이며 둘 다 현재 V2 결과만 반환한다. Backend는 HTTP body를 buffering하기
전에 32 MiB 상한을 적용하고, 중복 key 검사를 보존하려면 파싱 전 bytes를 `review_json()`에
전달해야 한다. `review()`의 dict 입력은 exact built-in JSON tree만 허용하고 constructed Pydantic
model이나 custom container를 받지 않는다. packaged JSON Schema를 먼저 적용해 문자열 숫자와
date-time의 Pydantic coercion을 막고, 그 다음 Schema로 표현할 수 없는 합산 text/page/byte
semantic Gate를 적용한다.
`customs_ai_agent.runner`와 호환용 `ReviewRunRequestV2`는 mock scenario와 임의
analyzer/retriever를 주입하는 CLI·test·evaluation 내부 경계라서 Backend에서 호출하지 않는다.
Backend는 exact
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
사용자 승인 뒤 수행한 과거 c28 Claude text normal/adversarial 평가는 당시 정상 Gold
`7cfe2cf8e2b88e55f80398425e6866d52bfa85e0886ec8a4b5862f736e01c7e5`와 adversarial Gold
`b4416cbad6df9fba62f2f636c4267a348bf5f11a1d3b78771aee8b16786db661`에서 각각
64/64와 21/21을 기록했다. ISO 6346 교정 뒤 현재 정상 Gold는
`1e300058f6d01330e82002d04e1a3238f42aaceeb9dc5716653a2aa1c67634e0`이다. 교정된 Gold와
post-c28 source의 exact-commit Claude text 진단도 normal 64/64, adversarial 21/21,
wrong/missing/hallucinated 0을 확인했다. 이 값은 합성 text-only 진단이며 OCR·이미지·Evidence
Release PASS가 아니다. 실행 SHA와 checksum의 권위 있는 원본은 외부 Artifact manifest다.

같은 과거 공개 AgentRuntime은 Claude 4문서 처리를 끝낸 뒤 runtime identity 강제 전 Evidence
endpoint에 POST해 `partial`, 최종 `needs_review`, `BLOCKED_DATA_READINESS`를 기록했다. 현재
Data readiness에는 필수 `evidence_contract` identity가 없고 corpus도 3/58이므로 Agent는
POST 전에 차단한다. 따라서 현재 성공하는 Data Live 또는 전체 서비스 Release E2E는 없다.

과거 c28의 고정 합성 CI/PL/B/L 이미지 triplet 비릴리스 진단도 PaddleOCR와 Claude 호출이
실행됐다는 역사 증거일 뿐이다. 그 실행을 판정한 12-Case image Gold는 후속 pixel 감사에서
무효화됐으므로 conflict 수, extraction 상태, 정확도 또는 성능을 현재 근거로 재사용하지 않는다.
현재 `run_claude_vision`은 report `status=FAIL`, failure code
`BLOCKED_EVALUATION_ASSET`으로, `run_image_runtime_diagnostic`은 report
`status=BLOCKED_EVALUATION_ASSET`으로 OCR·Claude·Data 호출 전에 중단한다. 별도의 1문서
OCR smoke에서도 Paddle과 Docling 모두 기준을 통과하지 못해 선택된 production provider가
없다.

pre-freeze offline 테스트 수, pre-commit wheel hash와 smoke report는 구현 이력일 뿐 최종
release 증거가 아니다. exact commit을 정한 뒤 생성한 JUnit, clean wheel install, 60+20 OCR
report/결합 판정과 safety 결과를 `_artifacts/runs/<RUN-ID>/`에 봉인하고 `_engineering`에서
commit과 함께 승인한 기록만 최종 판정의 권위 있는 근거로 사용한다.
