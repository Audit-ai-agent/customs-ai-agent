# Evaluations

모델·prompt·OCR·RAG 변경을 서로 분리해 검증하는 versioned 평가 영역이다. 실제 거래 문서와
PII가 포함된 원본은 커밋하지 않고 `_artifacts/datasets/`에 보관하며, 저장소에는 비민감 schema,
hash lock, 평가 코드와 허용 가능한 partial Gold만 둔다.

## 결과의 권위와 freeze

이 문서는 평가 계약과 재현 명령을 설명한다. 변동 중인 “최신” 테스트 수나 wheel hash를 정식
결과로 선언하지 않는다.

- 과거 `310 passed`, `321 passed`, pre-commit wheel/clean-install과 1문서·4문서 OCR smoke는
  pre-freeze 진단 이력이며 최종 release evidence가 아니다.
- 최종 판정은 코드 freeze 뒤 exact commit에서 만든 JUnit, wheel/sdist와 clean-install,
  canonical 60+20 OCR report/결합 Gate, safety 결과를 하나의
  `_artifacts/runs/<RUN-ID>/`에 봉인해야 한다.
- commit SHA, 파일 SHA-256, Gate 상태와 미검증 범위를 결박한 `_engineering` Task/Evaluation
  기록이 reader-facing 최종 판정의 권위 있는 인덱스다.
- 이 저장소 안의 고정 hash는 입력 계약을 잠그기 위한 값이다. 실행 결과를 자동으로 PASS로
  만드는 표식이 아니다.

## 평가 층위

| 층위 | 검증 대상 | 검증하지 않는 것 |
|---|---|---|
| Text adapter | Claude text 분류·필드 exact grounding | OCR/VLM, Live RAG |
| OCR provider | page/span/bbox 관측과 provenance | Claude 업무 필드 정확도 |
| Vision adapter | OCR ledger에 결박된 Claude field candidate | Live Data/RAG |
| Deterministic workflow | normalize/pair/validate/policy/route | 운영 Backend |
| Live RAG | Evidence HTTP/snapshot/citation | Claude 정확도 |
| System | Agent → Evidence → public result | Frontend/Backend 완성 |

한 층위의 PASS를 다른 층위의 PASS로 확대 해석하지 않는다.

## Evidence HTTP Live 평가

`run_evidence_live.py`는 Data Platform package나 DB를 직접 사용하지 않고, Agent가 실제
Evidence HTTP API를 소비하는 경계만 검증한다. 합성 `happy_path` 한 case를 readiness가 제공한
`supported_review_as_of`로 한 번 실행한다.

기본 실행은 계약 hash만 확인하고 네트워크를 허용하지 않으므로 의도적으로 FAIL한다.

```powershell
uv run python evals/run_evidence_live.py
```

실제 local Evidence Service가 준비된 경우에만 `--live`를 명시한다.

```powershell
uv run python evals/run_evidence_live.py `
  --live `
  --base-url http://127.0.0.1:8010 `
  --env-file ..\_secrets\.env `
  --out ..\_artifacts\evidence-live-eval.json
```

이 evaluator는 loopback URL만 허용한다. `--live`가 없거나 계약 preflight가 실패하면 readiness,
retrieval, workflow를 호출하지 않는다. Live에서도 최대 한 case와 한 Agent retrieval workflow만
실행한다.

판정 기준:

- status `ok`, coverage `complete`, snapshot 존재, 모든 item result 존재, HS가
  `exact|unique_descendant`, item error 없음이면 complete contract로 분류한다. 그 뒤 최종 상태는
  deterministic outcome 정책과 human-review flag의 일관성을 검증한다.
- partial, failed, ambiguous HS, unknown/partial coverage, snapshot/item 누락, corpus error는
  incomplete로 분류한다. readiness 자체가 통과한 실행에서는 `needs_review`와
  `requires_human_review=true`일 때 fail-closed 정책 검증을 PASS할 수 있다.
- 불완전 검색이 `completed`가 되면 즉시 FAIL한다.

readiness가 실패한 Data snapshot을 조사할 때만 다음 진단 경로를 사용한다.

```powershell
uv run python evals/run_evidence_live.py `
  --live `
  --allow-unready-diagnostic `
  --diagnostic-review-as-of 2026-07-10 `
  --base-url http://127.0.0.1:8010 `
  --env-file ..\_secrets\.env `
  --out ..\_artifacts\evidence-unready-diagnostic.json
```

이 경로는 한 workflow가 `needs_review`로 안전하게 닫히는지를 확인해도 항상 nonzero로 종료하고
report status를 `BLOCKED`로 유지한다. 2026-07-18 TASK-0009 pre-release 진단에서는 readiness가 실패한
3/58 corpus snapshot에 한 workflow를 실행했고, partial/unknown coverage와 ambiguous HS를
`needs_review`로 닫아 `fail_closed_policy_passed=true`를 확인했다. 그러나
`release_readiness_passed=false`이므로 Evidence Live/Release PASS가 아니다. redacted report
SHA-256은
`29277a6d8a9181843f811ae68c8a505b7f31fc97efdb5865f7658e73b937bcae`다.

Evidence consumer snapshot의 고정 SHA-256은
`531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a`다. evaluator는 실제 local
schema bytes, lock SHA, 이 기대 SHA가 모두 같은지 API 호출 전에 확인한다. lock의 release
metadata는 `evidence-api-v1.0.0`과
`a8935007262434523a84e2914271140c5e88489a`다.

Data 저장소의 작업 HEAD는 이 release pin과 다른 개념이다. TASK-0009 작업 시 관찰된 HEAD
`7c3ea37...`은 후속 구현 commit이고 lock release revision을 바꾸는 근거가 아니다. 이 evaluator는
HTTP-only 경계를 지키므로 provider Git HEAD를 읽거나 검증하지 않으며 report에
`provider_repository_head_status=not_observed_http_only`로 구분한다.

결과에는 retrieval status/coverage, readiness date, snapshot ID의 SHA-256, HS resolution별 개수,
safe error code, issue/finding 수, 최종 상태와 계약 hash만 남긴다. request field, evidence text,
source URI, token, `.env` 경로, 절대경로, raw snapshot ID는 기록하지 않는다.

## Text 평가

`claude_text_gold_v1.json`은 합성 `happy_path` 4종 텍스트 문서의 허용 필드 전체를
열거한 exhaustive Gold다. 문서에 실제 존재하고 prompt allowlist에 포함되지만 Gold에
누락된 필드는 환각으로 계산하지 않도록, Gold와 prompt allowlist를 함께 변경하고 hash를
다시 고정해야 한다.
`claude_text_adversarial_v1.json`은 잘못된 declared type, delimiter-like filename/content,
시스템 무시·허위 승인·tool 호출·prompt/key 공개 지시를 포함한 합성 공격 baseline이다.

```powershell
# 1문서 smoke
uv run python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 1 `
  --max-calls 1 `
  --max-total-input-tokens 20000 `
  --max-total-output-tokens 8192

# 4종 text Gold
uv run python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 4 `
  --max-calls 4 `
  --max-total-input-tokens 80000 `
  --max-total-output-tokens 32768

# 실제 Claude + deterministic workflow + 명시적 Mock RAG 진단
uv run python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 4 `
  --max-calls 4 `
  --max-total-input-tokens 80000 `
  --max-total-output-tokens 32768 `
  --workflow

# adversarial baseline
uv run python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --gold evals/claude_text_adversarial_v1.json `
  --max-documents 1 `
  --max-calls 1 `
  --max-total-input-tokens 20000 `
  --max-total-output-tokens 8192
```

출력은 정확도 count와 model/prompt/request/token/latency metadata만 포함한다. key, 원문,
추출값, prompt, raw response는 포함하지 않는다. `--workflow`의 Mock retriever는
`EXPLICIT_MOCK_NOT_LIVE_RAG`이며 Agent–Data 통합 증거가 아니다.

위 명령은 재현 예시다. 사용자가 합성 fixture의 Anthropic 전송과 Claude API 사용을 승인한 뒤
EVAL-0011에서 exhaustive text 정상 64/64, prompt-injection 21/21 exact와 hallucinated 0을
실제로 확인했다. 공개 `AgentRuntime → Claude → Evidence HTTP`도 Claude 문서 처리 4/4와
fail-closed 결과를 확인했지만, Data 법령 corpus readiness가 58개 중 3개 loaded 상태라 전체
판정은 `BLOCKED_DATA_READINESS`, `release_ready=false`다. 따라서 과거
`NOT_EXECUTED_EXTERNAL_TRANSFER_APPROVAL_REQUIRED` 기록은 EVAL-0011로 대체됐다. evaluator는
SDK retry를 0으로 강제하고, 실패한 main/crop 시도도 `--max-calls` 예산을 소비한다.

## Image Gold 격리와 adjudication Gate

`trade_documents_images/v1/manifest_partial_gold.json`은 더 이상 Gold가 아니다. 과거 case
0500 suffix label 오류와 단일 검수 문제 때문에 artifact 전체가
`invalidated_by_label_error`로 격리됐다. `expected_*` 키를 제거했으며 남은 값은
`observed_candidate_*` 감사 기록일 뿐이다. 이 artifact를 사용한 결과는 정확도, 회귀 PASS,
release score로 사용할 수 없다.

Claude Vision evaluator는 다음 네 증거가 모두 일치하기 전에는 provider client도 생성하지
않는다.

1. 고정된 12-triplet `diagnostic-selection-v1.json`
2. Case마다 서로 다른 두 reviewer의 full-triplet worklist와 resolution
3. worklist에서 파생된 `gold_eligible=true` Gate
4. selection/worklist/gate hash와 36개 source image hash에 결박된 별도
   `trade-image-adjudicated-gold/v1` manifest

2026-07-18 기준 12개 Case/36개 image는 서로 다른 두 **독립 AI 시각 reviewer**가 검토했고
별도 canonical 경로에서 `passed_two_reviewer_adjudication`이 됐다. 이것은 사람 검수
완료를 뜻하지 않는다. reviewer independence는 서로 다른 ID·파일·attestation으로 구조적으로
검증했을 뿐 외부에서 독립성을 증명한 것도 아니다.

```text
_artifacts/datasets/trade_documents_images/source-v1/derived/
  independent-review-a/v1/annotations.jsonl
  independent-review-b/v1/annotations.jsonl
  adjudicated-gold/v1/
    adjudications.jsonl
    gate.json
    adjudicated-gold.json
    reconciliation-report.json
```

고정 SHA-256:

- reviewer A: `847dddfe7c7d59147c11997243293eb11b1e6d4093fa432d6c24ec0aa9fbf290`
- reviewer B: `cf1e88964bc69e06085959b09490c491802609a68abb584d8019e7d920ecd38a`
- worklist: `74b9ba353520dc15834353023ef2c96e7399c655b63c75ff6cc097fa97874c70`
- gate: `3c03dc8a592cf70551079606676e218fadbc77d5762c1ac70a627f16a24621b2`
- Gold: `75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6`
- reconciliation report:
  `c9bdd9a8f18a52a44f885e5bcc900308430bec258a85aaa9f68f220d862b9c8b`

과거
`derived/diagnostic-selection/v1/{adjudications.jsonl,gate.json}`은
`pending_first_review`인 역사적 초기 worklist/Gate로 그대로 보존한다. 이 파일을 현재
canonical adjudication으로 사용하거나 hash 상수만 갱신해 case 0500을 다시 허용하면 안 된다.

### 비릴리스 이미지 서비스 진단기

`run_image_runtime_diagnostic.py`는 canonical
`_artifacts/datasets/trade_documents_images/source-v1/raw`의 hash-pinned Case 0097
CI/PL/B/L 3장만 evaluation-only composition에 넣어 OCR/Layout → Claude Vision → Evidence
HTTP를 확인하는 별도 opt-in 진단기다. raw OCR provider를 공개 `AgentRuntime`에 연결하지
않는다. `--live`와 `--nonrelease-ocr-diagnostic`을 동시에 요구하고, OCR provider/Python,
env file, Evidence base URL, review date도 명시해야 한다.

이 진단기는 OCR release Gate를 우회하거나 Paddle/Docling 중 하나를 release provider로
선정하지 않는다. 결과 status도 `DIAGNOSTIC_PASS` 또는 `BLOCKED_DATA_READINESS`뿐이며 항상
`release_ready=false`다. Data readiness 503 이후 실행은
`--allow-unready-diagnostic`에서만 허용하고 불완전 Evidence가 자동 완료되지 않는지만
fail-closed로 확인한다. Report에는 source 값, OCR text, 추출값, 파일 위치, network
destination, credential, raw provider metadata를 기록하지 않는다.

runner의 mock 기반 안전성 테스트와 실제 실행 결과는 서로 다른 증거다. 실제
Paddle/Docling·Claude image 호출 여부와 결과는 저장소 문구가 아니라
`_artifacts/runs/<RUN-ID>`의 redacted report 및 `_engineering` Evaluation에서 판정한다.
어떤 경우에도 이 runner의 코드 존재나 `DIAGNOSTIC_PASS`를 이미지 release runtime PASS로
해석하지 않는다.

다음 예시는 향후 evaluation-v3 sealed runner와 C0/C1 freeze가 구현된 뒤 canonical Gold를
명시적으로 선택할 때의 목표 계약이다. 현재 evaluation-v2 공개 combiner는 입력 접근 전에
중단하므로 유효한 60+20 release Gate report가 없다. 향후 evaluator는 adjudication 증거와
별도로 v3 descriptor/rendered manifest, 결합 Gate report의 exact hash와 `status=PASS`,
selected provider/engine/wheel, provider dependency lock과 model-assets lock을 Claude client
생성 전에 검증해야 한다. report·caller pin·OCR Python installed wheel의 3-way hash,
설치 `RECORD`/executable, 전체 distribution inventory, model revision/tree, runtime ledger
provider/engine/version/model-set/model-assets lock이 모두 일치해야 Claude transport로 진행한다.
이 전제 중 하나라도 틀리면 외부 호출 전에 fail-closed한다.

```powershell
$Dataset = "<TRADE_IMAGE_SOURCE_V1>"
$OcrRenderedRoot = "<RENDERED_DATASET_ROOT>"
$OcrProvider = "<SELECTED_PROVIDER>"
$OcrPython = "<OCR_PROVIDER_PYTHON>"
$OcrReleaseGateReport = "<OCR_RELEASE_GATE_REPORT>"
$OcrReleaseGateSha256 = "<OCR_RELEASE_GATE_REPORT_SHA256>"
$FinalWheelSha256 = "<FINAL_AGENT_WHEEL_SHA256>"
$OcrEnvironmentLock = "<SELECTED_PROVIDER_REQUIREMENTS_LOCK>"
$OcrEnvironmentLockSha256 = "<SELECTED_PROVIDER_REQUIREMENTS_LOCK_SHA256>"
$OcrModelAssetsLock = "evals\ocr_environments\v1\model-assets.lock.json"
$OcrModelAssetsLockSha256 = "<MODEL_ASSETS_LOCK_SHA256>"
uv run python evals/run_claude_vision.py `
  --live `
  --env-file ..\_secrets\.env `
  --dataset-root "$Dataset\raw" `
  --gold "$Dataset\derived\adjudicated-gold\v1\adjudicated-gold.json" `
  --adjudication-worklist "$Dataset\derived\adjudicated-gold\v1\adjudications.jsonl" `
  --adjudication-gate "$Dataset\derived\adjudicated-gold\v1\gate.json" `
  --ocr-provider $OcrProvider `
  --ocr-python $OcrPython `
  --ocr-release-gate-report $OcrReleaseGateReport `
  --ocr-release-gate-report-sha256 $OcrReleaseGateSha256 `
  --ocr-release-descriptor "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>" `
  --ocr-release-rendered-root $OcrRenderedRoot `
  --expected-agent-wheel-sha256 $FinalWheelSha256 `
  --ocr-environment-lock $OcrEnvironmentLock `
  --ocr-environment-lock-sha256 $OcrEnvironmentLockSha256 `
  --ocr-model-assets-lock $OcrModelAssetsLock `
  --ocr-model-assets-lock-sha256 $OcrModelAssetsLockSha256 `
  --max-documents 1 `
  --max-calls 7 `
  --max-total-input-tokens 30000 `
  --max-total-output-tokens 16384
```

위 `--max-documents 1` 실행은 비용 제한 진단이며, 모든 metric을 만족해도
`status=DIAGNOSTIC_PASS`, `release_score_eligible=false`다. 이 evaluator에서 release
`PASS`가 되려면 `--case-id`와 `--document-id`를 사용하지 않고, two-reviewer Gate가 고정한
12개 Case의 36개 문서를 `--max-documents 36`으로 정확히 모두 평가해야 한다. Case나 문서가
하나라도 빠지면 release 점수로 승격되지 않는다.

`--workflow`는 field Gold만으로 outcome을 점수화하지 못하므로 별도 adjudicated Outcome
Gold가 생기기 전까지 항상 차단된다.

한편 `trade_document_triplets`의 320개 PDF/PNG는 source JSON에서 결정론적으로 렌더한 별도
Gold다. 이 rendered Gold Claude 평가는 renderer 기반 field/layout 회귀이고, 3,000장 JPEG의
AI reviewer adjudication 평가가 아니다. 두 결과의 score·분모·hash·scope를 합치지 않는다.

## OCR benchmark 범위와 현재 상태

기존 `evals/ocr_benchmark_v1.json`과 repository-visible `evaluation-v2`는 모두 release
입력으로 무효다. v2는 공개 seed와 일반 도구로 구성원을 사전 도출할 수 있으므로 authorization
receipt나 새 descriptor로 다시 unseen 상태가 될 수 없다. release runner와 v2 OCR descriptor
builder는 어떤 split 파일도 읽기 전에 `release_holdout_v2_publicly_derivable`로 중단한다.
현재 code-owned descriptor hash는 `NOT_FROZEN`이며 선택된 OCR provider도 없다.
공개 OCR release combiner 역시 descriptor·rendered Gold·provider report를 읽기 전에 같은
코드로 중단한다. 구 결합 알고리즘은 `NONRELEASE_DIAGNOSTIC_*` 상태만 생성하는 private
역사 회귀 경로이며 release report가 아니다.

다음 v2 hash는 development-only 진단 provenance다.

- evaluation-v2 split:
  `d7e6b86ef19bbae838a7f5da7b94428637481ab5c15d9599ee3af3e86bda7b64`
- rendered-v2 manifest:
  `e95a3a2f3d2539c8fb92c3988010c8aeb33736269f1bbfc1c5e637962ca25196`

replacement는 외부에서 사람이 공급하는 `evaluation-v3` sealed/unseen bundle이어야 한다.
저장소에는 count와 bundle SHA-256만 공개하며 case ID·membership·seed를 두지 않는다. 공개
manifest schema와 미구현 blocker는 `docs/RELEASE_HOLDOUT_PROTOCOL.md`를 따른다. sealed
runner가 아직 없으므로 v3도 `release_holdout_v3_sealed_bundle_not_provisioned`로 중단한다.

v3 runner가 구현된 뒤에도 descriptor를 만든 C0와 descriptor hash를 pin하는 C1을 같은
commit으로 취급하지 않는다. C0는 clean authorization commit이고 C1은 C0의 single-parent
child이며 `src/customs_ai_agent/ocr_release_pin.py`의 두 literal만 바꿀 수 있다. descriptor의
release context는 `authorization_commit_sha`로 C0를 기록하고 C1 SHA는 기록하지 않는다.
`tools/verify_ocr_release_freeze.py`가 clean C1, parent, exact diff, descriptor와 consumption
receipt를 검증한다. 자세한 절차는 `docs/OCR_RELEASE_FREEZE.md`를 따른다.

두 provider는 동일한 exact-commit Agent wheel이 설치된 격리 환경에서 각각 `locked`와
`co_regression`을 실행해야 한다.

```powershell
$RenderedRoot = "<RENDERED_DATASET_ROOT>"
$PaddlePython = "<PADDLE_ENV_PYTHON>"
$DoclingPython = "<DOCLING_ENV_PYTHON>"
$FinalWheelSha256 = "<FINAL_AGENT_WHEEL_SHA256>"
$PaddleEnvironmentLock = "evals\ocr_environments\v1\paddle.requirements.lock.txt"
$PaddleEnvironmentLockSha256 = "69361cadc005fc027cbdce20c11548f92e53d661d08b2b70fe67f345602f49db"
$DoclingEnvironmentLock = "evals\ocr_environments\v1\docling.requirements.lock.txt"
$DoclingEnvironmentLockSha256 = "9548eb2b4e4a3063ebb3466357734e5d28dfa646ca8224b8c41d339daed4ef5d"
$ModelAssetsLock = "evals\ocr_environments\v1\model-assets.lock.json"
$ModelAssetsLockSha256 = "6d93a74c6087e5ea3d06c9b069f0266b7a7fb714e07d5c18d473dbace3e9afca"
$LockedReport = "<LOCKED_REPORT>"
$CoReport = "<CO_REPORT>"
$OcrDescriptor = "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>"
$AuthorizationReceipt = "<EVALUATION_V3_CONSUMPTION_RECEIPT>"
$RepoRoot = "."

uv run python evals/run_ocr_benchmark.py `
  --repo-root $RepoRoot `
  --authorization-receipt $AuthorizationReceipt `
  --descriptor $OcrDescriptor `
  --rendered-root $RenderedRoot `
  --sample locked `
  --paddle-python $PaddlePython `
  --docling-python $DoclingPython `
  --wheel-sha256 $FinalWheelSha256 `
  --paddle-environment-lock $PaddleEnvironmentLock `
  --paddle-environment-lock-sha256 $PaddleEnvironmentLockSha256 `
  --docling-environment-lock $DoclingEnvironmentLock `
  --docling-environment-lock-sha256 $DoclingEnvironmentLockSha256 `
  --model-assets-lock $ModelAssetsLock `
  --model-assets-lock-sha256 $ModelAssetsLockSha256 `
  --out $LockedReport

uv run python evals/run_ocr_benchmark.py `
  --repo-root $RepoRoot `
  --authorization-receipt $AuthorizationReceipt `
  --descriptor $OcrDescriptor `
  --rendered-root $RenderedRoot `
  --sample co_regression `
  --paddle-python $PaddlePython `
  --docling-python $DoclingPython `
  --wheel-sha256 $FinalWheelSha256 `
  --paddle-environment-lock $PaddleEnvironmentLock `
  --paddle-environment-lock-sha256 $PaddleEnvironmentLockSha256 `
  --docling-environment-lock $DoclingEnvironmentLock `
  --docling-environment-lock-sha256 $DoclingEnvironmentLockSha256 `
  --model-assets-lock $ModelAssetsLock `
  --model-assets-lock-sha256 $ModelAssetsLockSha256 `
  --out $CoReport

uv run python evals/combine_ocr_release_gate.py `
  --descriptor "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>" `
  --rendered-root $RenderedRoot `
  --locked-report $LockedReport `
  --co-regression-report $CoReport `
  --out "<OCR_RELEASE_GATE_REPORT>"
```

Paddle 후보는 Windows CPU 환경에서 native crash 없이 실행 가능한
`PP-DocLayout-S + PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec`와 table recognition을
사용하고, Docling을 같은 Gold와 threshold로 비교한다. 더 큰 server OCR 후보는 합성 smoke에서
Windows native access violation으로 종료돼 release 후보에서 제외했다. 이는 임계치를 낮춘 것이
아니다. 기준은 critical identifier exact 1.0, CER 0.01 이하, table row/cell F1 0.98 이상,
bbox F1@IoU 0.5 0.95 이상, crash 0이다.

결합기는 smoke를 release 분모에서 제외하고, 두 report의 Schema·descriptor·dataset·threshold·
wheel·provider별 dependency lock·공통 model-assets lock·provider engine identity가 모두 같을
때만 판정한다. 같은 provider가 locked와 C/O에서 모두 PASS해야 선택 가능하다. 두 provider의
locked 품질이 완전히 같을 때만 benchmark runner가 identity-bound 상태에서 측정한 locked
end-to-end p95 latency, isolated worker peak memory, distribution+model disk footprint를 이
순서대로 비교한다. 문서별 latency로 p95를 다시 계산하며, 지표 누락·비정상, runtime identity
drift, 세 지표 전체 동률이면 자동 선택하지 않고 fail-closed한다.

구 evaluation-v2 결합 규칙은 두 report뿐 아니라 canonical descriptor와 rendered manifest,
문서 순서·benchmark ID·입력/layout hash, Gold 분모와 TP/FP/FN·문자 edit distance를
재계산하도록 구현됐다. 그러나 v2 membership이 공개 도출 가능하므로 이 로직은 이제 private
non-release diagnostic이며 `ocr-invalidated-v2-combination-diagnostic/v1`,
`NONRELEASE_DIAGNOSTIC_*`, `release_eligible=false`만 생성한다. 공식 `combine()`과 CLI는
descriptor·Gold·report를 읽기 전에 `release_holdout_v2_publicly_derivable`로 중단한다.

향후 evaluation-v3 결합기는 verified `OcrFreezeAttestation`과 sealed runner의 출력만 받고,
canonical descriptor/manifest, exact Agent wheel, selected engine, dependency/model-assets
lock과 입력 report hash를 함께 고정해야 한다. 그 v3 경로가 구현·freeze되기 전에는 Claude
Vision preflight에 전달할 `status=PASS` release report가 존재한다고 기록하지 않는다.

pre-freeze 1문서 smoke는 stable provider process가 실행됐는지 확인한 음성 진단일 뿐이다.
그 한 문서에서 두 provider metric이 threshold에 미달했지만 smoke는 결합기 분모에서 제외되므로,
provider 선택이나 80문서 FAIL/PASS를 대신하지 않는다.

과거 4문서 report
`ocr-benchmark-locked-final-attested.json`은 descriptor SHA-256
`52364b3d4f67dac442a1d61372400f5e63f6f2e5dc415f61477e7045001322a0`과 lightweight Paddle
configuration을 사용한 초기 진단이며 현재 descriptor·provider 선택 증거로 사용할 수 없다.
그 report 자체의 SHA-256은
`812460a25dbd583d823ed4b236f8dc5c92eee82a8234a81440b2c8fd2ca5a40b`다.
현재 60+20문서 재평가가 끝나기 전에는 선택된 OCR provider가 없다. 둘 다 threshold를
통과하지 못하면 Claude-only로 대체하지 않고 image release path를 `BLOCKED`로 남긴다.

OCR benchmark report는 top-level wheel hash만 신뢰하지 않는다. 각 provider record의
`installed_agent_wheel_sha256`, `environment_lock_sha256`, `model_assets_lock_sha256`이 해당
격리 runtime을 고정하고, 결합 Gate가 locked/C/O identity를 다시 확인한다. Claude Vision
evaluator와 제품 CLI는 report exact hash와 caller pin을 검증한 뒤 OCR Python의 PEP 610,
설치 `RECORD`와 console executable, 전체 distribution inventory를 독립 검증한다. model cache도
선택 provider의 locked revision/file count/bytes/content-tree hash와 같아야 한다. source/editable
install, untracked executable, dependency/model drift가 있으면 OCR provider와 Anthropic client를
생성하기 전에 fail-closed한다. 생성한 provider도 attested wrapper를 통과해야 하며 실제 ledger의
provider/engine/version이 결합 Gate의 selected engine과 다르면 Claude에 ledger를 보내지 않는다.

기존 OCR 가상환경은 이 provider별 requirements lock으로 sync한 release runtime이 아니므로 이전
smoke/PASS를 재사용하지 않는다. 두 환경을 lock에서 새로 만들고 exact release-candidate wheel을
설치한 뒤 locked 60과 C/O 20을 모두 재실행해야 한다.

rendered Gold의 Claude extraction 평가도 같은 결합 Gate를 강제한다. 다음 1문서 명령은 비용
제한 진단이며 metric이 모두 통과해도 `DIAGNOSTIC_PASS`, `release_score_eligible=false`다.
결합 Gate PASS나 외부 전송 승인이 아직 존재한다는 뜻도 아니다.

```powershell
$RenderedRoot = "<RENDERED_DATASET_ROOT>"
$OcrReleaseGateReport = "<OCR_RELEASE_GATE_REPORT>"
$OcrReleaseGateSha256 = "<OCR_RELEASE_GATE_REPORT_SHA256>"
$FinalWheelSha256 = "<FINAL_AGENT_WHEEL_SHA256>"
$OcrProvider = "<SELECTED_PROVIDER>"
$OcrPython = "<OCR_PROVIDER_PYTHON>"
$OcrEnvironmentLock = "<SELECTED_PROVIDER_REQUIREMENTS_LOCK>"
$OcrEnvironmentLockSha256 = "<SELECTED_PROVIDER_REQUIREMENTS_LOCK_SHA256>"
$OcrModelAssetsLock = "evals\ocr_environments\v1\model-assets.lock.json"
$OcrModelAssetsLockSha256 = "<MODEL_ASSETS_LOCK_SHA256>"
uv run python evals/run_rendered_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --descriptor "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>" `
  --rendered-root $RenderedRoot `
  --ocr-release-gate-report $OcrReleaseGateReport `
  --ocr-release-gate-report-sha256 $OcrReleaseGateSha256 `
  --expected-agent-wheel-sha256 $FinalWheelSha256 `
  --ocr-environment-lock $OcrEnvironmentLock `
  --ocr-environment-lock-sha256 $OcrEnvironmentLockSha256 `
  --ocr-model-assets-lock $OcrModelAssetsLock `
  --ocr-model-assets-lock-sha256 $OcrModelAssetsLockSha256 `
  --ocr-provider $OcrProvider `
  --ocr-python $OcrPython `
  --max-documents 1 `
  --max-calls 3 `
  --max-total-input-tokens 60000 `
  --max-total-output-tokens 24576 `
  --out "<RENDERED_CLAUDE_REPORT>"
```

rendered extraction release `PASS`는 위 명령에 `--release`를 추가하고
`--max-documents 80`으로 locked 60과 C/O regression 20을 정확히 모두 읽었을 때만 가능하다.
C/O sample이 없거나 hash/ID/count가 다르거나 `--max-documents`가 80이 아니면 provider와 Claude
생성 전에 실패한다. locked 60만 전부 실행해도 release가 아니라 진단이다.

OCR부터 전체 Agent 흐름까지 확인할 때는 provider가 명시된 `customs-ai-agent review`를 사용한다.

- `--ocr-provider paddle|docling`: 결합 report, dependency/model locks와 runtime attestation이
  모두 일치한 실제 OCR/layout provider
- `--ocr-provider ledger`: 같은 image hash에 고정된 관측으로 OCR 이후 회귀

image/PDF에서 `--ocr-provider`를 생략할 수 없으며 실제 provider에는 report/descriptor/rendered
root, wheel pin, dependency lock path/hash, model-assets lock path/hash를 모두 명시한다.
`ledger`에는 이 release 인자를 주지 않는다. ledger 결과와 실제 OCR 결과를 같은 열에 합치지
않는다.

## Trade document triplets

`trade_document_triplets/v1/dataset.lock.json`은 사용자가 제공한 CI/PL/B/L 100세트의 source
bytes와 등록 기대값을 고정한다. 원본과 파생 PDF/PNG/Gold는
`_artifacts/datasets/trade_document_triplets/source-v1/`에만 둔다.

원본 값이 업무상 정합하지 않아 보여도 바로 수정하거나 “의도된 오류”로 간주하지 않는다.
현재 `defect_intent_status=unknown`이므로 작성자 확인 또는 통제 generator 증거가 필요하다.

정답은 분리한다.

- Extraction Gold: 원본에 실제 저장/표시된 문자열
- Validation Gold: 문서 간 일치·불일치에 대한 adjudication
- OCR/layout Gold: render 결과의 word/field/table bbox와 text
- Outcome Gold: Agent가 내야 할 상태와 finding

추출 정답을 업무상 “올바를 것 같은 값”으로 고치면 OCR/추출 평가가 오염된다. 반대로 원문에
그 값이 있다는 사실만으로 정상 업무 case라고 판정해서도 안 된다.

## 불일치 의도 판정

case 상태는 다음 중 하나다.

```text
confirmed_normal
confirmed_intentional_defect
confirmed_generator_error
ambiguous
coverage_gap
not_applicable
```

판정 기준:

1. `confirmed_intentional_defect`는 dataset author 확인 또는 한 field만 변경하고 mutation
   manifest/hash를 남긴 controlled generator 증거가 필요하다.
2. 의도 근거 없이 source, render, Gold, 설명이 충돌하면 `ambiguous`다.
3. parser/OCR/업무 규칙이 아직 지원하지 않는 경우 `coverage_gap`이다.
4. generator invariant 위반이 재현되면 `confirmed_generator_error`다.
5. `confirmed_normal`과 `confirmed_intentional_defect`만 충분한 판정 근거가 있을 때
   validation precision/recall 분모에 포함한다.

따라서 사용자가 “일부러 틀리게 만들었을 수도 있다”고 말한 것은 중요한 가설이지만 Gold 확정
증거는 아니다. 데이터는 보존하고 quarantine/adjudication을 거쳐야 한다.

### Controlled-defect 결정론적 평가

`run_deterministic_validation.py`는 임의 JSONL 점수 계산기가 아니다. 다음 네 입력을 함께
제공해야 한다.

```powershell
uv run python evals/run_deterministic_validation.py `
  --defects "<DATASET_ROOT>/derived/evaluation/v2/injected-defects/cases.jsonl" `
  --manifest "<DATASET_ROOT>/derived/evaluation/v2/manifest.json" `
  --splits "<DATASET_ROOT>/derived/evaluation-splits/v2/splits.json" `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json `
  --out "<RUN_ROOT>/deterministic-validation.json"
```

Generator는 evaluation-v2 development 20 Case에만 한 필드 mutation과 provenance manifest를
생성하고, evaluator는 별도로 실제 검출 delta를 점수화한다. manifest schema는
`trade-document-evaluation-assets/v3`, 결과 schema는
`deterministic-validation-evaluation-report/v4`다. container/seal, freight both/neither,
C/O origin/exporter/importer/HS를 포함한 20개 business-field operator 전체를 요구한다.

기존 v1 locked partition과 evaluation-v2의 과거 locked partition은 모두 release 근거에서
제외한다. evaluation-v2는 공개 seed와 일반 tooling으로 membership이 도출되어 영구
`release_holdout_v2_publicly_derivable`이다. evaluator는 split bytes의 canonical hash만
확인하고 JSON을 파싱하지 않으며 결과에 `release_eligible=false`를 기록한다. replacement는
사람이 외부에서 공급하는 sealed evaluation-v3 bundle이어야 하지만 runner가 아직 없어
`release_holdout_v3_sealed_bundle_not_provisioned`로 차단된다.

워크플로 경계는 별도 descriptor/evaluator로 검증한다.

```powershell
uv run python evals/run_workflow_boundary_validation.py `
  --descriptor evals/trade_document_triplets/v1/workflow-boundary.descriptor.json `
  --out "<RUN_ROOT>/workflow-boundary-validation.json"
```

required document missing, duplicate type, OCR conflict, Evidence partial/failed/ambiguous 중
하나라도 descriptor에서 누락되거나 자동 `completed`가 발생하면 Gate는 실패한다.

## 필수 metric 분리

최소 보고 단위:

- document type accuracy
- field exact/wrong/missing/hallucinated
- line item exact count
- source span/bbox grounding completeness
- OCR character/word error와 bbox metric
- mismatch detection precision/recall by adjudicated case class
- workflow status와 HITL rate
- Evidence Hit@5, Recall@10, nDCG@10, citation completeness
- model/OCR provider error, token, latency

normal 정확도, intentional anomaly 탐지율, adversarial 방어율, ambiguous/quarantine 수를 각각
표시한다.

## 실행 증거와 공개 제한

각 evaluator는 API/network 전에 dataset, Gold, prompt, schema hash를 확인한다. 결과 artifact에는
다음만 허용한다.

- commit/model/prompt/OCR engine/schema/dataset version과 hash
- count/rate/latency/token
- safe error code와 문서 ID
- 평가 scope와 미검증 범위

원문, image bytes/base64, OCR text, field value, prompt/raw response, key/token, 절대경로는
기록하지 않는다. 포트폴리오 공개본은 이 redacted 결과에서 다시 파생한다.

TASK-0009 평가 구현은 AI Agent 저장소에 한정된다. Backend·Frontend와 Data Platform source를
변경하지 않으며 Data는 pinned Evidence HTTP API로만 검증한다. `project/_engineering/archive`의
legacy 자료는 평가기의 자동 입력이 아닌 `reference-only`이고, 사용하려면 active
Investigation/ADR로 승격해 현재 hash·schema·Gate에 맞춰 재검증해야 한다. Backend 기술 스택은
이 평가 범위에서 결정하지 않는다.
