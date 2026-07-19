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
uv run --locked python evals/run_evidence_live.py
```

실제 local Evidence Service가 준비된 경우에만 `--live`를 명시한다.

```powershell
uv run --locked python evals/run_evidence_live.py `
  --live `
  --base-url http://127.0.0.1:8010 `
  --env-file ..\_secrets\.env `
  --out ..\_artifacts\evidence-live-eval.json
```

이 evaluator는 loopback URL만 허용한다. `--live`가 없거나 계약 preflight가 실패하면 readiness,
retrieval, workflow를 호출하지 않는다. Live에서도 최대 한 case와 한 Agent retrieval workflow만
실행한다.

report v2는 서로 다른 실행 단위를 혼동하지 않도록 다음 계수를 분리한다.

- `agent_workflow_count`: `run_review`를 호출한 합성 case 수다. HTTP POST 수가 아니다.
- `readiness_get_count`: Evidence adapter의 `urlopen` 경계에서 관찰한 readiness GET 시도 수다.
- `retrieval_post_count`: 같은 경계에서 관찰한 retrieval POST 시도 수다.
- `other_http_count`: 위 두 endpoint가 아닌 예기치 않은 HTTP 시도 수다.

HTTP 계수의 단위는 `adapter_urlopen_attempt_including_retries`다. 즉 실패 후 retry도 각각
계수한다. `other_http_count`가 0이 아니면 evaluator는
`unexpected_evidence_http_attempt`로 반드시 FAIL하며 정상 업무 결과만으로 이를 상쇄할 수 없다.
readiness가 `ready`인 Live PASS는 현재 retry budget에서 readiness GET `2..4`, retrieval
POST `1..2`를 모두 만족해야 한다. 이 범위를 벗어나면 각각
`evidence_readiness_http_count_invalid`, `evidence_retrieval_http_count_invalid`로 FAIL한다.
따라서 retriever/readiness/workflow를 stub으로 바꿔 업무 state만 정상처럼 만든 `0 GET /
0 POST` 실행은 Live HTTP 증거가 될 수 없다.
`--allow-unready-diagnostic`에서 외부 readiness와 workflow 내부 required retriever의 재검사가
모두 실패하면 `agent_workflow_count=1`, `readiness_get_count>0`이어도
`retrieval_post_count=0`이다. 이는 workflow가 fail-closed 정책까지 실행되었다는 뜻이지,
Evidence retrieval API가 호출되었다는 뜻이 아니다.

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
uv run --locked python evals/run_evidence_live.py `
  --live `
  --allow-unready-diagnostic `
  --diagnostic-review-as-of 2026-07-10 `
  --base-url http://127.0.0.1:8010 `
  --env-file ..\_secrets\.env `
  --out ..\_artifacts\evidence-unready-diagnostic.json
```

이 경로는 한 workflow가 `needs_review`로 안전하게 닫히는지를 확인해도 항상 nonzero로 종료하고
report status를 `BLOCKED`로 유지한다. 2026-07-18 TASK-0009의 **runtime identity 강제 전**
진단에서는 readiness가 실패한 3/58 corpus snapshot에 한 workflow를 실행했고,
partial/unknown coverage와 ambiguous HS를 `needs_review`로 닫아
`fail_closed_policy_passed=true`를 확인했다. 그러나 현재 evaluator는
`contract_identity_mode=required`이고 Data readiness에는 필수 `evidence_contract`가 없다.
따라서 현재 같은 명령은 retrieval POST 전에 차단되며, 과거 workflow 실행을 현재
Agent–Data Live E2E로 재사용할 수 없다. 과거에도 `release_readiness_passed=false`였으므로
Evidence Live/Release PASS가 아니다. 역사 redacted report SHA-256은
`29277a6d8a9181843f811ae68c8a505b7f31fc97efdb5865f7658e73b937bcae`다.

Evidence consumer snapshot의 고정 SHA-256은
`531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a`다. evaluator는 실제 local
schema bytes, lock SHA, 이 기대 SHA가 모두 같은지 API 호출 전에 확인한다. lock의 release
metadata는 local candidate `evidence-api-v1.1.0`,
`7fa46fb487c08c4230312ffe539978f7141fa56f`와 build SHA-256
`97c7c99886f333547b84b1f93584dcc7729ce021f2f365b4714f04d8ec75f0df`다.

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
uv run --locked python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 1 `
  --max-calls 1 `
  --max-total-input-tokens 20000 `
  --max-total-output-tokens 8192

# 4종 text Gold
uv run --locked python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 4 `
  --max-calls 4 `
  --max-total-input-tokens 80000 `
  --max-total-output-tokens 32768

# 실제 Claude + deterministic workflow + 명시적 Mock RAG 진단
uv run --locked python evals/run_claude_live.py `
  --live `
  --env-file ..\_secrets\.env `
  --max-documents 4 `
  --max-calls 4 `
  --max-total-input-tokens 80000 `
  --max-total-output-tokens 32768 `
  --workflow

# adversarial baseline
uv run --locked python evals/run_claude_live.py `
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
EVAL-0011의 과거 c28 실행은 정상 Gold
`7cfe2cf8e2b88e55f80398425e6866d52bfa85e0886ec8a4b5862f736e01c7e5`에서
64/64, adversarial Gold
`b4416cbad6df9fba62f2f636c4267a348bf5f11a1d3b78771aee8b16786db661`에서 21/21
exact와 hallucinated 0을 기록했다. ISO 6346 교정 뒤 정상 Gold
`1e300058f6d01330e82002d04e1a3238f42aaceeb9dc5716653a2aa1c67634e0`과 post-c28
source를 결박한 후속 exact-commit 진단도 normal 64/64, adversarial 21/21,
wrong/missing/hallucinated 0을 확인했다. 이는 합성 text-only 진단이며 release PASS가 아니다.
정확한 실행 SHA와 checksum은 외부 Artifact manifest에서만 판정한다.

같은 과거 공개 `AgentRuntime → Claude → Evidence HTTP`도 Claude 문서 처리 4/4 뒤,
runtime identity 강제 전 Data endpoint에 POST하여 fail-closed 결과를 확인했다. 현재는 Data
readiness에 필수 `evidence_contract` identity가 없고 법령 corpus도 58개 중 3개 loaded라
POST 전에 차단된다. 따라서 성공하는 현재 Agent–Data Live E2E는 없고, 과거 판정도
`BLOCKED_DATA_READINESS`, `release_ready=false`다. evaluator는 SDK retry를 0으로 강제하고,
실패한 main/crop 시도도 `--max-calls` 예산을 소비한다.

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

2026-07-18에 12개 Case/36개 image를 두 AI 시각 reviewer가 검토했고 당시 도구는
`passed_two_reviewer_adjudication`으로 판정했다. 후속 source-pixel 재감사에서 6개
Packing List의 B/L label에 원본 픽셀에 없는 suffix가 확인됐다. 합의한 AI reviewer 두 개는
정답 독립성을 제공하지 못했으므로 해당 Gold 전체를 영구 무효화했다.

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
- 무효화된 Gold:
  `75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6`
- reconciliation report:
  `c9bdd9a8f18a52a44f885e5bcc900308430bec258a85aaa9f68f220d862b9c8b`

위 Gold와 이를 사용한 모든 report는
`adjudicated_gold_invalidated_by_label_error`이며 정확도·진단·release 점수로 재사용할 수
없다. runner와 builder는 exact byte hash뿐 아니라 output 경로, `gold_set_id`, JSON formatting,
reviewer reference를 제외한 source-bound value semantics hash
`ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133`도 차단한다.
독립 사람 검수를 완료한 replacement Gold가 새 exact hash로 발행·승인되기 전에는 canonical
image Gold가 없다.

Vision evaluator의 기존 code-owned prompt pin은 commit `8fe0058`의 1.2.0 hash
`531ad80b62016c5bd65ea292ac584f36413dfb230b13cffb5761b6862d1705ba`다.
commit `c28e72d`에서 runtime prompt를 1.3.0으로 바꾸었고 현재 hash는
`8d64bfa5673f90e092e56740e732ba07012e566fffec154457b59ca35590f3c8`이지만, 별도 평가
승인 없이 evaluator pin을 자동 갱신하지 않는다. 따라서 replacement Gold와 prompt 변경을
각각 승인하기 전까지 `adjudicated_gold_replacement_not_provisioned`와
`prompt_hash_mismatch`가 fail-closed blocker다.

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

현재 고정 Case 0097이 포함된 Gold가 무효화됐고 replacement hash가 설정되지 않았으므로 이
진단기는 OCR·Claude·network 생성 전에
`adjudicated_gold_invalidated_by_label_error` 또는
`adjudicated_gold_replacement_not_provisioned`로 중단한다.

이 진단기는 OCR release Gate를 우회하거나 Paddle/Docling 중 하나를 release provider로
선정하지 않는다. 결과 status는 `PREFLIGHT_BLOCKED`, `BLOCKED_EVALUATION_ASSET`,
`DIAGNOSTIC_EXECUTION_FAILED`, `BLOCKED_DATA_READINESS`, `DIAGNOSTIC_PASS`를 구분하며 항상
`release_ready=false`다. 특히 Gold·source hash 문제를 Data readiness 실패로 기록하지 않는다.
Data readiness 503 이후 실행은
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
uv run --locked python -m evals.run_claude_vision `
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

위 명령은 향후 독립 사람 검수 replacement Gold와 모든 exact pin이 준비된 뒤 사용할
비용 제한 진단 예시다. 현재는 replacement Gold가 없고 과거 12 Case/36문서 Gold가
영구 무효이므로, 문서 수와 관계없이 provider 생성 전에
`run_claude_vision` report의 `status=FAIL`, failure code
`BLOCKED_EVALUATION_ASSET`으로 종료한다. 위의 별도 `run_image_runtime_diagnostic`은 같은
자산 문제를 report `status=BLOCKED_EVALUATION_ASSET`으로 기록한다. 향후
`--max-documents 1` 실행도 모든 metric을 만족하더라도 `status=DIAGNOSTIC_PASS`,
`release_score_eligible=false`다. Release 판정의 분모·coverage 조건은 새 사람 검수 Gold와
sealed holdout protocol에서 별도로 고정해야 하며, 무효화된 과거 12 Case 기준을 재사용하지
않는다.

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
$PaddleEnvironmentLockSha256 = "607459fa093bd01ca1706dfa07222038f05fa190deccc58565ad1d846fc3bdf4"
$DoclingEnvironmentLock = "evals\ocr_environments\v1\docling.requirements.lock.txt"
$DoclingEnvironmentLockSha256 = "1d9aa9b9f2ac28dc2d2bff3e61b33ebd37bbf4c052910f3412f130eb79af9e05"
$ModelAssetsLock = "evals\ocr_environments\v1\model-assets.lock.json"
$ModelAssetsLockSha256 = "acab257b285e1e174d5c039a022cc41d190215fc0e71283dad28470bbf1d7e8d"
$LockedReport = "<LOCKED_REPORT>"
$CoReport = "<CO_REPORT>"
$OcrDescriptor = "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>"
$AuthorizationReceipt = "<EVALUATION_V3_CONSUMPTION_RECEIPT>"
$RepoRoot = "."

uv run --locked python evals/run_ocr_benchmark.py `
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

uv run --locked python evals/run_ocr_benchmark.py `
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

uv run --locked python evals/combine_ocr_release_gate.py `
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

현재 Paddle engine behavior는 `customs-paddle-layout-table-v4`다. text detection은
`PP-OCRv5_mobile_det`의 `text_det_unclip_ratio=0.75`를 고정하고 provider-native bbox를
그대로 사용한다. raster background/foreground 후처리로 text bbox를 다시 만들지 않는다.
실제 table object가 없으면
page pixel과 동일 실행의 OCR token bbox에서 nominal/strict threshold topology, 연속선,
네 모서리, connected component, 최소 `2행×2열`, 행별 column signature와 전 cell grounding을
모두 검증해 generic ruled table을 만든다. layout image 후보가 있으면 복구 table 면적의 `0.5`
초과 겹침을 요구하되 잘린 후보가 전체 page geometry를 축소하지 못한다. 후보가 전혀 없어도
동일한 강한 구조·token 조건은 허용하지만 품목표 의미는 추론하지 않는다. page 복구가 없을
때만 최대 4개 bounded image crop의 ruled 검증과 strict PP table object 검증으로 내려간다.
Gold/template 좌표, document type, header keyword는 런타임 입력이 아니다. empty cell,
모호한 token, threshold topology drift, 구조·좌표 오류와 복수 crop 성공은 fail-closed한다.
table key는 algorithm revision, page, 분석 bbox와 RGB pixel SHA-256을 결박한다. 이 behavior
변경으로 engine ID와 model-assets lock hash가 바뀌었으므로 과거 OCR 수치는 재사용할 수 없고
canonical 60+20을 새 exact wheel에서 다시 실행해야 한다.

Docling 비교 환경은 exact hub ref/snapshot tree를 preflight에서 검증한 뒤
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 강제한
`DocumentConverter:default-image-pipeline-offline-v2`로 실행한다. 부모 환경의 `0` 또는
`false` 설정은 worker가 덮어쓴다. 평가 중 누락 모델을 다운로드하거나 ref를 갱신하지 않으므로
cache가 lock과 다르면 복구 성공이 아니라 provider 실패로 기록한다.

OCR text metric의 순서 계약은 다음처럼 고정한다.

- Gold text: `renderer-bbox-yx/v1`
  - renderer `layout.json` word bbox의 top-left y/x
  - 같은 좌표는 source array index로 tie-break
- Prediction text: OCR ledger span의 명시적 `reading_order`
- bbox/polygon: Gold 시각 순서와 bbox·layout metric에 사용하되 prediction 순서를 바꾸지 않음

worker의 성공 envelope는
`reading_order_policy_id=worker-emitted-sequence/v1`을 emit하고 adapter는 그 exact 값을
검증한다. policy ID와 ordered spans는 ledger hash에 결박된다. 현재 runtime ledger는
`schema_version=2.0`과 packaged `internal/v2/ocr-layout-ledger` 계약을 사용한다.
`internal/v1` schema는 역사 호환·감사용 package resource로만 남기며, policy가 없던 v1
ledger/checkpoint는 현재 runtime provenance 또는 benchmark 입력으로 재사용하지 않는다.

측정 이력은 입력 identity별로 분리한다.

- 과거 pre-policy internal/v1 1문서 smoke: Paddle CER `22.17%`, table row/cell F1 `0/0`
- 현재 rendered-v2에서 관찰한 CER `17.07%`와 `0%`: 과거 v1 smoke와 image hash가 다르므로
  `22.17% → 17.07%/0%` 개선으로 비교 금지
- 선행 고정 1문서 ordering 실험: prediction XY-cut CER `44.08%`, Paddle native source
  order CER `0%`
- 현재 v4 적용 전 CI/PL/B/L/C/O 각 1문서 unsealed ordering 감사:
  - worker/source order vs Gold bbox-yx CER: 문서별 `0%`, micro `0%`
  - renderer array order counterfactual CER: `23.31%`, `19.30%`, `29.58%`, `21.19%`,
    micro `24%`
  - OCR line geometry resort counterfactual: micro CER `36.55%`
  - table row/cell F1: 모든 문서 `0/0`

이 감사는 Gold의 시각 순서와 worker-attested prediction 순서가 선택한 4문서에서 일치하고,
renderer 배열 또는 prediction geometry resort가 의미 순서를 훼손한다는 것을 확인한
non-release 개발 진단이다. durable Artifact Run으로 봉인된 권위 있는 평가가 아니며 OCR text
자체의 전체 corpus 정확도나 Paddle production PASS를 뜻하지 않는다. canonical 60+20이
아니므로 `selection_status=NO_PROVIDER_PASSED`와 선택 provider 없음 상태를 유지한다.

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
uv run --locked python evals/run_rendered_claude_live.py `
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

역사적 evaluation-v2 진단은 임의 JSONL 점수 계산기가 아니다. 먼저 정확히 알려진 generator
오류 bundle에서 원본을 보존하는 one-time repair를 만들고, 다음 네 입력을 함께 제공해야 한다.

```powershell
uv run --locked python -m tools.repair_trade_triplet_container_mutation `
  --source-root "<DATASET_ROOT>/derived/evaluation/v2" `
  --output-root "<DATASET_ROOT>/derived/evaluation/v2-container-repair-v1"

uv run --locked python -m evals.run_deterministic_validation `
  --defects "<DATASET_ROOT>/derived/evaluation/v2-container-repair-v1/injected-defects/cases.jsonl" `
  --manifest "<DATASET_ROOT>/derived/evaluation/v2-container-repair-v1/manifest.json" `
  --splits "<DATASET_ROOT>/derived/evaluation-splits/v2/splits.json" `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json `
  --out "<RUN_ROOT>/deterministic-validation.json"
```

이 historical evaluator는 evaluation-v2 development 20 Case의 한 필드 mutation에서 실제
검출 delta를 issue-code 단위로 점수화한다. manifest schema는
`trade-document-evaluation-assets/v3`, 결과 schema는
`deterministic-validation-evaluation-report/v4`다. container/seal, freight both/neither,
C/O origin/exporter/importer/HS를 포함한 20개 business-field operator 전체를 요구한다.

기존 v1 locked partition과 evaluation-v2의 과거 locked partition은 모두 release 근거에서
제외한다. evaluation-v2는 공개 seed와 일반 tooling으로 membership이 도출되어 영구
`release_holdout_v2_publicly_derivable`이다. evaluator는 split bytes의 canonical hash만
확인하고 JSON을 파싱하지 않으며 결과에 `release_eligible=false`를 기록한다. replacement는
사람이 외부에서 공급하는 sealed evaluation-v3 bundle이어야 하지만 runner가 아직 없어
`release_holdout_v3_sealed_bundle_not_provisioned`로 차단된다.

현재 전체 규칙 커버리지는 별도의 공개 development diagnostic으로 검증한다.

```powershell
uv run --locked python -m tools.build_trade_triplet_rule_coverage `
  --dataset-root "<DATASET_ROOT>" `
  --output-root "<DATASET_ROOT>/derived/evaluation/rule-coverage-v1" `
  --descriptor evals/trade_document_triplets/v1/rule-coverage-v1.descriptor.json `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json

uv run --locked python -m evals.run_rule_coverage_validation `
  --defects "<DATASET_ROOT>/derived/evaluation/rule-coverage-v1/injected-defects/cases.jsonl" `
  --negative-controls "<DATASET_ROOT>/derived/evaluation/rule-coverage-v1/negative-controls/cases.jsonl" `
  --manifest "<DATASET_ROOT>/derived/evaluation/rule-coverage-v1/manifest.json" `
  --selection-descriptor evals/trade_document_triplets/v1/rule-coverage-v1.descriptor.json `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json `
  --out "<RUN_ROOT>/rule-coverage-validation.json"
```

`deterministic-validation-evaluation-report/v5`는 29개 positive operator를 relation-aware finding
단위(`issue_code`, `source_document_id`, `target_document_id`)로 평가하고 PL-only 추가 품목
1건을 negative control로 검사한다. negative control은 baseline 대비 unmatched PL item이 정확히
1개 증가하되 finding 추가·삭제와 review 상태 변화가 모두 0이어야 한다. 총 30개 variant는
공개 등록 합성 데이터와 공개 seed에서 재현되므로 `release_eligible=false`이며 모델 선택 또는
production 성능 근거로 사용할 수 없다.

워크플로 경계는 별도 descriptor/evaluator로 검증한다.

```powershell
uv run --locked python -m evals.run_workflow_boundary_validation `
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
