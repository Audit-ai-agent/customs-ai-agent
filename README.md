# Customs AI Agent

통관 문서 묶음을 분석하고, 문서 간 불일치와 규정 근거를 결합해 사람이 검토할 수 있는 XAI
결과를 만드는 **Evidence-bound AI Agent**다. 현재 작업 목표는 `0.3.0` AI Agent
release candidate이며, 제품 Frontend와 Backend 또는 전체 서비스 release가 아니다.

이 버전은 다음 두 입력 경로를 지원한다.

- V1: 기존 UTF-8 텍스트 문서
- V2: 검증된 JPEG/PNG/WebP `image_pages` 또는 명시적 `pdf_pages`

Claude는 문서 유형과 필드 후보를 제안한다. OCR/layout 원장, 정규화, 계산, 교차문서 검증,
Evidence API, Evidence Gate, severity, 최종 상태는 코드가 통제한다. 모델이 법률·Evidence ID·
severity·최종 승인 상태를 자유 생성하는 구조가 아니다.

## 구현된 코드 경계와 검증 상태

다음은 구현된 Agent 코드 경계다. 이 목록은 OCR·Claude·Evidence·Release Gate가 모두
PASS했다는 뜻이 아니다.

- LangGraph 기반 분류 → 추출 → 정규화 → pairing → 검증 → 검색 → compliance → report 흐름
- CI/PL/B/L/C/O 교차검증과 ISO 6346 컨테이너 형식·check digit의 결정론적 fail-closed 검증
- Claude text/vision structured output adapter
- 이미지 hash·byte·dimension·page 수·provenance 검증
- 독립 OCR/layout 원장과 Claude field candidate의 span/bbox grounding
- worker가 소유한 `worker-emitted-sequence/v1` reading order와 이를 hash-bound하는
  internal OCR ledger v2
- PDF를 제한된 page 수와 DPI로 rasterize하는 로컬 adapter
- Paddle/Docling 격리 실행 provider 경계와 고정 ledger 평가 provider
- invalidated v2 결합 경로의 non-release 격리와 향후 v3 wheel 3-way
  attestation·runtime ledger engine pre-call lock
- Data Platform의 Evidence Service를 소비하는 HTTP-only retriever
- 실패·부분 검색·모호성·근거 부족을 자동 완료하지 않는 Evidence Gate
- Agent V1/V2 request/result JSON Schema
- 원문·base64·로컬 경로를 제외한 공개 결과

TASK-0009의 pre-freeze 상태는 다음과 같다.

- 과거 `310 passed`와 이후 `321 passed` offline 실행은 구현 중간 회귀를 찾는 데 사용한
  역사적 스냅샷이다. 최종 exact commit의 검증 수치나 release 증거가 아니다.
- pre-commit wheel, clean-install 결과, 1문서 또는 과거 4문서 OCR smoke도 모두
  진단 이력이다. 최종 wheel이나 OCR provider 선택 근거로 사용할 수 없다.
- 과거 12 Case/36 image field Gold는 두 AI reviewer의 adjudication을 거쳤지만, 이후 6개
  Packing List B/L 라벨에 실제 pixel에는 없는 `EE`가 추가된 오류가 확인됐다. exact
  SHA-256 `75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6`과
  semantic fingerprint
  `ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133`은 모두
  `invalidated_by_label_error`이며 점수 또는 release 근거로 사용할 수 없다.
- 현재 canonical/scoreable image Gold는 없다. 독립 사람 검수로 만든 replacement Gold의 exact
  hash가 코드에 명시적으로 pin되기 전에는 evaluator가 evaluation-asset preflight에서
  OCR·Claude·Evidence 호출 전에 차단된다.
- controlled-defect 평가는 고정 manifest/report V3 경계에서 별도 sub-Gate로 검증한다. 이
  결과만으로 OCR·Claude·Evidence 또는 release 전체가 PASS한 것은 아니다.
- 2026-07-18의 과거 Claude text Live는 당시 정상 Gold SHA-256
  `7cfe2cf8e2b88e55f80398425e6866d52bfa85e0886ec8a4b5862f736e01c7e5`에서
  64/64, adversarial Gold SHA-256
  `b4416cbad6df9fba62f2f636c4267a348bf5f11a1d3b78771aee8b16786db661`에서 21/21
  exact match와 wrong/missing/hallucinated 0을 기록했다. 이후 ISO 6346에 맞는 컨테이너
  번호로 정상 Gold를 수정해 현재 SHA-256은
  `1e300058f6d01330e82002d04e1a3238f42aaceeb9dc5716653a2aa1c67634e0`이다. 현재 Gold로
  별도 exact-commit Claude text 진단을 다시 수행해 normal 64/64, adversarial 21/21과
  wrong/missing/hallucinated 0을 확인했다. 이 결과는 합성 text-only 진단이며 OCR·이미지·
  Evidence Release PASS가 아니다. 실행 SHA와 보고서 checksum은 source 문서에 자기참조로
  고정하지 않고 외부 Harness의 Evaluation·Artifact manifest를 권위 있는 근거로 사용한다.
- 과거 고정 합성 CI/PL/B/L 이미지 진단의 실행 로그는 구현 진단 이력으로만 보존한다. 그 결과를
  판정한 image Gold가 위 라벨 오류로 무효화됐으므로 정확도·성능·release 증거로 재사용할 수
  없다. 현재 evaluator는 legacy Gold 무효화, replacement Gold 부재, release lock 불일치를
  오류 코드로 남기고 provider 호출 전에 fail-closed한다. 구체적인 실행 상태와 오류 코드 역시
  외부 Artifact manifest를 따른다.
- 별도의 과거 공개 runtime Evidence HTTP 진단은 runtime identity 강제 전 Data corpus
  readiness 실패 상태에서 POST를 실행해 `partial`, 최종 `needs_review`,
  `BLOCKED_DATA_READINESS`를 기록했다. 현재 Agent는 readiness의 필수
  `evidence_contract` identity를 요구하지만 Data Provider가 이를 제공하지 않고 corpus도
  58개 중 3개만 적재돼 있다. 따라서 현재 Evidence POST는 0건이고 성공하는
  Agent–Data Live E2E는 없다. 과거 진단은 현재 image evaluation PASS 또는 Evidence Release
  PASS가 아니다.
- OCR provider는 exact-commit wheel로 canonical locked 60 CI/PL/B/L와 C/O 20을 모두
  검증하기 전까지 선택하지 않는다.
- OCR release pin은 descriptor SHA와 C0 authorization commit SHA 모두 `None`이다. 향후
  evaluation-v3가 준비돼도 C0 승인과 C1 literal-only pin 검증을 통과하기 전에는
  `NOT_FROZEN` 상태를 유지한다.
- 이 Task는 Backend·Frontend와 Data Platform source를 변경하지 않는다. Data는 pinned
  Evidence HTTP API의 소비자 경계에서만 관찰한다.

과거 c28 exact commit의 611-test·package·Live 결과 뒤에도 Gold 무효화, Evidence identity,
HS/date/ISO 6346과 OCR ordering 보완이 추가됐다. 보완된 후보는 exact-commit Fast·Contract·
local secret scan, JUnit, Claude text 진단, wheel/sdist build와 fresh install로 다시 검증한다.
변동 가능한 test 수, commit SHA, 배포물 hash와 Gate 결과는 이 source 문서에 복제하지 않고
`_artifacts/runs/<RUN-ID>` 및 개인 Harness Evaluation에서만 확정한다. OCR benchmark의 Gold text order는
`renderer-bbox-yx/v1`, prediction order는 worker-attested
`worker-emitted-sequence/v1`이다. 과거 pre-policy internal/v1 1문서 smoke의 Paddle CER
`22.17%`와 별도 rendered-v2에서 관찰한 `17.07%`/`0%` CER은 입력 image hash가 달라
전후 성능 비교에 사용할 수 없다.

Paddle engine behavior `customs-paddle-layout-table-v4`는 provider의 실제 table object가
없을 때 page pixel과 같은 실행에서 나온 OCR token으로 generic ruled table을 복구한다.
Otsu 기준의 서로 다른 두 threshold·line coverage에서 topology가 같고, 연속 수평·수직선,
네 모서리, 실제 변을 공유하는 connected component, 행별 동일 column signature와 최소
`2행×2열`을 모두 만족해야 한다. 각 token은 bbox 면적의 `0.7`을 초과해 정확히 한 cell에만
귀속되고 모든 cell에 text/confidence 근거가 있어야 한다. Paddle layout 후보가 있으면 복구한
table 면적의 절반을 초과해 겹쳐야 하며, 후보가 표의 끝을 잘라도 page geometry가 전체 경계를
소유한다. 후보가 전혀 없는 경우에도 위의 강한 구조·token 조건은 사용할 수 있지만 결과는
품목표라는 의미를 부여하지 않은 generic table observation이다. page 구조가 승인되지 않으면
bounded image crop에서 같은 ruled 검증 후 strict PP table object 검증으로 내려간다.
Gold/template 좌표, 문서 유형, header keyword는 런타임 입력이 아니다. empty cell, 모호한
token, 불안정 topology, 구조·좌표 오류는 fail-closed한다.

Paddle text bbox는 `PP-OCRv5_mobile_det`의 provider-native detection 결과를 그대로 사용한다.
text detection polygon의 확장 비율인 `text_det_unclip_ratio=0.75`는 engine behavior에
고정하지만, page raster의 색·corner·edge를 다시 분할해 좌표를 사후 생성하거나 축소하지 않는다.
따라서 일반 text span bbox의 출처는 detector output 하나이며, 표 선·좁은 glyph·색상
foreground를 text 후처리가 잘못 지워 false grounding을 만드는 경로가 없다. 앞 문단의
generic ruled-grid table/row/cell geometry는 이 text bbox와 달리 worker가 page pixel에서
파생하며 algorithm revision·page·crop hash에 결박한다. table token grounding에는 같은
provider-native text token bbox를 사용한다. provider-neutral adapter는 table당 하나의 root,
연속 row anchor, 범위 안의 row/cell, 물리 row 순서·비중첩, 논리 span 비중첩, cell이 주장한
모든 row band의 실제 수직 범위, 물리 bbox 비중첩과 완전한 직사각형 cell graph를 다시
검증한다. Docling cell은 cell-level bbox가 없으면 table 전체 bbox를 대신 복제하지 않고
fail-closed한다. Docling worker는 preflight에서 hash 검증한 Hugging Face cache만 사용하도록
`HF_HUB_OFFLINE=1`과 `TRANSFORMERS_OFFLINE=1`을 강제하며 부모 프로세스의 완화된 값을
덮어쓴다. 이 정책은 `default-image-pipeline-offline-v2` engine ID에 결박된다. engine ID와
model-assets lock이 변경됐으므로 이 경로는 canonical 60+20 benchmark를 새 exact wheel에서
다시 통과하기 전까지 release provider로 승인되지 않는다.

현재 v4 적용 전 CI/PL/B/L/C/O 각 1문서에 대한 unsealed non-release
ordering 감사에서는 worker/source order와 Gold
bbox-yx order의 CER가 문서별 모두 `0%`였다. 반면 renderer 배열 순서를 Gold로 사용한
counterfactual CER는 각각 `23.31%`, `19.30%`, `29.58%`, `21.19%`, micro `24%`였고,
prediction을 OCR line geometry로 다시 정렬하면 micro CER `36.55%`였다. 이는 ordering 계약을
선택한 근거이지 OCR 인식 품질의 production benchmark가 아니다. table row/cell F1은 여전히
`0/0`이다. 이 4문서 관찰은 durable Artifact Run으로 봉인되지 않은 개발 진단이며,
production 기준을 통과한 provider는 없다. evaluation-v3 60+20 OCR 두 report와
결합 판정도 pending이다. 최종 값은
`_artifacts/runs/<RUN-ID>/`에 봉인하고 `_engineering`의 Task/Evaluation 기록에서 commit과 함께
승인한다. 이 README의 과거 숫자나 이름에 `latest` 의미를 부여하지 않는다.

따라서 현재 상태를 전체 서비스, production-ready 또는 최종 release로 표시하지 않는다.
OCR 기준을 통과한 provider가 없으면 Claude-only 방식으로 대체하지 않고 image release path를
`BLOCKED`로 유지한다.

아직 전체 서비스 완료를 뜻하지 않는 항목:

- 사용자 인증·권한, Case/Job/HITL 영속화, 업로드 저장소
- 제품 Backend API와 Frontend
- 운영 개인정보 보존·삭제·감사 정책
- 운영 배포, tenant isolation, rate limit, 관제
- 세관 신고의 자동 승인 또는 법적 판단

Backend 기술 스택은 의도적으로 확정하지 않았다. Spring/Java와 FastAPI/Python의 선택은 Backend
구현을 시작할 때 인증, 장기 운영 인력, 배포 환경, Agent 호출 방식, 트랜잭션 요구사항을 비교해
별도 ADR로 결정한다. 현재 Agent는 설치 가능한 Python wheel, V1/V2 JSON Schema, CLI라는
기술 중립 경계를 제공하므로 그 결정을 선행할 필요가 없다.

`project/_engineering/archive`의 legacy 문서와 플랜은 `reference-only`다. runtime, evaluator,
prompt 또는 Agent 자동 context로 읽지 않는다. 내용을 다시 쓰려면 active Investigation/ADR로
승격하고 현재 코드·계약·평가 기준으로 재검증해야 한다.

## 저장소 경계

| 소유자 | 책임 | Agent와 연결 |
|---|---|---|
| AI Agent | 문서 이해, workflow, 정책, 평가, Agent 계약 | 이 저장소 |
| Data Platform | 법령·HSK·CCCT 원장, RAG, Evidence API | HTTP `POST /internal/rag/v1/retrieve`만 사용 |
| Backend | 인증, Case, Job, HITL, Operational DB | 추후 Agent wheel/계약 버전을 고정해 호출 |
| Frontend | 사용자 작업대 | 추후 Backend API만 호출 |

Agent는 Data 저장소의 Python 코드나 DB를 import/조회하지 않는다. `contracts/evidence/v1/`은 Data
provider 원본을 hash로 고정한 consumer snapshot일 뿐 canonical 계약이 아니다. 반대로
`contracts/agent/v1|v2/`는 이 저장소가 소유하며 Backend가 소비한다.

## 실행 모드

| 모드 | 문서 분석 | 규정 검색 | 판정 용도 |
|---|---|---|---|
| Mock | fixture/규칙 | deterministic Mock | 빠른 단위·회귀 테스트 |
| Live RAG | fixture/규칙 | 실제 Evidence HTTP | Agent–Data 계약/통합 |
| Claude text | Claude | 명시적 Evidence HTTP | 실제 텍스트 Agent 평가 |
| Claude vision | OCR/layout + Claude vision | 명시적 Evidence HTTP | 실제 이미지/PDF Agent 평가 |
| Ledger vision | 고정 OCR ledger + Claude vision | 명시적 Evidence HTTP | 재현 가능한 OCR 이후 Agent 평가 |

`llm_mode=llm`은 실제 Evidence retriever가 없으면 실행 전에 실패한다. Claude를 Mock RAG와
결합한 별도 evaluator는 진단용일 뿐 Live 통합 PASS로 표시하지 않는다. provider·schema·grounding
실패도 규칙 extractor로 fallback하지 않는다.

현재 Claude 성공 결과도 기본적으로 `needs_review`다. exact quote/span grounding은 문자열과
좌표 근거를 증명하지만, 업무 필드 연결 정확도가 운영 임계치로 보정됐다는 뜻은 아니기 때문이다.
사람 승인 계약과 충분한 평가 근거가 생기기 전에는 Claude 결과만으로 자동 승인·자동 차단하지 않는다.

## 개발 환경

Python 3.11과 `uv` lockfile을 기준으로 한다.

```powershell
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest -q
uv build --wheel
```

기존 텍스트 회귀 CLI:

```powershell
uv run --locked customs-ai-agent list-scenarios
uv run --locked customs-ai-agent run --scenario happy_path
uv run --locked customs-ai-agent run --scenario happy_path `
  --rag-url http://127.0.0.1:8010 --review-as-of 2026-07-10
uv run --locked customs-ai-agent run --scenario happy_path --llm-mode llm `
  --env-file ..\_secrets\.env --rag-url http://127.0.0.1:8010
```

`run`은 V1 scenario/text 호환 명령이다. 실제 로컬 파일은 새 `review` 명령을 사용한다.

## Python application API

Backend의 untrusted serialized JSON 경계는 `AgentRuntime.review_json(raw_bytes)`다. 이 API는
32 MiB 상한, strict UTF-8 JSON, 중복 key, NaN/Infinity, 구조 깊이를 nested model/PIL 처리 전에
검사한다. Backend transport도 body를 메모리에 전부 올리기 전에 같은 32 MiB stream/body 상한을
적용하고, 중복 key 정보가 사라지지 않도록 원본 bytes를 이 API에 전달해야 한다.

`AgentRuntime.review()`는 이미 신뢰 경계 안에 들어온 `ServiceReviewRunRequestV2` 또는 plain
JSON built-in tree를 위한 in-process API다. custom container와 nested constructed Pydantic
model은 거절하고 전체 요청을 다시 검증한다. 두 API 모두 scenario/mock을 통과시키지 않고
Claude와 Evidence HTTP 경로만 사용한다.

```python
from customs_ai_agent import AgentRuntime, AgentRuntimeError, RuntimeConfig

evidence_url = "http://127.0.0.1:8010"
try:
    with AgentRuntime(
        RuntimeConfig.from_environment(
            evidence_base_url=evidence_url,
            evidence_allowed_origins=frozenset({evidence_url}),
            env_file="../_secrets/.env",
        )
    ) as runtime:
        result = runtime.review_json(
            b"""{
                "case_id": "case-001",
                "documents": [{
                    "document_id": "invoice-001",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-001"
                }],
                "llm_mode": "llm",
                "review_as_of": "2026-07-10"
            }"""
        )
except AgentRuntimeError as exc:
    safe_error_code = exc.code
else:
    public_result = result.model_dump(mode="json")
```

`RuntimeInputError`, `RuntimeConfigurationError`, `RuntimeExecutionError`는 모두
`AgentRuntimeError`이며 외부 응답에는 안전한 `code`만 사용한다. 현재 Data readiness에
필수 contract identity가 없으므로 위 호출은 Data가 그 계약을 제공할 때까지 fail-closed한다.
image/PDF 호출은 추가로 canonical OCR release attestation을 통과한 provider가 필요하며,
현재 OCR release pin이 비어 있으므로 성공 예제로 표시하지 않는다.

장기 실행 Backend는 `AgentRuntime`을 context manager로 사용하거나 종료 시 `close()`를
호출한다. Runtime은 자신이 생성한 Anthropic client만 닫는다. `RuntimeConfig`로 주입한 OCR
provider는 호출자 소유이므로 Runtime이 닫지 않으며 Backend가 별도 lifecycle을 관리한다.

required Evidence 경로는 readiness의 schema/release/revision/build SHA-256을 네
`X-Evidence-*` POST header로 전달하고 response의 같은 네 header를 body보다 먼저 exact match로
검증한다. 현재 lock의 build SHA-256은 TASK-0010 transport 회귀용 임시 값이며 Data 최종
artifact digest가 발행되면 교체해야 한다.

Data의 contract/header 구현만 확인하는 서버를 사용할 때는 runtime Live evaluator에
`--provider-conformance`를 명시한다. 이 모드의 `status=PASS`는
`scope=PROVIDER_CONFORMANCE_NOT_RAG_QUALITY`,
`implementation_candidate_ready=true`, `release_ready=false`로 기록한다. 합성 conformance
provider의 PASS를 실제 법령 corpus·RAG 품질·서비스 release PASS로 승격하지 않는다.

## 로컬 text/image/PDF review

입력 파일을 CLI 인자에 반복해서 노출하지 않고 한 manifest로 결박한다.

```json
{
  "schema_version": "customs-agent-local-review-manifest/v1",
  "contract_version": "v2",
  "llm_mode": "llm",
  "case_id": "case-eval-001",
  "review_run_id": "run-eval-001",
  "review_as_of": "2026-07-10",
  "trade_type": "import",
  "jurisdiction": "KR",
  "documents": [
    {
      "document_id": "invoice-1",
      "filename": "invoice.png",
      "declared_type": "invoice",
      "source": {
        "kind": "image",
        "path": "inputs/invoice.png"
      }
    },
    {
      "document_id": "packing-list-1",
      "filename": "packing-list.pdf",
      "source": {
        "kind": "pdf",
        "path": "inputs/packing-list.pdf"
      }
    },
    {
      "document_id": "bl-1",
      "filename": "bill-of-lading.txt",
      "source": {
        "kind": "text",
        "path": "inputs/bill-of-lading.txt"
      }
    }
  ]
}
```

모든 `source.path`는 manifest 디렉터리 기준 상대경로다. 절대경로, `..` traversal, symlink,
지원하지 않는 image/PDF, 한도 초과 파일은 입력 단계에서 거부한다. 파일 경로는
`ReviewRunRequest`에 들어가지 않는다.

Paddle OCR/layout 실행 형태:

```powershell
$OcrReleaseGateReport = "<OCR_RELEASE_GATE_REPORT>"
$OcrReleaseGateSha256 = "<OCR_RELEASE_GATE_REPORT_SHA256>"
$FinalWheelSha256 = "<FINAL_AGENT_WHEEL_SHA256>"
$RenderedRoot = "<CANONICAL_RENDERED_ROOT>"
$OcrEnvironmentLock = "evals\ocr_environments\v1\paddle.requirements.lock.txt"
$OcrEnvironmentLockSha256 = "<PADDLE_ENVIRONMENT_LOCK_SHA256>"
$OcrModelAssetsLock = "evals\ocr_environments\v1\model-assets.lock.json"
$OcrModelAssetsLockSha256 = "<MODEL_ASSETS_LOCK_SHA256>"
uv run --locked customs-ai-agent review `
  --manifest .\local-eval\review-manifest.json `
  --out ..\_artifacts\local-review-v2.json `
  --env-file ..\_secrets\.env `
  --rag-url http://127.0.0.1:8010 `
  --ocr-provider paddle `
  --ocr-python <OCR_PROVIDER_PYTHON> `
  --ocr-release-gate-report $OcrReleaseGateReport `
  --ocr-release-gate-report-sha256 $OcrReleaseGateSha256 `
  --ocr-release-descriptor "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>" `
  --ocr-release-rendered-root $RenderedRoot `
  --expected-agent-wheel-sha256 $FinalWheelSha256 `
  --ocr-environment-lock $OcrEnvironmentLock `
  --ocr-environment-lock-sha256 $OcrEnvironmentLockSha256 `
  --ocr-model-assets-lock $OcrModelAssetsLock `
  --ocr-model-assets-lock-sha256 $OcrModelAssetsLockSha256
```

Docling provider를 검증할 때는 `--ocr-provider docling`과
`docling.requirements.lock.txt`를 사용한다. release preflight는 exact Hugging Face
revision/tree와 cache inventory를 먼저 검증하고 isolated worker는 hub와 Transformers를
항상 offline으로 실행한다. 필요한 snapshot이 없거나 ref/tree가 lock과 다르면 다운로드로
복구하지 않고 실패한다. OCR 의존성은 Agent 프로세스에 섞지 않고 별도 Python
환경에서 실행한다. image/PDF 입력에는 암묵적 provider가 없으며, 실제 provider에서는 위 release
인자를 하나라도 생략하면 입력 파일 처리·OCR provider 생성·Claude 설정 확인 전에 실패한다.
위 예시는 실행 형태이며 Paddle을 기본 provider로 채택했다는 뜻이 아니다. 실제 사용은
향후 sealed evaluation-v3 OCR benchmark를 통과한 provider의 exact-commit wheel/environment에
한정한다. repository-visible evaluation-v2는 공개 seed로 구성원이 도출되어 영구 release
부적격이다. 현재 v3 sealed runner도 구현되지 않았으므로 이 명령은 release 용도로 실행할 수
없다. 공개 v2 combiner도 입력 접근 전에 영구 무효 코드로 중단한다. `review`는 새 PASS를
만드는 evaluator가 아니라 향후 v3 경계에서 승인된 결합 report만 소비한다.
두 provider의 locked 품질이 완전히 같을 때는 임의 기본값을 사용하지 않는다. 동일 benchmark
runner가 기록한 p95 latency, isolated worker peak memory, distribution+model disk footprint를
순서대로 비교하며 지표 누락·비정상·identity drift·전체 동률은 자동 선택 없이 차단한다.
따라서 report·caller pin·설치 wheel/RECORD, dependency inventory, model revision/tree,
runtime ledger engine/model-set/model-assets lock이 모두 일치해야 하며 명령 실행만으로 새 OCR
Gate PASS를 주장할 수 없다.
환경 lock hash, model-assets lock hash, provider별 engine ID는 코드 소유 canonical 값과도
일치해야 하므로 임의 lock과 재작성 report를 함께 제시해 이 경계를 바꿀 수 없다.
현재 기존 OCR 가상환경은 새 requirements lock으로 재생성·sync한 release runtime이 아니므로
`environment_inventory_mismatch`로 차단된다. evaluation-v3 sealed runner와 C0/C1 freeze를
구현하고 새 exact wheel로 canonical 60+20을 실행해 결합 PASS를 만든 뒤에만 실제 provider
명령을 사용할 수 있다.

고정 ledger는 OCR 자체의 품질을 시험하는 모드가 아니라, OCR 관측을 byte/hash로 고정해 Claude
vision 이후 workflow를 반복 검증하는 모드다.

```json
{
  "document_id": "invoice-1",
  "filename": "invoice.png",
  "source": {
    "kind": "image",
    "path": "inputs/invoice.png",
    "ocr_ledger": "ledgers/invoice.ocr.json"
  }
}
```

```powershell
uv run --locked customs-ai-agent review `
  --manifest .\local-eval\review-manifest.json `
  --out ..\_artifacts\local-review-v2.json `
  --env-file ..\_secrets\.env `
  --rag-url http://127.0.0.1:8010 `
  --ocr-provider ledger
```

ledger의 document ID, page 순서, image SHA-256, width/height, canonical ledger ID가 현재 입력과
일치하지 않으면 실행하지 않는다. `ledger` 모드에 `--ocr-python`을 주거나, 실제 provider 모드에
precomputed ledger를 섞는 것도 거부한다. dependency/model/release lock 인자는 `ledger` 모드에
주지 않는다.

공개 result에는 다음이 포함되지 않는다.

- 원문 `documents`/`content_text`
- `image_pages`/`pdf_pages`/`data_base64`
- manifest/source/OCR-ledger 로컬 경로
- raw prompt/raw provider response

CLI 표준 출력에도 입력 경로와 원문을 표시하지 않는다. 공개 result를 Backend의 원문 저장소로
오해하면 안 된다. 추후 Backend가 원본 객체 저장소와 접근권한을 별도로 소유해야 한다.

## 단일 `.env`

local secret 기준은 네 저장소 밖 `project/_secrets/.env` 한 파일이다. 각 repo/worktree에 `.env`를
복제하지 않는다.

```dotenv
ANTHROPIC_API_KEY=
CUSTOMS_AGENT_CLAUDE_MODEL=claude-sonnet-5
CUSTOMS_AGENT_CLAUDE_MAX_TOKENS=8192
CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS=90
CUSTOMS_AGENT_CLAUDE_MAX_RETRIES=2
EVIDENCE_SERVICE_BASE_URL=http://127.0.0.1:8010
EVIDENCE_SERVICE_ALLOWED_ORIGINS=
EVIDENCE_SERVICE_TOKEN=
EVIDENCE_SERVICE_TOKEN_HEADER=X-Demo-Token
```

Agent는 `--env-file`로 지정한 파일만 allowlist 방식으로 읽고 `os.environ`을 변경하지 않는다.
process 환경값이 있으면 그것이 우선한다. key가 있어도 기본 Mock 테스트가 자동으로 과금 API를
호출하지 않는다.

Anthropic endpoint는 공식 HTTPS endpoint로 고정한다. 임의 base URL, custom auth/header,
SDK debug logging은 fail-closed한다. 모델 변경은 환경변수만 바꾸는 운영 토글이 아니라 새
평가셋·ADR·Gate 갱신이 필요한 변경이다.

일반 Agent 설정은 bounded SDK retry를 허용할 수 있지만, 비용 상한을 검증하는 Claude Live
evaluator는 환경의 retry 값과 무관하게 `max_retries=0`을 강제한다. `--live`,
`--max-calls`, `--max-total-input-tokens`, `--max-total-output-tokens`를 모두 명시해야 하며
실패한 main/crop 시도도 call budget을 소비한다. 합성 문서도 Anthropic 외부 전송이므로 전송
범위와 위험을 고지받은 사용자의 명시적 승인이 필요하다.

## Agent 계약

계약은 사용 경계에 따라 세 scope로 분리한다.

| scope | 소비자 | 포함 계약 |
|---|---|---|
| `backend` | 향후 제품 Backend | V1/V2 review request/result |
| `local` | 로컬 CLI adapter | 상대 파일 경로를 갖는 review manifest |
| `internal` | Agent 내부 adapter/evaluator | OCR ledger와 Claude candidate |

Backend가 소비할 수 있는 것은 `backend` scope뿐이다. `local` manifest의 `source.path`와
`ocr_ledger`는 파일을 읽어 public `ReviewRunRequest`로 바꾸기 위한 로컬 입력이며, Backend wire
계약이 아니다. 반대로 OCR 원장과 Claude candidate는 모델/OCR 경계의 검증 계약이지 제품
Backend가 저장하거나 조립할 요청/응답 모델이 아니다.

```powershell
uv run --locked customs-ai-agent export-schema --version v1 --kind request `
  --out contracts/agent/v1/review_request.schema.json
uv run --locked customs-ai-agent export-schema --version v2 --kind result `
  --out contracts/agent/v2/review_result.schema.json
uv run --locked customs-ai-agent validate-contract --version v2 --kind request `
  --input request.json
```

- V1 request: text only
- V2 Backend request: 필수 `case_id`, 1개 이상의 `documents`, 명시적
  `llm_mode=llm`만 허용한다. 각 문서는 text, bounded image pages 또는 bounded PDF pages 중
  정확히 하나이며, scenario/mock 필드는 Backend 계약에 존재하지 않는다.
- V1 result: `schema_version=1.0`
- V2 result: `schema_version=2.0`

설치된 wheel에서 계약 원문과 그 정확한 resource byte의 SHA-256을 읽을 수 있다.

```python
from customs_ai_agent.contracts import get_schema, schema_sha256

result_schema = get_schema("backend", "v2", "review-result")
result_schema_sha256 = schema_sha256("backend", "v2", "review-result")
```

`get_schema()`가 반환한 dict는 호출자용 복사본이며 package resource를 변경하지 않는다.
`verify_packaged_schemas()`는 wheel에 등록된 10개 JSON Schema가 현재 Pydantic 원본과 어긋나면
실패한다. `contracts/agent/v1|v2/`는 코드 리뷰용 checkout 사본이며 wheel의 `backend` resource와
의미상 동일해야 한다.

현재 OCR runtime ledger는 `schema_version=2.0`,
`internal/v2/ocr-layout-ledger.schema.json`이다. OCR worker의 성공 envelope가
`reading_order_policy_id=worker-emitted-sequence/v1`을 emit하고 Agent adapter가 이 exact 값을
검증한다. 이 policy ID와 ordered spans는 canonical `ledger_id` hash 재료에 포함되므로 순서
정책을 바꾸고 같은 ledger identity를 재사용할 수 없다. `internal/v1` OCR ledger schema는
역사 호환·감사용 package resource로만 유지한다. policy가 없던 v1 ledger 또는 그 ledger가
들어 있는 checkpoint를 현재 runtime provenance로 재사용하지 않는다.

OCR benchmark는 Gold에 `renderer-bbox-yx/v1`을 사용한다. 즉 renderer word bbox의 top-left
y/x와 source index tie-break로 시각 순서를 만들고, prediction에는 ledger `reading_order`를
사용한다. bbox geometry는 Gold 시각 순서와 bbox metric에만 권한이 있으며 worker/provider
prediction 순서를 다시 정렬하지 않는다.

Backend는 wheel commit, package version, schema scope/version/kind와 SHA-256을 함께 고정한다.
현재 V2는 아직 외부 release/tag 또는 Backend consumer pin이 없는 implementation candidate라
service runtime과 일치하도록 위 계약을 동결 전 교정했다. 동결 이후 Breaking change는 기존
V1/V2를 덮어쓰지 않고 다음 버전으로 추가한다. Backend용 공개 실행
API는 raw JSON bytes용 `AgentRuntime.review_json()`과 신뢰된 in-process object용
`AgentRuntime.review()`이며 현재 서비스 결과는 V2다. `customs_ai_agent.runner`의
V1/V2 함수는 mock scenario와 임의 dependency injection을 허용하는 CLI·test·evaluation 내부
도구이므로 Backend에서 호출하면 안 된다. 기존 V1 결과를 `schema_version=2.0`으로 재표시해서도
안 된다.

## 평가 데이터 해석

합성 문서가 실제 업무상 정합하지 않다는 사실만으로 즉시 “데이터 오류”로 수정하지 않는다.
불일치가 탐지·HITL·fail-closed를 검증하기 위해 의도된 것인지 먼저 분류해야 한다.

1. dataset manifest와 expected outcome에 anomaly 목적이 명시돼 있고 이미지·OCR ledger·Gold가
   같은 관측값을 가리키면 **의도된 negative/adversarial case**로 보존한다.
2. 원본 이미지의 가시값, OCR ledger, Gold, 설명 중 둘 이상이 충돌하고 의도가 명시되지 않으면
   **ambiguous/quarantined**로 두며 pass/fail 분모에서 제외한다.
3. 생성 과정과 근거가 확인된 뒤에만 Gold를 수정하고 dataset version/hash를 올린다.

따라서 “모순을 잘 잡았다”와 “정상 문서를 정확히 처리했다”를 하나의 정확도로 합치지 않는다.
normal, intentional anomaly, adversarial, quarantine을 분리해 보고한다.

세부 구조와 불변식은 [Architecture](docs/ARCHITECTURE.md), 평가 기준은
[Evaluations](evals/README.md), 취약점 보고는 [Security](SECURITY.md)를 참고한다.
