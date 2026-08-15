# 합성 PDF source-attested 자동 활성화

이 경로는 사람이 PDF를 다시 검수하지 않아도 되는 **합성 PDF 전용** 품질
활성화 경로다. 기존 사람이 만든 합성 source truth와 final PDF의 authored
content stream에서 만든 layout truth를 해시로 결속한다. 사람이 검수한 Gold
경로를 대체하거나 수정하지 않으며, 두 증거를 섞을 수 없다.

## 의미와 범위

- `evaluation_basis`는 정확히 `source_attested_synthetic`이어야 한다.
- `layout_truth_kind`는 정확히 `pdf_content_stream_layout_v1`이어야 한다.
- `evaluation_scope`는 정확히 `synthetic_content_stream_compatibility`이다.
- 따라서 native PDF text-layer 후보의 통과는 **PDF content-stream 호환성**
  증거이며, 이미지 기반 광학 OCR 품질 증거가 아니다.
- PDF, 선택한 synthetic dataset revision, sealed packet에만 적용된다. JPEG와
  PNG는 이 pointer가 있어도 계속 `unavailable`이다.
- 실제 기업 문서, 스캔, 손글씨, 도장 겹침, 저화질 이미지에 대한 품질 주장은
  만들지 않는다.

layout truth compiler는 candidate OCR output, Gold UI, human review draft를
읽으면 안 된다. 이 자료의 dataset/packet/PDF hash·parser specification 등은
layout truth receipt 안에서 검증된다. 개발용 60개와 봉인용 20개는 서로 다른
packet이므로 서로 다른 layout truth receipt를 가진다. 개발 selection은 개발
receipt를, sealed result와 pointer는 봉인 receipt를 각각 참조한다. 두 receipt는
같은 dataset revision·dataset receipt·parser specification을 공유해야 하지만,
hash 자체가 같으면 오히려 잘못된 결속이다.

## 자동 활성화 전제

자동 sealed evaluator result는 canonical JSON이어야 하며 다음을 모두 가져야
한다.

- `evaluation_basis: source_attested_synthetic`
- `evaluation_scope: synthetic_content_stream_compatibility`
- `layout_truth_kind: pdf_content_stream_layout_v1`
- `layout_truth_receipt_sha256`
- `layout_truth_parser_spec_sha256`
- `development_selection_receipt_sha256`
- `sealed_run_receipt_sha256`
- sealed packet/dataset/engine/configuration/model inventory hash
- field exact match `>= 0.99`, CER `<= 0.01`, table-cell F1 `>= 0.98`
- `passed: true` 및 self-excluding result SHA-256

manual human Gold result, `gold_receipt_sha256`, reviewer/approval ID, 빠진
provenance field, 다른 scope/kind는 이 경로에서 모두 거부된다. 반대로 기존
`approve_document_preview_runtime.py`는 기존 human-adjudicated result만 받는다.

또한 아래 local-only 증거 파일이 모두 존재하고 canonical self-hash를 통과해야
한다. 파일은 regular file이어야 하며 symlink, 누락, byte 변조, 다른 hash 이름은
모두 fail-closed 한다.

- `source-layout-truths/<sealed-layout-truth-sha>.json`: 봉인 20개 전체의
  source PDF SHA와 6개 strata(각 최소 3개)를 보존한다.
- `source-layout-truths/<development-layout-truth-sha>.json`: 개발 60개 전체의
  source PDF SHA와 6개 strata(각 10개)를 보존한다. 봉인 packet과 PDF SHA/asset ID가
  겹치면 거부한다.
- `source-attested-development-selections/<selection-sha>.json`: 정확히
  `pdf_text_layer`, `paddleocr`, `docling` 세 후보와 선택된 passing 후보의 engine,
  revision, configuration, inventory, development result hash를 결속한다.
- `source-attested-sealed-runs/<sealed-run-sha>.json`: 선택 결과, 봉인 layout,
  candidate bundle/parser 및 engine/configuration/inventory를 결속한다.
- `candidate-run-reservations/sealed-<sealed-packet-sha>.json`: 봉인 후보 실행을
  한 번만 허용한 create-only 예약의 exact canonical-byte SHA를 결속한다.

runtime은 pointer 생성 전, 다음 Agent process 시작 시, 그리고 PDF preview 직전에
이 체인을 다시 검증한다. 따라서 이미 떠 있는 process가 변경된 pointer나 receipt를
새 승인으로 받아들이지 않는다.

모델과 package는 여전히 별도 명시 provisioning으로 local-only inventory에
기록해야 한다. 이 활성화 명령은 다운로드, network fallback, model 설치를
수행하지 않는다.

## 활성화

source-attested evaluator가 `evaluation-results/` 아래에 passing sealed result를
기록한 뒤, evaluator/orchestrator의 명시적
`--auto-activate-source-attested-synthetic` switch는 아래 runtime helper를 같은
local workflow에서 호출한다. 개발 평가, 실패한 sealed 평가, switch가 없는 일반
평가는 pointer를 절대 쓰지 않는다.

다음 명령은 같은 helper를 직접 호출하는 local-only 동등 경로다. 결과·configuration·inventory·설치 package·local model artifact를 다시 검증하고 active pointer를 원자적으로 교체한다.

```powershell
uv run --locked --no-sync --group dev python tools/activate_source_attested_synthetic_pdf_preview.py `
  --document-preview-root <workspace>/_local/data/document-preview `
  --evaluation-result <workspace>/_local/data/document-preview/evaluation-results/<result-sha>.json `
  --configuration-sha256 <configuration-sha> `
  --model-inventory-sha256 <inventory-sha>
```

출력의 `activation_sha256`을 기존 paired setting에 정확히 넣고 **새 Agent
process를 시작한다**. 실행 중인 Agent가 pointer를 live reload하지 않으며, 그
process에 환경값이나 파일을 바꾸는 행위는 활성화가 아니다.

```powershell
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH = '<workspace>/_local/data/document-preview/approvals/pdf-preview-active.json'
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256 = '<activation-sha256>'
```

두 setting이 없거나, pointer/result/configuration/inventory/model artifact 또는 위
증거 체인 중 하나라도 달라지면 runtime은 기본 disabled 상태로 fail-closed 한다.
서비스 시작만으로 candidate 평가나 pointer 활성화가 자동 실행되지는 않는다.
