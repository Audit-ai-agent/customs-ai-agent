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
layout truth receipt 안에서 검증되고, sealed evaluator result는 그 receipt의
`layout_truth_receipt_sha256`만 참조한다.

## 자동 활성화 전제

자동 sealed evaluator result는 canonical JSON이어야 하며 다음을 모두 가져야
한다.

- `evaluation_basis: source_attested_synthetic`
- `evaluation_scope: synthetic_content_stream_compatibility`
- `layout_truth_kind: pdf_content_stream_layout_v1`
- `layout_truth_receipt_sha256`
- sealed packet/dataset/engine/configuration/model inventory hash
- field exact match `>= 0.99`, CER `<= 0.01`, table-cell F1 `>= 0.98`
- `passed: true` 및 self-excluding result SHA-256

manual human Gold result, `gold_receipt_sha256`, reviewer/approval ID, 빠진
provenance field, 다른 scope/kind는 이 경로에서 모두 거부된다. 반대로 기존
`approve_document_preview_runtime.py`는 기존 human-adjudicated result만 받는다.

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

출력의 `activation_sha256`을 기존 paired setting에 정확히 넣고 새 Agent
process를 시작한다.

```powershell
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH = '<workspace>/_local/data/document-preview/approvals/pdf-preview-active.json'
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256 = '<activation-sha256>'
```

두 setting이 없거나, pointer/result/configuration/inventory/model artifact 중
하나라도 달라지면 runtime은 기본 disabled 상태로 fail-closed 한다. 서비스
시작만으로 candidate 평가나 pointer 활성화가 자동 실행되지는 않는다.
