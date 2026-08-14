# 합성 PDF OCR 후보 실행기

이 도구는 승인된 OCR 엔진을 서비스에 올리는 도구가 아닙니다. 사람 검수가
완료된 합성 PDF Gold와 나중에 비교할 수 있도록, 한 엔진의 원시 추출 후보를
로컬에 한 번 기록하는 오프라인 평가 도구입니다.

## 경계와 의도

- 입력은 PDF dataset receipt, stratified-v2 packet, 실제 PDF source directory,
  provisioned configuration/inventory sidecar뿐입니다.
- `Gold`, `clean_manifest.csv`, `error_labels.csv`, Case 문서, 대화 이력은
  후보 실행기와 label normalizer의 입력이 아닙니다.
- 원시 OCR span, field 후보, table cell 후보는 Git·일반 로그·Transcript에
  쓰지 않고 `_local/data/document-preview/candidate-outputs/`에만 저장합니다.
- 실행 결과에는 `approved_preview: false`가 명시됩니다. active approval
  pointer를 읽거나 만들지 않습니다.
- 모든 adapter 호출은 `local_files_only()` 안에서 실행됩니다. 네트워크,
  자동 모델 다운로드, 다른 엔진으로의 fallback은 허용하지 않습니다.

## 준비 조건

1. 운영자가 별도의 provisioning 명령으로 하나의 엔진의 package/model을
   명시적으로 설치하고, immutable sidecar를 기록합니다.
2. sidecar는 다음 정확한 위치에 있어야 합니다.

   - `_local/data/document-preview/runtime-configurations/<sha256>.json`
   - `_local/data/document-preview/model-inventories/<sha256>.json`

3. 실행기는 sidecar의 canonical byte/hash, package version, local model
   artifact byte/hash, 설정된 model directory를 모두 다시 확인합니다.
   active pointer는 필요하지 않습니다. 평가는 승인보다 먼저 수행되기
   때문입니다.

## 개발 평가 실행

각 엔진을 별도 configuration/inventory로 정확히 한 번씩 실행합니다.
PaddleOCR와 Docling은 text-layer fallback을 사용하지 않으므로 세 결과가
실제로 비교 가능한 독립 후보가 됩니다.

```powershell
uv run python tools/run_synthetic_pdf_ocr_candidate.py `
  --receipt <local-receipt> `
  --packet <development-packet> `
  --source-directory <local-pdf-dataset> `
  --document-preview-root <_local/data/document-preview> `
  --configuration <runtime-configurations/sha.json> `
  --inventory <model-inventories/sha.json> `
  --engine-kind pdf_text_layer `
  --output-directory <_local/data/document-preview/candidate-outputs/pdf-text> `
  --confirm-local-evaluation
```

출력에는 다음 결속 정보가 포함됩니다.

- engine kind와 extractor revision
- configuration SHA-256와 model inventory SHA-256
- `synthetic-pdf-label-normalizer-v1` parser revision
- `offline_synthetic_pdf_evaluation` profile mode
- 모든 위 값과 candidate document를 묶은 `candidate_bundle_sha256`

평가 명령은 이 bundle의 위 값을 독립적으로 검증한 sidecar와 정확히
대조합니다. 따라서 PDF를 실행한 뒤 다른 엔진·모델·parser 이름으로
재표기할 수 없습니다.

```powershell
uv run python tools/evaluate_synthetic_pdf_ocr.py `
  --receipt <local-receipt> `
  --packet <development-packet> `
  --gold-packet <human-adjudicated-gold> `
  --candidate-output <candidate-outputs/.../bundle.json> `
  --document-preview-root <_local/data/document-preview> `
  --configuration <runtime-configurations/sha.json> `
  --inventory <model-inventories/sha.json> `
  --engine-kind pdf_text_layer
```

개발 선택은 `pdf_text_layer`, `paddleocr`, `docling` 결과가 각각 정확히
한 개씩 있어야만 수행됩니다. 누락·중복·서로 다른 packet/Gold receipt는
선택 실패입니다.

## field와 table의 제한

`synthetic-pdf-label-normalizer-v1`은 문서 유형(B/L, Invoice, Packing
List)별 고정 label과 `:`/`#`/명시적 dash 구분자를 가진 span만 읽습니다.
값이 없거나 충돌하면 field를 비워 둡니다. Gold나 CSV 값으로 빈 field를
채우지 않습니다.

현재 runtime preview adapter는 structured table cell을 공개하지 않습니다.
실제 adapter가 `CandidateExtractionV1.table_cells`를 제공하면 실행기는
그 값을 그대로 보존하지만, 그렇지 않으면 빈 목록입니다. 특히 현재
Docling Markdown 경로는 모든 텍스트 chunk를 보존하지만 진짜 table
coordinate를 만들지 않습니다. 이 경우 table-cell F1 Gate는 자연스럽게
실패하며, table adapter를 구현하기 전에는 통과했다고 주장할 수 없습니다.

## 봉인 평가

봉인 packet은 PDF byte를 읽거나 adapter를 만들기 전에
`candidate-run-reservations/sealed-<packet-sha>.json`을 create-only로
기록합니다. 한 번 예약된 packet은 실행이 실패하거나 프로세스가 중단돼도
같은 root에서 재실행할 수 없습니다. 새 packet/Gold revision이 필요합니다.
이는 결과를 보고 재시도하여 좋은 결과만 고르는 일을 막기 위한
fail-closed 규칙입니다.

봉인 결과의 Gate 통과만으로는 서비스가 활성화되지 않습니다. 별도의
사람 승인 명령이 Gold receipt, sealed evaluation result, engine revision,
configuration, inventory를 대조한 뒤에만 PDF preview active pointer를
만듭니다. JPEG/PNG는 이 승인 범위 밖이며 계속 `unavailable`입니다.
