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
   artifact byte/hash, 설정된 model directory를 모두 다시 확인합니다. PaddleOCR와
   Docling의 각 설정 directory는 하위까지 재귀적으로 검사하며, 실제 regular file
   집합이 inventory의 path·size·hash 선언과 정확히 같아야 합니다. 미신고 file,
   누락 file, symlink 또는 Windows junction(reparse point)은 후보 실행 전에
   fail-closed 합니다. Docling inventory에는 패키지 본체뿐 아니라 layout/table/GLM
   실행 closure의 고정 version도 포함해야 하므로, 평가 후 전이 의존성 교체도
   fail-closed 합니다. active pointer는 필요하지 않습니다. 평가는 승인보다 먼저
   수행되기 때문입니다.

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
- `synthetic-pdf-visible-layout-normalizer-v3` parser revision
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

`synthetic-pdf-visible-layout-normalizer-v3`은 후보 엔진이 낸 span과
table cell만 읽습니다. B/L 13개, Invoice 15개, Packing List 12개라는
**렌더링된 field 계약**을 대상으로 다음 두 가지 visible 의미만 사용합니다.

- form field는 명시적 `:`/`#`/dash label-value 또는 form label 바로 다음의
  같은 page value만 읽습니다. table caption은 다음 cell을 form value로
  사용하지 않습니다.
- table field는 동일 table의 row-0 header와 단 하나의 row-1 body가 모두
  명시적으로 있을 때만 읽습니다. 열 위치·bbox·다른 문서·PDF content stream은
  사용하지 않습니다. Invoice/PL의 `qty + unit`은 `숫자 한 개 + 영문 unit 한
  개`이라는 명시적 cell token에서만 분리하고, PL `net_gross_weight`는 같은
  표의 Net/Gross cell 두 개를 `net / gross`로 결합합니다.

Packing List의 비렌더링 `containers` column은 평가 field에 포함하지 않습니다.
label/header의 중복, 값 충돌, body row 추가, 누락 또는 형식 불명확성은 해당
field를 비워 두며, Gold/CSV/다른 문서의 값으로 채우지 않습니다. 따라서
normalizer가 source-truth content parser나 `clean_manifest.csv`,
`error_labels.csv`를 import하거나 받지 않습니다.

현재 runtime preview adapter는 structured table cell을 공개하지 않습니다.
실제 adapter가 `CandidateExtractionV1.table_cells`를 제공하면 실행기는
그 값을 그대로 보존합니다. 현 Docling 후보는 예외적으로 page-1 Markdown
span 안의 **명시적 pipe table**만 고정 문법으로 읽을 수 있습니다.

- header row → `| --- |` separator row → 최소 한 개의 같은 폭 data row가
  모두 있어야 table로 인정합니다.
- page number는 1, Markdown 순서의 table index는 1부터, header row index는
  0, data row는 1부터, column index는 0부터입니다. separator row는 cell로
  기록하지 않습니다.
- Gold 검수자도 위 좌표 규칙으로 table cell을 확인해야 합니다. Gold나 CSV
  값, bbox, 추정한 행·열 경계로 빈 cell을 채우지 않습니다.

Pdf text layer와 PaddleOCR가 structured cell을 내지 않으면 빈 목록이며 F1
Gate를 통과하지 못합니다. Docling도 pipe table을 내지 않거나 Markdown이
깨져 있으면 동일하게 실패합니다. 따라서 이 추가 parser는 합성 PDF의
명시적 Markdown table 범위에서만 Gate를 가능하게 하며, 일반 문서 table
품질을 주장하는 기능이 아닙니다.

## 봉인 평가

봉인 packet은 PDF byte를 읽거나 adapter를 만들기 전에
`candidate-run-reservations/sealed-<packet-sha>.json`을 create-only로
기록합니다. 한 번 예약된 packet은 실행이 실패하거나 프로세스가 중단돼도
같은 root에서 재실행할 수 없습니다. 새 packet/Gold revision이 필요합니다.
이는 결과를 보고 재시도하여 좋은 결과만 고르는 일을 막기 위한
fail-closed 규칙입니다.

사람이 검수한 Gold 경로에서는 봉인 결과 뒤에도 별도의 사람 승인 명령이 Gold receipt,
sealed evaluation result, engine revision, configuration, inventory를 대조한 뒤에만 PDF
preview active pointer를 만듭니다. 반대로 `source_attested_synthetic` 경로는 별도 layout
truth·개발 selection·sealed state chain이 세 Gate를 통과했을 때만 evaluator가 pointer를
자동으로 만듭니다. 두 provenance를 섞을 수 없으며 상세 조건은
[`LOCAL_SOURCE_ATTESTED_SYNTHETIC_PDF_ACTIVATION.md`](LOCAL_SOURCE_ATTESTED_SYNTHETIC_PDF_ACTIVATION.md)에
있습니다. JPEG/PNG는 두 경로 모두 계속 `unavailable`입니다.
