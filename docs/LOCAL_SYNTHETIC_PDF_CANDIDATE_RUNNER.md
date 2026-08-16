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
   fail-closed 합니다. PaddleOCR inventory의 25개는 pinned `uv.lock`의 2.9.1
   direct profile dependency boundary와 adapter의 `paddlepaddle`·`pypdf`·
   `pypdfium2`이며, 전체 Python import closure는 아니다. 특히 세
   OpenCV 배포본이 공존할 수 있으므로 `opencv-contrib-python`·`opencv-python`·
   `opencv-python-headless`를 빠짐없이 선언하고, 실제 `cv2` 파일이 designated
   contrib wheel `RECORD`의 regular file 집합·hash·size와 일치하지 않으면 후보 실행을 거부한다. active pointer는
   필요하지 않습니다. 평가는 승인보다 먼저 수행되기 때문입니다.

### Windows Docling native path 조건

Pinned Docling 2.15.1의 `docling-parse`와 `deepsearch-glm` native component는
Windows에서 non-ASCII `site-packages` 경로의 package-owned resource를 안정적으로
찾지 못할 수 있다. 이 경우 모델·PDF·network 문제가 아니어도 source PDF를 열기
전 `document_preview_docling_native_runtime_unavailable`으로 중단한다. 제품 코드가
drive mapping이나 junction을 자동으로 만들지는 않는다.

Docling configuration은 `pdf_backend: "pypdfium2"`를 반드시 명시한다. adapter는
이를 `PyPdfiumDocumentBackend`로 직접 전달하므로 pinned Docling의 기본
`DoclingParseV2` PDF backend를 선택하거나 다른 backend로 fallback하지 않는다.
다만 StandardPdfPipeline의 native GLM 실행은 남아 있으므로 위 ASCII-only
`site-packages` 검사와 memory-only native self-test는 그대로 필수이다.

프로젝트 경로에 한글 등이 있어도 **Docling을 설치해 실행하는 virtual environment
자체**를 ASCII-only 위치에 명시적으로 둘 수 있다. 이는 재생성 가능한 Python 환경일
뿐이며 PDF·Gold·OCR 결과·모델을 그 위치에 복사하지 않는다.

```powershell
$env:VIRTUAL_ENV = 'C:\customs-agent-docling-venv' # ASCII-only local path
uv venv $env:VIRTUAL_ENV
uv sync --active --locked --extra document-preview-docling --group dev

# provisioning은 sidecar 검증 뒤 memory-only blank PDF native self-test를 수행한다.
& "$env:VIRTUAL_ENV\Scripts\python.exe" tools/provision_document_preview_runtime.py `
  --document-preview-root <workspace>/_local/data/document-preview `
  --configuration <workspace>/_local/data/document-preview/staging/configuration-docling.json `
  --inventory <workspace>/_local/data/document-preview/staging/inventory-docling.json `
  --confirm-provision
```

그 self-test는 socket을 차단한 상태에서 메모리의 빈 1-page PDF만 변환하고 결과를
즉시 버린다. source PDF, OCR text, candidate bundle, sealed reservation은 만들지
않는다. 이 단계가 실패하면 environment를 바로잡은 뒤 provisioning부터 다시 해야
하며 development/sealed candidate를 실행해서는 안 된다.

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
- `synthetic-pdf-visible-layout-normalizer-v5` parser revision
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

`synthetic-pdf-visible-layout-normalizer-v5`은 후보 엔진이 낸 span과
table cell만 읽습니다. B/L 13개, Invoice 15개, Packing List 12개라는
**렌더링된 field 계약**을 대상으로 다음 두 가지 visible 의미만 사용합니다.

- form field는 명시적 `:`/`#`/dash label-value 또는 form label 바로 다음의
  같은 page value만 읽습니다. table caption은 다음 cell을 form value로
  사용하지 않습니다. label/header의 `/` 앞뒤 공백만 같은 label로 취급하며,
  후보가 낸 field value의 slash 표기는 바꾸지 않습니다.
- table field는 동일 table의 row-0 header와 단 하나의 row-1 body가 모두
  명시적으로 있을 때만 읽습니다. 열 위치·bbox·다른 문서·PDF content stream은
  사용하지 않습니다. Invoice/PL의 `qty + unit`은 `숫자 한 개 + 영문 unit 한
  개`이라는 명시적 cell token에서만 분리합니다. 단, Packing List의 Goods
  Description 한 cell이 줄바꿈 없이 평탄화된 경우에는 **종단의 단 하나인**
  `숫자 + ASCII 영문 unit` token만 분리합니다. token이 둘 이상이거나 종단이
  아니거나 ASCII unit이 아니면 item boundary를 추측하지 않습니다. PL
  `net_gross_weight`는 같은 표의 Net/Gross cell 두 개를 `net / gross`로
  결합합니다.

Docling의 **직접 구조화 table cell**을 bundle에 넣기 전에는 header label이나
문서 종류를 읽지 않는 별도 좌표 정책을 적용합니다. 정확히 한 table만 row 0과
row 1에 각각 column 0--4의 다섯 cell을 모두 가져야 합니다. 뒤따르는 footer는
다섯 column의 엄격한 부분집합인 partial row만 허용하며, 두 번째 완전 5-column
body, 좌표 누락·중복·gap·여섯 번째 column, 또는 target table이 0개/2개 이상이면
cell을 전부 비웁니다. 통과한 header/body 10개 cell만 `page=1`, `table=1`로
재색인합니다. 이 선택은 후보가 낸 좌표 외의 Gold·CSV·source truth·PDF content
stream·raw header label을 읽지 않습니다.

Packing List의 비렌더링 `containers` column은 평가 field에 포함하지 않습니다.
label/header의 중복, 값 충돌, body row 추가, 누락 또는 형식 불명확성은 해당
field를 비워 두며, Gold/CSV/다른 문서의 값으로 채우지 않습니다. 따라서
normalizer가 source-truth content parser나 `clean_manifest.csv`,
`error_labels.csv`를 import하거나 받지 않습니다.

runtime preview API는 structured table cell을 별도 필드로 공개하지 않습니다.
그러나 Docling 후보와 preview는 같은 in-memory structured extractor를 사용합니다.
이 extractor는 `DoclingDocument.iterate_items()`의 실제 page provenance와
`TableItem.data.grid`의 `(row, column, text)`를 그대로 읽고, `data.table_cells`의
원문 cell text를 CER span으로 한 번씩 유지합니다. 따라서 page·table·row·column은
Markdown 순서나 pipe 문법으로 추정하지 않습니다.

adapter가 보존한 raw table cell은 candidate bundle 직전에 별도 target-table
selection hook을 거칩니다. 문서 유형별 필수 header와 정확히 한 body row를 가진
완전한 대상 표가 **정확히 하나**일 때만 그 표를 page 1, table index 1로 재번호화합니다.
0개 또는 둘 이상의 대상 표는 빈 cell로 끝납니다. adapter가 다른 표를 합치거나
Gold·CSV·source truth·PDF content stream을 보고 셀을 만드는 경로는 없습니다.

page provenance가 하나가 아니거나, page/bbox가 유효 범위를 벗어나거나, grid와
cell offset이 불완전하면 Docling 후보는 fail-closed 합니다. 빈 cell은 채우지 않으며,
Gold·CSV·PDF content stream도 읽지 않습니다. Pdf text layer와 PaddleOCR가 자체적으로
검증된 structured cell을 내지 않으면 여전히 빈 목록이고 F1 Gate를 통과하지 못합니다.
이는 합성 PDF의 고정된 Docling structure/table output만 측정하는 범위이며 일반 문서
table 품질을 보장한다는 뜻은 아닙니다.

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
