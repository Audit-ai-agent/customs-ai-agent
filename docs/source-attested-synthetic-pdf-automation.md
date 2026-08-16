# 합성 PDF source-attested 자동 평가

이 경로는 사람이 재입력한 Gold를 대체하는 일반 OCR 인증이 아니다. 생성기
소스가 현재 workspace에 없기 때문에, 최종 born-digital 합성 PDF의 제한된
content stream에서 source-attested layout truth를 만들고, 명시적으로 감독 학습한
native template 후보를 비교하는
`synthetic_r1_supervised_native_template_compatibility` 평가다.

즉 이 결과는 이 정확한 합성 PDF 템플릿에서의 추출 호환성 증거다. 실제 기업
문서, 스캔 품질, 손글씨, 도장 중첩 또는 독립 optical OCR 품질을 보장하지
않는다. JPEG/PNG는 이 경로에 포함되지 않으며 계속 `unavailable`이다.

현재 r1 v11 감독형 profile의 실제 문서 field, 학습 분리, 개발·봉인 결과와
한계를 이해하기 쉬운 형태로 정리한 안내는
[`R1_SUPERVISED_NATIVE_PDF_EVALUATION.md`](R1_SUPERVISED_NATIVE_PDF_EVALUATION.md)를
참조한다. 아래의 여러 엔진 비교·실행 설명은 이전 후보를 검증하고 감사하기 위한
일반 경로를 포함한다. canonical r1 v11 활성화는 순위로 세 엔진을 고르는 방식이
아니라, 고정된 supervised `pdf_text_layer` profile 하나만 허용한다.

## 안전 경계

- PDF, layout truth, candidate OCR, 모델, receipt는 모두 Git 밖
  `_local/data/document-preview/` 아래에만 둔다.
- compiler는 PDF 1.3의 관측된 ASCII85+Flate content-stream subset만 받고,
  page/template/field position/table rectangle/asset SHA가 달라지면 중단한다.
- 자동 승인 기본 경로는 `synthetic-customs-pdf-2000`의 `20260811-r1`, 정확히
  6,000 PDF(정상 3,000/오류 3,000)와 그 receipt SHA, 그리고 고정된 개발 60개·봉인
  20개 packet receipt만 받는다. 같은 이름의 작은 샘플·임의 packet은 평가 증거가
  될 수 없다.
- field gate는 실제 최종 PDF에 렌더링된 구조화 field를 모두 요구한다. 문서 한 장당
  B/L 13개, Invoice 15개, Packing List 12개이며, evaluator result에는 문서 유형별
  `document_count × fields_per_document = total_fields` denominator도 함께 기록한다.
  Packing List의 이전 manifest `containers` 열은 최종 PDF에 label/value/table slot으로
  렌더링돼 있지 않으므로 12개 계약에 포함하지 않는다. carton 값, 같은 set의 B/L,
  manifest/CSV로 이를 추론하거나 보완하지 않는다.
- 각 mapping은 source stream의 고정 run 또는 고정 table column에 결속한다. Invoice와
  Packing List의 `qty`/`unit`은 같은 visible run에서 정확히 하나인 수치 token과 영문
  unit token을 분리하고, Packing List `net_gross_weight`는 visible net/gross 두 slot을
  고정 `" / "` 구분자로 결합한다. 이 변환 규칙도 parser-spec self-hash에 포함된다.
  run 수·좌표·table geometry·token 개수가 달라져 모호해지면 PDF preview는
  `unavailable`로 남는다.
- candidate parser revision은 정확히
  `synthetic-pdf-supervised-template-normalizer-v11`만 허용한다. 따라서 v10 또는 그 이전의
  candidate/execution/result/selection/sealed-run이 모든 SHA와 서로 일치해도
  새 자동 pointer를 만들 수 없다. v11이 아닌 revision은 source-truth writer와
  runtime loader 양쪽에서 fail-closed 한다.
- `error` stratum은 error label만으로 인정하지 않는다. 같은 set/document의
  clean sibling과 decoded content stream이 실제로 달라야 하며, label의
  `correct_value → wrong_value`가 해당 source field에 실제로 렌더링된 것도
  확인해야 한다. 표기 단위만 다른 numeric/table label은 값 하나 또는 net/gross
  값쌍의 fixed numeric projection으로 mutation만 검증한다. 이 projection은 source
  truth 값을 만들지 않으며, source truth에는 여전히 PDF에서 읽은 원문 slot 값만 쓴다.
- `clean_manifest.csv`와 `error_labels.csv`는 dataset 결속, strata와 compact
  mismatch receipt용이다. PDF에 실제로 그려진 값을 조용히 고치거나 OCR 후보
  출력으로 정답을 만들지 않는다.
- 기존 PaddleOCR 2.9 후보는 `PP-OCRv4`, CPU 1 thread와
  detection/recognition/classifier 세 local directory를 모두 configuration·inventory에
  고정한다. PaddleOCR 3 후보는 `runtime_api: "paddleocr3-paddlex-3.2"`와 CPU 1 thread,
  `models/paddleocr3/det`·`models/paddleocr3/rec`, `raster_scale: 3`만을 별도
  configuration·inventory에 고정한다. P3 configuration은
  `paddleocr-3.2.0-ppocrv5-mobile-en-raster216-v1` revision만 허용하며, inventory는
  해당 ASCII-only CPython 3.11/Windows AMD64 profile의 56개 metadata dependency
  closure와 고정 version을 정확히 모두 가져야 한다. 두 Paddle profile은 같은
  `paddleocr` distribution의 호환되지 않는 API를 쓰므로 environment와 fallback을
  공유하지 않는다. Docling 후보는 CPU 1 thread의 local
  layout/table parser로만 동작하며 EasyOCR downloader를 피하기 위해 `ocr_enabled: false`,
  `table_structure_mode: "fast" | "accurate"`, `pdf_backend: "pypdfium2"`를 고정한다.
  두 table mode는 서로 다른 configuration·inventory·extractor revision의 별도 profile이며
  mode 자동 대체는 없다.
  adapter는 `PyPdfiumDocumentBackend`를 직접 선택하므로 기본 `DoclingParseV2`와 다른
  backend는 fallback 후보가 아니다. StandardPdfPipeline native GLM의 Windows 경계는
  별도로 남으므로 ASCII-only environment preflight도 계속 필요하다. 이 둘의 실제 실행
  profile은 source-attested runtime에서도 fallback 없이 그대로 사용된다.

## 불변 local artifact

모든 파일은 canonical sorted JSON + newline이고 hash 이름과 일치해야 한다.
hash의 semantic payload는 별도 self-hash 필드가 있으면 그 필드만 제외한 compact
sorted JSON이다.

- `source-attested-dataset-receipts/<dataset receipt SHA>.json`
- `source-attested-packets/<packet receipt SHA>.json`
- `source-layout-truths/<layout truth SHA>.json`
- `source-attested-candidate-bundles/<candidate bundle SHA>.json`
- `source-attested-candidate-executions/<execution receipt SHA>.json` — raw
  candidate JSON의 SHA, parser revision, engine/configuration/inventory 및
  source-layout/packet 결속만 담는 content-free sidecar
- `source-attested-evaluation-results/<evaluation result SHA>.json`
- `source-attested-development-selections/<selection SHA>.json`
- `source-attested-sealed-runs/<sealed run SHA>.json`

기존 candidate runner의 sealed one-shot reservation은
`candidate-run-reservations/sealed-<packet SHA>.json`이며, 그 raw canonical
bytes SHA가 sealed-run receipt에 결속된다.

evaluation result·development selection·sealed run은 모두 candidate parser
revision과 execution receipt SHA를 함께 결속한다. runtime은 활성 pointer를 읽을
때 candidate 원문을 반환하거나 로그에 쓰지 않고 local canonical bytes를 메모리에서
다시 점수화한다. 재계산 metric, candidate/output SHA, source-layout/packet/profile
중 하나라도 달라지면 pointer를 사용하지 않는다.

## 실행 순서

먼저 개발 packet과 봉인 packet 각각의 layout truth를 만든다.

```powershell
uv run python tools/compile_source_attested_synthetic_pdf_layout_truth.py `
  --receipt <local-dataset-receipt> --packet <development-or-sealed-packet> `
  --source-directory <local-synthetic-pdf-root> `
  --clean-manifest <local-clean-manifest> --error-labels <local-error-labels> `
  --document-preview-root <_local/data/document-preview>
```

명시적으로 provision된 세 후보를 기존 `run_synthetic_pdf_ocr_candidate.py`로
개발 packet에 한 번씩 실행한다. 각 candidate output을 다음 evaluator로
평가한다.

```powershell
uv run python tools/evaluate_source_attested_synthetic_pdf.py `
  --receipt <local-dataset-receipt> --packet <development-packet> `
  --layout-truth <development-layout-truth> --candidate-output <candidate-output> `
  --document-preview-root <_local/data/document-preview> `
  --configuration <configuration-sidecar> --inventory <model-inventory-sidecar> `
  --engine-kind <pdf_text_layer|paddleocr|docling>
```

세 development result를 함께 선택한다. 일부 후보만 비교하면 선택이 실패한다.

```powershell
uv run python tools/select_source_attested_synthetic_pdf_engine.py `
  --result <development-result-1> --result <development-result-2> `
  --result <development-result-3> `
  --document-preview-root <_local/data/document-preview>
```

선정된 정확한 engine/configuration/inventory로 sealed candidate를 한 번 실행한
후, sealed evaluation에 development selection을 전달한다. evaluator는 candidate
runner가 만든 reservation을 확인한 뒤 sealed-run receipt를 먼저 write한다.

```powershell
uv run python tools/evaluate_source_attested_synthetic_pdf.py `
  --receipt <local-dataset-receipt> --packet <sealed-packet> `
  --layout-truth <sealed-layout-truth> --candidate-output <sealed-candidate-output> `
  --development-selection <development-selection> `
  --document-preview-root <_local/data/document-preview> `
  --configuration <selected-configuration-sidecar> `
  --inventory <selected-model-inventory-sidecar> --engine-kind <selected-engine> `
  --auto-activate-source-attested-synthetic
```

마지막 switch는 sealed packet이고 세 Gate가 모두 통과했을 때만 hardened runtime
pointer writer를 호출한다. 개발 평가, failed sealed 평가, receipt/hash/model/config
불일치에서는 pointer가 생성되지 않는다. 활성화 뒤에도 OCR 결과는 Finding·통관
판단·자동 review 입력이 아니며, 사용자가 확인/수정한 텍스트만 기존 text-only
review로 전달된다.
