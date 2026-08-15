# 합성 PDF source-attested 자동 평가

이 경로는 사람이 재입력한 Gold를 대체하는 일반 OCR 인증이 아니다. 생성기
소스가 현재 workspace에 없기 때문에, 최종 born-digital 합성 PDF의 제한된
content stream을 독립적인 pinned parser와 template profile로 읽어 만든
`synthetic_content_stream_compatibility` 평가다.

즉 이 결과는 이 정확한 합성 PDF 템플릿에서의 추출 호환성 증거다. 실제 기업
문서, 스캔 품질, 손글씨, 도장 중첩 또는 독립 optical OCR 품질을 보장하지
않는다. JPEG/PNG는 이 경로에 포함되지 않으며 계속 `unavailable`이다.

## 안전 경계

- PDF, layout truth, candidate OCR, 모델, receipt는 모두 Git 밖
  `_local/data/document-preview/` 아래에만 둔다.
- compiler는 PDF 1.3의 관측된 ASCII85+Flate content-stream subset만 받고,
  page/template/field position/table rectangle/asset SHA가 달라지면 중단한다.
- `error` stratum은 error label만으로 인정하지 않는다. 같은 set/document의
  clean sibling과 decoded content stream이 실제로 달라야 한다.
- `clean_manifest.csv`와 `error_labels.csv`는 dataset 결속, strata와 compact
  mismatch receipt용이다. PDF에 실제로 그려진 값을 조용히 고치거나 OCR 후보
  출력으로 정답을 만들지 않는다.

## 불변 local artifact

모든 파일은 canonical sorted JSON + newline이고 hash 이름과 일치해야 한다.
hash의 semantic payload는 별도 self-hash 필드가 있으면 그 필드만 제외한 compact
sorted JSON이다.

- `source-attested-dataset-receipts/<dataset receipt SHA>.json`
- `source-attested-packets/<packet receipt SHA>.json`
- `source-layout-truths/<layout truth SHA>.json`
- `source-attested-candidate-bundles/<candidate bundle SHA>.json`
- `source-attested-evaluation-results/<evaluation result SHA>.json`
- `source-attested-development-selections/<selection SHA>.json`
- `source-attested-sealed-runs/<sealed run SHA>.json`

기존 candidate runner의 sealed one-shot reservation은
`candidate-run-reservations/sealed-<packet SHA>.json`이며, 그 raw canonical
bytes SHA가 sealed-run receipt에 결속된다.

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
