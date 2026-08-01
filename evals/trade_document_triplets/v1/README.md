# Trade Document Triplets source-v1 / evaluation-v2

이 디렉터리는 Git에 넣을 수 있는 계약·설계만 보관한다. 사용자가 제공한 CI/PL/B/L 값과
파생 PDF·PNG·Gold는 로컬
`_local/data/trade_document_triplets/source-v1/derived/`에만 생성한다. 모든 runner는
DatasetProvenanceV2·relocation manifest·retired legacy path 부재를 먼저 검증한다.

## 정답 계층

- Extraction Gold는 원문에 실제 저장된 문자열만 보존한다. 계산상 의심되는 값을 고치지 않는다.
- Validation Gold는 별도 adjudication으로 관리한다. 원문 추출 정답과 업무 판정을 섞지 않는다.
- 원본 데이터의 불일치는 작성 의도가 확인되지 않았으므로 `ambiguous` 또는 `coverage_gap`이며
  precision/recall 점수에서 제외한다.
- `confirmed_intentional_defect`는 이 저장소의 결정론적 generator가 한 필드만 변경하고 mutation
  manifest를 남긴 통제 변형에만 자동 부여한다.

6개 상태는 다음과 같다.

```text
confirmed_normal
confirmed_intentional_defect
confirmed_generator_error
ambiguous
coverage_gap
not_applicable
```

`confirmed_normal`과 `confirmed_intentional_defect`만 판정 증거가 있을 때 점수화할 수 있다.

## 재현 명령

```powershell
uv run --locked python -m tools.register_trade_document_triplets `
  --source-root <USER_SOURCE> `
  --dataset-root <CANONICAL_DATASET_ROOT>

uv run --locked python -m tools.repair_trade_triplet_container_mutation `
  --source-root <CANONICAL_DATASET_ROOT>/derived/evaluation/v2 `
  --output-root <CANONICAL_DATASET_ROOT>/derived/evaluation/v2-container-repair-v1

uv run --locked python -m evals.run_deterministic_validation `
  --defects <CANONICAL_DATASET_ROOT>/derived/evaluation/v2-container-repair-v1/injected-defects/cases.jsonl `
  --manifest <CANONICAL_DATASET_ROOT>/derived/evaluation/v2-container-repair-v1/manifest.json `
  --splits <CANONICAL_DATASET_ROOT>/derived/evaluation-splits/v2/splits.json `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json `
  --out <RUN_ARTIFACT_ROOT>/deterministic-validation.json

uv run --locked python -m tools.build_trade_triplet_rule_coverage `
  --dataset-root <CANONICAL_DATASET_ROOT> `
  --output-root <CANONICAL_DATASET_ROOT>/derived/evaluation/rule-coverage-v1 `
  --descriptor evals/trade_document_triplets/v1/rule-coverage-v1.descriptor.json `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json

uv run --locked python -m evals.run_rule_coverage_validation `
  --defects <CANONICAL_DATASET_ROOT>/derived/evaluation/rule-coverage-v1/injected-defects/cases.jsonl `
  --negative-controls <CANONICAL_DATASET_ROOT>/derived/evaluation/rule-coverage-v1/negative-controls/cases.jsonl `
  --manifest <CANONICAL_DATASET_ROOT>/derived/evaluation/rule-coverage-v1/manifest.json `
  --selection-descriptor evals/trade_document_triplets/v1/rule-coverage-v1.descriptor.json `
  --dataset-lock evals/trade_document_triplets/v1/dataset.lock.json `
  --out <RUN_ARTIFACT_ROOT>/rule-coverage-validation.json

uv run --locked python -m evals.run_workflow_boundary_validation `
  --descriptor evals/trade_document_triplets/v1/workflow-boundary.descriptor.json `
  --out <RUN_ARTIFACT_ROOT>/workflow-boundary-validation.json

uv run --locked python -m tools.render_trade_document_fixtures `
  --dataset-root <CANONICAL_DATASET_ROOT> `
  --evaluation-root <CANONICAL_DATASET_ROOT>/derived/evaluation/v2-container-repair-v1 `
  --output-root <CANONICAL_DATASET_ROOT>/derived/rendered/v2-container-repair-v1
```

등록은 source bytes와 pinned `dataset.lock.json`을 함께 검증한다. Renderer는 ReportLab invariant
PDF를 만들고 pypdfium2로 200 DPI 원본 PNG를 생성한다. Claude 입력 제한을 넘는 PNG는 종횡비를
유지해 축소한 별도 `.claude.png`를 만든다. word·field·table cell마다 PDF-point, 200 DPI
top-left pixel, Claude-input pixel 좌표를 모두 기록하고, 두 단계 좌표 변환과 scale을 layout
Gold에 고정한다.

기존 `derived/splits.json`, `derived/evaluation/v1`, EVAL-0003은 삭제하지 않지만
`holdout_compromised_by_precommit_evaluation`인 역사 증거다. release 성능 근거로 재사용할 수
없다. evaluation-v2의 재배치도 repository-visible deterministic seed를 사용했으므로 release
holdout이 아니다. 두 version은 모두 audit/development 진단 이력으로만 보존한다.

evaluation manifest v3는 dataset lock·evaluation-v2 provenance bytes·controlled-defect input
hash와 operator assignment를 함께 고정한다. 역사적 v2 evaluator는 공개된 split JSON을 정확한
hash로 고정하고 development 20 Case에서 기존 20개 business-field operator만 재현한다.
evaluation-v2 전체 asset builder는 무효화되어 재실행을 차단한다. 별도 workflow
descriptor는 required document, duplicate type, OCR conflict, Evidence partial/failed/ambiguous
6개 경계를 점검하며 하나라도 없으면 평가 시작 전에 실패한다.

`rule-coverage-v1`은 등록된 100개 공개 합성 Case 전체에서 별도 descriptor와 seed로 30개를
선택한다. 29개 positive variant는 29개 business operator를 relation-aware finding 단위
(`issue_code`, `source_document_id`, `target_document_id`)로 점수화한다. 나머지 1개는 PL-only
추가 품목을 자동 위반으로 만들지 않는 negative control이며, 기존 unmatched 품목 수 대비 정확히
1개 증가하고 finding 추가·삭제와 review 상태 변화가 모두 0인지 검증한다. 이 suite는 공개
development diagnostic이며 release split membership을 읽지 않지만, 공개 seed와 데이터에서
재현 가능하므로 release·모델 선택 근거가 아니다.

evaluation-v2의 locked 구성은 공개 seed와 일반 tooling에서 사전 도출 가능하므로 영구
`release_holdout_v2_publicly_derivable`이다. 한 번의 authorization이나 descriptor freeze로
이 상태를 되돌릴 수 없다. 기존 development 결과는 non-release 진단으로만 보존한다.

기존 v2 controlled defect 중 `pl_container_mismatch`는 문자열 suffix를 붙여 container mismatch와
ISO 6346 format 오류를 동시에 만든 `confirmed_generator_error`다. 원본 v2 bytes는 역사 증거로
보존하고, exact old bundle hash만 받는 one-time migration이 BL/PL controlled parent를 유효한
check digit으로 보정한 뒤 서로 다른 유효 번호 하나만 주입한다. repaired sibling도 evaluation-v2의
공개 membership을 그대로 사용하므로 development diagnostic일 뿐 release Gold가 아니다.
수리 명령은 출력 디렉터리가 없을 때 한 번만 실행하며 기존 v2를 수정하지 않는다. nested
adjudication evidence까지 mutation과 byte-level로 일치해야 historical evaluator가 입력을
수락한다.

replacement는 사람이 저장소 밖에서 공급하는 evaluation-v3 sealed/unseen bundle이어야 한다.
공개 manifest에는 count와 bundle SHA-256만 허용하며 case ID·membership·deterministic seed를
금지한다. 자세한 계약과 현재 `release_holdout_v3_sealed_bundle_not_provisioned` blocker는
`docs/RELEASE_HOLDOUT_PROTOCOL.md`를 따른다.

## C/O fixture

20개 C/O는 원본 데이터의 일부가 아니라 명시적인 controlled synthetic extension이다. 10자리
값은 OCR/field-format 테스트용 synthetic identifier이며 법적 HS 분류가 아니다. 이 값으로
Evidence API를 검색하거나 관세 분류 정확도를 주장하면 안 된다.
