# 로컬 PDF Gold 검수 도구

`tools/review_document_gold.py`는 사람이 이미 확인한 합성 PDF의 Gold 초안을
검수하는 **로컬 전용** 도구다. Agent API·OCR runtime·일반 Review workflow와
분리되어 있으며, 이 도구가 Gold를 승인하거나 OCR engine을 활성화하지는 않는다.

## 실행 경계

- `--host`는 `127.0.0.1` 또는 `::1`만 허용한다. `0.0.0.0`, LAN IP,
  `localhost` 별칭은 거부한다.
- receipt, stratified-v2 packet, PDF dataset root, draft output은 모두
  workspace의 `_local/data/` 아래여야 한다.
- packet에 포함된 `application/pdf` asset만 열 수 있다. packet·receipt에 없는
  asset, dataset root 밖의 path, symlink를 통해 밖으로 나가는 PDF는 거부한다.
- PDF를 열거나 저장할 때마다 파일 크기와 SHA-256을 receipt·packet의 source
  SHA와 대조한다. receipt 또는 packet 파일이 서버 실행 뒤 변경되면 fail-closed
  한다.
- PDF bytes와 사람이 입력한 field/span/table text는 Git, 일반 로그, Agent
  transcript에 쓰지 않는다. local output의 `drafts/<asset-id>.json`에만 쓴다.

실행 예시는 다음과 같다. 모든 경로는 실제 workspace의 `_local/data` 경계 안을
가리켜야 한다.

```powershell
uv run --locked --no-sync --group dev python tools/review_document_gold.py `
  --receipt <workspace>/_local/data/document-preview/<pdf-dataset>/metadata/dataset-receipt.json `
  --packet <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/packets/<development-packet>.json `
  --dataset-root <workspace>/_local/data/document-preview/<pdf-dataset> `
  --output-directory <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/review-drafts `
  --gold-draft <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/drafts/<seeded-draft>.json
```

브라우저는 도구가 표시한 loopback 주소에서만 연다. 장기 실행 server나 외부 OCR
model 다운로드를 이 도구가 수행하지 않는다.

`--gold-draft`는 선택 사항이다. 지정하면 source-attested **draft** Gold의 field
값만 화면의 초기 후보로 채운다. receipt·dataset revision·packet receipt·각 PDF
source SHA·exact asset coverage가 모두 같고 Gold 상태가 `draft`일 때만 허용한다.
이 경우에도 span/table cell, 검수자 식별자, 제출 상태는 사람이 직접 확인·입력해야
하며, 화면은 `human_adjudicated`로 바꾸지 않는다.

## stratified-v2 입력과 handoff

기본 입력은 다음 필드를 가진 packet v2다.

```text
schema_version=2.0, packet_kind=stratified-v2, split,
dataset_id, dataset_revision, receipt_sha256, error_labels_sha256, packet_receipt_sha256,
prepared_as_of, items, required_gates, release_thresholds
```

`packet_receipt_sha256`는 자신을 제외한 canonical JSON의 SHA-256이어야 한다.
`error_labels_sha256`는 실제 오류 PDF만 error strata에 넣은 원본 오류 label CSV의
SHA-256이다. 이 필드가 없거나 바뀌면 구판 packet으로 간주해 열지 않는다.
item에는 asset ID, source SHA, PDF MIME, fixture kind, document type, set ID가
있어야 한다. 이전 v1 packet은 기존 schema를 완전히 만족할 때만 읽기 호환으로
열 수 있다.

검수 후 `GET /api/drafts/<asset-id>/handoff`는 아래처럼 Gold pipeline에 넘길
local-only intermediate를 결정론적으로 반환한다.

```text
dataset/revision + receipt SHA + packet receipt SHA + packet kind/split
+ asset ID/source SHA + reviewer ID/approval ID/review status
+ fields/spans/table cells
```

`reviewer_submitted`는 사람이 검수했다는 local 제출 상태일 뿐이다.
이 handoff에는 `human_adjudicated`가 없으며, Gold pipeline이 source SHA, packet
coverage, dataset revision, approval ID를 다시 검증한 뒤에만 최종 Gold로 승격할 수
있다.

## 사람 승인 Gold 확정

`tools/finalize_synthetic_pdf_ocr_gold.py`가 이 별도 승격 단계다. PDF 원본·Gold·검수
draft는 모두 `_local/data/` 아래에 있어야 하며, 이 명령은 네트워크·OCR·서버 실행을
하지 않는다.

```powershell
uv run --locked --no-sync --group dev python tools/finalize_synthetic_pdf_ocr_gold.py `
  --receipt <workspace>/_local/data/document-preview/<pdf-dataset>/metadata/dataset-receipt.json `
  --packet <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/packets/<development-packet>.json `
  --seeded-gold <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/drafts/<seeded-draft>.json `
  --source-directory <workspace>/_local/data/document-preview/<pdf-dataset> `
  --review-draft-directory <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/review-drafts/drafts `
  --output-directory <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/stratified-v2/human-adjudicated `
  --approval-id <opaque-approval-id> `
  --reviewer-id <opaque-reviewer-id> `
  --confirm-human-adjudication
```

`--confirm-human-adjudication`은 실제 렌더링 PDF를 검수한 사람이 실행할 때만
지정한다. 선택된 **모든** asset에 대해 `loopback_review_v1` origin,
`reviewer_submitted`, 동일한 reviewer/approval ID, nonempty field/span/table cell,
receipt·packet·source SHA 결속이 있어야 한다. 하나라도 누락·중복·혼합·변조되면
fail-closed한다. 최종 Gold에는 reviewer ID 원문 대신 SHA-256만 저장하고, 명령 출력도
원문 field/OCR text/사람 식별자를 포함하지 않는다.
