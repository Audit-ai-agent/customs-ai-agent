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
  --receipt <workspace>/_local/data/receipts/<dataset>/<revision>/dataset-receipt.json `
  --packet <workspace>/_local/data/evaluation/<dataset>/<revision>/document-preview-development-stratified-v2.json `
  --dataset-root <workspace>/_local/data/document-preview/<pdf-dataset> `
  --output-directory <workspace>/_local/data/gold-review/<dataset>/<revision>
```

브라우저는 도구가 표시한 loopback 주소에서만 연다. 장기 실행 server나 외부 OCR
model 다운로드를 이 도구가 수행하지 않는다.

## stratified-v2 입력과 handoff

기본 입력은 다음 필드를 가진 packet v2다.

```text
schema_version=2.0, packet_kind=stratified-v2, split,
dataset_id, dataset_revision, receipt_sha256, packet_receipt_sha256,
prepared_as_of, items, required_gates, release_thresholds
```

`packet_receipt_sha256`는 자신을 제외한 canonical JSON의 SHA-256이어야 한다.
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
