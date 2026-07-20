# Trade image diagnostic-v1

## 입력 경로와 등록 경계

사용자가 복원한 workspace root `trade_documents_images/`는 byte-for-byte 원본 참고본이다.
평가 runner는 `_artifacts/datasets/trade_documents_images/source-v1/raw/`만 canonical
입력으로 사용하며 root를 대체 경로로 탐색하지 않는다. 양쪽은 3,002개 파일,
260,248,817 bytes, 상대경로·크기·SHA-256 mismatch 0으로 2026-07-18에 재검증됐다.

이 등록은 source bytes 무결성을 증명할 뿐 3,000장 전체의 field/OCR/bbox/table/finding
Gold를 만들지 않는다. README가 참조하는 `shipments_master.json`은 여전히 누락됐고, 전체
corpus 상태는 `candidate_unverified`다. 공개 재배포 권리도 확인되지 않아 raw는 Git 또는
공개 포트폴리오에 포함하지 않는다.

## 무효화된 case 0500 기록

`manifest_partial_gold.json`은 호환 경로만 유지하는 과거 후보 annotation이다. 과거 suffix
anomaly 라벨 오류로 annotation 과정이 오염됐고, 수정 관측도 한 명만 확인했기 때문에 artifact
전체 상태는 `invalidated_by_label_error`다.

- `gold_set_id`와 `expected_*` 키를 제거했다.
- `release_scoring_allowed=false`다.
- 값은 감사와 비점수 debugging용 `observed_candidate_*`로만 보존한다.
- 이 파일을 사용한 과거 PASS·정확도·회귀 결과는 release score로 재사용할 수 없다.
- 파일 hash만 새 값으로 갱신해 다시 허용하는 행위를 금지한다.

이 디렉터리의 12개 triplet 선택은 원본 1,000 Case의 document hash metadata를 기준으로
SHA-256 순위를 계산한 결정론적 진단 표본이다. CI·PL·B/L은 Case 단위로 함께 선택하므로
문서 간 leakage가 없다.

`diagnostic-selection-v1.json`은 선택 근거와 36개 image SHA-256을 고정하는 descriptor이며
Gold가 아니다. 최초로 생성된 pending worklist는 다음 경로에 역사적 증거로 남아 있다.

```text
_artifacts/datasets/trade_documents_images/source-v1/
  derived/diagnostic-selection/v1/adjudications.jsonl
```

각 Case는 고정된 세 이미지를 모두 확인한 서로 다른 두 reviewer의 독립성 확인이 필요하다.
동일 결론은 `two_reviewer_consensus` resolution을, 다른 결론은 두 reviewer와 다른 제3의
adjudicator ID·독립성 확인·resolution을 요구한다. 도구는 기존 worklist를 덮어쓰지 않고
파생 `gate.json`만 갱신한다.

다음 상태는 모두 비점수·비Gold다.

- `pending_first_review`
- `pending_second_review`
- `consensus_requires_resolution`
- `disagreement_requires_adjudicator`
- `excluded`

12개가 모두 `adjudicated`이고 `gold_eligible=true`일 때만
`passed_two_reviewer_adjudication`이다. 그 전에는 `--require-gold-ready`가 exit code 2로
실패한다.

`run_claude_vision.py`는 selection, worklist, 파생 gate를 함께 재검증한다. Gate JSON만
`passed`로 바꾸어도 worklist가 pending이면 API client를 만들기 전에 실패한다. Gate가
통과하더라도 `schemas/adjudicated-gold.schema.json`을 만족하고 selection/worklist/gate hash와
12개 case·36개 source image hash에 결박된 별도 Gold manifest가 있어야 점수화를 시작한다.

## 무효화된 12-Case adjudicated Gold

2026-07-18에 서로 다른 두 **AI 시각 reviewer**가 hash로 고정된 12개 full triplet을
각각 검토했고 당시 도구는 12개 모두 `adjudicated`, `gold_eligible=true`로 판정했다.
후속 source-pixel 재감사에서 6개 Packing List의 B/L label에 원본 픽셀에는 없는 suffix가
포함된 사실을 확인했다. 두 reviewer의 합의는 정답성을 보장하지 않았으므로 이 Gold 전체와
이를 사용한 report를 `adjudicated_gold_invalidated_by_label_error`로 영구 무효화한다.
exact Gold hash뿐 아니라 output 경로, `gold_set_id`, JSON formatting, reviewer reference를
제외한 source-bound value semantics hash
`ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133`도 code-owned
invalidation으로 고정한다.

최초 `derived/diagnostic-selection/v1` worklist와 Gate는 12개
`pending_first_review`인 역사적 snapshot으로 그대로 보존한다. 현재 점수화에 사용하는
canonical 파생물은 별도 경로에 있다.

```text
_artifacts/datasets/trade_documents_images/source-v1/derived/
  independent-review-a/v1/annotations.jsonl
  independent-review-b/v1/annotations.jsonl
  adjudicated-gold/v1/
    adjudications.jsonl
    gate.json
    adjudicated-gold.json
    reconciliation-report.json
```

| 증거 | SHA-256 |
|---|---|
| reviewer A | `847dddfe7c7d59147c11997243293eb11b1e6d4093fa432d6c24ec0aa9fbf290` |
| reviewer B | `cf1e88964bc69e06085959b09490c491802609a68abb584d8019e7d920ecd38a` |
| canonical worklist | `74b9ba353520dc15834353023ef2c96e7399c655b63c75ff6cc097fa97874c70` |
| canonical Gate | `3c03dc8a592cf70551079606676e218fadbc77d5762c1ac70a627f16a24621b2` |
| 무효화된 adjudicated Gold | `75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6` |
| reconciliation report | `c9bdd9a8f18a52a44f885e5bcc900308430bec258a85aaa9f68f220d862b9c8b` |

과거 reconciliation의 source observation 일치/불일치 판정과 모든 성능 수치는 release 또는
diagnostic 점수로 재사용하지 않는다. 원본 이미지는 수정하지 않고, 새 Gold는 서로 독립된
사람 검수와 resolution을 완료한 뒤 새로운 hash로만 발행한다. 현재 runner와 builder는 위
무효 Gold를 경로 변경이나 재생성으로 되살릴 수 없도록 exact byte hash와 value semantics
hash 기준으로 차단한다.

0001·0041·0500의 과거 잘못된 suffix 주장을 무효화한 직접 시각 검수는 별도의 correction
provenance다. 위 hash-ranked 12 Case adjudication과 같은 표본이 아니며, 그 정정 이력만으로
모델 정확도나 데이터 전체 정상성을 주장하지 않는다.

## Deterministic rendered Gold와의 경계

`trade_document_triplets/source-v1/derived/rendered/v2`의 320문서는 source JSON에서
결정론적으로 생성한 PDF·200 DPI PNG·Claude PNG·layout Gold다. 고정 Vera font와 license
hash, renderer/layout schema v2를 사용한다. `rendered/v1`은 역사적 reference-only이며
evaluation-v2 또는 OCR release 입력으로 재사용하지 않는다. 이 경로는 renderer가 알고 있는
source field와 좌표를 평가하므로, 사람이 만든 3,000장 이미지 corpus의 adjudication 상태와
독립적이다.

- rendered Gold Claude 평가는 합성 renderer 회귀와 field/layout 추출 정확도만 주장한다.
- 과거 12-triplet adjudicated Gold는 파생 라벨 오류로 무효화됐으므로 현재 어떤 점수도
  주장하지 않는다. 독립 사람 검수 replacement가 별도 exact hash로 승인된 뒤에만 해당 JPEG
  template corpus의 제한된 field-extraction 평가를 재개할 수 있다.
- 어느 결과도 OCR provider, 실제 스캔 대표성, RAG, workflow outcome, Backend 저장, 시스템
  release를 자동으로 증명하지 않는다.
- 두 dataset의 score, 분모, artifact hash, report scope를 합치지 않는다.
