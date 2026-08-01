# Trade Documents Images — source-v1 descriptor

이 디렉터리는 이미지 원본이 아니라 사용자가 제공한 합성 통관 문서 데이터셋의 추적 가능한
descriptor다. 2026-07-18에 복원한 root 참고본은 canonical 등록본과 3,002개
파일의 상대경로·크기·SHA-256을 두 번 대조한 뒤 TASK-0028에서 retired됐다. Agent
runner가 읽는 유일한 canonical raw는 Git 밖의
`_local/data/trade_documents_images/source-v1/raw/`다. 실행 전에 DatasetProvenanceV2와
`_local/data/dataset-relocation-manifest.v1.json`을 검증하며, 폐기된 root·`_artifacts/datasets`
경로로의 fallback은 허용하지 않는다.

복원 원본과 canonical raw는 두 방식으로 대조했다. 전체 3,002개 파일의 상대경로·크기·SHA-256
비교 결과 mismatch는 0이며, 양쪽 모두 260,248,817 bytes다. 기존 3,001개
image+manifest checksum을 양쪽에 다시 적용한 결과도 누락·불일치가 0이고, checksum 목록에
포함되지 않은 `README.txt`는 별도 SHA-256으로 일치함을 확인했다.

## 판정

- 파일 무결성: 사용 가능
  - shipment 1,000개
  - Commercial Invoice, Packing List, Bill of Lading 각 1,000장
  - JPEG 3,000장 모두 decode 성공, 1000×1414, 중복 SHA-256 0
  - manifest 참조 누락·미참조 이미지 0
  - source 전체 3,002개 파일, 260,248,817 bytes
  - manifest SHA-256:
    `a1ebe0e18953b6712f044ee4d7dd3ede8ad426ae73ed3158dd0eab38e2111965`
  - source README SHA-256:
    `80edd388bd40c68b6c1c4cae6ddb6b9538502bc2a85d801707844ba1e6d591f0`
- 의미 정답: 제한적으로만 사용 가능
  - README가 언급하는 `shipments_master.json`이 제공되지 않았다.
  - OCR text, field bbox, table cell, 모든 필드 값, finding 정답이 없다.
  - 과거 4건에 기록된 suffix/footer 불일치 주장은 직접 시각 재검수에서 모두 무효화됐다.
  - 이는 데이터 전체가 clean임을 증명하지 않으며, 현재 확정된 source anomaly가 없다는 뜻이다.
- 대표성: synthetic clean-template baseline에 한정
  - 회전, 흐림, 그림자, crop, 도장, 필기, 다국어, 다양한 layout이 없다.
  - 실제 스캔 문서나 production 정확도를 대표하지 않는다.

따라서 3,000장 전체 데이터셋의 의미·field·bbox 상태는 현재 `candidate_unverified`다. 파일
입력 smoke와 비점수 진단에는 쓸 수 있지만, 검수되지 않은 나머지 corpus를 OCR/VLM/Agent
정확도 분모나 PASS 근거로 사용할 수 없다.

별도로 hash-ranked 12 Case/36 image 표본은 2026-07-18에 두 독립 AI 시각 reviewer의
annotation과 resolution을 거쳤지만, 이후 라벨 재감사에서 6개 Packing List의 expected
`bl_no`에 실제 pixel에는 없는 `EE`가 추가된 오류가 확인됐다. 해당 Case는 `0097`, `0311`,
`0092`, `0591`, `0372`, `0250`이다. 이는 source 문서의 의도된 anomaly나 discrepancy가
아니라 파생 Gold의 라벨 오류다.

따라서 이 12 Case Gold 전체는 `invalidated_by_label_error`이며 점수화할 수 없다. 현재
canonical/scoreable image Gold는 없다. 독립 사람 검수로 replacement Gold를 만들고 그 exact
hash를 코드에 명시적으로 pin하기 전까지 evaluator는 evaluation-asset preflight에서
OCR·Claude·Evidence 호출 전에 차단된다.

생성기·버전·seed와 공개 재배포 권리는 확인되지 않았다. 사용자가 지시한 이 프로젝트의
비공개 개발·평가에는 사용하되 raw를 Git/GitHub나 공개 포트폴리오에 포함하지 않는다. Claude
API를 이용한 비공개 평가 허용은 raw의 공개 재배포 권리를 뜻하지 않는다.

## 사용 규칙

1. canonical raw의 bytes와 파일명은 수정하지 않는다.
2. runner에는 canonical `_local/data/.../source-v1/raw`만 명시하고 폐기 경로 fallback을 금지한다.
3. 실행 전에 `dataset.lock.json`, Agent-owned relocation binding, workspace relocation
   manifest, DatasetProvenanceV2, 로컬 inventory/checksum을 모두 대조한다.
4. CI/PL/BL은 같은 shipment 단위로 split하여 case leakage를 금지한다.
5. manifest 값은 `manifest_observation`이며 사람이 확정한 field/layout gold가 아니다.
6. 모델 출력을 다시 gold로 저장하지 않는다.
7. full gold는 생성 원장 복구 또는 독립 annotation과 adjudication으로 별도 파생한다.
8. 검수되지 않은 anomaly 추정은 점수화하지 않고, 직접 재검수로 무효화된 라벨을 재사용하지 않는다.
9. retired root와 이전 `_artifacts/datasets` 등록본이 다시 생기면 검증을 실패시킨다.

## 초기 고정 표본

- Vision 입력 smoke(비점수 진단): `0500`
- item-count 1..8 annotation 후보:
  `0002`, `0004`, `0013`, `0001`, `0003`, `0012`, `0009`, `0031`
- 이전 anomaly 라벨 재사용 금지 회귀: `0001`, `0041`, `0500`, `1000`

표본 선택은 정답 확정을 뜻하지 않는다. 기존
`evals/trade_documents_images/v1/manifest_partial_gold.json`은 이름과 달리 현재
과거 label 오류로 artifact 전체가 `invalidated_by_label_error`인 단일 검수자 후보 기록이다.
`expected_*` 키와 release score 권한은 없으며 감사·비점수 debugging 외 사용을 금지한다.

대표 진단 표본 선택 자체는
`evals/trade_documents_images/v1/diagnostic-selection-v1.json`의 해시 순위 12개
triplet(36장)으로 고정한다. 최초
`derived/diagnostic-selection/v1` worklist/Gate는 12개 `pending_first_review`인 역사적
snapshot으로 유지한다. 로컬 canonical data의
`derived/{independent-review-a,independent-review-b,adjudicated-gold}/v1`도 감사용 역사
artifact일 뿐 현재 canonical Gold가 아니다.

두 reviewer는 독립 AI 시각 reviewer이며 사람 reviewer가 아니다. reviewer annotation SHA-256은
각각
`847dddfe7c7d59147c11997243293eb11b1e6d4093fa432d6c24ec0aa9fbf290`,
`cf1e88964bc69e06085959b09490c491802609a68abb584d8019e7d920ecd38a`다.
무효화된 과거 Gate와 Gold SHA-256은 각각
`3c03dc8a592cf70551079606676e218fadbc77d5762c1ac70a627f16a24621b2`,
`75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6`다. 같은 의미 내용을
경로·서식·Gold ID만 바꿔 재직렬화한 경우에도 semantic fingerprint
`ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133`으로 차단한다.
