# Diagnostic selection adjudication workflow

`diagnostic-selection-v1.json`의 12개 triplet은 대표성 진단 표본이지 그 자체로 Gold가 아니다.
각 Case는 서로 독립적인 두 reviewer가 같은 고정 image SHA-256을 확인해야 한다.

상태 전이는 다음과 같다.

```text
0 reviews → pending_first_review
1 independent review → pending_second_review
2 reviews, same conclusion → consensus_requires_resolution
consensus resolution recorded → adjudicated
2 reviews, different conclusions → disagreement_requires_adjudicator
서로 다른 제3의 adjudicator가 독립성 확인과 resolution 기록 → adjudicated
resolution concludes unreadable/insufficient → excluded
```

Gate는 다음을 모두 확인한다.

- 선택된 12개 Case가 정확히 한 번씩 존재한다.
- reviewer ID가 서로 다르고 독립 검수 확인이 `true`다.
- 검수한 image SHA-256이 selection manifest의 고정 triplet에 속한다.
- 두 결론이 같으면 명시적인 `two_reviewer_consensus` resolution을 기록하고, 다르면 별도
  reviewer ID와 겹치지 않는 adjudicator ID, 독립성 확인, resolution을 기록한다.
- 모든 Case가 `adjudicated`이고 `gold_eligible=true`다.

한 항목이라도 만족하지 않으면 해당 worklist의 전체 selection은
`blocked_pending_independent_reviews`이며 모델 정확도, Gold 통과 또는 Release Gate 근거로
사용할 수 없다.

기존 0001·0041·0500의 잘못된 suffix 주장을 무효화한 직접 시각 재검수는 별도의 correction
provenance다. 이 기록은 hash-ranked 12 Case adjudication과 같은 표본이 아니며 단독으로
field Gold나 모델 정확도 근거가 되지 않는다.

worklist JSONL은 독립 reviewer가 작성하는 검수 증거다. reviewer 종류가 사람인지 AI인지는
reconciliation report에 명시해야 하며 서로 바꾸어 주장하면 안 된다. 생성 도구는 파일이 없을
때만 초기화하며 기존 worklist를 덮어쓰지 않는다. 이후 실행은 기존 worklist를 읽어 파생
`gate.json`만 갱신한다. selection descriptor와 worklist를 분리함으로써 review 진행 중에도
선택된 36개 이미지와 순위 근거는 변하지 않는다.

## 2026-07-18 상태

최초
`_local/data/trade_documents_images/source-v1/derived/diagnostic-selection/v1`
worklist/Gate는 12개 `pending_first_review`인 역사적 snapshot으로 보존된다.

현재 canonical adjudication은 두 **독립 AI 시각 reviewer**의 별도 annotation을 reconciliation한
다음 경로다.

```text
_local/data/trade_documents_images/source-v1/derived/
  independent-review-a/v1/annotations.jsonl
  independent-review-b/v1/annotations.jsonl
  adjudicated-gold/v1/
    adjudications.jsonl
    gate.json
    adjudicated-gold.json
    reconciliation-report.json
```

당시 Gate는 12개 모두 `adjudicated`, `gold_eligible=true`,
`passed_two_reviewer_adjudication`으로 계산됐다. reviewer A/B annotation SHA-256은 각각
`847dddfe7c7d59147c11997243293eb11b1e6d4093fa432d6c24ec0aa9fbf290`,
`cf1e88964bc69e06085959b09490c491802609a68abb584d8019e7d920ecd38a`다.
Gate와 Gold SHA-256은 각각
`3c03dc8a592cf70551079606676e218fadbc77d5762c1ac70a627f16a24621b2`,
`75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6`다.

후속 source-pixel 재감사에서 6개 Packing List label 오류가 확인되어 위 Gold hash는
`adjudicated_gold_invalidated_by_label_error`로 영구 무효화됐다. 경로를 바꾸거나 같은
review 입력으로 재생성해도 runner와 builder가 exact byte hash와 source-bound value
semantics hash
`ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133` 기준으로 차단한다.
`gold_set_id`, JSON formatting, output/reference 경로를 바꾸는 것은 우회가 아니다. 독립 사람
검수를 완료한 새 Gold가 새 exact hash로 발행·승인되기 전에는 canonical adjudicated Gold가
없다.
