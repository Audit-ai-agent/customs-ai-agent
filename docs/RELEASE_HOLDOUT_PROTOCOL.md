# Release Holdout Protocol

## 현재 판정

`evaluation-v2`는 release holdout이 아니다. 구성원이 저장소에 공개된 seed와 일반
builder/evaluator 입력에서 사전 도출 가능하므로 `release_holdout_v2_publicly_derivable`로
영구 무효화한다. 이미 알려질 수 있는 구성을 authorization receipt, 파일명 변경, 새 descriptor
또는 “sealed” 표기로 다시 unseen 상태로 만들 수 없다.

evaluation-v2의 hash-locked development 20-case 결과는 규칙 회귀를 확인하는
`CONTROLLED_SINGLE_MUTATION_DEVELOPMENT` 진단으로만 유지한다. 이 결과에는
`release_eligible=false`, `release_holdout_used=false`가 기록되며 release metric으로 승격할
수 없다.

다음 entry point는 evaluation-v2 입력을 파일 접근 전에 차단한다.

- `evals/run_release_holdout_validation.py`
- `tools/build_ocr_benchmark_descriptor.py`
- `tools/build_trade_triplet_eval_assets.py`의 전체 evaluation-v2 재생성 entry point
- `evals/combine_ocr_release_gate.py`의 공개 release combiner

따라서 현재 release holdout, release OCR descriptor, 선택된 OCR provider는 모두 없다.
구 evaluation-v2 보고서 결합 규칙은 역사 회귀를 위한 private non-release diagnostic으로만
남으며 `status=PASS` 또는 release-eligible report를 만들 수 없다.

## replacement evaluation-v3

향후 replacement는 사람이 저장소 밖에서 제공하는 sealed/unseen bundle이어야 한다. 저장소와
일반 개발 도구에 공개 가능한 것은 다음뿐이다.

- dataset ID와 version
- evaluation version
- sealed bundle SHA-256과 byte size
- case count
- 사람이 외부에서 공급했다는 provenance
- deterministic seed를 사용하지 않았다는 불변식
- ordinary development tooling이 사전 접근하지 않았다는 불변식

공개 manifest 계약은
`evals/release_holdout_protocol/v1/sealed-bundle-public-manifest.schema.json`이다. case ID,
membership, selection/rebalance/holdout seed, 원문, Gold 값은 공개 manifest에 넣을 수 없다.
public deterministic seed로 replacement를 생성하는 방식도 금지한다.

## 아직 구현하지 않은 경계

evaluation-v3 sealed bundle의 안전한 보관, 암호화/복호화, one-time human authorization,
access receipt, isolated evaluation runner와 결과 redaction은 아직 구현되지 않았다. 따라서
`evaluation-v3` 요청도 `release_holdout_v3_sealed_bundle_not_provisioned`로 fail-closed한다.
이 blocker가 해소되기 전에는 release holdout 결과나 OCR release Gate PASS를 생성할 수 없다.

sealed runner가 구현되면 descriptor 생성 C0 authorization commit과 descriptor hash를 고정하는
C1 literal-only pin commit을 분리해야 한다. C1 SHA를 descriptor에 넣지 않으며, exact parent와
허용 diff·receipt를 검증하는 절차는 `docs/OCR_RELEASE_FREEZE.md`를 따른다.
