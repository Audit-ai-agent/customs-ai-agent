# OCR Release Two-Commit Freeze

## 목적

OCR release descriptor의 SHA-256을 그 descriptor를 만든 동일 commit에 다시 넣으면 commit
hash가 바뀌어 순환 참조가 생긴다. 이 문서는 그 순환을 만들지 않으면서 descriptor, 사람 승인,
실행 코드와 OCR runtime을 하나의 검증 가능한 release 경계로 묶는 절차를 정의한다.

현재 `evaluation-v2`는 구성원이 공개적으로 도출 가능해 영구 무효이고 evaluation-v3 sealed
runner도 아직 없다. 따라서 아래 절차는 아직 실행할 수 있는 release 절차가 아니라
evaluation-v3가 구현된 뒤 적용할 freeze 계약이다. 현재 두 code-owned pin은 모두 `None`이며
runtime과 benchmark는 provider 생성 전에 `NOT_FROZEN`으로 중단한다. 공개 OCR report
combiner도 evaluation-v2 입력 파일을 읽기 전에 영구 무효 코드로 중단한다.

## C0: authorization commit

C0는 holdout 접근 전에 사람이 승인하는 exact clean commit이다.

- C0에는 최종 행위 코드, threshold, schema, prompt, OCR 환경·model lock과
  `ocr_release_pin.py`의 `None`/`None` 상태가 포함된다.
- 사람 authorization, one-time access receipt와 OCR descriptor는 C0에 결박한다.
- descriptor의 `release_context.authorization_commit_sha`는 C0를 기록한다.
- evaluation-v3 receipt는 sealed bundle SHA-256, C0, authorization SHA-256을 결박한다.
- descriptor나 source에 아직 존재하지 않는 C1 SHA를 넣지 않는다.

## C1: literal-only pin commit

C1은 C0의 단일 parent child다. 허용되는 변경은
`src/customs_ai_agent/ocr_release_pin.py`의 다음 두 literal을 `None`에서 실제 값으로 바꾸는
것뿐이다.

```python
CANONICAL_OCR_DESCRIPTOR_SHA256: str | None = "<DESCRIPTOR_SHA256>"
CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA: str | None = "<C0_SHA>"
```

파일 추가, 주석 변경, import 변경, formatting 변경, 다른 코드 변경은 같은 C1에 넣지 않는다.
검증기는 C0와 C1의 Git blob을 canonical source와 byte 단위로 비교하므로 “같은 의미”의 다른
코드도 허용하지 않는다.

```powershell
uv run --locked python tools/verify_ocr_release_freeze.py `
  --repo-root . `
  --descriptor "<SEALED_EVALUATION_V3_OCR_DESCRIPTOR>" `
  --authorization-receipt "<EVALUATION_V3_CONSUMPTION_RECEIPT>"
```

검증기는 다음을 모두 요구한다.

- 현재 worktree가 clean C1
- C1 parent가 정확히 하나이고 그 parent가 pin에 기록한 C0
- C0→C1 변경 경로가 literal-only pin module 하나
- C0 pin source가 `None`/`None`, C1 pin source가 descriptor SHA/C0 SHA
- descriptor 실제 SHA와 code-owned descriptor pin 일치
- descriptor의 `authorization_commit_sha`와 code-owned C0 pin 일치
- evaluation-v3 receipt의 실제 hash, C0, sealed bundle hash, registry key 일치
- descriptor와 source에 C1 SHA가 들어 있지 않음

## benchmark와 runtime

OCR benchmark는 wheel/environment/model attestation이나 provider 생성보다 먼저 위 source
freeze 검증을 수행한다. 제품 runtime은 Git 저장소를 전제로 하지 않으므로 code-owned
descriptor SHA와 C0 SHA를 descriptor의 release context와 대조하고, 이후 combined report의
exact wheel hash를 설치된 wheel과 다시 대조한다.

C1 뒤 C2에서 행위 코드가 바뀌면 C2의 parent는 C0가 아니므로 source freeze가 무효다. 기존
combined report도 C1 wheel hash에 결박되어 C2 wheel에서 사용할 수 없다. 변경을 release하려면
새 C0 authorization과 새 C1 pin 절차를 다시 수행해야 한다.

## 금지 사항

- 무효인 evaluation-v2 descriptor를 C1에 pin하지 않는다.
- C0와 C1 사이에 다른 변경을 섞지 않는다.
- C1 SHA를 descriptor나 pin source에 기록하지 않는다.
- dirty worktree, merge commit, 기존 receipt 재사용 상태에서 freeze를 승인하지 않는다.
- 두 pin 중 하나만 채운 상태를 release로 취급하지 않는다.
