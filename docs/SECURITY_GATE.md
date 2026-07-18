# AI Agent Safety Gate

이 문서는 `customs-ai-agent`의 CI Safety Gate를 로컬에서 같은 기준으로 재현하기 위한
운영 문서다. 이 Gate는 세 가지를 모두 통과해야 PASS이며, 도구 설치 실패·취약점 DB 조회 실패·
보고서 파싱 실패를 skip 또는 성공으로 바꾸지 않는다.

## 고정 도구와 판정 범위

| Gate | 도구 | 고정 기준 | 판정 대상 |
|---|---|---|---|
| Secret | Gitleaks 8.30.1 | Linux x64 archive SHA-256 `551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb` | exact commit을 포함한 전체 tracked Git history |
| Dependency | pip-audit 2.9.0 | `uv.lock`에서 `uv export --locked --all-groups`한 hash 포함 requirements | runtime·build·dev Python dependency |
| Python SAST | Ruff 0.15.21의 `S` rule family | 정확한 버전은 `uv.lock` 소유 | `src/customs_ai_agent`, `tools`, `evals` |

버전·다운로드 URL·체크섬의 단일 기준은
`security/toolchain.lock.toml`이다. CI는 Gitleaks archive를 SHA-256으로 확인한 뒤 실행하며,
`pip-audit`은 별도 격리 환경에 정확히 `2.9.0`을 설치한다. Ruff는 애플리케이션의 잠긴 개발
환경만 사용한다. SARIF 업로드와 `security-events: write`, OIDC 권한은 사용하지 않는다.
외부 GitHub Action도 이동 tag가 아니라 검토한 full commit SHA로 고정하고 checkout credential은
후속 step에 남기지 않는다.

## 로컬 재현

Windows에서 프로젝트의 기존 uv와 Gitleaks 증거 바이너리를 사용하는 예시는 다음과 같다.
경로는 로컬 설치 위치에 맞게 지정하되 `.env` 값이나 token을 명령 인자로 넣지 않는다.

```powershell
$UV = "..\..\..\_artifacts\tools\uv\v0.11.28\uv.exe"
$GITLEAKS = "..\..\..\_artifacts\tools\gitleaks\v8.30.1\bin\gitleaks.exe"

& $UV run --locked python tools/run_security_gate.py verify
& $UV run --locked python tools/run_security_gate.py secrets --gitleaks $GITLEAKS --scope history

New-Item -ItemType Directory -Force dist/security | Out-Null
& $UV export --locked --all-groups --no-emit-project --no-emit-local --no-sources `
  --format requirements.txt --output-file dist/security/locked-requirements.txt
& $UV run --isolated --no-project --with "pip-audit==2.9.0" python -I -X utf8 `
  tools/run_security_gate.py dependencies `
  --requirements dist/security/locked-requirements.txt

& $UV run --locked python tools/run_security_gate.py sast
```

Gitleaks 결과는 `--redact=100`으로 생성되며 wrapper는 secret 본문을 출력하지 않는다. 결과에는
rule ID, 저장소 상대 파일, 줄 번호만 표시한다. `gitleaks:allow`로 값을 숨기는 방식은 사용하지
않는다. Release Gate는 clean exact commit만 판정하므로 untracked build extraction이나 로컬
`.env`를 Gate 입력으로 삼지 않는다. 아직 commit되지 않은 변경의 보조 진단이 필요할 때만
명시적으로 `--scope worktree`를 사용할 수 있지만, 그 결과는 release evidence가 아니다.
`.gitleaks.toml`의 경로 allowlist는 Git에서 제외되는 로컬 virtual environment, cache, build
output만 대상으로 하며 source·test·fixture를 제외하지 않는다.

Dependency Gate는 PyPI advisory service를 조회하므로 네트워크가 필요하다. 조회 실패나 응답
파싱 실패는 FAIL이다. 애플리케이션 감사 입력은 `uv.lock`에서 hash와 함께 완전히 고정되지만,
격리된 감사 도구 자체의 전이 dependency는 `pip-audit==2.9.0` 설치 시점에 해석된다. 이것이 현재
알려진 재현성 한계다. 운영 release 전에는 감사 도구 전용 hash lock 또는 검증된 container digest로
이 경계도 고정해야 한다.

호출자는 `uv run --isolated`와 `python -I -X utf8`을 사용하고, wrapper는 version 확인과 실제
`pip-audit`을 모두 source checkout이 아닌 동일한 비어 있는 전용 temporary directory에서
동일한 isolated UTF-8 Python 모드로 실행한다. 이는 Windows의 legacy code page로 표현할 수
없는 저장소 경로에서 외부 감사 환경이나 자식 Python이 `site` import 전에 종료되는 문제를
막고, 동시에 project-local 파일이 version 확인 또는 감사 환경에 영향을 주지 못하게 한다.
temporary directory 생성·version 확인·감사 실행·JSON 파싱 중 하나라도 실패하면 Gate는
성공으로 간주하지 않는다.

Ruff `S`는 Bandit 계열 보안 규칙의 정적 검사로, 동적 taint 분석이나 악성 PDF/image fuzzing을
대체하지 않는다. `# noqa: S...`는 검증된 interpreter 경로·고정 argv·`shell=False`, 사전 URL
scheme/origin 검증, destructor/외부 SDK 호환 탐색처럼 해당 줄의 오탐 근거가 코드 바로 옆에
설명된 경우만 허용한다. 파일 단위 또는 전체 `S` 규칙 disable은 금지한다.

## Clean wheel 설치 출처 검증

CI는 Agent wheel을 위치 인자로 바로 설치하지 않는다. 먼저 wheel byte의 SHA-256을 계산하고,
그 hash가 포함된 단일 requirements 파일을 만든 뒤 `uv pip install --require-hashes`로 설치한다.
그 다음 verifier에 동일한 실제 wheel 파일과 caller expected SHA-256을 함께 전달한다.

uv 0.11.28은 hash-pinned local wheel을 정상 설치해도 PEP 610 `direct_url.json`의
`archive_info`에 hash를 기록하지 않는다. 따라서 hash 필드가 비어 있다는 사실만으로 설치를
허용하거나 거부하지 않는다. hash가 비어 있을 때도 다음 조건을 모두 만족해야 한다.

- 제공된 wheel archive 자체가 bounded inventory·metadata·entry point·RECORD 검사를 통과한다.
- wheel byte SHA-256이 caller expected SHA-256과 정확히 같다.
- `direct_url.json`의 local file URI가 제공된 exact wheel 경로를 가리킨다.
- wheel RECORD의 모든 package·metadata byte와 설치 RECORD 및 실제 설치 파일이 일치한다.
- 추가 RECORD 행은 pinned uv가 생성하는 제한된 provenance 파일과 정확히 하나의
  `customs-ai-agent` launcher만 허용하며, 각 파일의 hash와 size도 설치 RECORD와 일치한다.
- `INSTALLER`는 `uv`여야 하고 console entry point metadata도 canonical 값과 같아야 한다.

`direct_url.json`에 SHA-256이 있으면 그 값도 caller expected hash와 일치해야 한다. 다른 wheel
경로, editable/directory origin, 알 수 없는 설치 파일, 수정된 package byte, RECORD 또는 launcher
불일치는 모두 fail-closed한다. 이 검사는 신뢰할 수 없는 caller가 임의 expected hash를 정하는
문제를 해결하지 않으므로, Release Gate의 expected hash는 exact-commit artifact manifest가
소유해야 한다.

## 완료 기준과 한계

이 절은 판정 기준이다. workflow 파일이 존재하거나 로컬 일부 검사가 성공했다는 사실만으로
GitHub CI Safety Gate 또는 exact-commit Release Gate가 실행·봉인됐다고 간주하지 않는다.

- 세 명령의 exit code가 모두 0이다.
- Gitleaks tracked history scan은 `actions/checkout`의 `fetch-depth: 0`에서 수행된다.
- 취약점이나 새 SAST finding을 baseline으로 자동 승인하지 않는다.
- 도구 장애, timeout, advisory service 장애, invalid JSON은 모두 FAIL이다.
- CI 로그와 보고서에 API key, token, `.env` 값, raw scanner stderr를 출력하지 않는다.
- 이 Gate 통과는 Claude Live, OCR 60+20, Evidence API, clean wheel 및 사람 승인을 대체하지 않는다.
