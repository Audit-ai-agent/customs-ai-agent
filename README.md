# Customs AI Agent

문서 텍스트를 분석하고 Data Evidence API를 조회해, 담당자가 검토할 결과를 만드는 서비스입니다.
사용자·Case·업로드·최종 결정은 Backend가 소유합니다. Agent는 Data DB를 직접 읽지 않고 HTTP만 사용합니다.

이 저장소에는 두 실행 방식만 있습니다.

| 방식 | 목적 | 외부 입력 |
| --- | --- | --- |
| `local-development` | 팀의 기본 전체 개발 | 없음. Data의 합성 Evidence와 결정론 분석기를 사용합니다. |
| `functional-uat` | 실제 관세 3법·Data·Claude 흐름을 한 번 확인 | 개인 Law API 권한, `ANTHROPIC_API_KEY`, Data의 로컬 모델 캐시 |

두 방식 모두 텍스트만 처리하고 결과를 항상 `human_review_required`로 반환합니다. 자동완료나 배포 판단 기능은 없습니다.

## 설치

기본 개발에는 Claude SDK가 설치되지 않습니다.

```powershell
uv sync --locked --group dev
```

실제 기능 UAT를 실행하는 사람만 Claude SDK를 추가합니다.

```powershell
uv sync --locked --group dev --extra functional-uat
```

## 로컬 입력

비밀과 로컬 상태는 Git이 아닌 workspace의 `_local/`에 둡니다. `dev-fixture`에는 팀 공용 비밀도 전달물도 없습니다. 각 팀원이 자신의 PC에서 Backend와 Data 초기화를 실행해 두 개의 loopback 토큰을 **로컬 생성**하며, 이 토큰은 팀원끼리 전달·복사·공유하지 않습니다.

Agent가 필요한 값은 다음뿐입니다. 이 값들은 화면에 복사·출력하지 않습니다.

```text
CUSTOMS_AGENT_SERVICE_BEARER_TOKEN  # Backend가 만든 Agent 호출 토큰
EVIDENCE_SERVICE_BASE_URL           # Data의 loopback URL
EVIDENCE_SERVICE_TOKEN              # Data가 만든 Evidence 호출 토큰
EVIDENCE_SERVICE_TOKEN_HEADER=Authorization
CUSTOMS_AGENT_SERVICE_MODE          # local-development 또는 functional-uat
```

기능 UAT에는 여기에 개인 `ANTHROPIC_API_KEY`를 추가합니다. 기본 입력 위치는 Git이 무시하는 workspace의 `../_local/secrets/.env`이며, 파일에는 아래 한 항목만 둡니다.

```text
ANTHROPIC_API_KEY=personal-key
```

`functional-uat-3law` importer만 이 파일을 읽어 현재 Agent PowerShell 세션과 그 자식 프로세스에 값을 넣습니다. 빈 줄과 `#` 주석 외의 다른 키·중복 키·공백이 있는 값은 거부하며, 값은 출력하지 않습니다. 파일이 없을 때만 기존의 현재-세션 환경변수 주입을 호환용으로 허용합니다. `dev-fixture`와 legacy `../_local/secrets/uat/agent.env`는 이 파일을 읽지 않습니다. 법령 원문, DB dump, 모델 파일, 토큰을 저장소·Docker image·공유 ZIP에 넣지 않습니다.

Backend profile 초기화는 `../_local/profiles/<profile>/backend.env`에 `CUSTOMS_AGENT_SERVICE_BEARER_TOKEN`과 `CUSTOMS_AGENT_SERVICE_MODE`를 만듭니다. Data의 `dev-fixture` 첫 실행 또는 `functional-uat-3law init`은 Evidence 토큰을 `../_local/profiles/dev-fixture/evidence-service-token` 또는 `../_local/profiles/functional-uat-3law/evidence-service-token`에 만듭니다. Agent는 값을 복사하거나 생성하지 않습니다.

`tools/Import-AgentProfile.ps1`는 위의 두 **생성된** 파일을 읽어 현재 PowerShell 세션의 Agent 환경변수에 넣습니다. `functional-uat-3law`일 때만 추가로 `../_local/secrets/.env`의 개인 Claude 키를 좁게 읽습니다. 토큰·원문·환경변수 값을 출력하지 않고, 파일 누락·profile/mode 불일치·형식 오류만 코드로 실패합니다. 반드시 점(`.`)으로 dot-source 하여 실행합니다.

## 상세 시나리오 transcript

일반 Agent 로그와 CI는 문서 본문·prompt·응답 본문을 남기지 않습니다.
`functional-uat`은 시작 시 고정된 workspace 경로
`../_local/state/scenario-runs/`을 자동 생성·검증하고, 각 실제 review run의 상세
transcript를 반드시 남깁니다. 이 경로를 다른 값으로 덮어쓰거나 생성·기록·비밀 제거에
실패하면 UAT는 불완전한 것으로 실패합니다.

기본 `local-development`는 계속 경량입니다. 사람이 명시적 시나리오 실행기를 재검토할
때만 Agent를 시작하기 **전에** 아래의 절대 경로를 설정해 같은 형식의 기록을 켤 수
있습니다.

```powershell
$env:CUSTOMS_SCENARIO_TRANSCRIPT_ROOT = [System.IO.Path]::GetFullPath(
  (Join-Path $PWD.Path "..\_local\state\scenario-runs")
)
```

상세 transcript는 `_local/state/scenario-runs/<review-run-id>/`에 다음의 실제 non-secret
body를 기록합니다.

- `agent-data.jsonl`: Agent → Data readiness/retrieve 요청과 응답
- `claude.jsonl`: `functional-uat`에서 실제 호출한 system/user prompt와 Claude 텍스트
  응답 전문, 또는 안전한 실패 코드. `local-development`에서는 Claude를 호출하지 않았다는
  항목만 기록

`backend-agent.jsonl`과 `public-api.jsonl`은 Backend가 같은 run 디렉터리에 기록합니다.
Agent는 `agent-data.jsonl`과 `claude.jsonl`만 소유합니다.

HTTP Authorization/Cookie 헤더, 세션·서비스 token, API key, 비밀번호, DSN, 환경변수는
기록하지 않으며 credential-shaped 문자열도 `[REDACTED]` 처리합니다. transcript는
현재 PC의 `_local`에만 두고 Git, `_artifacts`, 채팅, PR, 공유 압축 파일에는 넣지 않습니다.
공개 API 사용자 여정과 합성 문서 본문은 Backend의 상세 시나리오 실행기가 같은 run
디렉터리에 기록합니다.

## 실행 순서

전체 서비스는 sibling 저장소를 같은 workspace에 둔 뒤 다음 순서로 준비·실행합니다.

```text
준비(서버 시작 전): Backend profile init → Data profile init
실행: Data → Agent → Backend → Frontend
```

여기서 Backend profile init은 Agent용 bearer token을 로컬에 만들기 위한 준비 단계일 뿐 Backend 서버를 먼저 띄우는 뜻이 아닙니다. Data `dev-fixture`는 첫 실행 때 토큰을 생성하므로, 실행 순서의 첫 단계가 끝난 뒤 Agent importer를 실행할 수 있습니다.

기본 개발에서는 Data의 `dev-fixture`를 먼저 실행하고 Agent를 시작합니다. 기본 Agent 포트는 Backend와 겹치지 않는 `127.0.0.1:8090`입니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile dev-fixture
uv run --locked --no-sync --group dev customs-ai-agent-dev-serve
```

기능 UAT에서는 Data가 `functional-uat-3law` 법령 수집·store build를 끝낸 뒤 같은 방식으로 Agent를 시작합니다. 먼저 `../_local/secrets/.env`에 새 개인 키를 한 줄로 넣고, importer를 dot-source 합니다. importer는 그 키를 현재 세션에만 주입하고 출력하지 않습니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-3law
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

Data profile을 같은 PC에서 병렬로 띄울 때에는 Data를 다른 loopback 포트로 실행하고, Agent importer에도 같은 포트를 지정합니다. 예를 들어 UAT Data를 `18183`으로 실행했다면 `-EvidencePort 18183`을 추가합니다. 표준 전체 서비스 topology에서 Agent는 Backend가 가리키는 `8090`을 유지합니다.

모든 내부 요청은 `127.0.0.1`에만 바인딩되고, 요청·응답의 `X-Customs-Service-Mode`가 일치해야 합니다.

## HTTP 계약

Backend가 사용하는 경로는 아래 네 개뿐입니다.

```text
GET  /health/live
GET  /health/ready
GET  /internal/agent/v2/metadata
POST /internal/agent/v2/reviews
```

`POST /internal/agent/v2/reviews`는 텍스트 문서, Case ID, 실행 ID, 기준일, 무역 유형, `KR`만 받습니다. 응답은 사람 검토 상태, 간단한 요약, Data Evidence, 확인 항목, 결과 hash를 담습니다.

패키지 안 `src/customs_ai_agent/contracts/schemas/backend/v3/`가 유일한 Agent HTTP 계약 원본입니다. Backend 소비자는 이 schema bytes를 자신의 저장소에 갱신해 사용합니다. Data Evidence 계약은 Data 저장소가 소유하며, Agent는 실행에 필요한 Data V3 schema bytes만 package에 두고 Harness가 sibling 바이트를 비교합니다.

## 확인

```powershell
uv run --locked --group dev ruff check src tests
uv run --locked --group dev mypy
uv run --locked --group dev pytest -q
```

실제 기능 UAT는 GitHub Actions에서 실행하지 않습니다. 개인 권한과 로컬 Data 입력이 준비된 사람이 명시적으로 실행하며, 요약 manifest는 workspace `_artifacts/functional-uat/`에 남고 상세 non-secret body는 항상 별도의 `_local/state/scenario-runs/`에 남습니다. 기본 개발에서는 명시적으로 transcript를 켠 경우에만 같은 경로에 남습니다.
