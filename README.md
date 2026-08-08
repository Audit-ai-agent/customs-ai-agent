# Customs AI Agent

문서 텍스트를 분석하고 Data Evidence API를 조회해, 담당자가 검토할 결과를 만드는 서비스입니다.
사용자·Case·업로드·최종 결정은 Backend가 소유합니다. Agent는 Data DB를 직접 읽지 않고 HTTP만 사용합니다.

이 저장소에는 두 실행 방식만 있습니다.

| 방식 | 목적 | 외부 입력 |
| --- | --- | --- |
| `local-development` | 팀의 기본 전체 개발 | 없음. Data의 합성 Evidence와 결정론 분석기를 사용합니다. |
| `functional-uat` | 실제 관세 3법 Evidence·Data·Claude 흐름을 한 번 확인 | 개인 Law API 권한, `ANTHROPIC_API_KEY`, Data의 로컬 모델 캐시 |

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
# 아래 세 값은 functional-uat-kr-trade 조합에서만 함께 설정합니다.
GUIDANCE_SERVICE_BASE_URL           # 별도 일반 통관 지식 snapshot의 loopback URL
GUIDANCE_SERVICE_TOKEN              # 그 snapshot이 만든 별도 호출 토큰
GUIDANCE_SERVICE_TOKEN_HEADER=Authorization
CUSTOMS_AGENT_SERVICE_MODE          # local-development 또는 functional-uat
```

`GUIDANCE_SERVICE_BASE_URL`과 `GUIDANCE_SERVICE_TOKEN`은 반드시 한 쌍으로만
설정합니다. 둘 다 없으면 일반 안내 조회도 기존 Evidence URL·토큰을 그대로 사용하므로
`dev-fixture`와 `functional-uat-3law`의 기존 단일 Data peer 동작은 바뀌지 않습니다.
하나만 있거나 loopback이 아닌 URL·잘못된 토큰·다른 mode header를 받으면 Agent는 일반
안내를 추측해 이어가지 않고 실패로 처리합니다. 문서 심사의 readiness·`/retrieve`는 어떤
경우에도 Evidence URL·토큰만 사용합니다.

기능 UAT에는 여기에 개인 `ANTHROPIC_API_KEY`를 추가합니다. 기본 입력 위치는 Git이 무시하는 workspace의 `../_local/secrets/.env`이며, 파일에는 아래 한 항목만 둡니다.

```text
ANTHROPIC_API_KEY=personal-key
```

두 Functional-UAT importer(`functional-uat-3law`, `functional-uat-kr-trade`)만 이 파일을 읽어 현재 Agent PowerShell 세션과 그 자식 프로세스에 값을 넣습니다. 빈 줄과 `#` 주석 외의 다른 키·중복 키·공백이 있는 값은 거부하며, 값은 출력하지 않습니다. 파일이 없을 때만 기존의 현재-세션 환경변수 주입을 호환용으로 허용합니다. `dev-fixture`와 legacy `../_local/secrets/uat/agent.env`는 이 파일을 읽지 않습니다. 법령 원문, DB dump, 모델 파일, 토큰을 저장소·Docker image·공유 ZIP에 넣지 않습니다.

Backend profile 초기화는 `../_local/profiles/<profile>/backend.env`에 `CUSTOMS_AGENT_SERVICE_BEARER_TOKEN`과 `CUSTOMS_AGENT_SERVICE_MODE`를 만듭니다. Data의 `dev-fixture` 첫 실행 또는 `functional-uat-3law init`은 Evidence 토큰을 `../_local/profiles/dev-fixture/evidence-service-token` 또는 `../_local/profiles/functional-uat-3law/evidence-service-token`에 만듭니다. 별도 일반 안내 snapshot은 `functional-uat-kr-trade init` 뒤 자신의 `../_local/profiles/functional-uat-kr-trade/evidence-service-token`을 만듭니다. 이름은 Data의 공통 서비스-token 파일명일 뿐, Agent에서는 guidance 전용으로만 사용합니다. Agent는 어떤 값을 복사하거나 생성하지 않습니다.

`tools/Import-AgentProfile.ps1`는 생성된 local 파일을 읽어 현재 PowerShell 세션의 Agent 환경변수에만 넣습니다. `functional-uat-3law`은 기존 Evidence peer 하나만, `functional-uat-kr-trade`는 **3법 Evidence peer + 별도 guidance peer**를 조합합니다. 후자는 Evidence 기본 포트 `18083`, guidance 기본 포트 `18093`을 사용하며 같은 포트는 거부합니다. 기존 profile을 다시 가져오면 남아 있던 guidance 환경변수도 지워 단일 peer fallback을 복구합니다. 두 Functional-UAT profile만 추가로 `../_local/secrets/.env`의 개인 Claude 키를 좁게 읽습니다. 토큰·원문·환경변수 값을 출력하지 않고, 파일 누락·profile/mode 불일치·형식 오류만 코드로 실패합니다. 반드시 점(`.`)으로 dot-source 하여 실행합니다.

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

사건과 독립된 **통관 지식 도우미**는 review run 디렉터리를 재사용하지 않습니다.
`functional-uat`에서는 항상
`_local/state/scenario-runs/knowledge/<conversation-id>/turn-<무작위-식별자>/`에 한 턴씩
기록합니다. `knowledge`는 Case·Review Run이 사용할 수 없는 예약 namespace이며, 이
디렉터리에는 Agent가 소유한 두 stream만 생깁니다.

- `agent-data.jsonl`: 한 번으로 제한된 공식 지식 Data 조회의 요청·응답 또는 안전한 실패 코드
- `claude.jsonl`: 실제 Claude가 호출된 경우에만 공식 자료 기반 prompt와 텍스트 응답 전문,
  또는 안전한 실패 코드

일반 `local-development` 서버는 지식 대화를 자동 기록하지 않습니다. 합성 fixture 시나리오를
명시적으로 재현할 때만 아래의 정확한 절대 경로를 설정합니다. 이 경로는 dev-fixture 실행기의
두 turn 검증과 같으며, 이전 `_local/state/knowledge-runs/` 기록은 옮기거나 지우지 않고
역사적 explicit-dev 기록으로만 보존합니다.

```powershell
$env:CUSTOMS_KNOWLEDGE_TRANSCRIPT_ROOT = [System.IO.Path]::GetFullPath(
  (Join-Path $PWD.Path "..\_local\state\scenario-runs\knowledge")
)
```

지식 transcript는 Case ID, Review Run, 문서 ID·파일명·본문, finding·정정 제안 구조를
재귀적으로 거부합니다. 경로·식별자·비밀 제거·쓰기 중 하나라도 실패하면 해당 명시적
시나리오 또는 Functional-UAT 대화는 성공으로 처리하지 않습니다. 이 기록도 Git,
`_artifacts`, PR, 공유 파일에 넣지 않습니다.

Functional-UAT 지식 대화가 끝난 뒤에는 Agent 전용 verifier로 해당 **한 턴**만 안전하게
확인합니다. verifier는 경로와 namespace, 필수 stream(`agent-data`, `claude`), JSONL 형식,
비밀 형태, Case·Review·문서 계열 필드를 fail-closed로 검사하며 transcript 본문·경로·prompt·
Claude 응답은 출력하지 않습니다.

```powershell
uv run --locked --no-sync --group dev customs-ai-agent-verify-knowledge-transcript `
  --service-mode functional-uat `
  --turn-directory ([System.IO.Path]::GetFullPath(
    "..\_local\state\scenario-runs\knowledge\<conversation-id>\turn-<turn-id>"
  ))
```

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

기본 개발에서는 Data의 `dev-fixture`를 먼저 실행하고 Agent를 시작합니다. 기본 Agent 포트는 Backend와 겹치지 않고 Windows의 예약 80xx 범위를 피하는 `127.0.0.1:18090`입니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile dev-fixture
uv run --locked --no-sync --group dev customs-ai-agent-dev-serve
```

기능 UAT에서는 Data가 `functional-uat-3law` 법령 수집·store build를 끝낸 뒤 같은 방식으로 Agent를 시작합니다. 먼저 `../_local/secrets/.env`에 새 개인 키를 한 줄로 넣고, importer를 dot-source 합니다. importer는 그 키를 현재 세션에만 주입하고 출력하지 않습니다. 넓은 공식 통관 지식용 `functional-uat-kr-trade`는 별도 공식 자료 snapshot build·검증과 `serve-guidance`가 완료된 뒤에만 **조합 profile**로 사용합니다. 그 전에는 기존 3법 UAT를 유지하며, 넓은 주제는 근거 부족으로 안전하게 답합니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-3law
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

넓은 일반 통관 안내를 함께 확인할 때에는 3법 Evidence 서비스를 계속 `18083`에 두고,
Data의 `functional-uat-kr-trade serve-guidance --port 18093`를 별도로 시작한 뒤 다음처럼
조합합니다. 이 guidance-only snapshot은 문서 심사 Evidence store를 대체하지 않습니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-kr-trade `
  -EvidencePort 18083 -GuidancePort 18093
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

Data profile을 같은 PC에서 병렬로 띄울 때에는 각 Data peer를 다른 loopback 포트로 실행하고, Agent importer에 정확한 포트를 지정합니다. 예를 들어 3법 Evidence를 `18183`, 별도 guidance snapshot을 `18193`으로 실행했다면 `-EvidencePort 18183 -GuidancePort 18193`을 사용합니다. 단일 peer profile에는 `-EvidencePort`만 필요합니다. 표준 전체 서비스 topology에서 Agent는 Backend가 가리키는 `18090`을 유지합니다.

모든 내부 요청은 `127.0.0.1`에만 바인딩되고, 요청·응답의 `X-Customs-Service-Mode`가 일치해야 합니다.

## HTTP 계약

Backend가 사용하는 경로는 아래 여섯 개뿐입니다.

```text
GET  /health/live
GET  /health/ready
GET  /internal/agent/v2/metadata
POST /internal/agent/v2/reviews
POST /internal/agent/v2/assistant/answers
POST /internal/agent/v2/knowledge/answers
```

`POST /internal/agent/v2/reviews`는 텍스트 문서, Case ID, 실행 ID, 기준일, 무역 유형, `KR`만 받습니다. 상업송장·포장명세서·B/L·AWB·원산지증명서·수출신고서·보완 증빙 문서를 받아 수입 또는 수출 심사를 수행합니다. V3.2 응답은 사람 검토 상태와 함께 실제 workflow 단계, 최대 2 cycle·6 action의 제한된 계획 trace, 문서별 추출 사실과 원문 인용, 코드 기반 비교 결과, Evidence에 연결된 finding·담당자 조치, 8구역 근거 보고서를 담습니다. 승인된 사람 수정은 원문을 덮어쓰지 않고 `human_provided` 사실로 별도 표시합니다. 수량·금액·원산지·문서 식별자 판정은 LLM이 아니라 코드가 수행하며, 결과는 언제나 자동완료 대상이 아닙니다.

`POST /internal/agent/v2/assistant/answers`는 Backend가 저장한 Review Run snapshot, 질문, 최근 검증 대화 최대 8턴만 받습니다. 답변은 현재 사건 기준과 공식 일반 통관 안내를 최대 두 구역으로 나눌 수 있으며, 사건 답변에는 사실·비교·Evidence 인용이, 일반 안내에는 공식 출처·기준일·snapshot 인용이 붙습니다. 이 호출은 Case, 문서, finding, 수정 제안을 직접 바꾸지 않습니다.

`POST /internal/agent/v2/knowledge/answers`는 사건·문서 원문을 받지 않는 별도 통관 지식 도우미 경로입니다. Backend가 대화 ID, 질문, 수출입 방향, 기준일, 최근 대화 최대 8턴만 전달합니다. Agent 코드는 허용된 공식 자료 레지스트리에서 질문당 한 번만 조회하고, coverage가 없거나 출처가 맞지 않으면 모델 지식으로 채우지 않습니다. 기본 profile은 Evidence peer 하나를 재사용하고, 조합 profile은 guidance URL·token으로만 이 호출을 보냅니다. Claude는 분류와 한국어 문장화에만 쓰며, 웹 검색·임의 URL·DB·파일 접근·네이티브 Tool Calling은 사용하지 않습니다.

패키지 안 `src/customs_ai_agent/contracts/schemas/backend/v3/`가 유일한 Agent HTTP 계약 원본입니다. Backend 소비자는 이 schema bytes를 자신의 저장소에 갱신해 사용합니다. Data Evidence 계약은 Data 저장소가 소유하며, Agent는 실행에 필요한 Data V3 schema bytes만 package에 두고 Harness가 sibling 바이트를 비교합니다.

## 확인

```powershell
uv run --locked --group dev ruff check src tests
uv run --locked --group dev mypy
uv run --locked --group dev pytest -q
```

실제 기능 UAT는 GitHub Actions에서 실행하지 않습니다. 개인 권한과 로컬 Data 입력이 준비된 사람이 명시적으로 실행하며, 요약 manifest와 모든 상세 non-secret transcript는 workspace `_local/state/scenario-runs/` 아래에만 남습니다. 독립 통관 지식 대화는 그 안의 예약 `knowledge/` namespace에 분리되고, 기존 `_local/state/knowledge-runs/`는 과거 explicit-dev 기록으로 보존됩니다. 기본 개발에서는 각각의 transcript를 명시적으로 켠 경우에만 기록합니다.
