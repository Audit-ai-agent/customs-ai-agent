# Customs AI Agent

문서 텍스트를 분석하고 Data Evidence API를 조회해, 담당자가 검토할 결과를 만드는 서비스입니다.
사용자·Case·업로드·최종 결정은 Backend가 소유합니다. Agent는 Data DB를 직접 읽지 않고 HTTP만 사용합니다.

이 저장소에는 두 실행 방식만 있습니다.

| 방식 | 목적 | 외부 입력 |
| --- | --- | --- |
| `local-development` | 팀의 기본 전체 개발 | 없음. Data의 합성 Evidence와 결정론 분석기를 사용합니다. |
| `functional-uat` | 실제 법령 Evidence·Data·Claude 흐름을 한 번 확인 | 개인 Law API 권한, `ANTHROPIC_API_KEY`, Data의 로컬 모델 캐시 |

두 방식의 **심사**는 텍스트만 처리하고 결과를 항상 `human_review_required`로 반환합니다. 별도 문서 미리보기 경로는 PDF/JPEG/PNG의 사람 검수용 추출 후보만 만들며 기본적으로 비활성화되어 있습니다. 미리보기 결과는 사람이 확인해 텍스트로 확정하기 전까지 심사·Finding·통관 결론에 사용되지 않습니다. 자동완료나 배포 판단 기능은 없습니다.

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
# 아래 네 값은 versioned functional-UAT guidance 조합에서만 함께 설정합니다.
GUIDANCE_SERVICE_BASE_URL           # 별도 일반 통관 지식 snapshot의 loopback URL
GUIDANCE_SERVICE_TOKEN              # 그 snapshot이 만든 별도 호출 토큰
GUIDANCE_SERVICE_TOKEN_HEADER=Authorization
GUIDANCE_SERVICE_REGISTRY_VERSION   # 1.0 또는 2.0
CUSTOMS_AGENT_SERVICE_MODE          # local-development 또는 functional-uat
CUSTOMS_AGENT_EVIDENCE_RETRIEVE_TIMEOUT_SECONDS  # Evidence /retrieve POST 전용, 기본 90초
```

`GUIDANCE_SERVICE_BASE_URL`·`GUIDANCE_SERVICE_TOKEN`·`GUIDANCE_SERVICE_REGISTRY_VERSION`은
반드시 함께 설정합니다. `dev-fixture`만 guidance peer가 없을 때 Evidence peer를 호환용으로
재사용합니다. Functional-UAT에서는 versioned guidance profile이 없으면 일반 안내 조회를
시도하지 않습니다. V2는 별도 peer와 `2.0`을 명시해야 하므로 V2 scope 요청이 V1 endpoint로
조용히 전송되지 않습니다. URL·token·version 중 하나가 빠지거나 loopback이 아닌 URL,
잘못된 token, 다른 mode header, registry version/hash 불일치가 있으면 Agent는 일반 안내를
추측해 이어가지 않고 coverage 부족으로 처리합니다. 문서 심사의 readiness·`/retrieve`는
어떤 경우에도 Evidence URL·token만 사용합니다. V2 대화의 첫 성공 응답에서 나온 immutable
snapshot ID는 Backend가 대화별로 보존하고, 후속 요청에만 `expected_snapshot_id`로 Agent에
전달합니다. Agent는 이 값을 V2 Data 요청에만 그대로 보내며, Data가 같은 ID를 다시 선택하지
못하거나 다른 ID를 응답하면 active snapshot으로 대체하지 않고 안전하게 coverage 부족으로
끝냅니다. V1은 이 값을 받으면 조회 전에 거부합니다.

`CUSTOMS_AGENT_EVIDENCE_RETRIEVE_TIMEOUT_SECONDS`는 생략하면 90초이며 1~120초만 허용합니다.
이는 초기화 중인 Evidence 모델의 `/retrieve` POST에만 적용됩니다. readiness GET과 guidance POST는
기존 5초 제한을 유지합니다. 기본 importer 수정은 필요 없고, 필요할 때만 현재 PowerShell 세션에서
비밀이 아닌 이 값을 명시해 조정할 수 있습니다.

기능 UAT에는 여기에 개인 `ANTHROPIC_API_KEY`를 추가합니다. 기본 입력 위치는 Git이 무시하는 workspace의 `../_local/secrets/.env`이며, 파일에는 아래 한 항목만 둡니다.

```text
ANTHROPIC_API_KEY=personal-key
```

다섯 Functional-UAT importer(`functional-uat-3law`, `functional-uat-law-history`,
`functional-uat-kr-trade`, `functional-uat-kr-trade-v2`,
`functional-uat-kr-trade-v2-law-history`)만 이 파일을 읽어 현재 Agent PowerShell 세션과 그 자식
프로세스에 값을 넣습니다. 빈 줄과 `#` 주석 외의 다른 키·중복 키·공백이 있는 값은 거부하며,
값은 출력하지 않습니다. 파일이 없을 때만 기존의 현재-세션 환경변수 주입을 호환용으로
허용합니다. `dev-fixture`와 legacy `../_local/secrets/uat/agent.env`는 이 파일을 읽지
않습니다. 법령 원문, DB dump, 모델 파일, 토큰을 저장소·Docker image·공유 ZIP에 넣지
않습니다.

Backend profile 초기화는 `../_local/profiles/<profile>/backend.env`에 `CUSTOMS_AGENT_SERVICE_BEARER_TOKEN`과 `CUSTOMS_AGENT_SERVICE_MODE`를 만듭니다. Data의 `dev-fixture` 첫 실행 또는 `functional-uat-3law`·`functional-uat-law-history` 초기화는 각각의 Evidence token을 같은 profile-local `evidence-service-token`에 만듭니다. V1과 V2 일반 안내는 Data가 공유하는 `../_local/profiles/functional-uat-kr-trade/evidence-service-token`을 사용합니다. `functional-uat-kr-trade-v2`와 `functional-uat-kr-trade-v2-law-history`는 Agent의 명시적 runtime/port/registry 선택 이름이며 별도 guidance token 디렉터리가 아닙니다. 이름은 Data의 공통 service-token 파일명일 뿐, Agent에서는 guidance 전용으로만 사용합니다. Agent는 어떤 값을 복사하거나 생성하지 않습니다.

`tools/Import-AgentProfile.ps1`는 생성된 local 파일을 읽어 현재 PowerShell 세션의 Agent 환경변수에만 넣습니다. `functional-uat-3law`은 보존된 3법 Evidence peer, `functional-uat-law-history`는 새 65법령 이력 Evidence peer를 사용합니다. `functional-uat-kr-trade`는 **3법 Evidence + V1 guidance(18093)**, `functional-uat-kr-trade-v2`는 **3법 Evidence + readiness-gated V2 guidance(18094)**, `functional-uat-kr-trade-v2-law-history`는 **65법령 이력 Evidence + V2 guidance(18094)**를 조합한 기본 UAT 선택지입니다. Importer는 Data token과 함께 비밀이 아닌 `CUSTOMS_AGENT_EVIDENCE_PROFILE`도 설정합니다. 이 값이 `functional-uat-law-history`이면 Agent는 실제 `/retrieve` 응답에 세 날짜가 모두 든 `temporal_context`, 심사 기준일과 같은 `law_as_of`, 그리고 법령·HSK·요건별 `covered|unavailable`를 모두 담은 `temporal_coverage`를 함께 요구합니다. 두 필드는 원자적으로 검증되며 하나라도 없거나 형식이 다르면 이력 Evidence를 사용하지 않습니다. 보존된 3법·개발 profile은 null/누락 문맥을 계속 읽을 수 있습니다. V1은 scope 계약이 없으므로 Agent가 V2 분류 결과를 명시적 legacy topic으로 축소하고 `scope_hint` 키를 보내지 않습니다. V2만 정확한 `topic + scope_hint`를 보냅니다. V2 importer는 `GUIDANCE_SERVICE_REGISTRY_VERSION=2.0`을 함께 설정합니다. 기존 profile을 다시 가져오면 남아 있던 guidance 환경변수도 지워 단일 peer fallback을 복구합니다. 다섯 Functional-UAT profile만 추가로 `../_local/secrets/.env`의 개인 Claude 키를 좁게 읽습니다. token·원문·환경변수 값을 출력하지 않고, 파일 누락·profile/mode 불일치·형식 오류만 코드로 실패합니다. 반드시 점(`.`)으로 dot-source 하여 실행합니다.

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

기능 UAT의 기본 경로는 Data가 `functional-uat-law-history` 65법령 기준 snapshot·store build를 끝낸 뒤 Agent를 시작하는 것입니다. Agent는 Data가 응답한 법령·HSK·요건 snapshot 기준일을 `temporal_context`로 그대로 보존하며, 법령 기준일이 현재 심사 기준일과 다르면 응답을 거부합니다. 동시에 `temporal_coverage`의 법령·HSK·요건 상태를 보존하므로 archive가 없는 HSK/요건을 법령 이력으로 보완한 것처럼 표시하지 않습니다. 기존 `functional-uat-3law`은 호환 UAT로 남습니다. V1 일반 안내 `functional-uat-kr-trade`와 V2 공식 페이지 snapshot은 공유 Data local state에서 각각 build·verify·guidance service가 완료된 뒤에만 명시적 조합 profile로 사용합니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-law-history
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

기존 V1 일반 통관 안내를 함께 확인할 때에는 3법 Evidence 서비스를 계속 `18083`에 두고,
Data의 `functional-uat-kr-trade serve-guidance --port 18093`를 별도로 시작한 뒤 다음처럼
조합합니다. 이 guidance-only snapshot은 문서 심사 Evidence store를 대체하지 않습니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-kr-trade `
  -EvidencePort 18083 -GuidancePort 18093
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

V2 공식 페이지 snapshot은 Data의 공유 `functional-uat-kr-trade` local state에서 `build-v2`, `verify-v2`,
`serve-guidance-v2 --port 18094`가 모두 성공한 뒤에만 사용합니다. Agent는 18094의
authenticated readiness가 `ready`이고 query 응답의 registry version·hash·snapshot·기준일이
copied V2 manifest와 정확히 일치할 때만 공식 안내로 채택합니다. 그렇지 않으면 일반 모델
지식으로 메우지 않고 coverage 부족으로 답합니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-kr-trade-v2 `
  -EvidencePort 18083 -GuidancePort 18094
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

새 기본 조합 UAT는 위 V2 안내 peer를 유지하면서 Evidence만 법령 이력 profile로 바꿉니다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile functional-uat-kr-trade-v2-law-history `
  -EvidencePort 18083 -GuidancePort 18094
uv run --locked --no-sync --group dev --extra functional-uat customs-ai-agent-functional-uat-serve
```

Data profile을 같은 PC에서 병렬로 띄울 때에는 각 Data peer를 다른 loopback 포트로 실행하고, Agent importer에 정확한 포트를 지정합니다. 예를 들어 law-history Evidence를 `18183`, V2 guidance snapshot을 `18194`으로 실행했다면 `-EvidencePort 18183 -GuidancePort 18194`를 `functional-uat-kr-trade-v2-law-history`와 함께 사용합니다. 단일 peer profile에는 `-EvidencePort`만 필요합니다. 표준 전체 서비스 topology에서 Agent는 Backend가 가리키는 `18090`을 유지합니다.

모든 내부 요청은 `127.0.0.1`에만 바인딩되고, 요청·응답의 `X-Customs-Service-Mode`가 일치해야 합니다.

## HTTP 계약

Backend가 사용하는 경로는 아래 일곱 개뿐입니다.

```text
GET  /health/live
GET  /health/ready
GET  /internal/agent/v2/metadata
POST /internal/agent/v2/reviews
POST /internal/agent/v2/document-extractions
POST /internal/agent/v2/assistant/answers
POST /internal/agent/v2/knowledge/answers
```

`POST /internal/agent/v2/reviews`는 텍스트 문서, Case ID, 실행 ID, 기준일, 무역 유형, `KR`만 받습니다. 상업송장·포장명세서·B/L·AWB·원산지증명서·수출신고서·보완 증빙 문서를 받아 수입 또는 수출 심사를 수행합니다. V3.2 응답은 사람 검토 상태와 함께 실제 workflow 단계, 최대 2 cycle·6 action의 제한된 계획 trace, 문서별 추출 사실과 원문 인용, 코드 기반 비교 결과, Evidence에 연결된 finding·담당자 조치, 8구역 근거 보고서를 담습니다. 이력 Evidence가 사용된 경우에는 법령·HSK·요건의 기준일을 담은 nullable `temporal_context`와 각 영역의 `covered|unavailable` 상태를 담은 nullable `temporal_coverage`가 함께 추가됩니다. 승인된 사람 수정은 원문을 덮어쓰지 않고 `human_provided` 사실로 별도 표시합니다. 수량·금액·원산지·문서 식별자 판정은 LLM이 아니라 코드가 수행하며, 결과는 언제나 자동완료 대상이 아닙니다.

`POST /internal/agent/v2/document-extractions`는 `/reviews`와 분리된 binary boundary입니다. `Content-Type`은 `application/pdf`·`image/jpeg`·`image/png` 중 하나이고, body에는 identity-encoded raw bytes만 담습니다. Backend는 `X-Document-Asset-ID`, `X-Extraction-Request-ID`, `X-Document-SHA256`을 함께 보내며 Agent는 size·streaming SHA-256·magic·PDF page hint·image pixel limit을 독립 확인합니다. 응답에는 asset/request ID, hash, adapter revision, page/span/normalized bbox, canonical text hash와 content-free warning만 있으며 raw bytes·base64·filename·Case ID는 없습니다. span text는 검수 화면 표시용 후보일 뿐이며 Agent의 일반 로그·scenario transcript·review contract에는 들어가지 않습니다. 기본 preview engine은 `unavailable`입니다. 일반 경로는 사람이 승인한 pointer SHA-256을 process 시작에 고정하며, 합성 source-attested PDF 전용의 명시 opt-in만 sealed chain을 다음 PDF 요청에 자동 반영할 수 있습니다. JPEG·PNG는 어느 경로에서도 계속 `unavailable`입니다.

`POST /internal/agent/v2/assistant/answers`는 Backend가 저장한 Review Run snapshot, 질문, 최근 검증 대화 최대 8턴과 선택적 `expected_snapshot_id`만 받습니다. 기존 단일 일반 안내는 그대로 `guidance_resolution` 하나를 사용합니다. V2에서만 방향이 같은 독립 scope 정확히 두 개를 순차 조회할 수 있으며, 첫 응답의 immutable snapshot으로 두 번째 조회를 고정합니다. 두 scope가 모두 `covered`이고 사건의 독립 citation도 있을 때만 `guidance_resolutions` 정확히 두 개, `answer_format="multi_scope_checklist"`, `사건 분석 → 주제 1 공식 안내 → 주제 2 공식 안내 → 사건 확인 목록`의 네 구역을 반환합니다. 각 공식 인용·안내 구역에는 `resolution_index`가 붙고, 마지막 확인 목록은 법적 결론·HS·세액·허가·통관 승인 판단을 만들지 않습니다. 한 scope라도 partial/no coverage/계약 오류이면 인용이나 사건 적용을 섞지 않고 항목화된 `insufficient_evidence`로 끝냅니다. 세 개 이상이거나 방향이 충돌하면 기존 명확화 질문을 반환합니다. 문서 본문처럼 보이는 다줄 붙여넣기는 새 심사 근거가 아니며 Agent·Data·Claude·transcript에 전달하지 않고 문서 등록 또는 교체 후 재심사를 안내합니다. 이 호출은 Case, 문서, finding, 수정 제안을 직접 바꾸지 않습니다.

`POST /internal/agent/v2/knowledge/answers`는 사건·문서 원문을 받지 않는 별도 통관 지식 도우미 경로입니다. Backend가 대화 ID, 질문, 수출입 방향, 기준일, 최근 대화 최대 8턴과 필요할 때의 V2 `expected_snapshot_id`만 전달합니다. Agent는 V2에서 방향이 호환되는 독립 `주제+세부범위` 정확히 두 개까지 순차 조회하고, 각 응답을 독립 검증한 뒤 각 scope의 인용만 가진 두 안내 구역을 반환합니다. `guidance_resolutions`와 단일 `guidance_resolution`은 XOR이며, 세 개 이상·방향 충돌은 명확화, partial/no coverage/계약 오류는 인용 없는 항목화 `insufficient_evidence`입니다. 질문 원문·Case·문서·대화 이력은 Data에 보내지 않으며 coverage가 없거나 출처·version·hash가 맞지 않으면 모델 지식으로 채우지 않습니다. V2 조합 profile은 readiness-gated guidance URL·token·registry version으로만 이 호출을 보냅니다. Claude는 제한된 단일 안내 표현 선택에만 쓰며, 다중 scope 결과는 항상 결정론적으로 구성합니다. 웹 검색·임의 URL·DB·파일 접근·네이티브 Tool Calling은 사용하지 않습니다.

패키지 안 `src/customs_ai_agent/contracts/schemas/backend/v3/`가 유일한 Agent HTTP 계약 원본입니다. Backend 소비자는 이 schema bytes를 자신의 저장소에 갱신해 사용합니다. Data Evidence 계약은 Data 저장소가 소유하며, Agent는 실행에 필요한 Data V3 schema bytes만 package에 두고 Harness가 sibling 바이트를 비교합니다.

## OCR Gold와 PDF preview 승인

`tools/create_document_dataset_receipt.py`는 PDF·JPEG·PNG·JSON 원본을 **그 자리에서** SHA-256으로 읽고, 원문·텍스트·절대경로 없이 `dataset-receipt.json`만 `_local/data/`에 만듭니다. JSON은 `structured_reference`로 기록되어 OCR 점수 대상이 되지 않습니다. `tools/prepare_document_preview_gold_packet.py`는 이 receipt에서 정확히 개발 60개와 봉인 20개를 결정론적으로 골라 `awaiting_human_adjudication` draft packet만 만듭니다. 두 도구 모두 OCR 실행·원본 복사·사람 승인·성능 pass를 수행하지 않습니다.

```powershell
uv run --locked --no-sync --group dev python tools/create_document_dataset_receipt.py `
  --source-directory <local-source> --dataset-id <id> --dataset-revision <revision> `
  --received-as-of 2026-08-11 --output-directory <workspace>/_local/data/receipts/<id>/<revision>

uv run --locked --no-sync --group dev python tools/prepare_document_preview_gold_packet.py `
  --receipt <workspace>/_local/data/receipts/<id>/<revision>/dataset-receipt.json `
  --as-of 2026-08-11 --output-directory <workspace>/_local/data/evaluation/<id>/<revision>
```

위의 v1 packet 도구는 기존 호환용이다. 현재 첫 승인 대상인 **합성 PDF 2,000세트**에는
반드시 `stratified-v2` 흐름을 사용한다. 이 흐름은 정상/실제 오류 × B/L·Invoice·Packing
List의 여섯 strata를 development 60개(각 10개), sealed 20개(각 최소 3개와 SHA 순위
추가 2개)로 고정한다. 오류 directory만으로 고르지 않고 `error_labels.csv`에서 같은
`E####`·문서 유형의 nonblank `wrong_value`가 있는 PDF만 오류 표본으로 인정한다.

```powershell
uv run --locked --no-sync --group dev python tools/prepare_stratified_synthetic_pdf_gold_packet.py `
  --receipt <dataset>/metadata/dataset-receipt.json `
  --error-labels <dataset>/error_labels.csv `
  --output-directory <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/packets `
  --as-of 2026-08-15

uv run --locked --no-sync --group dev python tools/seed_synthetic_pdf_ocr_gold.py `
  --receipt <dataset>/metadata/dataset-receipt.json `
  --packet <development-or-sealed-stratified-v2-packet> `
  --source-directory <dataset> --clean-manifest <dataset>/clean_manifest.csv `
  --error-labels <dataset>/error_labels.csv `
  --output-directory <workspace>/_local/data/document-preview/ocr-gold/<dataset>/<revision>/drafts
```

두 명령은 SHA-결속 packet과 field Gold **초안**만 만든다. PDF 렌더링을 확인한 검수자가
loopback 검수 UI에서 span/table cell을 보완하고 제출한 뒤에만
`tools/finalize_synthetic_pdf_ocr_gold.py --confirm-human-adjudication`으로
`human_adjudicated` Gold를 만들 수 있다. 상세 검수·handoff 절차는
[`docs/LOCAL_PDF_GOLD_REVIEW.md`](docs/LOCAL_PDF_GOLD_REVIEW.md), 세 OCR 후보의
offline 실행·평가·봉인 reservation 절차는
[`docs/LOCAL_SYNTHETIC_PDF_CANDIDATE_RUNNER.md`](docs/LOCAL_SYNTHETIC_PDF_CANDIDATE_RUNNER.md)에
있다. 이때도 OCR 결과는 자동 심사 입력이 아니다.

OCR preview engine은 human-Gold 경로에서는 Gold 평가와 사람 승인을 거치기 전에는 켜지지 않습니다. 합성 PDF의 source-attested 경로는 별도 local receipt chain과 sealed Gate가 통과한 경우에만 자동 pointer를 만들 수 있으며, 두 provenance를 섞지 않습니다. runtime이 읽는 모든 파일은 workspace의 Git 제외 경로 한 곳에만 둡니다.

```text
<workspace>/_local/data/document-preview/
  models/                         # 사람이 명시적으로 받은 package/model 파일
  staging/                        # provisioning 입력 JSON
  runtime-configurations/<SHA>.json
  model-inventories/<SHA>.json
  evaluation-results/<SHA>.json   # sealed evaluator의 canonical 결과
  source-attested-evaluation-results/<SHA>.json  # source-attested sealed 결과
  approvals/pdf-preview-active.json
```

승인 pointer는 다음을 정확히 묶습니다.

- `approval_id`, PDF 전용 media type, engine kind/revision
- runtime configuration SHA-256과 model inventory SHA-256
- sealed packet의 dataset revision, Gold receipt SHA-256, evaluator result SHA-256

runtime은 시작 시 pointer·configuration·inventory·sealed result·설치 package version·모든 model artifact SHA-256을 검사합니다. Docling inventory에는 `docling`만이 아니라 실제 layout/table/GLM 실행에 관여하는 `docling-core`, `docling-ibm-models`, `docling-parse`, `deepsearch-glm`, `torch`, `torchvision` 등 고정 실행 closure도 모두 version으로 선언해야 합니다. PaddleOCR/Docling의 설정된 model directory는 재귀적으로 훑어 inventory에 적힌 regular file 집합과 정확히 같아야 하므로, 누락·미신고 파일·symlink/Windows junction(reparse point)은 모두 fail-closed 합니다. 요청 중에도 이 집합을 다시 확인하고, package version 또는 model file의 크기·수정시각·regular-file 상태가 달라지면 PDF도 즉시 `unavailable`으로 끝냅니다. 모델을 다시 다운로드하거나 다른 OCR engine으로 fallback하지 않습니다. OCR 결과는 계속 사람 확인 전 후보이며 Review·Finding·통관 결론에 자동으로 들어가지 않습니다.

실제 package/model provisioning은 기본 서비스 명령과 분리된 **명시적 운영자 작업**입니다. optional extra는 하나만 고르고, package/model 다운로드가 필요한 명령도 운영자가 직접 실행합니다. `provision`과 `approve` 도구 자체는 네트워크에 접속하지 않습니다.

```powershell
# 예: PDF text layer 후보. PaddleOCR/Docling도 각각의 pinned extra 하나만 선택한다.
uv sync --locked --extra document-preview-pdf

# 이 명령 전에는 models/와 staging/의 configuration·inventory JSON을 사람이 준비한다.
# 기본 12 GiB free-space gate는 실제 PaddleOCR/Docling provisioning 전에 보수적으로 막는다.
uv run --locked --no-sync --group dev python tools/provision_document_preview_runtime.py `
  --document-preview-root <workspace>/_local/data/document-preview `
  --configuration <workspace>/_local/data/document-preview/staging/configuration.json `
  --inventory <workspace>/_local/data/document-preview/staging/inventory.json `
  --cache-directory <local-package-or-model-cache> --confirm-provision

# human-Gold sealed evaluator 결과를 사람이 검토한 뒤에만 pointer를 atomic activation한다.
uv run --locked --no-sync --group dev python tools/approve_document_preview_runtime.py `
  --document-preview-root <workspace>/_local/data/document-preview `
  --evaluation-result <workspace>/_local/data/document-preview/evaluation-results/<evaluation-result-sha>.json `
  --configuration-sha256 <configuration-sha> --model-inventory-sha256 <inventory-sha> `
  --approval-id <human-approval-id> --confirm-human-approval

# approve 출력의 approval_sha256을 정확히 설정한 다음 새 Agent process를 시작한다.
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH = '<workspace>/_local/data/document-preview/approvals/pdf-preview-active.json'
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256 = '<approval-sha256>'
```

합성 PDF의 `source_attested_synthetic` sealed 결과는 별도 evaluator의
`--auto-activate-source-attested-synthetic` switch가 pointer를 원자적으로 쓸 수
있다. 이를 이미 실행 중인 Agent에 반영하려면, Agent를 처음 시작할 때에만 importer의
명시 switch를 사용한다. pointer가 없거나 chain이 불완전하면 process는 계속
`unavailable`이며, sealed pointer가 생긴 뒤 다음 PDF 요청에서 전체 local chain을 다시
검증해 반영한다. 이후 activation마다 SHA 환경변수 변경이나 Agent 재시작은 없다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile dev-fixture `
  -EnableSourceAttestedPdfPreviewAutoActivation
uv run --locked --no-sync --group dev customs-ai-agent-dev-serve
```

이 모드는 PDF content-stream compatibility를 검증한 이 정확한 합성 데이터셋에만
적용된다. 실제 기업 문서·스캔·손글씨·저화질 이미지의 OCR 품질 근거가 아니며,
동일 local write 권한을 가진 악성 프로세스까지 구별하는 외부 signer도 아니다. 상세
제약과 증거 chain은 [`docs/LOCAL_SOURCE_ATTESTED_SYNTHETIC_PDF_ACTIVATION.md`](docs/LOCAL_SOURCE_ATTESTED_SYNTHETIC_PDF_ACTIVATION.md)에 있다.

`configuration`은 `schema_version`, `engine_kind`, `extractor_revision`, `options`만 허용합니다. `pdf_text_layer`의 `options`는 빈 객체이다. PaddleOCR은 `PP-OCRv4`, CPU 1 thread, detection/recognition/classifier의 세 local model directory를 모두 고정해야 하며, classifier를 쓰지 않아도 생략할 수 없다. Docling은 CPU 1 thread·table structure `fast`의 local layout/table parser만 허용하고 `ocr_enabled: false`를 강제해 EasyOCR downloader를 차단한다. `inventory`는 pinned package version과 각 model artifact의 relative path·byte size·SHA-256·license ID·공식 HTTPS source URI를 모두 요구하며, Docling은 output에 관여하는 고정 실행 closure 전체도 빠짐없이 적어야 한다. OCR engine의 설정 directory 아래 regular file은 inventory에 모두 한 번씩 선언되어야 하며, extra file·누락 file·link/junction은 허용하지 않는다. 경로 이탈·symlink·누락 artifact·버전 불일치·sealed 평가 Gate 실패·pointer 변조는 모두 preview 비활성으로 끝난다.

PaddleOCR/Docling은 대용량 wheel·모델·cache가 필요할 수 있습니다. provisioning 도구의 기본 free-space 기준은 12 GiB이며 `--minimum-free-gib`로 더 높은 운영 기준을 지정할 수 있습니다. 여유 공간이 부족하면 다운로드나 부분 설치 전에 중단해야 하며, 일반 Agent runtime은 언제나 no-download 상태를 유지합니다.

## 확인

```powershell
uv run --locked --group dev ruff check src tests
uv run --locked --group dev mypy
uv run --locked --group dev pytest -q
```

실제 기능 UAT는 GitHub Actions에서 실행하지 않습니다. 개인 권한과 로컬 Data 입력이 준비된 사람이 명시적으로 실행하며, 요약 manifest와 모든 상세 non-secret transcript는 workspace `_local/state/scenario-runs/` 아래에만 남습니다. 독립 통관 지식 대화는 그 안의 예약 `knowledge/` namespace에 분리되고, 기존 `_local/state/knowledge-runs/`는 과거 explicit-dev 기록으로 보존됩니다. 기본 개발에서는 각각의 transcript를 명시적으로 켠 경우에만 기록합니다.
