# Customs AI Agent

통관 문서 검토 과정을 재현하는 **Evidence-bound Agent workflow**다. `0.2.0`부터
Anthropic Claude의 structured output으로 텍스트 문서를 분류·추출할 수 있다. Claude는
비결정적인 문서 이해 후보만 만들고, 정규화·계산·교차문서 검증·Evidence API·Evidence Gate·
severity·최종 상태는 결정론적 코드가 통제한다.

현재 입력 계약은 `content_text`다. 따라서 이 버전은 실제 Claude 기반 **텍스트 문서 Agent**이며,
PDF/image 업로드, OCR, layout/table 좌표, 실제 bbox를 구현한 버전은 아니다. 텍스트 경로의
성공을 OCR/VLM 또는 전체 서비스 완성으로 해석하면 안 된다.

## 저장소 책임과 경계

- 이 저장소가 소유한다: Agent workflow, Claude 문서 분석 adapter와 versioned prompt,
  문서 분류·추출·정규화·검증 정책, Evidence Gate, 결과 스키마, Mock adapter와 Agent
  평가/회귀 테스트.
- `customs-data-platform`이 소유한다: 법령·HSK·CCCT 데이터, PostgreSQL/Qdrant/Neo4j,
  Evidence Service와 canonical Evidence API schema.
- Agent는 Data 저장소의 DB나 Python 모듈을 직접 사용하지 않는다. 실제 연결은 오직
  `POST /internal/rag/v1/retrieve` HTTP 계약을 구현한 `HttpRegulationRetriever`를 통해서만 한다.
- Backend는 설치된 Agent package를 호출하고 실행 상태를 저장한다. Agent는 Operational DB에
  직접 쓰지 않는다.

`contracts/evidence/v1/`의 schema는 provider 원본의 **검증용 pinned snapshot**이며 이 저장소의
canonical 계약이 아니다. 버전과 hash는 `contracts/evidence/v1.lock.json`으로 고정한다.

## 실행 모드

| 구분 | 문서/LLM | 규정 검색 | 용도 |
|---|---|---|---|
| 기본 Mock | fixture + 규칙 | `MockRegulationRetriever` | 빠르고 결정론적인 단위·회귀 테스트 |
| Live RAG | fixture + 규칙 | HTTP Evidence Service | Agent–Data 계약·통합 검증 |
| Claude text | Claude structured output | 명시적 HTTP/주입 retriever | 실제 텍스트 분류·추출 검증 |

Mock은 테스트 대역이다. 운영 profile에서는 Mock fallback을 금지하고 Backend가 허용된 Evidence
Service 주소로 `HttpRegulationRetriever`를 명시적으로 주입해야 한다. 검색 실패·부분 응답·HS 모호성은
자동 완료하지 않고 `needs_review`로 닫힌다.

`llm_mode=llm`에서 retriever를 생략하면 실행 전에 실패한다. 테스트에서 Claude와 Mock RAG를
결합하려면 `MockRegulationRetriever`를 dependency로 직접 주입해야 하며, 그 실행은 실제 RAG
검증으로 표시할 수 없다. 모델 API·schema·grounding 실패도 규칙 extractor로 fallback하지 않고
안전한 오류 코드와 HITL 경로로 닫힌다.

중요하게도 0.2.x에서 Claude 경로의 정상 성공 상태 역시 `needs_review`다. exact quote
grounding은 문자열 존재를 증명하지만 Claude가 그 문자열을 올바른 업무 필드에 연결했다는
정확도를 아직 보정하지 않았기 때문이다. 따라서 필드 confidence는 0.85
`source_quote_grounded_uncalibrated`이고, Claude 유래 critical issue도 별도 사람 승인 계약 전에는
자동 `blocked`로 만들지 않는다. 이 정책에서 성공은 “모델 호출·strict schema·exact grounding·
결정론적 검증·Evidence 흐름이 오류 없이 끝나고 사람 검토 대상으로 안전하게 전달됨”을 뜻한다.

Claude 응답은 다음 경계를 통과해야 한다.

- `extra=forbid` structured candidate schema
- 문서 유형별 document/line-item field allowlist
- 모든 비어 있지 않은 값의 source quote 존재
- quote가 입력 문서의 case-sensitive exact span이고 value가 quote 안의 exact bounded span이라는
  결정론적 대조
- 모델이 생성하지 않는 코드 소유 item ID
- 성공 audit에는 모델·prompt version·request/message ID·token·latency만 허용하고, 실패에는
  safe code와 제한된 provider status metadata만 허용

Claude는 법률, Evidence ID, requirement, severity, finding, final status를 생성할 권한이 없다.
텍스트 입력에는 실제 좌표가 없으므로 Claude 경로의 `bbox`는 `None`이며, 합성 좌표를 만들지 않는다.

CLI에서 `--rag-url`을 명시하면 process 또는 명시적 env file의
`EVIDENCE_SERVICE_BASE_URL`보다 우선한다. 둘 중 하나가
존재하면 Live RAG로 간주하며, URL·readiness·retrieval 실패를 Mock으로 전환하지 않는다.
`--review-as-of`를 생략한 Live RAG 실행은
`GET /internal/rag/v1/health/ready`의 `supported_review_as_of`를 가져와 현재 Knowledge
snapshot과 날짜를 맞춘다. 명시한 `--review-as-of` 값은 자동 탐색보다 우선한다.

HTTP base URL은 `localhost`, `127.0.0.1`, `::1` loopback에서만 허용한다. 외부
Evidence Service는 HTTPS를 쓰고 그 exact origin을 `EVIDENCE_SERVICE_ALLOWED_ORIGINS`에
명시해야 하며 URL credential·query·fragment·path prefix는 거부한다. redirect는 따르지
않고 응답 body는 2 MiB로 제한한다.
Data 데모에 `DEMO_TOKEN`이 있다면 Agent에 `EVIDENCE_SERVICE_TOKEN`을 주입하고,
기본 `X-Demo-Token` 헤더를 쓴다. token은 URL·CLI 인자·오류 메시지에 포함하지 않는다.

## 개발 환경

Python 3.11과 uv를 기준으로 한다.

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run mypy
uv run pytest -q
uv build --wheel
```

CLI 예시:

```powershell
uv run customs-ai-agent list-scenarios
uv run customs-ai-agent run --scenario happy_path
uv run customs-ai-agent run --scenario happy_path --rag-url http://127.0.0.1:8010
uv run customs-ai-agent run --scenario happy_path --rag-url http://127.0.0.1:8010 --review-as-of 2026-07-10
uv run customs-ai-agent run --scenario happy_path --llm-mode llm `
  --env-file ..\_secrets\.env --rag-url http://127.0.0.1:8010
```

두 번째 명령은 `EVIDENCE_SERVICE_BASE_URL`이 설정되지 않았을 때만 Mock RAG 회귀
테스트다. 세 번째는 실제 Evidence Service readiness에서 지원 날짜를 읽고, 네 번째는
사용자가 명시한 날짜로 호출한다. 마지막 명령만 실제 Claude와 Evidence Service를 함께 사용한다.

## 단일 `.env`와 Claude 설정

로컬 secret의 기준은 제품 저장소 밖 개인 `_secrets/.env` 한 파일이다.

```dotenv
ANTHROPIC_API_KEY=
CUSTOMS_AGENT_CLAUDE_MODEL=claude-sonnet-5
CUSTOMS_AGENT_CLAUDE_MAX_TOKENS=8192
CUSTOMS_AGENT_CLAUDE_TIMEOUT_SECONDS=90
CUSTOMS_AGENT_CLAUDE_MAX_RETRIES=2
EVIDENCE_SERVICE_BASE_URL=http://127.0.0.1:8010
EVIDENCE_SERVICE_ALLOWED_ORIGINS=
EVIDENCE_SERVICE_TOKEN=
EVIDENCE_SERVICE_TOKEN_HEADER=X-Demo-Token
```

저장소나 worktree마다 `.env`를 복제하지 않는다. Agent는 부모 디렉터리를 무인 탐색하지 않고
`--env-file`로 명시된 한 파일에서 Claude/Evidence allowlist만 settings 객체로 읽는다.
`os.environ`을 바꾸지 않으며 이미 프로세스 값이 있으면 그것이 우선한다.
`CUSTOMS_AGENT_ENABLE_LLM` 같은 두 번째 활성화 토글은 없다. key가 존재해도 기본 mode는 계속
`mock`이므로 테스트가 자동으로 과금 API를 호출하지 않는다.

비민감 기본 설정은 다음과 같으며 환경변수로 선택적으로 덮어쓸 수 있다.

- model: `claude-sonnet-5` (0.2.x 평가 기준으로 exact pin)
- max output: 8,192 tokens
- timeout: 90초
- SDK retry: 최대 2회

Sonnet 5 요청에는 temperature/top-p/top-k 또는 구식 manual thinking budget을 보내지 않는다.
provider base URL은 코드에서 `https://api.anthropic.com`으로 고정한다.
`ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_AUTH_TOKEN`, Anthropic SDK debug
logging은 문서/key 유출 위험 때문에 fail-closed한다. 다른 Claude 모델은 새 평가셋·ADR·Gate
갱신 없이 환경변수만으로 바꿀 수 없다.
실제 문서를 전송하기 전 조직의 Anthropic 데이터 보존/ZDR 조건과 문서 전송 권한을 별도로 확인한다.

## 실제 Claude 합성 평가

기본 pytest는 실제 네트워크를 사용하지 않는다. 실호출은 versioned 합성 문서만 사용하는
다음 명령으로 명시적으로 실행한다.

```powershell
# 1문서 smoke
uv run python evals/run_claude_live.py `
  --env-file ..\_secrets\.env --max-documents 1

# 4종 문서 gold
uv run python evals/run_claude_live.py `
  --env-file ..\_secrets\.env --max-documents 4

# 실제 Claude + 전체 Agent workflow + 명시적 Mock RAG 진단
uv run python evals/run_claude_live.py `
  --env-file ..\_secrets\.env --max-documents 4 --workflow

# 실제 Claude adversarial prompt-injection baseline
uv run python evals/run_claude_live.py `
  --env-file ..\_secrets\.env `
  --gold evals/claude_text_adversarial_v1.json --max-documents 1
```

평가 출력은 유형 정확도, 필드 exact/missing/wrong/hallucinated 수, 모델·prompt version,
token·latency만 포함한다. API key, 입력 문서, 추출값, prompt, raw response는 포함하지 않는다.
이 PASS는 `SYNTHETIC_TEXT_ONLY_NOT_PDF_OCR_OR_RELEASE`이며 Data corpus, OCR/VLM, Backend 저장,
전체 Release를 증명하지 않는다. `--workflow`도 규정 검색은
`EXPLICIT_MOCK_NOT_LIVE_RAG`이므로 Agent–Data Live 통합 PASS로 표시할 수 없다. 이 진단의
기대 최종 상태는 위 안전 정책에 따라 `needs_review`다.

## 계약 갱신 절차

1. Data provider가 새 schema를 기존 버전 수정이 아닌 새 버전으로 release하고
   annotated tag와 full 40-character commit SHA를 확정한다.
2. provider 경로에서 schema를 byte-for-byte 복사해 consumer snapshot을 갱신한다.
3. `v1.lock.json`의 version, provider release tag, provider revision, SHA-256을 갱신한다.
4. `uv run pytest tests/test_evidence_contract_lock.py -q`로 hash와 wire example을 검증한다.
5. 상위 Contract Gate에서 tag target, provider commit의 schema bytes, consumer snapshot,
   lock SHA-256이 모두 같은지 검증한다.
6. Agent/Backend 통합 검증 후 같은 PR에서 pin 변경 이유를 기록한다.

구조와 설계 의도는 [Architecture](docs/ARCHITECTURE.md), 기여 기준은
[Contributing](CONTRIBUTING.md), 취약점 보고는 [Security](SECURITY.md)를 참고한다.
