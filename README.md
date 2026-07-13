# Customs AI Agent

통관 문서 검토 과정을 재현하는 **Evidence-bound Agent workflow**다. 현재 버전은 Mock 문서와
결정론적 규칙을 이용하는 검증용 MVP이며, 실제 LLM/VLM 문서 인식 adapter는 아직 구현되어 있지
않다. 따라서 `llm_mode=llm`을 운영 가능한 기능으로 표시하거나, Mock 실행 결과를 실제 AI 성능으로
해석해서는 안 된다.

## 저장소 책임과 경계

- 이 저장소가 소유한다: Agent workflow, 문서 분류·추출·정규화·검증 정책, Evidence Gate,
  결과 스키마, Mock adapter와 Agent 평가/회귀 테스트.
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
| LLM | 구현되지 않음 | 주입된 retriever | 실제 adapter 구현 전 fail-fast |

Mock은 테스트 대역이다. 운영 profile에서는 Mock fallback을 금지하고 Backend가 허용된 Evidence
Service 주소로 `HttpRegulationRetriever`를 명시적으로 주입해야 한다. 검색 실패·부분 응답·HS 모호성은
자동 완료하지 않고 `needs_review`로 닫힌다.

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
uv run customs-ai-agent run --scenario happy_path --rag-url http://127.0.0.1:8001
```

두 번째 명령은 Mock RAG 회귀 테스트다. 세 번째 명령만 실제 Evidence Service를 호출한다.

## 계약 갱신 절차

1. Data provider가 새 schema를 기존 버전 수정이 아닌 새 버전으로 release한다.
2. provider 경로에서 schema를 byte-for-byte 복사해 consumer snapshot을 갱신한다.
3. `v1.lock.json`의 version, provider revision, SHA-256을 갱신한다.
4. `uv run pytest tests/test_evidence_contract_lock.py -q`로 hash와 wire example을 검증한다.
5. Agent/Backend 통합 검증 후 같은 PR에서 pin 변경 이유를 기록한다.

구조와 설계 의도는 [Architecture](docs/ARCHITECTURE.md), 기여 기준은
[Contributing](CONTRIBUTING.md), 취약점 보고는 [Security](SECURITY.md)를 참고한다.
