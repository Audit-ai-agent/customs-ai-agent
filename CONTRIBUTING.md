# 기여 가이드

## 변경 원칙

1. 하나의 PR은 하나의 문제와 검증 가능한 성공 기준을 다룬다.
2. Data/RAG 구현이나 DB client를 복제하지 않는다. 외부 규정 정보는 Evidence HTTP 계약으로만 받는다.
3. 모델 출력은 후보이며 최종 상태·Evidence Gate·정규화 불변식은 결정론적 코드가 통제한다.
4. Mock과 Live 실행을 명시적으로 구분하고, Live 실패를 Mock 성공으로 대체하지 않는다.
5. schema 변경은 provider release를 먼저 확인하고 lock hash와 회귀 테스트를 함께 갱신한다.

## 로컬 검증

```powershell
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check src tests evals tools
uv run --locked mypy
uv run --locked python tools/verify_service_openapi.py
uv run --locked pytest -q `
  tests/service/test_openapi.py `
  tests/test_packaged_contracts.py `
  tests/test_evidence_contract_lock.py `
  tests/test_evidence_runtime_release_v2.py `
  tests/test_evidence_v3_consumer.py `
  tests/test_v1_wire_compatibility.py `
  tests/test_source_snapshot.py
```

전체 회귀는 계약·모델·데이터 경계 변경에서만 실행한다. API key나 token 값은 명령 인자·로그·PR
첨부 파일에 넣지 않는다.

PR 설명에는 문제·의도·대안·선택 이유·영향 계약·실행 명령·결과·남은 위험을 기록한다.
실제 문서, 개인정보, 거래정보, token, `.env`, 실행 결과 전체는 커밋하지 않는다.

## 완료 기준

- clean checkout에서 lock 기반 설치, lint, format, typecheck와 공개 계약 회귀가 통과한다.
- 계약·모델·데이터 경계 변경은 일반 PR CI 결과로 대체하지 않고 해당 전체 회귀를 별도로 통과한다.
- 새로운 High/Critical finding은 실제 item-bound evidence ID를 가진다.
- 검색 실패·partial·unknown·ambiguous 상태가 자동 완료되지 않는다.
- 공개 API나 wire schema가 바뀌면 호환성 테스트와 문서가 함께 바뀐다.
