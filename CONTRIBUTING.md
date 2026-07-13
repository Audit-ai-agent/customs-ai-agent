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
uv run ruff check .
uv run mypy
uv run pytest -q
uv build --wheel
```

PR 설명에는 문제·의도·대안·선택 이유·영향 계약·실행 명령·결과·남은 위험을 기록한다.
실제 문서, 개인정보, 거래정보, token, `.env`, 실행 결과 전체는 커밋하지 않는다.

## 완료 기준

- clean checkout에서 lock 기반 설치, lint, typecheck, 28개 이상의 회귀 테스트와 wheel build가 통과한다.
- 새로운 High/Critical finding은 실제 item-bound evidence ID를 가진다.
- 검색 실패·partial·unknown·ambiguous 상태가 자동 완료되지 않는다.
- 공개 API나 wire schema가 바뀌면 호환성 테스트와 문서가 함께 바뀐다.
