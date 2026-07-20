# Local Agent Execution Inspector

이 도구는 제품 Frontend나 Backend가 아니다. 사용자가 CI·PL·B/L·C/O 텍스트를
입력하고 다음 실제 실행 경로의 공개 가능한 입출력을 한 화면에서 확인하기 위한
localhost 전용 엔지니어링 도구다.

```text
브라우저
→ Local Agent Execution Inspector
→ 실제 Claude 분류·추출
→ 결정론적 정규화·Pairing·교차검증
→ 실제 Evidence API v2 HTTP
→ 현재 PostgreSQL·Qdrant·Neo4j의 3법령 진단 corpus
→ Finding·XAI·최종 Outcome
```

## 판정 경계

- 기능 실행 확인용이며 정량 평가·RAG 품질 검증 도구가 아니다.
- 입력은 text-only다. OCR 경로는 실행하지 않는다.
- Data 범위는 정확히 3/58법령이며 `release_ready=false`다.
- 실제 Claude와 실제 Data HTTP만 허용하고 Mock fallback은 금지한다.
- 입력 원문은 브라우저 메모리와 해당 실행 프로세스 안에서만 사용한다.
- 서버 응답에는 원문, API key, bearer token, raw prompt/response, 내부 graph state를
  포함하지 않는다.
- 민감하거나 실거래인 문서를 넣지 않는다. 입력 텍스트는 Claude API로 전송된다.

## 준비물

1. clean commit 상태의 `customs-data-platform` TASK worktree
2. 같은 Data commit으로 생성한 production provider context
3. 새 bearer token 파일
4. 새 diagnostic manifest 출력 경로
5. `ANTHROPIC_API_KEY`와 Claude 설정이 있는 절대경로 `.env`
6. 보존된 PostgreSQL·Qdrant·Neo4j 3법령 stack

`provider context`, token, manifest는 저장소 밖의 `_artifacts` 또는 `_secrets`에 둔다.
Git에 추가하지 않는다.

## 1. Data provider context 생성

Data worktree가 clean committed HEAD인지 확인한 뒤 실행한다.

```powershell
& $UV run --locked python tools/build_production_context.py `
  --repository $DATA_WORKTREE `
  --output $PROVIDER_CONTEXT
```

`$PROVIDER_CONTEXT`는 존재하지 않거나 비어 있어야 하며 Data 저장소 밖이어야 한다.

## 2. 3법령 read-only Evidence v2 진단 서버 실행

Data 환경에서 PostgreSQL·Qdrant·Neo4j 접속 설정을 주입한 뒤 실행한다.

```powershell
& $UV run --locked python tools/run_three_law_v2_diagnostic.py `
  --diagnostic-only-three-law `
  --provider-context $PROVIDER_CONTEXT `
  --service-token-file $TOKEN_FILE `
  --diagnostic-manifest-output $DIAGNOSTIC_MANIFEST `
  --host 127.0.0.1 `
  --port 18012 `
  --expected-loaded-law-count 3
```

서버는 시작 전에 다음을 확인한다.

- 고정 법령 ID `001556`, `002421`, `006392`
- 법령 원문 hash와 MST version
- PostgreSQL/Qdrant 식별자 집합·개수 일치
- PostgreSQL/Neo4j 식별자 집합·개수 일치
- 실제 Knowledge snapshot과 지원 기준일
- PostgreSQL transaction read-only

하나라도 다르면 서버를 열지 않거나 요청을 503으로 닫는다.

## 3. Inspector 실행

Agent worktree에서 실행한다.

```powershell
& $UV run --locked python tools/agent_execution_inspector/server.py `
  --live `
  --three-law-diagnostic `
  --env-file $CLAUDE_ENV_FILE `
  --evidence-base-url http://127.0.0.1:18012 `
  --evidence-token-file $TOKEN_FILE `
  --diagnostic-manifest $DIAGNOSTIC_MANIFEST `
  --host 127.0.0.1 `
  --port 8021 `
  --run-timeout-seconds 300
```

브라우저에서 `http://127.0.0.1:8021`을 연다. CI·PL·B/L·C/O 텍스트를 붙여 넣고
`실제 Agent 전체 실행`을 선택한다.

### 미리 만든 합성 시나리오 사용

화면의 `미리 만든 가상 테스트 시나리오`에서 다음 8개 입력을 선택할 수 있다.

1. 정확한 HSK `0303540000`과 요건서류 누락 기준 흐름
2. 같은 거래에 요건서류 코드가 제출됐다고 선언하는 metadata-only 비교
3. CI–PL 품목 수량 불일치
4. Invoice 표시 합계와 수량·단가 산술 불일치
5. CI–C/O 원산지 불일치
6. CI와 C/O 양쪽의 HS Code 누락
7. 문서 원문 안의 Prompt Injection 공격 문구
8. 동일 Case 안의 Invoice 유형 중복

선택만으로 현재 입력을 덮어쓰거나 네트워크 실행을 시작하지 않는다. `선택 시나리오
입력 불러오기`를 눌러야 Case·거래·문서 입력에 반영되고, 이때도 Claude는 호출되지
않는다. 내용을 확인한 다음 `실제 Agent 전체 실행`을 별도로 눌러야 실행된다.

- 시나리오를 불러올 때 Evidence health가 제공한 `supported_review_as_of`를 사용한다.
- 새 Review Run ID를 생성하고 화면의 이전 결과·오류·입력 digest를 제거한다.
- 실행 중에는 시나리오 선택과 불러오기가 잠긴다.
- 불러온 문서를 수정하면 화면에 `수정됨`을 표시하고 기존 관찰 포인트는 참고로만 본다.
- 시나리오는 합성값이며 정답 라벨·평가셋·법적 판정이 아니다.
- 요건서류 제출 시나리오는 code metadata만 비교하며 실제 다섯 번째 문서의 진위·내용을
  검증하지 않는다.

카탈로그는 서버 시작 시
`scenario_catalog.schema.json`으로 검증된다. 같은 origin의 `GET /api/scenarios`로 한 번
메모리에 적재하며 브라우저 저장소나 서버 run record에 저장하지 않는다. 카탈로그를
불러오지 못해도 수동 입력과 Agent 실행은 계속 사용할 수 있다.

각 실행은 서버가 생성한 추측 불가능한 32자리 run ID를 사용한다. 동기 응답 제한인
`--run-timeout-seconds` 안에 끝나지 않아도 실행은 취소하거나 유실하지 않는다. UI가
`GET /api/runs/{run_id}`를 polling하여 worker 종료 후 공개 결과를 가져온다.

- 브라우저와 서버 모두 동시에 하나의 실행만 허용한다. 브라우저는 첫 `await` 전에
  입력을 잠그고 단조 증가 generation을 검사하므로 중복 제출이나 늦게 도착한 이전
  응답이 현재 결과·오류·busy 상태를 덮어쓰지 못한다.
- 서버는 입력 request나 secret을 run registry에 저장하지 않는다.
- 완료 run에는 안전성·크기 검사를 통과한 공개 envelope만 메모리에 저장한다.
- 완료 run은 15분 후 제거하며 최대 8개만 보존한다.
- 새 실행을 누르는 즉시 이전 결과 영역을 숨기고 비운다. 새 실행이 실패해도 이전
  Evidence·Finding·XAI를 현재 결과처럼 표시하지 않는다.

## 화면에서 확인할 항목

1. 상단 고정 경계: 실제 Claude, 실제 Evidence HTTP, 3/58 진단, OCR 미사용
2. 입력 원문과 browser/server SHA-256 일치
3. 문서별 분류·추출·정규화 결과
4. Invoice–Packing List line-item Pairing
5. CI–PL, PL–B/L, C/O–CI 교차검증 issue
6. Claude model call·error audit
7. Evidence retrieval 결과, snapshot, citation metadata
8. Finding과 근거 연결
9. XAI report와 최종 `completed`/`needs_review`/`blocked`
10. Safe Raw JSON에서 Mock fallback 0과 비공개 필드 부재

최종 상태는 현재 정책과 3법령 검색 결과를 그대로 표시한다. 성공처럼 보이게
`completed`로 강제하지 않는다.

## 종료

Inspector와 진단 서버 프로세스를 먼저 종료한 뒤 Data stack에는 다음만 사용한다.

```powershell
docker compose down
```

`down -v`, `docker system prune`, `docker volume prune`은 사용하지 않는다. 세 Data
named volume은 후속 58법령 적재와 비교 검증을 위해 보존한다.
