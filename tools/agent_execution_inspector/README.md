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
2. 같은 Data commit으로 생성한 Local UAT provider context
3. 새 bearer token 파일
4. 새 diagnostic manifest 출력 경로
5. `ANTHROPIC_API_KEY`와 Claude 설정이 있는 절대경로 `.env`
6. 보존된 PostgreSQL·Qdrant·Neo4j 3법령 stack

`provider context`, token, manifest는 저장소 밖의 `_artifacts` 또는 `_secrets`에 둔다.
Git에 추가하지 않는다.

## 1. Data provider context 생성

Data worktree가 clean committed HEAD인지 확인한 뒤 실행한다.

```powershell
& $UV run --locked python tools/build_provider_context.py `
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

## 별도 Evidence V3 HTTP Conformance Runner

위 Inspector는 의도적으로 3법령 Evidence V2 진단 경로다. Evidence V3의 17개
GET/POST release binding과 Supporting Document policy를 검증했다고 해석하면 안 된다.
V3 상호운용성은 loopback Data conformance 서버를 띄운 뒤 다음 별도 명령으로 실행한다.

저장소에는 재현 가능한 text-only 요청을 다음 경로에 고정한다.

```powershell
$AGENT_V3_REQUEST_JSON = Resolve-Path `
  tools/agent_execution_inspector/clean_text_production_candidate_v3.request.json
```

이 fixture는 CI·PL·B/L·C/O 4종과 합성 Supporting Document를 포함한다. assertion은 명시적으로
`test_issuer`이며 trusted connector나 사람 승인으로 인정되지 않는다. 기대 disposition은
`human_review_required`다. 파일 SHA-256은
`bb5c87b544a9a46c6261832fd24b25b988b9f1fd56ba07f3bdb815c00e674567`, 정규화된 request
hash는 `7c68df85e2d38a42a094c89df14f328a189beeb736b40900de963cd865152e4f`다. 실제 기관
문서·법적 Gold·Production 자동완료 증거로 사용하면 안 된다. Provider identity의
`supported_review_as_of`도 fixture의 `2026-07-10`과 일치해야 한다. 다르면 Claude를
호출하기 전에 `evidence_review_date_mismatch`로 fail-closed한다.

```powershell
& $UV run --locked python tools/agent_execution_inspector/run_v3_conformance.py `
  --base-url http://127.0.0.1:18013 `
  --identity $EVIDENCE_V3_IDENTITY_JSON `
  --request $AGENT_V3_REQUEST_JSON `
  --token-file $TOKEN_FILE `
  --env-file $CLAUDE_ENV_FILE `
  --output-dir $ARTIFACT_RUN_DIR\agent-v3-conformance `
  --acknowledge-non-release
```

이 명령은 실제 `AgentRuntime.review_v3_json` 경계, 실제 Claude, 실제 Evidence V3
readiness GET과 retrieval POST를 사용한다. Provider identity는 파일로 명시하며
literal `127.0.0.1` 또는 `::1` 이외 주소는 거절한다(`localhost`도 허용하지 않는다).
PASS는 성공한 readiness GET과 retrieval POST를 실제 counter로 각각 1회 이상 확인하고,
Evidence `ok/complete`, Claude model error 0, 실제 Anthropic 문서 호출 4회 이상,
CI·PL·B/L·C/O 분류 일치, citation metadata 완전성, material narrative citation,
High/Critical Finding의 Evidence 결박, test issuer에 대한 사람 검토 disposition을 모두
요구한다. 출력은 원문·API key·token·raw provider 응답을 제외한 `report.json`과
`junit.xml`뿐이며 `provider_scope=TEST_ONLY_LOOPBACK_CONFORMANCE_PROVIDER`와
`production_release_evidence=false`를 항상 표시한다. 현재 packaged Data V3 lock은
`unreleased`이므로 이 결과는 HTTP 계약 conformance 증거이며 Production Release 증거가
아니다. 실패 보고서는 원본 예외 메시지 대신 allowlist 기반 `failure_stage`,
`error_code`, `error_type`만 기록한다.

요청 JSON은 반드시 실행 중인 Agent checkout의 `HEAD`에 정확히 추적된 파일이어야
한다. untracked/수정된 복사본은 credential이나 외부 API를 사용하기 전에 거부한다.
`report.json.source_input`은 원문 대신 코드가 고정한 Agent 저장소 ID, history-root
identity hash, Agent checkout 기준 상대경로, Git object format, exact HEAD/blob object ID,
raw-byte SHA-256, V3 canonical request SHA-256를 기록한다. 다른 Git 저장소에 같은
fixture를 commit해도 Agent checkout source로 인정하지 않는다. 따라서 민감한
문서 내용을 artifact에 복제하지 않고도 각 실행이 어떤 검토된 source를
사용했는지 독립적으로 확인할 수 있다.

## 종료

Inspector와 진단 서버 프로세스를 먼저 종료한 뒤 Data stack에는 다음만 사용한다.

```powershell
docker compose down
```

`down -v`, `docker system prune`, `docker volume prune`은 사용하지 않는다. 세 Data
named volume은 후속 58법령 적재와 비교 검증을 위해 보존한다.
