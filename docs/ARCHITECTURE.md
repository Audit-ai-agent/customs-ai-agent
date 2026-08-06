# Agent 구조

```text
Frontend → Backend → Agent → Data Evidence API
```

- Frontend는 사용자 화면을 소유합니다.
- Backend는 로그인, Case, 업로드, Job, 최종 사람 결정을 소유합니다.
- Agent는 문서 텍스트 분석, Data HTTP 검색, 검토용 결과 변환을 소유합니다.
- Data는 합성 fixture 또는 실제 3법령의 수집·store·Evidence API를 소유합니다.

저장소 사이에는 HTTP와 versioned schema만 사용합니다. Agent는 sibling 경로 import, DB 연결, Data container 제어를 하지 않습니다.

## 실행 방식

`local-development`는 결정론 텍스트 분석기와 Data `dev-fixture`를 사용합니다. Claude 키, 법령 API, 모델 cache, Docker가 필요하지 않은 기본 개발 경로입니다.

`functional-uat`는 실제 Claude와 Data `functional-uat-3law` store를 사용합니다. Data가 기준일·source hash·모델 revision을 자기 manifest에 기록합니다. Agent는 그 manifest를 승인 증거로 해석하지 않고, 정상 HTTP 흐름과 사용자 여정만 확인합니다.

두 방식은 loopback에만 바인딩되고 `X-Customs-Service-Mode`가 정확히 같을 때만 서로 호출합니다. Backend→Agent 토큰과 Agent→Data 토큰은 분리됩니다.

## 한 번의 검토

1. Backend가 텍스트 문서와 Case 정보를 Agent에 보냅니다.
2. Agent가 결정론 분석기 또는 Claude로 문서 유형·HS 후보를 읽습니다.
3. Agent가 코드로 문서·품목을 대응하고 수량·금액·원산지·B/L 값을 비교합니다.
4. 필요한 문서만 최대 한 번 재추출한 뒤 Data Evidence API에 한 번 일괄 조회합니다.
5. Agent가 최대 2 cycle·6 action의 trace와 8구역 근거 보고서를 돌려줍니다.
6. Backend가 사용자에게 보여 주고 담당자가 finding 판단, 수정 제안, child Run 재심사를 수행합니다.

## 근거 기반 대화와 수정 입력

- Backend는 Review Run별 대화와 사람 수정 제안을 저장합니다. Agent는 한 번의 질문과 최근 8턴, 현재 Run의 구조화 결과만 받아 답변합니다.
- `case_grounded`와 `evidence_grounded` 답변은 해당 Run의 fact·comparison·Evidence ID만 인용합니다. 일반 통관 안내는 사건 판단이 아니라는 한계를 함께 표시합니다.
- Agent는 대화로 상태를 변경하지 않습니다. 수정값은 Backend의 독립 승인 뒤 child Run에만 전달되며, Agent는 원문 인용을 지우지 않고 `human_provided` 사실을 추가해 비교에 우선 사용합니다.

Agent 응답은 언제나 사람 검토 상태입니다. 자동완료, 이미지/OCR/PDF 처리, 서명·승인·commit pin·신뢰 저장소·배포 판정은 이 구조에 없습니다.
