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
3. Agent가 Data Evidence API에 한 번 조회합니다.
4. Agent가 Evidence와 확인 항목을 작은 응답으로 돌려줍니다.
5. Backend가 사용자에게 보여 주고 담당자가 다음 결정을 합니다.

Agent 응답은 언제나 사람 검토 상태입니다. 자동완료, 이미지/OCR/PDF 처리, 서명·승인·commit pin·신뢰 저장소·배포 판정은 이 구조에 없습니다.
