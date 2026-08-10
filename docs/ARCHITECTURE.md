# Agent 구조

```text
Frontend → Backend → Agent → Data Evidence API
```

- Frontend는 사용자 화면을 소유합니다.
- Backend는 로그인, Case, 업로드, Job, 최종 사람 결정을 소유합니다.
- Agent는 문서 텍스트 분석, Data HTTP 검색, 검토용 결과 변환을 소유합니다.
- Data는 합성 fixture, 보존된 3법령 또는 65법령 이력의 수집·store·Evidence API를 소유합니다.

저장소 사이에는 HTTP와 versioned schema만 사용합니다. Agent는 sibling 경로 import, DB 연결, Data container 제어를 하지 않습니다.

## 실행 방식

`local-development`는 결정론 텍스트 분석기와 Data `dev-fixture`를 사용합니다. Claude 키, 법령 API, 모델 cache, Docker가 필요하지 않은 기본 개발 경로입니다.

`functional-uat`의 기본 Evidence profile은 Data `functional-uat-law-history`입니다. Data가 65법령의 기준일·source hash·모델 revision과 법령·HSK·요건 snapshot 날짜를 검증합니다. Agent는 `temporal_context`의 세 날짜를 그대로 Backend에 보존하고 법령 날짜가 Review Run 기준일과 다르면 실패로 처리합니다. 보존된 `functional-uat-3law`은 명시적 legacy profile로 유지합니다. 기존 일반 통관 안내는 별도 `functional-uat-kr-trade` V1 guidance peer(18093)로 보존합니다. V1 request는 legacy topic만 포함하고 `scope_hint`를 키 자체로 생략합니다. Agent가 V2 scope를 V1 topic으로 명시적으로 매핑하되, V1 manifest가 선언한 방향만 적용합니다. 공식 페이지 단위 V2는 `functional-uat-kr-trade-v2-law-history` runtime profile에서 peer(18094)를 명시적으로 선택하고 정확한 `topic + scope_hint`와 readiness를 요구합니다. Data local token/state는 V1 profile과 공유하지만, readiness와 query의 registry version·hash·snapshot·기준일이 copied manifest와 맞을 때만 사용합니다. 문서 심사의 `/retrieve`는 언제나 선택된 Evidence peer 하나만 사용합니다.

두 방식은 loopback에만 바인딩되고 `X-Customs-Service-Mode`가 정확히 같을 때만 서로 호출합니다. Backend→Agent token과 Agent→Data token은 분리됩니다. 별도 guidance peer를 쓸 때에는 URL·token·registry version이 완전한 묶음이어야 하며, 누락·mode 불일치·비loopback 주소·V2 readiness/provenance 불일치는 실패로 처리합니다.

## 한 번의 검토

1. Backend가 텍스트 문서와 Case 정보를 Agent에 보냅니다.
2. Agent가 결정론 분석기 또는 Claude로 문서 유형·HS 후보를 읽습니다.
3. Agent가 코드로 문서·품목을 대응하고 수량·금액·원산지·B/L 값을 비교합니다.
4. 필요한 문서만 최대 한 번 재추출한 뒤 Data Evidence API에 한 번 일괄 조회합니다.
5. Agent가 최대 2 cycle·6 action의 trace와 8구역 근거 보고서를 돌려주며, 이력 Evidence의 세 기준일은 nullable `temporal_context`로 함께 보존합니다.
6. Backend가 사용자에게 보여 주고 담당자가 finding 판단, 수정 제안, child Run 재심사를 수행합니다.

## 근거 기반 대화와 수정 입력

- Backend는 Review Run별 대화와 사람 수정 제안을 저장합니다. Agent는 한 번의 질문과 최근 8턴, 현재 Run의 구조화 결과만 받아 답변합니다.
- `case_grounded`와 `evidence_grounded` 답변은 해당 Run의 fact·comparison·Evidence ID만 인용합니다. 일반 통관 안내는 사건 판단이 아니라는 한계를 함께 표시하고, 하나의 `주제+세부범위` bounded `/guidance/query`만 사용합니다. 독립 scope가 둘 이상이면 Data나 Claude를 호출하지 않고 code-owned 선택지를 반환합니다.
- 혼합 질문은 `현재 사건 기준`·`일반 통관 안내`를 분리하며, 적용·영향·보완·다음 확인을 명시한 경우에만 두 citation 영역을 함께 가리키는 결정론 `사건 적용·다음 확인` 구역을 추가합니다. 이 구역은 법적 결론이나 자동 승인·통관 판단을 만들지 않습니다.
- 사건 채팅의 문서형 다줄 붙여넣기는 신규 심사 근거가 아닙니다. Agent는 해당 원문을 Data, Claude, transcript에 전달하지 않고 문서 등록·교체 후 새 Review Run을 안내합니다.
- Agent는 대화로 상태를 변경하지 않습니다. 수정값은 Backend의 독립 승인 뒤 child Run에만 전달되며, Agent는 원문 인용을 지우지 않고 `human_provided` 사실을 추가해 비교에 우선 사용합니다.

Agent 응답은 언제나 사람 검토 상태입니다. 자동완료, 이미지/OCR/PDF 처리, 서명·승인·commit pin·신뢰 저장소·배포 판정은 이 구조에 없습니다.
