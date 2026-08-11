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

`functional-uat`의 기본 Evidence profile은 Data `functional-uat-law-history`입니다. Data가 65법령의 기준일·source hash·모델 revision과 법령·HSK·요건 snapshot 날짜를 검증합니다. Agent는 `temporal_context`의 세 날짜와 법령·HSK·요건별 `covered|unavailable`인 `temporal_coverage`를 원자적으로 검증·보존하고, 법령 날짜가 Review Run 기준일과 다르거나 law-history 응답에서 둘 중 하나가 없으면 실패로 처리합니다. 보존된 `functional-uat-3law`은 명시적 legacy profile로 유지합니다. 기존 일반 통관 안내는 별도 `functional-uat-kr-trade` V1 guidance peer(18093)로 보존합니다. V1 request는 legacy topic만 포함하고 `scope_hint`를 키 자체로 생략합니다. Agent가 V2 scope를 V1 topic으로 명시적으로 매핑하되, V1 manifest가 선언한 방향만 적용합니다. 공식 페이지 단위 V2는 `functional-uat-kr-trade-v2-law-history` runtime profile에서 peer(18094)를 명시적으로 선택하고 정확한 `topic + scope_hint`와 readiness를 요구합니다. Data local token/state는 V1 profile과 공유하지만, readiness와 query의 registry version·hash·snapshot·기준일이 copied manifest와 맞을 때만 사용합니다. V2 대화의 첫 immutable snapshot ID는 Backend가 conversation에 보존하고 다음 turn에서만 `expected_snapshot_id`로 전달합니다. Agent는 이 값을 V2 Data에만 전달하고, unavailable·mismatch이면 active snapshot fallback 없이 coverage 부족으로 끝냅니다. 문서 심사의 `/retrieve`는 언제나 선택된 Evidence peer 하나만 사용합니다.

두 방식은 loopback에만 바인딩되고 `X-Customs-Service-Mode`가 정확히 같을 때만 서로 호출합니다. Backend→Agent token과 Agent→Data token은 분리됩니다. 별도 guidance peer를 쓸 때에는 URL·token·registry version이 완전한 묶음이어야 하며, 누락·mode 불일치·비loopback 주소·V2 readiness/provenance 불일치는 실패로 처리합니다.

## 한 번의 검토

1. Backend가 텍스트 문서와 Case 정보를 Agent에 보냅니다.
2. Agent가 결정론 분석기 또는 Claude로 문서 유형·HS 후보를 읽습니다.
3. Agent가 코드로 문서·품목을 대응하고 수량·금액·원산지·B/L 값을 비교합니다.
4. 필요한 문서만 최대 한 번 재추출한 뒤 Data Evidence API에 한 번 일괄 조회합니다.
5. Agent가 최대 2 cycle·6 action의 trace와 8구역 근거 보고서를 돌려주며, 이력 Evidence의 세 기준일과 영역별 coverage는 nullable `temporal_context`·`temporal_coverage`로 함께 보존합니다.
6. Backend가 사용자에게 보여 주고 담당자가 finding 판단, 수정 제안, child Run 재심사를 수행합니다.

## 근거 기반 대화와 수정 입력

- Backend는 Review Run별 대화와 사람 수정 제안을 저장합니다. Agent는 한 번의 질문과 최근 8턴, 현재 Run의 구조화 결과만 받아 답변합니다.
- `case_grounded`와 `evidence_grounded` 답변은 해당 Run의 fact·comparison·Evidence ID만 인용합니다. 일반 통관 안내는 사건 판단이 아니라는 한계를 함께 표시합니다. V2는 같은 수출입 방향에 맞는 독립 `주제+세부범위` 정확히 두 개만 순차 `/guidance/query`할 수 있고, 첫 검증 snapshot으로 두 번째 요청을 pin합니다. Data에는 safe query·topic·scope·방향·기준일·필요한 snapshot ID만 보내며 질문 원문, Case, 문서, 대화 이력은 보내지 않습니다. 세 개 이상 또는 방향 충돌은 Data/Claude 없이 명확화로 끝냅니다.
- 두 scope의 `covered` 응답과 사건 citation이 모두 있을 때만 `guidance_resolutions` 정확히 두 개, `answer_format="multi_scope_checklist"`, `사건 분석 → 주제 1 공식 안내 → 주제 2 공식 안내 → 사건 확인 목록`의 결정론 네 구역을 만듭니다. 각 일반 안내 citation은 자기 `resolution_index`만 가지며, 마지막 구역은 법적 결론이나 자동 승인·통관 판단을 만들지 않습니다. 하나라도 partial/no coverage/계약 오류이면 citation을 섞거나 사건 적용을 만들지 않고 항목화된 `insufficient_evidence`를 반환합니다. 단일 `guidance_resolution`과 다중 `guidance_resolutions`는 XOR입니다.
- 사건 채팅의 문서형 다줄 붙여넣기는 신규 심사 근거가 아닙니다. Agent는 해당 원문을 Data, Claude, transcript에 전달하지 않고 문서 등록·교체 후 새 Review Run을 안내합니다.
- Agent는 대화로 상태를 변경하지 않습니다. 수정값은 Backend의 독립 승인 뒤 child Run에만 전달되며, Agent는 원문 인용을 지우지 않고 `human_provided` 사실을 추가해 비교에 우선 사용합니다.

Agent 응답은 언제나 사람 검토 상태입니다. 자동완료, 이미지/OCR/PDF 처리, 서명·승인·commit pin·신뢰 저장소·배포 판정은 이 구조에 없습니다.
