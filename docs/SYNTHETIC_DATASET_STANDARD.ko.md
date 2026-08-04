# 완전한 가상데이터 생성 및 검증 명세

> 문서 상태: 팀 표준 초안
>
> 적용 범위: Customs AI Agent 개발·회귀·평가용 가상데이터
>
> 적용 제외: 실제 통관 데이터의 법적 효력, 모델 자동 승인, 전체 서비스 Production Release
>
> 문서 형식: GitHub와 Notion에서 함께 사용할 수 있는 Markdown

> Notion으로 복사할 때 저장소 상대 링크가 자동 변환되지 않으면 마지막 “관련 문서”의 제목을
> Notion 페이지 링크 또는 Private GitHub URL로 교체한다.

---

## 1. 한눈에 보는 결론

이 프로젝트에서 **완전한 가상데이터**란 모든 레코드가 정상이라는 뜻이 아니다.

정상 문서, 의도적으로 오류를 넣은 문서, 판단이 모호한 문서, 누락·중복 문서가 함께 있어도
된다. 대신 정의한 평가 범위마다 다음 정보가 빠짐없이 있어야 한다.

1. 거래의 원래 사실인 `Case Master`
2. 문서에 실제로 기록된 값
3. 그 값이 존재하는 page·span·bbox·table 위치
4. 업무·표준 규칙상 유효성
5. 오류가 의도적으로 주입됐는지 여부
6. Agent가 생성해야 하는 Finding과 최종 Outcome
7. 사용해야 하는 법령·조문·Evidence
8. 생성기·template·model·seed·source·license와 파일 hash
9. 개발·검증·Release 평가 사이의 누출을 막는 split 통제

따라서 “100 Case가 있다”, “60/20/20으로 나눴다”, “두 reviewer가 합의했다”만으로 완전성을
증명할 수 없다. 완전성은 **수량이 아니라 목적별 정답성과 재현 가능성**으로 판정한다.

가상데이터가 완전해도 합성 데이터만으로 실제 통관 문서 일반화, Production OCR 성능,
58개 법령 RAG 품질 또는 전체 서비스 Production-ready가 자동으로 증명되지는 않는다.

---

## 2. 목적과 비목적

### 2.1 목적

이 명세는 다음 기능을 서로 혼동하지 않고 평가하기 위한 데이터 생성 기준을 정한다.

- 문서 유형 분류
- CI·PL·B/L·C/O 필드 추출
- 원문 값 정규화
- 같은 Case의 문서 Pairing
- 문서 내부·문서 간 결정론적 검증
- OCR text·reading order·bbox·table 복원
- Claude field candidate와 OCR 근거의 grounding
- Evidence API 검색과 citation 결박
- Finding·severity·Outcome 생성
- 오류·모호성·Evidence 부족의 fail-closed 처리

### 2.2 비목적

이 데이터는 다음을 직접 수행하거나 보장하지 않는다.

- Claude 기반 모델의 사전학습
- 실제 세관 신고의 자동 승인 또는 법적 판단
- Data Platform 법령 원장의 대체
- 실제 운영 환경의 개인정보·보존·삭제 정책 검증
- 실제 문서 모집단 전체에 대한 통계적 대표성 보장
- Frontend·Backend·HITL·운영 배포 완성

---

## 3. 핵심 원칙

### 원칙 1 — 원본 관측값과 업무 정답을 분리한다

문서에 잘못된 컨테이너 번호가 적혀 있다면 Extraction Gold는 그 문자열 그대로다.
Validation Gold만 “표준 위반”이라고 판정한다. 추출 평가를 통과시키기 위해 원본 문자열을
정상 값으로 바꾸지 않는다.

### 원칙 2 — 결함 의도를 추정하지 않는다

표준 위반이 확인돼도 데이터 작성자가 테스트를 위해 넣은 것인지 생성기 실수인지 모를 수
있다. 의도가 확인되지 않은 관측은 `ambiguous`로 관리하고 의도된 결함 탐지 점수에 넣지
않는다. 동일 입력·생성기에서 결함이 재현되고 generator invariant 위반 증거가 있을 때만
`confirmed_generator_error`로 승격한다.

### 원칙 3 — 원본은 불변이다

한 번 등록한 `dataset_id + dataset_version`의 raw bytes를 수정하지 않는다. 수정이 필요하면
새 version이나 hash로 결박된 corrected/counterfactual variant를 만든다.

### 원칙 4 — 한 평가층의 PASS를 다른 평가층으로 확대하지 않는다

Text extraction PASS는 OCR PASS가 아니며, Evidence API conformance PASS는 실제 법령 RAG 품질
PASS가 아니다. 합성 renderer의 bbox PASS도 실제 스캔 문서의 layout PASS가 아니다.

### 원칙 5 — 자동 완료보다 근거 부족을 우선한다

필수 문서, OCR 근거, HS Code, Evidence 또는 citation이 없거나 모호하면 정답 Outcome은
`needs_review` 또는 `blocked`다. 테스트를 통과시키기 위해 `completed`로 보정하지 않는다.

---

## 4. 데이터셋 포트폴리오

하나의 데이터 묶음에 모든 목적을 섞지 않는다.

| 데이터 클래스 | 목적 | 대표 입력 | 점수 사용 |
|---|---|---|---|
| Development fixture | 빠른 단위·계약 개발 | 최소 문서와 고정 응답 | 개발 전용 |
| Clean baseline | 정상 Case 처리 | 모든 불변식을 통과한 문서 묶음 | 가능 |
| Single-defect | 규칙 하나의 탐지 | Case당 통제된 mutation 1개 | 가능 |
| Multi-defect | 복합 Finding과 우선순위 | 서로 독립적인 오류 여러 개 | 가능 |
| Ambiguous/fail-closed | 불확실성 안전 처리 | 의도 불명·근거 부족 | Outcome 평가만 |
| Missing/duplicate documents | 문서 집합 안전성 | 누락·중복 CI/PL/B/L/C/O | Outcome 평가 |
| OCR/Layout robustness | 이미지·표·좌표 성능 | 여러 template와 열화 이미지 | 사람 Gold 필요 |
| Prompt-injection/adversarial | 모델 안전성 | 문서 내부 악성 지시 | 안전 평가 |
| Evidence conformance | HTTP 계약·identity | sealed test provider | RAG 품질 주장 불가 |
| Evidence/RAG quality | 검색 품질·citation | 실제 sealed Knowledge generation | 별도 Gold 필요 |
| Sealed release holdout | 최종 Release 판단 | 사전 비공개 Case와 Gold | 일회성 승인 |

Clean, defect, ambiguous, OCR, Evidence 데이터를 하나의 정확도 숫자로 합치지 않는다. 각 클래스의
분모와 실패 의미가 다르기 때문이다.

---

## 5. 데이터 단위와 식별자

### 5.1 Grain

기본 평가 단위는 `Case`다. 하나의 Case는 동일 거래에 속하는 문서와 파생 variant를 소유한다.

```text
Dataset
└─ Case
   ├─ Case Master
   ├─ CI
   ├─ PL
   ├─ B/L
   ├─ C/O
   ├─ clean rendered documents
   ├─ controlled mutations
   └─ Gold layers
```

### 5.2 필수 ID

| ID | 역할 | 요구사항 |
|---|---|---|
| `dataset_id` | 데이터셋 계열 | 의미가 안정적인 문자열 |
| `dataset_version` | 불변 버전 | raw 변경 시 반드시 증가 |
| `case_id` | 거래 단위 | 전역 또는 dataset 내 고유 |
| `document_id` | 문서 단위 | Case·문서 유형·variant와 결박 |
| `variant_id` | clean/defect 파생본 | parent Case와 연결 |
| `mutation_id` | 결함 주입 단위 | 하나의 의도와 기대 결과를 소유 |
| `span_id` | OCR 근거 단위 | page·image hash와 결박 |
| `finding_id` | 기대 Finding | rule·source·target·evidence와 결박 |
| `knowledge_generation_id` | Data snapshot | 검색 Gold가 참조한 불변 generation |

배열 index만으로 Case를 Pairing하지 않는다. 기존 source에 명시적 Case ID가 없다면 등록 과정에서
안정적 ID를 만들고, source record index와 source hash를 함께 보존한다. 그 ID는 해당 dataset
version의 로컬 계약이지 작성자의 원래 식별자로 주장하지 않는다.

---

## 6. Authoritative Case Master

모든 clean 문서는 하나의 Case Master에서 생성한다. 문서를 먼저 만들고 나중에 서로 맞추는
방식은 문서 간 모순의 원인을 추적하기 어렵게 만든다.

### 6.1 필수 정보

- 당사자: Seller, Buyer, Exporter, Importer, Consignee, Notify Party
- 식별자: Invoice No., B/L No., C/O No.
- 운송: Vessel, Voyage, From, To, Port, 선적·출항·발행일
- 컨테이너: equipment profile, Container No., Seal No.
- 포장: package type과 count
- 품목: stable item ID, description, quantity, unit, unit price, amount
- 금액: currency, precision, rounding rule, total
- 물류 수치: net weight, gross weight, CBM과 단위
- 원산지와 검증된 HS Code
- 필요한 문서 유형과 의도적인 누락·중복 여부
- 기대 Evidence scope와 review date
- 생성 provenance와 source hash

### 6.2 예시 구조

아래 예시는 schema 방향을 설명하기 위한 비민감 예시이며 실제 Gold가 아니다.

```json
{
  "dataset_id": "customs-agent-synthetic",
  "dataset_version": "2.0.0",
  "case_id": "CASE-000001",
  "scenario": "clean_export",
  "parties": {
    "seller_id": "PARTY-SELLER-001",
    "consignee_id": "PARTY-CONSIGNEE-001"
  },
  "transport": {
    "equipment_profile_id": "EQUIPMENT-40HQ-VERSIONED",
    "container_no": "ISO6346-VALID-PLACEHOLDER",
    "seal_no": "SEAL-PLACEHOLDER"
  },
  "items": [
    {
      "item_id": "ITEM-001",
      "description": "SYNTHETIC ITEM",
      "quantity": "100",
      "unit": "PCS",
      "unit_price": "12.34",
      "currency": "USD",
      "hs_code": "VALIDATED-HS-PLACEHOLDER"
    }
  ],
  "expected_documents": [
    "invoice",
    "packing_list",
    "bill_of_lading",
    "certificate_of_origin"
  ],
  "provenance_ref": "PROV-000001"
}
```

실제 생성 시 placeholder가 아닌 검증된 값을 사용해야 한다. 특히 HS Code와 equipment capacity는
공식 source·version 또는 승인된 project reference에 결박한다.

### 6.3 문서 유형 vocabulary

synthetic-v2 Case Master와 Agent 평가 결과의 canonical document type은 Agent wire enum을 따른다.

| Canonical type | 화면·업무 약어 | 기존 dataset source alias |
|---|---|---|
| `invoice` | CI | `commercial_invoice` |
| `packing_list` | PL | `packing_list` |
| `bill_of_lading` | B/L | `bill_of_lading` |
| `certificate_of_origin` | C/O | `certificate_of_origin` |
| `unknown` | 미분류 | 해당 없음 |

파일 디렉터리나 기존 source schema의 `commercial_invoice`를 Agent wire 값 `invoice`와 같은
문자열로 오인하지 않는다. 등록 adapter가 source alias를 canonical type으로 변환하고, 변환
규칙과 version을 manifest에 기록한다.

---

## 7. Clean baseline 생성 기준

Clean baseline은 “눈으로 보기에 자연스러운 데이터”가 아니라 명시한 모든 불변식을 통과한
Case다.

### 7.1 식별자와 형식

- Invoice No., B/L No., C/O No.가 각 계약에 맞는 형식이어야 한다.
- Container No.는 고정한 ISO 6346 규칙과 check digit을 통과해야 한다.
- Container owner/equipment/serial/check digit 구성요소를 별도로 검증한다.
- 날짜와 번호의 공백·대소문자·구두점 정책을 명시한다.

### 7.2 금액

- `quantity × unit_price`와 line amount의 관계를 검증한다.
- 표시 단가의 정밀도에서 가능한 반올림 구간을 계산한다.
- 통화별 소수 자릿수와 rounding mode를 versioned policy로 고정한다.
- 표시 정밀도로 설명 가능한 차이를 산술 결함으로 만들지 않는다.
- line amount의 합과 invoice total의 관계를 검증한다.

### 7.3 중량·용적·포장

- `net weight ≤ gross weight`를 만족해야 한다.
- 품목·포장별 중량 합과 문서 총중량의 관계를 명시한다.
- Container capacity는 보편적인 단일 숫자가 아니라 고정한 equipment profile과 source version에
  결박한다.
- 포장 수량과 단위는 CI·PL·B/L 간 일관돼야 한다.

### 7.4 문서 간 관계

- CI–PL: Invoice 번호·날짜, 당사자, 품목·수량, 선적 정보
- PL–B/L: Shipper, Consignee, Vessel/Voyage, Port, Container, Seal, Package, Weight, CBM
- C/O–CI: Exporter, Importer, Origin, 품목, HS Code
- 동일 항목은 stable item ID로 연결하고 description 문자열만으로 Pairing하지 않는다.

PL의 추가 품목이 허용되는 업무 시나리오라면 “정상 추가 품목”이라는 명시적 scenario와 rule이
있어야 한다. 정의가 없다면 clean baseline에 포함하지 않는다.

### 7.5 결제조건

- T/T, L/C 등 결제조건별 필수·금지·선택 필드를 정의한다.
- 의도 없이 모든 필드를 채우는 생성 방식을 사용하지 않는다.
- 복수 조건을 허용한다면 그 조건과 기대 Agent 동작을 명시한다.

---

## 8. 문서 생성과 결정론적 렌더링

### 8.1 생성 단계

```text
Case Master validation
→ clean semantic documents 동결
├─ clean render → clean Extraction/Layout Gold
├─ semantic controlled mutation
│  → variant semantic documents
│  → variant render
│  → variant Extraction/Layout/Validation Gold
└─ visual degradation mutation
   → degraded image
   → geometry transformation 또는 독립 사람 Layout Gold
→ 전체 schema·hash·relationship validation
```

문서의 의미를 바꾸는 mutation은 반드시 rendering 전에 적용하고 variant용 Extraction/Layout
Gold를 새로 생성한다. clean Gold를 의미가 달라진 variant에 복사하지 않는다.

blur·rotation·crop·compression 같은 visual degradation은 문서 의미 mutation과 분리한다.
이 변환이 text truth를 바꾸지 않더라도 bbox geometry는 달라질 수 있으므로 transformation
matrix로 좌표를 검증하거나 독립 사람 Layout Gold를 만든다.

같은 입력과 generator release에서 같은 결과를 만들 수 있어야 한다.

### 8.2 필수 provenance

- generator name·version·commit
- template ID·version·hash
- prompt ID·version·hash를 사용했다면 해당 hash
- random seed와 deterministic 여부
- renderer·PDF engine·image codec version
- font file·license·SHA-256
- DPI, page size, color mode, compression
- 생성 시각과 실행 manifest
- Case Master·문서·page image의 SHA-256
- 원본과 파생본의 parent-child 관계

재현 가능한 synthetic renderer는 layout 개발에 유용하지만 실제 스캔 대표성을 자동으로
제공하지 않는다.

---

## 9. Gold 레이어

Gold는 하나의 JSON에 모든 의미를 섞지 않고 다음 레이어로 분리한다.

| Gold | 질문 | 대표 필드 |
|---|---|---|
| Source/Case Master | 거래의 원래 사실은 무엇인가? | parties, items, transport |
| Extraction | 문서에 실제로 무엇이 적혀 있는가? | exact `raw_value`, source hash |
| Normalization | 비교 가능한 표준값은 무엇인가? | normalized value, unit, rule |
| OCR/Layout | 정답 text가 어디에 어떻게 적혀 있는가? | page, span, bbox, table |
| Validation | 어떤 규칙을 통과·위반하는가? | rule ID, result, Finding |
| Evidence/RAG | 어떤 법령·조문이 근거인가? | generation, law/article/citation |
| Outcome | Agent가 어떻게 닫혀야 하는가? | completed/needs_review/blocked |

### 9.1 Extraction Gold

- 문서에 보이는 exact raw string을 보존한다.
- 빈 값과 누락을 서로 구분한다.
- source file·record·case·document hash를 기록한다.
- 필드가 존재하는 source span을 참조한다.
- 원문이 업무상 틀렸다는 이유로 값을 교정하지 않는다.

### 9.2 Normalization Gold

- normalization rule ID와 version을 기록한다.
- 날짜, 숫자, 통화, 단위, 공백, 대소문자 변환을 구분한다.
- 원문과 normalized value를 모두 보존한다.
- 정보가 부족하면 임의 보완하지 않고 `unknown` 또는 conflict로 둔다.

### 9.3 Validation Gold

- rule ID와 rule version
- source document·field·span
- 비교 대상 document·field·span
- expected pass/fail/unknown
- expected Finding code와 severity
- expected human-review 여부
- 판정 근거와 reviewer/adjudication record

### 9.4 Outcome Gold

최종 Outcome은 Finding 수만으로 결정하지 않는다.

- `completed`: 필수 문서·근거·정책 조건을 모두 충족
- `needs_review`: 모호성, conflict, Evidence 부족, 사람 승인 필요
- `blocked`: 계약·입력·보안·provider readiness 등으로 안전한 실행 불가

필수 Evidence가 없는데 `completed`로 설정한 Gold는 허용하지 않는다.

---

## 10. 판정 상태와 점수 포함 기준

모든 업무 관측은 다음 상태 중 하나를 가져야 한다.

| 상태 | 의미 | Validation 점수 포함 |
|---|---|---|
| `confirmed_normal` | 정상임이 근거로 확인됨 | 포함 |
| `confirmed_intentional_defect` | 의도적 결함과 기대 결과가 확인됨 | 포함 |
| `confirmed_generator_error` | 생성기 실수로 확인됨 | 제외 |
| `ambiguous` | 작성 의도 또는 업무 의미가 불명확 | 제외 |
| `coverage_gap` | 판단에 필요한 truth·source가 없음 | 제외 |
| `not_applicable` | 해당 규칙 대상이 아님 | 제외 |

`confirmed_generator_error`, `ambiguous`, `coverage_gap`은 삭제하지 않는다. 데이터 생성·검수
품질을 개선하는 중요한 증거다. 다만 의도된 결함 탐지 precision/recall의 분모에 넣지 않는다.

판정 변경은 기존 값을 덮어쓰지 않고 append-only event로 남긴다.

```text
상태 변경 시각
이전 상태 → 새 상태
판정자 종류
근거 문서·hash
변경 이유
영향받는 Gold·평가 run
```

AI reviewer는 pre-label과 검수 보조에 사용할 수 있지만, 두 AI reviewer의 합의만으로 Release
Gold를 확정하지 않는다. Release subset의 중요 필드는 독립된 사람 검수와 불일치 adjudication이
필요하다.

---

## 11. Controlled mutation

의도된 오류는 clean baseline을 직접 수정해 덮어쓰지 않고 parent hash에 결박된 variant로
생성한다.

### 11.1 필수 mutation record

```json
{
  "mutation_id": "MUT-RULE-000001",
  "parent_case_id": "CASE-000001",
  "parent_case_sha256": "SHA256-PLACEHOLDER",
  "variant_id": "CASE-000001-VARIANT-0001",
  "operator_id": "replace_single_field/v1",
  "operator_version": "1.0.0",
  "seed": 1001,
  "target_document_id": "DOC-BL-000001",
  "target_field": "container_no",
  "before": "VALID-SOURCE-VALUE",
  "after": "CONTROLLED-INVALID-VALUE",
  "output_document_id": "DOC-BL-000001-VARIANT-0001",
  "output_document_sha256": "SHA256-PLACEHOLDER",
  "intent": "confirmed_intentional_defect",
  "expected_rule_id": "CONTAINER_CHECK_DIGIT",
  "expected_finding_code": "INVALID_CONTAINER_NUMBER",
  "expected_outcome": "needs_review"
}
```

### 11.2 단일 결함 세트

- Case당 mutation 의도는 하나만 둔다.
- mutation 하나가 우연히 여러 rule을 동시에 위반하지 않는지 검증한다.
- 각 rule에 정상 negative control과 결함 positive case가 있어야 한다.
- 원본 clean Case와 모든 variant를 같은 split에 둔다.

### 11.3 복합 결함 세트

- 각 mutation을 별도 record로 유지한다.
- 상호작용으로 새 Finding이 생기는지 명시한다.
- expected Finding의 순서가 아니라 집합·severity·evidence 연결을 평가한다.
- 하나의 오류가 다른 오류를 가려도 누락 허용 여부를 policy로 정의한다.

### 11.4 모호성 세트

의도적으로 `ambiguous`를 만드는 경우에도 기대 동작을 기록한다.

- 자동 위반으로 단정하지 않음
- 무근거 값을 채우지 않음
- `needs_review` 또는 정해진 blocker
- conflict와 관측 근거를 결과에 남김

---

## 12. OCR/Layout 데이터

OCR Release 평가에는 field 값만이 아니라 관측 가능한 layout truth가 필요하다.

### 12.1 Authoritative OCR/Layout Gold

Gold는 renderer truth 또는 독립 사람 annotation이 소유한다. 평가 대상 OCR provider가 만든
confidence·reading order·bbox를 정답으로 다시 사용하지 않는다.

- source document·page·image SHA-256
- page width·height·DPI
- stable token/span ID
- exact text
- gold reading order와 그 생성·검수 정책 version
- bbox 또는 계약이 지원하는 geometry
- table·row·cell ID와 관계
- field candidate가 참조한 span ID
- crop이 있다면 page·crop hash와 좌표 변환

### 12.2 OCR prediction ledger

평가 대상 provider 출력은 별도의 prediction ledger로 기록한다.

- OCR provider·engine·model·version
- model asset lock과 hash
- source page·image hash
- prediction token/span ID와 text·confidence
- provider reading order와 정책 version
- prediction bbox·table·row·cell
- retry·crop·fallback provenance

평가기는 Gold와 prediction을 source image hash·page·metric contract로 결박해 비교한다. provider
prediction을 Gold로 승격하거나 provider 순서로 Gold를 재정렬하지 않는다.

현재 runtime envelope와 다른 page 수·해상도·geometry를 요구하려면 기존 계약을 몰래 확장하지
않고 새 contract version을 정의해야 한다.

### 12.3 대표성 축

최소한 다음 변형을 독립적으로 조절할 수 있어야 한다.

- 여러 CI·PL·B/L·C/O template
- 글꼴·크기·자간·행간
- 한국어·영어·숫자·기호 혼합
- 단일·다중 page
- 표의 row/column 수와 merged cell
- scan·camera 촬영
- DPI·compression·noise
- 회전·perspective·crop
- blur·shadow·조명
- stamp·signature·handwriting
- 겹치는 문자·선·도장
- 빈 셀·잘린 셀·페이지 경계 표

clean digital render와 degraded render를 같은 평가 분모에 무표시로 섞지 않는다.

### 12.4 사람 검수

Release subset은 OCR/VLM과 독립된 사람이 다음을 확인해야 한다.

- 전체 text와 critical identifier
- reading order
- field-to-span 연결
- table row/cell 구조
- bbox와 page/image hash
- reviewer kind, 시각, disagreement와 adjudicator

사람이 보지 않은 renderer-generated layout Gold는 renderer 회귀에는 사용할 수 있지만 실제
문서 OCR 일반화의 최종 증거가 아니다.

### 12.5 현재 프로젝트 목표 임계값

| 지표 | 목표 |
|---|---:|
| Critical identifier exact accuracy | 1.0 |
| OCR CER | 0.01 이하 |
| Table row F1 | 0.98 이상 |
| Table cell F1 | 0.98 이상 |
| Bbox F1 @ IoU 0.5 | 0.95 이상 |
| Crash·unhandled document | 0 |

이 임계값은 적격한 sealed holdout과 승인된 provider에서 측정해야 한다. 현재 공개 development
평가 결과를 Release PASS로 승격할 수 없다.

---

## 13. Evidence/RAG Gold

Evidence conformance와 실제 RAG 품질을 분리한다.

### 13.1 Evidence conformance fixture

다음을 검증한다.

- readiness와 retrieval HTTP 계약
- schema/release/revision/build identity
- request/response header 결박
- request ID와 snapshot 일치
- partial/failed/ambiguous의 fail-closed 처리

sealed test provider의 성공은 네트워크·계약 상호운용성을 증명하지만 실제 법령 검색 품질을
증명하지 않는다.

### 13.2 RAG quality Gold

실제 RAG 평가 Case는 다음 exact identity에 결박한다.

- 공식 source manifest와 source hash
- law ID·article ID·effective date
- HSK·CCCT snapshot
- 58-law target을 충족한 sealed Knowledge generation
- PostgreSQL/Qdrant/Neo4j projection과 attestation
- embedding model·revision·dimension
- index·hybrid search·rerank config
- 기대 relevant law/article/citation 집합
- review date와 applicability
- 사람 검수·승인 record

합성 형식 확인용 HS Code나 placeholder를 법적 분류·검색 정확도 Gold로 사용하지 않는다.
현재 Knowledge corpus가 불완전하면 올바른 기대 결과는 성공 retrieval을 꾸미는 것이 아니라
readiness 실패와 `needs_review`다.

---

## 14. Split과 Release holdout

### 14.1 일반 split 규칙

- Case 단위로 분리한다.
- 같은 Case의 clean·mutation·render variant를 같은 split에 둔다.
- 동일 거래 entity, template, generator family가 split 사이에 누출되는지 검사한다.
- 모델·prompt·rule 선택에 사용한 데이터는 Release holdout에서 제외한다.
- 데이터 수량만으로 split 적격성을 주장하지 않는다.

### 14.2 권장 역할

```text
Development
→ 구현·디버깅·prompt/rule 선택에 반복 사용

Validation
→ 선택한 구현의 일반화·회귀 확인

Sealed Release Holdout
→ 사전 접근 없이 승인된 시점에만 실행
```

비율은 데이터 크기와 규칙 coverage에 따라 결정한다. `60/20/20`은 가능한 개발 구성일 뿐
완전성 또는 Release 적격성의 충분조건이 아니다.

### 14.3 현재 evaluation-v2의 경계

현재 evaluation-v2는 membership을 공개된 정보에서 도출할 수 있고 이미 개발 평가에 사용됐기
때문에 영구적으로 Release holdout이 아니다. authorization receipt, 파일명 변경, 새 descriptor,
“locked” 또는 “sealed” 표기로 unseen 상태를 복원할 수 없다.

### 14.4 evaluation-v3

replacement Release holdout은 저장소 밖의 human-supplied sealed/unseen bundle이어야 한다.
공개 manifest에는 다음만 허용한다.

- dataset/evaluation ID와 version
- sealed bundle SHA-256과 byte size
- Case count
- 외부 사람이 공급했다는 provenance
- ordinary development tooling 사전 접근 없음
- 공개 deterministic seed로 생성하지 않았음

다음은 공개 manifest와 GitHub에 넣지 않는다.

- Case ID와 membership
- selection·rebalance·holdout seed
- 원문·이미지
- Gold 값
- 복호화 정보

secure storage, one-time human authorization, access receipt, isolated runner, 결과 redaction이 모두
준비되기 전에는 Release holdout PASS를 생성하지 않는다. 세부 경계는
[Release Holdout Protocol](RELEASE_HOLDOUT_PROTOCOL.md)을 따른다.

---

## 15. Schema versioning

기존 source-v1 schema를 새 요구에 맞게 조용히 수정하지 않는다.

| 현재 자산 | 재사용 가능 범위 | 새 version이 필요한 이유 |
|---|---|---|
| Extraction Gold v1 | source-v1 CI/PL/B/L raw string | C/O·normalized value·span 참조 필요 |
| Validation Gold v1 | non-scoreable source-v1 진단 | scoreable clean/intentional defect 필요 |
| Evaluation split v2 | 개발·감사 이력 | 공개 membership이라 Release 부적격 |
| Image adjudicated Gold v1 | 무효화 회귀 차단 | 독립 사람 Gold 구조 필요 |
| Sealed public manifest v1 | 기존 대상의 공개 descriptor | synthetic-v2 identity 지원 필요 |
| OCR ledger v2 | 현재 runtime envelope | envelope·geometry 변경 시 새 계약 필요 |

Breaking change는 기존 v1/v2를 덮어쓰지 않고 새 version으로 추가한다.

권장 신규 계약:

- `synthetic-case-master/v2`
- `document-extraction-gold/v2`
- `document-layout-gold/v2`
- `document-validation-gold/v2`
- `controlled-mutation/v2`
- `evidence-rag-gold/v2`
- `agent-outcome-gold/v2`
- `release-holdout-public-manifest/v2`

---

## 16. 품질 Gate

### Gate D0 — Provenance

- dataset ID·version·owner 존재
- generator·template·font·license 존재
- source·Case·document hash 존재
- raw 재배포·외부 전송 권리 확인

### Gate D1 — Schema·grain

- schema validation 100%
- Case·document·variant·mutation ID 중복 0
- 명시적 grain 위반 0
- required Gold 누락 0

### Gate D2 — Integrity

- manifest 참조 누락 0
- orphan document/span/finding 0
- parent-child hash 불일치 0
- file checksum 불일치 0

### Gate D3 — Clean validity

- clean baseline rule 위반 0
- ISO·equipment·amount·unit·date invariant 실패 0
- CI–PL–B/L–C/O 관계 오류 0
- 검증되지 않은 HS·Evidence 참조 0

### Gate D4 — Mutation

- mutation manifest coverage 100%
- parent hash·before·after 존재 100%
- single-defect의 예상하지 않은 추가 rule 위반 0
- positive와 negative control 누락 0
- 의도 미확인 항목의 scoring 포함 0

### Gate D5 — Gold·review

- scoreable observation의 `ambiguous/coverage_gap` 0
- critical field의 사람 검수 coverage 100%
- reviewer independence와 adjudication record 존재
- Finding–source span–Evidence 연결 누락 0

### Gate D6 — Split·leakage

- Case/variant split leakage 0
- template/entity/generator leakage 검사 완료
- Release holdout 사전 접근 0
- 개발 결과를 Release metric으로 승격한 사례 0

### Gate D7 — Safety

- PII·secret·token·DSN 0
- prompt injection 지시 수행 0
- raw image/base64의 result·log 포함 0
- 권리 미확인 raw의 Git/GitHub 게시 0

### Gate D8 — Agent evaluation

- 문서 유형 정확도 1.0
- critical identifier exact accuracy 1.0
- required field exact accuracy 0.99 이상
- hallucinated required field 0
- line-item/table cell F1 0.98 이상
- bbox F1 @ IoU 0.5 0.95 이상
- controlled defect precision/recall 1.0
- 잘못된 자동 완료 0
- 근거 없는 High/Critical Finding 0
- Evidence citation metadata completeness 1.0
- 적격한 실제 RAG-quality set에서 Hit@5 1.0, Recall@10 1.0, nDCG@10 0.8 이상

Gate 결과에는 dataset version·Gold hash·model·prompt·OCR·Data generation·Agent commit을 함께
기록한다.

### Gate D9 — 표본과 metric 계약

데이터 건수만으로 완전성을 판정하지 않지만, 분모가 너무 작거나 특정 계층이 비어 있는 평가도
허용하지 않는다. 각 evaluation descriptor는 실행 전에 다음을 동결한다.

- 전체 Case·문서·page·field·table·Finding 분모
- 문서 유형·rule·severity·template·언어·image degradation별 최소 분모
- clean positive·defect positive·negative control의 최소 분모
- micro/macro 집계 방식과 중복 observation 처리
- null·unknown·not-applicable·zero-denominator 정책
- confidence interval 또는 목표 실패율에 맞춘 표본 설계 근거
- metric ID·version·계산식·threshold

필수 계층의 분모가 0이거나 동결한 최소 분모보다 작으면 점수를 1.0으로 만들지 않고
`INSUFFICIENT_SAMPLE`로 Gate를 실패시킨다. 규칙당 한 건의 성공을 Production 일반화 근거로
사용하지 않는다. 정확한 최소 Case 수는 목표 실패율, 허용 오차, rule prevalence와
대표성 계층에 따라 평가 계획에서 계산하며 이 명세가 임의의 보편 숫자를 정하지 않는다.

---

## 17. 저장·GitHub 게시 정책

### 17.1 로컬 비공개 저장

다음은 원칙적으로 Git에서 제외한 private artifact 영역에 둔다.

- raw JSON·PDF·이미지
- business-valued 문서
- derived render·OCR ledger·Gold bundle
- 평가 원본·모델 raw response
- sealed holdout과 복호화 정보
- 권리·PII 검토 전 자료

### 17.2 Agent 저장소에 게시 가능한 것

- 이 명세
- 비민감 JSON Schema
- generator·validator·evaluator 코드
- raw 값을 포함하지 않는 dataset descriptor와 hash lock
- 권리가 확인되고 비식별이며 처음부터 공개 목적으로 생성한 소형 synthetic fixture
- redacted 평가 요약

소형 fixture 예외는 license·owner·generator provenance, PII/secret scan, 공개 의도와 파일
hash를 모두 통과해야 한다. 기존 권리 미확인 raw에서 일부 행이나 이미지를 잘라 공개 fixture로
전환하지 않는다.

### 17.3 게시 금지

- `_local/data` canonical 이전을 완료한 `trade_documents_images` raw
- `_local/data` canonical 이전을 완료한 `가상데이터(CI_PL_BL)` raw
- private artifact 전체
- 개인 엔지니어링 기록
- 로컬 절대경로
- API key·token·DSN
- 재배포 권리가 확인되지 않은 자료

raw가 GitHub에 없더라도 schema·generator·검증 명령과 공개 descriptor만으로 팀원이 데이터
계약과 생성 방식을 검토할 수 있어야 한다.

raw의 유일한 로컬 권한 경로는 `_local/data/<dataset-id>/source-v1`이다. 실행 전에
Agent-owned relocation binding, workspace relocation manifest, DatasetProvenanceV2, inventory·checksum,
폐기된 root와 `_artifacts/datasets` 경로의 부재를 모두 검증한다. 중복 원본이 다시
생기면 편의상 fallback하지 않고 fail-closed한다.

---

## 18. 현재 두 데이터셋의 역할

현재 등록 상태를 다음과 같이 해석한다.

| 데이터셋 | 파일 무결성 | 현재 핵심 한계 | 정확한 역할 |
|---|---|---|---|
| 구조화 CI/PL/B/L source-v1 | 등록 raw와 hash 일치 | 작성자 Case ID·결함 의도·HS·C/O truth 없음 | 텍스트·관계·규칙 개발 |
| 이미지 source-v1 | manifest·3,000 image 연결 정상 | 참조된 master truth 누락, scoreable image Gold 0 | ingestion·비점수 진단·annotation 후보 |
| 공개 evaluation-v2 | 개발용 hash·membership 고정 | membership 도출 가능, 이미 개발 평가에 사용 | non-release 회귀 이력 |
| rendered synthetic documents | 결정론적 생성·layout export | 실제 scan 대표성과 사람 검수 부족 | renderer·layout 개발 |

이미지 source-v1의 과거 12-Case 파생 Gold는 원본과 다른 식별자 라벨이 확인돼 전체가
무효화됐다. 이것은 원본 이미지 파일의 손상과 다른 문제다. 무효 Gold를 경로 변경이나
재생성으로 되살리지 않고, 독립 사람 검수 replacement가 생기기 전까지 점수화를 차단한다.

### 18.1 구조화 CI/PL/B/L source-v1

현재 안전한 역할:

- source-exact text extraction 개발
- dataset-local index Pairing 회귀
- CI–PL–B/L 공통 필드 비교
- ambiguous·coverage gap의 fail-closed 처리
- 별도로 manifest가 있는 controlled mutation 개발 평가
- 결정론적 renderer 개발

현재 금지하는 주장:

- 모든 Case가 clean baseline
- 관측된 이상이 모두 의도된 결함
- HS/RAG/C/O 법적 정확도
- 실제 통관 문서 대표성
- 공개 development split의 Release 적격성

### 18.2 이미지 source-v1

현재 안전한 역할:

- 파일 ingestion·decode·MIME·dimension·hash 검사
- manifest와 파일 연결
- 비점수 분류·identifier debugging
- 무효 Gold를 거부하는 fail-closed 회귀
- 향후 사람 annotation 후보 모집단

현재 금지하는 주장:

- 전체 field/OCR/layout Gold 존재
- OCR CER·bbox·table·field 정확도
- 과거 12-Case Gold 기반 성능
- 실제 스캔 문서 Production 일반화

### 18.3 두 데이터셋의 관계

두 source는 Case 수와 provenance가 다른 독립 데이터셋이다. authoritative Case Master와 hash
mapping이 없으므로 번호·문자열 유사성으로 서로 결합하지 않는다.

새 synthetic-v2는 하나의 Case Master에서 structured source와 rendered documents를 함께
만들어 이 문제를 해결한다.

---

## 19. 신규 synthetic-v2 생성 순서

### Phase 1 — 계약

1. 평가 용도와 지원 문서 유형을 고정한다.
2. Case Master v2 schema를 만든다.
3. Gold·mutation·split·public manifest의 새 versioned schema를 만든다.
4. Data Platform이 제공해야 할 Evidence identity를 합의한다.

### Phase 2 — Clean master

1. 검증된 party·route·item·amount·container·HS source를 준비한다.
2. 모든 clean invariant를 자동 검사한다.
3. Case Master와 생성 provenance를 hash로 고정한다.
4. Case 수가 아니라 rule·template·entity·언어 coverage를 검토한다.

### Phase 3 — 문서·layout

1. CI·PL·B/L·C/O를 동일 Case Master에서 생성한다.
2. 여러 template와 page 구성을 만든다.
3. clean PDF·PNG·JPEG와 span·bbox·table Gold를 동시에 출력한다.
4. clean 결과 hash와 parent-child manifest를 생성한다.

### Phase 4 — Defect·ambiguity

1. clean semantic document에 rule별 single mutation과 negative control을 적용한다.
2. 각 semantic variant를 다시 rendering하고 variant용 Extraction/Layout Gold를 생성한다.
3. 복합 defect를 별도 세트로 만들고 같은 절차로 rendering·Gold를 생성한다.
4. visual degradation은 semantic mutation과 분리해 좌표 변환·사람 검수를 적용한다.
5. 누락·중복·Evidence partial·prompt injection을 만든다.
6. 모든 mutation의 variant ID·output hash·기대 Finding·Evidence·Outcome을 기록한다.

### Phase 5 — 검수

1. schema·integrity·business invariant를 자동 검사한다.
2. 사람 reviewer가 critical field와 image/layout subset을 독립 검수한다.
3. disagreement를 adjudicator가 판정한다.
4. generator error와 ambiguous를 scoring에서 격리한다.

### Phase 6 — 평가 분리

1. development와 validation을 Case 단위로 분리한다.
2. 모든 variant를 parent Case와 같은 split에 둔다.
3. leakage 검사를 실행한다.
4. 별도 평가 담당자가 evaluation-v3 sealed bundle을 준비한다.

### Phase 7 — Release 증거

1. exact Agent wheel·model·prompt·OCR provider·Data generation을 고정한다.
2. 한 번 승인된 sealed holdout에서 평가한다.
3. JUnit·metrics·model usage·checksums·승인 record를 한 run에 봉인한다.
4. 미검증 범위와 blocker를 포함해 Release 판정을 기록한다.

---

## 20. 완료 정의

다음을 모두 만족할 때 해당 **평가 범위에 대해** 가상데이터가 완전하다고 판정한다.

- [ ] 목적과 평가 grain이 명시됐다.
- [ ] Case Master와 모든 ID가 안정적이고 고유하다.
- [ ] raw·source·case·document·page hash가 고정됐다.
- [ ] generator·template·font·model·seed·license provenance가 있다.
- [ ] clean baseline의 모든 불변식이 통과한다.
- [ ] Extraction·Normalization·Layout·Validation·Evidence·Outcome Gold가 분리됐다.
- [ ] 모든 scoreable defect에 의도·before/after·rule·expected result가 있다.
- [ ] ambiguous·coverage gap이 score 분모에서 제외됐다.
- [ ] 모든 reference가 해소되고 orphan이 없다.
- [ ] 사람 검수와 adjudication이 필요한 범위가 완료됐다.
- [ ] parent·variant split leakage가 없다.
- [ ] 평가별 최소 분모·집계·zero-denominator·metric version 계약이 동결되고 Gate D9를 통과했다.
- [ ] Release holdout이 사전 비공개 상태다.
- [ ] Data/RAG Gold가 sealed Knowledge generation에 결박됐다.
- [ ] raw·secret·PII·권리 미확인 자료가 GitHub에 없다.
- [ ] exact commit과 artifact hash로 평가 결과를 재현할 수 있다.
- [ ] 합성 데이터가 증명하지 못하는 Production 한계를 명시했다.

체크리스트를 통과하지 못한 데이터도 폐기할 필요는 없다. `development fixture`,
`candidate_unverified`, `diagnostic-only`, `non-release` 등 정확한 역할로 분류해 사용할 수
있다.

---

## 21. 의사결정 요약

| 질문 | 결정 |
|---|---|
| 기존 두 원본을 수정하는가? | 아니오. source-v1은 불변 보존한다. |
| 두 원본을 하나로 합치는가? | 아니오. authoritative mapping 없이는 독립 dataset으로 유지한다. |
| 잘못된 값도 Extraction Gold가 될 수 있는가? | 예. 문서에 실제로 적혀 있다면 raw extraction Gold다. |
| 이상값을 의도된 결함으로 자동 간주하는가? | 아니오. 의도 확인 전에는 non-scoreable이다. |
| 두 AI reviewer 합의로 Release Gold가 되는가? | 아니오. 독립 사람 검수와 adjudication이 필요하다. |
| 공개 evaluation-v2를 Release holdout으로 쓰는가? | 아니오. 영구 development-only다. |
| 현재 합성 C/O·HS로 법령 정확도를 평가하는가? | 아니오. 검증된 Data generation과 연결된 Gold가 필요하다. |
| 가상데이터를 GitHub에 올리는가? | raw는 올리지 않고 schema·code·descriptor·문서만 올린다. |
| 합성 데이터만으로 Production-ready를 선언하는가? | 아니오. 실제·사람 검수 대표셋과 운영 Gate가 별도 필요하다. |

---

## 관련 문서

- [Release Holdout Protocol](RELEASE_HOLDOUT_PROTOCOL.md)
- [OCR Release Freeze](OCR_RELEASE_FREEZE.md)
- [Data Provider Handoff](DATA_PROVIDER_HANDOFF.md)
- [Evaluation Guide](../evals/README.md)
- [Trade Document Triplets Evaluation](../evals/trade_document_triplets/v1/README.md)
- [Trade Document Images Evaluation](../evals/trade_documents_images/v1/README.md)
