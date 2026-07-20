# Agent Request/Result 계약

이 디렉터리는 AI Agent가 소유하고 Backend가 소비하는 공개 wire 계약이다. Data Platform이
소유하는 `contracts/evidence/` consumer snapshot과 소유권이 다르다.

## 버전 의미

- `v1`: 기존 텍스트 `ReviewRunRequest`와 `schema_version=1.0` 공개 결과를 보존한다.
- `v2`: Backend 서비스 전용 요청은 필수 `case_id`, 비어 있지 않은 `documents`, 명시적
  `llm_mode="llm"`을 요구한다. 한 문서는 text/image/PDF 중 정확히 하나만 가지며,
  scenario/mock 호환 필드는 포함하지 않는다. 공개 결과는 `schema_version=2.0`이다.
- V1 파일을 제자리에서 breaking change하지 않는다. 호환되지 않는 변경은 새 버전 디렉터리로
  추가한다.
- JSON Schema는 wire shape를 검증한다. 실행 시에는 Pydantic 도메인 검증도 적용되어 이미지
  byte/hash/dimension/provenance, 전체 page/byte 한도, `llm_mode=llm` 조건을 다시 확인한다.

## 생성과 검증

```powershell
uv run --locked customs-ai-agent export-schema --version v1 --kind request `
  --out contracts/agent/v1/review_request.schema.json
uv run --locked customs-ai-agent export-schema --version v1 --kind result `
  --out contracts/agent/v1/review_result.schema.json
uv run --locked customs-ai-agent export-schema --version v2 --kind request `
  --out contracts/agent/v2/review_request.schema.json
uv run --locked customs-ai-agent export-schema --version v2 --kind result `
  --out contracts/agent/v2/review_result.schema.json

uv run --locked customs-ai-agent validate-contract --version v2 --kind request `
  --input request.json
```

CI는 생성 결과와 체크인 파일을 byte 의미상 비교한다. 결과 계약에는 원문 문서,
`content_text`, `image_pages`, `pdf_pages`, `data_base64`, 로컬 source/manifest/OCR-ledger
path를 포함하지 않는다. Backend는 선택한 Agent wheel commit과 이 계약 버전을 함께 고정해야
한다.

이 디렉터리는 설치 전 코드 리뷰를 위한 `backend` scope 사본이다. 배포된 wheel은 같은 계약을
package resource로 포함하며 소비자는 다음 API로 schema와 exact resource hash를 얻는다.

```python
from customs_ai_agent.contracts import get_schema, schema_sha256

schema = get_schema("backend", "v2", "review-result")
digest = schema_sha256("backend", "v2", "review-result")
```

Backend V2 request는 서비스 전용 계약이다. `case_id`, 비어 있지 않은 `documents`, 명시적
`llm_mode="llm"`이 모두 필수이며 text/image/PDF 문서에 같은 조건을 적용한다. scenario와
mock mode는 runner·CLI·evaluation 내부 호환 기능일 뿐 이 Backend wire에는 존재하지 않는다.
JSON Schema에 없는 필드, `scenario_id: null`, 누락 또는 null인 필수값도 모두 거절한다.
Backend는 untrusted request body에 32 MiB streaming 상한을 먼저 적용한 뒤 원본 bytes를
`AgentRuntime.review_json()`에 전달해야 한다. JSON을 먼저 일반 dict로 바꾸면 duplicate-key
정보를 복원할 수 없다. 신뢰된 in-process 호출만 `AgentRuntime.review()`를 사용한다.

JSON Schema 통과만으로 실행 가능한 요청이 되는 것은 아니다. Schema는 각 문서·페이지의 wire
shape를 표현하고 runtime은 nested image decode/PIL보다 먼저 다음 run 합산 Gate를 추가 적용한다.

- 문서 최대 20개
- 전체 text 최대 500,000 characters
- image/PDF raster page 전체 최대 10개
- decoded image bytes 전체 최대 20 MiB
- declared byte length와 base64 encoded length의 합산·상호 일치

따라서 Backend는 Schema 검증 성공을 최종 수용으로 처리하지 않고 `AgentRuntime`의 semantic
validation 결과까지 확인해야 한다. 문자열 숫자와 date-time을 Pydantic이 암묵 변환하지 않도록
runtime과 `validate-contract` CLI는 packaged JSON Schema와 `date` format을 먼저 검증한다.

`local/v1/review-manifest`와 `internal/v1/*`는 wheel에는 함께 포함되지만 Backend 공개 계약이
아니다. Backend가 로컬 manifest path 또는 OCR/Claude 내부 candidate를 직접 소비하면 경계
위반이다.
