# Agent Request/Result 계약

이 디렉터리는 AI Agent가 소유하고 Backend가 소비하는 공개 wire 계약이다. Data Platform이
소유하는 `contracts/evidence/` consumer snapshot과 소유권이 다르다.

## 버전 의미

- `v1`: 기존 텍스트 `ReviewRunRequest`와 `schema_version=1.0` 공개 결과를 보존한다.
- `v2`: V1 필드를 유지하면서 검증된 `image_pages` 또는 명시적 `pdf_pages` 입력을 허용하고
  공개 결과를 `schema_version=2.0`으로 식별한다. 한 문서는 text/image/PDF 중 정확히 하나만
  가진다.
- V1 파일을 제자리에서 breaking change하지 않는다. 호환되지 않는 변경은 새 버전 디렉터리로
  추가한다.
- JSON Schema는 wire shape를 검증한다. 실행 시에는 Pydantic 도메인 검증도 적용되어 이미지
  byte/hash/dimension/provenance, 전체 page/byte 한도, `llm_mode=llm` 조건을 다시 확인한다.

## 생성과 검증

```powershell
uv run customs-ai-agent export-schema --version v1 --kind request `
  --out contracts/agent/v1/review_request.schema.json
uv run customs-ai-agent export-schema --version v1 --kind result `
  --out contracts/agent/v1/review_result.schema.json
uv run customs-ai-agent export-schema --version v2 --kind request `
  --out contracts/agent/v2/review_request.schema.json
uv run customs-ai-agent export-schema --version v2 --kind result `
  --out contracts/agent/v2/review_result.schema.json

uv run customs-ai-agent validate-contract --version v2 --kind request `
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

`local/v1/review-manifest`와 `internal/v1/*`는 wheel에는 함께 포함되지만 Backend 공개 계약이
아니다. Backend가 로컬 manifest path 또는 OCR/Claude 내부 candidate를 직접 소비하면 경계
위반이다.
