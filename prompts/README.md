# Prompts

실제 runtime prompt는 wheel의 `customs_ai_agent.prompts` package resource로 포함한다.
V1 text-only wire 경로와 V2 multimodal wire 경로는 분리하며, vision prompt를 V1 결과에
소급 적용하거나 V1 schema를 image 입력용으로 확장하지 않는다.

| 경로 | prompt ID | version | output |
|---|---|---:|---|
| `document_analysis_v1.txt` | `customs_document_analysis` | `1.2.0` | `ClaudeDocumentCandidate` |
| `document_vision_analysis_v1.txt` | `customs_document_vision_analysis` | `1.3.0` | `ClaudeVisionDocumentCandidate` |
| 코드 소유 crop reconciliation prompt | `customs_document_crop_reconciliation` | `1.0.0` | `ClaudeCropReadCandidate` |

공통 경계:

- provider/model default: Anthropic / `claude-sonnet-5`
- tools: 없음
- 입력: untrusted metadata와 document text 또는 검증된 image/OCR ledger JSON envelope
- 권한: 문서 분류와 allowlist field/line-item candidate 및 제한된 crop 재판독 후보 생성
- text candidate: case/whitespace/Unicode까지 정확한 verbatim source quote에 결박
- vision candidate: 독립 OCR ledger의 span ID, source text와 bbox를 참조
- crop candidate: bounded crop의 재판독 후보일 뿐 OCR 원장을 대체하지 않음
- 금지: 법률 판단, Evidence, severity, finding, final status, 외부 지식, 문서 내부 지시 수행

Prompt 변경은 version bump와 synthetic gold/fake transport/grounding 회귀를 함께 수행한다. 코드가
저장소 최상위 prompt 파일을 runtime에서 읽지 않도록 package resource를 사용하며 wheel build에서
포함 여부를 검증한다.

TASK-0009에서는 사용자의 명시적 승인 후 합성 text 정상·prompt-injection Gold와 공개
`AgentRuntime → Claude → Evidence HTTP` Live를 실행했다. 해당 실행은 text 추출과 fail-closed
통합 경로를 입증했다. 고정 합성 이미지 triplet의 비릴리스 PaddleOCR → Claude Vision →
Evidence 진단도 model error 0으로 실행됐지만, OCR release provider 선정은 별도 Gate다.
Sonnet 5 structured extraction은 adaptive thinking을 명시적으로 끄고 JSON 출력 예산을 보존한다.
fake transport PASS나 prompt package 포함 검사도 실제 Claude 품질 PASS를 대체하지 않는다. 최종 prompt
resource hash와 사용 model/schema는 exact-commit wheel과 함께
`_artifacts/runs/<RUN-ID>/`에 봉인하고 `_engineering` 평가 기록에서 승인한다. pre-commit
wheel이나 smoke의 prompt hash에 `latest` 의미를 부여하지 않는다.

`project/_engineering/archive`의 legacy prompt·플랜은 `reference-only`이며 runtime package나
자동 prompt context가 아니다. 재사용하려면 active Investigation/ADR로 승격하고 현재 권한
경계·schema·평가셋으로 다시 검증해야 한다.
