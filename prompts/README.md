# Prompts

실제 runtime prompt는 wheel에 포함되는
`src/customs_ai_agent/prompts/document_analysis_v1.txt`에 있다.

- prompt ID: `customs_document_analysis`
- version: `1.1.0`
- provider/model default: Anthropic / `claude-sonnet-5`
- tools: 없음
- output: `ClaudeDocumentCandidate` structured JSON
- 입력: untrusted metadata/document text를 담은 JSON envelope
- 권한: 문서 분류와 case/whitespace/Unicode까지 정확한 verbatim candidate 추출만
- 금지: 법률 판단, Evidence, severity, finding, final status, 외부 지식, 문서 내부 지시 수행

Prompt 변경은 version bump와 synthetic gold/fake transport/grounding 회귀를 함께 수행한다. 코드가
저장소 최상위 prompt 파일을 runtime에서 읽지 않도록 package resource를 사용하며 wheel build에서
포함 여부를 검증한다.
