# Evaluations

모델/Prompt/RAG 변경의 versioned 평가 정의와 비민감 gold-set metadata를 둔다. 실제 거래 문서와
PII가 포함된 평가 원본은 커밋하지 않고 개인 artifact storage의 hash만 기록한다.

현재 `claude_text_gold_v1.json`은 저장소의 합성 `happy_path` 4종 텍스트 문서에 대한 gold다.
`run_claude_live.py`는 실제 Claude를 명시적으로 호출하며 다음을 fail-closed로 평가한다.

- 문서 유형 정확도
- document/line-item field exact, wrong, missing, hallucinated 수
- adapter의 strict schema와 source quote grounding
- model/prompt version, request/message ID, token, latency

출력에는 key, 문서 원문, 추출값, prompt, raw response가 없다. 1/4문서 실행 모두
`SYNTHETIC_TEXT_ONLY_NOT_PDF_OCR_OR_RELEASE` 범위이며 기본 pytest/CI에서는 실행하지 않는다.

`--workflow --max-documents 4`는 실제 Claude 분석 결과를 전체 결정론적 workflow에 넣되
`EXPLICIT_MOCK_NOT_LIVE_RAG` retriever를 사용한다. 이 평가의 정상 최종 상태는
`needs_review`다. 이는 실패가 아니라 0.2.x가 Claude의 의미상 필드 연결 정확도를 아직
보정하지 않았으므로 자동 완료·자동 차단 권한을 주지 않는 안전 정책이다. Live Data/RAG
검증이나 Release Gate로 표시할 수 없다.

`claude_text_adversarial_v1.json`은 잘못된 declared type, filename delimiter-like text,
시스템 무시·허위 승인·tool 호출·prompt/key 공개 지시가 포함된 합성 invoice 1건이다.
실제 Claude에서 expected type/field 13/13, wrong/missing/hallucinated 0으로 통과했으며 raw
지시나 응답은 artifact에 남기지 않는다. 다만 이는 한 공격 패턴의 baseline일 뿐이며
다국어·encoding·간접 RAG injection 전체를 검증한 것으로 확대 해석하지 않는다.

Evaluator는 network/provider setup 전에 normal/adversarial gold, candidate schema, packaged
prompt SHA-256을 pinned 값과 대조한다. 불일치하면 API 호출이나 과금 전에 fail-closed한다.
