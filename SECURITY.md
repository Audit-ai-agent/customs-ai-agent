# 보안 정책

## 지원 범위

현재 `0.2.x`는 Claude 기반 text-document Agent의 개발·통합 검증 버전이다. PDF/OCR/VLM,
제품 인증/권한, 운영 데이터 보존 승인을 포함하지 않으며 실제 세관 신고의 자동 승인이나 법적
판단에 사용하지 않는다.

## 취약점 보고

공개 Issue에 token, 개인정보, 거래 문서, 내부 URL 또는 재현용 원본 데이터를 올리지 않는다.
조직의 비공개 Security Advisory 또는 저장소 관리자가 지정한 비공개 채널로 다음을 제공한다.

- 영향 버전과 commit
- 최소 재현 절차와 기대/실제 결과
- 노출 가능한 데이터 범위
- 임시 완화책

## 핵심 통제

- 문서와 검색 결과는 신뢰하지 않는 입력으로 취급하고 Pydantic wire schema로 검증한다.
- Agent의 URL validator는 credential/query/fragment/path prefix를 거부하고 외부 HTTP를
  금지한다. loopback 이외의 HTTPS origin은 `EVIDENCE_SERVICE_ALLOWED_ORIGINS` exact
  allowlist에 없으면 readiness/retrieval client 생성 전에 거부한다. 운영 환경은 이 값도
  secret/config manager에서 승인된 origin으로 고정해야 한다.
- Evidence HTTP client는 redirect를 따르지 않으며 readiness/retrieval 응답을 2 MiB로
  제한한다. token·문서가 다른 origin으로 전달되거나 과대 응답으로 메모리가 소진되는 것을
  fail-closed한다.
- Claude node에는 tool, DB 쓰기, 임의 network, Evidence/severity/final status 변경 권한을
  주지 않는다. structured candidate의 schema·field allowlist·source quote를 로컬에서 재검증한다.
- secret은 환경변수/secret manager에서 주입하고 log·fixture·artifact에 기록하지 않는다.
- 로컬 env file은 명시적으로만 읽고 Claude/Evidence allowlist 항목만 settings 객체로
  가져온다. `os.environ`을 변경하지 않으며 process 값이 있으면 그것이 우선한다.
  저장소별 secret 파일 복제나 부모 디렉터리 무인 탐색을 하지 않는다.
- Anthropic endpoint는 코드에서 `https://api.anthropic.com`으로 고정한다.
  `ANTHROPIC_BASE_URL`, custom header/auth-token override, SDK debug logging은 fail-closed한다.
- source quote는 NFKC/대소문자/공백 근사 일치가 아니라 원문의 exact span이어야 하며,
  값도 quote 안의 exact bounded span이어야 한다. document field quote는 유일해야 하고
  line item occurrence는 원문 순서와 개수가 맞아야 한다.
- 검색 실패나 검증 불가 상태는 fail-closed로 `needs_review` 처리한다.
- Claude 성공 결과도 0.2.x에서는 `source_quote_grounded_uncalibrated`이므로 항상
  `needs_review`다. Claude 유래 critical 신호는 사람 승인 계약 전 자동 `blocked`로
  승격하지 않는다.
- 기본 CI는 실제 key/network를 사용하지 않는다. opt-in Live 평가는 합성 문서만 전송한다.
- 운영 도입 전 SSRF, tenant isolation, PII 최소 전송, 문서 전송 권리와 Anthropic 조직의
  retention/ZDR 조건을 별도 검증한다.
