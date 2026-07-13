# 보안 정책

## 지원 범위

현재 `0.1.x`는 개발·통합 검증용 MVP다. 실제 세관 신고의 자동 승인이나 법적 판단에 사용하지 않는다.

## 취약점 보고

공개 Issue에 token, 개인정보, 거래 문서, 내부 URL 또는 재현용 원본 데이터를 올리지 않는다.
조직의 비공개 Security Advisory 또는 저장소 관리자가 지정한 비공개 채널로 다음을 제공한다.

- 영향 버전과 commit
- 최소 재현 절차와 기대/실제 결과
- 노출 가능한 데이터 범위
- 임시 완화책

## 핵심 통제

- 문서와 검색 결과는 신뢰하지 않는 입력으로 취급하고 Pydantic wire schema로 검증한다.
- Evidence Service URL은 운영 설정/allowlist로 고정하며 사용자 임의 URL을 허용하지 않는다.
- LLM/VLM node에는 DB 쓰기, 임의 network, 최종 상태 변경 권한을 주지 않는다.
- secret은 환경변수/secret manager에서 주입하고 log·fixture·artifact에 기록하지 않는다.
- 검색 실패나 검증 불가 상태는 fail-closed로 `needs_review` 처리한다.
- 운영 도입 전 prompt injection, SSRF, tenant isolation, PII 최소 전송과 provider retention을 별도 검증한다.
