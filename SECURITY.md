# 보안 정책

## 지원 범위

`0.3.0`은 합성 text/image/PDF를 이용한 AI Agent release-candidate 목표다. 제품 인증·인가,
tenant isolation, 운영 문서 저장·삭제, Backend/Frontend, 배포 승인을 포함하지 않는다.
exact-commit Gate가 끝나기 전에는 Agent release candidate로도 확정하지 않으며, 실제 세관 신고의
자동 승인이나 법적 판단에 사용하지 않는다.

## 취약점 보고

공개 Issue에 token, 개인정보, 거래 문서, 원문 image/PDF, 내부 URL, OCR ledger 또는 재현용 원본
데이터를 올리지 않는다. 조직의 비공개 Security Advisory 또는 관리자가 지정한 비공개 채널로
다음만 제공한다.

- 영향 버전과 commit
- 최소·비식별 재현 절차
- 기대/실제 결과와 노출 가능 범위
- 임시 완화책

## 입력 통제

- 문서, OCR, 모델, RAG 응답은 모두 신뢰하지 않는 입력이다.
- local manifest source는 manifest 디렉터리 아래 상대경로만 허용한다. 절대경로, `..`, symlink,
  NUL, plain filename이 아닌 display label은 거부한다.
- text는 UTF-8, 파일 byte/문자 수를 제한한다.
- image는 실제 decode 결과로 format, byte length, SHA-256, width/height, visual budget을
  검증한다. 선언된 media type만 신뢰하지 않는다.
- PDF는 signature/trailer, polyglot·암호화 여부, 전체 byte, page 수, MediaBox, DPI,
  page/전체 pixel과 rasterized PNG 한도를 render 전에 적용한다. 초과 페이지를 조용히
  생략하지 않는다.
- Pillow decompression bomb, decode 오류, PDF renderer 오류는 원문·경로 없는 safe error로
  바꾼다.
- OCR ledger는 canonical ID와 document/page/image hash/dimension을 현재 요청과 대조한다.
- 제품 CLI 명령 실패는 stderr의 stable error code와 일반화된 message로만 반환한다. 입력 경로,
  provider exception, secret, traceback은 출력하지 않으며 예상 밖 오류는 `internal_error`로
  축약한다.

PDF rasterization은 `python -I` subprocess, neutral working directory, 환경변수 allowlist,
한 줄 bounded JSON protocol과 suppressed stderr 경계에서 실행한다. parent는 worker 응답의
protocol/request ID, PDF hash, page 순서, PNG hash·signature·dimension을 다시 검증한다.

현재 한도는 PDF/전체 raster PNG 20 MiB, page당 PNG 5 MiB, worker request/response 각각
30 MiB, CLI 문서당 5 page, run/protocol 10 page, CLI 180 DPI, protocol 72–600 DPI,
long edge 2,576 px, worker 30초와 종료 grace 2초다.

이것은 process/protocol isolation이지 OS-level sandbox가 아니다. Windows Job Object,
container/seccomp, memory/CPU quota 같은 강제 격리는 운영 전 별도 보안 Gate가 필요하다.

## OCR/VLM 통제

- Paddle/Docling은 격리 subprocess 경계에서 실행한다. OCR worker에 Agent secret과 Evidence
  token을 넘기지 않는다.
- worker의 stdout만 bounded JSON protocol로 처리하며 원문이 포함될 수 있는 stderr/provider
  exception을 공개 audit에 넣지 않는다.
- OCR 미설치, timeout, crash, malformed output, provenance mismatch는 fail-closed한다.
- Claude Vision Live evaluator와 제품 `review` CLI는 canonical 60+20 결합 Gate report의 exact
  hash, PASS provider/engine/wheel, provider별 dependency lock과 공통 model-assets lock을 OCR
  provider와 외부 호출 전에 대조한다. locked-only 또는 smoke report는 이 Gate를 통과하지 못한다.
- 결합 report·caller expected·OCR Python installed wheel hash의 3-way 일치와 설치 Agent
  `RECORD`/console executable 무결성을 요구한다. source/editable install과 untracked executable은
  거부한다.
- Agent clean-install Gate는 wheel을 hash-pinned requirements로 설치하고, verifier가 동일한
  actual wheel의 SHA-256·exact local origin과 wheel/installed RECORD 및 설치 payload를 다시
  대조한다. uv가 local wheel의 `direct_url.json`에 hash를 생략해도 이 대체 증거 전체가
  일치할 때만 허용하며, 단순 `.whl` URL 또는 caller hash만으로 통과시키지 않는다.
- OCR Python의 전체 distribution name/version inventory는 선택 provider의 hash-locked
  requirements와 정확히 같아야 한다. model cache도 lock의 revision, file count, byte count,
  path-free content-tree SHA-256, engine ID와 일치해야 한다. 새 lock 경로만 제시한 기존 환경이나
  old model cache의 과거 PASS는 재사용할 수 없다.
- provider별 dependency lock hash, 공통 model-assets lock hash, engine ID는 코드 소유
  canonical 값이기도 하다. caller/report/runtime의 상호 일치만으로 임의 평가 환경을 release
  환경으로 승격할 수 없다.
- runtime provider가 만든 각 ledger의 provider/engine/version, canonical model-set ID,
  model-assets lock SHA-256도 다시 대조하며, 불일치하면 Claude transport 전에 중단한다.
- Live evaluator의 bounded prefix/subset 성공은 `DIAGNOSTIC_PASS`일 뿐 release PASS가 아니다.
  3,000장 계열 평가는 adjudicated 12 Case/36문서 전체, rendered 평가는 locked 60+C/O 20
  전체의 사전 고정 ID·count·hash가 맞을 때만 release score eligible이다.
- Claude에는 검증된 image bytes와 OCR/layout 관측만 전달한다. arbitrary URL, local path, tool,
  DB write 권한을 주지 않는다.
- structured candidate는 schema, document field allowlist, OCR span, exact text/value,
  page/bbox를 로컬에서 재검증한다.
- crop reread는 검증된 page의 bounded crop과 run-wide budget 안에서만 허용한다.
- 모델이 법률, Evidence ID, requirement, severity, finding, final status를 생성하거나 변경하지
  못한다.

## Prompt/RAG injection 통제

- 문서와 검색 snippet 안의 지시는 데이터이며 시스템 지시로 실행하지 않는다.
- filename과 문서 내용은 JSON envelope로 경계화한다.
- provider tool use와 임의 network를 허용하지 않는다.
- source quote/span/bbox grounding이 없는 field를 downstream에 전달하지 않는다.
- Evidence는 provider schema, request/item ID, context date, HS context와 citation metadata를
  검증한다.

## Network와 secret

- local secret의 단일 기준은 저장소 밖 `project/_secrets/.env`다. repo/worktree별 `.env` 복제를
  금지한다.
- env file은 `--env-file`로만 지정하고 Claude/Evidence allowlist만 읽는다. process 환경을
  변경하지 않는다.
- Anthropic endpoint는 공식 HTTPS endpoint로 고정한다. base URL/custom auth/custom header/debug
  override는 fail-closed한다.
- Evidence URL은 credential/query/fragment/path prefix를 거부한다. loopback 이외에는 HTTPS와
  exact origin allowlist가 필요하다.
- Evidence client는 redirect를 따르지 않고 response body를 제한한다. token은 header에만 둔다.
- 기본 pytest/CI는 key와 실제 network를 사용하지 않는다. Live 평가는 명시적 opt-in이다.
- 합성 fixture도 Claude Live에서는 Anthropic으로 외부 전송된다. 전송 대상·범위·보존 위험을
  고지받은 사용자의 명시적 승인이 없으면 실행하지 않는다.
- `.env`에 key가 있다는 사실 자체는 전송 승인이 아니다. TASK-0009의 text 정상·adversarial 및
  공개 runtime Live는 별도의 명시적 사용자 승인 후 실행했으며, 이미지 전송·재실행도 계속
  명시적인 opt-in과 비용 상한을 요구한다.
- Claude Live evaluator는 SDK `max_retries=0`을 강제하고 `--live`, call/input/output token
  budget을 모두 요구한다. 실패한 provider 시도도 call budget에서 제외하지 않는다.

## 출력과 로그

공개 Agent result와 CLI 표준 출력에는 다음을 남기지 않는다.

- `documents`, `content_text`, `image_pages`, `data_base64`
- manifest/source/OCR-ledger 로컬 경로
- raw prompt, raw Claude/OCR response
- API key, Evidence token, DSN

성공 audit는 provider/model/prompt/version/request ID/token usage/latency와 immutable ledger ID만
허용한다. 실패 audit는 safe code와 제한된 provider status만 허용한다. traceback, SDK response
body, subprocess stderr는 민감한 로컬 artifact로 취급하고 공개 결과에 합치지 않는다.

## 운영 전 필수 검증

저장소 CI의 secret scan, Python dependency vulnerability audit, Python SAST 기준과 동일한 로컬
재현 명령은 [AI Agent Safety Gate](docs/SECURITY_GATE.md)를 따른다. 세 Gate 중 하나라도 도구
실행·외부 advisory 조회·보고서 검증에 실패하면 PASS가 아니다.

- 실제 문서 전송 권리와 최소 수집 원칙
- Anthropic 조직의 retention/ZDR 조건
- 원문 object storage encryption, 권한, 삭제/보존
- OCR/PDF subprocess의 OS-level sandbox와 CPU/memory resource limit
- 악성 image/PDF parser fuzzing
- SSRF, tenant isolation, rate limit, audit access
- dependency/SAST/secret scan
- 공개 포트폴리오의 PII·token·DSN·절대경로 0건

Backend 스택이 아직 정해지지 않았다는 사실은 이 통제를 유예하는 근거가 아니다. Backend 구현을
시작할 때 Spring/Java 또는 FastAPI/Python 선택과 함께 인증·저장·권한 threat model을 별도 ADR과
보안 검증으로 추가해야 한다.

TASK-0009는 Backend·Frontend와 Data Platform source를 변경하지 않는다. Data 접근은 pinned
Evidence HTTP API에 한정한다. `project/_engineering/archive`의 legacy 자료도
`reference-only`이며 prompt, evaluator 또는 runtime의 자동 context로 사용하지 않는다.

pre-freeze 테스트 수, wheel hash와 OCR smoke는 최종 보안/release 판정이 아니다. exact commit의
secret/dependency/SAST 결과와 clean wheel, 60+20 OCR 결합 결과를 동일한
`_artifacts/runs/<RUN-ID>/`에 봉인하고 `_engineering`에서 승인한 기록으로만 최종 상태를
판정한다.
