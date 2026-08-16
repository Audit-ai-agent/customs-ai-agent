# 합성 PDF source-attested 자동 활성화

이 경로는 사람이 PDF를 다시 검수하지 않아도 되는 **합성 PDF 전용** 품질
활성화 경로다. 기존 사람이 만든 합성 source truth와 final PDF의 authored
content stream에서 만든 layout truth를 해시로 결속한다. 사람이 검수한 Gold
경로를 대체하거나 수정하지 않으며, 두 증거를 섞을 수 없다.

## 의미와 범위

- `evaluation_basis`는 정확히 `source_attested_synthetic`이어야 한다.
- `layout_truth_kind`는 정확히 `pdf_content_stream_layout_v1`이어야 한다.
- `evaluation_scope`는 정확히 `synthetic_content_stream_compatibility`이다.
- 따라서 native PDF text-layer 후보의 통과는 **PDF content-stream 호환성**
  증거이며, 이미지 기반 광학 OCR 품질 증거가 아니다.
- PDF, 선택한 synthetic dataset revision, sealed packet에만 적용된다. JPEG와
  PNG는 이 pointer가 있어도 계속 `unavailable`이다.
- 실제 기업 문서, 스캔, 손글씨, 도장 겹침, 저화질 이미지에 대한 품질 주장은
  만들지 않는다.

layout truth compiler는 candidate OCR output, Gold UI, human review draft를
읽으면 안 된다. 이 자료의 dataset/packet/PDF hash·parser specification 등은
layout truth receipt 안에서 검증된다. 개발용 60개와 봉인용 20개는 서로 다른
packet이므로 서로 다른 layout truth receipt를 가진다. 개발 selection은 개발
receipt를, sealed result와 pointer는 봉인 receipt를 각각 참조한다. 두 receipt는
같은 dataset revision·dataset receipt·parser specification을 공유해야 하지만,
hash 자체가 같으면 오히려 잘못된 결속이다.

## 자동 활성화 전제

자동 sealed evaluator result는 canonical JSON이어야 하며 다음을 모두 가져야
한다.

- `evaluation_basis: source_attested_synthetic`
- `evaluation_scope: synthetic_content_stream_compatibility`
- `layout_truth_kind: pdf_content_stream_layout_v1`
- `layout_truth_receipt_sha256`
- `layout_truth_parser_spec_sha256`
- `development_selection_receipt_sha256`
- `sealed_run_receipt_sha256`
- sealed packet/dataset/engine/configuration/model inventory hash
- field exact match `>= 0.99`, CER `<= 0.01`, table-cell F1 `>= 0.98`
- `passed: true` 및 self-excluding result SHA-256

manual human Gold result, `gold_receipt_sha256`, reviewer/approval ID, 빠진
provenance field, 다른 scope/kind는 이 경로에서 모두 거부된다. 반대로 기존
`approve_document_preview_runtime.py`는 기존 human-adjudicated result만 받는다.

또한 아래 local-only 증거 파일이 모두 존재하고 canonical self-hash를 통과해야
한다. 파일은 regular file이어야 하며 symlink, 누락, byte 변조, 다른 hash 이름은
모두 fail-closed 한다.

- `source-layout-truths/<sealed-layout-truth-sha>.json`: 봉인 20개 전체의
  source PDF SHA와 6개 strata(각 최소 3개)를 보존한다.
- `source-layout-truths/<development-layout-truth-sha>.json`: 개발 60개 전체의
  source PDF SHA와 6개 strata(각 10개)를 보존한다. 봉인 packet과 PDF SHA/asset ID가
  겹치면 거부한다.
- `source-attested-development-selections/<selection-sha>.json`: 정확히
  `pdf_text_layer`, `paddleocr`, `docling` 세 후보와 선택된 passing 후보의 engine,
  revision, configuration, inventory, development result hash를 결속한다.
- `source-attested-sealed-runs/<sealed-run-sha>.json`: 선택 결과, 봉인 layout,
  candidate bundle/parser 및 engine/configuration/inventory를 결속한다.
- `candidate-run-reservations/sealed-<sealed-packet-sha>.json`: 봉인 후보 실행을
  한 번만 허용한 create-only 예약의 exact canonical-byte SHA를 결속한다.

기본 static 승인 runtime은 시작 시 SHA-256으로 pointer를 고정하고, 요청마다 drift를
`unavailable`로 끝낸다. 이 문서의 source-attested 경로만 별도의 명시 opt-in으로
동작한다. 그 모드는 PDF 요청마다 fixed local pointer를 다시 읽고, pointer hash가
바뀌면 전체 체인을 재검증한 뒤에만 새 engine으로 교체한다. 누락·변조·human pointer·
새 chain 검증 실패가 있으면 이전 engine을 유지하지 않고 `unavailable`로 끝낸다.

모델과 package는 여전히 별도 명시 provisioning으로 local-only inventory에
기록해야 한다. Docling은 본체만이 아니라 layout/table/GLM 실행에 실제로 관여하는
고정 dependency closure의 package version도 inventory에 기록해야 한다. PaddleOCR의 25개
package는 pinned `uv.lock`의 direct profile dependency boundary와 adapter 직접 의존성일
뿐, 전체 Python import closure가 아니다. `cv2`를 제공할 수 있는 세 OpenCV 배포본의
version도 모두 기록해야 한다. runtime은 실제 `cv2` tree의 regular file 집합·hash·size가
designated `opencv-contrib-python` wheel `RECORD`와 맞지 않으면 provisioning·activation·
engine binding load를 거부한다. PaddleOCR/Docling은 설정된 model directory 아래의 모든 regular
file을 inventory의 path·size·SHA-256으로 선언해야 하며, extra·누락 file 또는
symlink/Windows junction(reparse point)은 activation과 runtime 모두에서 거부된다.
이 활성화 명령은 다운로드, network fallback, model 설치를 수행하지 않는다.

현재 pinned PaddleOCR profile은 `PP-OCRv4`, CPU (`use_gpu: false`,
`cpu_threads: 1`)와 detection, recognition, classifier의 **세** local model directory를
모두 명시해야 한다. classifier를 사용하지 않아도 PaddleOCR 2.9.1은 기본 classifier
downloader를 초기화할 수 있으므로, `cls_model_directory`를 생략할 수 없다. Docling profile은
`ocr_enabled: false`, `device: cpu`, `num_threads: 1`, `table_structure_mode: "fast" | "accurate"`,
cell matching, `pdf_backend: "pypdfium2"` 고정만 허용한다. 두 table mode는 서로 다른
configuration·inventory·extractor revision의 별도 profile이며 자동 대체는 없다. 마지막 값은 adapter가
`PyPdfiumDocumentBackend`를 직접 선택하게 하며 기본 `DoclingParseV2` 또는 다른 PDF
backend fallback을 허용하지 않는다. 이 후보는 inventory된 Docling layout/table artifact를
쓰는 로컬 구조·표 parser이며 EasyOCR downloader를 쓰지 않는다. StandardPdfPipeline의
native GLM 경계 때문에 기존 ASCII-only `site-packages` preflight는 유지한다. 따라서 자동
certificate에서 Docling을 일반 optical OCR로 표현하지 않는다.

PaddleOCR 3 profile은 위 2.9 profile과 독립된
`runtime_api: "paddleocr3-paddlex-3.2"`다. configuration은
`paddleocr-3.2.0-ppocrv5-mobile-en-raster216-v1` revision, PP-OCRv5 mobile det/rec
두 모델 directory, CPU 1 thread 및 `raster_scale: 3`만 허용한다. 이 revision은
PDFium 3배 raster(216 dpi)와 PaddleOCR 3.2.0/PaddleX 3.2.0/PaddlePaddle 3.1.1 API
family를 함께 뜻하므로 다른 revision이나 2.9 fallback으로 바꿀 수 없다.

P3 inventory의 `packages`는 현재 ASCII-only CPython 3.11/Windows AMD64 environment에서
installed metadata를 재귀적으로 해석한 **정확히 56개** activated dependency closure와
각 고정 version으로 구성해야 한다. 시작점은 `paddleocr`(여기서
`paddlex[ocr-core]`를 활성화), `paddlepaddle`, adapter의 `pypdfium2`이고, Windows와
AMD64 marker도 적용한다. 활성화하지 않은 PaddleX extras와 서비스·테스트 패키지는
포함하지 않는다. 누락, 추가 package, 또는 version drift는 모두 provisioning과 runtime
binding에서 fail-closed 한다.

```text
aistudio-sdk, annotated-types, anyio, bce-python-sdk, certifi, chardet,
charset-normalizer, click, colorama, colorlog, crc32c, filelock, fsspec,
future, h11, hf-xet, httpcore, httpx, huggingface-hub, idna, imagesize,
modelscope, modelscope-hub, networkx, numpy, opencv-contrib-python,
opt-einsum, packaging, paddleocr, paddlepaddle, paddlex, pandas, pillow,
prettytable, protobuf, psutil, py-cpuinfo, pyclipper, pycryptodome, pydantic,
pydantic-core, pypdfium2, python-dateutil, pyyaml, requests, ruamel-yaml,
setuptools, shapely, six, tqdm, typing-extensions, typing-inspection, tzdata,
ujson, urllib3, wcwidth
```

## 활성화

source-attested evaluator가 `source-attested-evaluation-results/` 아래에 passing sealed result를
기록한 뒤, evaluator/orchestrator의 명시적
`--auto-activate-source-attested-synthetic` switch는 아래 runtime helper를 같은
local workflow에서 호출한다. 개발 평가, 실패한 sealed 평가, switch가 없는 일반
평가는 pointer를 절대 쓰지 않는다.

다음 명령은 같은 helper를 직접 호출하는 local-only 동등 경로다. 결과·configuration·inventory·설치 package·local model artifact를 다시 검증하고 active pointer를 원자적으로 교체한다.

```powershell
uv run --locked --no-sync --group dev python tools/activate_source_attested_synthetic_pdf_preview.py `
  --document-preview-root <workspace>/_local/data/document-preview `
  --evaluation-result <workspace>/_local/data/document-preview/source-attested-evaluation-results/<result-sha>.json `
  --configuration-sha256 <configuration-sha> `
  --model-inventory-sha256 <inventory-sha>
```

기본 static human-approved 경로는 기존처럼 출력의 `activation_sha256`을 paired
setting에 정확히 넣고 새 Agent process를 시작한다.

```powershell
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_PATH = '<workspace>/_local/data/document-preview/approvals/pdf-preview-active.json'
$env:CUSTOMS_AGENT_DOCUMENT_PREVIEW_APPROVAL_SHA256 = '<activation-sha256>'
```

반대로 합성 source-attested 자동 경로는 Agent를 한 번 다음 importer switch로
시작한다. 이 switch는 engine·model·URL을 받지 않고 workspace의 유일한 local active
pointer path만 설정한다. SHA setting과는 함께 쓸 수 없다.

```powershell
. .\tools\Import-AgentProfile.ps1 -Profile dev-fixture `
  -EnableSourceAttestedPdfPreviewAutoActivation
uv run --locked --no-sync --group dev customs-ai-agent-dev-serve
```

이렇게 시작한 process는 pointer가 아직 없으면 계속 `unavailable`이며, sealed
evaluator가 passing 결과로 pointer를 원자적으로 쓴 뒤 **다음 PDF 요청**에서 새
chain을 검증해 자동 반영한다. PaddleOCR/Docling source-attested pointer는 평가했던
직접 profile을 그대로 실행하며, native text-layer fallback으로 바뀌지 않는다. 기존
human-approved pointer의 text-layer-first 동작은 그대로다. 이후 activation마다 SHA 복사나
Agent 재시작은 필요하지 않다. 단, 이미 일반 mode로 떠 있는 process를 이 mode로 바꾸려면
처음 한 번은 이 switch로 다시 시작해야 한다.

이 자동 모드는 source-attested synthetic PDF 전용의 local trust boundary다. 같은
local write 권한의 악성 프로세스까지 암호학적으로 식별하는 외부 signer는 아니므로,
해당 `_local/data/document-preview/` 경로는 evaluator와 Agent 사용자 권한으로만
보호해야 한다. 두 mode 모두 서비스 시작만으로 candidate 평가·model download·pointer
생성을 수행하지 않으며, pointer/result/configuration/inventory/model artifact 또는 위
증거 체인 중 하나라도 달라지면 fail-closed 한다.
