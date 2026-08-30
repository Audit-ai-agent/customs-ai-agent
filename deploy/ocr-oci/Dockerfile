# syntax=docker/dockerfile:1.10

ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.28@sha256:5c3ab83183a73c5d319a77009eb425b60d5bb937f339fb7876788ebf567baf48
# `scratch` lets the asset-free app-structure target build in ordinary PRs.
# The release target necessarily fails at COPY unless the publisher supplies
# an immutable linux/amd64 runtime image reference ending in @sha256:<digest>.
ARG RASTER_RUNTIME_IMAGE=scratch

FROM ${UV_IMAGE} AS uv-bin

FROM ${PYTHON_IMAGE} AS app-build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/customs/app/.venv
WORKDIR /build
COPY --from=uv-bin /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY deploy/ocr-oci/create_app_license_manifest.py ./create_app_license_manifest.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable && \
    /opt/customs/app/.venv/bin/python create_app_license_manifest.py \
      --output /build/app-python-licenses.json

FROM ${PYTHON_IMAGE} AS app-base
ARG TARGETARCH
ARG TARGETOS
RUN test "${TARGETOS}" = "linux" && test "${TARGETARCH}" = "amd64" && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
      libgl1=1.6.0-1 \
      libglib2.0-0=2.74.6-2+deb12u9 \
      libgomp1=12.2.0-14+deb12u1 && \
    rm -rf /var/lib/apt/lists/* && \
    python -c \
      "import ctypes; [ctypes.CDLL(name) for name in ('libgomp.so.1', 'libGL.so.1', 'libgthread-2.0.so.0')]" && \
    groupadd --gid 10001 customs && \
    useradd --uid 10001 --gid 10001 --no-create-home \
      --home-dir /opt/customs --shell /usr/sbin/nologin customs && \
    install -d -o root -g root -m 0755 \
      /opt/customs /opt/customs/app /opt/customs/deploy /opt/customs/licenses
ENV HOME=/tmp/customs-agent \
    XDG_CACHE_HOME=/tmp/customs-agent/cache \
    PADDLE_HOME=/tmp/customs-agent/paddle \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUSTOMS_AGENT_HOST=127.0.0.1 \
    CUSTOMS_AGENT_PORT=18090 \
    CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_AUTO_ACTIVATION=source-attested-synthetic-raster-text-v1 \
    CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH=/opt/customs/_local/data/document-preview/approvals/raster-text-active.json

FROM app-base AS app-structure
COPY --from=app-build --chown=0:0 /opt/customs/app/.venv/ /opt/customs/app/.venv/
COPY --from=app-build --chown=0:0 /build/app-python-licenses.json \
    /opt/customs/licenses/app-python-licenses.json
COPY --chown=0:0 deploy/ocr-oci/ocr_readiness.py /opt/customs/deploy/ocr_readiness.py
RUN test -x /opt/customs/app/.venv/bin/customs-ai-agent-serve && \
    test "$(head -n 1 /opt/customs/app/.venv/bin/customs-ai-agent-serve)" = \
      '#!/opt/customs/app/.venv/bin/python' && \
    test ! -e /opt/customs/_local

FROM ${RASTER_RUNTIME_IMAGE} AS raster-runtime

# This layer changes only when the externally attested OCR runtime digest
# changes.  Application source is copied later, so ordinary code releases can
# reuse the large sidecar/model layer.
FROM app-base AS runtime-material
COPY --from=raster-runtime --link --chown=0:0 \
    /document-preview/ /opt/customs/_local/data/document-preview/

FROM runtime-material AS runtime-verified
COPY --from=app-build --chown=0:0 /opt/customs/app/.venv/ /opt/customs/app/.venv/
COPY --chown=0:0 tools/verify_raster_text_oci_runtime.py \
    /opt/customs/deploy/verify_raster_text_oci_runtime.py
COPY --chown=0:0 tools/raster_text_oci_evidence.py \
    /opt/customs/deploy/raster_text_oci_evidence.py
COPY --chown=0:0 tools/raster_text_oci_metadata.py \
    /opt/customs/deploy/raster_text_oci_metadata.py
COPY --chown=0:0 deploy/ocr-oci/runtime-license-policy.json \
    /opt/customs/deploy/expected-runtime-license-policy.json
COPY --chown=0:0 deploy/ocr-oci/verify_agent_runtime_binding.py \
    /opt/customs/deploy/verify_agent_runtime_binding.py
RUN install -d -o 10001 -g 10001 -m 0755 /opt/customs/deploy/verified
USER 10001:10001
RUN --network=none \
    /opt/customs/app/.venv/bin/python \
      /opt/customs/deploy/verify_raster_text_oci_runtime.py \
      --document-preview-root /opt/customs/_local/data/document-preview \
      --expected-license-policy /opt/customs/deploy/expected-runtime-license-policy.json \
      > /opt/customs/deploy/verified/runtime-verification.json && \
    /opt/customs/app/.venv/bin/python \
      /opt/customs/deploy/verify_agent_runtime_binding.py \
      --document-preview-root /opt/customs/_local/data/document-preview \
      > /opt/customs/deploy/verified/agent-binding-verification.json

FROM runtime-material AS release
ARG RASTER_RUNTIME_IMAGE
ARG SOURCE_REVISION
LABEL org.opencontainers.image.title="Customs AI Agent with Raster Text OCR" \
      org.opencontainers.image.source="https://github.com/Audit-ai-agent/customs-ai-agent" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.customs.raster-runtime.image="${RASTER_RUNTIME_IMAGE}" \
      io.customs.raster-runtime.contract="raster-text-runtime-oci-v1"
COPY --from=app-structure --chown=0:0 /opt/customs/app/ /opt/customs/app/
COPY --from=app-structure --chown=0:0 /opt/customs/licenses/ /opt/customs/licenses/
COPY --from=app-structure --chown=0:0 /opt/customs/deploy/ocr_readiness.py \
    /opt/customs/deploy/ocr_readiness.py
COPY --from=runtime-verified --chown=0:0 \
    /opt/customs/deploy/verified/runtime-verification.json \
    /opt/customs/deploy/runtime-verification.json
COPY --from=runtime-verified --chown=0:0 \
    /opt/customs/deploy/verified/agent-binding-verification.json \
    /opt/customs/deploy/agent-binding-verification.json
COPY --from=runtime-verified --chown=0:0 \
    /opt/customs/deploy/expected-runtime-license-policy.json \
    /opt/customs/deploy/runtime-license-policy.json
COPY --chown=0:0 tools/verify_raster_text_oci_runtime.py \
    /opt/customs/deploy/verify_raster_text_oci_runtime.py
COPY --chown=0:0 tools/raster_text_oci_metadata.py \
    /opt/customs/deploy/raster_text_oci_metadata.py
RUN SOURCE_REVISION_VALUE="${SOURCE_REVISION}" \
    RASTER_RUNTIME_IMAGE_VALUE="${RASTER_RUNTIME_IMAGE}" \
    python -c \
      "import os, re; assert re.fullmatch(r'[a-f0-9]{40}', os.environ['SOURCE_REVISION_VALUE']); assert re.fullmatch(r'ghcr[.]io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}', os.environ['RASTER_RUNTIME_IMAGE_VALUE'])" && \
    test -x /opt/customs/app/.venv/bin/customs-ai-agent-serve && \
    test "$(head -n 1 /opt/customs/app/.venv/bin/customs-ai-agent-serve)" = \
      '#!/opt/customs/app/.venv/bin/python' && \
    test ! -e /opt/customs/_local/data/document-preview/synthetic-raster-v1

USER 10001:10001
WORKDIR /opt/customs
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=3 \
  CMD ["/opt/customs/app/.venv/bin/python", "/opt/customs/deploy/ocr_readiness.py"]
ENTRYPOINT ["/opt/customs/app/.venv/bin/customs-ai-agent-serve"]
