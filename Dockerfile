# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG AGENT_WHEEL=dist/wheel/customs_ai_agent-0.3.0-py3-none-any.whl
ARG BUILD_SHA
ARG WHEEL_SHA256

LABEL org.opencontainers.image.title="customs-ai-agent" \
      org.opencontainers.image.description="Text-only evidence-bound Customs AI Agent service" \
      org.opencontainers.image.revision="${BUILD_SHA}"

ENV CUSTOMS_AGENT_BUILD_SHA="${BUILD_SHA}" \
    CUSTOMS_AGENT_ENVIRONMENT="production" \
    CUSTOMS_AGENT_HOST="0.0.0.0" \
    CUSTOMS_AGENT_PORT="8080" \
    CUSTOMS_AGENT_WHEEL_SHA256="${WHEEL_SHA256}" \
    HOME="/tmp" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1" \
    TMPDIR="/tmp"

COPY ${AGENT_WHEEL} /tmp/customs_ai_agent.whl
COPY dist/runtime-requirements.txt /tmp/runtime-requirements.txt

RUN set -eu; \
    test "$(printf '%s' "${BUILD_SHA}" | wc -c)" -eq 40; \
    test "$(printf '%s' "${WHEEL_SHA256}" | wc -c)" -eq 64; \
    echo "${WHEEL_SHA256}  /tmp/customs_ai_agent.whl" | sha256sum --check --strict; \
    python -c "import json,re,sys,zipfile; p='/tmp/customs_ai_agent.whl'; raw=zipfile.ZipFile(p).read('customs_ai_agent/build-manifest.json'); m=json.loads(raw); expected={'contract','distribution','git_sha','package_version','schema_version','source_snapshot_algorithm','source_snapshot_sha256'}; canonical=(json.dumps(m,ensure_ascii=True,allow_nan=False,separators=(',',':'),sort_keys=True)+'\n').encode(); valid=set(m)==expected and raw==canonical and m['contract']=='customs-ai-agent-build-manifest' and m['schema_version']=='1.0' and m['distribution']=='customs-ai-agent' and m['package_version']=='0.3.0' and m['source_snapshot_algorithm']=='git-archive-files-sha256-v1' and m['git_sha']=='${BUILD_SHA}' and re.fullmatch('[a-f0-9]{64}',m['source_snapshot_sha256']); sys.exit(0 if valid else 1)"; \
    python -m pip install --no-cache-dir --disable-pip-version-check \
      --require-hashes --requirement /tmp/runtime-requirements.txt; \
    python -m pip install --no-cache-dir --disable-pip-version-check \
      --no-deps /tmp/customs_ai_agent.whl; \
    python -m pip check; \
    rm /tmp/customs_ai_agent.whl /tmp/runtime-requirements.txt; \
    groupadd --gid 65532 agent; \
    useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin agent

USER 65532:65532
WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2).read()"]

ENTRYPOINT ["customs-ai-agent-serve"]
