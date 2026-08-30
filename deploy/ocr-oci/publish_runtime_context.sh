#!/usr/bin/env bash
set -euo pipefail

# Publish one already-exported runtime context.  Authentication is deliberately
# external: run `docker login ghcr.io` first or use a credential helper.  This
# script never accepts or prints a registry token.

usage() {
  echo "usage: $0 <absolute-export-directory> <ghcr.io/owner/runtime-package> <owner/repository>" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
context_root=$1
image_repository=$2
source_repository=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "${script_dir}/../.." && pwd -P)

[[ "${context_root}" = /* ]] || { echo "absolute_runtime_context_required" >&2; exit 1; }
[[ -d "${context_root}/document-preview" ]] || { echo "runtime_context_missing" >&2; exit 1; }
[[ "${image_repository}" =~ ^ghcr\.io/[a-z0-9][a-z0-9._/-]*$ ]] || {
  echo "ghcr_runtime_repository_invalid" >&2
  exit 1
}
[[ "${source_repository}" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$ ]] || {
  echo "github_source_repository_invalid" >&2
  exit 1
}
command -v docker >/dev/null 2>&1 || { echo "docker_unavailable" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "docker_buildx_unavailable" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv_unavailable" >&2; exit 1; }

# The package must already exist and be private.  First-time setup uses an
# asset-free placeholder; this script never risks model bytes on an unknown
# package visibility default.
python3 "${repository_root}/deploy/ocr-oci/verify_ghcr_private_package.py" \
  --image "${image_repository}" \
  --repository "${source_repository}" >/dev/null

verification_json=$(python3 "${repository_root}/tools/verify_raster_text_oci_runtime.py" \
  --document-preview-root "${context_root}/document-preview")
# The data-only verifier owns transport/packaging constraints.  Publication
# additionally requires the authoritative V12/V19/V9 Agent loader to start the
# bundled child, complete native imports/model load, and report warm liveness.
uv run --directory "${repository_root}" --locked --no-sync python \
  deploy/ocr-oci/verify_agent_runtime_binding.py \
  --document-preview-root "${context_root}/document-preview" >/dev/null
manifest_sha=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["manifest_sha256"])' \
  <<<"${verification_json}")
profile_revision=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["raster_text_profile_revision"])' \
  <<<"${verification_json}")
file_count=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["file_count"])' \
  <<<"${verification_json}")
total_bytes=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["total_bytes"])' \
  <<<"${verification_json}")
[[ "${manifest_sha}" =~ ^[a-f0-9]{64}$ ]] || { echo "runtime_manifest_digest_invalid" >&2; exit 1; }
[[ "${profile_revision}" =~ ^synthetic-raster-text-v1:[a-f0-9]{24}$ ]] || {
  echo "runtime_profile_revision_invalid" >&2
  exit 1
}
[[ "${file_count}" =~ ^[0-9]+$ && "${file_count}" -le 25000 ]] || {
  echo "runtime_file_count_invalid" >&2
  exit 1
}
[[ "${total_bytes}" =~ ^[0-9]+$ && "${total_bytes}" -le 2147483648 ]] || {
  echo "runtime_total_bytes_invalid" >&2
  exit 1
}

temporary_root=$(mktemp -d)
cleanup() {
  rm -f -- "${temporary_root}/metadata.json"
  rmdir -- "${temporary_root}" 2>/dev/null || true
}
trap cleanup EXIT
runtime_tag="runtime-${manifest_sha:0:24}"

docker buildx build \
  --file "${repository_root}/deploy/ocr-oci/runtime-context.Dockerfile" \
  --platform linux/amd64 \
  --provenance=mode=max \
  --sbom=true \
  --label "io.customs.runtime.manifest-sha256=${manifest_sha}" \
  --label "io.customs.raster-text.profile-revision=${profile_revision}" \
  --metadata-file "${temporary_root}/metadata.json" \
  --tag "${image_repository}:${runtime_tag}" \
  --push \
  "${context_root}"

runtime_digest=$(python3 -c \
  'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value["containerimage.digest"])' \
  "${temporary_root}/metadata.json")
[[ "${runtime_digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || {
  echo "runtime_image_digest_missing" >&2
  exit 1
}
digest_reference="${image_repository}@${runtime_digest}"
docker buildx imagetools inspect "${digest_reference}" >/dev/null
python3 "${repository_root}/deploy/ocr-oci/verify_ghcr_private_package.py" \
  --image "${digest_reference}" \
  --repository "${source_repository}" >/dev/null

# This is the only value the final-image workflow accepts.  The mutable helper
# tag is never consumed by a release build.
printf 'runtime_image=%s\n' "${digest_reference}"
printf 'runtime_manifest_sha256=%s\n' "${manifest_sha}"
printf 'raster_text_profile_revision=%s\n' "${profile_revision}"
printf 'runtime_file_count=%s\n' "${file_count}"
printf 'runtime_total_bytes=%s\n' "${total_bytes}"
