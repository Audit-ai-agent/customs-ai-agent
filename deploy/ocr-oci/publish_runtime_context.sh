#!/usr/bin/env bash
set -euo pipefail

# Publish one already-exported runtime context.  Authentication is deliberately
# external: run `docker login ghcr.io` first or use a credential helper.  This
# script never accepts or prints a registry token.

usage() {
  echo "usage: $0 <absolute-export-directory> <ghcr.io/owner/runtime-package>" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
context_root=$1
image_repository=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "${script_dir}/../.." && pwd -P)

[[ "${context_root}" = /* ]] || { echo "absolute_runtime_context_required" >&2; exit 1; }
[[ -d "${context_root}/document-preview" ]] || { echo "runtime_context_missing" >&2; exit 1; }
[[ "${image_repository}" =~ ^ghcr\.io/[a-z0-9][a-z0-9._/-]*$ ]] || {
  echo "ghcr_runtime_repository_invalid" >&2
  exit 1
}
command -v docker >/dev/null 2>&1 || { echo "docker_unavailable" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "docker_buildx_unavailable" >&2; exit 1; }

verification_json=$(python3 "${repository_root}/tools/verify_raster_text_oci_runtime.py" \
  --document-preview-root "${context_root}/document-preview")
manifest_sha=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["manifest_sha256"])' \
  <<<"${verification_json}")
profile_revision=$(python3 -c \
  'import json,sys; value=json.load(sys.stdin); print(value["raster_text_profile_revision"])' \
  <<<"${verification_json}")
[[ "${manifest_sha}" =~ ^[a-f0-9]{64}$ ]] || { echo "runtime_manifest_digest_invalid" >&2; exit 1; }
[[ "${profile_revision}" =~ ^synthetic-raster-text-v1:[a-f0-9]{24}$ ]] || {
  echo "runtime_profile_revision_invalid" >&2
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

# This is the only value the final-image workflow accepts.  The mutable helper
# tag is never consumed by a release build.
printf 'runtime_image=%s\n' "${digest_reference}"
printf 'runtime_manifest_sha256=%s\n' "${manifest_sha}"
printf 'raster_text_profile_revision=%s\n' "${profile_revision}"
