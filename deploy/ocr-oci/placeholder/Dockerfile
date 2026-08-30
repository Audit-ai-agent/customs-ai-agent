# syntax=docker/dockerfile:1.10

# This image contains no filesystem layer or OCR material. Its only purpose is
# to let the repository workflow create/link a GHCR package before any private
# runtime byte is considered for publication.
FROM scratch
ARG SOURCE_REPOSITORY
LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      io.customs.artifact.kind="asset-free-ghcr-bootstrap-v1"
