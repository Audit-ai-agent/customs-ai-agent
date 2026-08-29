# syntax=docker/dockerfile:1.10

# This image is a data-only, content-addressed build input.  The context must
# be the directory produced by tools/export_raster_text_oci_runtime.py and
# therefore contains exactly document-preview/ plus its immutable manifest.
FROM scratch
COPY --chown=0:0 document-preview/ /document-preview/

LABEL org.opencontainers.image.title="Customs Raster Text OCR runtime context" \
      org.opencontainers.image.description="Verified Linux/amd64 OCR sidecar, models and compact activation closure" \
      io.customs.artifact.kind="raster-text-runtime-context-v1"

USER 65532:65532
