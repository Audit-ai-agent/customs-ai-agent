import importlib
import io
import json

import pytest
from PIL import Image

from customs_ai_agent.schemas.models import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_LONG_EDGE,
    MAX_IMAGE_VISUAL_TOKENS,
)
from tools.render_trade_document_fixtures import (
    FONT_ASSETS,
    FONT_LICENSE,
    RENDER_DPI,
    RenderError,
    render_dataset,
    render_document,
    require_render_dependencies,
)


def _records() -> dict[str, dict]:
    return {
        "commercial_invoice": {
            "Shipper / Seller": "Synthetic Seller",
            "Consignee (or For account & risk of Messrs)": "Synthetic Consignee",
            "Departure Date": "2025-08-13",
            "Vessel / Flight": "Synthetic Vessel / V.001E",
            "From": "Busan, South Korea",
            "To": "Los Angeles, USA",
            "Invoice No. and Date": "KR-INV-0001 / 2025-08-05",
            "L/C No. and Date": "KR-LC-0001 / 2025-07-09",
            "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
            "Other References (Country of Origin)": ("Country of Origin: Republic of Korea"),
            "Terms of Delivery and Payment": "FOB / T/T 30 Days",
            "Shipping Marks": "SYNTHETIC // C/NO. 0001",
            "No. & Kind of Packages": "10 Cartons",
            "Goods Description": "Synthetic Item",
            "Quantity": "3 PCS",
            "Unit Price": "USD 10.00",
            "Amount": "USD 30.00",
            "Signed By": "Synthetic Signer",
        },
        "packing_list": {
            "Seller": "Synthetic Seller",
            "Consignee": "Synthetic Consignee",
            "Departure Date": "2025-08-13",
            "Vessel / Flight": "Synthetic Vessel / V.001E",
            "From": "Busan, South Korea",
            "To": "Los Angeles, USA",
            "Invoice No. and Date": "KR-INV-0001 / 2025-08-05",
            "Buyer (if other than consignee)": "SAME AS CONSIGNEE",
            "Other References": "Country of Origin: Republic of Korea",
            "Shipping Marks": "SYNTHETIC // C/NO. 0001",
            "No. & Kind of Packages": "10 Cartons",
            "Goods Details": [
                {
                    "item": "Synthetic Item",
                    "quantity": "3 PCS",
                    "weight": "20.00 KGS",
                }
            ],
            "Gross Weight": "30.00 KGS",
            "Measurement": "1.50 CBM",
            "Signed By": "Synthetic Signer",
        },
        "bill_of_lading": {
            "Shipper / Exporter": "Synthetic Seller",
            "Consignee": "Synthetic Consignee",
            "Notify Party": "SAME AS CONSIGNEE",
            "Ocean Vessel": "Synthetic Vessel",
            "Port of Loading": "Busan, South Korea",
            "Place of Receipt": "Busan CY",
            "Voyage No.": "V.001E",
            "Port of Discharge": "Los Angeles, USA",
            "Place of Delivery": "Los Angeles CY",
            "Final Destination": "Los Angeles, USA",
            "B/L No. (Bill of Lading Number)": "BL0001",
            "Flag": "Republic of Korea",
            "Container No.": "CONT0001",
            "Seal No. / Marks & No.": "SEAL0001 // SYNTHETIC",
            "No. & Kinds of Containers or Packages": "1x40HQ / 10 Cartons",
            "Description of Goods": "SAID TO CONTAIN: Synthetic Item",
            "Gross Weight": "30.00 KGS",
            "Freight and Charges": "FREIGHT AS ARRANGED",
            "Revenue Tons": "1.50 RT",
            "Rate": "AS PER TARIFF",
            "Per": "RT",
            "Prepaid": "Y",
            "Collect": "",
            "Freight Prepaid At": "Busan, South Korea",
            "Freight Payable At": "N/A",
            "Place and Date of Issue": "Busan / 2025-08-13",
            "No. of Original B/L": "THREE (3)",
            "Laden on Board (Date / Signature)": "2025-08-13 / SIGNED ON BOARD",
            "Carrier / Agent Signature": "SYNTHETIC CARRIER",
        },
        "certificate_of_origin": {
            "Certificate No.": "CO-SYNTHETIC-0001",
            "Exporter": "Synthetic Seller",
            "Consignee": "Synthetic Consignee",
            "Means of Transport and Route": ("Synthetic Vessel / V.001E | Busan -> Los Angeles"),
            "Country of Origin": "Republic of Korea",
            "Invoice No. and Date": "KR-INV-0001 / 2025-08-05",
            "Goods Details": [
                {
                    "item": "Synthetic Item",
                    "quantity": "3 PCS",
                    "hs_code": "0000000001",
                    "origin_criterion": "SYNTHETIC-CRITERION",
                }
            ],
            "Gross Weight": "20.00 KGS",
            "Issuing Authority": "SYNTHETIC TEST AUTHORITY - NOT VALID",
            "Issue Date": "2025-08-13",
            "Signature": "SYNTHETIC FIXTURE",
            "Document Disclaimer": ("Generated records are test fixtures, not legal certificates."),
        },
    }


def _assert_bbox_within_page(bbox, width, height):
    assert bbox is not None
    assert 0 <= bbox[0] <= bbox[2] <= width
    assert 0 <= bbox[1] <= bbox[3] <= height


def test_optional_renderer_dependency_failure_is_explicit():
    def importer(name):
        if name == "reportlab":
            raise ImportError("not installed")
        return importlib.import_module(name)

    with pytest.raises(RenderError, match="reportlab"):
        require_render_dependencies(importer)


def test_rendered_pdf_and_png_are_byte_deterministic():
    record = _records()["commercial_invoice"]
    kwargs = {
        "document_id": "trade-triplet-0001:commercial_invoice",
        "case_id": "trade-triplet-0001",
        "document_type": "commercial_invoice",
        "record": record,
        "source_record_sha256": "0" * 64,
    }

    first_pdf, first_png, first_claude_png, first_layout = render_document(**kwargs)
    second_pdf, second_png, second_claude_png, second_layout = render_document(**kwargs)

    assert first_pdf == second_pdf
    assert first_png == second_png
    assert first_claude_png == second_claude_png
    assert first_layout == second_layout
    assert first_pdf.startswith(b"%PDF-")
    assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(first_png)) as image:
        assert image.mode == "RGB"
        assert image.size == (
            first_layout["page"]["width_image_px"],
            first_layout["page"]["height_image_px"],
        )
    with Image.open(io.BytesIO(first_claude_png)) as image:
        assert image.size == (
            first_layout["page"]["claude_input_width_px"],
            first_layout["page"]["claude_input_height_px"],
        )
    assert first_layout["page"]["render_dpi"] == 200
    assert {item["font_name"]: item["sha256"] for item in first_layout["font_assets"]} == {
        name: value["sha256"] for name, value in FONT_ASSETS.items()
    }
    assert first_layout["font_license"]["sha256"] == FONT_LICENSE["sha256"]
    resize = first_layout["coordinate_transform_chain"][1]["resize"]
    assert resize["resized"] is True
    assert "visual_token_limit" in resize["reasons"]
    assert max(resize["target_width_px"], resize["target_height_px"]) <= MAX_IMAGE_LONG_EDGE
    assert resize["target_visual_tokens"] <= MAX_IMAGE_VISUAL_TOKENS
    assert resize["target_byte_length"] <= MAX_IMAGE_BYTES


@pytest.mark.parametrize(
    "document_type",
    [
        "commercial_invoice",
        "packing_list",
        "bill_of_lading",
        "certificate_of_origin",
    ],
)
def test_word_field_and_table_bboxes_are_inside_the_raster(document_type):
    record = _records()[document_type]
    _, _, _, layout = render_document(
        document_id=f"fixture:{document_type}",
        case_id="fixture-0001",
        document_type=document_type,
        record=record,
        source_record_sha256="1" * 64,
    )
    width = layout["page"]["width_image_px"]
    height = layout["page"]["height_image_px"]
    claude_width = layout["page"]["claude_input_width_px"]
    claude_height = layout["page"]["claude_input_height_px"]
    assert layout["page"]["render_dpi"] == RENDER_DPI
    assert layout["words"]
    assert layout["fields"]
    assert layout["tables"]
    for word in layout["words"]:
        _assert_bbox_within_page(word["bbox_image_px"], width, height)
        _assert_bbox_within_page(word["bbox_claude_input_px"], claude_width, claude_height)
    for field in layout["fields"]:
        if field["rendered"]:
            _assert_bbox_within_page(field["bbox_image_px"], width, height)
            _assert_bbox_within_page(field["bbox_claude_input_px"], claude_width, claude_height)
            assert field["word_ids"]
        else:
            assert field["bbox_image_px"] is None
            assert field["bbox_claude_input_px"] is None
            assert field["raw_value"] == ""
    for table in layout["tables"]:
        _assert_bbox_within_page(table["bbox_image_px"], width, height)
        _assert_bbox_within_page(table["bbox_claude_input_px"], claude_width, claude_height)
        for row in table["rows"]:
            _assert_bbox_within_page(row["bbox_image_px"], width, height)
            _assert_bbox_within_page(row["bbox_claude_input_px"], claude_width, claude_height)
            for cell in row["cells"]:
                _assert_bbox_within_page(cell["bbox_image_px"], width, height)
                _assert_bbox_within_page(cell["bbox_claude_input_px"], claude_width, claude_height)


def test_render_dataset_writes_only_artifact_outputs(tmp_path):
    dataset_root = tmp_path / "registered"
    evaluation_root = tmp_path / "evaluation"
    output_root = tmp_path / "rendered"
    case_id = "trade-triplet-0001"
    documents = {}
    for document_type in (
        "commercial_invoice",
        "packing_list",
        "bill_of_lading",
    ):
        documents[document_type] = {"raw_record": _records()[document_type]}
    case = {"case_id": case_id, "documents": documents}
    certificate = {
        "fixture_id": "co-synthetic-0001",
        "record": _records()["certificate_of_origin"],
    }
    (dataset_root / "derived").mkdir(parents=True)
    (evaluation_root / "certificate-of-origin").mkdir(parents=True)
    (dataset_root / "derived" / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    (evaluation_root / "certificate-of-origin" / "cases.jsonl").write_text(
        json.dumps(certificate) + "\n", encoding="utf-8"
    )

    manifest = render_dataset(dataset_root, evaluation_root, output_root)

    assert manifest["counts"]["document_count"] == 4
    assert manifest["counts"]["generated_document_file_count"] == 16
    assert manifest["counts"]["claude_input_resized_document_count"] == 4
    assert {item["font_name"]: item["sha256"] for item in manifest["font_assets"]} == {
        name: value["sha256"] for name, value in FONT_ASSETS.items()
    }
    assert manifest["font_license"]["sha256"] == FONT_LICENSE["sha256"]
    assert len(list(output_root.rglob("*.pdf"))) == 4
    assert len(list(output_root.rglob("*.png"))) == 8
    assert len(list(output_root.rglob("*.claude.png"))) == 4
    assert len(list(output_root.rglob("*.layout-gold.json"))) == 4
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "checksums.sha256").is_file()
    assert not (dataset_root / "raw").exists()
