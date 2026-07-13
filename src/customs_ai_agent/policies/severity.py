from customs_ai_agent.schemas.models import Severity


SEVERITY_POLICY: dict[str, Severity] = {
    "INV_TOTAL_MISMATCH": "high",
    "QTY_MISMATCH_INVOICE_PACKING": "high",
    "GROSS_WEIGHT_MISMATCH": "high",
    "NET_GROSS_WEIGHT_INVALID": "critical",
    "CONTAINER_NO_MISMATCH": "high",
    "SHIPPER_CONSIGNEE_MISMATCH": "medium",
    "HS_CODE_MISSING": "medium",
    "HS_CODE_FORMAT_INVALID": "medium",
    "ORIGIN_MISMATCH": "critical",
    "LINE_ITEM_UNMATCHED": "medium",
}
