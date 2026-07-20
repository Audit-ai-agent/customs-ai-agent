import hashlib
import json
from types import SimpleNamespace

import pytest

import tools.repair_trade_triplet_container_mutation as repair


def test_repair_cli_redacts_filesystem_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = "C:" + r"\Users" + r"\private-reviewer\missing-bundle"
    monkeypatch.setattr(
        repair,
        "_parse_args",
        lambda: SimpleNamespace(
            source_root="unused",
            output_root="unused",
        ),
    )
    monkeypatch.setattr(
        repair,
        "repair_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(private_path)),
    )

    assert repair.main() == 1
    output = capsys.readouterr().out
    assert "filesystem_operation_failed" in output
    assert private_path not in output


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in values
        )
        + "\n",
        encoding="utf-8",
    )


def _source_bundle(tmp_path, monkeypatch):
    root = tmp_path / "source-v2"
    defects_path = root / repair.DEFECTS_RELATIVE_PATH
    before = "HMMU7610271"
    after = f"{before}-CONTROLLED-MISMATCH"
    source_case = {
        "documents": {
            "bill_of_lading": {
                "raw_record": {
                    "Container No.": before,
                }
            },
            "packing_list": {"raw_record": {}},
        }
    }
    controlled_parent = json.loads(json.dumps(source_case))
    controlled_parent["documents"]["packing_list"]["raw_record"]["Container No."] = before
    mutated_case = json.loads(json.dumps(controlled_parent))
    mutated_case["documents"]["packing_list"]["raw_record"]["Container No."] = after
    records = [
        {
            "variant_id": "trade-triplet-0040--pl_container_mismatch",
            "source_case_sha256": repair._canonical_sha256(source_case),
            "controlled_parent_sha256": repair._canonical_sha256(controlled_parent),
            "controlled_parent_setup": [
                {
                    "op": "add",
                    "path": [
                        "documents",
                        "packing_list",
                        "raw_record",
                        "Container No.",
                    ],
                    "value": before,
                }
            ],
            "mutation": {
                "operator_id": repair.OPERATOR_ID,
                "document_type": "packing_list",
                "field_path": ["Container No."],
                "before": before,
                "after": after,
            },
            "expected_rule_ids": ["pl_bl.container_numbers"],
            "expected_outcome": "needs_review",
            "adjudication": {
                "rule_id": "pl_bl.container_numbers",
                "expected_findings": ["pl_bl.container_numbers"],
                "expected_outcome": "needs_review",
                "evidence": {
                    "operator_id": repair.OPERATOR_ID,
                    "document_type": "packing_list",
                    "field_path": ["Container No."],
                    "before": before,
                    "after": after,
                },
            },
            "mutated_case": mutated_case,
        }
    ]
    _write_jsonl(defects_path, records)
    defects_sha256 = _sha256(defects_path)
    manifest_path = root / repair.MANIFEST_RELATIVE_PATH
    _write_json(
        manifest_path,
        {
            "schema_version": "trade-document-evaluation-assets/v3",
            "evaluation_version": "evaluation-v2",
            "controlled_defects": {
                "relative_path": repair.DEFECTS_RELATIVE_PATH.as_posix(),
                "sha256": defects_sha256,
                "byte_size": defects_path.stat().st_size,
            },
            "generated_files": [
                {
                    "relative_path": repair.DEFECTS_RELATIVE_PATH.as_posix(),
                    "sha256": defects_sha256,
                    "byte_size": defects_path.stat().st_size,
                }
            ],
            "release_holdout": {
                "release_eligible": False,
                "status": "historical",
            },
        },
    )
    checksums_path = root / repair.CHECKSUMS_RELATIVE_PATH
    checksums_path.write_text(
        f"{defects_sha256}  {repair.DEFECTS_RELATIVE_PATH.as_posix()}\n"
        f"{_sha256(manifest_path)}  {repair.MANIFEST_RELATIVE_PATH.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(repair, "KNOWN_SOURCE_DEFECTS_SHA256", defects_sha256)
    monkeypatch.setattr(repair, "KNOWN_SOURCE_MANIFEST_SHA256", _sha256(manifest_path))
    monkeypatch.setattr(repair, "KNOWN_SOURCE_CHECKSUMS_SHA256", _sha256(checksums_path))
    return root, records


def test_repair_preserves_source_and_isolates_one_valid_container_mismatch(tmp_path, monkeypatch):
    source, original_records = _source_bundle(tmp_path, monkeypatch)
    output = tmp_path / "repaired-v2"

    report = repair.repair_bundle(source, output)

    assert report["status"] == "PASS"
    assert report["release_eligible"] is False
    assert (
        json.loads((source / repair.DEFECTS_RELATIVE_PATH).read_text().splitlines()[0])
        == (original_records[0])
    )
    repaired = json.loads((output / repair.DEFECTS_RELATIVE_PATH).read_text().splitlines()[0])
    before = repaired["mutation"]["before"]
    after = repaired["mutation"]["after"]
    assert after != before
    assert (
        after
        == repaired["mutated_case"]["documents"]["packing_list"]["raw_record"]["Container No."]
    )
    assert repair._different_valid_container_number(before) == after
    assert repaired["adjudication"]["evidence"] == repaired["mutation"]
    assert (output / repair.PROVENANCE_RELATIVE_PATH).is_file()
    assert "repair-provenance.json" in (output / repair.CHECKSUMS_RELATIVE_PATH).read_text()


def test_repair_rejects_existing_output_without_overwriting(tmp_path, monkeypatch):
    source, _records = _source_bundle(tmp_path, monkeypatch)
    output = tmp_path / "repaired-v2"
    output.mkdir()

    with pytest.raises(repair.RepairError, match="already exists"):
        repair.repair_bundle(source, output)


def test_repair_rejects_undeclared_source_file(tmp_path, monkeypatch):
    source, _records = _source_bundle(tmp_path, monkeypatch)
    (source / "undeclared.txt").write_text("must not be copied\n", encoding="utf-8")

    with pytest.raises(repair.RepairError, match="undeclared or missing"):
        repair.repair_bundle(source, tmp_path / "repaired-v2")
