"""Validate generated Stage 3 tables, figures, report values, and mirrors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results_clean_split" / "final_artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional JSON validation report path.",
    )
    args = parser.parse_args()

    required_figures = [
        "task1/class_distribution.png",
        "task1/representative_samples.png",
        "task3/optimizer_comparison_curves.png",
        "task4/model_a_training_curves.png",
        "task4/model_b_training_curves.png",
        "task4/model_a_confusion_matrix.png",
        "task4/model_b_confusion_matrix.png",
        "task5/mobilenet_v2_training_curves.png",
        "task5/shufflenet_v2_training_curves.png",
        "task5/mobilenet_v2_confusion_matrix.png",
        "task5/shufflenet_v2_confusion_matrix.png",
        "task6/test_accuracy_comparison.png",
        "task6/parameter_count_comparison.png",
        "task6/model_size_comparison.png",
        "task6/mac_comparison.png",
        "task6/accuracy_vs_macs.png",
        "task6/accuracy_vs_model_size.png",
    ]
    required_tables = [
        "optimizer_comparison.csv",
        "optimizer_comparison.json",
        "optimizer_comparison.tex",
        "model_a_vs_b.csv",
        "model_a_vs_b.json",
        "model_a_vs_b.tex",
        "sota_comparison.csv",
        "sota_comparison.json",
        "sota_comparison.tex",
        "final_model_comparison.csv",
        "final_model_comparison.json",
        "final_model_comparison.tex",
        "model_b_vs_sota_comparison.csv",
        "model_b_vs_sota_comparison.json",
        "model_b_vs_sota_comparison.tex",
        "model_a_architecture.csv",
        "model_b_architecture.csv",
        "derived_tradeoffs.json",
    ]
    for relative in required_figures:
        source = CANONICAL / "figures" / relative
        mirror = ROOT / "figures" / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise AssertionError(f"Missing canonical figure: {relative}")
        if not mirror.is_file() or sha256(source) != sha256(mirror):
            raise AssertionError(f"Figure mirror mismatch: {relative}")
    for relative in required_tables:
        source = CANONICAL / "tables" / relative
        mirror = ROOT / "tables" / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise AssertionError(f"Missing canonical table: {relative}")
        if not mirror.is_file() or sha256(source) != sha256(mirror):
            raise AssertionError(f"Table mirror mismatch: {relative}")

    final_rows = read_rows(CANONICAL / "tables" / "final_model_comparison.csv")
    final_json = json.loads(
        (CANONICAL / "tables" / "final_model_comparison.json").read_text(encoding="utf-8")
    )
    if len(final_rows) != len(final_json):
        raise AssertionError("Final comparison CSV/JSON row-count mismatch")
    for csv_row, json_row in zip(final_rows, final_json):
        for key, json_value in json_row.items():
            csv_value = csv_row[key]
            if isinstance(json_value, str):
                if csv_value != json_value:
                    raise AssertionError(f"Final comparison CSV/JSON mismatch: {key}")
            elif abs(float(csv_value) - float(json_value)) > 0.00005:
                raise AssertionError(f"Final comparison CSV/JSON mismatch: {key}")
    if [row["Model"] for row in final_rows] != [
        "Model A", "Model B", "ShuffleNetV2 1.0x", "MobileNetV2"
    ]:
        raise AssertionError("Final comparison model ordering mismatch")

    optimizer_rows = read_rows(CANONICAL / "tables" / "optimizer_comparison.csv")
    if len(optimizer_rows) != 3 or optimizer_rows[-1]["Optimizer"] != "Adam (selected)":
        raise AssertionError("Optimizer table is incomplete or selected optimizer is wrong")

    for name, expected in [("model_a_architecture.csv", 2_553_349),
                           ("model_b_architecture.csv", 37_957)]:
        rows = read_rows(CANONICAL / "tables" / name)
        if sum(int(row["Trainable parameters"]) for row in rows) != expected:
            raise AssertionError(f"Architecture parameter total mismatch: {name}")

    report_tokens = {
        "Draco_A03_EN3150.pdf": [
            "67.64", "70.57", "86.09", "83.46",
            "cross-split duplicate content equals zero",
        ],
        "Section_6_Final_Comparison.pdf": ["70.57", "86.09", "83.46"],
    }
    forbidden = ["68.90", "70.43", "86.31", "83.76", "70.14"]
    page_counts: dict[str, int] = {}
    for filename, tokens in report_tokens.items():
        pdf = ROOT / "reports" / filename
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise AssertionError(f"Missing report: {filename}")
        text = subprocess.run(
            ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        for token in tokens:
            if token not in text:
                raise AssertionError(f"{filename}: missing canonical token {token}")
        for token in forbidden:
            if token in text:
                raise AssertionError(f"{filename}: stale metric token {token}")
        info = subprocess.run(
            ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
        ).stdout
        page_line = next(line for line in info.splitlines() if line.startswith("Pages:"))
        page_counts[filename] = int(page_line.split(":", 1)[1])

    canonical_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", ROOT / "Model_A_Design.md",
                     ROOT / "Model_B_Design.md", ROOT / "SOTA_Evaluation.md",
                     ROOT / "docs" / "STAGE3_4_SUMMARY.md",
                     ROOT / "reports" / "Draco_A03_EN3150.tex",
                     ROOT / "reports" / "Section_6_Final_Comparison.tex"]
    )
    if "TODO" in canonical_text or "PLACEHOLDER" in canonical_text:
        raise AssertionError("TODO or placeholder remains in canonical text")
    for token in forbidden:
        if token in canonical_text:
            raise AssertionError(f"Stale metric token remains in canonical text: {token}")

    report = {
        "status": "PASS",
        "canonical_figures_validated": len(required_figures),
        "canonical_tables_validated": len(required_tables),
        "report_pages": page_counts,
        "stale_metric_tokens_in_final_pdfs": 0,
        "architecture_parameter_totals": {"Model A": 2_553_349, "Model B": 37_957},
    }
    if args.write_report:
        output = args.write_report if args.write_report.is_absolute() else ROOT / args.write_report
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("STAGE 3 FINAL ARTIFACT VALIDATION = PASS")


if __name__ == "__main__":
    main()
