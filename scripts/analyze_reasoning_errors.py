
"""Analyze S4 risk-flag errors from existing pipeline evaluation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def analyze(results: dict[str, Any]) -> dict[str, Any]:
    """Compare predicted risk-rule IDs against gold annotations."""
    documents = results["document_results"]

    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    true_positives: Counter[str] = Counter()
    document_errors = []
    execution_error_docs = 0

    for doc in documents:
        expected = set(doc.get("expected_flags", []))
        predicted = set(doc.get("predicted_flags", []))

        tp = sorted(expected & predicted)
        fp = sorted(predicted - expected)
        fn = sorted(expected - predicted)

        true_positives.update(tp)
        false_positives.update(fp)
        false_negatives.update(fn)

        errors = doc.get("execution_errors") or []
        if errors:
            execution_error_docs += 1

        document_errors.append(
            {
                "doc_id": doc.get("doc_id"),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "grounding_rate": doc.get("grounding_rate"),
                "numerical_verification_failure_rate": doc.get(
                    "hallucination_rate"
                ),
                "execution_errors": errors,
            }
        )

    return {
        "documents_analyzed": len(documents),
        "total_tp": sum(true_positives.values()),
        "total_fp": sum(false_positives.values()),
        "total_fn": sum(false_negatives.values()),
        "documents_with_execution_errors": execution_error_docs,
        "most_common_false_positives": sorted(
            false_positives.items(),
            key=lambda item: (-item[1], item[0]),
        ),
        "most_common_false_negatives": sorted(
            false_negatives.items(),
            key=lambda item: (-item[1], item[0]),
        ),
        "document_errors": sorted(
            document_errors,
            key=lambda doc: str(doc["doc_id"]),
        ),
    }


def format_report(analysis: dict[str, Any]) -> str:
    """Produce a Markdown summary of confirmed rule-ID mismatches."""
    tp = analysis["total_tp"]
    fp = analysis["total_fp"]
    fn = analysis["total_fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    lines = [
        "# Member 6 — S4 Risk-Flag Error Analysis",
        "",
        f"Documents analyzed: {analysis['documents_analyzed']}",
        f"True positives: {tp}",
        f"False positives: {fp}",
        f"False negatives: {fn}",
        f"Precision: {precision:.2%}",
        f"Recall: {recall:.2%}",
        f"F1: {f1:.2%}",
        "Documents with execution errors: "
        f"{analysis['documents_with_execution_errors']}",
        "",
        "## Most common false-positive rules",
        "",
    ]

    for rule_id, count in analysis["most_common_false_positives"]:
        lines.append(f"- {rule_id}: {count}")

    lines.extend(["", "## Most common missed rules", ""])

    for rule_id, count in analysis["most_common_false_negatives"]:
        lines.append(f"- {rule_id}: {count}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results identify risk-rule ID mismatches against "
            "the gold annotations. They do not independently identify "
            "whether each mismatch originated in extraction, grounding, "
            "rule coverage, or inference.",
            "",
            "## Document-level errors",
            "",
        ]
    )

    for doc in analysis["document_errors"]:
        if not doc["false_positives"] and not doc["false_negatives"]:
            continue

        lines.extend(
            [
                f"### {doc['doc_id']}",
                "",
                f"- False positives: {', '.join(doc['false_positives']) or 'None'}",
                f"- Missed flags: {', '.join(doc['false_negatives']) or 'None'}",
                f"- Grounding rate: {doc['grounding_rate']}",
                f"- Execution errors: {doc['execution_errors']}",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/eval_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    args = parser.parse_args()

    results = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "reasoning_error_analysis.json"
    md_path = args.output_dir / "reasoning_error_analysis.md"

    json_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        format_report(analysis),
        encoding="utf-8",
    )

    print(f"Analyzed {analysis['documents_analyzed']} documents")
    print(f"TP={analysis['total_tp']}")
    print(f"FP={analysis['total_fp']}")
    print(f"FN={analysis['total_fn']}")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
