"""Evaluate the LLM-only comparator on a JSONL annotation set.

Each input line must have ``id`` and ``text``. Add optional ``gold_family`` and
``gold_flags`` fields to score clause-family accuracy and document-level flag
micro/macro precision/recall/F1. Predictions can be generated live via Gemini
or evaluated from saved prediction records offline for reproducible scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from termsheet_ai.baseline import LLMOnlyBaseline  # noqa: E402
from termsheet_ai.schemas.contracts import ClauseFamily  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load validated evaluation records from a JSONL file."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"Line {line_number} must contain a string 'text' field")
        records.append(record)
    return records


def filter_by_split(records: list[dict[str, Any]], split: str = "all") -> list[dict[str, Any]]:
    """Filter records by evaluation split with strict metadata validation.

    Safeguard: Never silently invent or assign splits when held-out evaluation is requested.
    """
    if split == "all":
        return records

    unassigned = [r for r in records if not r.get("split") or r.get("split") not in ("dev", "test")]
    if len(unassigned) == len(records):
        raise ValueError(
            f"Dataset split error: Cannot perform held-out '{split}' evaluation. "
            f"None of the {len(records)} records have declared 'split' assignments "
            "(metadata is missing or null). As documented in the project proposal (┬º3.1) and audit, "
            "held-out testing requires explicit split metadata. Refusing to invent split partitions."
        )

    matching = [r for r in records if r.get("split") == split]
    if not matching:
        raise ValueError(
            f"Dataset split error: No records found with split='{split}' among evaluated examples."
        )
    return matching


def flag_metrics(
    gold_sets: Iterable[set[str]], predicted_sets: Iterable[set[str]]
) -> dict[str, float]:
    """Return micro-averaged multi-label precision, recall, and F1."""
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(gold_sets, predicted_sets, strict=True):
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def macro_flag_metrics(
    gold_sets: Iterable[set[str]], predicted_sets: Iterable[set[str]]
) -> dict[str, Any]:
    """Return macro-averaged multi-label precision, recall, and F1 plus per-flag breakdown."""
    all_labels = set()
    for g in gold_sets:
        all_labels.update(g)
    for p in predicted_sets:
        all_labels.update(p)

    if not all_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_flag": {}}

    per_flag: dict[str, dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []
    gold_list = list(gold_sets)
    pred_list = list(predicted_sets)

    for label in sorted(all_labels):
        tp = sum(1 for g, p in zip(gold_list, pred_list, strict=True) if label in g and label in p)
        fp = sum(
            1 for g, p in zip(gold_list, pred_list, strict=True) if label not in g and label in p
        )
        fn = sum(
            1 for g, p in zip(gold_list, pred_list, strict=True) if label in g and label not in p
        )

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        per_flag[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1": round(macro_f1, 4),
        "per_flag": per_flag,
    }


def evaluate(
    records: list[dict[str, Any]],
    baseline: Optional[LLMOnlyBaseline] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
    split: str = "all",
    detailed: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run predictions or score pre-saved predictions against input gold labels."""
    if baseline is None and predictions is None:
        raise ValueError("Either baseline or pre-saved predictions must be supplied to evaluate()")

    pred_by_id: dict[str, dict[str, Any]] = {}
    if predictions is not None:
        for idx, p in enumerate(predictions):
            p_id = str(p.get("id", idx))
            pred_by_id[p_id] = p

    output: list[dict[str, Any]] = []
    family_total = family_correct = 0
    gold_flags: list[set[str]] = []
    predicted_flags: list[set[str]] = []

    for index, record in enumerate(records):
        rec_id = str(record.get("id", index))
        if baseline is not None:
            pred_obj = baseline.predict(record["text"])
            result = {"id": rec_id, **pred_obj.to_dict()}
            predicted_family_val = pred_obj.clause_family
            predicted_flags_val = set(pred_obj.risk_flags)
        else:
            # Use pre-saved prediction
            pred_dict = pred_by_id.get(
                rec_id, predictions[index] if index < len(predictions) else {}
            )
            raw_family = pred_dict.get("clause_family", "other")
            try:
                predicted_family_val = ClauseFamily(str(raw_family).lower())
            except ValueError:
                predicted_family_val = ClauseFamily.OTHER
            raw_flags = pred_dict.get("risk_flags", [])
            predicted_flags_val = (
                {str(f) for f in raw_flags} if isinstance(raw_flags, (list, tuple, set)) else set()
            )
            result = {
                "id": rec_id,
                "clause_family": predicted_family_val.value,
                "risk_flags": sorted(predicted_flags_val),
                "summary": str(pred_dict.get("summary", "")),
            }

        output.append(result)

        gold_family = record.get("gold_family")
        if gold_family is not None:
            family_total += 1
            try:
                family_correct += predicted_family_val == ClauseFamily(str(gold_family).lower())
            except ValueError:
                pass

        if "gold_flags" in record:
            raw_gold_flags = record["gold_flags"]
            if not isinstance(raw_gold_flags, list):
                raise ValueError("gold_flags must be a list when supplied")
            gold_flags.append({str(flag) for flag in raw_gold_flags})
            predicted_flags.append(predicted_flags_val)

    doc_ids = sorted({str(r.get("doc_id") or r.get("id", idx)) for idx, r in enumerate(records)})

    metrics: dict[str, Any] = {
        "metadata": {
            "sample_count": len(records),
            "evaluated_document_ids": doc_ids,
            "split": split,
            "evaluation_mode": "saved_predictions" if predictions is not None else "live_llm",
        },
        "records": len(records),
    }

    if family_total:
        family_acc = round(family_correct / family_total, 4)
        metrics["family_accuracy"] = family_acc
        metrics["clause_family_classification"] = {
            "evaluated_samples": family_total,
            "accuracy": family_acc,
        }

    if gold_flags:
        micro = flag_metrics(gold_flags, predicted_flags)
        metrics["risk_flag_micro"] = micro
        risk_detection: dict[str, Any] = {
            "evaluated_samples": len(gold_flags),
            "micro": micro,
        }
        if detailed:
            macro = macro_flag_metrics(gold_flags, predicted_flags)
            metrics["risk_flag_macro"] = macro
            risk_detection["macro"] = macro
        metrics["risk_flag_detection"] = risk_detection

    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TermSheet AI LLM-only baseline.")
    parser.add_argument(
        "input", type=Path, help="JSONL records containing text and optional gold labels"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional path to saved JSONL predictions to score offline without live API calls",
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "test"],
        default="all",
        help="Dataset split to evaluate ('all', 'dev', 'test'). Defaults to 'all'.",
    )
    parser.add_argument("--output", type=Path, default=Path("llm_baseline_predictions.jsonl"))
    parser.add_argument("--model", default=None, help="Optional Gemini model override")
    parser.add_argument("--no-cache", action="store_true", help="Disable the existing prompt cache")
    parser.add_argument(
        "--detailed", action="store_true", help="Include macro and per-flag breakdowns"
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    records = filter_by_split(records, split=args.split)

    if args.predictions is not None:
        saved_predictions = load_jsonl(args.predictions)
        metrics, predictions = evaluate(
            records,
            predictions=saved_predictions,
            split=args.split,
            detailed=args.detailed,
        )
    else:
        baseline = LLMOnlyBaseline.with_gemini(model=args.model, use_cache=not args.no_cache)
        metrics, predictions = evaluate(
            records,
            baseline=baseline,
            split=args.split,
            detailed=args.detailed,
        )
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        print(f"Predictions written to {args.output}")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
