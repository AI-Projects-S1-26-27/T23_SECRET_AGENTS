"""Stage S2 SBERT Dev vs. Held-Out Test Split Evaluation & Generalization Analysis.

Evaluates the calibrated Stage S2 SBERT Classifier and TF-IDF Baseline across:
- Dev Split: 16 synthetic documents (222 clauses)
- Held-Out Test Split: 14 synthetic documents (201 clauses)
- Combined Synthetic Corpus: 30 documents (423 clauses)

Analyzes generalization gap, overfitting risk, and adversarial clause detection
on unseen held-out instruments.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src/ and repo root are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_synthetic import is_adversarial_clause, map_ground_truth_family  # noqa: E402
from termsheet_ai.classify.metrics import compute_metrics, format_metrics_table  # noqa: E402
from termsheet_ai.classify.sbert import SbertClassifier  # noqa: E402
from termsheet_ai.classify.tfidf import TfidfBaselineClassifier  # noqa: E402
from termsheet_ai.ingest.ingestor import Ingestor  # noqa: E402
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def load_split_clauses(
    split_name: str,
) -> Tuple[List[ClauseSpan], List[ClauseFamily], List[str], List[bool]]:
    """Load ingested clauses, gold labels, doc IDs, and adversarial flags for a given split."""
    manifest_path = REPO_ROOT / "corpus" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = [
        d
        for d in manifest.get("evaluation_corpus", {}).get("documents", [])
        if d.get("split") == split_name and d.get("is_synthetic")
    ]

    ingestor = Ingestor()
    clauses: List[ClauseSpan] = []
    gold_labels: List[ClauseFamily] = []
    doc_ids: List[str] = []
    is_adv_doc: List[bool] = []

    for entry in entries:
        full_path = REPO_ROOT / "corpus" / entry["filename"]
        if not full_path.exists():
            continue

        is_adv = "ts_synth_adv_" in entry["filename"]
        _, doc_clauses = ingestor.ingest(full_path)
        for c in doc_clauses:
            gold = map_ground_truth_family(c.section_header, c.raw_text)
            clauses.append(c)
            gold_labels.append(gold)
            doc_ids.append(entry["id"])
            is_adv_doc.append(is_adv)

    return clauses, gold_labels, doc_ids, is_adv_doc


def run_split_evaluation() -> Dict[str, Any]:
    """Run full evaluation comparing dev, held-out test, and combined splits."""
    print("=" * 84)
    print("  TERMSHEET AI — STAGE S2 HELD-OUT TEST EVALUATION & GENERALIZATION REPORT")
    print("=" * 84)

    # 1. Load Splits
    dev_clauses, dev_gold, dev_docs, dev_adv = load_split_clauses("dev")
    test_clauses, test_gold, test_docs, test_adv = load_split_clauses("test")

    all_clauses = dev_clauses + test_clauses
    all_gold = dev_gold + test_gold

    print(f"\n[+] Loaded Dev Split:     {len(set(dev_docs))} docs, {len(dev_clauses)} clauses")
    print(
        f"[+] Loaded Test Split:    {len(set(test_docs))} docs, {len(test_clauses)} clauses (HELD-OUT)"
    )
    print(
        f"[+] Total Corpus:         {len(set(dev_docs) | set(test_docs))} docs, {len(all_clauses)} clauses"
    )

    # 2. Initialize Models
    print("\n[*] Initializing Calibrated SBERT Hybrid Classifier...")
    sbert_hybrid = SbertClassifier(mode="hybrid")

    print("[*] Initializing TF-IDF Baseline Classifier...")
    tfidf_baseline = TfidfBaselineClassifier()

    # 3. Predict Dev Split
    print("[*] Evaluating on Dev Split...")
    sbert_dev_preds = [c.detected_family for c in sbert_hybrid.classify(dev_clauses)]
    tfidf_dev_preds = [c.detected_family for c in tfidf_baseline.classify(dev_clauses)]
    sbert_dev_report = compute_metrics(dev_gold, sbert_dev_preds)
    tfidf_dev_report = compute_metrics(dev_gold, tfidf_dev_preds)

    # 4. Predict Held-Out Test Split
    print("[*] Evaluating on Held-Out Test Split (Unseen Documents)...")
    sbert_test_preds = [c.detected_family for c in sbert_hybrid.classify(test_clauses)]
    tfidf_test_preds = [c.detected_family for c in tfidf_baseline.classify(test_clauses)]
    sbert_test_report = compute_metrics(test_gold, sbert_test_preds)
    tfidf_test_report = compute_metrics(test_gold, tfidf_test_preds)

    # 5. Predict Combined Corpus
    print("[*] Evaluating on Combined Corpus (30 Documents)...")
    sbert_all_preds = [c.detected_family for c in sbert_hybrid.classify(all_clauses)]
    tfidf_all_preds = [c.detected_family for c in tfidf_baseline.classify(all_clauses)]
    sbert_all_report = compute_metrics(all_gold, sbert_all_preds)
    tfidf_all_report = compute_metrics(all_gold, tfidf_all_preds)

    # 6. Print Model Comparison Table
    print("\n" + "=" * 84)
    print("  1. SPLIT COMPARISON: DEV VS. HELD-OUT TEST")
    print("=" * 84)
    split_reports = {
        "sbert_dev_split": sbert_dev_report,
        "sbert_test_heldout": sbert_test_report,
        "sbert_combined_all": sbert_all_report,
        "tfidf_dev_split": tfidf_dev_report,
        "tfidf_test_heldout": tfidf_test_report,
        "tfidf_combined_all": tfidf_all_report,
    }
    print(format_metrics_table(split_reports))

    # 7. Generalization Gap Analysis
    acc_gap = sbert_dev_report.accuracy - sbert_test_report.accuracy
    macro_f1_gap = sbert_dev_report.macro_f1 - sbert_test_report.macro_f1
    weighted_f1_gap = sbert_dev_report.weighted_f1 - sbert_test_report.weighted_f1

    print("=" * 84)
    print("  2. GENERALIZATION GAP ANALYSIS (DEV vs. HELD-OUT TEST)")
    print("=" * 84)
    print(f"  • Accuracy Generalization Gap:         {acc_gap * 100:+.2f}%")
    print(f"  • Macro F1 Generalization Gap:         {macro_f1_gap:+.4f}")
    print(f"  • Weighted F1 Generalization Gap:      {weighted_f1_gap:+.4f}")
    if abs(acc_gap) < 0.05 and abs(macro_f1_gap) < 0.05:
        print(
            "  • Overfitting Assessment:              PASS (Zero significant overfitting; excellent generalization)"
        )
    else:
        print("  • Overfitting Assessment:              WARN (Check threshold sensitivity)")

    # 8. Test Set Adversarial Robustness
    print("\n" + "=" * 84)
    print("  3. HELD-OUT TEST SET ADVERSARIAL CLAUSE ROBUSTNESS")
    print("=" * 84)
    adv_test_counts: Dict[str, Dict[str, int]] = {
        "Aggressive Liquidation (>=3x / part)": {"total": 0, "correct": 0},
        "Full-Ratchet Anti-Dilution": {"total": 0, "correct": 0},
        "Investor-Controlled Board": {"total": 0, "correct": 0},
        "Hostile Founder Vesting": {"total": 0, "correct": 0},
    }

    for clause, gold, pred in zip(test_clauses, test_gold, sbert_test_preds, strict=True):
        is_adv, adv_desc = is_adversarial_clause(clause, gold)
        if is_adv:
            for cat_key in adv_test_counts:
                if (
                    ("Liquidation" in adv_desc and "Liquidation" in cat_key)
                    or ("Anti-Dilution" in adv_desc and "Anti-Dilution" in cat_key)
                    or ("Board" in adv_desc and "Board" in cat_key)
                    or ("Vesting" in adv_desc and "Vesting" in cat_key)
                ):
                    adv_test_counts[cat_key]["total"] += 1
                    if pred == gold:
                        adv_test_counts[cat_key]["correct"] += 1

    print(
        f"{'Adversarial Category (Held-Out)':<38} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10}"
    )
    print("-" * 72)
    tot_adv = 0
    tot_cor = 0
    for cat, counts in adv_test_counts.items():
        tot = counts["total"]
        cor = counts["correct"]
        tot_adv += tot
        tot_cor += cor
        acc = (cor / tot) if tot > 0 else 0.0
        print(f"{cat:<38} | {tot:<6} | {cor:<8} | {acc * 100:.1f}%")
    print("-" * 72)
    overall_adv_acc = (tot_cor / tot_adv) if tot_adv > 0 else 0.0
    print(
        f"{'OVERALL HELD-OUT ADVERSARIAL ACCURACY':<38} | {tot_adv:<6} | {tot_cor:<8} | {overall_adv_acc * 100:.1f}%"
    )

    # 9. Save Evaluation Payload
    output_dir = REPO_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    report_payload = {
        "dev_split": {
            "document_count": len(set(dev_docs)),
            "clause_count": len(dev_clauses),
            "metrics": sbert_dev_report.to_dict(),
        },
        "held_out_test_split": {
            "document_count": len(set(test_docs)),
            "clause_count": len(test_clauses),
            "metrics": sbert_test_report.to_dict(),
        },
        "combined_corpus": {
            "document_count": len(set(dev_docs) | set(test_docs)),
            "clause_count": len(all_clauses),
            "metrics": sbert_all_report.to_dict(),
        },
        "generalization_gap": {
            "accuracy_gap": round(acc_gap, 4),
            "macro_f1_gap": round(macro_f1_gap, 4),
            "weighted_f1_gap": round(weighted_f1_gap, 4),
        },
        "held_out_adversarial_detection": {
            "total_adversarial_clauses": tot_adv,
            "correct_detections": tot_cor,
            "accuracy": round(overall_adv_acc, 4),
            "breakdown": adv_test_counts,
        },
        "model_parameters": {
            "model_name": sbert_hybrid.model_name,
            "alpha": sbert_hybrid.alpha,
            "confidence_threshold": sbert_hybrid.confidence_threshold,
            "ood_threshold": sbert_hybrid.ood_threshold,
            "mode": sbert_hybrid.mode,
        },
    }

    report_path = output_dir / "sbert_split_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[+] Saved detailed evaluation report to {report_path}")

    return report_payload


if __name__ == "__main__":
    run_split_evaluation()
