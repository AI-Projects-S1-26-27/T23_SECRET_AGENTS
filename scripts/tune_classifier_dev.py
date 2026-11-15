"""Dev Set Threshold Tuning & Hyperparameter Calibration for Stage S2 SBERT Classifier.

Evaluates hyperparameter combinations over the 16 registered dev-split term sheets
in corpus/manifest.json (221 clauses across both standard and adversarial instruments).

Tuning parameters:
- confidence_threshold: Minimum probability required before falling back to ClauseFamily.OTHER
- ood_threshold: Minimum cosine similarity to any exemplar required to avoid OOD rejection
- alpha: Hybrid ensemble weight balancing Logistic Regression head vs. ChromaDB k-NN
- k_neighbors: Number of nearest neighbors queried in ChromaDB
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

import numpy as np  # noqa: E402

from scripts.benchmark_synthetic import map_ground_truth_family  # noqa: E402
from termsheet_ai.classify.metrics import compute_metrics  # noqa: E402
from termsheet_ai.classify.sbert import (  # noqa: E402
    SbertClassifier,
    _is_noise_or_empty,
    prepare_clause_text,
)
from termsheet_ai.ingest.ingestor import Ingestor  # noqa: E402
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def load_dev_corpus() -> Tuple[List[ClauseSpan], List[ClauseFamily], List[str]]:
    """Load and ingest all auto-labeled dev-split term sheets from manifest.json."""
    manifest_path = REPO_ROOT / "corpus" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    dev_entries = [
        d
        for d in manifest.get("evaluation_corpus", {}).get("documents", [])
        if d.get("split") == "dev" and d.get("is_synthetic")
    ]

    ingestor = Ingestor()
    all_clauses: List[ClauseSpan] = []
    gold_labels: List[ClauseFamily] = []
    doc_names: List[str] = []

    for entry in dev_entries:
        rel_path = entry["filename"]
        full_path = REPO_ROOT / "corpus" / rel_path
        if not full_path.exists():
            continue

        _, clauses = ingestor.ingest(full_path)
        for c in clauses:
            gold = map_ground_truth_family(c.section_header, c.raw_text)
            all_clauses.append(c)
            gold_labels.append(gold)
            doc_names.append(entry["id"])

    return all_clauses, gold_labels, doc_names


def run_dev_tuning() -> Dict[str, Any]:
    """Execute grid search over hyperparameters on the dev corpus."""
    print("=" * 84)
    print("  TERMSHEET AI — STAGE S2 SBERT DEV SET HYPERPARAMETER TUNING")
    print("=" * 84)

    clauses, gold_labels, doc_names = load_dev_corpus()
    print(
        f"\n[+] Loaded {len(set(doc_names))} dev-split term sheets ({len(clauses)} clauses total)."
    )

    # Base classifier instance (model weights loaded once)
    base_clf = SbertClassifier(mode="hybrid")

    # Pre-filter and pre-encode dev clauses once
    valid_indices: List[int] = []
    valid_texts: List[str] = []
    is_noise_mask = [False] * len(clauses)

    for idx, c in enumerate(clauses):
        prep = prepare_clause_text(c.raw_text, c.section_header)
        if _is_noise_or_empty(prep):
            is_noise_mask[idx] = True
        else:
            valid_indices.append(idx)
            valid_texts.append(prep)

    print(f"[*] Encoding {len(valid_texts)} valid clauses with SBERT...")
    embeddings = base_clf._encode(valid_texts)
    print("[*] Computing base LR and k-NN probabilities...")
    p_knn, max_sims = base_clf._predict_knn_proba_and_max_sim(embeddings)
    p_lr = base_clf._predict_lr_proba(embeddings)

    # Hyperparameter search grid
    confidence_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    ood_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
    alpha_grid = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]

    best_score = -1.0
    best_config: Dict[str, Any] = {}
    tuning_records: List[Dict[str, Any]] = []

    print("\n[*] Evaluating grid search configurations in-memory...")
    for alpha in alpha_grid:
        hybrid_p = alpha * p_lr + (1.0 - alpha) * p_knn
        best_class_indices = np.argmax(hybrid_p, axis=1)
        best_confidences = np.max(hybrid_p, axis=1)

        for ood in ood_grid:
            ood_mask = max_sims < ood

            for conf in confidence_grid:
                conf_mask = best_confidences < conf
                fallback_mask = ood_mask | conf_mask

                # Construct predictions
                preds: List[ClauseFamily] = [ClauseFamily.OTHER] * len(clauses)
                for local_i, global_i in enumerate(valid_indices):
                    if fallback_mask[local_i]:
                        preds[global_i] = ClauseFamily.OTHER
                    else:
                        cls_idx = best_class_indices[local_i]
                        preds[global_i] = base_clf._classes[cls_idx]

                report = compute_metrics(gold_labels, preds)

                # Composite metric: 60% Macro F1 + 40% Accuracy
                composite_score = 0.60 * report.macro_f1 + 0.40 * report.accuracy

                record = {
                    "alpha": alpha,
                    "ood_threshold": ood,
                    "confidence_threshold": conf,
                    "accuracy": round(report.accuracy, 4),
                    "macro_precision": round(report.macro_precision, 4),
                    "macro_recall": round(report.macro_recall, 4),
                    "macro_f1": round(report.macro_f1, 4),
                    "weighted_f1": round(report.weighted_f1, 4),
                    "composite_score": round(composite_score, 4),
                }
                tuning_records.append(record)

                if composite_score > best_score:
                    best_score = composite_score
                    best_config = record

    # Sort records by composite score
    tuning_records.sort(key=lambda r: r["composite_score"], reverse=True)

    print("\n" + "=" * 84)
    print("  TOP 5 CALIBRATED CONFIGURATIONS (RANKED BY COMPOSITE F1/ACCURACY)")
    print("=" * 84)
    print(
        f"{'Rank':<5} | {'Alpha':<6} | {'OOD Thr':<8} | {'Conf Thr':<9} | "
        f"{'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Composite':<9}"
    )
    print("-" * 84)
    for rank, rec in enumerate(tuning_records[:5], start=1):
        print(
            f"{rank:<5} | {rec['alpha']:<6.1f} | {rec['ood_threshold']:<8.2f} | "
            f"{rec['confidence_threshold']:<9.2f} | {rec['accuracy']:<9.4f} | "
            f"{rec['macro_f1']:<9.4f} | {rec['weighted_f1']:<11.4f} | {rec['composite_score']:<9.4f}"
        )

    print("\n" + "=" * 84)
    print("  OPTIMAL DEV-CALIBRATED HYPERPARAMETERS")
    print("=" * 84)
    print(f"  • Best Alpha (LR Head Weight):        {best_config['alpha']}")
    print(f"  • Best OOD Threshold:                 {best_config['ood_threshold']}")
    print(f"  • Best Confidence Threshold:          {best_config['confidence_threshold']}")
    print(f"  • Dev Set Accuracy:                   {best_config['accuracy'] * 100:.2f}%")
    print(f"  • Dev Set Macro F1:                   {best_config['macro_f1']:.4f}")
    print(f"  • Dev Set Weighted F1:                {best_config['weighted_f1']:.4f}")
    print("=" * 84 + "\n")

    # Save results to reports/sbert_dev_tuning.json
    output_dir = REPO_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    tuning_payload = {
        "best_config": best_config,
        "dev_document_count": len(set(doc_names)),
        "dev_clause_count": len(clauses),
        "top_configurations": tuning_records[:10],
    }
    with open(output_dir / "sbert_dev_tuning.json", "w", encoding="utf-8") as f:
        json.dump(tuning_payload, f, indent=2)

    return best_config


if __name__ == "__main__":
    run_dev_tuning()
