"""Generate report figures from the JSON files written by the evaluation scripts.

Inputs (a missing input skips only the figures that need it):
  reports/eval_results.json         written by scripts/evaluate_pipeline.py
  reports/baseline_comparison.json  written by scripts/benchmark_llm_baseline.py
  reports/corpus_eval.json          EvaluationReport.to_dict() from termsheet_ai.evaluation.corpus

Outputs: PNG files in reports/figures/.   Run:  python scripts/make_plots.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: no display needed (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE, GREY = "#1F5A6B", "#C4491C", "#7A7F87"
FACT_LABELS = {
    "multiple": "Liquidation multiple",
    "participation": "Participation mode",
    "anti_dilution": "Anti-dilution type",
    "founder_seats": "Founder board seats",
    "investor_seats": "Investor board seats",
}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _label_bars(ax: plt.Axes, bars, fmt: str = "{:.0%}", horizontal: bool = False) -> None:
    for bar in bars:
        if horizontal:
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(bar.get_width()),
                va="center",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fmt.format(bar.get_height()),
                ha="center",
                fontsize=9,
            )


def plot_flag_prf(results: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    s = results.get("summary_metrics", {})
    keys = ["risk_flag_precision", "risk_flag_recall", "risk_flag_f1"]
    if not all(k in s for k in keys):
        return None
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["Precision", "Recall", "F1"], [s[k] for k in keys], color=[BLUE, BLUE, ORANGE])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Risk-flag detection ({s.get('total_documents_evaluated', '?')} documents)")
    _label_bars(ax, bars)
    return _save(fig, out_dir, "fig_flag_prf.png")


def plot_fact_accuracy(results: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    acc = results.get("summary_metrics", {}).get("fact_extraction_accuracy")
    if not acc:
        return None
    names = [FACT_LABELS.get(k, k) for k in acc]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(names, list(acc.values()), color=BLUE)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel("Field-level accuracy against gold labels")
    ax.set_title("Fact extraction accuracy (S3)")
    _label_bars(ax, bars, horizontal=True)
    return _save(fig, out_dir, "fig_fact_accuracy.png")


def plot_risk_distribution(results: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    """Histogram of per-document risk, standard vs adversarial documents.

    Uses ``risk_score`` when the evaluation script recorded it; otherwise falls back to
    the number of predicted flags per document and labels the axis accordingly.
    """
    docs: List[Dict[str, Any]] = results.get("document_results", [])
    if not docs:
        return None
    use_score = all(d.get("risk_score") is not None for d in docs)

    def value(d: Dict[str, Any]) -> float:
        return float(d["risk_score"]) if use_score else float(len(d.get("predicted_flags", [])))

    standard = [value(d) for d in docs if not d.get("is_adversarial")]
    adversarial = [value(d) for d in docs if d.get("is_adversarial")]
    top = max(standard + adversarial + [1.0])
    bins = 10 if use_score else int(top) + 1
    rng = (0, 100) if use_score else (0, int(top) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(
        [standard, adversarial],
        bins=bins,
        range=rng,
        color=[BLUE, ORANGE],
        label=[f"Standard (n={len(standard)})", f"Adversarial (n={len(adversarial)})"],
    )
    ax.set_xlabel("Document risk score (0-100)" if use_score else "Predicted flags per document")
    ax.set_ylabel("Documents")
    ax.set_title("Risk distribution: standard vs adversarial term sheets")
    ax.legend()
    return _save(fig, out_dir, "fig_risk_distribution.png")


def plot_confusion_matrix(corpus_report: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    matrix = corpus_report.get("confusion_matrix")
    if not matrix:
        return None
    labels = sorted(matrix)
    counts = [[matrix[g].get(p, 0) for p in labels] for g in labels]
    row_sums = [sum(r) or 1 for r in counts]
    norm = [[c / row_sums[i] for c in r] for i, r in enumerate(counts)]
    size = max(5.5, 0.55 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    for i, row in enumerate(counts):
        for j, c in enumerate(row):
            if c:
                ax.text(
                    j,
                    i,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[i][j] > 0.55 else "black",
                )
    ax.set_xlabel("Predicted clause family")
    ax.set_ylabel("Gold clause family")
    ax.set_title("Clause classification confusion matrix (row-normalised, counts shown)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of gold clauses")
    return _save(fig, out_dir, "fig_confusion_matrix.png")


def plot_family_f1(corpus_report: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    metrics = corpus_report.get("family_metrics")
    if not metrics:
        return None
    names = sorted(metrics)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(names) + 1.6))
    bars = ax.barh(names, [metrics[n]["f1"] for n in names], color=BLUE)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel("F1")
    ax.set_title("Per-family clause classification F1")
    _label_bars(ax, bars, "{:.2f}", horizontal=True)
    return _save(fig, out_dir, "fig_family_f1.png")


def plot_baseline_comparison(comp: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    ours, base = comp.get("neuro_symbolic_pipeline"), comp.get("llm_only_baseline")
    keys = [
        ("numerical_hallucination_rate", "Numerical\nhallucination rate"),
        ("proof_tree_justification_rate", "Proof-tree\njustification rate"),
    ]
    if not ours or not base or not all(k in ours and k in base for k, _ in keys):
        return None
    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.36
    for i, (k, _) in enumerate(keys):
        b1 = ax.bar(
            i - width / 2, ours[k], width, color=BLUE, label="TermSheet AI" if i == 0 else None
        )
        b2 = ax.bar(
            i + width / 2, base[k], width, color=GREY, label="LLM-only baseline" if i == 0 else None
        )
        _label_bars(ax, list(b1) + list(b2))
    ax.set_xticks(range(len(keys)), labels=[label for _, label in keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title(f"Pipeline vs LLM-only baseline ({comp.get('total_documents', '?')} documents)")
    ax.legend()
    return _save(fig, out_dir, "fig_baseline_comparison.png")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate evaluation figures")
    p.add_argument("--results", default="reports/eval_results.json")
    p.add_argument("--baseline", default="reports/baseline_comparison.json")
    p.add_argument("--corpus-report", default="reports/corpus_eval.json")
    p.add_argument("--out-dir", default="reports/figures")
    args = p.parse_args(argv)

    def resolve(s: str) -> Path:
        path = Path(s)
        return path if path.is_absolute() else REPO_ROOT / path

    out_dir = resolve(args.out_dir)
    results, baseline, corpus = (
        _load(resolve(a)) for a in (args.results, args.baseline, args.corpus_report)
    )
    jobs = [
        ("flag precision/recall/F1", results, plot_flag_prf),
        ("fact accuracy", results, plot_fact_accuracy),
        ("risk distribution", results, plot_risk_distribution),
        ("confusion matrix", corpus, plot_confusion_matrix),
        ("per-family F1", corpus, plot_family_f1),
        ("baseline comparison", baseline, plot_baseline_comparison),
    ]
    made = 0
    for name, data, fn in jobs:
        path = fn(data, out_dir) if data else None
        if path:
            made += 1
            print(f"wrote   {path}")
        else:
            print(f"skipped {name} (input file missing or lacks the needed fields)")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
