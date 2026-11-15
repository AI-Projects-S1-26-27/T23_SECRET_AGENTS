"""Inter-annotator agreement between Annotator A and Annotator B.

Reads two directories of ``<document>.gold.json`` files (guide format, one per document),
matches clauses by (document file name, clause_id) and reports:
  - Cohen's kappa on clause_family
  - Cohen's kappa on expected_flags (binary decision per clause x rule)
  - exact-match rate per attribute (only where at least one annotator filled it in)
  - a list of disagreements for the reconciliation meeting

Writes reports/annotator_agreement.json and reports/annotator_agreement.md.

Run:  python scripts/compute_kappa.py --a-dir corpus/gold/annotator_a --b-dir corpus/gold/annotator_b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> Optional[float]:
    """Cohen's kappa for two equal-length label sequences.

    Returns None when there is nothing to compare. When both annotators used one single
    identical label throughout, kappa is 0/0; we report 1.0 (perfect agreement).
    """
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def landis_koch(kappa: Optional[float]) -> str:
    """Landis & Koch (1977) verbal labels."""
    if kappa is None:
        return "n/a"
    if kappa < 0.0:
        return "poor"
    for upper, name in [
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
    ]:
        if kappa <= upper:
            return name
    return "almost perfect"


Pair = Tuple[str, str, Dict[str, Any], Dict[str, Any]]  # (doc, clause_id, clause_a, clause_b)


def load_dir(directory: Path) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    for f in sorted(directory.glob("*.gold.json")):
        docs[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return docs


def align(
    docs_a: Dict[str, Dict[str, Any]], docs_b: Dict[str, Dict[str, Any]]
) -> Tuple[List[Pair], Dict[str, List[str]]]:
    """Match clauses across annotators; return pairs and everything that failed to match."""
    pairs: List[Pair] = []
    unmatched: Dict[str, List[str]] = {
        "documents_only_a": [],
        "documents_only_b": [],
        "clauses": [],
    }
    unmatched["documents_only_a"] = sorted(set(docs_a) - set(docs_b))
    unmatched["documents_only_b"] = sorted(set(docs_b) - set(docs_a))
    for name in sorted(set(docs_a) & set(docs_b)):
        ca = {c["clause_id"]: c for c in docs_a[name].get("clauses", [])}
        cb = {c["clause_id"]: c for c in docs_b[name].get("clauses", [])}
        for cid in sorted(set(ca) & set(cb)):
            pairs.append((name, cid, ca[cid], cb[cid]))
        for cid in sorted(set(ca) ^ set(cb)):
            side = "A" if cid in ca else "B"
            unmatched["clauses"].append(f"{name}:{cid} (only in {side})")
    return pairs, unmatched


def family_agreement(pairs: List[Pair]) -> Dict[str, Any]:
    a = [p[2].get("clause_family") or "unlabelled" for p in pairs]
    b = [p[3].get("clause_family") or "unlabelled" for p in pairs]
    kappa = cohen_kappa(a, b)
    return {
        "n_clauses": len(pairs),
        "percent_agreement": (sum(x == y for x, y in zip(a, b, strict=True)) / len(a))
        if a
        else None,
        "kappa": kappa,
        "interpretation": landis_koch(kappa),
    }


def flag_agreement(pairs: List[Pair], rule_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Binary kappa over every (clause, rule) decision: was the rule listed or not."""
    universe = set(rule_ids or set())
    if not universe:
        for _, _, ca, cb in pairs:
            universe |= set(ca.get("expected_flags", [])) | set(cb.get("expected_flags", []))
    rules = sorted(universe)
    a: List[int] = []
    b: List[int] = []
    exact = 0
    for _, _, ca, cb in pairs:
        fa, fb = set(ca.get("expected_flags", [])), set(cb.get("expected_flags", []))
        exact += fa == fb
        for rule in rules:
            a.append(int(rule in fa))
            b.append(int(rule in fb))
    kappa = cohen_kappa(a, b)
    return {
        "n_decisions": len(a),
        "n_rules": len(rules),
        "clause_exact_match_rate": (exact / len(pairs)) if pairs else None,
        "kappa": kappa,
        "interpretation": landis_koch(kappa),
    }


def attribute_agreement(pairs: List[Pair]) -> Dict[str, Dict[str, Any]]:
    """Exact-match rate per attribute among clauses where either annotator filled it in."""
    totals: Dict[str, List[int]] = {}
    for _, _, ca, cb in pairs:
        attrs_a, attrs_b = ca.get("attributes") or {}, cb.get("attributes") or {}
        for key in set(attrs_a) | set(attrs_b):
            va, vb = attrs_a.get(key), attrs_b.get(key)
            if va is None and vb is None:
                continue
            totals.setdefault(key, [0, 0])
            totals[key][1] += 1
            totals[key][0] += va == vb
    return {
        k: {"matches": v[0], "compared": v[1], "exact_match_rate": v[0] / v[1]}
        for k, v in sorted(totals.items())
    }


def disagreements(pairs: List[Pair]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc, cid, ca, cb in pairs:
        text = (ca.get("clause_text") or cb.get("clause_text") or "").strip().replace("\n", " ")
        fam_a, fam_b = ca.get("clause_family"), cb.get("clause_family")
        if fam_a != fam_b:
            out.append(
                {
                    "doc": doc,
                    "clause_id": cid,
                    "field": "clause_family",
                    "A": fam_a,
                    "B": fam_b,
                    "clause_text": text[:200],
                }
            )
        fl_a, fl_b = sorted(ca.get("expected_flags", [])), sorted(cb.get("expected_flags", []))
        if fl_a != fl_b:
            out.append(
                {
                    "doc": doc,
                    "clause_id": cid,
                    "field": "expected_flags",
                    "A": fl_a,
                    "B": fl_b,
                    "clause_text": text[:200],
                }
            )
    return out


def build_report(
    docs_a: Dict[str, Dict[str, Any]],
    docs_b: Dict[str, Dict[str, Any]],
    rule_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    pairs, unmatched = align(docs_a, docs_b)
    return {
        "documents_compared": len({p[0] for p in pairs}),
        "clause_family": family_agreement(pairs),
        "expected_flags": flag_agreement(pairs, rule_ids),
        "attributes": attribute_agreement(pairs),
        "unmatched": unmatched,
        "disagreements": disagreements(pairs),
    }


def _fmt(x: Optional[float], pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def to_markdown(report: Dict[str, Any]) -> str:
    fam, flg = report["clause_family"], report["expected_flags"]
    lines = [
        "# Inter-Annotator Agreement (Annotator A vs Annotator B)",
        "",
        f"Documents compared: **{report['documents_compared']}**, "
        f"clauses compared: **{fam['n_clauses']}**. Agreement is measured **before** reconciliation.",
        "",
        "| Measure | Percent agreement | Cohen's kappa | Interpretation (Landis & Koch) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Clause family | {_fmt(fam['percent_agreement'], True)} | {_fmt(fam['kappa'])} | {fam['interpretation']} |",
        f"| Expected flags (clause x rule decisions: {flg['n_decisions']}) | "
        f"{_fmt(flg['clause_exact_match_rate'], True)} (clause exact match) | {_fmt(flg['kappa'])} | {flg['interpretation']} |",
        "",
    ]
    if report["attributes"]:
        lines += [
            "## Attribute exact-match rate",
            "",
            "| Attribute | Matches | Compared | Rate |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for key, v in report["attributes"].items():
            lines.append(
                f"| {key} | {v['matches']} | {v['compared']} | {_fmt(v['exact_match_rate'], True)} |"
            )
        lines.append("")
    un = report["unmatched"]
    if un["documents_only_a"] or un["documents_only_b"] or un["clauses"]:
        lines += ["## Unmatched items", ""]
        lines += [f"- Only annotated by A: {n}" for n in un["documents_only_a"]]
        lines += [f"- Only annotated by B: {n}" for n in un["documents_only_b"]]
        lines += [f"- Clause {c}" for c in un["clauses"]]
        lines.append("")
    lines += [f"## Disagreements to reconcile ({len(report['disagreements'])})", ""]
    for d in report["disagreements"]:
        lines.append(f"- `{d['doc']}` {d['clause_id']} / {d['field']}: A = {d['A']}, B = {d['B']}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute inter-annotator agreement")
    p.add_argument("--a-dir", default="corpus/gold/annotator_a")
    p.add_argument("--b-dir", default="corpus/gold/annotator_b")
    p.add_argument("--rules", default="docs/rules_v1.yaml")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args(argv)

    def resolve(s: str) -> Path:
        path = Path(s)
        return path if path.is_absolute() else REPO_ROOT / path

    a_dir, b_dir = resolve(args.a_dir), resolve(args.b_dir)
    for d in (a_dir, b_dir):
        if not d.is_dir():
            print(f"error: directory not found: {d}")
            return 1
    docs_a, docs_b = load_dir(a_dir), load_dir(b_dir)
    if not docs_a or not docs_b:
        print("error: need at least one *.gold.json file from each annotator")
        return 1

    rule_ids: Optional[Set[str]] = None
    rules_path = resolve(args.rules)
    if rules_path.is_file():
        import yaml

        rule_ids = {
            r["id"] for r in yaml.safe_load(rules_path.read_text(encoding="utf-8"))["rules"]
        }

    report = build_report(docs_a, docs_b, rule_ids)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotator_agreement.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = to_markdown(report)
    (out_dir / "annotator_agreement.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
