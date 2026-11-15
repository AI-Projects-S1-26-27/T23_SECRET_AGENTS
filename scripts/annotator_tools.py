"""Helpers for independent human annotation (Annotator B) of the real term sheets.

The tool NEVER proposes labels: templates contain the segmented clause text only, so the
annotator's judgement stays independent of the pipeline and of Annotator A.

Commands
  template   write blank gold files (one per real document in corpus/manifest.json)
  check      validate filled gold files against docs/annotation_guide_v0.md
  families   print the family list and attribute names from the guide

Examples
  python scripts/annotator_tools.py template --annotator B
  python scripts/annotator_tools.py check corpus/gold/annotator_b
  python scripts/annotator_tools.py families
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Mirrors termsheet_ai.schemas.contracts.ClauseFamily (a test keeps them in sync).
CLAUSE_FAMILIES: Tuple[str, ...] = (
    "liquidation_preference",
    "anti_dilution",
    "board_composition",
    "protective_provisions",
    "founder_vesting",
    "information_rights",
    "rofr_co_sale",
    "drag_along",
    "option_pool",
    "exclusivity_no_shop",
    "conversion_rights",
    "redemption_rights",
    "other",
)

# Attribute names per family, copied from docs/annotation_guide_v0.md.
FAMILY_ATTRIBUTES: Dict[str, Tuple[str, ...]] = {
    "liquidation_preference": (
        "liq_multiple",
        "liq_participation",
        "liq_cap",
        "liq_seniority",
    ),
    "anti_dilution": (
        "anti_dilution_type",
        "broad_based",
        "narrow_based",
        "adjustment_trigger",
        "excluded_issuances",
    ),
    "board_composition": (
        "total_seats",
        "investor_seats",
        "founder_seats",
        "independent_seats",
        "appointment_rights",
    ),
    "protective_provisions": (
        "consent_threshold",
        "requires_series_majority",
        "veto_day_to_day_operations",
        "protected_actions",
    ),
    "founder_vesting": (
        "vesting_months",
        "cliff_months",
        "acceleration",
        "repurchase_right",
        "applies_to",
    ),
    "information_rights": (
        "report_types",
        "frequency",
        "inspection_right",
        "eligibility_threshold",
    ),
    "rofr_co_sale": (
        "investor_rofr",
        "investor_co_sale",
        "covered_holders",
        "notice_period_days",
        "transfer_exceptions",
    ),
    "drag_along": (
        "approval_threshold",
        "requires_founder_consent",
        "requires_preferred_consent",
        "sale_scope",
    ),
    "option_pool": (
        "pool_percentage",
        "pre_money_pool",
        "post_money_pool",
        "dilution_holder",
        "pool_purpose",
    ),
    "exclusivity_no_shop": (
        "duration_days",
        "restricted_activities",
        "starts_on",
        "termination_event",
        "exceptions",
    ),
    "conversion_rights": (
        "conversion_ratio",
        "optional_conversion",
        "automatic_conversion",
        "conversion_triggers",
        "majority_threshold",
    ),
    "redemption_rights": (
        "redemption_years",
        "redemption_price",
        "installments",
        "approval_threshold",
        "redemption_trigger",
    ),
    "other": ("topic", "parties", "effective_date", "amounts", "conditions"),
}


def load_rule_ids(rules_path: Path) -> Set[str]:
    import yaml

    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    return {r["id"] for r in data.get("rules", [])}


def build_template(
    doc_id: str, filename: str, annotator: str, clauses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Blank gold structure. ``clauses`` items need clause_id, raw_text, section_header."""
    return {
        "doc_id": doc_id,
        "filename": filename,
        "annotator": annotator,
        "clauses": [
            {
                "clause_id": c["clause_id"],
                "section_header": c.get("section_header"),
                "clause_text": c["raw_text"],
                "clause_family": None,
                "source_span": None,
                "attributes": {},
                "expected_flags": [],
                "notes": "",
            }
            for c in clauses
        ],
    }


def _number_forms(value: float) -> List[str]:
    forms = {str(value)}
    if float(value).is_integer():
        forms.update({str(int(value)), f"{int(value)}.0"})
    return sorted(forms)


def check_gold(
    gold: Dict[str, Any], rule_ids: Optional[Set[str]] = None
) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings) for one gold file."""
    errors: List[str] = []
    warnings: List[str] = []
    if not gold.get("doc_id"):
        errors.append("missing doc_id")
    clauses = gold.get("clauses")
    if not isinstance(clauses, list):
        return errors + ["'clauses' must be a list"], warnings
    seen: Set[str] = set()
    for clause in clauses:
        cid = clause.get("clause_id")
        where = f"clause {cid or '<no id>'}"
        if not cid:
            errors.append("a clause has no clause_id")
            continue
        if cid in seen:
            errors.append(f"{where}: duplicate clause_id")
        seen.add(cid)

        family = clause.get("clause_family")
        if family is None:
            errors.append(f"{where}: clause_family not filled in (use 'other' if nothing fits)")
        elif family not in CLAUSE_FAMILIES:
            errors.append(f"{where}: unknown clause_family '{family}'")

        text = clause.get("clause_text", clause.get("raw_text", ""))
        span = clause.get("source_span")
        if span is not None:
            if not isinstance(span, str) or not span.strip():
                errors.append(f"{where}: source_span must be a non-empty string or null")
            elif text and span not in text:
                errors.append(f"{where}: source_span is not an exact substring of the clause text")

        attrs = clause.get("attributes") or {}
        allowed = FAMILY_ATTRIBUTES.get(family or "", ())
        for key, value in attrs.items():
            if allowed and key not in allowed:
                warnings.append(f"{where}: attribute '{key}' is not listed for family '{family}'")
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if is_number and isinstance(span, str):
                haystack = span.replace(",", "")
                if not any(f in haystack for f in _number_forms(value)):
                    warnings.append(
                        f"{where}: numeric attribute {key}={value} not found literally in source_span"
                        " (the guide requires it; use null if the text does not state it)"
                    )
        if attrs and span is None:
            warnings.append(f"{where}: attributes given but source_span is null")

        flags = clause.get("expected_flags", [])
        if not isinstance(flags, list):
            errors.append(f"{where}: expected_flags must be a list")
        elif rule_ids is not None:
            for flag in flags:
                if flag not in rule_ids:
                    errors.append(f"{where}: unknown rule id '{flag}' (see docs/rules_v1.yaml)")
    return errors, warnings


def _real_documents(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = manifest["evaluation_corpus"]["documents"]
    return [d for d in docs if not d.get("is_synthetic") and d.get("filename")]


def cmd_template(args: argparse.Namespace) -> int:
    from termsheet_ai.ingest.ingestor import Ingestor  # lazy: needs the parsing extras

    corpus_dir = REPO_ROOT / args.corpus
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / (args.out_dir or f"corpus/gold/annotator_{args.annotator.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for entry in _real_documents(manifest):
        doc_path = corpus_dir / entry["filename"]
        out_path = out_dir / f"{Path(entry['filename']).name}.gold.json"
        if not doc_path.is_file():
            print(f"skip  {entry['id']}: file not found: {doc_path}")
            skipped += 1
            continue
        if out_path.exists() and not args.force:
            print(
                f"keep  {out_path.name} already exists (use --force to overwrite blank work only)"
            )
            skipped += 1
            continue
        _, clauses = Ingestor().ingest(doc_path)
        rows = [
            {"clause_id": c.clause_id, "raw_text": c.raw_text, "section_header": c.section_header}
            for c in clauses
        ]
        template = build_template(entry["id"], entry["filename"], args.annotator.upper(), rows)
        out_path.write_text(
            json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} clauses)")
        written += 1
    print(f"done: {written} written, {skipped} skipped")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    rules_path = REPO_ROOT / args.rules
    rule_ids = load_rule_ids(rules_path) if rules_path.is_file() else None
    if rule_ids is None:
        print(f"note: {rules_path} not found, rule IDs not checked")
    target = Path(args.path)
    files = sorted(target.glob("*.gold.json")) if target.is_dir() else [target]
    if not files:
        print(f"no *.gold.json files in {target}")
        return 1
    bad = 0
    for f in files:
        errors, warnings = check_gold(json.loads(f.read_text(encoding="utf-8")), rule_ids)
        status = "OK" if not errors else "ERRORS"
        print(f"{status:6} {f.name}: {len(errors)} errors, {len(warnings)} warnings")
        for line in errors:
            print(f"    error:   {line}")
        for line in warnings:
            print(f"    warning: {line}")
        bad += bool(errors)
    return 1 if bad else 0


def cmd_families(_: argparse.Namespace) -> int:
    for family in CLAUSE_FAMILIES:
        print(f"{family}\n    " + ", ".join(FAMILY_ATTRIBUTES[family]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="write blank gold files for the real documents")
    t.add_argument("--annotator", default="B")
    t.add_argument("--corpus", default="corpus")
    t.add_argument("--out-dir", default=None)
    t.add_argument("--force", action="store_true")
    t.set_defaults(fn=cmd_template)

    c = sub.add_parser("check", help="validate filled gold file(s)")
    c.add_argument("path", help="a .gold.json file or a directory of them")
    c.add_argument("--rules", default="docs/rules_v1.yaml")
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("families", help="list clause families and their attributes")
    f.set_defaults(fn=cmd_families)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
