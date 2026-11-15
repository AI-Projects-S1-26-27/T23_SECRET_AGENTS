"""Tests for the M7 tooling: kappa, annotation helpers and report plots."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kappa = _load_script("compute_kappa")
tools = _load_script("annotator_tools")


# --- Cohen's kappa -----------------------------------------------------------------


def test_kappa_known_values():
    assert kappa.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert kappa.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert kappa.cohen_kappa([0, 1], [1, 0]) == pytest.approx(-1.0)


def test_kappa_edge_cases():
    assert kappa.cohen_kappa([], []) is None
    assert kappa.cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    with pytest.raises(ValueError):
        kappa.cohen_kappa([1], [1, 2])


def test_landis_koch_labels():
    assert kappa.landis_koch(-0.1) == "poor"
    assert kappa.landis_koch(0.1) == "slight"
    assert kappa.landis_koch(0.5) == "moderate"
    assert kappa.landis_koch(0.9) == "almost perfect"
    assert kappa.landis_koch(None) == "n/a"


def _doc(*clauses):
    return {"doc_id": "d", "clauses": list(clauses)}


def _clause(cid, family, flags=(), attrs=None, text="text"):
    return {
        "clause_id": cid,
        "clause_family": family,
        "expected_flags": list(flags),
        "attributes": attrs or {},
        "clause_text": text,
    }


def test_report_counts_disagreements_and_unmatched():
    a = {
        "x.gold.json": _doc(
            _clause("c1", "liquidation_preference", ["R-LP-01"], {"liq_multiple": 2.0}),
            _clause("c2", "board_composition", [], {"investor_seats": 3}),
            _clause("c3", "other"),
        ),
        "only_a.gold.json": _doc(_clause("c1", "other")),
    }
    b = {
        "x.gold.json": _doc(
            _clause("c1", "liquidation_preference", ["R-LP-01"], {"liq_multiple": 2.0}),
            _clause("c2", "protective_provisions", ["R-CT-02"], {"investor_seats": 2}),
            _clause("c4", "other"),
        )
    }
    report = kappa.build_report(a, b, rule_ids={"R-LP-01", "R-CT-02"})

    assert report["documents_compared"] == 1
    assert report["clause_family"]["n_clauses"] == 2  # c1 and c2 matched
    assert report["clause_family"]["percent_agreement"] == pytest.approx(0.5)
    assert report["expected_flags"]["n_decisions"] == 4  # 2 clauses x 2 rules
    assert report["attributes"]["liq_multiple"]["exact_match_rate"] == 1.0
    assert report["attributes"]["investor_seats"]["exact_match_rate"] == 0.0
    assert report["unmatched"]["documents_only_a"] == ["only_a.gold.json"]
    assert len(report["unmatched"]["clauses"]) == 2  # c3 only in A, c4 only in B
    assert {d["field"] for d in report["disagreements"]} == {"clause_family", "expected_flags"}
    assert "Cohen's kappa" in kappa.to_markdown(report)


def test_main_writes_reports(tmp_path):
    a_dir, b_dir, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a_dir.mkdir()
    b_dir.mkdir()
    doc = _doc(_clause("c1", "other"))
    (a_dir / "d.gold.json").write_text(json.dumps(doc))
    (b_dir / "d.gold.json").write_text(json.dumps(doc))
    code = kappa.main(
        [
            "--a-dir",
            str(a_dir),
            "--b-dir",
            str(b_dir),
            "--out-dir",
            str(out),
            "--rules",
            "none.yaml",
        ]
    )
    assert code == 0
    assert (out / "annotator_agreement.json").is_file()
    assert (out / "annotator_agreement.md").is_file()


def test_main_fails_cleanly_when_directory_missing(tmp_path):
    assert kappa.main(["--a-dir", str(tmp_path / "nope"), "--b-dir", str(tmp_path)]) == 1


# --- Annotation helpers ------------------------------------------------------------


def test_template_is_blank_and_never_prefilled():
    template = tools.build_template(
        "eval-001",
        "ts_001.docx",
        "B",
        [{"clause_id": "c001", "raw_text": "Investors get 1x.", "section_header": "Liquidation"}],
    )
    clause = template["clauses"][0]
    assert template["annotator"] == "B"
    assert clause["clause_text"] == "Investors get 1x."
    assert clause["clause_family"] is None
    assert clause["expected_flags"] == [] and clause["attributes"] == {}


def test_check_accepts_valid_gold():
    gold = {
        "doc_id": "eval-001",
        "clauses": [
            {
                "clause_id": "c001",
                "clause_family": "liquidation_preference",
                "clause_text": "Series A receives 2x the Original Purchase Price, non-participating.",
                "source_span": "2x the Original Purchase Price",
                "attributes": {"liq_multiple": 2.0, "liq_participation": "non_participating"},
                "expected_flags": ["R-LP-01"],
            }
        ],
    }
    errors, warnings = tools.check_gold(gold, {"R-LP-01"})
    assert errors == [] and warnings == []


def test_check_reports_problems():
    gold = {
        "doc_id": "eval-001",
        "clauses": [
            {"clause_id": "c1", "clause_family": None, "clause_text": "abc"},
            {"clause_id": "c1", "clause_family": "made_up", "clause_text": "abc"},
            {
                "clause_id": "c2",
                "clause_family": "founder_vesting",
                "clause_text": "Vesting over four years with a one-year cliff.",
                "source_span": "Vesting over four years",
                "attributes": {"vesting_months": 48, "favourite_colour": "blue"},
                "expected_flags": ["R-NOPE"],
            },
            {
                "clause_id": "c3",
                "clause_family": "other",
                "clause_text": "abc",
                "source_span": "not in the text",
            },
        ],
    }
    errors, warnings = tools.check_gold(gold, {"R-VS-01"})
    text = "\n".join(errors)
    assert "not filled in" in text
    assert "duplicate clause_id" in text
    assert "unknown clause_family" in text
    assert "unknown rule id 'R-NOPE'" in text
    assert "not an exact substring" in text
    warn_text = "\n".join(warnings)
    assert "vesting_months=48 not found literally" in warn_text
    assert "favourite_colour" in warn_text


def test_numeric_attribute_found_with_commas_and_decimals():
    gold = {
        "doc_id": "d",
        "clauses": [
            {
                "clause_id": "c1",
                "clause_family": "other",
                "clause_text": "Fees up to $50,000 and a 1.0x multiple.",
                "source_span": "up to $50,000 and a 1.0x multiple",
                "attributes": {"amounts": 50000, "conditions": 1.0},
            }
        ],
    }
    assert tools.check_gold(gold) == ([], [])


def test_families_and_attributes_cover_the_ontology():
    from termsheet_ai.schemas.contracts import ClauseFamily

    assert set(tools.CLAUSE_FAMILIES) == {f.value for f in ClauseFamily}
    assert set(tools.FAMILY_ATTRIBUTES) == set(tools.CLAUSE_FAMILIES)


# --- Plots -------------------------------------------------------------------------


def test_make_plots_writes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    plots = _load_script("make_plots")

    results = {
        "summary_metrics": {
            "total_documents_evaluated": 2,
            "risk_flag_precision": 0.8,
            "risk_flag_recall": 0.7,
            "risk_flag_f1": 0.75,
            "fact_extraction_accuracy": {"multiple": 0.9, "participation": 0.8},
        },
        "document_results": [
            {"doc_id": "a", "is_adversarial": False, "predicted_flags": ["x"]},
            {"doc_id": "b", "is_adversarial": True, "predicted_flags": ["x", "y", "z"]},
        ],
    }
    baseline = {
        "total_documents": 2,
        "neuro_symbolic_pipeline": {
            "numerical_hallucination_rate": 0.0,
            "proof_tree_justification_rate": 1.0,
        },
        "llm_only_baseline": {
            "numerical_hallucination_rate": 0.5,
            "proof_tree_justification_rate": 0.0,
        },
    }
    corpus = {
        "confusion_matrix": {
            "other": {"other": 3, "anti_dilution": 1},
            "anti_dilution": {"other": 0, "anti_dilution": 4},
        },
        "family_metrics": {
            "other": {"precision": 1.0, "recall": 0.75, "f1": 0.86},
            "anti_dilution": {"precision": 0.8, "recall": 1.0, "f1": 0.89},
        },
    }
    paths = {}
    for name, data in [("results", results), ("baseline", baseline), ("corpus", corpus)]:
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data))
    out = tmp_path / "figs"

    code = plots.main(
        [
            "--results",
            str(paths["results"]),
            "--baseline",
            str(paths["baseline"]),
            "--corpus-report",
            str(paths["corpus"]),
            "--out-dir",
            str(out),
        ]
    )
    assert code == 0
    assert {p.name for p in out.glob("*.png")} == {
        "fig_flag_prf.png",
        "fig_fact_accuracy.png",
        "fig_risk_distribution.png",
        "fig_confusion_matrix.png",
        "fig_family_f1.png",
        "fig_baseline_comparison.png",
    }


def test_make_plots_skips_missing_inputs_and_fails_when_nothing_made(tmp_path):
    pytest.importorskip("matplotlib")
    plots = _load_script("make_plots")
    code = plots.main(
        [
            "--results",
            str(tmp_path / "missing1.json"),
            "--baseline",
            str(tmp_path / "missing2.json"),
            "--corpus-report",
            str(tmp_path / "missing3.json"),
            "--out-dir",
            str(tmp_path / "figs"),
        ]
    )
    assert code == 1
