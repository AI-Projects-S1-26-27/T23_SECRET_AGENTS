"""Offline tests for the single-prompt LLM-only comparison baseline."""

import importlib.util
import json
from pathlib import Path

import pytest

from termsheet_ai.baseline import LLMOnlyBaseline
from termsheet_ai.schemas.contracts import ClauseFamily

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_llm_baseline.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_llm_baseline", _SCRIPT)
assert _SPEC and _SPEC.loader
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_baseline_uses_one_direct_prompt_and_normalizes_output():
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "clause_family": "liquidation_preference",
                "risk_flags": ["R-LP-01", "NOT-A-RULE", "R-LP-01"],
                "summary": "The investor receives a 2x preference.",
            }
        )

    prediction = LLMOnlyBaseline(fake_llm).predict("Investor receives a 2x liquidation preference.")

    assert len(prompts) == 1
    assert "retrieval" in prompts[0].lower()
    assert prediction.clause_family == ClauseFamily.LIQUIDATION_PREFERENCE
    assert prediction.risk_flags == ("R-LP-01",)
    assert prediction.summary


def test_baseline_invalid_response_is_visible_as_empty_other_prediction():
    prediction = LLMOnlyBaseline(lambda _: "not JSON").predict("Board clause")

    assert prediction.clause_family == ClauseFamily.OTHER
    assert prediction.risk_flags == ()
    assert prediction.summary == ""
    assert prediction.raw_response == "not JSON"


def test_baseline_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        LLMOnlyBaseline(lambda _: "{}").predict(" ")


def test_evaluation_reports_family_and_multilabel_metrics():
    responses = iter(
        [
            '{"clause_family":"liquidation_preference","risk_flags":["R-LP-01"],"summary":"x"}',
            '{"clause_family":"board_composition","risk_flags":["R-CT-02"],"summary":"y"}',
        ]
    )
    records = [
        {
            "id": "a",
            "text": "x",
            "gold_family": "liquidation_preference",
            "gold_flags": ["R-LP-01"],
        },
        {"id": "b", "text": "y", "gold_family": "anti_dilution", "gold_flags": ["R-AD-01"]},
    ]

    metrics, output = evaluation.evaluate(records, LLMOnlyBaseline(lambda _: next(responses)))

    assert metrics["records"] == 2
    assert metrics["family_accuracy"] == 0.5
    assert metrics["risk_flag_micro"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert metrics["metadata"]["sample_count"] == 2
    assert metrics["metadata"]["evaluated_document_ids"] == ["a", "b"]
    assert metrics["clause_family_classification"]["accuracy"] == 0.5
    assert metrics["risk_flag_detection"]["micro"]["f1"] == 0.5
    assert [row["id"] for row in output] == ["a", "b"]


def test_evaluation_from_saved_predictions_offline():
    """Verify that evaluate() can score pre-saved predictions without making LLM calls."""
    records = [
        {
            "id": "doc1",
            "text": "clause 1",
            "gold_family": "liquidation_preference",
            "gold_flags": ["R-LP-01"],
        },
        {
            "id": "doc2",
            "text": "clause 2",
            "gold_family": "anti_dilution",
            "gold_flags": ["R-AD-01"],
        },
    ]
    saved_preds = [
        {
            "id": "doc1",
            "clause_family": "liquidation_preference",
            "risk_flags": ["R-LP-01"],
            "summary": "2x liq pref",
        },
        {"id": "doc2", "clause_family": "other", "risk_flags": [], "summary": "uncategorized"},
    ]

    metrics, output = evaluation.evaluate(records, predictions=saved_preds)

    assert metrics["metadata"]["evaluation_mode"] == "saved_predictions"
    assert metrics["metadata"]["sample_count"] == 2
    assert metrics["metadata"]["evaluated_document_ids"] == ["doc1", "doc2"]
    assert metrics["clause_family_classification"]["accuracy"] == 0.5
    assert metrics["risk_flag_micro"]["precision"] == 1.0
    assert metrics["risk_flag_micro"]["recall"] == 0.5
    assert len(output) == 2


def test_evaluation_split_safeguards_reject_unassigned_splits():
    """Verify that filter_by_split strictly rejects held-out evaluation on unassigned datasets."""
    records = [
        {"id": "doc1", "text": "clause 1", "split": None},
        {"id": "doc2", "text": "clause 2"},
    ]

    with pytest.raises(ValueError, match="Dataset split error"):
        evaluation.filter_by_split(records, split="test")


def test_baseline_risk_flag_ids_contains_all_25_canonical_rules():
    """Verify that RISK_FLAG_IDS covers all 25 rules defined in rules_v1.yaml."""
    from termsheet_ai.baseline.llm_only import RISK_FLAG_IDS

    assert len(RISK_FLAG_IDS) == 25
    expected_prefixes = {
        "R-LP-",
        "R-AD-",
        "R-CT-",
        "R-VS-",
        "R-PP-",
        "R-DA-",
        "R-OP-",
        "R-RD-",
        "R-EX-",
        "R-DV-",
        "R-RO-",
    }
    for flag_id in RISK_FLAG_IDS:
        assert any(flag_id.startswith(p) for p in expected_prefixes)


def test_evaluation_detailed_metrics():
    """Verify detailed evaluation calculates macro precision, recall, f1, and per-flag metrics."""
    responses = iter(
        [
            '{"clause_family":"liquidation_preference","risk_flags":["R-LP-01"],"summary":"x"}',
            '{"clause_family":"anti_dilution","risk_flags":["R-AD-01"],"summary":"y"}',
        ]
    )
    records = [
        {
            "id": "1",
            "text": "x",
            "gold_flags": ["R-LP-01"],
        },
        {
            "id": "2",
            "text": "y",
            "gold_flags": ["R-AD-01"],
        },
    ]

    metrics, _ = evaluation.evaluate(
        records, LLMOnlyBaseline(lambda _: next(responses)), detailed=True
    )

    assert "risk_flag_macro" in metrics
    macro = metrics["risk_flag_macro"]
    assert macro["precision"] == 1.0
    assert macro["recall"] == 1.0
    assert macro["f1"] == 1.0
    assert "R-LP-01" in macro["per_flag"]
    assert "R-AD-01" in macro["per_flag"]
