"""Tests for reasoning audit and proof-tree JSON exports."""

import json

from termsheet_ai.reasoning.engine import ForwardChainingEngine, HornRule, WorkingMemory
from termsheet_ai.reasoning.export import (
    export_audit_trace,
    export_proof_tree,
    save_audit_trace,
)
from termsheet_ai.schemas.contracts import SeverityLevel


def make_test_trace():
    """Create a deterministic audit trace containing two risk flags."""
    rules = [
        HornRule(
            rule_id="R-TEST-01",
            family="liquidation_preference",
            condition="multiple >= 3",
            severity=SeverityLevel.CRITICAL,
            message="High liquidation preference",
            source="Test standard",
        ),
        HornRule(
            rule_id="R-TEST-02",
            family="liquidation_preference",
            condition="participation == True",
            severity=SeverityLevel.HIGH,
            message="Participating liquidation preference",
            source="Test standard",
        ),
    ]

    engine = ForwardChainingEngine(rules=rules)
    wm = WorkingMemory()
    wm.assert_fact("multiple", 3.0, clause_id="c_1")
    wm.assert_fact("participation", True, clause_id="c_1")

    return engine.infer(wm, "export-test-document")


def test_audit_trace_is_valid_json():
    trace = make_test_trace()
    exported = json.loads(export_audit_trace(trace))

    assert exported["doc_id"] == "export-test-document"
    assert len(exported["flags"]) == 2
    assert len(exported["rule_traces"]) == 2
    assert exported["risk_score"]["critical_count"] == 1
    assert exported["risk_score"]["high_count"] == 1


def test_export_is_deterministic():
    first = export_audit_trace(make_test_trace())
    second = export_audit_trace(make_test_trace())

    assert first == second


def test_proof_tree_export():
    trace = make_test_trace()
    tree = json.loads(export_proof_tree(trace.flags[0]))

    assert tree["rule_id"] == "R-TEST-01"
    assert tree["conclusion"] == "High liquidation preference"
    assert isinstance(tree["premises"], list)
    assert tree["premises"][0]["fact"] == "multiple"
    assert tree["premises"][0]["value"] == 3.0


def test_save_audit_trace(tmp_path):
    trace = make_test_trace()
    path = tmp_path / "audit_trace.json"

    save_audit_trace(trace, path)

    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["doc_id"] == trace.doc_id
    assert len(saved["flags"]) == 2
