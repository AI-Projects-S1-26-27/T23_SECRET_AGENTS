"""Deterministic JSON export utilities for TermSheet AI reasoning."""

import json
from pathlib import Path

from termsheet_ai.schemas.contracts import AuditTrace, RiskFlag


def export_audit_trace(trace: AuditTrace) -> str:
    """Convert an audit trace, including flags and proof trees, into JSON."""
    return json.dumps(
        trace.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def export_proof_tree(flag: RiskFlag) -> str:
    """Convert the proof tree of a risk flag into JSON."""
    return json.dumps(
        flag.justification.proof_tree,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def save_audit_trace(trace: AuditTrace, path: str | Path) -> None:
    """Save an audit trace to a UTF-8 JSON file."""
    destination = Path(path)
    destination.write_text(
        export_audit_trace(trace),
        encoding="utf-8",
    )
