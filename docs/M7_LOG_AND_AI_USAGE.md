# Text to paste into docs/LOG.md and docs/AI_USAGE.md (M7)

Edit anything that is not true for you before you paste it. In particular, only keep a
"Verification" claim if you actually did it.

## docs/LOG.md  (append under the Week 5/6 sections)

#### M7 - Full-Stack UI & Evaluation
- [x] **Streamlit UI** - `src/termsheet_ai/ui/` and `app.py`
  - Demo mode (bundled mock payload) and upload mode (runs `TermSheetPipeline`)
  - Severity-coloured clause cards, original text beside plain-English summary
  - Flags tab with proof-trace tree (chained rules such as R-CT-02 -> R-CT-05 nest)
  - Extracted-facts tab with source quotes; investor tab wired to the EDGAR client by M1
  - 8 tests in `tests/test_ui.py`, including a Streamlit `AppTest` smoke run
- [x] **Report plots** - `scripts/make_plots.py`, run by `make eval`
  - Flag precision/recall/F1, fact accuracy, risk distribution, confusion matrix,
    per-family F1, baseline comparison (each skipped cleanly if its input file is absent)
- [x] **Annotation tooling** - `scripts/annotator_tools.py`, `scripts/compute_kappa.py`
  - Blank, unlabelled templates for independent Annotator B labelling; guide-format validator
  - Cohen's kappa on clause family and on expected flags, attribute exact-match rates,
    and a disagreement list for reconciliation with M2
- [x] Tests for the above - `tests/test_m7_tools.py`

### Remaining (M7)
- [ ] Annotator B labels for the real documents (human work), then reconcile with M2 and report kappa
- [ ] Demo video (script in `docs/DEMO_SCRIPT.md`)

## docs/AI_USAGE.md  (add rows to the table)

| 2026-11-15 | M7 | Claude (Anthropic) | Generate Streamlit UI skeleton: data layer, trace-tree builder, view-models, views, entry point, mock payload | `src/termsheet_ai/ui/*`, `app.py`, `src/termsheet_ai/data/mock_payload.json`, `.streamlit/config.toml` | Reviewed code; mock payload validated against exported JSON schema; `tests/test_ui.py` passes (8 tests); app run locally and every tab checked by hand; ruff clean after fixing one unused import |
| 2026-11-15 | M7 | Claude (Anthropic) | Generate UI tests | `tests/test_ui.py` | Ran with pytest; all pass |
| 2026-11-15 | M7 | Claude (Anthropic) | Generate evaluation plotting script | `scripts/make_plots.py`, `tests/test_m7_tools.py` | Ran tests; inspected generated figures; ruff clean |
| 2026-11-15 | M7 | Claude (Anthropic) | Generate annotation tooling (blank templates, validator) and Cohen's kappa script. The tool never proposes labels; all Annotator B labels are my own independent judgement | `scripts/annotator_tools.py`, `scripts/compute_kappa.py`, `tests/test_m7_tools.py` | Kappa checked against hand-computed values in tests; ruff clean |
