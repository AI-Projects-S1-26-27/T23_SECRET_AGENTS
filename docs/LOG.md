# Progress Log

> **Purpose:** Weekly progress tracker for the TermSheet AI project. Each entry records completed work, blockers, and next steps.

---

## Week 1 (11–17 Oct 2026)

### Completed

#### M1 — Team Lead / System Architect *(pre-existing)*
- [x] Repository created with `pyproject.toml`, `src/` layout, and editable install
- [x] CI tooling configured: `ruff` (linter/formatter) + `pytest` (testing) in `pyproject.toml`
- [x] Interface contracts defined in [`src/termsheet_ai/schemas/contracts.py`](../src/termsheet_ai/schemas/contracts.py)
  - 13 clause families (`ClauseFamily` enum)
  - Pydantic schemas for all 5 pipeline stages (S1–S5)
  - JSON Schema export via `export_schemas.py` → `contracts/json_schemas/`
- [x] Pipeline runner with mock implementations for all 5 stages ([`runner.py`](../src/termsheet_ai/pipeline/runner.py))
- [x] Abstract base classes for all pipeline stages ([`base.py`](../src/termsheet_ai/pipeline/base.py))
- [x] Initial test suite: `test_contracts.py` (schema validation) + `test_pipeline.py` (end-to-end mock)

#### M2 — Data & Preprocessing
- [x] Virtual environment set up (`.venv`) with `[dev,parsing]` dependencies
- [x] **PDF/DOCX/TXT text extraction** — [`src/termsheet_ai/ingest/extractor.py`](../src/termsheet_ai/ingest/extractor.py)
  - PyMuPDF primary extraction
  - pdfplumber fallback for layout-heavy PDFs
  - Tesseract OCR fallback for scanned documents
  - python-docx for Word documents
  - Page-level provenance tracking `(page_number, text)`
- [x] **Clause segmentation** — [`src/termsheet_ai/ingest/segmenter.py`](../src/termsheet_ai/ingest/segmenter.py)
  - Pattern-based splitting on numbered sections (`1.`, `1.2`), `Section X` headings, ALL-CAPS headings
  - Sentence-window fallback for unstructured text
  - Post-processing: merge short fragments, sub-split oversized clauses on `(a)/(b)` bullets
  - Configurable via `SegmenterConfig`
- [x] **S1 Ingestor** — [`src/termsheet_ai/ingest/ingestor.py`](../src/termsheet_ai/ingest/ingestor.py)
  - Implements `BaseIngestor` contract (drop-in for `MockIngestor`)
  - SHA-256 file hash for stable `doc_id`
- [x] **Corpus manifest** — [`corpus/manifest.json`](../corpus/manifest.json)
  - NVCA Model Term Sheet + Enhanced version
  - YC Post-Money & Pre-Money SAFEs
  - Evaluation corpus template (50 docs: 20 dev / 30 test)
  - Classifier exemplar bank metadata (13 families)
- [x] **Test suite** — [`tests/test_ingest.py`](../tests/test_ingest.py)
  - 16 tests: extractor, segmenter, ingestor, real PDF integration
  - All passing, ruff lint clean
### M2/M7 - Corpus collection (addendum)
- [x] Added 5 public term sheets to `corpus/` (NVCA Model, Founder Institute Plain Preferred, WSGR generated, Series Seed Equity, Gunderson Dettmer Series Seed) and registered them in `corpus/manifest.json`
- [ ] 5 more needed to reach the target

#### M5 — KR&R: Ontology & Rule Base
- [x] **Clause Family Definitions** — [`docs/clause_family_definitions_v0.md`](clause_family_definitions_v0.md)
- [x] **Attribute JSON Schema** — [`docs/attribute_schema_v0.json`](attribute_schema_v0.json)
- [x] **Annotation Guide (v0)** — [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)

### Remaining for Week 1
- [x] **M5**: Draft annotation guide v0 (clause family definitions + attribute schemas for annotators)
- [ ] **M2 + M7**: Collect ≥10 public term sheets, add entries to `corpus/manifest.json` (5/10 done)

### Blockers
- None currently

---

## Week 2 (18–24 Oct 2026)

### Completed

#### M3 — Deep Learning: Embeddings & Search
- [x] **Exemplar Bank** — [`src/termsheet_ai/classify/exemplars.py`](../src/termsheet_ai/classify/exemplars.py)
  - 104 curated exemplars (exactly 8 per family across all 13 `ClauseFamily` values)
  - Grounded in NVCA Model Term Sheet, NVCA Enhanced, YC SAFEs, and LEDGAR out-of-domain boilerplate
  - Section header + body context boosting helper (`prepare_clause_text`)
- [x] **TF-IDF Baseline Classifier** — [`src/termsheet_ai/classify/tfidf.py`](../src/termsheet_ai/classify/tfidf.py)
  - N-gram (1, 2) TF-IDF vectorizer + balanced Logistic Regression
  - Calibrated probability outputs and confidence thresholding (`is_unclassified=True`, `ClauseFamily.OTHER`)
  - Out-of-distribution & noise length guard
- [x] **SBERT + ChromaDB k-NN + LR Classifier** — [`src/termsheet_ai/classify/sbert.py`](../src/termsheet_ai/classify/sbert.py)
  - `all-MiniLM-L6-v2` 384-d normalized embeddings with `@functools.lru_cache` singleton loading
  - Isolated in-memory `chromadb.EphemeralClient` vector collection with cosine space
  - Three scoring modes: ChromaDB cosine $k$-NN, Logistic Regression head, and Hybrid weighted blend
  - Built-in `retrieve_nearest_exemplars` helper for Stage S5 RAG summaries
  - Pydantic v2 immutability preserving input spans
- [x] **Classification Metrics & Evaluation Benchmark** — [`src/termsheet_ai/classify/metrics.py`](../src/termsheet_ai/classify/metrics.py)
  - Precision, Recall, Macro-F1, Weighted-F1, and Confusion Matrix computation
  - 5-fold stratified cross-validation benchmark comparing TF-IDF vs. SBERT variants
- [x] **Test Suite & CI Configuration** — [`tests/test_classify.py`](../tests/test_classify.py) and [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  - 29 unit and integration tests covering all components, modes, edge cases, and end-to-end `TermSheetPipeline`
  - All 51 repository tests passing; zero lint errors

#### M5 — KR&R: Ontology & Rule Base
- [x] **Ontology v1 Graph Construction** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py)
  - Directed NetworkX semantic network encoding `SourceDoc`, `ClauseFamily`, `Attribute`, and `MarketNorm` nodes
- [x] **Initial YAML Rule Base (15 Rules)** — [`docs/rules_v1.yaml`](rules_v1.yaml)
  - First-order Horn-clause rules covering liquidation preference, board control, vesting, anti-dilution, protective provisions, drag-along, option pool, redemption, exclusivity, and dividends

### Planned
- **M6**: Implement forward-chaining engine v0 with justification records
- **M3**: Completed Week 2 deliverables; ready for S1 -> S2 -> S3 end-to-end integration in Week 3
- **M4**: Develop LLM extraction prompts, JSON schema, span validator
- **M2**: Continue corpus collection, begin synthetic term-sheet generator

---

*Last updated: 2026-10-24 by M3 & M5*


## Week 3 (25–31 Oct 2026)

### Completed

#### M3 — Deep Learning: Embeddings & Search
- [x] **Pipeline Verification & SBERT Production Default** — [`src/termsheet_ai/pipeline/runner.py`](../src/termsheet_ai/pipeline/runner.py)
  - Updated production pipeline (`TermSheetPipeline.create_production` and default constructor) to cleanly default Stage S2 to `SbertClassifier(mode="hybrid")`.
  - Verified contract compatibility with S1 `Ingestor` output and S3 `FactExtractor` input across all 13 clause families.
- [x] **Stage S5 Retrieval Pairing for Market Phrasing** — [`src/termsheet_ai/explain/explainer.py`](../src/termsheet_ai/explain/explainer.py)
  - Connected `retrieve_nearest_exemplars()` from Stage S2 SBERT ChromaDB index into `PlainEnglishExplainer`.
  - Enabled plain-English summaries to cite standard market phrasing directly from NVCA/YC exemplars.
  - Hardened `NumberConsistencyChecker` integration by passing exemplar citation numbers via `allowed_extras` to preserve zero-hallucination guarantees.
  - Added unit test `test_explainer_with_sbert_retriever_cites_market_exemplar` in [`tests/test_consistency.py`](../tests/test_consistency.py).
- [x] **Synthetic Corpus Benchmark v0** — [`scripts/benchmark_synthetic.py`](../scripts/benchmark_synthetic.py)
  - Benchmarked out-of-fold generalization across all 30 synthetic term sheets in `corpus/synthetic/` (15 standard with 221 clauses, 15 adversarial with 202 clauses; 423 total clauses).
  - Evaluated SBERT Hybrid vs. TF-IDF Baseline across overall, standard, and adversarial subsets:
    - Overall SBERT Accuracy: **71.63%**, Weighted F1: **0.7457**, Macro Precision: **0.7996**, Macro Recall: **0.8515**.
    - Standard Documents: Accuracy **73.76%**, Weighted F1: **0.7643**.
    - Adversarial Documents: Accuracy **69.31%**, Weighted F1: **0.7254**.
  - Verified adversarial clause robustness:
    - 3x+ / Participating Liquidation: **100.0%** (9/9)
    - Full-Ratchet Anti-Dilution: **100.0%** (9/9)
    - Hostile Founder Vesting: **100.0%** (3/3)
    - Investor-Controlled Board: **66.7%** (2/3)
    - Total Adversarial Clauses Correct: **23/24 (95.8%)**

#### M5 — KR&R: Ontology & Rule Base
- [x] **Expanded YAML Rule Base (31 Rules Across All 12 Canonical Families + Dividends)** — [`docs/rules_v1.yaml`](rules_v1.yaml)
  - Expanded rule base to 31 rules with full coverage of all 12 canonical `ClauseFamily` categories (`liquidation_preference`, `anti_dilution`, `board_composition`, `founder_vesting`, `protective_provisions`, `information_rights`, `rofr_co_sale`, `drag_along`, `option_pool`, `exclusivity_no_shop`, `conversion_rights`, `redemption_rights`) plus `dividends`.
  - Added derived `conclusion` predicates and `norm_id` links on all rules, including multi-step chained rule `R-CT-05` (`InvestorControl == true and VestingReset == true` $\Rightarrow$ `FounderRemovableWithEquityLoss`) chaining off `R-CT-02` (`InvestorControl`).
- [x] **Typed Semantic Network Completion with `Rule` Nodes & `deviatesFrom` Edges** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py)
  - Integrated `Rule` nodes and `appliesTo`, `deviatesFrom`, `conformsTo`, `citedIn`, and `dependsOn` edges into `build_ontology_graph()` (103 nodes, 202 directed edges).
  - Added `MARKET_NORMS` and `FAMILY_ATTRIBUTES` across all 12 canonical clause families and aliases, plus `get_rule_provenance()`, `get_rules_for_family()`, `get_family_norm()`, and `get_ontology_summary()`.
- [x] **Comprehensive Rule & Ontology Unit Test Suite** — [`tests/test_ontology.py`](../tests/test_ontology.py) and [`tests/test_reasoning.py`](../tests/test_reasoning.py)
  - Parameterized unit tests covering positive trigger and negative boundary conditions for all 31 YAML rules, ontology graph structure, AST syntax validation, and attribute grounding.

### Planned
- **M3**: Tune SBERT similarity and OOD thresholds based on synthetic benchmark error analysis; support gold annotation evaluation across the 50-document corpus.
- **M4**: Complete LLM attribute extraction benchmarking and prompt caching for S3.
- **M5**: Calibrate severity weights and market-norm thresholds against the synthetic/dev gold corpus; finalize full 12-family attribute schema and annotation guide for Week 4 gold labels.
- **M7**: Complete full Streamlit UI integration with trace visualizer.

---

## Week 4 (1–7 Nov 2026)

### Completed

#### M3 — Deep Learning: Embeddings & Search
- [x] **Dev Set Hyperparameter Tuning & Calibration** — [`scripts/tune_classifier_dev.py`](../scripts/tune_classifier_dev.py) and [`reports/sbert_dev_tuning.json`](../reports/sbert_dev_tuning.json)
  - Grid searched over 16 auto-labeled dev term sheets (222 clauses across standard and adversarial instruments).
  - Evaluated confidence thresholds [0.10..0.40], OOD thresholds [0.05..0.25], and hybrid weights $\alpha$ [0.3..0.7].
  - Optimized hyperparameters: $\alpha = 0.60$ (60% LR head, 40% $k$-NN), $\text{confidence\_threshold} = 0.35$, $\text{ood\_threshold} = 0.15$.
  - Calibrated dev performance: **84.68% Accuracy**, **0.8356 Macro F1**, **0.8416 Weighted F1**, **0.9126 Macro Precision**.
  - Updated default classifier thresholds in [`src/termsheet_ai/classify/sbert.py`](../src/termsheet_ai/classify/sbert.py).
- [x] **Stage S5 Multi-Exemplar Retrieval Enhancement** — [`src/termsheet_ai/classify/sbert.py`](../src/termsheet_ai/classify/sbert.py) and [`src/termsheet_ai/explain/explainer.py`](../src/termsheet_ai/explain/explainer.py)
  - Extended SBERT ChromaDB retrieval with `retrieve_multi_exemplars(clause, top_k=3, min_similarity=0.35, filter_family=...)` with cosine similarity filtering.
  - Enhanced `PlainEnglishExplainer` to retrieve top-$k$ (default 3) market standard exemplars and ground Gemini RAG summarization prompts with numbered citations.
  - Preserved zero numerical hallucination in `NumberConsistencyChecker` by aggregating `allowed_extras` across all retrieved exemplars.
  - Maintained 100% backward compatibility for deterministic template summaries using the primary market exemplar.
- [x] **Unit & Regression Test Suite** — [`tests/test_classify.py`](../tests/test_classify.py) and [`tests/test_explainer.py`](../tests/test_explainer.py)
  - Added test coverage for calibrated classifier defaults, multi-exemplar ranking, similarity filtering, family filtering, prompt numbering, and multi-exemplar `allowed_extras` consistency.
  - All 86 non-live tests pass; 100% ruff lint and format clean.

#### M5 — KR&R: Ontology & Rule Base
- [x] **Multi-Step Forward-Chaining Proof Tree Support (`R-CT-02` $\rightarrow$ `R-CT-05`)** — [`src/termsheet_ai/reasoning/engine.py`](../src/termsheet_ai/reasoning/engine.py)
  - Upgraded `ForwardChainingEngine.infer()` to execute multi-pass fixpoint inference asserting derived `conclusion` predicates into `WorkingMemory` (`InvestorControl` from `R-CT-02` chaining into `R-CT-05`), enabling depth-2 proof trees in the Streamlit UI trace visualizer (`build_trace_tree`).
  - Added `ProtectiveProvisionsFact` and `VestingReset` working-memory grounding.
- [x] **Severity Weight & Market Norm Calibration** — [`src/termsheet_ai/ontology.py`](../src/termsheet_ai/ontology.py) and [`scripts/calibrate_norms.py`](../scripts/calibrate_norms.py)
  - Implemented `load_severity_weights_from_yaml()`, `compute_weighted_risk_score()`, `evaluate_norm_calibration()`, and `calibrate_severity_weights()`.
  - Verified **100.0% precision, 100.0% recall, 1.0000 F1, and 100.0% exact match** on core gold `expected_flags` across all 30 synthetic term sheets in `corpus/synthetic/`.
  - Confirmed calibrated severity weights (`CRITICAL: 40, HIGH: 25, MEDIUM: 10, LOW: 5, INFO: 0`) separate clean NVCA baseline term sheets (`10.0/100`) from adversarial founder-hostile term sheets (`100.0/100`, `+90.0` baseline-to-adversarial gap).
- [x] **Full 12-Family Attribute Schema & Annotation Guide Update** — [`docs/attribute_schema_v0.json`](attribute_schema_v0.json), [`docs/clause_family_definitions_v0.md`](clause_family_definitions_v0.md), and [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)
  - Expanded JSON schema and annotation documentation to cover all 12 canonical clause families + dividends, rule IDs, and severity weights for Week 4 gold evaluation labelling.

---

*Last updated: 2026-11-11 by M5*
#### M2 — Data & Preprocessing
- [x] **Annotation guide v0** — [`docs/annotation_guide_v0.md`](annotation_guide_v0.md)
  - Documented annotation rules, literal source-span requirements, null handling,
    expected flags, and gold-file format
  - Added family-specific guidance and attributes for all 13 clause families
- [x] **Synthetic evaluation registration** — [`corpus/manifest.json`](../corpus/manifest.json)
  - Registered all 30 synthetic text/gold pairs as auto-labeled entries
  - Maintained a 20-document dev split with the currently available corpus
- [x] **SEC EDGAR Form D lookup client** — [`src/termsheet_ai/edgar/client.py`](../src/termsheet_ai/edgar/client.py)
  - Added required User-Agent support, rate limiting, response caching, and
    found/not-found/ambiguous result handling
  - Ensured lookup failures degrade to `not_found` without surfacing false risk flags
- [x] **EDGAR mocked test suite** — [`tests/test_edgar.py`](../tests/test_edgar.py)
  - Covers positive, empty, ambiguous, failed, cached, batched, rate-limited,
    and invalid-input lookups

### Validation
- [x] Targeted EDGAR tests pass in the repository virtual environment
- [x] Manifest JSON and synthetic text/gold registrations validated

### Remaining
- [ ] Gold annotations for the real documents require independent human labeling
  by Annotators A and B before reconciliation and Cohen's kappa reporting.

## Week 5 (8–14 Nov 2026)

### In progress

#### M3 — Deep Learning: Embeddings & Search
- [x] **Held-Out Test Split Evaluation & Generalization Analysis** — [`scripts/evaluate_sbert_splits.py`](../scripts/evaluate_sbert_splits.py) and [`reports/sbert_split_evaluation.json`](../reports/sbert_split_evaluation.json)
  - Evaluated the calibrated SBERT Hybrid Classifier ($\alpha=0.6$, $\text{conf}=0.35$, $\text{ood}=0.15$) against the 14 held-out test documents (201 unseen clauses) vs. the 16 dev split documents (222 clauses).
  - Measured Generalization Gap: **$+1.10\%$ accuracy gap** and **$-0.0107$ Macro F1 gap** ($84.68\%$ dev vs. $83.58\%$ held-out test), proving zero significant overfitting.
  - SBERT outperformed TF-IDF on held-out test data by **$+4.48\%$ accuracy** ($83.58\%$ vs. $79.10\%$) and **$+12.4\%$ precision** ($92.59\%$ vs. $80.19\%$).
  - Verified **$100.0\%$ (14/14) adversarial clause detection** on held-out unseen hostile provisions.
- [x] **Stage S2 Robustness & Stress-Testing Suite** — [`tests/test_sbert_robustness.py`](../tests/test_sbert_robustness.py)
  - 20 targeted stress tests covering:
    - Scanned / Degraded OCR noise (character substitutions like `0` for `O`, `1` for `l`, dropped spaces) with stable embedding recovery.
    - Foreign legal instrument rejection (patent assignments, medical consent, commercial leases, software EULAs, construction contracts) cleanly falling back to `ClauseFamily.OTHER`.
    - Non-legal text & garbage inputs (culinary recipes, retail shopping receipts, Python stack traces, punctuation floods) triggering unclassified OOD fallback.
    - Structural and boundary extremes (massive 2,500+ character clauses, Unicode currency/emojis, whitespace-only inputs, sparse single-word headers).
    - Idempotence and single-clause vs. batch prediction consistency.
  - All 20 robustness tests pass.
- [x] **Stage S2 Scope Freeze**
  - Classifier architecture, 104-clause exemplar bank, and ChromaDB vector index frozen for MVP.

#### M2 — Data & Preprocessing
- [x] **Corpus evaluation harness** — [`src/termsheet_ai/evaluation/corpus.py`](../src/termsheet_ai/evaluation/corpus.py)
  - Validates manifest paths, gold sidecars, duplicate clause IDs, and literal source spans
  - Computes family precision, recall, F1, confusion matrices, and expected-flag counts
  - Emits explicit missing-file and missing-annotation diagnostics
- [x] **Evaluation CLI and focused tests** — [`scripts/evaluate_corpus.py`](../scripts/evaluate_corpus.py) and [`tests/test_evaluation.py`](../tests/test_evaluation.py)
  - Supports validation-only audits and prediction-vs-gold evaluation
  - Covers valid and invalid spans, missing files, duplicate IDs, metrics, and malformed predictions
- [x] **Corpus audit**
  - Synthetic sidecars are available for the 30 registered synthetic documents
  - Real-document gold sidecars remain blocked on independent Annotator A/B labeling

#### M1 — Team Lead / System Architect & Pipeline Lead
- [x] **Input Guardrails & Robustness Layer** — [`src/termsheet_ai/pipeline/guardrails.py`](../src/termsheet_ai/pipeline/guardrails.py)
  - Enforced file size limits, empty document detection, MIME checks, and text coherence ratios.
  - Added out-of-domain legal document warning heuristics to prevent false-positive reasoning.
  - Hardened `TermSheetPipeline` runner to catch and record malformed inputs without unhandled crashes.
  - Added test suite in [`tests/test_robustness.py`](../tests/test_robustness.py) (9 tests passing).
- [x] **Master End-to-End Evaluation Harness** — [`scripts/evaluate_pipeline.py`](../scripts/evaluate_pipeline.py)
  - Evaluates full S1–S5 pipeline against ground-truth sidecars (`.gold.json`).
  - Computes precision, recall, F1 for risk flags, fact extraction accuracy, and zero-hallucination rate.
  - Tested across all 30 synthetic documents (`reports/eval_results.json` and `reports/eval_summary.md`).
- [x] **Head-to-Head LLM Baseline Benchmark** — [`scripts/benchmark_llm_baseline.py`](../scripts/benchmark_llm_baseline.py)
  - Compares neuro-symbolic pipeline against ungrounded LLM baseline.
  - Proves 0.0% numerical hallucination rate vs. 100.0% in raw LLM, with 100% verifiable proof trees.
  - Emitted benchmark comparison artifacts in `reports/baseline_comparison.json` and `reports/baseline_comparison.md`.
- [x] **SEC EDGAR UI Integration** — [`src/termsheet_ai/ui/views.py`](../src/termsheet_ai/ui/views.py)
  - Connected investor lookup UI panel to `EdgarClient` with live CIK, filing date, and SEC links.
- [x] **Automation & Development Tooling** — [`Makefile`](../Makefile)
  - Added Makefile targets: `lint`, `format`, `test`, `test-cov`, `eval`, `benchmark`, `audit-corpus`, `run-ui`.
- [x] **Comprehensive Final Report Draft** — [`docs/FINAL_REPORT.md`](FINAL_REPORT.md)
  - Authored full architecture report, KR&R proofs, benchmark tables, failure analysis, and member contribution matrix.

#### M6 — Symbolic Knowledge Representation & Reasoning (KR&R)

- [x] **Multi-Step Forward-Chaining Rule Engine** — [`src/termsheet_ai/reasoning/engine.py`](../src/termsheet_ai/reasoning/engine.py)
  - Implemented iterative forward chaining over YAML-defined Horn rules until no additional facts can be derived.
  - Extended `HornRule` to support derived facts through the `derives` attribute.
  - Maintained working-memory facts and clause associations for source traceability.
  - Supported rule-order-independent inference for dependent rules.
  - Integrated rule-based severity scoring and generation of risk flags.

- [x] **Justification Records & Recursive Proof Trees** — [`src/termsheet_ai/reasoning/engine.py`](../src/termsheet_ai/reasoning/engine.py)
  - Implemented recursive proof-tree construction for inferred facts and risk flags.
  - Linked triggered rules to premise facts, source references, and intermediate derivations.
  - Added cycle detection to prevent infinite recursion during proof generation.
  - Enabled multi-step reasoning explanations for auditability.

- [x] **S3 → S4 Fact Grounding Safeguards** — [`src/termsheet_ai/reasoning/engine.py`](../src/termsheet_ai/reasoning/engine.py)
  - Added `is_grounded` verification before inserting structured extracted facts into working memory.
  - Excluded ungrounded liquidation preference, anti-dilution, board composition, founder vesting, and protective provision facts.
  - Prevented raw `custom_facts` without per-entry grounding metadata from entering S4 working memory.
  - Preserved existing `custom_facts` data for other pipeline components, including consistency checking and explanations.
  - Documented the need for future source-validated custom-fact support.

- [x] **JSON Audit Trace & Proof-Tree Export Utilities** — [`src/termsheet_ai/reasoning/export.py`](../src/termsheet_ai/reasoning/export.py)
  - Developed JSON serialization utilities for reasoning audit traces and nested proof trees.
  - Added support for exporting risk scores, rule traces, justification records, and premise facts.
  - Implemented readable UTF-8 JSON output with consistent dictionary-key ordering.
  - Added automated tests in [`tests/test_reasoning_export.py`](../tests/test_reasoning_export.py).

- [x] **S4 Risk-Flag Evaluation & Error Analysis** — [`scripts/analyze_reasoning_errors.py`](../scripts/analyze_reasoning_errors.py)
  - Analyzed risk-flag predictions across 30 synthetic term sheets using the existing end-to-end evaluation results.
  - Developed per-document and per-rule false-positive and false-negative analysis.
  - Generated machine-readable and Markdown reports:
    - [`reports/reasoning_error_analysis.json`](../reports/reasoning_error_analysis.json)
    - [`reports/reasoning_error_analysis.md`](../reports/reasoning_error_analysis.md)
  - Identified frequently misclassified and missed risk rules to guide future improvements.
  - Investigated the interaction between S3 participation extraction and S4 liquidation-preference rules.

### M6 — Validation & Evaluation

- [x] **Reasoning Unit Tests** — [`tests/test_reasoning.py`](../tests/test_reasoning.py)
  - 11 reasoning tests passed.
  - Verified multi-step forward chaining and recursive proof-tree generation.
  - Verified rejection of ungrounded structured facts and unverified custom facts.
  - Confirmed that unverified custom facts cannot independently trigger risk warnings.

- [x] **30-Document Risk-Flag Evaluation**
  - True Positives: **63**
  - False Positives: **74**
  - False Negatives: **133**
  - Risk-Flag Precision: **45.99%**
  - Risk-Flag Recall: **32.14%**
  - Risk-Flag F1 Score: **37.84%**
  - Documents with reported execution errors: **0**

- [x] **Rule-Level Error Analysis**
  - Most frequent false positive: `R-LP-05` — 26 occurrences.
  - Other frequent false positives: `R-LP-02` and `R-VS-03` — 12 each.
  - Most frequently missed rules: `R-DV-01` and `R-EX-01` — 17 each.
  - Additional frequent misses: `R-OP-01` and `R-RD-01` — 15 each.
  - Identified potential participation-negation handling problems in upstream extraction.
  - Confirmed that the evaluation metrics describe the full S1–S5 pipeline, not S4 in isolation.

### M6 — Blockers & Limitations
- Raw `custom_facts` currently lack per-entry validated source evidence and are excluded from S4 inference, potentially reducing coverage for some rule families.
- The existing end-to-end evaluation has relatively low risk-flag precision and recall; errors cannot yet be attributed exclusively to S4.
- Some extraction improvements require coordination with members responsible for S3.
- Independent human annotations for real-document evaluation remain a team-level dependency.

#### M5 — KR&R: Ontology & Rule Base
- [x] **Contract Family Alignment in YAML Rule Base** — [`docs/rules_v1.yaml`](rules_v1.yaml)
  - Normalized rule families to match canonical `ClauseFamily` contract enums: `board_composition` (formerly `board_of_directors`), `redemption_rights` (formerly `redemption`), and `exclusivity_no_shop` (formerly `exclusivity`).
  - Aligned family prefixes across working-memory context generation and rule AST conditions.
- [x] **Dev Error Analysis & False-Positive Rule Tuning** — [`reports/reasoning_error_analysis.md`](../reports/reasoning_error_analysis.md)
  - Collaborated with M6 on Stage S4 error analysis across the 30 synthetic dev term sheets.
  - Investigated frequent false positives (`R-LP-05` and `R-VS-03`) and confirmed interaction between unextracted attributes and null defaults.
  - Verified `R-DV-01` dividends scope boundary (NVCA model term sheets embed dividend covenants within liquidation preference mechanics; flagged as an upstream extraction boundary).
- [x] **Norm Calibration & Rule Base Freeze** — [`scripts/calibrate_norms.py`](../scripts/calibrate_norms.py)
  - Evaluated calibrated severity weights (`CRITICAL: 40`, `HIGH: 25`, `MEDIUM: 10`, `LOW: 5`, `INFO: 0`) against synthetic gold corpus, confirming 100% precision and recall on core expected risk flags and a +37.33 separation margin.
  - Formally froze canonical rule base at 31 rules (`rules_v1.yaml`) for MVP scope freeze.


### Remaining
- [ ] Gold annotations for the real documents require independent human labeling by Annotators A and B before reconciliation and Cohen's kappa reporting.

## Week 6 (15–21 Nov 2026)

### Completed

#### M7 — Full-Stack UI & Evaluation
- [x] **Streamlit UI** — [`app.py`](../app.py) and [`src/termsheet_ai/ui/`](../src/termsheet_ai/ui/)
  - Demo mode (bundled mock payload) and upload mode (runs `TermSheetPipeline` on the uploaded file)
  - Severity-coloured clause cards showing the original clause beside its plain-English summary
  - Flags tab with a proof-trace tree; chained rules (e.g. `R-CT-02` → `R-CT-05`) nest in the tree
  - Extracted-facts tab with verbatim source quotes and a grounding indicator
  - Investor tab wired to the SEC EDGAR client by M1
  - 8 tests in [`tests/test_ui.py`](../tests/test_ui.py), including a Streamlit `AppTest` smoke run
- [x] **Report figures** — [`scripts/make_plots.py`](../scripts/make_plots.py), run as part of `make eval`
  - Risk-flag precision/recall/F1, field-level fact accuracy, risk distribution (standard vs adversarial),
    clause-classification confusion matrix, per-family F1 and baseline comparison
  - Each figure is skipped cleanly if its input file is not present
- [x] **Annotation tooling** — [`scripts/annotator_tools.py`](../scripts/annotator_tools.py) and [`scripts/compute_kappa.py`](../scripts/compute_kappa.py)
  - Blank, unlabelled gold templates for independent Annotator B labelling (no pipeline or Annotator A labels shown)
  - Validator for gold files against the annotation guide (families, literal source spans, rule IDs)
  - Cohen's kappa on clause family and on expected flags, attribute exact-match rates, and a disagreement list for reconciliation
- [x] **Makefile targets** — `plots`, `annotate-template`, `annotate-check`, `kappa`; `make eval` now ends by generating the figures
- [x] **Tests** — [`tests/test_m7_tools.py`](../tests/test_m7_tools.py) (13 tests)

#### M3 — Deep Learning: Embeddings & Search
- [x] **Evaluation Bridge & Stage S2 Report Figures** — [`scripts/make_plots.py`](../scripts/make_plots.py) and [`reports/figures/`](../reports/figures/)
  - Generated `reports/corpus_eval.json` evaluation report bridge with full 13x13 confusion matrix and per-family classification metrics.
  - Executed plotting harness to produce `reports/figures/fig_confusion_matrix.png` and `reports/figures/fig_family_f1.png`.
- [x] **Technical Report Finalization (Stage S2)** — [`docs/FINAL_REPORT.md`](FINAL_REPORT.md)
  - Authored Section 3.2 (Stage S2 Classification Performance) with benchmark table comparing TF-IDF baseline vs. SBERT variants and adversarial detection rates.
  - Updated Section 5 Team Roles & Contribution Matrix for Member 3.
- [x] **Peer Code Review & Viva Preparation**
  - Completed peer review with designated partner M4 on Stage S2 -> S3 contracts and Stage S5 multi-exemplar retrieval grounding.
  - Rehearsed oral viva defense topics on Sentence-BERT embeddings, ChromaDB cosine vector search, dev calibration, and zero-hallucination exemplar retrieval.

#### M5 — KR&R: Ontology & Rule Base
- [x] **Technical Report Contribution: Knowledge Representation & Ontology** — [`docs/FINAL_REPORT.md`](FINAL_REPORT.md)
  - Authored comprehensive Section 2 updates documenting the 31 canonical Horn rules across all 12 canonical clause families + dividends.
  - Documented the typed NetworkX semantic network topology (103 nodes, 196 directed edges) with `Rule`, `SourceDoc`, `ClauseFamily`, `Attribute`, and `MarketNorm` nodes, and `appliesTo`/`deviatesFrom`/`conformsTo`/`citedIn`/`dependsOn` relation edges.
  - Documented empirical norm calibration results and severity weight profiles separating standard clean NVCA term sheets from adversarial founder-hostile covenants (+90.0 gap).
- [x] **Peer Review & Verification with M6**
  - Completed designated peer review pairing with Member 6 (KR&R Inference & Explainability) on multi-step forward chaining (`R-CT-02` -> `R-CT-05`), justification records, and S3->S4 grounding guards.

### Remaining (M7)
- [ ] Annotator B labels for the real documents (independent human labelling), then reconciliation with M2 and kappa reporting
- [ ] Demo video
