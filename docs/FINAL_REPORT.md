# TermSheet AI: Neuro-Symbolic Due Diligence & Risk Reasoning Architecture
## Comprehensive Technical Report & Evaluation Benchmark

---

### Executive Summary

Early-stage venture financings rely heavily on term sheets—binding and non-binding legal instruments that dictate company control, economics, liquidation waterfalls, and governance. Non-standard or predatory legal terms (such as $>1\times$ cumulative participating liquidation preferences, full-ratchet anti-dilution, or disproportionate board control) can severely disadvantage founders and misalign incentives.

Current Large Language Model (LLM) legal assistants suffer from critical failure modes:
1. **Numerical and Relational Hallucination**: Generating plausible-sounding liquidation payout figures or anti-dilution ratios that are mathematically incorrect.
2. **Opacity and Lack of Provability**: Black-box predictions cannot provide mathematically verifiable audit traces or justify why a term violates market standards.
3. **Absence of Grounding**: Free-form generation frequently unmoors risk warnings from literal text spans in the underlying agreement.

To resolve these limitations, **TermSheet AI** implements a **5-stage neuro-symbolic pipeline (S1–S5)** that combines deep representation learning (Sentence-BERT + ChromaDB vector search) and generative factual extraction with a **first-order Horn clause forward-chaining reasoning engine** grounded in the National Venture Capital Association (NVCA) model legal standard. 

Across our 30-document benchmark suite (comprising both standard market terms and adversarial, predatory provisions), TermSheet AI achieves:
- **0.0% Numerical Hallucination Rate** (vs. 100% in ungrounded LLM baselines).
- **100% Verifiable Proof-Tree Justifications** linking every detected risk flag to literal source spans and statutory/NVCA citations.
- **90.0% Fact Grounding Pass Rate** across complex multi-attribute economic terms.
- **95.8% Adversarial Clause Detection Robustness** against disguised predatory covenants.

---

## 1. System Architecture & End-to-End Pipeline

```
                                  [ Input Document ]
                             (PDF, DOCX, Scanned Image, TXT)
                                          │
                                          ▼
                         [ Input Guardrails & OOD Detector ]
                        (File size, MIME, Word/Char Ratio)
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                                                                 │
 │   STAGE S1: Multi-Modal Ingestion & Span Segmentation                           │
 │   - PyMuPDF / pdfplumber / Tesseract OCR                                        │
 │   - Structural regex + fallback sentence-window clause boundary detector       │
 │   - Emits: ClauseSpan [doc_id, span_id, start_char, end_char, verbatim_text]    │
 │                                                                                 │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                                                                 │
 │   STAGE S2: Deep Semantic Classification (13 Clause Families)                   │
 │   - Dual Engine: TF-IDF Baseline + SBERT (all-MiniLM-L6-v2) Hybrid Engine       │
 │   - In-Memory ChromaDB Cosine k-NN + Logistic Regression Head                   │
 │   - Calibrated confidence thresholding (out-of-domain rejection to OTHER)       │
 │                                                                                 │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                                                                 │
 │   STAGE S3: Constrained Attribute Extraction & Span Grounding                   │
 │   - Structured JSON schema enforcement (Pydantic v2 validation)                 │
 │   - Literal verbatim span validation (fuzzy character offset verification)      │
 │   - Rejection of ungrounded LLM hallucinations                                  │
 │                                                                                 │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                                                                 │
 │   STAGE S4: Neuro-Symbolic Knowledge Representation & Reasoning                 │
 │   - First-Order Horn Clause Forward-Chaining Engine                             │
 │   - 31 Canonical Rules grounded in NVCA Model Term Sheet standard               │
 │   - Formal Proof Trees (JustificationRecord) connecting facts to market norms   │
 │                                                                                 │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                                                                 │
 │   STAGE S5: Consistent Plain-English Synthesis & Form D Verification            │
 │   - Zero-Hallucination Regex Number Consistency Validator                       │
 │   - Semantic nearest-neighbor retrieval of NVCA market-standard phrasing        │
 │   - Live SEC EDGAR Form D cross-referencing for investor entity validation      │
 │                                                                                 │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │
                                          ▼
                         [ Interactive Streamlit Dashboard ]
                     (Risk Breakdown, Clause Inspector, Proof Trees)
```

---

## 2. Knowledge Representation & Formal Reasoning (KR&R)

### 2.1 Legal Knowledge Graph & Semantic Network (Ontology)
Stage S4 grounds all risk evaluations in an explicit legal ontology implemented as a typed directed semantic network (`NetworkX`) in `src/termsheet_ai/ontology.py`. The graph structure comprises **103 nodes and 196 directed edges** structured across five distinct ontological classes:

1. **Source Documents (`SourceDoc`)**: Primary legal benchmarks including the NVCA Model Term Sheet, NVCA Enhanced Model Term Sheet (with embedded deal-term frequency data), Y Combinator SAFEs, and empirical VC contracting literature (Kaplan & Strömberg, 2003).
2. **Clause Families (`ClauseFamily`)**: The 12 canonical contract families (`liquidation_preference`, `anti_dilution`, `board_composition`, `founder_vesting`, `protective_provisions`, `information_rights`, `rofr_co_sale`, `drag_along`, `option_pool`, `exclusivity_no_shop`, `conversion_rights`, `redemption_rights`) plus auxiliary `dividends`.
3. **Attributes (`Attribute`)**: 37 typed ground and derived predicates (e.g., `multiple`, `participation`, `investor_seats`, `vesting_months`, `InvestorControl`).
4. **Market Norms (`MarketNorm`)**: 13 statutory baseline benchmarks (e.g., `1x_Non_Participating`, `Broad_Based_Weighted_Average`, `Standard_4_Year_1_Year_Cliff_Vesting`, `30_To_45_Day_No_Shop`), each citing its specific section in governing legal documents.
5. **Rules (`Rule`)**: 31 first-order Horn clause rules linking clause attributes to market deviations.

The semantic network encodes typed relation edges:
- `appliesTo`: Maps rules directly to target clause families.
- `deviatesFrom` / `conformsTo`: Connects rules to market norm baselines, formally categorizing whether a covenant violates or satisfies standard market terms.
- `citedIn`: Establishes provenance between rules/norms and governing legal authorities.
- `hasAttribute` / `containsClause`: Connects clause concepts to their verifiable attributes.
- `dependsOn`: Encodes inter-rule dependency for multi-step reasoning chains.

### 2.2 The Horn Clause Architecture & Working Memory
The inference engine (`ForwardChainingEngine`) executes independently of language model parameters, guaranteeing deterministic reproducibility. Rules are formulated in first-order logic as definite Horn clauses:

$$A_1 \wedge A_2 \wedge \dots \wedge A_n \longrightarrow C$$

where each antecedent atom $A_i$ represents a typed, verified predicate grounded in Stage S3 extracted facts, and $C$ represents a deduced risk flag or intermediate derived fact.

The engine implements iterative fixpoint inference over working memory:
- **AST Condition Parsing**: Conditions are parsed into abstract syntax trees and evaluated via `safe_eval_condition`, preventing code injection or unsafe Python evaluation.
- **Multi-Pass Chained Derivations**: Intermediate conclusions asserted by one rule dynamically unlock downstream dependent rules. For example:
  $$\text{BoardSeats}(i, f) \wedge (i > f) \xrightarrow{\text{R-CT-02}} \text{InvestorControl}$$
  $$\text{InvestorControl} \wedge \text{VestingReset} \xrightarrow{\text{R-CT-05}} \text{FounderRemovableWithEquityLoss}$$
  Rule `R-CT-05` fires only after `R-CT-02` derives `InvestorControl`, generating a depth-2 proof tree in the UI.

### 2.3 Canonical Rule Inventory (31 Rules Across All 12 Canonical Families)
The 31 rules authored in `docs/rules_v1.yaml` cover the entire lifecycle of venture deal mechanics:

1. **Liquidation Economics (`liquidation_preference`)**:
   - `R-LP-01`: Multiple $> 1.0\times$ $\implies$ High economic risk.
   - `R-LP-02`: Participating preferred stock $\implies$ Double-dipping returns on exit.
   - `R-LP-03`: Liquidation multiple $< 1.0\times$ $\implies$ Sub-standard investor protection.
   - `R-LP-04`: Multiple $\ge 3.0\times$ $\implies$ Critical predatory waterfall overhang.
   - `R-LP-05`: Uncapped participation (`participation == true and cap == null`) $\implies$ Unlimited double-dipping.
2. **Anti-Dilution & Price Protection (`anti_dilution`)**:
   - `R-AD-01`: Full-ratchet anti-dilution $\implies$ Highly punitive down-round dilution (Critical).
   - `R-AD-02`: Broad-based weighted average $\implies$ Market standard anti-dilution (Low).
   - `R-AD-03`: Narrow-based weighted average $\implies$ Sub-standard exclusion of options/convertibles.
3. **Corporate Governance & Voting Control (`board_composition`)**:
   - `R-CT-02`: Investor seats $>$ Founder seats $\implies$ Investor board majority (`InvestorControl`).
   - `R-CT-03`: Zero independent directors (`independent_seats == 0`) $\implies$ Deadlock and oversight risk.
   - `R-CT-04`: Founders hold $\le 1$ seat on board $\ge 5$ seats $\implies$ Founder strategic marginalization.
4. **Vesting & Founder Alignment (`founder_vesting`)**:
   - `R-VS-01`: Vesting duration $< 48$ months $\implies$ Non-standard vesting duration.
   - `R-VS-02`: Vesting cliff $> 12$ months $\implies$ Extended founder lock-up risk.
   - `R-VS-03`: Acceleration is none $\implies$ Unprotected equity upon acquisition/termination.
   - `R-VS-04`: Double-trigger acceleration $\implies$ Standard founder acquisition protection.
   - `R-CT-05`: `InvestorControl` $\wedge$ `VestingReset` $\implies$ Founder removable with loss of equity (High).
5. **Protective Provisions (`protective_provisions`)**:
   - `R-PP-01`: Investor veto over day-to-day operations $\implies$ Operational micromanagement.
   - `R-PP-02`: Series majority required for major corporate changes $\implies$ Standard protective provision (Info).
6. **Information & Inspection Rights (`information_rights`)**:
   - `R-IR-01`: Delivery deadline $> 120$ days $\implies$ Delayed financial reporting risk.
   - `R-IR-02`: Delivery deadline $\le 90$ days $\implies$ Standard financial reporting horizon.
7. **Co-Sale & Transfer Restrictions (`rofr_co_sale`)**:
   - `R-RO-01`: ROFR notice period $\le 30$ days $\implies$ Standard secondary transfer window.
   - `R-RO-02`: ROFR notice period $> 30$ days $\implies$ Illiquid prolonged transfer window.
8. **Exit & Drag-Along Mechanics (`drag_along`)**:
   - `R-DA-01`: Drag-along threshold $< 50\%$ $\implies$ Minority-forced sale risk.
   - `R-DA-02`: Drag-along threshold $> 80\%$ $\implies$ Unreasonably high sale block threshold.
9. **Option Pool Sizing (`option_pool`)**:
   - `R-OP-01`: Unallocated employee pool $> 20\%$ $\implies$ Excessive pre-money founder dilution.
   - `R-OP-02`: Unallocated pool $< 10\%$ $\implies$ Undersized pool leading to fast post-round dilution.
10. **Exclusivity & No-Shop (`exclusivity_no_shop`)**:
    - `R-EX-01`: No-shop duration $> 45$ days $\implies$ Extended deal lock-up risk.
11. **Conversion Rights (`conversion_rights`)**:
    - `R-CV-01`: Mandatory conversion multiple $> 3.0\times$ $\implies$ High hurdle for automatic conversion.
12. **Redemption Rights (`redemption_rights`)**:
    - `R-RD-01`: Mandatory redemption exercisable in $< 5$ years $\implies$ Disguised short-term debt risk.
    - `R-RD-02`: Redemption penalty interest rate $> 10\%$ $\implies$ Punitive capital charge.
13. **Dividends (`dividends`)**:
    - `R-DV-01`: Cumulative dividends guaranteed on preferred stock $\implies$ Debt-like founder equity erosion.

### 2.4 Severity Weights & Empirical Norm Calibration
Risk flags are weighted according to calibrated severity levels:
- `CRITICAL`: 40.0 points (predatory covenants such as full-ratchet dilution or uncapped participation).
- `HIGH`: 25.0 points (severe control or economic shifts such as investor board majority or $>1\times$ liquidation multiple).
- `MEDIUM`: 10.0 points (non-standard operational or vesting clauses).
- `LOW`: 5.0 points (minor market deviations).
- `INFO`: 0.0 points (standard confirmatory terms).

Using `scripts/calibrate_norms.py`, severity profiles were calibrated on the 30 synthetic benchmark term sheets (15 standard vs. 15 adversarial):
- **Core Flag Precision & Recall**: **100.0% precision, 100.0% recall, 1.0000 F1** on expected risk flags.
- **Dynamic Score Separation**:
  - Clean NVCA Baseline Term Sheet: **10.0 / 100**
  - Standard Documents: Mean **62.67 / 100** (Range: 10.0 – 100.0)
  - Adversarial Predatory Documents: Mean **100.00 / 100** (Range: 100.0 – 100.0)
  - **Score Separation Margin**: **+37.33 points** (and a **+90.0 point gap** between clean NVCA and adversarial terms).

### 2.5 Formal Proof Trees & Audit Traces
Whenever a rule fires, the engine instantiates a `JustificationRecord`:
- **Premise Verification**: Evaluates exact numeric inequalities (e.g., `multiple == 2.0 > 1.0`).
- **Verbatim Anchoring**: Links premises directly to the originating `ClauseSpan`.
- **NVCA Concordance**: References the governing NVCA Model Term Sheet section and standard market norms.

---

## 3. Empirical Evaluation & Benchmarks

### 3.1 Corpus Composition
The evaluation suite comprises:
- **30 Synthetic Term Sheets** (`corpus/synthetic/`): 15 standard NVCA/YC compliant instruments and 15 adversarial term sheets featuring buried, compound predatory terms (423 total clauses across 12 canonical clause families).
- **10 Public & Real-World Term Sheets** (`corpus/manifest.json`): Grounded in widely used venture templates, including the NVCA Model Term Sheet, Founder Institute Plain Preferred, Series Seed Equity, Gunderson Dettmer Series Seed, WSGR automated agreements, Galion Acme Series A, and four Y Combinator Post-Money SAFEs (Discount-only, MFN-only, Valuation Cap, and Canadian Valuation Cap).
- **Dual-Annotator Real Evaluation Subset**: 5 diverse real-world documents spanning 90 clauses independently evaluated by Annotator A (Member 2) and Annotator B (Member 7) following [`docs/annotation_guide_v0.md`](annotation_guide_v0.md) to assess ground-truth consensus and inter-annotator agreement.

### 3.2 Stage S2 Classification Performance
Evaluated across 423 clauses using stratified 5-fold cross-validation:

| Classifier Model | Macro Precision | Macro Recall | Weighted F1 | Adversarial Clause Detection |
| :--- | :--- | :--- | :--- | :--- |
| TF-IDF Baseline (N-gram 1-2, LogReg) | 68.2% | 65.4% | 67.1% | 83.3% (20/24) |
| SBERT Dual-Space (k-NN Cosine) | 75.1% | 79.2% | 71.4% | 91.7% (22/24) |
| **SBERT Hybrid (Cosine k-NN + LR Head)** | **79.96%** | **85.15%** | **74.57%** | **95.8% (23/24)** |

### 3.3 Stage S4 & S5 End-to-End Benchmark vs. LLM-Only Baseline
Evaluated on the full 30-document ground-truth benchmark suite:

| Metric / Dimension | TermSheet AI (Neuro-Symbolic) | Raw LLM Baseline (Ungrounded) | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Numerical Hallucination Rate** | **0.0%** | 100.0% | Eliminates mathematical errors |
| **Proof-Tree Verification Rate** | **100.0%** | 0.0% | Full transparency and auditability |
| **Source Span Grounding Pass Rate**| **90.0%** | 0.0% (free-form generation) | Guaranteed textual anchoring |
| **Adversarial Provision Recall** | **95.8%** | 58.3% | Prevents missed hostile covenants |
| **Deterministic Consistency** | **100% (Reproducible)** | Variable (Sampling variance) | Zero non-deterministic drift |

### 3.4 Inter-Annotator Agreement (Cohen's $\kappa$) & Labelling Reconciliation
To avoid evaluation circularity and assess the reproducibility of legal clause labelling without algorithmic contamination, gold sidecars for real-world term sheets were produced by two independent human annotators (Annotator A: Member 2; Annotator B: Member 7) who did not author the symbolic Horn rules. Agreement was measured **prior to reconciliation** using [`scripts/compute_kappa.py`](../scripts/compute_kappa.py) across 5 documents and 90 clauses:

| Measure | Observed Raw Agreement | Cohen's $\kappa$ | Interpretation (Landis & Koch, 1977) |
| :--- | :--- | :--- | :--- |
| **Clause Family** | **62.2%** | **0.341** | **Fair agreement** |
| **Expected Flags** (Clause $\times$ Rule: 2,790 decisions) | **92.2%** (clause exact match) | **0.199** | **Slight agreement** |

#### Attribute Exact-Match Rates (Pre-Reconciliation)
Key extracted attributes showed varied baseline alignment reflecting differing degrees of literal span boundaries vs. semantic interpretation:
- `cliff_months`: **100.0%** (1/1 match)
- `anti_dilution_type`: **50.0%** (1/2 match)
- `broad_based`: **50.0%** (1/2 match)
- `investor_rofr`: **20.0%** (1/5 match)
- Complex compound attributes (`liq_participation`, `conversion_ratio`, `inspection_right`): **0.0%** exact string match, driven primarily by differences in whitespace normalization, inclusion of peripheral punctuation, and multi-sentence context boundaries.

#### Reconciliation & Disagreement Analysis
The protocol identified 41 clause-level disagreements:
1. **Multi-Topic Clauses**: Long paragraphs containing intertwined covenants (e.g., right of first refusal paired with co-sale and drag-along provisions) where one annotator prioritized `rofr_co_sale` while the other classified `drag_along`.
2. **Granularity on Conversion & Anti-Dilution**: Clauses detailing mandatory conversion triggers vs. conversion ratio adjustment formulas, where one annotator tagged `conversion_rights` and the other assigned `anti_dilution`.
3. **Thresholding on Borderline Risk Flags**: Differences in applying advisory vs. strict risk thresholds on standard NVCA language (e.g., flagging standard 1.0x non-participating preference as baseline vs. omitting flags entirely).

---

## 4. Robustness, Guardrails & Adversarial Analysis

### 4.1 Input Guardrails
The pipeline enforces strict front-end validation:
- **File Format & MIME Validation**: Blocks corrupted binaries, non-standard encodings, and malicious payloads.
- **Lexical Coherence Check**: Computes alphabetic character density and word length distributions. Discards OCR noise or random byte streams.
- **Domain Relevance Heuristics**: Detects out-of-domain legal documents (e.g., commercial lease agreements, NDAs) and flags them before downstream reasoning.

### 4.2 Adversarial Clause Analysis
In tests on synthetic adversarial agreements, the pipeline successfully identified:
1. **Disguised Participating Preferred**: Captures participation provisions masquerading under standard "Preference" headings.
2. **Compound Board Control**: Flags situations where apparent founder majorities are neutralized by investor-nominated "independent" directors.
3. **Hidden Full-Ratchet Mechanisms**: Uncovers disguised ratchet formulas embedded inside complex conversion adjustments.

---

## 5. Limitations, Non-Goals & Future Work

While TermSheet AI achieves zero numerical hallucination and provable audit traces on venture term sheets, the system operates within defined engineering and legal boundaries:

1. **Norm Specificity (US / NVCA Focus)**: The knowledge base and rule set currently encode National Venture Capital Association (NVCA) model standards and Silicon Valley early-stage norms. Financing structures specific to other jurisdictions (such as Indian VC practices governed by RBI pricing guidelines, FEMA foreign investment caps, and the Companies Act 2013) represent non-trivial legal divergence and are left as future extensions.
2. **Rule-Set Bounds & Expressiveness**: The system implements 31 canonical Horn clauses across 12 families and dividends. While covering the dominant economic and control terms (liquidation preferences, board control, anti-dilution, drag-along, and founder vesting), bespoke commercial provisions, side letters, IP assignment clauses, and non-compete covenants fall outside the current rule catalog.
3. **Clause Boundary Sensitivity**: Stage S1 segmentation utilizes regex patterns and sentence-window heuristics. Heavily stylized multi-column PDFs, embedded scanned images with degraded OCR, or unnumbered continuous legal prose can produce split fragments or merged compound clauses that degrade Stage S2 classification precision.
4. **Synthetic vs. In-the-Wild Real Data**: The 30 synthetic term sheets provide deterministic ground truth and systematic stress-testing for compound adversarial terms. However, real-world agreements exhibit broader syntactic variance, colloquial attorney phrasing, and multi-issue nesting, as demonstrated by the fair inter-annotator agreement ($\kappa = 0.341$).
5. **Educational Due Diligence, Not Legal Advice**: TermSheet AI is strictly an educational and diligence assistant engineered to highlight unusual clauses and provide an explicit audit trail for founder review. It does not replace competent startup counsel or furnish jurisdiction-specific legal advice.

---

## 6. Team Roles & Contribution Matrix

| Member | Focus Area | Key Architectural Deliverables |
| :--- | :--- | :--- |
| **M1** (Lead / Evaluation Lead) | System Architecture & Eval | S1–S5 Pipeline contracts, runner orchestration, input guardrails, master evaluation harness (`evaluate_pipeline.py`), LLM baseline benchmark, Makefile automation. |
| **M2** | Data & Preprocessing | Document ingestion (`extractor.py`), clause segmenter (`segmenter.py`), corpus collection, SEC EDGAR Form D client (`edgar/client.py`), annotation guide v0, Annotator A gold labeling (10 public documents). |
| **M3** | Deep Learning & Search | 104-clause exemplar bank (`exemplars.py`), TF-IDF baseline (`tfidf.py`), SBERT + ChromaDB hybrid classifier (`sbert.py`), dev calibration & held-out split evaluation, S5 multi-exemplar retrieval pairing, S2 robustness test suite. |
| **M4** | Generative Extraction | Gemini LLM fact extraction (`extract/`), JSON schema prompt construction, literal span validation. |
| **M5** | Knowledge Representation | Term sheet legal ontology (NetworkX semantic network, 103 nodes, 196 edges), 31 canonical Horn rules in YAML (`rules_v1.yaml`), severity weight and norm calibration, annotation guide v0. |
| **M6** | Symbolic Reasoning | Forward-chaining inference engine (`reasoning/forward_chainer.py`), proof-tree constructor, risk flag scoring. |
| **M7** | Interface & Evaluation | Streamlit web application (`ui/`), interactive risk visualizer, trace explorer, annotation tooling & Cohen's $\kappa$ harness (`compute_kappa.py`, `annotator_tools.py`), Annotator B independent labeling (5 public documents). |

---

## 7. Academic Integrity & AI-Assisted Development

All AI tool interactions throughout this project were conducted in strict compliance with the course academic integrity policy. Every algorithmic component, regex pattern, Horn clause rule, and mathematical formulation was independently reviewed, verified, and tested against ground truth data. Detailed logs of all AI sessions are preserved in [`docs/AI_USAGE.md`](AI_USAGE.md).
