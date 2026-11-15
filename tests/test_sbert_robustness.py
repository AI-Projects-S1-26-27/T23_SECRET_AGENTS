"""Stage S2 SBERT Embedding & Classification Robustness Stress-Testing.

Week 5 Deliverable (Member 3 — Deep Learning: Embeddings & Search).

Validates the SBERT hybrid classifier under severe operational stress:
1. Scanned / degraded OCR noise (character substitution, dropped spaces, typo corruption)
2. Out-of-Domain (OOD) foreign legal agreements (NDAs, leases, EULAs, employment contracts)
3. Non-legal text & garbage inputs (culinary recipes, retail receipts, stack traces, symbol floods)
4. Extreme structural & boundary inputs (massive 2500+ char clauses, unicode/emojis, whitespace)
5. Idempotence and prediction consistency across repeated invocations
"""

from __future__ import annotations

import numpy as np
import pytest

from termsheet_ai.classify.sbert import SbertClassifier
from termsheet_ai.schemas.contracts import ClauseFamily, ClauseSpan


@pytest.fixture(scope="module")
def classifier() -> SbertClassifier:
    """Instantiate a shared SbertClassifier with dev-calibrated default thresholds."""
    return SbertClassifier(mode="hybrid")


# ============================================================================
# 1. Scanned and Degraded OCR Noise Robustness
# ============================================================================


class TestScannedAndOcrNoiseRobustness:
    """Evaluates embedding stability against typical OCR errors and character degradation."""

    def test_ocr_degraded_liquidation_preference(self, classifier: SbertClassifier):
        """Typo-corrupted liquidation clause must be correctly recognized."""
        # Simulated OCR noise: 0 for O, 1 for l, missing spaces
        corrupted_text = (
            "Series A Pr3ferred sh4ll receive a 1.0x n0n-partic1pating liquidati0n preference "
            "pri0r to any distributi0n to c0mmon st0ckholders."
        )
        clause = ClauseSpan(
            clause_id="ocr_lp", raw_text=corrupted_text, section_header="LIQUIDATI0N"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert result.is_unclassified is False
        assert result.classification_confidence > 0.35

    def test_ocr_degraded_anti_dilution(self, classifier: SbertClassifier):
        """Corrupted anti-dilution formula text maintains correct classification."""
        corrupted_text = (
            "The c0nversi0n price sha11 be subject t0 standard br0ad-based "
            "weighted average ant1-dilution pr0tection in the event of down-r0und financing."
        )
        clause = ClauseSpan(
            clause_id="ocr_ad", raw_text=corrupted_text, section_header="ANTI-DILUTI0N"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.ANTI_DILUTION
        assert result.is_unclassified is False

    def test_ocr_degraded_board_composition(self, classifier: SbertClassifier):
        """Board clause with noisy numbers and substitutions correctly resolves."""
        corrupted_text = (
            "The Board of Directors shall consist of three (3) directors: "
            "one des1gnated by Common, two by Series A Pr3ferred."
        )
        clause = ClauseSpan(
            clause_id="ocr_board", raw_text=corrupted_text, section_header="BOARD OF DIRECTORS"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.BOARD_COMPOSITION
        assert result.is_unclassified is False

    def test_ocr_degraded_founder_vesting(self, classifier: SbertClassifier):
        """Vesting schedule with OCR substitutions maps to FOUNDER_VESTING."""
        corrupted_text = (
            "F0under sh4res sha11 be subject t0 a f0ur-year (4-yr) vesting schedule "
            "with a 0ne-year (1-yr) cl1ff."
        )
        clause = ClauseSpan(
            clause_id="ocr_vesting", raw_text=corrupted_text, section_header="VEST1NG"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.FOUNDER_VESTING
        assert result.is_unclassified is False


# ============================================================================
# 2. Out-of-Domain (OOD) Foreign Legal Document Rejection
# ============================================================================


class TestForeignAndOutOfDomainRejection:
    """Verifies that non-term-sheet legal instruments cleanly fall back to OTHER."""

    def test_patent_assignment_clause_rejected(self, classifier: SbertClassifier):
        """Patent Assignment agreement clause must be rejected as OTHER (unclassified)."""
        patent_text = (
            "Inventor hereby assigns and transfers to Assignee all entire right, title, and "
            "interest in and to any patentable inventions, letters patent, and patent applications "
            "throughout the world, free and clear of all liens and encumbrances."
        )
        clause = ClauseSpan(
            clause_id="ood_patent", raw_text=patent_text, section_header="PATENT ASSIGNMENT"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_medical_consent_agreement_rejected(self, classifier: SbertClassifier):
        """Informed medical consent text must be rejected as OTHER (unclassified)."""
        med_text = (
            "Patient authorizes the attending physician and hospital medical staff to administer "
            "anesthesia and perform surgical diagnostic procedures as deemed necessary during operation."
        )
        clause = ClauseSpan(
            clause_id="ood_med", raw_text=med_text, section_header="INFORMED CONSENT"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_real_estate_commercial_lease_rejected(self, classifier: SbertClassifier):
        """Commercial lease agreement text must be rejected as OTHER."""
        lease_text = (
            "Tenant shall pay to Landlord as Base Rent for the Leased Premises the sum of "
            "$12,500 per month, payable in advance on the first day of each calendar month. "
            "Tenant shall also deposit a Security Deposit equal to two months rent."
        )
        clause = ClauseSpan(clause_id="ood_lease", raw_text=lease_text, section_header="BASE RENT")
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_software_eula_license_rejected(self, classifier: SbertClassifier):
        """Software End-User License Agreement clause must be rejected."""
        eula_text = (
            "Licensor grants you a revocable, non-exclusive, non-transferable, limited license "
            "to download, install, and use the Application strictly in accordance with the terms "
            "of this Agreement. You shall not decompile, reverse engineer, or disassemble the binary."
        )
        clause = ClauseSpan(
            clause_id="ood_eula", raw_text=eula_text, section_header="GRANT OF LICENSE"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_employment_non_compete_rejected(self, classifier: SbertClassifier):
        """Employment non-competition covenant must be rejected."""
        non_compete_text = (
            "During the term of employment and for twelve (12) months following termination, "
            "Employee shall not directly or indirectly engage in, perform services for, "
            "or acquire an equity interest in any business competing with Employer within North America."
        )
        clause = ClauseSpan(
            clause_id="ood_nc", raw_text=non_compete_text, section_header="NON-COMPETE"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_construction_contract_rejected(self, classifier: SbertClassifier):
        """Construction contract clause must be rejected."""
        construction_text = (
            "Contractor shall supply all labor, materials, scaffoldings, tools, and construction "
            "equipment necessary for the framing and dry-walling of the commercial structure "
            "in accordance with architectural specifications Exhibit B."
        )
        clause = ClauseSpan(
            clause_id="ood_const", raw_text=construction_text, section_header="SCOPE OF WORK"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER


# ============================================================================
# 3. Non-Legal Text & Garbage Input Rejection
# ============================================================================


class TestNonLegalGarbageRejection:
    """Verifies that arbitrary non-legal texts and garbage are rejected."""

    def test_culinary_recipe_rejected(self, classifier: SbertClassifier):
        """Pancake culinary recipe instructions must be rejected."""
        recipe_text = (
            "Combine 2 cups of unbleached flour, 2 tablespoons sugar, 2 teaspoons baking powder, "
            "and 1/2 teaspoon salt in a large bowl. Beat two eggs with 1.5 cups whole milk "
            "and melted butter. Cook on a greased griddle until bubbles form."
        )
        clause = ClauseSpan(
            clause_id="garb_recipe", raw_text=recipe_text, section_header="DIRECTIONS"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_retail_shopping_receipt_rejected(self, classifier: SbertClassifier):
        """Retail store receipt with itemized prices must be rejected."""
        receipt_text = (
            "STORE #4029 - ORGANIC MARKET\n"
            "01/14/2026  14:32:01  CASHIER 04\n"
            "1. ORGANIC OAT MILK        $4.49\n"
            "2. ARTISAN SOURDOUGH LOAF   $5.99\n"
            "3. HONEYCRISP APPLES (2LB) $6.20\n"
            "SUBTOTAL:                 $16.68\n"
            "TAX (8.25%):               $1.38\n"
            "TOTAL:                    $18.06"
        )
        clause = ClauseSpan(
            clause_id="garb_receipt", raw_text=receipt_text, section_header="RECEIPT"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_programming_stack_trace_rejected(self, classifier: SbertClassifier):
        """Python exception traceback must be rejected."""
        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "src/pipeline.py", line 142, in execute\n'
            "    result = engine.infer(memory)\n"
            '  File "src/reasoning.py", line 89, in infer\n'
            "ZeroDivisionError: float division by zero"
        )
        clause = ClauseSpan(clause_id="garb_trace", raw_text=traceback_text, section_header="ERROR")
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True

    def test_punctuation_symbol_flood_rejected(self, classifier: SbertClassifier):
        """Repeated punctuation and random ASCII symbols must be rejected."""
        symbol_text = "!@#$%^&*()_+=-~`{}[]|\\:;\"'<>,.?/!@#$%^&*()_+"
        clause = ClauseSpan(clause_id="garb_symbols", raw_text=symbol_text, section_header="###")
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True
        assert result.classification_confidence == 0.0


# ============================================================================
# 4. Extreme Structural & Boundary Inputs
# ============================================================================


class TestExtremeStructuralAndBoundaryInputs:
    """Verifies edge cases including length extremes, Unicode, and empty strings."""

    def test_massive_clause_length_handling(self, classifier: SbertClassifier):
        """Massive legal clause (2,500+ characters) is safely truncated/embedded without crashing."""
        standard_lp = (
            "In the event of any liquidation, dissolution, or winding up of the Company, "
            "the proceeds shall be distributed first to holders of Series A Preferred Stock "
            "at 1.0x original purchase price plus declared dividends. "
        )
        massive_text = standard_lp * 15  # ~3,000 characters
        assert len(massive_text) > 2500

        clause = ClauseSpan(
            clause_id="massive_lp", raw_text=massive_text, section_header="LIQUIDATION"
        )
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert result.is_unclassified is False

    def test_emoji_and_unicode_symbols(self, classifier: SbertClassifier):
        """Clauses containing emojis and varied currency symbols embed cleanly."""
        emoji_text = (
            "🚀 Series A Preferred Stock receives €1,000,000 aggregate purchase price "
            "with a 1.0x non-participating liquidation preference 💰📈."
        )
        clause = ClauseSpan(clause_id="emoji_lp", raw_text=emoji_text, section_header="LIQUIDATION")
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.LIQUIDATION_PREFERENCE
        assert result.is_unclassified is False

    def test_whitespace_and_newline_only_clause(self, classifier: SbertClassifier):
        """Clauses containing only spaces, tabs, and newlines evaluate to unclassified OTHER."""
        whitespace_text = "   \n\t\r\n   "
        clause = ClauseSpan(clause_id="blank_c", raw_text=whitespace_text, section_header="")
        result = classifier.classify([clause])[0]

        assert result.detected_family == ClauseFamily.OTHER
        assert result.is_unclassified is True
        assert result.classification_confidence == 0.0

    def test_single_word_header_empty_body(self, classifier: SbertClassifier):
        """Single-word header with empty body text evaluates gracefully without crash."""
        clause = ClauseSpan(clause_id="sparse_c", raw_text="", section_header="Dividends")
        result = classifier.classify([clause])[0]

        # Should either resolve based on header or gracefully fall back to OTHER
        assert isinstance(result.detected_family, ClauseFamily)
        assert 0.0 <= result.classification_confidence <= 1.0


# ============================================================================
# 5. Idempotence & Determinism
# ============================================================================


class TestModelIdempotenceAndConsistency:
    """Verifies that classification outputs are strictly deterministic and reproducible."""

    def test_idempotent_predictions(self, classifier: SbertClassifier):
        """Classifying the same clause twice in separate calls produces identical results."""
        clause = ClauseSpan(
            clause_id="c_test",
            raw_text="The board shall consist of 3 members: two common and one preferred.",
            section_header="Board of Directors",
        )

        res1 = classifier.classify([clause])[0]
        res2 = classifier.classify([clause])[0]

        assert res1.detected_family == res2.detected_family
        assert np.isclose(res1.classification_confidence, res2.classification_confidence, atol=1e-6)
        assert res1.is_unclassified == res2.is_unclassified

    def test_batch_vs_single_item_probability_equivalence(self, classifier: SbertClassifier):
        """Classifying a clause in a batch vs. individually yields identical probabilities."""
        c1 = ClauseSpan(
            clause_id="c1",
            raw_text="Series A has a 1.0x non-participating liquidation preference.",
            section_header="Liquidation",
        )
        c2 = ClauseSpan(
            clause_id="c2",
            raw_text="Standard broad-based weighted average anti-dilution applies.",
            section_header="Anti-Dilution",
        )

        batch_results = classifier.classify([c1, c2])
        single_res1 = classifier.classify([c1])[0]
        single_res2 = classifier.classify([c2])[0]

        assert batch_results[0].detected_family == single_res1.detected_family
        assert np.isclose(
            batch_results[0].classification_confidence,
            single_res1.classification_confidence,
            atol=1e-4,
        )
        assert batch_results[1].detected_family == single_res2.detected_family
        assert np.isclose(
            batch_results[1].classification_confidence,
            single_res2.classification_confidence,
            atol=1e-4,
        )
