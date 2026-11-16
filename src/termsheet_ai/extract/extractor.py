"""Structured Fact Extractor (Stage S3).

Extracts structured VC term-sheet facts conforming to Pydantic contract schemas:
- LiquidationPreferenceFact
- AntiDilutionFact
- BoardCompositionFact
- VestingFact
- ProtectiveProvisionsFact

Every extracted fact is verified and grounded against source text using validate_source_span.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from termsheet_ai.extract.llm_cache import make_cached_callable
from termsheet_ai.extract.llm_client import make_gemini_callable
from termsheet_ai.extract.span_validator import validate_source_span
from termsheet_ai.pipeline.base import BaseExtractor
from termsheet_ai.schemas.contracts import (
    AntiDilutionFact,
    AntiDilutionType,
    BoardCompositionFact,
    ClauseFamily,
    ClauseSpan,
    ExtractedFacts,
    LiquidationPreferenceFact,
    ParticipationType,
    ProtectiveProvisionsFact,
    SourceSpan,
    VestingFact,
)

logger = logging.getLogger(__name__)

# Protocol type for an LLM text generation function: receives (prompt: str) -> str
LLMCallable = Callable[[str], str]


EXTRACTION_PROMPTS = {
    ClauseFamily.LIQUIDATION_PREFERENCE: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "multiple": <float or null, e.g. 1.0, 2.0>,\n'
        '  "participation": <"non_participating" | "fully_participating" | "capped_participating" | "unknown">,\n'
        '  "cap_multiple": <float or null>,\n'
        '  "seniority": <string or null, e.g. "senior", "pari_passu">,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.ANTI_DILUTION: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "anti_dilution_type": <"broad_based_weighted_average" | "narrow_based_weighted_average" | "full_ratchet" | "none" | "unknown">,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.BOARD_COMPOSITION: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "founder_seats": <integer or null>,\n'
        '  "investor_seats": <integer or null>,\n'
        '  "independent_seats": <integer or null>,\n'
        '  "total_seats": <integer or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.FOUNDER_VESTING: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "total_vesting_years": <float or null, e.g. 4.0>,\n'
        '  "cliff_months": <integer or null, e.g. 12>,\n'
        '  "acceleration_type": <"none" | "single_trigger" | "double_trigger" | "unknown">,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.PROTECTIVE_PROVISIONS: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "requires_series_majority": <boolean>,\n'
        '  "covered_actions": [<list of strings describing restricted corporate actions>],\n'
        '  "veto_day_to_day_operations": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.DRAG_ALONG: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "approval_threshold": <float or null, e.g. 50.0, 60.0>,\n'
        '  "requires_founder_consent": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.OPTION_POOL: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "pool_percentage": <float or null, e.g. 10.0, 15.0, 20.0>,\n'
        '  "pre_money_pool": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.REDEMPTION_RIGHTS: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "redemption_years": <float or null, e.g. 5.0>,\n'
        '  "interest_rate": <float or null, e.g. 8.0>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.EXCLUSIVITY_NO_SHOP: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "duration_days": <integer or null, e.g. 30, 45, 60>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.ROFR_CO_SALE: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "investor_co_sale": <boolean or null>,\n'
        '  "investor_rofr": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.INFORMATION_RIGHTS: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "financial_statements_days": <integer or null, e.g. 45, 90>,\n'
        '  "inspection_rights": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
    ClauseFamily.CONVERSION_RIGHTS: (
        "You are an expert legal fact extraction assistant. Analyze the term sheet clause below "
        "and output ONLY valid JSON matching this schema:\n"
        "{{\n"
        '  "conversion_ratio": <float or null, e.g. 1.0>,\n'
        '  "optional_conversion": <boolean or null>,\n'
        '  "quote": <verbatim substring from clause text supporting this fact>\n'
        "}}\n\n"
        "Clause text:\n{clause_text}\n"
    ),
}


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first valid JSON dictionary from LLM output."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()

    # 1. Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    # 2. Markdown fenced JSON block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Outer bracket extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _quote_contains_numeric(quote_text: str, value: float | int) -> bool:
    """Verify whether a candidate numeric value is reflected in the quote text."""
    if value is None:
        return True

    # Check for direct number or formatted float (e.g. 2.0 -> '2.0' or '2' or '2x')
    val_str = str(value)
    if val_str in quote_text:
        return True

    if isinstance(value, float) and value.is_integer():
        int_str = str(int(value))
        if int_str in quote_text:
            return True

    # Word forms for single-digit integers
    num_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    int_val = int(value) if isinstance(value, (int, float)) and float(value).is_integer() else None
    if int_val in num_words:
        pattern = r"\b" + num_words[int_val] + r"\b"
        if re.search(pattern, quote_text, re.IGNORECASE):
            return True

    return False


class FactExtractor(BaseExtractor):
    """Schema-compliant fact extractor with source-span validation and LLM/heuristic support."""

    def __init__(self, llm_callable: Optional[LLMCallable] = None):
        """Initialize extractor.

        Args:
            llm_callable: Optional function (prompt: str) -> str that queries an LLM.
                          If None or if generation fails, deterministic extraction runs.
        """
        self.llm_callable = llm_callable
        # Tracks whether each clause fact was extracted via 'llm' or 'fallback'
        self.extraction_modes: Dict[str, str] = {}

    @classmethod
    def with_gemini(
        cls,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> "FactExtractor":
        """Instantiate FactExtractor configured with Gemini API and prompt caching."""
        gemini_callable = make_gemini_callable(model=model, api_key=api_key)
        llm = make_cached_callable(gemini_callable) if use_cache else gemini_callable
        return cls(llm_callable=llm)

    def extract(self, doc_id: str, clauses: List[ClauseSpan]) -> ExtractedFacts:
        """Extract structured domain attributes anchored to source text spans."""
        facts = ExtractedFacts(doc_id=doc_id)

        for clause in clauses:
            family = clause.detected_family

            if family == ClauseFamily.LIQUIDATION_PREFERENCE:
                fact = self._extract_liquidation_preference(clause)
                if fact is not None:
                    if facts.liquidation_preference is None:
                        facts.liquidation_preference = fact
                    else:
                        # Secondary liquidation clause: register for cross-clause checking
                        facts.custom_facts[f"liquidation_multiple_{clause.clause_id}"] = (
                            fact.multiple
                        )

            elif family == ClauseFamily.ANTI_DILUTION and facts.anti_dilution is None:
                fact = self._extract_anti_dilution(clause)
                if fact is not None:
                    facts.anti_dilution = fact

            elif family == ClauseFamily.BOARD_COMPOSITION and facts.board_composition is None:
                fact = self._extract_board_composition(clause)
                if fact is not None:
                    facts.board_composition = fact

            elif family == ClauseFamily.FOUNDER_VESTING:
                fact = self._extract_vesting(clause)
                if fact is not None:
                    if facts.founder_vesting is None:
                        facts.founder_vesting = fact
                    else:
                        # Secondary vesting clause: register for cross-clause checking
                        facts.custom_facts[f"vesting_years_{clause.clause_id}"] = (
                            fact.total_vesting_years
                        )
                        facts.custom_facts[f"vesting_cliff_{clause.clause_id}"] = fact.cliff_months

            elif family == ClauseFamily.PROTECTIVE_PROVISIONS:
                fact = self._extract_protective_provisions(clause)
                if fact is not None:
                    if facts.protective_provisions is None:
                        facts.protective_provisions = fact
                    if getattr(fact, "_veto_ops", False):
                        facts.custom_facts["veto_day_to_day_operations"] = True
                        facts.custom_facts["protective_provisions.veto_day_to_day_operations"] = (
                            True
                        )

            elif family == ClauseFamily.DRAG_ALONG:
                self._process_drag_along(clause, facts)

            elif family == ClauseFamily.OPTION_POOL:
                self._process_option_pool(clause, facts)

            elif family == ClauseFamily.REDEMPTION_RIGHTS:
                self._process_redemption_rights(clause, facts)

            elif family == ClauseFamily.EXCLUSIVITY_NO_SHOP:
                self._process_exclusivity(clause, facts)

            elif family == ClauseFamily.ROFR_CO_SALE:
                self._process_rofr_co_sale(clause, facts)

            elif family == ClauseFamily.INFORMATION_RIGHTS:
                self._process_information_rights(clause, facts)

            elif family == ClauseFamily.CONVERSION_RIGHTS:
                self._process_conversion_rights(clause, facts)

            # Global check for cumulative dividends (R-DV-01) in text
            lower_text = clause.raw_text.lower()
            if "cumulative dividend" in lower_text or "dividends shall be cumulative" in lower_text:
                facts.custom_facts["cumulative"] = True
                facts.custom_facts["dividends.cumulative"] = True

        return facts

    def _ground_and_create_span(
        self, clause: ClauseSpan, quote: Optional[str]
    ) -> tuple[SourceSpan, bool]:
        """Validate candidate quote and return (SourceSpan, is_grounded).

        If the quote is missing or cannot be grounded verbatim in clause.raw_text,
        is_grounded is marked False.
        """
        if not quote or not quote.strip():
            span = SourceSpan(
                text=clause.raw_text[: min(60, len(clause.raw_text))],
                page_number=clause.page_number,
            )
            return span, False

        candidate_span = SourceSpan(text=quote.strip(), page_number=clause.page_number)
        val_result = validate_source_span(clause, candidate_span)

        if val_result.is_valid and val_result.resolved_span is not None:
            return val_result.resolved_span, True

        return candidate_span, False

    def _extract_liquidation_preference(
        self, clause: ClauseSpan
    ) -> Optional[LiquidationPreferenceFact]:
        data = None
        mode = "none"

        if self.llm_callable:
            prompt = EXTRACTION_PROMPTS[ClauseFamily.LIQUIDATION_PREFERENCE].format(
                clause_text=clause.raw_text
            )
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for liquidation preference: %s", e)

        if not data:
            data = self._fallback_extract_liquidation(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        if not data:
            return None

        # Guard against fabricated multiples when no multiple is present in data
        raw_mult = data.get("multiple")
        if raw_mult is None:
            # If both LLM and text provide no multiple, do not invent 1.0x
            return None

        try:
            multiple_val = float(raw_mult)
        except (ValueError, TypeError):
            return None

        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)

        # Numerical consistency check against quote text
        if is_grounded and not _quote_contains_numeric(span.text, multiple_val):
            is_grounded = False

        part_raw = (
            str(data.get("participation", "unknown")).lower().replace(" ", "_").replace("-", "_")
        )
        try:
            part = ParticipationType(part_raw)
        except ValueError:
            part = ParticipationType.UNKNOWN

        cap_mult = None
        if data.get("cap_multiple") is not None:
            try:
                cap_mult = float(data["cap_multiple"])
                if is_grounded and not _quote_contains_numeric(span.text, cap_mult):
                    is_grounded = False
            except (ValueError, TypeError):
                cap_mult = None

        return LiquidationPreferenceFact(
            clause_id=clause.clause_id,
            multiple=multiple_val,
            participation=part,
            cap_multiple=cap_mult,
            seniority=data.get("seniority", "pari_passu"),
            source_span=span,
            is_grounded=is_grounded,
        )

    def _extract_anti_dilution(self, clause: ClauseSpan) -> Optional[AntiDilutionFact]:
        data = None
        mode = "none"

        if self.llm_callable:
            prompt = EXTRACTION_PROMPTS[ClauseFamily.ANTI_DILUTION].format(
                clause_text=clause.raw_text
            )
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for anti-dilution: %s", e)

        if not data:
            data = self._fallback_extract_anti_dilution(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        if not data:
            return None

        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)

        ad_raw = (
            str(data.get("anti_dilution_type", "unknown"))
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        try:
            ad_type = AntiDilutionType(ad_raw)
        except ValueError:
            ad_type = AntiDilutionType.UNKNOWN

        return AntiDilutionFact(
            clause_id=clause.clause_id,
            anti_dilution_type=ad_type,
            source_span=span,
            is_grounded=is_grounded,
        )

    def _extract_board_composition(self, clause: ClauseSpan) -> Optional[BoardCompositionFact]:
        data = None
        mode = "none"

        if self.llm_callable:
            prompt = EXTRACTION_PROMPTS[ClauseFamily.BOARD_COMPOSITION].format(
                clause_text=clause.raw_text
            )
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for board composition: %s", e)

        if not data:
            data = self._fallback_extract_board(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        if not data:
            return None

        try:
            f_seats = int(data.get("founder_seats") or 0)
            i_seats = int(data.get("investor_seats") or 0)
            ind_seats = int(data.get("independent_seats") or 0)
            tot_raw = data.get("total_seats")
            tot_seats = int(tot_raw) if tot_raw is not None else (f_seats + i_seats + ind_seats)
        except (ValueError, TypeError):
            return None

        # If all counts are zero, do not fabricate a board fact
        if f_seats == 0 and i_seats == 0 and ind_seats == 0 and tot_seats == 0:
            return None

        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)

        # Verify numerical support in quote
        if is_grounded:
            has_num_support = (
                _quote_contains_numeric(span.text, f_seats)
                or _quote_contains_numeric(span.text, i_seats)
                or _quote_contains_numeric(span.text, tot_seats)
            )
            if not has_num_support:
                is_grounded = False

        inv_majority = i_seats > f_seats

        return BoardCompositionFact(
            clause_id=clause.clause_id,
            founder_seats=f_seats,
            investor_seats=i_seats,
            independent_seats=ind_seats,
            total_seats=tot_seats,
            investor_majority=inv_majority,
            source_span=span,
            is_grounded=is_grounded,
        )

    def _extract_vesting(self, clause: ClauseSpan) -> Optional[VestingFact]:
        data = None
        mode = "none"

        if self.llm_callable:
            prompt = EXTRACTION_PROMPTS[ClauseFamily.FOUNDER_VESTING].format(
                clause_text=clause.raw_text
            )
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for vesting: %s", e)

        if not data:
            data = self._fallback_extract_vesting(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        if not data:
            return None

        years_raw = data.get("total_vesting_years")
        cliff_raw = data.get("cliff_months")

        # Do not invent vesting schedule if both fields are missing
        if years_raw is None and cliff_raw is None:
            return None

        try:
            years = float(years_raw) if years_raw is not None else 4.0
            cliff = int(cliff_raw) if cliff_raw is not None else 0
        except (ValueError, TypeError):
            return None

        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)

        # Numerical consistency check against quote text
        if is_grounded and not _quote_contains_numeric(span.text, years):
            is_grounded = False

        accel = str(data.get("acceleration_type") or "none").lower().replace(" ", "_")
        if accel not in ("none", "single_trigger", "double_trigger"):
            accel = "none"

        return VestingFact(
            clause_id=clause.clause_id,
            total_vesting_years=years,
            cliff_months=cliff,
            acceleration_type=accel,
            source_span=span,
            is_grounded=is_grounded,
        )

    def _extract_protective_provisions(
        self, clause: ClauseSpan
    ) -> Optional[ProtectiveProvisionsFact]:
        data = None
        mode = "none"

        if self.llm_callable:
            prompt = EXTRACTION_PROMPTS[ClauseFamily.PROTECTIVE_PROVISIONS].format(
                clause_text=clause.raw_text
            )
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for protective provisions: %s", e)

        if not data:
            data = self._fallback_extract_protective(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        if not data:
            return None

        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)

        requires_majority = bool(data.get("requires_series_majority", True))
        actions = [str(a) for a in data.get("covered_actions", [])]
        veto_ops = bool(data.get("veto_day_to_day_operations", False))

        fact = ProtectiveProvisionsFact(
            clause_id=clause.clause_id,
            requires_series_majority=requires_majority,
            covered_actions=actions,
            source_span=span,
            is_grounded=is_grounded,
        )
        if veto_ops and is_grounded:
            object.__setattr__(fact, "_veto_ops", True)
        return fact

    # -------------------------------------------------------------
    # Deterministic regex / keyword fallbacks
    # Only produce facts when explicit evidence exists in the clause.
    # -------------------------------------------------------------

    def _fallback_extract_liquidation(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "liquidation" not in lower and "preference" not in lower:
            return None

        mult_match = re.search(r"(\d+(?:\.\d+)?)x", lower)
        # If no multiple is mentioned, return None rather than fabricating 1.0x
        if not mult_match:
            return None
        mult = float(mult_match.group(1))

        if "participating fully" in lower or "fully participating" in lower:
            part = "fully_participating"
        elif "capped" in lower:
            part = "capped_participating"
        elif "non-participating" in lower or "non participating" in lower:
            part = "non_participating"
        else:
            part = "unknown"

        quote = mult_match.group(0)
        phrase_match = re.search(r"(\d+(?:\.\d+)?x\s+liquidation\s+preference)", lower)
        if phrase_match:
            start, end = phrase_match.span()
            quote = text[start:end]

        return {
            "multiple": mult,
            "participation": part,
            "cap_multiple": None,
            "seniority": "pari_passu",
            "quote": quote,
        }

    def _fallback_extract_anti_dilution(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "anti-dilution" not in lower and "ratchet" not in lower and "dilution" not in lower:
            return None

        if "full ratchet" in lower or "full-ratchet" in lower:
            ad_type = "full_ratchet"
            quote_match = re.search(r"full[- ]ratchet(?: anti-dilution)?", text, re.IGNORECASE)
            quote = quote_match.group(0) if quote_match else "full ratchet"
        elif "narrow" in lower:
            ad_type = "narrow_based_weighted_average"
            quote_match = re.search(r"narrow[- ]based", text, re.IGNORECASE)
            quote = quote_match.group(0) if quote_match else "narrow"
        elif "weighted average" in lower or "broad" in lower:
            ad_type = "broad_based_weighted_average"
            quote_match = re.search(r"broad[- ]based weighted average", text, re.IGNORECASE)
            quote = quote_match.group(0) if quote_match else "weighted average"
        else:
            ad_type = "unknown"
            quote_match = re.search(r"anti[- ]dilution", text, re.IGNORECASE)
            quote = quote_match.group(0) if quote_match else text[: min(30, len(text))]

        return {
            "anti_dilution_type": ad_type,
            "quote": quote,
        }

    def _fallback_extract_board(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "board" not in lower and "director" not in lower:
            return None

        tot_match = re.search(r"(\d+)\s*(?:total\s+)?(?:member|director|seat)s?", lower)
        f_match = re.search(
            r"(\d+)\s*(?:seat|member|director)?s?\s*(?:designated|appointed|held)?\s*(?:by\s+)?(?:the\s+)?(?:founder|common)",
            lower,
        )
        i_match = re.search(
            r"(\d+)\s*(?:seat|member|director)?s?\s*(?:designated|appointed|held)?\s*(?:by\s+)?(?:the\s+)?(?:investor|preferred|series)",
            lower,
        )

        # Do not fabricate board seats if no numbers exist
        if not tot_match and not f_match and not i_match:
            return None

        f_seats = int(f_match.group(1)) if f_match else 0
        i_seats = int(i_match.group(1)) if i_match else 0
        tot_seats = int(tot_match.group(1)) if tot_match else (f_seats + i_seats)
        ind_seats = max(0, tot_seats - f_seats - i_seats)

        quote = None
        if i_match:
            start, end = i_match.span()
            quote = text[start:end]
        elif tot_match:
            start, end = tot_match.span()
            quote = text[start:end]
        elif f_match:
            start, end = f_match.span()
            quote = text[start:end]

        return {
            "founder_seats": f_seats,
            "investor_seats": i_seats,
            "independent_seats": ind_seats,
            "total_seats": tot_seats,
            "quote": quote or text[: min(50, len(text))],
        }

    def _fallback_extract_vesting(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "vesting" not in lower and "cliff" not in lower:
            return None

        years_match = re.search(r"(\d+(?:\.\d+)?)[- ]year", lower)
        cliff_match = re.search(r"(\d+)[- ]month", lower)

        # Do not fabricate a 4-year schedule if no duration is stated
        if not years_match and not cliff_match:
            return None

        years = float(years_match.group(1)) if years_match else 4.0
        cliff = int(cliff_match.group(1)) if cliff_match else 0

        accel = "none"
        if "double" in lower and "trigger" in lower:
            accel = "double_trigger"
        elif "single" in lower and "trigger" in lower:
            accel = "single_trigger"

        quote_match = re.search(r"(\d+[- ]year vesting|\d+[- ]month cliff)", text, re.IGNORECASE)
        quote = quote_match.group(0) if quote_match else text[: min(50, len(text))]

        return {
            "total_vesting_years": years,
            "cliff_months": cliff,
            "acceleration_type": accel,
            "quote": quote,
        }

    def _fallback_extract_protective(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if (
            "protective" not in lower
            and "veto" not in lower
            and "consent" not in lower
            and "approval" not in lower
            and "negative covenants" not in lower
        ):
            return None

        requires_majority = "majority" in lower or "consent" in lower or "approval" in lower
        actions = []
        if "sale" in lower or "liquidation" in lower or "merger" in lower:
            actions.append("liquidation_or_sale")
        if "shares" in lower or "class" in lower or "equity" in lower:
            actions.append("authorization_of_senior_shares")

        veto_ops = (
            "day-to-day" in lower
            or "day to day" in lower
            or "operational decisions" in lower
            or "ordinary course" in lower
            or "operating budget" in lower
            or "hiring" in lower
        )
        if veto_ops:
            actions.append("veto_day_to_day_operations")

        quote_match = re.search(
            r"(?:require\w*\s+(?:the\s+)?(?:vote|consent|approval))", text, re.IGNORECASE
        )
        quote = quote_match.group(0) if quote_match else text[: min(50, len(text))]

        return {
            "requires_series_majority": requires_majority,
            "covered_actions": actions,
            "veto_day_to_day_operations": veto_ops,
            "quote": quote,
        }

    # -------------------------------------------------------------
    # Handlers for extended clause families (custom_facts)
    # -------------------------------------------------------------

    def _extract_generic_family(
        self,
        clause: ClauseSpan,
        family: ClauseFamily,
        fallback_fn: Callable[[str], Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Extract attributes for an extended clause family via LLM or deterministic fallback."""
        data = None
        mode = "none"

        if self.llm_callable and family in EXTRACTION_PROMPTS:
            prompt = EXTRACTION_PROMPTS[family].format(clause_text=clause.raw_text)
            try:
                raw_resp = self.llm_callable(prompt)
                data = _extract_json_block(raw_resp)
                if data:
                    mode = "llm"
            except Exception as e:
                logger.warning("LLM call failed for %s: %s", family.value, e)

        if not data:
            data = fallback_fn(clause.raw_text)
            if data:
                mode = "fallback"

        self.extraction_modes[clause.clause_id] = mode
        return data

    def _fallback_extract_drag_along(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "drag" not in lower:
            return None
        threshold_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
        threshold = float(threshold_match.group(1)) if threshold_match else None

        if "without" in lower and "founder" in lower and "consent" in lower:
            requires_consent = False
        elif (
            "founder consent" in lower
            or "consent of the founder" in lower
            or "founders' consent" in lower
            or "founders consent" in lower
        ):
            requires_consent = True
        elif "majority of common" in lower:
            requires_consent = True
        else:
            requires_consent = False if threshold is not None else None

        quote = None
        if threshold_match:
            quote = threshold_match.group(0)
        else:
            m = re.search(r"drag[- ]along[^\.\;]*", text, re.IGNORECASE)
            quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "approval_threshold": threshold,
            "requires_founder_consent": requires_consent,
            "quote": quote,
        }

    def _process_drag_along(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.DRAG_ALONG, self._fallback_extract_drag_along
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        threshold = data.get("approval_threshold")
        if threshold is not None:
            try:
                threshold_val = float(threshold)
                if is_grounded and not _quote_contains_numeric(span.text, threshold_val):
                    is_grounded = False
                if is_grounded:
                    facts.custom_facts["approval_threshold"] = threshold_val
                    facts.custom_facts["drag_along.approval_threshold"] = threshold_val
            except (ValueError, TypeError):
                pass
        consent = data.get("requires_founder_consent")
        if consent is not None and is_grounded:
            consent_val = bool(consent)
            facts.custom_facts["requires_founder_consent"] = consent_val
            facts.custom_facts["drag_along.requires_founder_consent"] = consent_val

    def _fallback_extract_option_pool(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if (
            "option pool" not in lower
            and "equity incentive" not in lower
            and "unallocated" not in lower
            and "pool" not in lower
            and "esop" not in lower
        ):
            return None
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
        pct = float(pct_match.group(1)) if pct_match else None

        pre_money = None
        if (
            "pre-money" in lower
            or "pre money" in lower
            or "prior to the closing" in lower
            or "prior to investment" in lower
        ):
            pre_money = True
        elif "post-money" in lower or "post money" in lower or "following the closing" in lower:
            pre_money = False

        quote = None
        if pct_match:
            quote = pct_match.group(0)
        else:
            m = re.search(
                r"(?:option\s+pool|incentive\s+pool|unallocated)[^\.\;]*", text, re.IGNORECASE
            )
            quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "pool_percentage": pct,
            "pre_money_pool": pre_money,
            "quote": quote,
        }

    def _process_option_pool(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.OPTION_POOL, self._fallback_extract_option_pool
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        pct = data.get("pool_percentage")
        if pct is not None:
            try:
                pct_val = float(pct)
                if is_grounded and not _quote_contains_numeric(span.text, pct_val):
                    is_grounded = False
                if is_grounded:
                    facts.custom_facts["pool_percentage"] = pct_val
                    facts.custom_facts["option_pool.pool_percentage"] = pct_val
            except (ValueError, TypeError):
                pass
        pre = data.get("pre_money_pool")
        if pre is not None and is_grounded:
            pre_val = bool(pre)
            facts.custom_facts["pre_money_pool"] = pre_val
            facts.custom_facts["option_pool.pre_money_pool"] = pre_val

    def _fallback_extract_redemption(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "redemption" not in lower and "redeem" not in lower:
            return None
        years_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:[- ]year|years)", lower)
        years = float(years_match.group(1)) if years_match else None

        interest_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:annual|dividend|interest|per annum)", lower
        )
        interest = float(interest_match.group(1)) if interest_match else None

        quote = None
        if years_match:
            quote = years_match.group(0)
        else:
            m = re.search(r"redemption[^\.\;]*", text, re.IGNORECASE)
            quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "redemption_years": years,
            "interest_rate": interest,
            "quote": quote,
        }

    def _process_redemption_rights(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.REDEMPTION_RIGHTS, self._fallback_extract_redemption
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        years = data.get("redemption_years")
        if years is not None:
            try:
                years_val = float(years)
                if is_grounded and not _quote_contains_numeric(span.text, years_val):
                    is_grounded = False
                if is_grounded:
                    facts.custom_facts["redemption_years"] = years_val
                    facts.custom_facts["redemption.redemption_years"] = years_val
            except (ValueError, TypeError):
                pass
        interest = data.get("interest_rate")
        if interest is not None:
            try:
                interest_val = float(interest)
                if is_grounded and _quote_contains_numeric(span.text, interest_val):
                    facts.custom_facts["interest_rate"] = interest_val
                    facts.custom_facts["redemption.interest_rate"] = interest_val
            except (ValueError, TypeError):
                pass

    def _fallback_extract_exclusivity(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "exclusivity" not in lower and "no-shop" not in lower and "no shop" not in lower:
            return None
        days_match = re.search(r"(\d+)\s*(?:[- ]day|days)", lower)
        days = int(days_match.group(1)) if days_match else None

        quote = None
        if days_match:
            quote = days_match.group(0)
        else:
            m = re.search(r"(?:exclusivity|no[- ]shop)[^\.\;]*", text, re.IGNORECASE)
            quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "duration_days": days,
            "quote": quote,
        }

    def _process_exclusivity(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.EXCLUSIVITY_NO_SHOP, self._fallback_extract_exclusivity
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        days = data.get("duration_days")
        if days is not None:
            try:
                days_val = int(days)
                if is_grounded and not _quote_contains_numeric(span.text, days_val):
                    is_grounded = False
                if is_grounded:
                    facts.custom_facts["duration_days"] = days_val
                    facts.custom_facts["exclusivity.duration_days"] = days_val
            except (ValueError, TypeError):
                pass

    def _fallback_extract_rofr(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if (
            "first refusal" not in lower
            and "rofr" not in lower
            and "co-sale" not in lower
            and "tag-along" not in lower
            and "tag along" not in lower
        ):
            return None
        co_sale = (
            "co-sale" in lower or "co sale" in lower or "tag-along" in lower or "tag along" in lower
        )
        rofr = "first refusal" in lower or "rofr" in lower

        m = re.search(
            r"(?:right of first refusal|co[- ]sale|tag[- ]along)[^\.\;]*", text, re.IGNORECASE
        )
        quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "investor_co_sale": co_sale,
            "investor_rofr": rofr,
            "quote": quote,
        }

    def _process_rofr_co_sale(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.ROFR_CO_SALE, self._fallback_extract_rofr
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        if is_grounded:
            if data.get("investor_co_sale") is not None:
                val = bool(data["investor_co_sale"])
                facts.custom_facts["investor_co_sale"] = val
                facts.custom_facts["rofr_co_sale.investor_co_sale"] = val
            if data.get("investor_rofr") is not None:
                val = bool(data["investor_rofr"])
                facts.custom_facts["investor_rofr"] = val
                facts.custom_facts["rofr_co_sale.investor_rofr"] = val

    def _fallback_extract_information_rights(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if (
            "information" not in lower
            and "financial" not in lower
            and "inspection" not in lower
            and "reporting" not in lower
        ):
            return None
        days_match = re.search(r"(\d+)\s*(?:[- ]day|days)", lower)
        days = int(days_match.group(1)) if days_match else None
        inspection = "inspection" in lower or "visit" in lower or "inspect" in lower

        quote = None
        if days_match:
            quote = days_match.group(0)
        else:
            m = re.search(
                r"(?:information rights|financial statements)[^\.\;]*", text, re.IGNORECASE
            )
            quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "financial_statements_days": days,
            "inspection_rights": inspection,
            "quote": quote,
        }

    def _process_information_rights(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.INFORMATION_RIGHTS, self._fallback_extract_information_rights
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        days = data.get("financial_statements_days")
        if days is not None:
            try:
                days_val = int(days)
                if is_grounded and not _quote_contains_numeric(span.text, days_val):
                    is_grounded = False
                if is_grounded:
                    facts.custom_facts["financial_statements_days"] = days_val
                    facts.custom_facts["information_rights.financial_statements_days"] = days_val
            except (ValueError, TypeError):
                pass
        insp = data.get("inspection_rights")
        if insp is not None and is_grounded:
            val = bool(insp)
            facts.custom_facts["inspection_rights"] = val
            facts.custom_facts["information_rights.inspection_rights"] = val

    def _fallback_extract_conversion(self, text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if "conversion" not in lower and "convertible" not in lower and "convert" not in lower:
            return None
        optional = "optional" in lower or "at the option" in lower or "at any time" in lower
        ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", lower)
        ratio = float(ratio_match.group(1)) if ratio_match else 1.0

        m = re.search(r"conversion[^\.\;]*", text, re.IGNORECASE)
        quote = m.group(0) if m else text[: min(50, len(text))]

        return {
            "conversion_ratio": ratio,
            "optional_conversion": optional,
            "quote": quote,
        }

    def _process_conversion_rights(self, clause: ClauseSpan, facts: ExtractedFacts) -> None:
        data = self._extract_generic_family(
            clause, ClauseFamily.CONVERSION_RIGHTS, self._fallback_extract_conversion
        )
        if not data:
            return
        quote = data.get("quote")
        span, is_grounded = self._ground_and_create_span(clause, quote)
        ratio = data.get("conversion_ratio")
        if ratio is not None:
            try:
                ratio_val = float(ratio)
                if is_grounded:
                    facts.custom_facts["conversion_ratio"] = ratio_val
                    facts.custom_facts["conversion_rights.conversion_ratio"] = ratio_val
            except (ValueError, TypeError):
                pass
        opt = data.get("optional_conversion")
        if opt is not None and is_grounded:
            val = bool(opt)
            facts.custom_facts["optional_conversion"] = val
            facts.custom_facts["conversion_rights.optional_conversion"] = val


def create_fact_extractor(
    use_gemini: bool = False,
    use_cache: bool = True,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> FactExtractor:
    """Factory creating a FactExtractor wired to Gemini or rule-based fallback.

    Parameters
    ----------
    use_gemini:
        If True, initializes Gemini via ``make_gemini_callable`` with optional caching.
        If False, uses deterministic extraction only.
    use_cache:
        If True and use_gemini is True, wraps the Gemini callable with ``make_cached_callable``.
    model:
        Optional Gemini model override (defaults to gemini-3.5-flash).
    api_key:
        Optional explicit API key override (defaults to GEMINI_API_KEY from .env).
    """
    if use_gemini:
        gemini_callable = make_gemini_callable(model=model, api_key=api_key)
        llm = make_cached_callable(gemini_callable) if use_cache else gemini_callable
        return FactExtractor(llm_callable=llm)
    return FactExtractor(llm_callable=None)
