"""Gemini REST API client for TermSheet AI (Stage S3).

This module provides a factory function ``make_gemini_callable`` that returns a
callable matching the ``LLMCallable`` protocol expected by ``FactExtractor``.

The key is loaded from the ``GEMINI_API_KEY`` environment variable (set via
``.env`` in the project root). It is **never** logged, printed, or embedded in
source code.

Usage
-----
>>> from termsheet_ai.extract.llm_client import make_gemini_callable
>>> call = make_gemini_callable()  # uses gemini-3.5-flash by default, timeout=90s
>>> response_text = call("Your prompt here")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions for clear, actionable failure reporting
# ---------------------------------------------------------------------------


class GeminiAPIError(RuntimeError):
    """Base exception for Gemini API errors."""


class GeminiQuotaError(GeminiAPIError):
    """Raised when Gemini API quota or rate limit is reached (HTTP 429)."""


class GeminiModelNotFoundError(GeminiAPIError):
    """Raised when the specified Gemini model is not found or unsupported (HTTP 404)."""


class GeminiAuthenticationError(GeminiAPIError):
    """Raised when API key is missing, invalid, or unauthorized (HTTP 401/403)."""


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_MODEL = "gemini-3.5-flash"
_DEFAULT_TIMEOUT_SECONDS = 90.0  # sensible default for extraction prompts on free/flash tier
_TIMEOUT_SECONDS = _DEFAULT_TIMEOUT_SECONDS  # backward-compatible alias
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_INITIAL_BACKOFF = 0.5


def _resolve_timeout(timeout: Optional[float] = None) -> float:
    """Resolve request timeout in seconds.

    Priority:
    1. Explicit ``timeout`` argument passed to ``make_gemini_callable``
    2. ``GEMINI_TIMEOUT_SECONDS`` environment variable
    3. ``GEMINI_TIMEOUT`` environment variable
    4. Sensible default of 90.0 seconds
    """
    if timeout is not None:
        return float(timeout)

    env_val = os.getenv("GEMINI_TIMEOUT_SECONDS") or os.getenv("GEMINI_TIMEOUT")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            logger.warning(
                "Invalid GEMINI_TIMEOUT_SECONDS value '%s'; using default %.1fs.",
                env_val,
                _DEFAULT_TIMEOUT_SECONDS,
            )
    return _DEFAULT_TIMEOUT_SECONDS


def _resolve_max_retries(max_retries: Optional[int] = None) -> int:
    """Resolve max retries for transient errors (parameter > GEMINI_MAX_RETRIES > 2)."""
    if max_retries is not None:
        return max(0, int(max_retries))
    env_val = os.getenv("GEMINI_MAX_RETRIES")
    if env_val:
        try:
            return max(0, int(env_val))
        except ValueError:
            pass
    return _DEFAULT_MAX_RETRIES


def _resolve_backoff(initial_backoff: Optional[float] = None) -> float:
    """Resolve initial backoff delay in seconds."""
    if initial_backoff is not None:
        return max(0.0, float(initial_backoff))
    env_val = os.getenv("GEMINI_INITIAL_BACKOFF")
    if env_val:
        try:
            return max(0.0, float(env_val))
        except ValueError:
            pass
    return _DEFAULT_INITIAL_BACKOFF


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_gemini_callable(
    model: Optional[str] = None,
    temperature: float = 0.0,
    *,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    initial_backoff: Optional[float] = None,
) -> "LLMCallable":  # noqa: F821  (avoid circular import; type hint only)
    """Return a callable ``(prompt: str) -> str`` backed by the Gemini REST API.

    Parameters
    ----------
    model:
        Gemini model name. Defaults to ``"gemini-3.5-flash"`` (or the
        ``GEMINI_MODEL`` environment variable). Only free-tier models should be used.
    temperature:
        Sampling temperature. Defaults to ``0.0`` for deterministic,
        structured extraction output.
    api_key:
        Optional override for the API key. When *None* (the default) the key
        is read from the ``GEMINI_API_KEY`` environment variable after loading
        ``.env``.
    timeout:
        Request timeout in seconds. When omitted or None, resolves from the
        ``GEMINI_TIMEOUT_SECONDS`` / ``GEMINI_TIMEOUT`` environment variable,
        defaulting to 90.0 seconds.
    max_retries:
        Number of retry attempts for transient timeout and connection errors.
        Defaults to 2 (total 3 attempts). Configurable via ``GEMINI_MAX_RETRIES``.
    initial_backoff:
        Initial backoff delay in seconds before first retry.
        Defaults to 0.5s. Configurable via ``GEMINI_INITIAL_BACKOFF``.

    Returns
    -------
    callable
        A function ``call(prompt: str) -> str`` that posts the prompt to
        Gemini and returns the model's text reply.

    Raises
    ------
    EnvironmentError
        If no API key can be resolved at call time.
    GeminiQuotaError
        If the Gemini API returns HTTP 429 (quota or rate-limit exceeded).
        Never retried.
    GeminiModelNotFoundError
        If the model is deprecated or not found (HTTP 404). Never retried.
    GeminiAuthenticationError
        If the API key is unauthorized or forbidden (HTTP 401/403). Never retried.
    """
    load_dotenv()

    resolved_model = model or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL
    resolved_timeout = _resolve_timeout(timeout)
    resolved_retries = _resolve_max_retries(max_retries)
    resolved_backoff = _resolve_backoff(initial_backoff)

    # _use_forced: when an explicit api_key is supplied (even ""), use it
    # directly and skip os.getenv() so tests can safely override.
    _use_forced: bool = api_key is not None
    _forced_key: str = api_key if api_key is not None else ""

    def _call(prompt: str) -> str:
        """Send *prompt* to Gemini and return the response text."""
        if _use_forced:
            key: str = _forced_key
        else:
            key = os.getenv("GEMINI_API_KEY") or ""

        if not key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

        url = f"{_API_BASE}/{resolved_model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }

        total_attempts = resolved_retries + 1
        response: Optional[httpx.Response] = None

        for attempt in range(total_attempts):
            try:
                response = httpx.post(
                    url,
                    params={"key": key},
                    json=payload,
                    timeout=resolved_timeout,
                )
                response.raise_for_status()
                # Request succeeded, break out of retry loop
                break

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                err_text = exc.response.text

                # 1. Quota / Rate limit (HTTP 429) - Fatal, do not retry
                if status == 429:
                    logger.error(
                        "Gemini API quota or rate limit reached (HTTP 429) for model %s: %s",
                        resolved_model,
                        err_text[:200],
                    )
                    raise GeminiQuotaError(
                        f"Gemini API quota or rate limit exceeded (HTTP 429) for model '{resolved_model}'. "
                        f"Free tier has 15 RPM / 1500 RPD. Check Google AI Studio (https://aistudio.google.com/) "
                        f"to view your quota and reset times."
                    ) from exc

                # 2. Model not found (HTTP 404) - Fatal, do not retry
                if status == 404:
                    logger.error(
                        "Gemini model %s not found or unsupported (HTTP 404): %s",
                        resolved_model,
                        err_text[:200],
                    )
                    raise GeminiModelNotFoundError(
                        f"Gemini model '{resolved_model}' not found or unsupported (HTTP 404). "
                        f"Verify model name against available free models."
                    ) from exc

                # 3. Authentication (HTTP 401 / 403) - Fatal, do not retry
                if status in (401, 403):
                    logger.error(
                        "Gemini API authentication failed (HTTP %s): %s",
                        status,
                        err_text[:200],
                    )
                    raise GeminiAuthenticationError(
                        f"Gemini API authentication failed (HTTP {status}). "
                        f"Check that your API key is valid in Google AI Studio."
                    ) from exc

                # 4. Other HTTP status errors (e.g. 400 Bad Request, 5xx) - Do not retry
                logger.error(
                    "Gemini API HTTP client/server error %s for model %s: %s",
                    status,
                    resolved_model,
                    err_text[:200],
                )
                return ""

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                # Transient timeout or connection error: retry with exponential backoff
                exc_name = type(exc).__name__
                if attempt < total_attempts - 1:
                    sleep_time = resolved_backoff * (2**attempt)
                    logger.warning(
                        "Gemini API %s on attempt %d/%d for model %s (timeout=%.1fs). Retrying in %.2fs: %s",
                        exc_name,
                        attempt + 1,
                        total_attempts,
                        resolved_model,
                        resolved_timeout,
                        sleep_time,
                        exc,
                    )
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    logger.error(
                        "Gemini API %s for model %s (timeout=%.1fs) exhausted all %d attempts: %s",
                        exc_name,
                        resolved_model,
                        resolved_timeout,
                        total_attempts,
                        exc,
                    )
                    return ""

            except httpx.RequestError as exc:
                # Non-transient transport/request error: do not retry
                logger.error(
                    "Gemini API non-retryable transport error for model %s: %s",
                    resolved_model,
                    exc,
                )
                return ""

        if response is None:
            return ""

        try:
            data = response.json()
            candidates = data.get("candidates")
            if not candidates:
                feedback = data.get("promptFeedback", {})
                logger.warning(
                    "Gemini API returned no candidates for model %s. Prompt feedback: %s",
                    resolved_model,
                    feedback,
                )
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "")
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning(
                "Gemini API unexpected response structure for model %s: %s",
                resolved_model,
                exc,
            )
            return ""

    return _call
