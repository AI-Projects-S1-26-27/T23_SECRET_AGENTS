"""Tests for the Gemini client and LLM cache.

This test module contains two distinct suites:
1. **Offline unit tests**: Fast, deterministic tests that mock all HTTP traffic
   and file I/O. They run completely offline without network access or API credentials.
2. **Live integration tests**: Real HTTP calls to the Gemini REST API. Marked with
   ``@pytest.mark.live``. Run only when outbound network access and ``GEMINI_API_KEY``
   are present.
"""

from __future__ import annotations

import json
import os
import socket
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config):
    """Register custom markers dynamically to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers", "live: marks tests requiring real outbound network and API credentials"
    )


# ---------------------------------------------------------------------------
# Network availability check for live integration tests
# ---------------------------------------------------------------------------

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))


def _network_available() -> bool:
    """Return True if the Gemini API hostname can be resolved."""
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
        return True
    except (socket.gaierror, OSError):
        return False


_NETWORK_OK = _network_available()

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from termsheet_ai.extract.llm_cache import (  # noqa: E402
    _hash_prompt,
    make_cached_callable,
)
from termsheet_ai.extract.llm_client import (  # noqa: E402
    GeminiAuthenticationError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
    make_gemini_callable,
)

# ===========================================================================
# 1. Offline Unit Tests for Gemini Client (No Network Required)
# ===========================================================================


class TestMakeGeminiCallableUnit:
    """Offline unit tests for make_gemini_callable with mocked HTTP layer."""

    def test_returns_callable(self):
        """make_gemini_callable() must return a callable object."""
        fn = make_gemini_callable()
        assert callable(fn)

    def test_raises_when_no_key(self):
        """Calling the returned function with an empty key raises EnvironmentError."""
        fn = make_gemini_callable(api_key="")
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            fn("test prompt")

    def test_quota_error_raises_gemini_quota_error(self):
        """HTTP 429 must raise GeminiQuotaError with clear actionable message."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        mock_resp = httpx.Response(
            429,
            request=mock_req,
            text='{"error": {"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}',
        )

        def _raise_429(*args, **kwargs):
            raise httpx.HTTPStatusError("quota exceeded", request=mock_req, response=mock_resp)

        with patch("httpx.post", side_effect=_raise_429):
            fn = make_gemini_callable(api_key="mock-key")
            with pytest.raises(GeminiQuotaError, match="429"):
                fn("extract this")

    def test_model_not_found_raises_gemini_model_error(self):
        """HTTP 404 must raise GeminiModelNotFoundError."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        mock_resp = httpx.Response(
            404,
            request=mock_req,
            text='{"error": {"code": 404, "message": "Model not found"}}',
        )

        def _raise_404(*args, **kwargs):
            raise httpx.HTTPStatusError("not found", request=mock_req, response=mock_resp)

        with patch("httpx.post", side_effect=_raise_404):
            fn = make_gemini_callable(model="nonexistent-model", api_key="mock-key")
            with pytest.raises(GeminiModelNotFoundError, match="404"):
                fn("extract this")

    def test_auth_error_raises_gemini_auth_error(self):
        """HTTP 401 or 403 must raise GeminiAuthenticationError."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        mock_resp = httpx.Response(
            403,
            request=mock_req,
            text='{"error": {"code": 403, "message": "API key not valid"}}',
        )

        def _raise_403(*args, **kwargs):
            raise httpx.HTTPStatusError("forbidden", request=mock_req, response=mock_resp)

        with patch("httpx.post", side_effect=_raise_403):
            fn = make_gemini_callable(api_key="invalid-key")
            with pytest.raises(GeminiAuthenticationError, match="authentication failed"):
                fn("extract this")

    def test_generic_server_error_returns_empty_string(self):
        """Transient HTTP 5xx errors return empty string to allow fallback."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        mock_resp = httpx.Response(500, request=mock_req, text="Internal Server Error")

        def _raise_500(*args, **kwargs):
            raise httpx.HTTPStatusError("server error", request=mock_req, response=mock_resp)

        with patch("httpx.post", side_effect=_raise_500):
            fn = make_gemini_callable(api_key="mock-key")
            result = fn("hello")

        assert result == ""

    def test_network_error_returns_empty_string(self):
        """A network connection error returns empty string to allow fallback."""
        import httpx

        with (
            patch("httpx.post", side_effect=httpx.ConnectError("connection refused")),
            patch("time.sleep"),
        ):
            fn = make_gemini_callable(api_key="mock-key")
            result = fn("hello")

        assert result == ""

    def test_default_timeout_is_90_seconds(self):
        """Default request timeout must be 90.0 seconds."""
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            fn = make_gemini_callable(api_key="mock-key")
            fn("hello")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 90.0

    def test_timeout_from_environment_variable(self, monkeypatch):
        """GEMINI_TIMEOUT_SECONDS environment variable configures timeout."""
        import httpx

        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "120.0")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            fn = make_gemini_callable(api_key="mock-key")
            fn("hello")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 120.0

    def test_timeout_parameter_overrides_env_var(self, monkeypatch):
        """Explicit timeout parameter takes precedence over environment variable."""
        import httpx

        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "120.0")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            fn = make_gemini_callable(api_key="mock-key", timeout=45.0)
            fn("hello")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 45.0

    def test_retry_on_timeout_recovers_on_second_attempt(self):
        """Transient ReadTimeout triggers backoff retry and succeeds on attempt 2."""
        import httpx

        mock_success = MagicMock(spec=httpx.Response)
        mock_success.raise_for_status.return_value = None
        mock_success.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "recovered"}]}}]
        }

        mock_req = httpx.Request("POST", "https://example.com")
        effects = [httpx.ReadTimeout("timed out", request=mock_req), mock_success]

        with (
            patch("httpx.post", side_effect=effects) as mock_post,
            patch("time.sleep") as mock_sleep,
        ):
            fn = make_gemini_callable(api_key="mock-key", max_retries=2, initial_backoff=0.5)
            result = fn("extract")

        assert result == "recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)

    def test_retry_on_connect_error_recovers_on_second_attempt(self):
        """Transient ConnectError triggers backoff retry and succeeds on attempt 2."""
        import httpx

        mock_success = MagicMock(spec=httpx.Response)
        mock_success.raise_for_status.return_value = None
        mock_success.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "connected"}]}}]
        }

        effects = [httpx.ConnectError("connection refused"), mock_success]

        with (
            patch("httpx.post", side_effect=effects) as mock_post,
            patch("time.sleep") as mock_sleep,
        ):
            fn = make_gemini_callable(api_key="mock-key", max_retries=2, initial_backoff=0.5)
            result = fn("extract")

        assert result == "connected"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(0.5)

    def test_exhausted_retries_returns_empty_string(self):
        """When transient timeout retries are exhausted, returns empty string."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        with (
            patch(
                "httpx.post", side_effect=httpx.ReadTimeout("timed out", request=mock_req)
            ) as mock_post,
            patch("time.sleep"),
        ):
            fn = make_gemini_callable(api_key="mock-key", max_retries=2)
            result = fn("extract")

        assert result == ""
        # 1 initial call + 2 retries = 3 calls
        assert mock_post.call_count == 3

    def test_fatal_errors_do_not_retry(self):
        """401/403, 404, 429, and 400 are never retried."""
        import httpx

        mock_req = httpx.Request("POST", "https://example.com")
        mock_resp_401 = httpx.Response(401, request=mock_req, text="Unauthorized")
        mock_resp_404 = httpx.Response(404, request=mock_req, text="Not Found")
        mock_resp_429 = httpx.Response(429, request=mock_req, text="Rate limit")
        mock_resp_400 = httpx.Response(400, request=mock_req, text="Bad Request")

        # 401: exactly 1 call
        with patch(
            "httpx.post",
            side_effect=httpx.HTTPStatusError("401", request=mock_req, response=mock_resp_401),
        ) as p:
            fn = make_gemini_callable(api_key="mock-key")
            with pytest.raises(GeminiAuthenticationError):
                fn("prompt")
            assert p.call_count == 1

        # 404: exactly 1 call
        with patch(
            "httpx.post",
            side_effect=httpx.HTTPStatusError("404", request=mock_req, response=mock_resp_404),
        ) as p:
            fn = make_gemini_callable(api_key="mock-key")
            with pytest.raises(GeminiModelNotFoundError):
                fn("prompt")
            assert p.call_count == 1

        # 429: exactly 1 call
        with patch(
            "httpx.post",
            side_effect=httpx.HTTPStatusError("429", request=mock_req, response=mock_resp_429),
        ) as p:
            fn = make_gemini_callable(api_key="mock-key")
            with pytest.raises(GeminiQuotaError):
                fn("prompt")
            assert p.call_count == 1

        # 400: exactly 1 call, returns ""
        with patch(
            "httpx.post",
            side_effect=httpx.HTTPStatusError("400", request=mock_req, response=mock_resp_400),
        ) as p:
            fn = make_gemini_callable(api_key="mock-key")
            res = fn("prompt")
            assert res == ""
            assert p.call_count == 1

    def test_malformed_response_returns_empty_string(self):
        """A response missing candidates returns empty string."""
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"promptFeedback": {"blockReason": "SAFETY"}}

        with patch("httpx.post", return_value=mock_response):
            fn = make_gemini_callable(api_key="mock-key")
            result = fn("hello")

        assert result == ""

    def test_successful_mock_response(self):
        """A well-formed mock response returns candidate text."""
        import httpx

        expected_text = '{"multiple": 1.0}'
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": expected_text}]}}]
        }

        with patch("httpx.post", return_value=mock_response):
            fn = make_gemini_callable(api_key="mock-key")
            result = fn("some prompt")

        assert result == expected_text


# ===========================================================================
# 2. Offline Unit Tests for LLM Cache (No Network Required)
# ===========================================================================


class TestLLMCache:
    """Unit tests for make_cached_callable and file-based caching helpers."""

    def test_hash_prompt_is_deterministic(self):
        """Same prompt must always produce the same SHA-256 hash."""
        h1 = _hash_prompt("liquidation preference clause")
        h2 = _hash_prompt("liquidation preference clause")
        assert h1 == h2

    def test_hash_prompt_differs_on_different_input(self):
        """Different prompts must produce distinct hashes."""
        assert _hash_prompt("prompt A") != _hash_prompt("prompt B")

    def test_cache_hit_does_not_call_inner(self, tmp_path):
        """When a cached response exists, the inner callable must not be invoked."""
        inner = MagicMock(return_value="should not be called")
        cache_file = tmp_path / "cache.json"

        seed_prompt = "cached prompt"
        seed_hash = _hash_prompt(seed_prompt)
        cache_file.write_text(json.dumps({seed_hash: "cached response"}), encoding="utf-8")

        fn = make_cached_callable(inner, cache_path=cache_file)
        result = fn(seed_prompt)

        assert result == "cached response"
        inner.assert_not_called()

    def test_cache_miss_calls_inner_and_stores(self, tmp_path):
        """On cache miss, the inner callable is called and the result persisted."""
        inner = MagicMock(return_value="fresh response")
        cache_file = tmp_path / "cache.json"

        fn = make_cached_callable(inner, cache_path=cache_file)
        result = fn("new prompt")

        assert result == "fresh response"
        inner.assert_called_once_with("new prompt")

        assert cache_file.exists()
        stored = json.loads(cache_file.read_text(encoding="utf-8"))
        assert _hash_prompt("new prompt") in stored
        assert stored[_hash_prompt("new prompt")] == "fresh response"

    def test_empty_response_not_cached(self, tmp_path):
        """An empty response from a transient failure must never be cached."""
        inner = MagicMock(return_value="")
        cache_file = tmp_path / "cache.json"

        fn = make_cached_callable(inner, cache_path=cache_file)
        result = fn("failing prompt")

        assert result == ""
        if cache_file.exists():
            stored = json.loads(cache_file.read_text(encoding="utf-8"))
            assert stored == {}

    def test_second_call_uses_cache(self, tmp_path):
        """A second identical call must return from cache with only one inner call."""
        inner = MagicMock(return_value="response")
        cache_file = tmp_path / "cache.json"

        fn = make_cached_callable(inner, cache_path=cache_file)
        fn("repeat prompt")
        fn("repeat prompt")

        assert inner.call_count == 1

    def test_corrupted_cache_file_handled_gracefully(self, tmp_path):
        """A corrupted cache file is treated as empty without crashing."""
        inner = MagicMock(return_value="ok")
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("invalid json {{{", encoding="utf-8")

        fn = make_cached_callable(inner, cache_path=cache_file)
        result = fn("prompt")

        assert result == "ok"
        inner.assert_called_once()


# ===========================================================================
# 3. Live Integration Tests (Requires Network and GEMINI_API_KEY)
# ===========================================================================


@pytest.mark.live
@pytest.mark.skipif(
    not _HAS_KEY,
    reason="GEMINI_API_KEY not configured — live Gemini test skipped",
)
@pytest.mark.skipif(
    not _NETWORK_OK,
    reason="No outbound network access — live Gemini test skipped",
)
class TestGeminiLiveConnectivity:
    """Real HTTP calls to the Gemini API (gemini-3.5-flash)."""

    _TEST_CLAUSE = (
        "The Series A Preferred Stock shall have a non-participating liquidation "
        "preference equal to 1.0x the original issue price, plus any declared "
        "but unpaid dividends."
    )

    _TEST_PROMPT = (
        "You are a legal term-sheet analyst. Extract the liquidation preference "
        "multiple from the following clause as JSON with a single key 'multiple' "
        "whose value is a number (or null if not found).\n\n"
        f"Clause: {_TEST_CLAUSE}"
    )

    def test_live_json_extraction(self):
        """Verify real Gemini API connectivity and valid JSON response."""
        fn = make_gemini_callable(model="gemini-3.5-flash")
        raw = fn(self._TEST_PROMPT)

        assert isinstance(raw, str)
        assert len(raw) > 0, "Live Gemini response was empty"

        data = json.loads(raw)
        assert "multiple" in data, f"Key 'multiple' not found in response: {data}"
        assert float(data["multiple"]) == pytest.approx(1.0, abs=0.01)

    def test_live_cached_callable(self, tmp_path):
        """Verify caching works end-to-end with live responses."""
        raw_fn = make_gemini_callable(model="gemini-3.5-flash")
        cached_fn = make_cached_callable(raw_fn, cache_path=tmp_path / "live_cache.json")

        result1 = cached_fn(self._TEST_PROMPT)
        result2 = cached_fn(self._TEST_PROMPT)

        assert result1 == result2
        assert len(result1) > 0
