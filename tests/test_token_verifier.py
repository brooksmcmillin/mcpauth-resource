"""Tests for IntrospectionTokenVerifier — RFC 7662 token introspection."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_authflow_resource.auth import token_verifier as token_verifier_module
from mcp_authflow_resource.auth.token_verifier import IntrospectionTokenVerifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTROSPECTION_URL = "http://localhost:8080/introspect"
SERVER_URL = "https://mcp.example.com"

_ACTIVE_TOKEN_DATA: dict[str, Any] = {
    "active": True,
    "client_id": "test-client",
    "scope": "read write",
    "exp": 9999999999,
    "aud": SERVER_URL,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verifier(
    introspection_endpoint: str = INTROSPECTION_URL,
    server_url: str = SERVER_URL,
    validate_resource: bool = False,
) -> IntrospectionTokenVerifier:
    return IntrospectionTokenVerifier(
        introspection_endpoint=introspection_endpoint,
        server_url=server_url,
        validate_resource=validate_resource,
    )


def _mock_http_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
) -> MagicMock:
    """Return a mock httpx.Response-like object."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    return response


def _patch_client(mock_post: AsyncMock) -> Any:
    """Return a patch context for httpx.AsyncClient wired to ``mock_post``."""
    patcher = patch("httpx.AsyncClient")
    mock_client_cls = patcher.start()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return patcher


class _FakeClock:
    """Controllable stand-in for the ``time`` module used by the verifier."""

    def __init__(self, monotonic: float = 1000.0, wall: float = 1_000_000.0) -> None:
        self._monotonic = monotonic
        self._wall = wall

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._wall += seconds

    def monotonic(self) -> float:
        return self._monotonic

    def time(self) -> float:
        return self._wall


# ---------------------------------------------------------------------------
# SSRF guard tests
# ---------------------------------------------------------------------------


class TestSSRFGuard:
    """verify_token must reject unsafe introspection endpoint URLs."""

    async def test_unsafe_url_returns_none_without_http_call(self) -> None:
        """An unsafe scheme (e.g. file://) is rejected before any HTTP call."""
        verifier = _make_verifier(introspection_endpoint="file:///etc/passwd")

        with patch("httpx.AsyncClient") as mock_client_cls:
            result = await verifier.verify_token("some-token")

        assert result is None
        mock_client_cls.assert_not_called()

    async def test_external_http_url_returns_none(self) -> None:
        """Plain http:// to an external host is rejected (SSRF guard)."""
        verifier = _make_verifier(introspection_endpoint="http://evil.example.com/introspect")

        with patch("httpx.AsyncClient") as mock_client_cls:
            result = await verifier.verify_token("some-token")

        assert result is None
        mock_client_cls.assert_not_called()

    async def test_https_url_is_accepted(self) -> None:
        """HTTPS endpoints pass the SSRF guard and proceed to HTTP."""
        verifier = _make_verifier(introspection_endpoint="https://auth.example.com/introspect")

        mock_response = _mock_http_response(200, _ACTIVE_TOKEN_DATA)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is not None
        mock_post.assert_called_once()

    async def test_localhost_http_url_is_accepted(self) -> None:
        """http://localhost is permitted (allow_localhost=True in the verifier)."""
        verifier = _make_verifier(introspection_endpoint="http://localhost:8080/introspect")

        mock_response = _mock_http_response(200, _ACTIVE_TOKEN_DATA)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is not None

    async def test_ssrf_unsafe_url_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unsafe URL should produce a WARNING log."""
        verifier = _make_verifier(introspection_endpoint="ftp://evil.example.com/")
        log_name = "mcp_authflow_resource.auth.token_verifier"

        with (
            patch("httpx.AsyncClient"),
            caplog.at_level(logging.WARNING, logger=log_name),
        ):
            await verifier.verify_token("tok")

        assert any("unsafe scheme" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Pooled HTTP client tests
# ---------------------------------------------------------------------------


class TestPooledHttpClient:
    """The verifier owns one client for its full active lifetime."""

    async def test_reuses_one_client_for_uncached_verifications(self) -> None:
        verifier = _make_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _ACTIVE_TOKEN_DATA))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            assert await verifier.verify_token("first") is not None
            assert await verifier.verify_token("second") is not None

        mock_client_cls.assert_called_once_with(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            verify=False,
        )
        assert mock_post.call_count == 2

    async def test_concurrent_verifications_create_one_client(self) -> None:
        verifier = _make_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _ACTIVE_TOKEN_DATA))
        entered = asyncio.Event()
        release = asyncio.Event()

        async def enter_client() -> MagicMock:
            entered.set()
            await release.wait()
            return MagicMock(post=mock_post)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(side_effect=enter_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            first = asyncio.create_task(verifier.verify_token("first"))
            await entered.wait()
            second = asyncio.create_task(verifier.verify_token("second"))
            await asyncio.sleep(0)
            release.set()
            assert await first is not None
            assert await second is not None

        mock_client_cls.assert_called_once()

    async def test_close_releases_the_pooled_client(self) -> None:
        verifier = _make_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _ACTIVE_TOKEN_DATA))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await verifier.verify_token("token")
            await verifier.close()

        mock_client_cls.return_value.__aexit__.assert_awaited_once_with(None, None, None)

    def test_endpoint_cannot_change_after_ssrf_validation(self) -> None:
        verifier = _make_verifier()

        with pytest.raises(AttributeError):
            object.__setattr__(verifier, "introspection_endpoint", "file:///etc/passwd")


# ---------------------------------------------------------------------------
# HTTP response error tests
# ---------------------------------------------------------------------------


class TestHttpErrors:
    """Non-200 responses and network errors should return None."""

    @pytest.mark.parametrize("status_code", [400, 401, 403, 500, 503])
    async def test_non_200_status_returns_none(self, status_code: int) -> None:
        """Any non-200 HTTP status code causes verify_token to return None."""
        verifier = _make_verifier()
        mock_response = _mock_http_response(status_code, {})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is None

    async def test_network_exception_returns_none(self) -> None:
        """Network-level exceptions (e.g. ConnectError) are caught and return None."""
        verifier = _make_verifier()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    post=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
                )
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is None

    async def test_arbitrary_exception_returns_none(self) -> None:
        """Any unexpected exception inside the HTTP block is caught and returns None."""
        verifier = _make_verifier()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(side_effect=RuntimeError("boom")))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is None

    async def test_exception_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Caught exceptions should be logged at WARNING level."""
        verifier = _make_verifier()
        log_name = "mcp_authflow_resource.auth.token_verifier"

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            caplog.at_level(logging.WARNING, logger=log_name),
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock(side_effect=RuntimeError("oops")))
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await verifier.verify_token("tok")

        assert any("introspection failed" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Active / inactive token tests
# ---------------------------------------------------------------------------


class TestTokenActiveFlag:
    """The `active` field in the introspection response controls token validity."""

    async def test_active_false_returns_none(self) -> None:
        """active: false must cause verify_token to return None."""
        verifier = _make_verifier()
        mock_response = _mock_http_response(200, {"active": False})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is None

    async def test_active_missing_returns_none(self) -> None:
        """A response with no `active` key (defaults to False) returns None."""
        verifier = _make_verifier()
        mock_response = _mock_http_response(200, {"client_id": "x"})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("some-token")

        assert result is None

    async def test_active_true_returns_access_token(self) -> None:
        """active: true with full data returns a populated AccessToken."""
        verifier = _make_verifier()
        mock_response = _mock_http_response(200, _ACTIVE_TOKEN_DATA)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("my-token")

        assert result is not None
        assert result.token == "my-token"
        assert result.client_id == "test-client"
        assert result.scopes == ["read", "write"]
        assert result.expires_at == 9999999999

    async def test_active_true_no_scope_returns_empty_scopes(self) -> None:
        """active: true with no scope field yields an empty scopes list."""
        verifier = _make_verifier()
        token_data = {"active": True, "client_id": "c"}
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is not None
        assert result.scopes == []

    async def test_list_format_scope_is_accepted(self) -> None:
        """An AS returning scope as a JSON array (Keycloak, Okta) is parsed,
        not rejected via AttributeError."""
        verifier = _make_verifier()
        token_data = {"active": True, "client_id": "c", "scope": ["read", "write"]}
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is not None
        assert result.scopes == ["read", "write"]

    async def test_unknown_client_id_defaults_to_unknown(self) -> None:
        """When client_id is absent, the AccessToken gets 'unknown' as client_id."""
        verifier = _make_verifier()
        token_data = {"active": True}
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is not None
        assert result.client_id == "unknown"


# ---------------------------------------------------------------------------
# Secure-by-default construction
# ---------------------------------------------------------------------------


class TestResourceValidationDefault:
    """RFC 8707 audience binding must be enabled by default (issue #4)."""

    def test_validate_resource_defaults_to_true(self) -> None:
        """Omitting validate_resource enables audience binding."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
        )
        assert verifier.validate_resource is True

    async def test_mismatched_aud_rejected_by_default(self) -> None:
        """A token for a different resource is rejected without opting in."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
        )
        token_data = {**_ACTIVE_TOKEN_DATA, "aud": "https://different.example.com"}
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is None


# ---------------------------------------------------------------------------
# Resource validation tests  (validate_resource=True)
# ---------------------------------------------------------------------------


class TestResourceValidation:
    """RFC 8707 resource validation via the `aud` claim."""

    async def _call_verifier_with_data(
        self, token_data: dict[str, Any], validate_resource: bool = True
    ) -> Any:  # noqa: ANN401
        verifier = _make_verifier(validate_resource=validate_resource)
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            return await verifier.verify_token("tok")

    async def test_string_aud_matches_server_url(self) -> None:
        """String `aud` matching the server URL passes resource validation."""
        token_data = {**_ACTIVE_TOKEN_DATA, "aud": SERVER_URL}
        result = await self._call_verifier_with_data(token_data)
        assert result is not None

    async def test_list_aud_containing_server_url_resource_validation_passes(self) -> None:
        """List `aud` containing the server URL passes _validate_resource.

        NOTE: verify_token subsequently passes the list as `resource` to
        AccessToken, which requires a string — Pydantic raises a validation
        error that is caught by the except block and returns None. This test
        documents the current behaviour. The _validate_resource logic itself
        correctly accepts a matching list (tested directly in
        TestValidateResourceMethod).
        """
        token_data = {
            **_ACTIVE_TOKEN_DATA,
            "aud": ["https://other.example.com", SERVER_URL],
        }
        # The list aud passes _validate_resource but AccessToken rejects the
        # list via Pydantic, so verify_token returns None here.
        result = await self._call_verifier_with_data(token_data)
        assert result is None

    async def test_list_aud_not_containing_server_url_returns_none(self) -> None:
        """List `aud` that does not include the server URL fails validation."""
        token_data = {
            **_ACTIVE_TOKEN_DATA,
            "aud": ["https://other.example.com", "https://another.example.com"],
        }
        result = await self._call_verifier_with_data(token_data)
        assert result is None

    async def test_string_aud_mismatch_returns_none(self) -> None:
        """String `aud` that does not match the server URL fails validation."""
        token_data = {**_ACTIVE_TOKEN_DATA, "aud": "https://different.example.com"}
        result = await self._call_verifier_with_data(token_data)
        assert result is None

    async def test_missing_aud_returns_none(self) -> None:
        """Token with no `aud` claim fails resource validation."""
        token_data = {k: v for k, v in _ACTIVE_TOKEN_DATA.items() if k != "aud"}
        result = await self._call_verifier_with_data(token_data)
        assert result is None

    async def test_resource_validation_disabled_skips_aud_check(self) -> None:
        """With validate_resource=False, a mismatched `aud` still returns a token."""
        token_data = {**_ACTIVE_TOKEN_DATA, "aud": "https://different.example.com"}
        result = await self._call_verifier_with_data(token_data, validate_resource=False)
        assert result is not None

    async def test_resource_validation_logs_warning_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Resource validation failure should log a WARNING."""
        token_data = {**_ACTIVE_TOKEN_DATA, "aud": "https://different.example.com"}
        log_name = "mcp_authflow_resource.auth.token_verifier"

        verifier = _make_verifier(validate_resource=True)
        mock_response = _mock_http_response(200, token_data)
        mock_post = AsyncMock(return_value=mock_response)

        with (
            patch("httpx.AsyncClient") as mock_client_cls,
            caplog.at_level(logging.WARNING, logger=log_name),
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is None
        assert any("resource validation failed" in record.message for record in caplog.records)

    async def test_readme_full_example_introspection_response_verifies(self) -> None:
        """The README "Full Example" introspect response must pass verification.

        Regression guard for issue #68: the documented example auth server used
        to omit the `aud` claim, so its tokens were rejected by the default
        validate_resource=True. This mirrors the shape the example now returns
        and asserts the verifier accepts it when the audience matches.
        """
        resource_server_url = "http://localhost:8001"
        introspection_response = {
            "active": True,
            "client_id": "test",
            "scope": "read",
            "exp": 9999999999,
            "aud": resource_server_url,
        }
        verifier = _make_verifier(server_url=resource_server_url, validate_resource=True)
        mock_response = _mock_http_response(200, introspection_response)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await verifier.verify_token("tok")

        assert result is not None


# ---------------------------------------------------------------------------
# _validate_resource unit tests
# ---------------------------------------------------------------------------


class TestValidateResourceMethod:
    """Direct unit tests for IntrospectionTokenVerifier._validate_resource."""

    def _make(self, server_url: str = SERVER_URL) -> IntrospectionTokenVerifier:
        return _make_verifier(server_url=server_url, validate_resource=True)

    def test_returns_false_when_server_url_empty(self) -> None:
        """_validate_resource returns False when server_url is empty."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url="",
            validate_resource=True,
        )
        # resource_url will be empty/None
        assert verifier._validate_resource({"aud": SERVER_URL}) is False

    def test_is_valid_resource_returns_false_when_resource_url_none(self) -> None:
        """_is_valid_resource short-circuits to False when resource_url is empty.

        This guard is distinct from the one in _validate_resource: a caller
        could reach _is_valid_resource directly with a still-empty resource_url,
        so it must fail closed on its own rather than rely on the earlier check.
        """
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url="",
            validate_resource=True,
        )
        # server_url="" resolves resource_url to an empty/None value.
        assert not verifier.resource_url
        assert verifier._is_valid_resource("https://mcp.example.com") is False

    def test_list_aud_all_mismatched_returns_false(self) -> None:
        verifier = self._make()
        assert (
            verifier._validate_resource({"aud": ["https://a.example.com", "https://b.example.com"]})
            is False
        )

    def test_list_aud_one_match_returns_true(self) -> None:
        verifier = self._make()
        assert (
            verifier._validate_resource({"aud": ["https://nope.example.com", SERVER_URL]}) is True
        )

    def test_empty_aud_list_returns_false(self) -> None:
        verifier = self._make()
        assert verifier._validate_resource({"aud": []}) is False

    def test_no_aud_returns_false(self) -> None:
        verifier = self._make()
        assert verifier._validate_resource({}) is False

    def test_string_aud_match_returns_true(self) -> None:
        verifier = self._make()
        assert verifier._validate_resource({"aud": SERVER_URL}) is True

    def test_string_aud_mismatch_returns_false(self) -> None:
        verifier = self._make()
        assert verifier._validate_resource({"aud": "https://wrong.example.com"}) is False


# ---------------------------------------------------------------------------
# Caller authentication on /introspect (RFC 7662 §2.1)
# ---------------------------------------------------------------------------


class TestClientAuth:
    """Optional client authentication on the introspection POST."""

    async def _captured_post(self, verifier: IntrospectionTokenVerifier) -> tuple[Any, Any]:
        """Run verify_token against a stub client and return (args, kwargs) of the POST."""
        mock_response = _mock_http_response(200, _ACTIVE_TOKEN_DATA)
        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=mock_post)
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await verifier.verify_token("tok")

        mock_post.assert_called_once()
        return mock_post.call_args.args, mock_post.call_args.kwargs

    # --- defaults / no auth ----------------------------------------------

    async def test_default_construction_sends_no_auth_header(self) -> None:
        """Backwards compat: with no client_secret, no Authorization header."""
        verifier = _make_verifier()
        _, kwargs = await self._captured_post(verifier)

        assert "Authorization" not in kwargs["headers"]
        # Only the token field is in the body.
        assert kwargs["data"] == {"token": "tok"}

    async def test_explicit_none_sends_no_auth_header(self) -> None:
        """client_auth_method='none' also sends no Authorization header."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_auth_method="none",
        )
        _, kwargs = await self._captured_post(verifier)

        assert "Authorization" not in kwargs["headers"]
        assert kwargs["data"] == {"token": "tok"}

    async def test_client_secret_omitted_overrides_method_to_none(self) -> None:
        """Setting only client_auth_method (no secret) still sends no auth."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_auth_method="client_secret_basic",
        )
        _, kwargs = await self._captured_post(verifier)

        assert "Authorization" not in kwargs["headers"]

    # --- client_secret_basic --------------------------------------------

    async def test_client_secret_basic_sends_basic_auth_header(self) -> None:
        """client_secret_basic puts base64(id:secret) in Authorization: Basic."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_id="my-client",
            client_secret="s3cret",
        )
        _, kwargs = await self._captured_post(verifier)

        expected = base64.b64encode(b"my-client:s3cret").decode()
        assert kwargs["headers"]["Authorization"] == f"Basic {expected}"
        assert kwargs["data"] == {"token": "tok"}

    def test_client_secret_basic_without_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id"):
            IntrospectionTokenVerifier(
                introspection_endpoint=INTROSPECTION_URL,
                server_url=SERVER_URL,
                client_secret="s3cret",
                client_auth_method="client_secret_basic",
            )

    # --- client_secret_post ---------------------------------------------

    async def test_client_secret_post_puts_credentials_in_body(self) -> None:
        """client_secret_post adds client_id and client_secret to the POST body."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_id="my-client",
            client_secret="s3cret",
            client_auth_method="client_secret_post",
        )
        _, kwargs = await self._captured_post(verifier)

        assert "Authorization" not in kwargs["headers"]
        assert kwargs["data"] == {
            "token": "tok",
            "client_id": "my-client",
            "client_secret": "s3cret",
        }

    def test_client_secret_post_without_client_id_raises(self) -> None:
        with pytest.raises(ValueError, match="client_id"):
            IntrospectionTokenVerifier(
                introspection_endpoint=INTROSPECTION_URL,
                server_url=SERVER_URL,
                client_secret="s3cret",
                client_auth_method="client_secret_post",
            )

    # --- bearer ----------------------------------------------------------

    async def test_bearer_sends_bearer_auth_header(self) -> None:
        """bearer puts the client_secret in Authorization: Bearer."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_secret="shared-secret",
            client_auth_method="bearer",
        )
        _, kwargs = await self._captured_post(verifier)

        assert kwargs["headers"]["Authorization"] == "Bearer shared-secret"
        assert kwargs["data"] == {"token": "tok"}

    async def test_bearer_does_not_require_client_id(self) -> None:
        """bearer auth works with only client_secret (no client_id needed)."""
        # Construction must not raise.
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_secret="x",
            client_auth_method="bearer",
        )
        assert verifier is not None

    def test_bearer_without_client_secret_raises(self) -> None:
        """Explicit bearer with no client_secret is a misconfiguration, not silent no-auth."""
        with pytest.raises(ValueError, match="bearer.*requires client_secret"):
            IntrospectionTokenVerifier(
                introspection_endpoint=INTROSPECTION_URL,
                server_url=SERVER_URL,
                client_auth_method="bearer",
            )

    # --- runtime credential guards (not assert-based) --------------------

    def test_missing_secret_at_call_time_raises_not_none_header(self) -> None:
        """A missing credential must fail loudly, never interpolate ``None``.

        The construction-time guards previously mirrored ``assert`` statements
        that ``python -O`` strips. This verifies the enforcement lives in a real
        runtime check by simulating a verifier whose credentials went missing
        after construction: it must raise rather than emit ``Basic None:None``.
        """
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_id="my-client",
            client_secret="s3cret",
            client_auth_method="client_secret_basic",
        )
        verifier._client_secret = None

        headers: dict[str, str] = {}
        data: dict[str, str] = {}
        with pytest.raises(RuntimeError, match="requires client_secret"):
            verifier._apply_client_auth(headers, data)
        assert "Authorization" not in headers

    def test_missing_client_id_at_call_time_raises(self) -> None:
        """client_secret_post with a missing client_id fails loudly, no partial body."""
        verifier = IntrospectionTokenVerifier(
            introspection_endpoint=INTROSPECTION_URL,
            server_url=SERVER_URL,
            client_id="my-client",
            client_secret="s3cret",
            client_auth_method="client_secret_post",
        )
        verifier._client_id = None

        headers: dict[str, str] = {}
        data: dict[str, str] = {}
        with pytest.raises(RuntimeError, match="requires client_id"):
            verifier._apply_client_auth(headers, data)
        assert data == {}


# ---------------------------------------------------------------------------
# Introspection caching tests
# ---------------------------------------------------------------------------

# Token whose `exp` is far enough out that the cache TTL, not the token expiry,
# governs the entry lifetime in tests that don't care about exp-capping.
_LONG_LIVED_TOKEN: dict[str, Any] = {**_ACTIVE_TOKEN_DATA, "exp": 9_999_999_999}


def _cached_verifier(ttl: float = 300.0, **kwargs: Any) -> IntrospectionTokenVerifier:
    return IntrospectionTokenVerifier(
        introspection_endpoint=INTROSPECTION_URL,
        server_url=SERVER_URL,
        validate_resource=False,
        introspection_cache_ttl=ttl,
        **kwargs,
    )


class TestIntrospectionCaching:
    """verify_token caches successful introspections when a TTL is configured."""

    async def test_caching_disabled_by_default(self) -> None:
        """With no TTL configured, every call performs a fresh introspection."""
        verifier = _make_verifier()  # default ttl == 0.0 -> disabled
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            assert await verifier.verify_token("tok") is not None
            assert await verifier.verify_token("tok") is not None
        finally:
            patcher.stop()

        assert mock_post.call_count == 2

    async def test_cache_hit_skips_second_introspection(self) -> None:
        """A second call for the same token is served from cache (no HTTP call)."""
        verifier = _cached_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            first = await verifier.verify_token("tok")
            second = await verifier.verify_token("tok")
        finally:
            patcher.stop()

        assert first is not None and second is not None
        assert second.client_id == first.client_id
        assert mock_post.call_count == 1

    async def test_distinct_tokens_are_cached_separately(self) -> None:
        """Different tokens do not collide in the cache."""
        verifier = _cached_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            await verifier.verify_token("tok-a")
            await verifier.verify_token("tok-b")
        finally:
            patcher.stop()

        assert mock_post.call_count == 2

    async def test_cache_expires_after_ttl(self) -> None:
        """Once the TTL elapses, the next call re-introspects."""
        verifier = _cached_verifier(ttl=10.0)
        clock = _FakeClock()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            with patch.object(token_verifier_module, "time", clock):
                await verifier.verify_token("tok")
                clock.advance(11.0)  # past the 10s TTL
                await verifier.verify_token("tok")
        finally:
            patcher.stop()

        assert mock_post.call_count == 2

    async def test_entry_lifetime_capped_by_token_exp(self) -> None:
        """A token expiring sooner than the TTL is never served past its expiry."""
        verifier = _cached_verifier(ttl=300.0)
        clock = _FakeClock()
        short_token = {**_ACTIVE_TOKEN_DATA, "exp": int(clock.time()) + 5}
        mock_post = AsyncMock(return_value=_mock_http_response(200, short_token))

        patcher = _patch_client(mock_post)
        try:
            with patch.object(token_verifier_module, "time", clock):
                await verifier.verify_token("tok")
                clock.advance(6.0)  # past token exp (+5) but well within TTL (300)
                await verifier.verify_token("tok")
        finally:
            patcher.stop()

        assert mock_post.call_count == 2

    async def test_revocation_reflected_after_ttl_expiry(self) -> None:
        """Once the cached entry expires, an active:false response is honored.

        This documents the revocation-latency tradeoff: a token cached as active
        keeps being accepted (without re-introspecting) until its entry expires,
        after which the next introspection observes the revocation and the cache
        is left empty.
        """
        verifier = _cached_verifier(ttl=10.0)
        clock = _FakeClock()
        active = _mock_http_response(200, _LONG_LIVED_TOKEN)
        inactive = _mock_http_response(200, {"active": False})
        mock_post = AsyncMock(side_effect=[active, inactive])

        patcher = _patch_client(mock_post)
        try:
            with patch.object(token_verifier_module, "time", clock):
                assert await verifier.verify_token("tok") is not None  # cached active
                clock.advance(11.0)  # fast-path entry expires
                assert await verifier.verify_token("tok") is None  # re-introspect -> inactive
        finally:
            patcher.stop()

        assert mock_post.call_count == 2
        assert len(verifier._cache) == 0

    async def test_burst_after_cache_hit_avoids_introspect(self) -> None:
        """The cache hit is what prevents 429s: repeated calls don't re-introspect.

        This is the core fix — during a burst, only the first call reaches the
        authorization server, so its ``/introspect`` rate limit is never tripped.
        """
        verifier = _cached_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            results = [await verifier.verify_token("tok") for _ in range(25)]
        finally:
            patcher.stop()

        assert all(r is not None for r in results)
        assert mock_post.call_count == 1

    async def test_non_200_is_not_cached(self) -> None:
        """A non-200 (e.g. 429) returns None and leaves the cache empty."""
        verifier = _cached_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(429, {}))

        patcher = _patch_client(mock_post)
        try:
            result = await verifier.verify_token("tok")
        finally:
            patcher.stop()

        assert result is None
        assert len(verifier._cache) == 0

    async def test_cache_key_does_not_store_raw_token(self) -> None:
        """The raw bearer token must not appear as a cache key."""
        verifier = _cached_verifier()
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            await verifier.verify_token("super-secret-token")  # noqa: S106
        finally:
            patcher.stop()

        assert "super-secret-token" not in verifier._cache
        assert len(verifier._cache) == 1

    async def test_invalid_max_size_raises(self) -> None:
        """A non-positive cache size is rejected at construction."""
        with pytest.raises(ValueError, match="introspection_cache_max_size"):
            _cached_verifier(introspection_cache_max_size=0)

    async def test_max_size_evicts_oldest_entry(self) -> None:
        """With max_size=1, caching a second token evicts the first."""
        verifier = _cached_verifier(introspection_cache_max_size=1)
        mock_post = AsyncMock(return_value=_mock_http_response(200, _LONG_LIVED_TOKEN))

        patcher = _patch_client(mock_post)
        try:
            await verifier.verify_token("tok-a")  # cached
            await verifier.verify_token("tok-b")  # evicts tok-a
            await verifier.verify_token("tok-a")  # must re-introspect
        finally:
            patcher.stop()

        assert verifier._cache_max_size == 1
        assert len(verifier._cache) == 1
        assert mock_post.call_count == 3
