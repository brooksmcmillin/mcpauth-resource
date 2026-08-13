"""Token verifier implementation using OAuth 2.0 Token Introspection (RFC 7662)."""

import asyncio
import base64
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Literal

import httpx
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url

from mcp_authflow_resource.auth.ssrf_protection import is_safe_url

logger = logging.getLogger(__name__)

# Default ceiling on the number of cached introspection results, bounding memory
# when caching is enabled (CWE-770). Oldest entries are evicted first.
_DEFAULT_CACHE_MAX_SIZE = 1024

ClientAuthMethod = Literal[
    "none",
    "client_secret_basic",
    "client_secret_post",
    "bearer",
]


class IntrospectionTokenVerifier(TokenVerifier):
    """Token verifier that uses OAuth 2.0 Token Introspection (RFC 7662).

    Optional caller authentication on ``/introspect`` (RFC 7662 §2.1) is
    supported via the ``client_id`` / ``client_secret`` / ``client_auth_method``
    parameters. When ``client_secret`` is unset the request is sent without
    any authentication (backwards-compatible default).

    Supported ``client_auth_method`` values:

    - ``"client_secret_basic"`` (default when credentials are given) — RFC 6749
      §2.3.1 HTTP Basic auth: ``Authorization: Basic base64(client_id:client_secret)``.
      Requires both ``client_id`` and ``client_secret``.
    - ``"client_secret_post"`` — RFC 6749 §2.3.1 form parameters: adds
      ``client_id`` and ``client_secret`` to the POST body. Requires both.
    - ``"bearer"`` — RFC 6750 bearer auth, used by deployments that protect
      the introspection endpoint with a single shared secret:
      ``Authorization: Bearer <client_secret>``. Requires only ``client_secret``.
    - ``"none"`` — explicit no-auth (same as omitting ``client_secret``).

    ``validate_resource`` defaults to ``True``, enforcing RFC 8707 audience
    binding so a token issued for resource server A cannot be replayed against
    resource server B that shares the same authorization server. Set it to
    ``False`` only for single-resource-server deployments where every token
    issued by the authorization server is intended for this resource.

    **Introspection caching.** By default the verifier introspects on every
    request. Set ``introspection_cache_ttl`` to a positive number of seconds to
    cache *successful* (active, resource-valid) introspections, so a burst of
    requests bearing the same token results in one introspection rather than one
    per request. This keeps the authorization server from being hammered (and
    rate-limited) under load — the failure mode this option exists to prevent.
    Caching is **opt-in** because it trades revocation latency for throughput: a
    revoked token may still be accepted until its cache entry expires. Each
    entry's lifetime is capped at ``min(introspection_cache_ttl, token_exp -
    now)`` so a cached token is never served past its own expiry, and a
    definitive ``active: false`` response immediately drops any cached entry for
    that token. Only successful results are cached — ``active: false``,
    resource-validation failures, transport errors, and non-200 responses are
    never cached and fall through to the usual ``None`` (auth-failure) result.
    """

    def __init__(
        self,
        introspection_endpoint: str,
        server_url: str,
        validate_resource: bool = True,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        client_auth_method: ClientAuthMethod = "client_secret_basic",
        introspection_cache_ttl: float = 0.0,
        introspection_cache_max_size: int = _DEFAULT_CACHE_MAX_SIZE,
    ):
        self._introspection_endpoint = introspection_endpoint
        self.server_url = server_url
        self.validate_resource = validate_resource
        self.resource_url = resource_url_from_server_url(server_url)
        self._client_id = client_id
        self._client_secret = client_secret
        self._client_auth_method: ClientAuthMethod = (
            "none"
            if client_secret is None and client_auth_method != "bearer"
            else client_auth_method
        )
        self._introspection_endpoint_is_safe = is_safe_url(
            introspection_endpoint, allow_localhost=True
        )
        if not self._introspection_endpoint_is_safe:
            logger.warning(
                "Rejecting introspection endpoint with unsafe scheme: %s",
                introspection_endpoint,
            )
        self._client: httpx.AsyncClient | None = None
        self._client_context: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        if self._client_auth_method in ("client_secret_basic", "client_secret_post"):
            if self._client_id is None or self._client_secret is None:
                raise ValueError(
                    f"client_auth_method={self._client_auth_method!r} requires "
                    "both client_id and client_secret"
                )
        elif self._client_auth_method == "bearer" and self._client_secret is None:
            raise ValueError("client_auth_method='bearer' requires client_secret")

        if introspection_cache_max_size <= 0:
            raise ValueError("introspection_cache_max_size must be a positive integer")

        # Introspection result cache. Keyed by a SHA-256 of the token (so raw
        # bearer tokens are not held in the cache dict), valued by
        # ``(AccessToken, expiry_monotonic)``. Disabled when TTL <= 0.
        self._cache_ttl = introspection_cache_ttl
        self._cache_max_size = introspection_cache_max_size
        self._cache: OrderedDict[str, tuple[AccessToken, float]] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    @property
    def _cache_enabled(self) -> bool:
        return self._cache_ttl > 0

    @property
    def introspection_endpoint(self) -> str:
        """The validated, immutable token-introspection endpoint."""
        return self._introspection_endpoint

    @staticmethod
    def _cache_key(token: str) -> str:
        """Derive a cache key that does not store the raw bearer token."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def _cache_get(self, key: str) -> AccessToken | None:
        """Return a non-expired cached token, or ``None``. Evicts if expired."""
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            access_token, expiry = entry
            if time.monotonic() >= expiry:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)  # LRU: mark recently used
            return access_token

    async def _cache_put(self, key: str, access_token: AccessToken, expires_at: int | None) -> None:
        """Store a successful introspection, capping lifetime at the token's expiry."""
        lifetime = self._cache_ttl
        if expires_at is not None:
            # Never serve a token past its own expiry.
            lifetime = min(lifetime, expires_at - time.time())
        if lifetime <= 0:
            return
        async with self._cache_lock:
            self._cache[key] = (access_token, time.monotonic() + lifetime)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_size:
                self._cache.popitem(last=False)  # evict oldest

    async def _cache_drop(self, key: str) -> None:
        """Remove any cached entry for ``key`` (e.g. token now reported inactive)."""
        async with self._cache_lock:
            self._cache.pop(key, None)

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the verifier-owned client, creating it once on first use."""
        async with self._client_lock:
            if self._client is None:
                self._client_context = httpx.AsyncClient(
                    timeout=httpx.Timeout(10.0, connect=5.0),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    verify=self.introspection_endpoint.startswith("https://"),
                )
                self._client = await self._client_context.__aenter__()
            return self._client

    async def close(self) -> None:
        """Close the pooled HTTP client when the owning application shuts down."""
        async with self._client_lock:
            if self._client_context is not None:
                await self._client_context.__aexit__(None, None, None)
                self._client = None
                self._client_context = None

    def _apply_client_auth(
        self,
        headers: dict[str, str],
        data: dict[str, str],
    ) -> None:
        """Apply the configured client auth to ``headers`` and ``data`` in place."""
        method = self._client_auth_method
        if method == "none":
            return

        # ``__init__`` guarantees the required credentials are present for every
        # authenticating method. Re-check here with real runtime guards rather
        # than ``assert`` (which ``python -O`` strips) so a missing credential
        # fails loudly instead of interpolating ``None`` into the Authorization
        # header. The locals also narrow ``str | None`` to ``str`` for the type
        # checker.
        client_secret = self._client_secret
        if client_secret is None:
            raise RuntimeError(f"client_auth_method={method!r} requires client_secret")

        if method == "bearer":
            headers["Authorization"] = f"Bearer {client_secret}"
            return

        client_id = self._client_id
        if client_id is None:
            raise RuntimeError(f"client_auth_method={method!r} requires client_id")

        if method == "client_secret_basic":
            creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"
        elif method == "client_secret_post":
            data["client_id"] = client_id
            data["client_secret"] = client_secret

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify token via introspection endpoint."""
        if not self._introspection_endpoint_is_safe:
            return None

        cache_key = self._cache_key(token) if self._cache_enabled else None

        # Cache fast path: a recent successful introspection for this token
        # short-circuits the network round-trip entirely.
        if cache_key is not None:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                logger.debug("Token introspection served from cache")
                return cached

        headers: dict[str, str] = {"Content-Type": "application/x-www-form-urlencoded"}
        data: dict[str, str] = {"token": token}
        self._apply_client_auth(headers, data)

        try:
            client = await self._get_client()
            response = await client.post(
                self.introspection_endpoint,
                data=data,
                headers=headers,
            )

            if response.status_code != 200:
                logger.debug("Token introspection returned status %s", response.status_code)
                return None

            data_resp = response.json()
            if not data_resp.get("active", False):
                logger.debug("Token marked as inactive")
                # Authoritative negative — reflect revocation immediately.
                if cache_key is not None:
                    await self._cache_drop(cache_key)
                return None

            # RFC 8707 resource validation (enabled by default)
            if self.validate_resource and not self._validate_resource(data_resp):
                logger.warning("Token resource validation failed. Expected: %s", self.resource_url)
                return None

            access_token = AccessToken(
                token=token,
                client_id=data_resp.get("client_id", "unknown"),
                scopes=self._parse_scopes(data_resp.get("scope")),
                expires_at=data_resp.get("exp"),
                resource=data_resp.get("aud"),  # Include resource in token
            )
            if cache_key is not None:
                await self._cache_put(cache_key, access_token, access_token.expires_at)
            return access_token
        except Exception as e:
            logger.warning("Token introspection failed: %s", e)
            return None

    @staticmethod
    def _parse_scopes(raw_scope: Any) -> list[str]:
        """Normalize the introspection ``scope`` claim into a list.

        RFC 7662 defines ``scope`` as a space-delimited string, but some
        authorization servers (e.g. Keycloak, Okta) return it as a JSON array.
        Handle both formats so list-format scopes do not raise AttributeError
        and reject otherwise-valid tokens.
        """
        if isinstance(raw_scope, list):
            return [str(scope) for scope in raw_scope]
        if isinstance(raw_scope, str) and raw_scope:
            return raw_scope.split()
        return []

    def _validate_resource(self, token_data: dict[str, Any]) -> bool:
        """Validate token was issued for this resource server."""
        if not self.server_url or not self.resource_url:
            return False  # Fail if strict validation requested but URLs missing

        # Check 'aud' claim first (standard JWT audience)
        aud = token_data.get("aud")
        if isinstance(aud, list):
            return any(self._is_valid_resource(audience) for audience in aud)
        elif aud:
            return self._is_valid_resource(aud)

        # No resource binding - invalid per RFC 8707
        return False

    def _is_valid_resource(self, resource: str) -> bool:
        """Check if resource matches this server using hierarchical matching."""
        if not self.resource_url:
            return False

        return check_resource_allowed(
            requested_resource=self.resource_url, configured_resource=resource
        )
