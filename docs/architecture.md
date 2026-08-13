# Architecture

`mcp-authflow-resource` is the **resource server** half of the OAuth 2.0 split. It takes Bearer tokens from MCP clients and gates tool execution on whether the authorization server says they're valid.

## The full picture

```
  MCP Client
      |
      | Bearer token
      v
+---------------------------+
|    Resource Server        |   <-- this package
|    (your MCP tools)       |
|                           |
|  1. Extract Bearer token  |
|  2. Introspect token  ----+---> Auth Server (/introspect)
|  3. Check scopes          |         |
|  4. Friction check        |    "active": true/false
|  5. Execute tool          |
+---------------------------+
```

## Token verification flow

1. **Client sends a request** with `Authorization: Bearer <token>`.
2. **[`IntrospectionTokenVerifier`][mcp_authflow_resource.IntrospectionTokenVerifier] calls** the auth server's `/introspect` endpoint ([RFC 7662](https://datatracker.ietf.org/doc/html/rfc7662)).
3. **Auth server responds** with token metadata (`active`, `scope`, `client_id`, `exp`, `aud`).
4. **If active and scopes match**, the tool executes.
5. **If `validate_resource=True`**, the `aud` claim must match the server's URL ([RFC 8707](https://datatracker.ietf.org/doc/html/rfc8707)), so a token issued for service A can't be replayed against service B.

## Discovery endpoints

When you call [`register_oauth_discovery_endpoints`][mcp_authflow_resource.register_oauth_discovery_endpoints], the resource server gets five `.well-known` routes that let MCP clients bootstrap without out-of-band configuration:

| Endpoint | Spec |
|---|---|
| `GET /.well-known/oauth-protected-resource` | [RFC 9728](https://datatracker.ietf.org/doc/html/rfc9728) |
| `GET /mcp/.well-known/oauth-protected-resource` | RFC 9728 (path-scoped) |
| `GET /.well-known/oauth-authorization-server` | [RFC 8414](https://datatracker.ietf.org/doc/html/rfc8414) |
| `GET /.well-known/oauth-authorization-server/mcp` | RFC 8414 (path-scoped) |
| `GET /.well-known/openid-configuration` | OIDC Discovery |

Clients hit these routes and start an authorization flow with no manual configuration.

## What's in the package

| Module | Responsibility |
|---|---|
| [`auth`](api/auth.md) | Token introspection client + SSRF protection. |
| [`oauth_discovery`](api/oauth-discovery.md) | `.well-known` endpoint registration. |
| [`friction`](api/friction.md) | Per-tool adaptive rate limiting. |
| [`middleware`](api/middleware.md) | Path normalization, logging. |
| [`validation`](api/validation.md) | Response validation helpers for tools. |

What this package deliberately does **not** provide:

- An MCP server. Bring your own; `FastMCP` from the MCP SDK is the usual choice.
- A token store. Storage lives on the authorization server side (see [mcp-authflow][mcp-authflow]).
- User identity. Tokens carry a `client_id` and optionally a `sub`; mapping to a user is up to your auth server.
- A consent UI. The OAuth flow itself happens at the auth server; this package only verifies the issued token at the resource boundary.

## SSRF protection

The introspection endpoint URL is a foot-gun: if you read it from config or a discovery document, an attacker could redirect it at internal services. [`is_safe_url`][mcp_authflow_resource.is_safe_url] enforces a small safe-by-default policy. HTTPS is always allowed. HTTP is allowed only to `localhost`, loopback addresses, or non-DNS hostnames (Docker / k8s service names); HTTP to any DNS-resolvable external host is rejected.

!!! warning "Not a general-purpose SSRF filter"
    `is_safe_url` does **not** resolve DNS — any `https://` URL with a non-IP hostname is accepted regardless of the address it resolves to. This is intentional for *operator-configured* endpoints, where you control the URL. Do not reuse it to validate untrusted, user-supplied URLs (webhooks, fetch targets); for those you must resolve the hostname and reject private/internal targets yourself (and pin the resolved IP to defend against DNS rebinding). See the function docstring for details.

[`IntrospectionTokenVerifier`][mcp_authflow_resource.IntrospectionTokenVerifier] runs this check once at construction time. A request against an endpoint with an unsafe URL is rejected (it fails auth and returns no token) rather than being introspected. The verifier also owns a pooled HTTP client; call `await verifier.close()` during application shutdown to release its connections.

[mcp-authflow]: https://github.com/brooksmcmillin/mcp-authflow
