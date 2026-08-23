/*
 * Supabase access-token verification at the edge.
 *
 * ASYMMETRIC, NOT THE SHARED SECRET. Supabase offers a "legacy JWT secret"
 * (HS256) and this project also publishes an ES256 key at its JWKS endpoint.
 * We verify against the JWKS, and the difference is not cosmetic: an HS256
 * shared secret both verifies AND forges tokens -- the same string does both,
 * so a Worker holding it is a Worker that can mint any session it likes if it
 * ever leaks. A public key can only verify. It is also rotatable without
 * coordination: Supabase rolls the signing key, the JWKS changes, and this
 * picks it up on the next cache miss instead of every session breaking at once.
 *
 * The blueprint originally specified the shared secret. That was right for a
 * project still on legacy HS256 and wrong for this one, which was only
 * discovered by reading the JWKS endpoint rather than the setting name.
 */

const JWKS_TTL_MS = 10 * 60 * 1000;   // keys rotate rarely; 10 min bounds a rollover
let jwksCache = { url: null, keys: null, fetchedAt: 0 };

function b64urlToBytes(s) {
  const pad = s.length % 4 ? "=".repeat(4 - (s.length % 4)) : "";
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/") + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64urlToJson(s) {
  return JSON.parse(new TextDecoder().decode(b64urlToBytes(s)));
}

async function getKeys(supabaseUrl) {
  const url = `${supabaseUrl}/auth/v1/.well-known/jwks.json`;
  const fresh = jwksCache.url === url && jwksCache.keys
    && (Date.now() - jwksCache.fetchedAt) < JWKS_TTL_MS;
  if (fresh) return jwksCache.keys;

  const res = await fetch(url, { cf: { cacheTtl: 300, cacheEverything: true } });
  if (!res.ok) throw new Error(`jwks ${res.status}`);
  const { keys } = await res.json();
  if (!Array.isArray(keys) || !keys.length) throw new Error("jwks empty");
  jwksCache = { url, keys, fetchedAt: Date.now() };
  return keys;
}

/**
 * Verify a Supabase access token. Returns its claims, or null.
 *
 * Returns null rather than throwing for anything that means "not a valid
 * session", because every caller does the same thing with a failure: treat
 * the request as anonymous. Throwing would invite a catch that accidentally
 * swallows a genuine outage into "logged out", which is a confusing bug to
 * chase. Infrastructure failures (JWKS unreachable) DO throw, so they can be
 * distinguished and logged.
 */
export async function verifySupabaseToken(token, supabaseUrl) {
  if (typeof token !== "string" || token.split(".").length !== 3) return null;
  const [headerB64, payloadB64, sigB64] = token.split(".");

  let header, claims;
  try {
    header = b64urlToJson(headerB64);
    claims = b64urlToJson(payloadB64);
  } catch { return null; }

  // ALGORITHM IS PINNED. Accepting whatever the token's own header asks for
  // is the classic JWT break: a token claiming alg:none, or alg:HS256 signed
  // with the public key as the HMAC secret, would verify against a naive
  // implementation. We only ever attempt ES256.
  if (header.alg !== "ES256") return null;

  const keys = await getKeys(supabaseUrl);
  const jwk = keys.find((k) => k.kid === header.kid) || (keys.length === 1 ? keys[0] : null);
  if (!jwk) return null;

  const key = await crypto.subtle.importKey(
    "jwk", jwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]
  );
  const ok = await crypto.subtle.verify(
    { name: "ECDSA", hash: "SHA-256" },
    key,
    b64urlToBytes(sigB64),
    new TextEncoder().encode(`${headerB64}.${payloadB64}`)
  );
  if (!ok) return null;

  // Signature valid; now the claims. A correctly signed token for the wrong
  // audience, or an expired one, is still not a session.
  const now = Math.floor(Date.now() / 1000);
  if (typeof claims.exp !== "number" || claims.exp <= now) return null;
  if (typeof claims.iat === "number" && claims.iat > now + 60) return null;  // clock skew allowance
  if (claims.iss && !String(claims.iss).startsWith(supabaseUrl)) return null;
  if (claims.aud && claims.aud !== "authenticated") return null;
  if (!claims.sub) return null;

  return claims;
}
