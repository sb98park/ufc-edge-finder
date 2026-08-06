/**
 * Octane Alpha access gate — Cloudflare Worker
 *
 * Sits in front of octanealpha.com. Shows a login page until a valid
 * session cookie is present, then passes the request through to GitHub
 * Pages unchanged. The actual site (repo, GitHub Action, generated
 * HTML) never changes -- this is a fully separate layer in front of it.
 *
 * THIS FILE LIVES IN THE REPO ON PURPOSE. It previously existed only in
 * the Cloudflare dashboard, which is why its logo stayed on an old mark
 * while every other brand asset was regenerated from one geometry --
 * nothing could see it to update it. Keep the dashboard copy and this
 * file in sync; this is the source of truth.
 *
 * Required Worker secrets (set in the Cloudflare dashboard, never
 * hardcoded here — same principle as ODDS_API_KEY in GitHub Secrets):
 *   GUEST_PASSWORD    - shared password for friends/testers
 *   ADMIN_PASSWORD    - your own password
 *   SESSION_SECRET     - random string used to sign session cookies
 *   MAGIC_LINK_TOKEN   - separate secret for the resume auto-login link,
 *                        deliberately NOT the same as GUEST_PASSWORD so
 *                        it can be rotated independently if it ever
 *                        needs to change without affecting friends who
 *                        already have the real password
 */

const COOKIE_NAME = "octane_auth";
const DISPLAY_COOKIE_NAME = "octane_role"; // readable by site JS for the Guest/Admin UI badge -- carries no security weight, the HttpOnly cookie above is the only thing the Worker actually trusts
const SESSION_DAYS = 90; // "stay logged in" -- long enough that a refresh or a return visit weeks later doesn't re-prompt

// Paths that must stay reachable WITHOUT a login, so external services
// that never carry your session cookie still work correctly.
const PUBLIC_PATHS = [
  "/og-share-card.png", // social media crawlers (iMessage, Twitter, etc.) fetch this directly for link previews
  // The SVG favicon has to be public too: the LOGIN PAGE ITSELF requests
  // it, and a gated request would hand back login HTML where an image
  // was expected -- the icon would silently never load.
  "/favicon.svg",
  "/favicon.ico", "/favicon-16.png", "/favicon-32.png",
  "/apple-touch-icon.png", "/icon-192.png", "/icon-512.png",
  "/icon-512-light.png", // dark-outer variant, for anywhere the mark sits on white
];

function appendSessionCookies(headers, role, sessionCookie) {
  headers.append("Set-Cookie", sessionCookie);
  const maxAge = SESSION_DAYS * 24 * 60 * 60;
  headers.append("Set-Cookie", `${DISPLAY_COOKIE_NAME}=${role}; Path=/; Max-Age=${maxAge}; Secure; SameSite=Lax`);
}

async function serveAuthenticatedSite(request, role, secret) {
  // Fetch the real site at a clean root path -- not the original
  // request URL, since that could carry a login POST body or a
  // ?access=... query string neither of which the origin should see.
  // cf.cacheTtl: -1 explicitly disables Cloudflare's own edge cache for
  // THIS subrequest -- a fetch() call inside a Worker passes through
  // Cloudflare's cache by default, completely separately from whatever
  // Cache-Control header ends up on the response sent back to the
  // browser. Without this, a cached copy of the authenticated page from
  // an earlier session could be served to a brand new, unauthenticated
  // visitor without the Worker's cookie check ever being consulted.
  const cleanUrl = new URL(request.url);
  cleanUrl.search = "";
  const originResponse = await fetch(new Request(cleanUrl.toString(), { method: "GET" }), {
    cf: { cacheTtl: -1, cacheEverything: false },
  });

  const headers = new Headers(originResponse.headers);
  headers.set("Cache-Control", "no-store, must-revalidate");
  appendSessionCookies(headers, role, await makeSessionCookie(role, secret));
  return new Response(originResponse.body, { status: originResponse.status, headers });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Launch images and the manifest are fetched by iOS at APP LAUNCH,
    // before any session cookie is presented. Gated, they'd return login
    // HTML where a PNG was expected -- and iOS, getting no usable image,
    // falls back to the white screen these exist to replace.
    // Prefix-matched rather than listing eight filenames, so adding a device
    // size never needs a worker change.
    if (path.startsWith("/splash-") || path === "/manifest.json") {
      return fetch(request);
    }
    if (PUBLIC_PATHS.includes(path)) {
      return fetch(request);
    }

    // Logout: clear both cookies AND serve the login page directly in
    // this same response, rather than clearing cookies then redirecting
    // to "/" as a separate request. A redirect means a second request
    // has to happen before the login page appears, and that follow-up
    // request is one more place a stale cached response could slip in;
    // serving the login page directly here removes that possibility
    // rather than depending on cache-control headers behaving exactly
    // as expected across browsers.
    if (path === "/logout" && request.method === "GET") {
      const headers = new Headers({
        "Content-Type": "text/html;charset=UTF-8",
        "Cache-Control": "no-store, must-revalidate",
      });
      headers.append("Set-Cookie", `${COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax`);
      headers.append("Set-Cookie", `${DISPLAY_COOKIE_NAME}=; Path=/; Max-Age=0; Secure; SameSite=Lax`);
      return new Response(renderLoginPage({ error: false }), { status: 200, headers });
    }

    // Magic link for the resume: ?access=<token> auto-logs in as guest.
    // Serves the real site directly rather than redirecting to a clean
    // URL, for the same reason logout no longer redirects -- one fewer
    // follow-up request that could theoretically be served stale.
    // Trade-off: the access token stays visible in the address bar
    // after landing, instead of being replaced by a clean "/".
    const accessToken = url.searchParams.get("access");
    if (accessToken) {
      if (await constantTimeEqual(accessToken, env.MAGIC_LINK_TOKEN)) {
        return serveAuthenticatedSite(request, "guest", env.SESSION_SECRET);
      }
      // Wrong/expired token: fall through to a normal login page rather
      // than a dead end, in case the link was mistyped or copied wrong.
    }

    // Login form submission. Serves the real site directly in this same
    // response rather than redirecting, so there's no separate follow-up
    // request to "/" that could theoretically be served from a stale
    // cache instead of picking up the just-set session cookie.
    if (request.method === "POST") {
      const form = await request.formData();
      const password = form.get("password") || "";
      let role = null;
      if (await constantTimeEqual(password, env.ADMIN_PASSWORD)) role = "admin";
      else if (await constantTimeEqual(password, env.GUEST_PASSWORD)) role = "guest";

      if (role) {
        return serveAuthenticatedSite(request, role, env.SESSION_SECRET);
      }
      return new Response(renderLoginPage({ error: true }), {
        status: 401,
        headers: { "Content-Type": "text/html;charset=UTF-8", "Cache-Control": "no-store, must-revalidate" },
      });
    }

    // Check for an existing, valid session.
    const role = await verifySession(request, env.SESSION_SECRET);
    if (role) {
      // Authenticated -- pass through to the real site, but strip any
      // caching so the browser never serves a stale copy of this page
      // without re-checking the cookie first. Without this, logging out
      // could redirect to "/" and the browser could serve its own
      // locally cached copy of the logged-in page directly, without
      // ever sending a new request for the Worker to re-check at all.
      const response = await fetch(request, { cf: { cacheTtl: -1, cacheEverything: false } });
      const newResponse = new Response(response.body, response);
      newResponse.headers.set("Cache-Control", "no-store, must-revalidate");
      return newResponse;
    }

    // No valid session: show the login page. Status 200, not 401 --
    // this page itself loads successfully, it just happens to show a
    // login form. A 401 here can cause link-preview crawlers (iMessage,
    // etc.) to treat the fetch as failed and never read the OG tags in
    // the body at all, even though they're technically present.
    return new Response(renderLoginPage({ error: false }), {
      status: 200,
      headers: { "Content-Type": "text/html;charset=UTF-8", "Cache-Control": "no-store, must-revalidate" },
    });
  },
};

// ---------- Session cookie: HMAC-signed so it can't be forged by
// manually setting a cookie value in the browser (e.g. typing
// "octane_auth=admin" into DevTools) -- the Worker only trusts a cookie
// whose signature it can verify against SESSION_SECRET, which never
// reaches the browser. ----------

async function makeSessionCookie(role, secret) {
  const expiry = Date.now() + SESSION_DAYS * 24 * 60 * 60 * 1000;
  const payload = `${role}.${expiry}`;
  const signature = await hmacSign(payload, secret);
  const value = `${payload}.${signature}`;
  const maxAge = SESSION_DAYS * 24 * 60 * 60;
  return `${COOKIE_NAME}=${encodeURIComponent(value)}; Path=/; Max-Age=${maxAge}; HttpOnly; Secure; SameSite=Lax`;
}

async function verifySession(request, secret) {
  const cookieHeader = request.headers.get("Cookie") || "";
  const match = cookieHeader.match(new RegExp(`${COOKIE_NAME}=([^;]+)`));
  if (!match) return null;

  const value = decodeURIComponent(match[1]);
  const parts = value.split(".");
  if (parts.length !== 3) return null;
  const [role, expiry, signature] = parts;

  if (Date.now() > Number(expiry)) return null; // expired
  const expectedSignature = await hmacSign(`${role}.${expiry}`, secret);
  if (!(await constantTimeEqual(signature, expectedSignature))) return null; // tampered or forged

  if (role !== "guest" && role !== "admin") return null;
  return role;
}

async function hmacSign(message, secret) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return Array.from(new Uint8Array(signature)).map(b => b.toString(16).padStart(2, "0")).join("");
}

// Constant-time string comparison -- prevents a timing attack from
// being able to guess the password/token one character at a time based
// on how long the comparison takes to fail.
async function constantTimeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const enc = new TextEncoder();
  const aBytes = enc.encode(a);
  const bBytes = enc.encode(b);
  if (aBytes.length !== bBytes.length) {
    // Still do a comparison of equal-length dummy data so a length
    // mismatch doesn't return measurably faster than a length match.
    let dummy = 0;
    for (let i = 0; i < aBytes.length; i++) dummy |= aBytes[i] ^ (bBytes[i % Math.max(bBytes.length, 1)] || 0);
    return false;
  }
  let result = 0;
  for (let i = 0; i < aBytes.length; i++) result |= aBytes[i] ^ bBytes[i];
  return result === 0;
}

// ---------- Login page, matching the site's actual brand ----------

function renderLoginPage({ error }) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Octane Alpha | Quantitative UFC Analytics &amp; Edge Detection</title>

<!-- SVG offered FIRST: desktop Chrome prefers it and rasterises at whatever
     size the UI needs, so there's no small bitmap to render with artifacts.
     Version pinned to v=5 -- these were stuck at v=2, two logo changes ago,
     which is why the login page kept showing the old mark no matter what was
     regenerated. -->
<link rel="icon" type="image/svg+xml" href="/favicon.svg?v=6">
<link rel="icon" type="image/x-icon" href="/favicon.ico?v=6">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png?v=6">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png?v=6">
<link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png?v=6">
<link rel="icon" type="image/png" sizes="512x512" href="/icon-512.png?v=6">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png?v=6">

<meta property="og:title" content="Octane Alpha | Quantitative UFC Analytics &amp; Edge Detection">
<meta property="og:description" content="Model probability vs. live sportsbook lines. Real fight predictions, tracked publicly, checked against the market -- misses included.">
<meta property="og:image" content="https://octanealpha.com/og-share-card.png?v=6">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:url" content="https://octanealpha.com">
<meta property="og:type" content="website">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Octane Alpha | Quantitative UFC Analytics &amp; Edge Detection">
<meta name="twitter:description" content="Model probability vs. live sportsbook lines. Real fight predictions, tracked publicly, checked against the market -- misses included.">
<meta name="twitter:image" content="https://octanealpha.com/og-share-card.png?v=6">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c10; --panel: #14171f; --panel2: #1a1e28; --border: #262b36;
    --text: #eef0f2; --muted: #8a8f9a; --accent: #d4af37; --accent2: #e8c766;
    --red: #ff5c5c; --font-display: 'Space Grotesk', -apple-system, Segoe UI, Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at top, #12151c, #0a0c10 60%);
    color: var(--text); font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    padding: 24px;
  }
  .card {
    width: 100%; max-width: 360px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 16px; padding: 32px 28px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
  }
  .logo-row { display: flex; flex-direction: column; align-items: center; gap: 12px; margin-bottom: 24px; }
  /* Per-word colour rather than one gradient across both. The old
     background-clip gradient applied a single treatment to the whole
     wordmark, which is structurally why the login page, the site header and
     the share card could never agree -- it couldn't express two colours.
     Weight drops 900 -> 800 to match the site's weight pass. */
  .brand-name {
    font-size: 22px; font-weight: 800; letter-spacing: 1px; text-align: center;
  }
  .brand-name .bn-octane { color: #ffffff; }
  .brand-name .bn-alpha { color: var(--accent); text-shadow: 0 0 18px rgba(212,175,55,0.35); }
  .brand-tagline { font-size: 12px; color: var(--muted); letter-spacing: 0.5px; text-align: center; }
  input[type="password"] {
    width: 100%; padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border);
    background: var(--panel2); color: var(--text); font-size: 15px; margin-bottom: 14px;
    font-family: var(--font-display);
  }
  input[type="password"]:focus { outline: none; border-color: var(--accent); }
  button {
    width: 100%; padding: 13px; border-radius: 10px; border: none;
    background: linear-gradient(135deg, #f0d97a, #d4af37); color: #14171f;
    font-size: 14px; font-weight: 800; cursor: pointer; letter-spacing: 0.3px;
  }
  button:active { transform: scale(0.98); }
  .error {
    color: var(--red); font-size: 12.5px; text-align: center;
    margin-bottom: 14px; ${error ? "" : "display: none;"}
  }
  .footer-note { text-align: center; font-size: 11px; color: var(--muted); margin-top: 20px; opacity: 0.7; }
</style>
</head>
<body>
  <div class="card">
    <div class="logo-row">
      <!-- APEX. Only the octagon's top three edges are drawn, so the cage is
           implied rather than closed. White outer / gold apex mirrors the
           wordmark (Octane white, Alpha gold).
           viewBox is 0 0 100 100 -- the geometry is built on a 100-unit grid,
           so the old 0 0 44 44 box would crop it. -->
      <svg width="52" height="52" viewBox="0 0 100 100" fill="none" role="img" aria-label="Octane Alpha logo">
              <!-- translate(0,6.48): the mark's ink spans y 13.04-74, so its optical
           centre is 43.5 rather than 50. Drawn on the raw grid it floats high
           against the wordmark beside it. The generated icons apply the same
           shift, so header and favicon stay identical. -->
      <g transform="translate(0,6.48)">
        <path d="M13.04 65.31 L13.04 34.69 L34.69 13.04 L65.31 13.04 L86.96 34.69 L86.96 65.31"
              stroke="#ffffff" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M32 74 L50 38 L68 74"
              stroke="#d4af37" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
      </svg>
      <div>
        <div class="brand-name"><span class="bn-octane">OCTANE</span> <span class="bn-alpha">ALPHA</span></div>
        <div class="brand-tagline">Access restricted to authorized users</div>
      </div>
    </div>
    <form method="POST" action="/">
      <div class="error">Incorrect password — try again.</div>
      <input type="password" name="password" placeholder="Access Key" autofocus required>
      <button type="submit">Enter</button>
    </form>
    <div class="footer-note">Contact admin for access</div>
  </div>
</body>
</html>`;
}
