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

import { verifySupabaseToken } from "./lib/jwt.js";
import { mintSession, readSession, clearSession } from "./lib/session.js";
import { isMember, forgetEntitlement } from "./lib/entitlement.js";
import { trialEndTimestamp, findOrCreateCustomer, createCheckoutSession,
         createPortalSession, verifyWebhook } from "./lib/stripe.js";
import { getProfile, setStripeCustomer, markTrialUsed,
         findUserByCustomer, upsertSubscription } from "./lib/db.js";

/*
 * TWO GATES IN ONE WORKER, chosen by env.GATE_MODE.
 *
 *   "password"  the original shared-password wall. Still the default, so
 *               deploying this file changes nothing about who can reach the
 *               site.
 *   "auth"      real accounts: Supabase identity, subscription entitlement,
 *               and the member payload streamed from R2.
 *
 * The flag exists because switching the two at once would mean the first
 * deployment of the new gate is also the moment the site becomes public. If
 * anything about sign-in, entitlement or R2 is wrong, that is discovered by
 * strangers rather than by us. With the flag, "auth" can be exercised end to
 * end on the live domain while the wall is still up, and the rollback is one
 * variable rather than a redeploy under pressure.
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
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const mode = env.GATE_MODE === "auth" ? "auth" : "password";

    // ACCOUNT AUTH RUNS IN BOTH MODES. Only the WALL is conditional.
    //
    // The obvious design -- password mode or auth mode, never both -- has a
    // hole in it: the only way to test sign-in, entitlement and R2 delivery
    // against the real domain would be to switch off the wall first, making
    // the verification step and the going-public step the same step. Running
    // auth underneath the wall means the owner can sign in, be recognised as
    // a member and be served the R2 payload while strangers still meet the
    // password prompt. Flipping GATE_MODE then only removes the wall, and it
    // removes it from something already known to work.
    const handled = await handleAuthMode(request, env, ctx, url, path, mode);
    if (handled) return handled;

    // Wall is off: the origin holds the free build, which is what everyone
    // without a member session should get.
    if (mode === "auth") return fetch(request);

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


/* ===================================================================== *
 *  AUTH MODE
 * ===================================================================== */

// Never gated, in either mode. The service worker and manifest are fetched
// before any cookie exists, and /welcome is the marketing page whose entire
// job is to be reachable by strangers.
const ALWAYS_PUBLIC = ["/sw.js", "/offline.html", "/welcome", "/welcome.html", "/favicon.svg"];

/**
 * Returns a Response when it owns the request, or null to fall through.
 */
async function handleAuthMode(request, env, ctx, url, path, mode) {
  // --- exchange a Supabase token for our own session -------------------
  if (path === "/auth/session" && request.method === "POST") {
    let body;
    try { body = await request.json(); } catch { return json({ error: "bad request" }, 400); }

    let claims = null;
    try {
      claims = await verifySupabaseToken(body && body.access_token, env.SUPABASE_URL);
    } catch (err) {
      // Distinguishable from an invalid token: this is the JWKS being
      // unreachable, which is our problem rather than the caller's.
      console.log(`jwks failure: ${err.message}`);
      return json({ error: "verification unavailable" }, 503);
    }
    if (!claims) return json({ error: "invalid token" }, 401);

    const member = await isMember(claims.sub, env, ctx);
    const headers = new Headers({ "Content-Type": "application/json", "Cache-Control": "no-store" });
    headers.append("Set-Cookie", await mintSession({ userId: claims.sub, member }, env.SESSION_SECRET));
    return new Response(JSON.stringify({ member }), { status: 200, headers });
  }

  // --- start a subscription --------------------------------------------
  if (path === "/billing/checkout" && request.method === "POST") {
    const session = await readSession(request, env.SESSION_SECRET);
    if (!session) return json({ error: "sign in first" }, 401);

    let plan = "year";
    try { plan = (await request.json()).plan === "month" ? "month" : "year"; } catch {}

    // TWO SETTINGS THAT MUST AGREE, ENFORCED RATHER THAN REMEMBERED.
    //
    // A public site (GATE_MODE=auth) running on test Stripe keys sends real
    // visitors to a checkout page stamped TEST MODE that cannot take their
    // money. Nothing else in the system couples those two values, so without
    // this the only thing preventing it is remembering to change both -- and
    // the failure is discovered by a customer, mid-purchase, at the exact
    // moment they were willing to pay.
    //
    // Refusing here is the safe direction: it breaks the subscribe button
    // rather than taking a payment that cannot complete, and it says exactly
    // what is wrong instead of failing mysteriously.
    if (env.GATE_MODE === "auth" && String(env.STRIPE_SECRET_KEY || "").startsWith("sk_test_")) {
      console.log("REFUSING CHECKOUT: gate is public but Stripe is in test mode");
      return json({ error: "billing is not live yet" }, 503);
    }

    // THE PRICE IS CHOSEN HERE, NOT SENT BY THE CLIENT. The page's monthly/
    // annual toggle is presentation only; if the browser named the price, a
    // tampered request would buy the annual plan at the monthly price.
    const priceId = plan === "month" ? env.STRIPE_PRICE_MONTHLY : env.STRIPE_PRICE_ANNUAL;

    try {
      const profile = await getProfile(session.userId, env);
      if (!profile) return json({ error: "no profile" }, 404);

      const customerId = await findOrCreateCustomer({
        email: profile.email, userId: session.userId,
        existingId: profile.stripe_customer_id,
      }, env);
      if (customerId !== profile.stripe_customer_id) {
        await setStripeCustomer(session.userId, customerId, env);
      }

      // ONE TRIAL PER ACCOUNT, ENFORCED HERE. Stripe would happily grant a
      // fresh trial to a new customer object, and a new customer object is
      // one new email address away -- so eligibility lives in our database.
      const allowTrial = !profile.trial_used_at;
      const nextEvent = await env.OCTANE_ENTITLEMENTS.get("next_event_date");

      const checkout = await createCheckoutSession({
        customerId, priceId, userId: session.userId,
        trialEnd: trialEndTimestamp(nextEvent),
        origin: url.origin, allowTrial,
      }, env);

      return json({ url: checkout.url, trial: allowTrial });
    } catch (err) {
      console.log(`checkout failed: ${err.message}`);
      return json({ error: "could not start checkout" }, 502);
    }
  }

  // --- manage an existing subscription ----------------------------------
  if (path === "/billing/portal" && request.method === "POST") {
    const session = await readSession(request, env.SESSION_SECRET);
    if (!session) return json({ error: "sign in first" }, 401);
    try {
      const profile = await getProfile(session.userId, env);
      if (!profile || !profile.stripe_customer_id) {
        return json({ error: "no subscription" }, 404);
      }
      const portal = await createPortalSession({
        customerId: profile.stripe_customer_id, origin: url.origin,
      }, env);
      return json({ url: portal.url });
    } catch (err) {
      console.log(`portal failed: ${err.message}`);
      return json({ error: "could not open portal" }, 502);
    }
  }

  // --- Stripe webhook ---------------------------------------------------
  if (path === "/stripe/webhook" && request.method === "POST") {
    // RAW TEXT, NOT JSON. The signature covers the exact bytes Stripe sent;
    // parsing and re-serialising changes them and verification fails.
    const raw = await request.text();
    const event = await verifyWebhook(raw, request.headers.get("Stripe-Signature"),
                                      env.STRIPE_WEBHOOK_SECRET);
    // Unsigned or stale: 400, and deliberately no detail. This endpoint is
    // public, and a descriptive error is a hint for someone probing it.
    if (!event) return json({ error: "bad signature" }, 400);

    try {
      await applyStripeEvent(event, env);
    } catch (err) {
      // 500 so Stripe RETRIES. Swallowing the error would return 200 and
      // lose the event permanently -- a subscription that silently never
      // grants access.
      console.log(`webhook ${event.type} failed: ${err.message}`);
      return json({ error: "handler failed" }, 500);
    }
    return json({ received: true });
  }

  // --- sign out ---------------------------------------------------------
  if (path === "/auth/logout") {
    const headers = new Headers({ "Location": "/", "Cache-Control": "no-store" });
    headers.append("Set-Cookie", clearSession());
    return new Response(null, { status: 302, headers });
  }

  // --- who am I (used by the page to render signed-in state) ------------
  if (path === "/auth/whoami") {
    const session = await readSession(request, env.SESSION_SECRET);
    return json(session ? { signedIn: true, member: session.member } : { signedIn: false });
  }

  // Only once the wall is down. While it is up, /welcome is not yet meant to
  // be reachable by strangers -- the site has not launched -- so it stays
  // behind the password like everything else.
  if (mode === "auth"
      && (ALWAYS_PUBLIC.includes(path) || path.startsWith("/splash-") || path === "/manifest.json")) {
    return fetch(request);
  }

  // --- the page itself --------------------------------------------------
  // Only the document is tiered. Every other asset -- icons, movement
  // fragments -- is identical in both builds and served straight from the
  // origin, so there is no reason to route it through entitlement.
  const wantsDocument = request.method === "GET"
    && (path === "/" || path === "/index.html")
    && (request.headers.get("Accept") || "").includes("text/html");
  if (!wantsDocument) return null;

  const session = await readSession(request, env.SESSION_SECRET);
  if (!session || !session.member) return null;      // free build from the origin

  const object = await env.MEMBER_PAYLOAD.get("index.html");
  if (!object) {
    // FAIL CLOSED. A missing member payload means the build has not
    // uploaded yet; serving the origin is correct, because the origin holds
    // the free build. The alternative -- erroring -- would take the site
    // down for members over a transient publishing gap.
    console.log("member payload missing from R2");
    return null;
  }

  return new Response(object.body, {
    status: 200,
    headers: {
      "Content-Type": "text/html;charset=UTF-8",
      // Private: this response is specific to one entitled user and must
      // never be held by a shared cache.
      "Cache-Control": "private, no-store",
      "X-Octane-Tier": "member",
    },
  });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}


/**
 * Apply one Stripe event to our own records.
 *
 * Only subscription state is acted on. Stripe sends dozens of event types and
 * subscribing to all of them means writing handlers for things that do not
 * affect entitlement -- so anything unrecognised is acknowledged and ignored
 * rather than treated as an error Stripe should retry.
 */
async function applyStripeEvent(event, env) {
  const obj = event.data && event.data.object;
  if (!obj) return;

  if (event.type === "checkout.session.completed") {
    // The trial is only marked used once checkout actually completes -- not
    // when the session is created -- so an abandoned checkout does not burn
    // the user's one trial.
    const userId = obj.client_reference_id;
    if (userId) {
      if (obj.customer) await setStripeCustomer(userId, obj.customer, env);
      await markTrialUsed(userId, env);
      await forgetEntitlement(userId, env);
    }
    return;
  }

  if (event.type.startsWith("customer.subscription.")) {
    const userId = (obj.metadata && obj.metadata.supabase_user_id)
      || await findUserByCustomer(obj.customer, env);
    if (!userId) {
      console.log(`no user for stripe customer ${obj.customer}`);
      return;
    }

    const item = obj.items && obj.items.data && obj.items.data[0];

    // WHERE THE BILLING PERIOD LIVES DEPENDS ON THE API VERSION. Stripe moved
    // current_period_end off the subscription and onto each subscription ITEM
    // in a recent version, and the account here is pinned to a 2026 one. Read
    // whichever is present rather than betting on the shape: the failure mode
    // if this is wrong is silent -- entitlement still works, because
    // is_member() treats a null period end as "no known expiry", so nothing
    // breaks visibly while access quietly stops expiring when it should.
    const periodEnd = obj.current_period_end
      || (item && item.current_period_end)
      || null;

    await upsertSubscription({
      user_id: userId,
      stripe_subscription_id: obj.id,
      status: obj.status,
      price_id: item ? item.price.id : "",
      plan_interval: item && item.price.recurring ? item.price.recurring.interval : "month",
      trial_end: obj.trial_end ? new Date(obj.trial_end * 1000).toISOString() : null,
      current_period_end: periodEnd ? new Date(periodEnd * 1000).toISOString() : null,
      cancel_at_period_end: Boolean(obj.cancel_at_period_end),
    }, env);

    // Entitlement just changed; drop the mirror so the next request reads
    // the truth instead of a cached answer up to 15 minutes stale.
    await forgetEntitlement(userId, env);
    return;
  }

  if (event.type === "invoice.payment_failed" && obj.customer) {
    const userId = await findUserByCustomer(obj.customer, env);
    if (userId) await forgetEntitlement(userId, env);
  }
}
