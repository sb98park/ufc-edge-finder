/*
 * The Worker's own session cookie.
 *
 * WHY MINT OUR OWN RATHER THAN CARRY THE SUPABASE TOKEN. Verifying an ES256
 * signature is an elliptic-curve operation on every single page request, and
 * entitlement still has to be looked up separately. Exchanging the Supabase
 * token once for a short-lived cookie of our own means the hot path is a
 * single HMAC verify over ~60 bytes -- the same cost the password gate
 * already paid, so the site does not get slower by becoming authenticated.
 *
 * The cookie carries the entitlement decision, not just identity. That is the
 * point: it is what lets a page request answer "member or not" with no KV
 * read, no database call and no network at all.
 *
 * SHORT TTL IS LOAD-BEARING. Because entitlement is baked in, a cancelled
 * subscriber keeps access until their cookie expires. Thirty minutes bounds
 * that. Longer would be cheaper and would mean someone who cancelled at 9am
 * still reading paid content at lunchtime.
 */

const COOKIE = "octane_session";

/*
 * THIRTY MINUTES WITH NO RENEWAL SIGNED PEOPLE OUT EVERY THIRTY MINUTES.
 * mintSession had exactly one call site -- the sign-in exchange -- so the
 * clock ran from the moment you signed in and no amount of using the app
 * extended it. On a home-screen PWA, which reloads when you return to it
 * after a spell in the background, that surfaced as being logged out
 * roughly every time you opened the thing. It read as deploy-related only
 * because the refresh job runs on a similar cadence.
 *
 * The window is now a week, and it SLIDES: every whoami and every member
 * page load re-checks entitlement and re-mints. See renewIfNeeded below for
 * why a longer cookie does not weaken the guarantee the old TTL was buying.
 */
const TTL_SECONDS = 7 * 24 * 60 * 60;

/*
 * Re-mint once the cookie is past halfway. Renewing on literally every
 * request would work too, but it puts a Set-Cookie on every response for no
 * benefit; half-life means an active session is always refreshed well before
 * it can lapse while most requests carry no cookie header at all.
 */
export function isPastHalfLife(session) {
  if (!session || !session.expiry) return true;
  const remaining = session.expiry - Date.now();
  return remaining < (TTL_SECONDS * 1000) / 2;
}

function bytesToHex(buf) {
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function hmac(message, secret) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  return bytesToHex(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message)));
}

// Constant-time comparison, so a forged signature cannot be refined one
// character at a time by measuring how long the rejection takes.
function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function mintSession({ userId, member }, secret) {
  const expiry = Date.now() + TTL_SECONDS * 1000;
  const tier = member ? "member" : "free";
  const payload = `2.${userId}.${tier}.${expiry}`;      // version prefix: see readSession
  const value = `${payload}.${await hmac(payload, secret)}`;
  return `${COOKIE}=${encodeURIComponent(value)}; Path=/; Max-Age=${TTL_SECONDS}; HttpOnly; Secure; SameSite=Lax`;
}

export function clearSession() {
  return `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax`;
}

export async function readSession(request, secret) {
  const raw = (request.headers.get("Cookie") || "").match(new RegExp(`${COOKIE}=([^;]+)`));
  if (!raw) return null;

  const parts = decodeURIComponent(raw[1]).split(".");
  // VERSION PREFIX. The password gate's cookie was `role.expiry.signature`,
  // three parts. This is five. Without the prefix, a stale cookie from the
  // old scheme could be parsed as a malformed new one; with it, anything that
  // is not version 2 is rejected outright and the user simply signs in again.
  if (parts.length !== 5 || parts[0] !== "2") return null;

  const [, userId, tier, expiry, signature] = parts;
  if (!(Number(expiry) > Date.now())) return null;
  if (tier !== "member" && tier !== "free") return null;

  const expected = await hmac(`2.${userId}.${tier}.${expiry}`, secret);
  if (!timingSafeEqual(signature, expected)) return null;

  return { userId, member: tier === "member", expiry: Number(expiry) };
}
