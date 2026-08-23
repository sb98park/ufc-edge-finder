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
const TTL_SECONDS = 30 * 60;

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
