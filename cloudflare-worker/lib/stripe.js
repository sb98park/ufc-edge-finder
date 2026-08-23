/*
 * Stripe: checkout, the customer portal, and webhook verification.
 *
 * The Stripe API is form-encoded, not JSON, which is easy to forget and
 * produces confusing 400s when you do -- hence the single request helper
 * every call goes through.
 */

const API = "https://api.stripe.com/v1";

/** Flatten a nested object into Stripe's bracket notation. */
function toForm(params, prefix = "", out = new URLSearchParams()) {
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    const key = prefix ? `${prefix}[${k}]` : k;
    if (typeof v === "object" && !Array.isArray(v)) toForm(v, key, out);
    else if (Array.isArray(v)) v.forEach((item, i) => {
      if (typeof item === "object") toForm(item, `${key}[${i}]`, out);
      else out.append(`${key}[${i}]`, String(item));
    });
    else out.append(key, String(v));
  }
  return out;
}

async function stripe(path, params, env, method = "POST") {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: {
      "Authorization": `Bearer ${env.STRIPE_SECRET_KEY}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: method === "POST" ? toForm(params || {}).toString() : undefined,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(`stripe ${path}: ${body.error?.message || res.status}`);
  return body;
}

/**
 * When the trial should end.
 *
 * THE WHOLE POINT IS THAT A TRIAL MUST CONTAIN A FIGHT CARD. A flat seven
 * days sounds fine and quietly fails on a gap week: someone who subscribes
 * the day after an event, with the next one nine days out, trials the product
 * without ever seeing it do the thing they are paying for, and then churns
 * having formed an opinion about a version of it that never showed up.
 *
 * So: a day after the next card, but never less than seven days. With UFC's
 * weekly cadence the seven-day floor usually wins, and the event clause only
 * bites in exactly the case it exists for.
 *
 * nextEventDate comes from KV, published by the build. If it is missing we
 * fall back to seven days rather than failing the checkout -- a slightly
 * mistimed trial is a far better outcome than a subscribe button that errors.
 */
export function trialEndTimestamp(nextEventDate) {
  const WEEK = 7 * 24 * 60 * 60 * 1000;
  const floor = Date.now() + WEEK;
  if (!nextEventDate) return Math.floor(floor / 1000);

  const event = Date.parse(`${nextEventDate}T12:00:00Z`);
  if (!Number.isFinite(event)) return Math.floor(floor / 1000);

  // Noon UTC the day after, so a card running past midnight local time is
  // still comfortably inside the trial.
  const dayAfter = event + 24 * 60 * 60 * 1000;
  return Math.floor(Math.max(floor, dayAfter) / 1000);
}

export async function findOrCreateCustomer({ email, userId, existingId }, env) {
  if (existingId) return existingId;
  const customer = await stripe("/customers", {
    email,
    // The Supabase user id travels with the customer so a webhook can map
    // back without a database lookup, and so a human reading the Stripe
    // dashboard can tell who a customer actually is.
    metadata: { supabase_user_id: userId },
  }, env);
  return customer.id;
}

export async function createCheckoutSession(
  { customerId, priceId, trialEnd, userId, origin, allowTrial }, env
) {
  const params = {
    mode: "subscription",
    customer: customerId,
    line_items: [{ price: priceId, quantity: 1 }],
    success_url: `${origin}/?checkout=success`,
    cancel_url: `${origin}/?checkout=cancelled`,
    client_reference_id: userId,
    // CARD REQUIRED, EVEN FOR THE TRIAL. Card-less trials widen the top of
    // the funnel and convert far worse, and they make trial farming free.
    payment_method_collection: "always",
    allow_promotion_codes: false,
    subscription_data: {
      metadata: { supabase_user_id: userId },
    },
  };

  if (allowTrial) {
    params.subscription_data.trial_end = trialEnd;
    // If the card is gone by the time the trial ends, cancel rather than
    // leaving a subscription that can never be collected.
    params.subscription_data.trial_settings = {
      end_behavior: { missing_payment_method: "cancel" },
    };
  }

  return stripe("/checkout/sessions", params, env);
}

export async function createPortalSession({ customerId, origin }, env) {
  return stripe("/billing_portal/sessions", {
    customer: customerId,
    return_url: `${origin}/`,
  }, env);
}

/**
 * Verify a webhook actually came from Stripe.
 *
 * WITHOUT THIS, THE ENDPOINT IS AN OPEN GRANT. It writes subscription rows,
 * so anyone who can POST to it could hand themselves a membership by
 * inventing a checkout.session.completed. The signature is the only thing
 * separating those two cases, and it must be computed over the RAW body --
 * parsing and re-serialising changes bytes and the signature stops matching,
 * which is why the caller passes text rather than an object.
 */
export async function verifyWebhook(rawBody, signatureHeader, secret) {
  if (!signatureHeader) return null;

  const parts = Object.fromEntries(
    signatureHeader.split(",").map((p) => p.split("=").map((s) => s.trim()))
  );
  const timestamp = parts.t;
  const signature = parts.v1;
  if (!timestamp || !signature) return null;

  // REPLAY WINDOW. A signature stays valid forever otherwise, so a captured
  // request could be resent indefinitely -- re-granting a membership that was
  // since cancelled.
  const age = Math.abs(Date.now() / 1000 - Number(timestamp));
  if (!Number.isFinite(age) || age > 300) return null;

  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const mac = await crypto.subtle.sign(
    "HMAC", key, new TextEncoder().encode(`${timestamp}.${rawBody}`)
  );
  const expected = Array.from(new Uint8Array(mac))
    .map((b) => b.toString(16).padStart(2, "0")).join("");

  if (expected.length !== signature.length) return null;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= expected.charCodeAt(i) ^ signature.charCodeAt(i);
  if (diff !== 0) return null;

  try { return JSON.parse(rawBody); } catch { return null; }
}
