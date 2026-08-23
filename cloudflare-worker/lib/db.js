/*
 * Supabase access from the Worker, via PostgREST.
 *
 * Everything here uses the service role, which bypasses row level security.
 * That is necessary -- the Worker acts on behalf of a user without holding
 * their token, and the RLS policies are written for a user session -- and it
 * is also why this file is deliberately small. Every function is a specific
 * operation with a specific reason to exist rather than a general "run SQL"
 * escape hatch, so the set of things the service role is used for stays
 * readable in one screen.
 */

function headers(env, extra = {}) {
  return {
    "apikey": env.SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

async function rest(path, init, env) {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1${path}`, init);
  if (!res.ok) throw new Error(`supabase ${path}: ${res.status} ${await res.text()}`);
  return res.status === 204 ? null : res.json();
}

export async function getProfile(userId, env) {
  const rows = await rest(
    `/profiles?id=eq.${userId}&select=id,email,stripe_customer_id,trial_used_at,lifetime_access`,
    { headers: headers(env) }, env
  );
  return rows && rows[0] ? rows[0] : null;
}

export async function setStripeCustomer(userId, customerId, env) {
  await rest(`/profiles?id=eq.${userId}`, {
    method: "PATCH",
    headers: headers(env, { Prefer: "return=minimal" }),
    body: JSON.stringify({ stripe_customer_id: customerId }),
  }, env);
}

export async function markTrialUsed(userId, env) {
  await rest(`/profiles?id=eq.${userId}&trial_used_at=is.null`, {
    method: "PATCH",
    headers: headers(env, { Prefer: "return=minimal" }),
    body: JSON.stringify({ trial_used_at: new Date().toISOString() }),
  }, env);
}

export async function findUserByCustomer(customerId, env) {
  const rows = await rest(
    `/profiles?stripe_customer_id=eq.${encodeURIComponent(customerId)}&select=id`,
    { headers: headers(env) }, env
  );
  return rows && rows[0] ? rows[0].id : null;
}

/**
 * Write a subscription's current state.
 *
 * UPSERT ON stripe_subscription_id, NOT INSERT. Stripe delivers webhooks at
 * least once, not exactly once, and it retries on any non-2xx -- so the same
 * event WILL arrive twice sooner or later. An insert would either duplicate
 * the row or fail the retry, and a failed retry means Stripe keeps resending
 * an event we have already handled. Upsert makes redelivery a no-op, which is
 * the only sane way to treat a system with at-least-once semantics.
 */
export async function upsertSubscription(row, env) {
  await rest(`/subscriptions?on_conflict=stripe_subscription_id`, {
    method: "POST",
    headers: headers(env, {
      Prefer: "resolution=merge-duplicates,return=minimal",
    }),
    body: JSON.stringify({ ...row, updated_at: new Date().toISOString() }),
  }, env);
}
