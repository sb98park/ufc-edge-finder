/*
 * Is this account entitled to member content?
 *
 * TWO LAYERS, ONE TRUTH. Supabase is authoritative; KV is a mirror that makes
 * the answer available in about a millisecond at the edge. The mirror is
 * written by the Stripe webhook when entitlement actually changes, and
 * back-filled here on a miss, so losing the whole namespace costs one slow
 * request per user rather than correctness.
 *
 * FAILS CLOSED, ALWAYS. Every error path in this file returns "not a member".
 * A paywall that fails open is not a paywall -- an outage would quietly make
 * the paid content free, and nobody would notice until it had been indexed.
 * The opposite failure, a member briefly seeing the free build during a
 * Supabase incident, is recoverable by reloading.
 */

const KV_PREFIX = "ent:";
const KV_TTL_SECONDS = 15 * 60;   // upper bound on how stale the mirror can be

/**
 * Ask Supabase directly. Uses the service role, which bypasses row level
 * security -- necessary because the Worker is acting on behalf of the user
 * without their token, and the RLS policies are written for a user session.
 *
 * is_member() is the same function the database uses, so entitlement is
 * defined in exactly one place. It already accounts for lifetime grants and
 * for past_due counting as entitled during dunning.
 */
async function askSupabase(userId, env) {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1/rpc/is_member`, {
    method: "POST",
    headers: {
      "apikey": env.SUPABASE_SERVICE_ROLE_KEY,
      "Authorization": `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ uid: userId }),
  });
  if (!res.ok) throw new Error(`is_member ${res.status}`);
  return (await res.json()) === true;
}

export async function isMember(userId, env, ctx) {
  if (!userId) return false;

  const key = KV_PREFIX + userId;

  try {
    const cached = await env.OCTANE_ENTITLEMENTS.get(key);
    if (cached === "1") return true;
    if (cached === "0") return false;
  } catch {
    // A KV read failure is not a reason to deny a paying member -- fall
    // through and ask the source of truth instead.
  }

  let member = false;
  try {
    member = await askSupabase(userId, env);
  } catch (err) {
    console.log(`entitlement lookup failed for ${userId}: ${err.message}`);
    return false;                                   // fail closed
  }

  // Written after the response is already decided, so populating the mirror
  // never delays the page.
  const write = env.OCTANE_ENTITLEMENTS.put(key, member ? "1" : "0",
                                            { expirationTtl: KV_TTL_SECONDS });
  if (ctx && ctx.waitUntil) ctx.waitUntil(write); else await write.catch(() => {});

  return member;
}

/**
 * Invalidate one account's mirror. Called by the Stripe webhook so a new
 * subscription, a cancellation or a failed payment takes effect on the next
 * request rather than up to KV_TTL_SECONDS later.
 */
export async function forgetEntitlement(userId, env) {
  try { await env.OCTANE_ENTITLEMENTS.delete(KV_PREFIX + userId); } catch {}
}
