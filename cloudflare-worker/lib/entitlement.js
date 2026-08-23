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
// The display label is mirrored under its own prefix so it can be dropped
// independently and can never be mistaken for the entitlement bit.
const PLAN_PREFIX = "plan:";
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
 * What to CALL this account's access: "basic" | "pro" | "lifetime".
 *
 * DISPLAY ONLY. isMember() above remains the sole gate on what anyone can
 * see; this decides a word in the account panel and nothing else. That is
 * why it is allowed to be wrong in a way isMember() is not -- every failure
 * path here returns the LEAST flattering answer that is still consistent
 * with the entitlement we already decided, so a lookup failure can never
 * upgrade someone's label.
 *
 * Takes `member` rather than re-deriving it: the caller has already resolved
 * entitlement through the cached, fail-closed path, and asking twice invites
 * the two answers to disagree.
 */
export async function accountPlan(userId, member, env, ctx) {
  if (!userId || !member) return "basic";

  const key = PLAN_PREFIX + userId;
  try {
    const cached = await env.OCTANE_ENTITLEMENTS.get(key);
    if (cached === "pro" || cached === "lifetime" || cached === "basic") return cached;
  } catch { /* fall through and ask */ }

  let plan = "pro";        // entitled but unclassified: the ordinary paid case
  try {
    const res = await fetch(`${env.SUPABASE_URL}/rest/v1/rpc/account_plan`, {
      method: "POST",
      headers: {
        "apikey": env.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ uid: userId }),
    });
    if (res.ok) {
      const value = await res.json();
      if (value === "pro" || value === "lifetime") plan = value;
      // A "basic" here would contradict isMember(), which already said this
      // account is entitled. Trust the gate, not the label, and say so.
      else if (value === "basic") {
        console.log(`account_plan says basic for entitled user ${userId}`);
      }
    }
  } catch (err) {
    console.log(`account_plan lookup failed for ${userId}: ${err.message}`);
  }

  const write = env.OCTANE_ENTITLEMENTS.put(key, plan, { expirationTtl: KV_TTL_SECONDS });
  if (ctx && ctx.waitUntil) ctx.waitUntil(write); else await write.catch(() => {});
  return plan;
}

/**
 * Invalidate one account's mirror. Called by the Stripe webhook so a new
 * subscription, a cancellation or a failed payment takes effect on the next
 * request rather than up to KV_TTL_SECONDS later.
 */
export async function forgetEntitlement(userId, env) {
  try { await env.OCTANE_ENTITLEMENTS.delete(KV_PREFIX + userId); } catch {}
  // The label is mirrored separately, so it has to be dropped separately --
  // otherwise a cancelled subscription loses access immediately but keeps
  // saying "Pro" until the TTL runs out.
  try { await env.OCTANE_ENTITLEMENTS.delete(PLAN_PREFIX + userId); } catch {}
}
