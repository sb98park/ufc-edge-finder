/**
 * Refresh cadence tick.
 *
 * WHY THIS EXISTS: refresh.yml asks GitHub for a five-minute cron (the exact
 * expression is on the `- cron:` line there; it can't be written inside a
 * block comment because the slash would close it) and GitHub does not
 * deliver it. Measured across 2026-08-12..14, scheduled runs actually
 * arrived roughly ONCE AN HOUR -- typical gaps of 43-137 minutes against the
 * 5 minutes requested, i.e. about one tick in twelve. GitHub documents
 * `schedule` as best-effort and drops ticks under load, so this is working as
 * designed on their side; no amount of tuning the cron expression fixes it.
 *
 * The damage is subtler than "the site is slow". scripts/should_refresh.py
 * exists to RAISE the cadence to every 5 minutes inside 12h of a card, which
 * is when late money actually moves a line. At one delivered tick per hour
 * that tier is unreachable: by the time the gate is asked, the last build is
 * always more than 15 minutes old, so it answers BUILD every single time.
 * The throttle it was written to apply has been a no-op for days, and the
 * fight-night cadence it was written to enable has never once fired.
 *
 * So Cloudflare fires the tick instead, every 5 minutes, UNCONDITIONALLY.
 *
 * THIS WORKER CONTAINS NO CADENCE LOGIC, ON PURPOSE. It does not know when
 * the next card is and must not learn. should_refresh.py stays the single
 * source of truth for whether a build is warranted; teaching this file the
 * same window rules in JavaScript is precisely how the two would drift out
 * of sync, and the failure would be silent -- a Worker that thinks a card is
 * 30h away while the gate thinks it is 10h away just quietly under-refreshes
 * on fight night. The Worker's only job is guaranteeing the gate gets ASKED
 * on schedule. Everything downstream of "did the tick arrive" is Python's
 * decision.
 *
 * It fires repository_dispatch, NOT workflow_dispatch. That distinction is
 * load-bearing: refresh.yml treats workflow_dispatch as an always-builds
 * manual override (so the "Run workflow" button and the phone Shortcut stay
 * a guaranteed immediate refresh), while repository_dispatch is throttled
 * through the gate exactly like `schedule`. Point this at workflow_dispatch
 * instead and you get 288 unthrottled builds a day, which is the Pages
 * deployment saturation the cadence gate was built to end.
 *
 * COST: 288 invocations/day, far inside the Workers free tier. The GitHub
 * side is ~288 runs/day of which most exit at the gate in ~90 seconds
 * without calling the odds API, committing, or deploying.
 *
 * SECURITY: the token is a fine-grained GitHub PAT scoped to this one repo,
 * held as a Worker Secret (never hardcoded here). The fetch handler below is
 * a manual test trigger and is shared-secret protected so that discovering
 * the Worker's URL doesn't hand someone a build button.
 */

const OWNER = "sb98park";
const REPO = "ufc-edge-finder";
const DISPATCH_URL = `https://api.github.com/repos/${OWNER}/${REPO}/dispatches`;

/**
 * POST the repository_dispatch. Throws on anything that isn't a 204.
 *
 * Throwing rather than returning a status is deliberate -- see the scheduled
 * handler. A tick that fails needs to be loud.
 */
async function dispatchTick(env) {
  if (!env.GITHUB_TOKEN) {
    throw new Error(
      "GITHUB_TOKEN secret is not set on this Worker -- add it under " +
      "Settings -> Variables as a Secret (not a plain-text variable)."
    );
  }

  const resp = await fetch(DISPATCH_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
      // REQUIRED, and the single easiest way to lose an hour on this. The
      // GitHub API rejects requests that send no User-Agent, and the Workers
      // runtime does not supply a default one the way curl does. Omit this
      // line and every dispatch returns 403 with a body that does not
      // obviously point at the header as the cause.
      "User-Agent": `${REPO}-refresh-tick`,
    },
    // event_type is what refresh.yml matches on in `types: [tick]`. Changing
    // this string means changing it there too, or the dispatch succeeds with
    // a 204 and silently triggers nothing at all.
    body: JSON.stringify({ event_type: "tick" }),
  });

  // 204 No Content is the documented success response. An empty body here is
  // success, not failure -- do not "fix" this by checking for a JSON payload.
  if (resp.status !== 204) {
    const detail = (await resp.text()).slice(0, 300);
    throw new Error(`repository_dispatch failed: HTTP ${resp.status} ${detail}`);
  }
}

export default {
  // Cron Trigger entry point. The five-minute expression is set in the
  // Cloudflare dashboard, not here -- see SETUP.md for the exact string.
  async scheduled(event, env, ctx) {
    // NOT wrapped in try/catch, deliberately. An uncaught throw marks the
    // invocation failed and surfaces it in the Worker's Cron Events log with
    // the message attached.
    //
    // The failure mode this protects against is the expensive one: a PAT
    // silently expires, every dispatch starts 401ing, and the site quietly
    // stops refreshing while GitHub's Actions tab shows nothing wrong at all
    // -- because from GitHub's point of view nothing IS wrong, it simply
    // stopped being asked. That is the same class of invisible starvation
    // that cost 12 hours of stale odds two days before a card. Swallowing
    // errors here would rebuild it exactly.
    await dispatchTick(env);
  },

  /**
   * Manual test trigger, so setup can be verified in seconds instead of by
   * deploying and waiting out a 5-minute cron window wondering which half is
   * broken. Token-protected: without this check, anyone who found the URL
   * could queue builds on the repo.
   */
  async fetch(request, env) {
    const token = new URL(request.url).searchParams.get("token");
    // Null-checked before .trim() -- a missing ?token= should be a clean 403,
    // not a runtime TypeError surfacing as an opaque 500.
    if (!env.TICK_TOKEN || !token || token.trim() !== env.TICK_TOKEN.trim()) {
      return new Response("Forbidden\n", { status: 403 });
    }

    try {
      await dispatchTick(env);
      return new Response(
        "OK -- repository_dispatch (event_type=tick) accepted, HTTP 204.\n" +
        "Check the repo's Actions tab: a run should appear within seconds.\n"
      );
    } catch (err) {
      // 502: this Worker is fine, the upstream call it makes is not.
      return new Response(`${err.message}\n`, { status: 502 });
    }
  },
};
