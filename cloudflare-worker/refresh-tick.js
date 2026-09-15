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
const RUNS_URL = `https://api.github.com/repos/${OWNER}/${REPO}/actions/runs`;

/**
 * How long a run may sit in `queued` before this treats it as wedged.
 *
 * A queued run is doing no work by definition -- it has no runner and no
 * job. refresh.yml's own `timeout-minutes: 18` cannot bound it, because that
 * clock only starts once a job is RUNNING. So the one failure mode that
 * silences every guard inside the workflow is invisible to all of them.
 *
 * 30 minutes clears everything legitimate with room: a build takes 2-5
 * minutes, the job cap is 18, and brief queueing while the previous run
 * finishes is normal. Anything past that is not waiting, it is stuck.
 */
const STUCK_QUEUED_MINUTES = 30;

function ghHeaders(env) {
  return {
    Authorization: `Bearer ${env.GITHUB_TOKEN}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    // Same reason as the dispatch below: the Workers runtime sends no
    // default User-Agent and GitHub 403s requests without one.
    "User-Agent": `${REPO}-refresh-tick`,
  };
}

/**
 * Cancel runs wedged in `queued`, and say how many. Never throws.
 *
 * WHY THIS LIVES IN THE WORKER. On 2026-09-13 a run sat queued for 8.7
 * hours holding refresh.yml's `refresh` concurrency group. That group is
 * declared `cancel-in-progress: false`, so GitHub keeps only the newest
 * PENDING run and cancels the rest: 99 consecutive runs cancelled, zero
 * successes, the site stale for nine hours. Nothing inside the workflow
 * could report it, because nothing inside the workflow ran. This Worker is
 * the only piece that keeps executing when Actions is wedged, so it is the
 * only place a guard can sit.
 *
 * ONLY `queued`, NEVER `in_progress`. An in-progress run may be mid-commit
 * or mid-push, and cancelling one risks leaving the repo half-written --
 * which is precisely why refresh.yml chose cancel-in-progress: false in the
 * first place. This must not quietly undo that decision.
 *
 * NEVER THROWS, and that is load-bearing. The tick is the primary job; a
 * broken health check must not stop the site refreshing. Failures are
 * returned for the caller to log, not raised.
 */
async function unwedgeStuckRuns(env, now = Date.now()) {
  const out = { checked: 0, cancelled: [], error: null };
  try {
    const resp = await fetch(`${RUNS_URL}?status=queued&per_page=50`, { headers: ghHeaders(env) });
    if (!resp.ok) {
      out.error = `list queued runs: HTTP ${resp.status}`;
      return out;
    }
    const body = await resp.json();
    const runs = Array.isArray(body?.workflow_runs) ? body.workflow_runs : [];
    out.checked = runs.length;
    for (const run of runs) {
      const startedAt = Date.parse(run?.created_at ?? "");
      if (!startedAt) continue;
      const minutes = (now - startedAt) / 60000;
      if (minutes < STUCK_QUEUED_MINUTES) continue;
      const cancel = await fetch(`${RUNS_URL}/${run.id}/cancel`, {
        method: "POST",
        headers: ghHeaders(env),
      });
      // 202 Accepted is the documented success. 409 means it already moved
      // on between the list and the cancel, which is a race this does not
      // need to care about -- the wedge is gone either way.
      if (cancel.status === 202 || cancel.status === 409) {
        out.cancelled.push({ id: run.id, number: run.run_number, minutes: Math.round(minutes) });
      } else if (cancel.status === 403) {
        // THE ONE FAILURE THAT LOOKS LIKE NO FAILURE. Listing runs needs
        // `actions: read` and cancelling needs `actions: write`. A PAT with
        // only read sails through the list above, finds the wedged run, and
        // is refused here -- so the guard appears installed, logs a bare
        // "HTTP 403", and cancels nothing for as long as nobody reads Cron
        // Events. That is the same invisible-starvation shape as a PAT
        // silently expiring, which this file already exists to prevent.
        out.error =
          `cancel ${run.id}: HTTP 403 -- the token listed runs but may not ` +
          `cancel them. That is the \`actions: write\` permission on the ` +
          `fine-grained PAT in GITHUB_TOKEN; read alone is not enough. ` +
          `Verify with GET /?token=...&check=queue`;
        out.scopeDenied = true;
      } else {
        out.error = `cancel ${run.id}: HTTP ${cancel.status}`;
      }
    }
  } catch (err) {
    out.error = String(err?.message ?? err);
  }
  return out;
}

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

/**
 * Can this token actually CANCEL, without cancelling anything?
 *
 * The queue guard's one real dependency is a permission that cannot be seen
 * until the moment it is needed -- by which point a run has been wedged for
 * half an hour and the answer arrives in a log nobody opens. This answers it
 * on demand instead.
 *
 * The trick is to aim the cancel at a run that is ALREADY COMPLETED. GitHub
 * checks authorization before it checks whether the resource can change, so
 * a token without `actions: write` is refused with 403, and a token that has
 * it gets 409 Conflict -- "that run is finished" -- having changed nothing.
 * Nothing in flight is ever touched: the run picked is, by definition, over.
 *
 * A 409 is therefore strong evidence and not quite proof. If GitHub ever
 * checked run state first, a scopeless token would also see 409 and this
 * would report healthy while the guard stayed broken. That is the one way to
 * be misled here, and it is recorded rather than hidden.
 *
 * NEVER THROWS, for the same reason unwedgeStuckRuns does not.
 */
async function probeCancelScope(env) {
  const out = { canList: false, canCancel: null, status: null, detail: null };
  try {
    const resp = await fetch(`${RUNS_URL}?status=completed&per_page=1`, { headers: ghHeaders(env) });
    if (!resp.ok) {
      out.detail = `list completed runs: HTTP ${resp.status} -- the token cannot even ` +
                   `READ Actions (\`actions: read\`)`;
      return out;
    }
    out.canList = true;
    const body = await resp.json();
    const run = (Array.isArray(body?.workflow_runs) ? body.workflow_runs : [])[0];
    if (!run) {
      out.detail = "no completed run to probe against yet -- scope unverified";
      return out;
    }
    const cancel = await fetch(`${RUNS_URL}/${run.id}/cancel`, {
      method: "POST",
      headers: ghHeaders(env),
    });
    out.status = cancel.status;
    if (cancel.status === 403) {
      out.canCancel = false;
      out.detail = `HTTP 403 on a finished run: the token may read Actions but not ` +
                   `cancel. Add \`actions: write\` to the fine-grained PAT in ` +
                   `GITHUB_TOKEN, or the queue guard cannot clear a wedge.`;
    } else if (cancel.status === 409) {
      out.canCancel = true;
      out.detail = `HTTP 409 on a finished run: refused for being already complete, ` +
                   `not for permission, so \`actions: write\` is present.`;
    } else if (cancel.status === 202) {
      // Should not happen against a completed run, but if GitHub accepted it
      // the permission plainly exists -- and the run was already over.
      out.canCancel = true;
      out.detail = `HTTP 202 on run ${run.id}, which had already completed -- ` +
                   `permission is present.`;
    } else {
      out.detail = `unexpected HTTP ${cancel.status} -- scope not determined`;
    }
  } catch (err) {
    out.detail = String(err?.message ?? err);
  }
  return out;
}

export { unwedgeStuckRuns, probeCancelScope, STUCK_QUEUED_MINUTES };

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
    // BEFORE the dispatch, so a wedged queue is cleared in the same tick
    // rather than one tick later. Its result is logged rather than thrown:
    // see unwedgeStuckRuns -- the tick is the primary job and must survive a
    // broken health check.
    const unwedged = await unwedgeStuckRuns(env);
    if (unwedged.cancelled.length) {
      console.log(
        `[refresh-tick] cancelled ${unwedged.cancelled.length} run(s) wedged in queued: ` +
        unwedged.cancelled.map((r) => `#${r.number} (${r.minutes}m)`).join(", ")
      );
    }
    if (unwedged.error) {
      console.log(`[refresh-tick] queue check did not complete: ${unwedged.error}`);
    }

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

    // ?check=queue verifies the queue guard WITHOUT dispatching anything.
    //
    // The guard's permission could not be confirmed except by waiting for a
    // real wedge, which meant the answer only ever arrived after the outage
    // it was supposed to prevent. This asks the question directly: it reads
    // what is queued, and probes cancel authority against an already
    // finished run so nothing in flight can be touched.
    if (new URL(request.url).searchParams.get("check") === "queue") {
      const scope = await probeCancelScope(env);
      const queued = await unwedgeStuckRuns(env);
      const verdict =
        scope.canCancel === true ? "OK -- the queue guard can cancel a wedged run."
        : scope.canCancel === false ? "BROKEN -- the queue guard cannot cancel anything."
        : "UNKNOWN -- scope could not be determined.";
      return new Response(
        `${verdict}\n\n` +
        `  can read Actions   ${scope.canList}\n` +
        `  can cancel runs    ${scope.canCancel === null ? "unknown" : scope.canCancel}\n` +
        `  probe status       ${scope.status ?? "n/a"}\n` +
        `  detail             ${scope.detail ?? ""}\n\n` +
        `  queued runs now    ${queued.checked}\n` +
        `  cancelled this call ${queued.cancelled.length}` +
        (queued.cancelled.length
          ? ` (${queued.cancelled.map((r) => `#${r.number} ${r.minutes}m`).join(", ")})`
          : ` -- nothing has been queued past ${STUCK_QUEUED_MINUTES}m`) + `\n` +
        (queued.error ? `  queue check error  ${queued.error}\n` : "") +
        `\nNo repository_dispatch was sent by this call.\n`,
        { status: scope.canCancel === false ? 503 : 200 }
      );
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
