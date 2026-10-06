// The one CI failure mode nothing inside CI can see.
//
// On 2026-09-13 a run sat in `queued` for 8.7 hours holding refresh.yml's
// `refresh` concurrency group. That group is cancel-in-progress: false, so
// GitHub keeps only the newest pending run and cancels the rest: 99
// consecutive cancelled runs, zero successes, the site stale for nine hours.
//
// timeout-minutes: 18 could not bound it -- that clock starts when a job
// RUNS, and this run never got one. The test gate, step health and the lint
// summary are all inside the workflow too, so a run that never starts is
// invisible to every guard we have. The Worker is the only piece still
// executing when Actions is wedged.
//
// These tests are mostly about what it must NOT cancel.

import { unwedgeStuckRuns, probeCancelScope, STUCK_QUEUED_MINUTES } from "../refresh-tick.js";

let pass = 0, fail = 0;
const check = (n, c) => { (c ? pass++ : fail++); console.log(`  ${c ? "PASS" : "FAIL"}  ${n}`); };

const NOW = Date.parse("2026-09-13T12:00:00Z");
const ago = (min) => new Date(NOW - min * 60000).toISOString();
const env = { GITHUB_TOKEN: "t" };

// Stub GitHub: a list endpoint returning `runs`, and a cancel endpoint that
// records what was asked for.
function stub(runs, { listStatus = 200, cancelStatus = 202 } = {}) {
  const cancelled = [];
  globalThis.fetch = async (url, opts = {}) => {
    if (String(url).includes("/cancel")) {
      cancelled.push(Number(String(url).match(/runs\/(\d+)\/cancel/)[1]));
      return new Response("", { status: cancelStatus });
    }
    if (listStatus !== 200) return new Response("nope", { status: listStatus });
    // The 409 re-check asks for ONE run by id. It is not the list call, so it
    // must not be measured by the list call's rule -- answering "completed"
    // keeps this helper's 409 meaning the race it was written for.
    if (/\/runs\/\d+$/.test(String(url))) {
      return new Response(JSON.stringify({ status: "completed" }), { status: 200 });
    }
    // The guard must ask only for queued runs -- never in_progress.
    check("only queued runs are listed", String(url).includes("status=queued"));
    return new Response(JSON.stringify({ workflow_runs: runs }), { status: 200 });
  };
  return cancelled;
}

// --- the regression -------------------------------------------------------
let cancelled = stub([{ id: 111, run_number: 10914, created_at: ago(519) }]);
let r = await unwedgeStuckRuns(env, NOW);
check("a run queued 8.7h is cancelled", cancelled.length === 1 && cancelled[0] === 111);
check("...and reported with its age", r.cancelled[0].minutes === 519);
check("...and its run number", r.cancelled[0].number === 10914);

// The 25-day zombie found alongside it.
cancelled = stub([{ id: 222, run_number: 3097, created_at: ago(36579) }]);
await unwedgeStuckRuns(env, NOW);
check("a 25-day-old queued run is cancelled", cancelled.length === 1);

// --- WHAT MUST NOT BE CANCELLED ------------------------------------------
// Brief queueing while the previous build finishes is normal.
cancelled = stub([{ id: 333, run_number: 1, created_at: ago(2) }]);
r = await unwedgeStuckRuns(env, NOW);
check("a run queued 2 minutes is left alone", cancelled.length === 0 && r.cancelled.length === 0);

cancelled = stub([{ id: 334, run_number: 2, created_at: ago(STUCK_QUEUED_MINUTES - 1) }]);
await unwedgeStuckRuns(env, NOW);
check("just under the threshold is left alone", cancelled.length === 0);

cancelled = stub([{ id: 335, run_number: 3, created_at: ago(STUCK_QUEUED_MINUTES + 1) }]);
await unwedgeStuckRuns(env, NOW);
check("just over the threshold is cancelled", cancelled.length === 1);

// The threshold must stay clear of the workflow's own 18-minute job cap, or
// this would start killing builds that are merely slow.
check("threshold clears refresh.yml's 18-minute job cap", STUCK_QUEUED_MINUTES > 18);

// Mixed: only the stuck one goes.
cancelled = stub([
  { id: 401, run_number: 10, created_at: ago(1) },
  { id: 402, run_number: 11, created_at: ago(600) },
  { id: 403, run_number: 12, created_at: ago(3) },
]);
r = await unwedgeStuckRuns(env, NOW);
check("only the wedged run is cancelled", cancelled.length === 1 && cancelled[0] === 402);
check("the others are counted but untouched", r.checked === 3);

// --- it must never throw, or the tick dies with it -----------------------
stub([], { listStatus: 500 });
r = await unwedgeStuckRuns(env, NOW);
check("a failed list is reported, not raised", r.error?.includes("HTTP 500") && r.cancelled.length === 0);

cancelled = stub([{ id: 501, run_number: 4, created_at: ago(600) }], { cancelStatus: 403 });
r = await unwedgeStuckRuns(env, NOW);
check("a refused cancel is reported, not raised", r.error?.includes("HTTP 403"));
check("...and nothing is claimed as cancelled", r.cancelled.length === 0);

// 409 means it moved on between list and cancel -- the wedge is gone either
// way, so that counts as handled.
cancelled = stub([{ id: 502, run_number: 5, created_at: ago(600) }], { cancelStatus: 409 });
r = await unwedgeStuckRuns(env, NOW);
check("a 409 race counts as cleared", r.cancelled.length === 1);

globalThis.fetch = async () => { throw new Error("network down"); };
r = await unwedgeStuckRuns(env, NOW);
check("a thrown fetch is caught", r.error === "network down" && r.cancelled.length === 0);

// Malformed payloads must not crash the tick either.
globalThis.fetch = async () => new Response(JSON.stringify({}), { status: 200 });
r = await unwedgeStuckRuns(env, NOW);
check("a payload with no workflow_runs is survivable", r.checked === 0 && !r.error);

globalThis.fetch = async () => new Response(JSON.stringify({ workflow_runs: [{ id: 9 }] }), { status: 200 });
r = await unwedgeStuckRuns(env, NOW);
check("a run with no created_at is skipped", r.cancelled.length === 0 && !r.error);


// --- the permission that looks like no failure ----------------------------
// Listing runs needs `actions: read`; cancelling needs `actions: write`. A
// read-only PAT lists the wedged run and is refused at the cancel, so the
// guard looks installed and clears nothing. It must say which permission.
cancelled = stub([{ id: 222, run_number: 10915, created_at: ago(600) }], { cancelStatus: 403 });
r = await unwedgeStuckRuns(env, NOW);
check("a 403 cancel is not counted as cancelled", r.cancelled.length === 0);
check("  ...and names the actual permission", /actions: write/.test(r.error ?? ""));
check("  ...and is flagged as a scope problem", r.scopeDenied === true);
check("  ...and points at the check that confirms it", /check=queue/.test(r.error ?? ""));

// A different failure must NOT be blamed on scope.
stub([{ id: 333, run_number: 10916, created_at: ago(600) }], { cancelStatus: 500 });
r = await unwedgeStuckRuns(env, NOW);
check("a 500 is reported without claiming a scope problem",
      /HTTP 500/.test(r.error ?? "") && !r.scopeDenied);

// --- the probe cancels nothing --------------------------------------------
// It aims at an already-completed run, so a 409 means "finished", not
// "forbidden" -- authority confirmed with nothing touched.
function stubProbe(cancelStatus, { listStatus = 200, runs = [{ id: 777 }] } = {}) {
  const attempted = [];
  globalThis.fetch = async (url) => {
    if (String(url).includes("/cancel")) {
      attempted.push(Number(String(url).match(/runs\/(\d+)\/cancel/)[1]));
      return new Response("", { status: cancelStatus });
    }
    if (listStatus !== 200) return new Response("nope", { status: listStatus });
    check("the probe asks for COMPLETED runs, never queued ones",
          String(url).includes("status=completed"));
    return new Response(JSON.stringify({ workflow_runs: runs }), { status: 200 });
  };
  return attempted;
}

let attempted = stubProbe(409);
let p409 = await probeCancelScope(env);
check("409 on a finished run means the scope is present", p409.canCancel === true);
check("  ...and the probe only ever touched a completed run",
      attempted.length === 1 && attempted[0] === 777);

stubProbe(403);
let p403 = await probeCancelScope(env);
check("403 means the scope is missing", p403.canCancel === false);
check("  ...and says which permission to add", /actions: write/.test(p403.detail ?? ""));

stubProbe(202, { listStatus: 401 });
let pNoRead = await probeCancelScope(env);
check("a token that cannot read Actions is reported as such",
      pNoRead.canList === false && /actions: read/.test(pNoRead.detail ?? ""));

stubProbe(202, { runs: [] });
let pEmpty = await probeCancelScope(env);
check("no completed run to probe -> unknown, not a false OK", pEmpty.canCancel === null);

stubProbe(418);
let pOdd = await probeCancelScope(env);
check("an unexpected status is unknown, not a false OK", pOdd.canCancel === null);

globalThis.fetch = async () => { throw new Error("network down"); };
let pDead = await probeCancelScope(env);
check("the probe never throws", pDead.canCancel === null && /network down/.test(pDead.detail));


// --- 409 is two different things -------------------------------------------
// It was read as "the run moved on between the list and the cancel", which is
// one of them. The other is a run GitHub will no longer let anyone cancel:
// #3097 sat `queued` from 2026-08-19, answered 409 every tick, and was
// reported as freshly cancelled for forty-eight days. On 2026-10-05 a REAL
// wedge then sat for 25.5 hours while the guard logged success.
function stub409(stillQueued, { createdMinAgo = 600 } = {}) {
  const runs = [{ id: 999, run_number: 3097, created_at: ago(createdMinAgo) }];
  globalThis.fetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("/cancel")) return new Response("", { status: 409 });
    // the re-check: same run id, no /cancel suffix
    if (/runs\/999$/.test(u)) {
      return new Response(JSON.stringify({ status: stillQueued ? "queued" : "completed" }),
                          { status: 200 });
    }
    return new Response(JSON.stringify({ workflow_runs: runs }), { status: 200 });
  };
}

stub409(true);
r = await unwedgeStuckRuns(env, NOW);
check("a 409 that is STILL queued is not counted as cancelled", r.cancelled.length === 0);
check("  ...it is reported as unkillable instead", r.unkillable.length === 1);
check("  ...naming the run", r.unkillable[0] && r.unkillable[0].number === 3097);
check("  ...and is not an error either", r.error === null);

stub409(false);
r = await unwedgeStuckRuns(env, NOW);
check("a 409 that genuinely moved on still counts as cancelled", r.cancelled.length === 1);
check("  ...and is not called unkillable", r.unkillable.length === 0);

// If the re-check itself fails we must not invent a zombie: unknowable reads
// as the race, which is the interpretation that does not raise a false alarm.
globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.includes("/cancel")) return new Response("", { status: 409 });
  if (/runs\/999$/.test(u)) throw new Error("network down");
  return new Response(JSON.stringify({
    workflow_runs: [{ id: 999, run_number: 3097, created_at: ago(600) }] }), { status: 200 });
};
r = await unwedgeStuckRuns(env, NOW);
check("an unreadable re-check falls back to the race reading",
      r.cancelled.length === 1 && r.unkillable.length === 0);

// A 202 must not pay for the extra request.
let probes = 0;
globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.includes("/cancel")) return new Response("", { status: 202 });
  if (/runs\/999$/.test(u)) { probes++; return new Response("{}", { status: 200 }); }
  return new Response(JSON.stringify({
    workflow_runs: [{ id: 999, run_number: 3097, created_at: ago(600) }] }), { status: 200 });
};
r = await unwedgeStuckRuns(env, NOW);
check("a clean 202 makes no re-check request", r.cancelled.length === 1 && probes === 0);


// --- "cancelled 0" has two very different meanings --------------------------
// The first real run of ?check=queue printed "cancelled this call 0 -- nothing
// has been queued past 30m" while reporting a 403 on a run queued since
// 2026-08-19, two lines below. Nothing stuck and stuck-but-unkillable are
// opposite findings and must not share a sentence.
stub([{ id: 7, run_number: 42, created_at: ago(600) }], { cancelStatus: 403 });
r = await unwedgeStuckRuns(env, NOW);
check("a run past the threshold is counted even when the cancel fails", r.qualified === 1);
check("  ...and nothing is reported as cancelled", r.cancelled.length === 0);

stub([{ id: 8, run_number: 43, created_at: ago(5) }]);
r = await unwedgeStuckRuns(env, NOW);
check("a fresh queued run does not count as qualified", r.qualified === 0);

console.log(`${fail ? "FAIL" : "PASS"}: refresh-tick -- ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
