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

import { unwedgeStuckRuns, STUCK_QUEUED_MINUTES } from "../refresh-tick.js";

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

console.log(`${fail ? "FAIL" : "PASS"}: refresh-tick -- ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
