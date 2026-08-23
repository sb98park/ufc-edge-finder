import { trialEndTimestamp, verifyWebhook } from "../lib/stripe.js";

let pass = 0, fail = 0;
const check = (n, c) => { (c ? pass++ : fail++); console.log(`  ${c ? "PASS" : "FAIL"}  ${n}`); };
const DAY = 86400, WEEK = 7 * DAY;
const now = () => Math.floor(Date.now() / 1000);
const days = (ts) => Math.round((ts - now()) / DAY);

// --- trial timing: must always contain a fight card -------------------
const soon = new Date(Date.now() + 2 * DAY * 1000).toISOString().slice(0, 10);
check("event in 2 days -> floor of 7 days still applies", days(trialEndTimestamp(soon)) === 7);

const gap = new Date(Date.now() + 9 * DAY * 1000).toISOString().slice(0, 10);
check("event in 9 days (gap week) -> trial stretches past it", days(trialEndTimestamp(gap)) === 10);

check("no event date -> falls back to 7 days", days(trialEndTimestamp(null)) === 7);
check("garbage date -> falls back to 7 days", days(trialEndTimestamp("not-a-date")) === 7);
check("trial always covers at least a week",
      [soon, gap, null, "x"].every((d) => trialEndTimestamp(d) >= now() + WEEK - 60));

// --- webhook signatures: the only thing guarding a public write endpoint
const SECRET = "whsec_test";
const body = JSON.stringify({ type: "customer.subscription.created", data: { object: { id: "sub_1" } } });
async function sign(payload, ts, secret = SECRET) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`${ts}.${payload}`));
  return Array.from(new Uint8Array(mac)).map(b => b.toString(16).padStart(2, "0")).join("");
}

const ts = now();
check("correctly signed webhook is accepted",
      (await verifyWebhook(body, `t=${ts},v1=${await sign(body, ts)}`, SECRET))?.type
        === "customer.subscription.created");

check("forged signature is rejected",
      (await verifyWebhook(body, `t=${ts},v1=${"0".repeat(64)}`, SECRET)) === null);

check("signature from a different secret is rejected",
      (await verifyWebhook(body, `t=${ts},v1=${await sign(body, ts, "whsec_other")}`, SECRET)) === null);

const tampered = JSON.stringify({ type: "customer.subscription.created", data: { object: { id: "sub_EVIL" } } });
check("body tampering invalidates a valid signature",
      (await verifyWebhook(tampered, `t=${ts},v1=${await sign(body, ts)}`, SECRET)) === null);

const old = now() - 600;
check("replayed old event is rejected (5 min window)",
      (await verifyWebhook(body, `t=${old},v1=${await sign(body, old)}`, SECRET)) === null);

check("missing signature header is rejected", (await verifyWebhook(body, null, SECRET)) === null);
check("malformed header is rejected", (await verifyWebhook(body, "garbage", SECRET)) === null);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
