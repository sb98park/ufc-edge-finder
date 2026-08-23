import { mintSession, readSession, clearSession }
  from "../lib/session.js";

const SECRET = "test-secret-not-the-real-one";
const req = (cookie) => new Request("https://x/", { headers: cookie ? { Cookie: cookie } : {} });
const cookieValue = (setCookie) => setCookie.split(";")[0];

let pass = 0, fail = 0;
const check = (name, cond) => { (cond ? pass++ : fail++); console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`); };

// 1. round trip
const asMember = await mintSession({ userId: "user-1", member: true }, SECRET);
const back = await readSession(req(cookieValue(asMember)), SECRET);
check("member cookie round-trips", back && back.userId === "user-1" && back.member === true);

const asFree = await mintSession({ userId: "user-2", member: false }, SECRET);
const back2 = await readSession(req(cookieValue(asFree)), SECRET);
check("free cookie round-trips as non-member", back2 && back2.member === false);

// 2. THE ATTACK: edit "free" to "member" in your own cookie
const forged = cookieValue(asFree).replace(".free.", ".member.");
check("tier tampering is rejected", (await readSession(req(forged), SECRET)) === null);

// 3. swap in someone else's user id, keep the signature
const swapped = cookieValue(asMember).replace("user-1", "user-9");
check("user id tampering is rejected", (await readSession(req(swapped), SECRET)) === null);

// 4. signature from a different secret
const otherSecret = await mintSession({ userId: "user-1", member: true }, "different-secret");
check("cookie signed with another secret is rejected",
      (await readSession(req(cookieValue(otherSecret)), SECRET)) === null);

// 5. expiry is enforced
const raw = decodeURIComponent(cookieValue(asMember).split("=")[1]).split(".");
const expired = `octane_session=${encodeURIComponent(["2", raw[1], raw[2], String(Date.now() - 1000), raw[4]].join("."))}`;
check("expired cookie is rejected", (await readSession(req(expired), SECRET)) === null);

// 6. the OLD password-gate cookie shape must not be misread
check("legacy 3-part cookie is rejected",
      (await readSession(req("octane_session=admin.9999999999999.abc"), SECRET)) === null);

// 7. no cookie
check("absent cookie yields null", (await readSession(req(null), SECRET)) === null);

// 8. clearing
check("clearSession expires the cookie", clearSession().includes("Max-Age=0"));

// 9. cookie flags
check("cookie is HttpOnly, Secure, SameSite=Lax",
      asMember.includes("HttpOnly") && asMember.includes("Secure") && asMember.includes("SameSite=Lax"));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
