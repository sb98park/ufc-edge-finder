import { verifySupabaseToken }
  from "../lib/jwt.js";

const URL_BASE = "https://proj.supabase.co";
const b64url = (buf) => Buffer.from(buf).toString("base64url");
const enc = new TextEncoder();

// A real P-256 keypair, and a JWKS served from a stubbed fetch.
const kp = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign","verify"]);
const jwk = await crypto.subtle.exportKey("jwk", kp.publicKey);
jwk.kid = "test-kid"; jwk.alg = "ES256"; jwk.use = "sig";
globalThis.fetch = async () => new Response(JSON.stringify({ keys: [jwk] }), { status: 200 });

const now = () => Math.floor(Date.now() / 1000);
async function makeToken(claims = {}, header = {}) {
  const h = b64url(JSON.stringify({ alg: "ES256", typ: "JWT", kid: "test-kid", ...header }));
  const p = b64url(JSON.stringify({ sub: "user-1", aud: "authenticated", iss: `${URL_BASE}/auth/v1`,
                                    iat: now(), exp: now() + 3600, ...claims }));
  const sig = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, kp.privateKey, enc.encode(`${h}.${p}`));
  return `${h}.${p}.${b64url(sig)}`;
}

let pass = 0, fail = 0;
const check = (n, c) => { (c ? pass++ : fail++); console.log(`  ${c ? "PASS" : "FAIL"}  ${n}`); };

check("valid token verifies", (await verifySupabaseToken(await makeToken(), URL_BASE))?.sub === "user-1");

// THE CLASSIC BREAKS
const t = await makeToken();
const [h, p, s] = t.split(".");
const noneHeader = b64url(JSON.stringify({ alg: "none", typ: "JWT", kid: "test-kid" }));
check("alg:none is rejected", (await verifySupabaseToken(`${noneHeader}.${p}.`, URL_BASE)) === null);

const hsHeader = b64url(JSON.stringify({ alg: "HS256", typ: "JWT", kid: "test-kid" }));
check("algorithm confusion (HS256) is rejected",
      (await verifySupabaseToken(`${hsHeader}.${p}.${s}`, URL_BASE)) === null);

// payload edited, original signature kept
const evil = b64url(JSON.stringify({ sub: "somebody-else", aud: "authenticated",
                                     iss: `${URL_BASE}/auth/v1`, iat: now(), exp: now() + 3600 }));
check("edited payload is rejected", (await verifySupabaseToken(`${h}.${evil}.${s}`, URL_BASE)) === null);

check("expired token is rejected",
      (await verifySupabaseToken(await makeToken({ exp: now() - 10 }), URL_BASE)) === null);
check("wrong audience is rejected",
      (await verifySupabaseToken(await makeToken({ aud: "anon" }), URL_BASE)) === null);
check("wrong issuer is rejected",
      (await verifySupabaseToken(await makeToken({ iss: "https://evil.example" }), URL_BASE)) === null);
check("token with no subject is rejected",
      (await verifySupabaseToken(await makeToken({ sub: undefined }), URL_BASE)) === null);
check("garbage is rejected", (await verifySupabaseToken("not.a.token", URL_BASE)) === null);
check("non-string is rejected", (await verifySupabaseToken(null, URL_BASE)) === null);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
