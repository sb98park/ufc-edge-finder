/**
 * Combat Edge fetch relay.
 *
 * WHY THIS EXISTS: combat-edge.com has real, useful KO/Sub/Dec method-of-
 * victory data and (unlike Sherdog or Tapology, both directly tested and
 * ruled out) a plain A-Z browsable directory that solves the name->URL
 * lookup problem without needing a JS-driven search. The ONLY issue is
 * that it blocks requests from GitHub Actions' shared IP range
 * specifically (confirmed: works fine from other network infrastructure,
 * 429s consistently and only from GH Actions runners -- an IP-reputation
 * block, not a per-request rate-limit, since realistic User-Agent headers
 * and delays don't help it).
 *
 * This Worker runs on CLOUDFLARE's own IP range, not GitHub's. It fetches
 * the target Combat Edge page server-side and hands the raw HTML back to
 * the GitHub Actions script. If Combat Edge isn't specifically blocking
 * Cloudflare's IPs too (no evidence they are -- that range is far too
 * broad and common to block without breaking huge amounts of legitimate
 * traffic, unlike a narrow, well-known "CI/CD cloud" range like GitHub
 * Actions'), this should route around the block entirely, for free, using
 * infrastructure you already have some familiarity with from the access
 * gate Worker.
 *
 * HONESTLY UNVERIFIED until you deploy and test it -- I can't confirm
 * Combat Edge's blocking behavior from this sandbox. Deploy it, run the
 * pipeline once, and check for the "[fighter_backfill] combat-edge: ...
 * found but no win/loss-by-method fields matched" vs the old 429 pattern
 * in the log to see whether this actually changes anything.
 *
 * SECURITY: this only proxies combat-edge.com URLs (checked below), and
 * requires a shared-secret token so it can't be discovered and abused as
 * a general-purpose open proxy by anyone who finds the Worker's URL.
 * Set your own token as a Cloudflare Worker environment variable/secret
 * named RELAY_TOKEN (Workers dashboard -> your worker -> Settings ->
 * Variables -> add a Secret, NOT a plain-text variable) before deploying
 * -- don't hardcode it here.
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const token = url.searchParams.get("token");
    const target = url.searchParams.get("url");

    if (!env.RELAY_TOKEN || token.trim() !== env.RELAY_TOKEN.trim()) {
      return new Response("Forbidden", { status: 403 });
    }
    if (!target) {
      return new Response("Missing 'url' query param", { status: 400 });
    }

    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch (e) {
      return new Response("Invalid target URL", { status: 400 });
    }
    // Only ever relay to combat-edge.com -- never becomes a general proxy
    // even if the token leaks, and rules out this being pointed at
    // anything else by mistake or malice.
    if (!targetUrl.hostname.endsWith("combat-edge.com")) {
      return new Response("Only combat-edge.com URLs are relayed", { status: 403 });
    }

    const upstream = await fetch(targetUrl.toString(), {
      headers: {
        // A realistic browser User-Agent -- matches what the Python side
        // already sends for Combat Edge specifically (see BASE_HEADERS
        // override in src/fighter_backfill.py), kept consistent here too.
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      },
    });

    // Pass the upstream response straight through -- status code included,
    // so the Python side's existing error handling (checking for a 429,
    // etc.) keeps working unchanged whether it's talking to Combat Edge
    // directly or through this relay.
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  },
};
