/*
 * CACHES EXACTLY ONE FILE: /offline.html. Nothing else, ever.
 *
 * WHAT WENT WRONG BEFORE. An earlier version used stale-while-revalidate:
 * serve the cached page instantly, fetch a fresh copy in the background, and
 * show that fresh copy on the NEXT launch. On most sites that's a good trade.
 * On this one it wasn't, and the symptom was exactly what the pattern
 * guarantees -- open the app, see a stale "Updated" timestamp from hours ago;
 * close and reopen, and only then see current data. Every launch displayed the
 * launch before it. That's a bad trade for a site whose entire value is how
 * current the odds and model output are.
 *
 * The version after that fixed it by registering no fetch handler at all. That
 * was right about the data and wrong about the failure mode: with no network,
 * the reader got the browser's own dinosaur rather than anything of ours.
 *
 * THE RULE THIS VERSION FOLLOWS: never serve stale CONTENT, but do serve a
 * useful FAILURE. So the fetch handler only ever touches navigation requests,
 * and only ever when the network has already failed. A successful response is
 * passed through untouched and is never stored -- there is no code path here
 * that can return yesterday's odds.
 *
 * Everything else (HTML, JSON, images, the movement fragments) goes straight
 * to the network exactly as it would with no service worker installed.
 */

const SHELL = "octane-offline-v1";
const OFFLINE_URL = "/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL)
      .then((c) => c.add(new Request(OFFLINE_URL, { cache: "reload" })))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      // Deletes the caches every previous version left behind, but must NOT
      // delete its own -- the old version cleared everything unconditionally,
      // and reusing that line here would evict the offline page moments after
      // installing it.
      .then((names) => Promise.all(
        names.filter((n) => n !== SHELL).map((n) => caches.delete(n))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // NAVIGATIONS ONLY. Sub-resource requests are left entirely alone: a failed
  // image or fragment should fail as itself, not be replaced by a page of
  // HTML that the caller cannot parse.
  if (req.mode !== "navigate") return;

  event.respondWith(
    // Network first, and network ONLY. The cache is consulted just once, in
    // the catch -- so there is no ordering, staleness or revalidation logic
    // to get wrong.
    fetch(req).catch(() => caches.match(OFFLINE_URL, { cacheName: SHELL }))
  );
});
