/*
 * Exists for exactly one reason: the gray flash between the iOS launch image
 * and the animated splash on a standalone home-screen app.
 *
 * THE ACTUAL PROBLEM, confirmed via screenshots (not the earlier CSS-ordering
 * bug, which was real but different): on a cold standalone launch there is no
 * Safari chrome to hide the wait for the FIRST BYTE of the page over the
 * network. WebKit paints its own neutral placeholder for that gap. No CSS
 * fix is possible -- your stylesheet hasn't arrived yet. The only fix is
 * removing the network wait itself, which means serving the page instantly
 * from local disk on repeat launches.
 *
 * SCOPE IS DELIBERATELY NARROW. This does ONE thing: cache the HTML
 * document (navigation requests only) and serve it instantly next time,
 * refreshing the cache in the background for the launch after that.
 * It does NOT touch:
 *   - the live-odds fetch to clob.polymarket.com (data-live-token spans)
 *   - the results fetch to sports.core.api.espn.com
 *   - the per-fight movement-chart fetch (docs/movements/*.html)
 * None of those are navigation requests, so the fetch handler below never
 * sees them -- they hit the network exactly as they do today, unchanged.
 *
 * STALE-WHILE-REVALIDATE, not cache-first-only. A launch always shows the
 * LAST cached copy instantly (killing the flash), while a fresh copy
 * downloads in the background and replaces the cache entry for the *next*
 * launch. Nothing here can show a copy staler than "as of the last time you
 * opened the app" -- and the model output it bakes in already only refreshes
 * every ~5 minutes server-side, so this adds no meaningfully new staleness.
 * The live-token/results/countdown elements keep updating client-side after
 * paint exactly as they already do, on top of whichever copy is showing.
 */

const CACHE_NAME = 'octane-shell-v1';

self.addEventListener('install', (event) => {
  // Take over immediately on next load rather than waiting for every open
  // tab to close first -- there's only ever one "tab" in standalone mode.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Only ever touch a real page navigation (the HTML document itself).
  // Everything else -- the live-odds JSON, ESPN results, movement-chart
  // fragments, any future asset -- falls straight through untouched.
  if (req.method !== 'GET' || req.mode !== 'navigate') return;

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(req);

      // Always kick off a fresh fetch, cache it for next time -- whether or
      // not we can use it THIS time. Failures (offline, a bad response) are
      // swallowed here rather than left to become an unhandled rejection;
      // the cached-fallback path below already covers that case.
      const network = fetch(req)
        .then((res) => {
          if (res && res.ok) cache.put(req, res.clone());
          return res;
        })
        .catch(() => null);

      // Cached copy exists -> serve it INSTANTLY, this is the fix. The
      // network fetch above still runs in the background and updates the
      // cache for the launch after this one.
      if (cached) return cached;

      // No cache yet (first-ever visit) -> behave exactly as today: wait on
      // the real network response. Fall back to any cached copy only if the
      // network genuinely fails.
      const res = await network;
      return res || cached || Response.error();
    })
  );
});
