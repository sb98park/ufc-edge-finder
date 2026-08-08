/*
 * DELIBERATELY DOES NO CACHING. This file exists only to neutralise the
 * caching service worker that shipped before it, and it must keep existing
 * (rather than simply being deleted) because a browser that already
 * installed the old one keeps running it until it fetches a CHANGED sw.js
 * at this exact path. Deleting the file would strand those installs on the
 * caching version indefinitely.
 *
 * WHAT WENT WRONG. The previous version used stale-while-revalidate: serve
 * the cached page instantly, fetch a fresh copy in the background, and show
 * that fresh copy on the NEXT launch. On most sites that's a good trade. On
 * this one it wasn't, and the symptom was exactly what the pattern
 * guarantees -- open the app, see a stale "Updated" timestamp from hours
 * ago; close and reopen, and only then see current data. Every launch
 * displayed the launch before it.
 *
 * That's a bad trade for a site whose entire value is how current the odds
 * and model output are. It was introduced to fix the gray flash between the
 * iOS launch image and the animated splash, and it did not fix that either,
 * so it was costing freshness and buying nothing.
 *
 * With no fetch handler registered, every request goes straight to the
 * network exactly as it would with no service worker at all. The activate
 * handler clears the caches the old version left behind, so no stale
 * response can survive this even once.
 */

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

// No 'fetch' listener, on purpose. Adding one back would reintroduce the bug.
