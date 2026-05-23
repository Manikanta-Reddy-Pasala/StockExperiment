// Service worker — minimal cache for shell + offline fallback.
// Bump CACHE_VERSION on any UI change so old clients refetch.
const CACHE_VERSION = 'v8-2026-05-23-lucide-gunicorn';
const CACHE_NAME = 'trading-pwa-' + CACHE_VERSION;
const PRECACHE_URLS = [
  '/static/logo.png',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/apple-touch-icon.png',
  '/static/css/custom.css',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n.startsWith('trading-pwa-') && n !== CACHE_NAME)
                       .map((n) => caches.delete(n)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  // Only handle GET; never cache POST / API calls / signals data
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // Network-first for HTML + APIs (always live data); cache-first for static
  if (url.pathname.startsWith('/static/') || url.pathname === '/favicon.ico') {
    event.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        const copy = res.clone();
        if (res.ok) caches.open(CACHE_NAME).then((c) => c.put(req, copy));
        return res;
      }))
    );
  } else {
    // network-first; on failure, return cached shell if any
    event.respondWith(
      fetch(req).catch(() => caches.match(req).then((hit) => hit ||
        new Response('<h1>Offline</h1><p>No cached copy. Reconnect and retry.</p>',
                     {headers: {'Content-Type': 'text/html'}})))
    );
  }
});
