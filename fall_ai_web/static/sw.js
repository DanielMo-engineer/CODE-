// ══════════════════════════════════════════════════════
// RDK X5 智能平台 — Service Worker
// 功能: 离线缓存应用外壳，设备关机时显示离线页面
// ══════════════════════════════════════════════════════

const CACHE_NAME = 'rdk-x5-v1';
const OFFLINE_URL = '/offline';

// 需要缓存的资源（应用外壳）
const APP_SHELL = [
  '/',
  '/camera',
  '/chat',
  '/dashboard',
  '/alerts',
  '/mood',
  '/about',
  '/offline',
  '/static/manifest.json',
];

// 安装：缓存应用外壳
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(APP_SHELL);
    })
  );
  self.skipWaiting();
});

// 激活：清理旧缓存
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      );
    })
  );
  self.clients.claim();
});

// 请求拦截：优先网络，失败则用缓存
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API 请求不缓存，但失败时返回离线提示
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        if (url.pathname === '/api/status') {
          return new Response(
            JSON.stringify({status: 'OFFLINE', alarm: false, people: 0, fps: 0}),
            {headers: {'Content-Type': 'application/json'}}
          );
        }
        return new Response(
          JSON.stringify({error: 'device_offline', message: '设备已离线'}),
          {status: 503, headers: {'Content-Type': 'application/json'}}
        );
      })
    );
    return;
  }

  // 导航请求：先尝试网络，失败后返回缓存的离线页面
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match(OFFLINE_URL).then((cached) => {
          return cached || caches.match('/');
        });
      })
    );
    return;
  }

  // 静态资源：缓存优先
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request).then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      });
    })
  );
});
