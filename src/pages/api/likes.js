// src/pages/api/likes.js
// Per-post thumbs-up counter in the LIKES KV namespace, one key per slug.
// Same shape as high-score.js: a server endpoint inside the Worker bundle,
// reaching its binding through the cloudflare:workers virtual module.
//
// GET  /api/likes?slug=<post-id>  → { likes }
// POST /api/likes?slug=<post-id>  → { likes } after one increment

import { env } from 'cloudflare:workers';

export const prerender = false;

// Post ids are kebab-case file paths; anything else never touches KV.
const SLUG = /^[a-z0-9][a-z0-9/-]{0,63}$/;

function json(data, init = {}) {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json');
  return new Response(JSON.stringify(data), { ...init, headers });
}

function slugOf(url) {
  const slug = new URL(url).searchParams.get('slug') || '';
  return SLUG.test(slug) ? slug : null;
}

export async function GET({ request }) {
  const slug = slugOf(request.url);
  if (!slug) return json({ error: 'invalid slug' }, { status: 400 });
  const likes = parseInt((await env.LIKES.get(slug)) || '0', 10);
  return json({ likes }, {
    headers: { 'Cache-Control': 'no-store' },
  });
}

export async function POST({ request }) {
  const slug = slugOf(request.url);
  if (!slug) return json({ error: 'invalid slug' }, { status: 400 });

  // One counted like per IP per post per day. The guard key stores a hash,
  // never the address itself, and expires on its own. This is friction, not
  // security: a like counter has nothing worth defending beyond making a
  // console loop pointless. Absent header (local dev) skips the guard.
  const ip = request.headers.get('cf-connecting-ip');
  if (ip) {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(ip));
    const hash = [...new Uint8Array(digest.slice(0, 12))].map((b) => b.toString(16).padStart(2, '0')).join('');
    const guard = `ip:${slug}:${hash}`;
    if (await env.LIKES.get(guard)) {
      const likes = parseInt((await env.LIKES.get(slug)) || '0', 10);
      return json({ likes });
    }
    await env.LIKES.put(guard, '1', { expirationTtl: 86400 });
  }

  // Read-then-write. KV can drop one of two simultaneous likes; for a blog
  // counter that is an acceptable trade for zero extra infrastructure.
  const likes = parseInt((await env.LIKES.get(slug)) || '0', 10) + 1;
  await env.LIKES.put(slug, String(likes));
  return json({ likes });
}
