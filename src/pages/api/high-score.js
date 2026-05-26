// src/pages/api/high-score.js
// Astro server endpoint for the global high score, deployed inside the
// Cloudflare Worker bundle by @astrojs/cloudflare. The SESSION KV binding
// (declared in wrangler.jsonc) is reached through the cloudflare:workers
// virtual module — Astro 6's replacement for the removed locals.runtime.env.
//
// GET  /api/high-score                 → { score }
// POST /api/high-score { score: int }  → { score, updated }

import { env } from 'cloudflare:workers';

// Opt out of static prerendering so this file runs as a server endpoint
// inside the Worker, not as a baked HTML response.
export const prerender = false;

const KEY = 'stack:high';

// 20 million caps the most obvious devtools fakes while leaving plenty of
// headroom — the real-world NES Tetris record is ~13.6M.
const MAX_SCORE = 20_000_000;

function json(data, init = {}) {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json');
  return new Response(JSON.stringify(data), { ...init, headers });
}

export async function GET() {
  const stored = await env.SESSION.get(KEY);
  const score = parseInt(stored || '0', 10);
  return json({ score }, {
    // Always serve the live value; the CDN must not cache this endpoint.
    headers: { 'Cache-Control': 'no-store' },
  });
}

export async function POST({ request }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: 'invalid json' }, { status: 400 });
  }

  const score = Number.isFinite(body?.score) ? Math.floor(body.score) : -1;
  if (score < 0 || score > MAX_SCORE) {
    return json({ error: 'invalid score' }, { status: 400 });
  }

  // Read-then-write. KV is eventually consistent and concurrent writes can
  // race; for a single global high score the if-check ensures we never go
  // backwards, which is the only invariant that matters here.
  const current = parseInt((await env.SESSION.get(KEY)) || '0', 10);
  if (score > current) {
    await env.SESSION.put(KEY, String(score));
    return json({ score, updated: true });
  }
  return json({ score: current, updated: false });
}
