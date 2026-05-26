// functions/api/high-score.js
// Cloudflare Pages Function. Single global high score for /stack, stored in
// the SESSION KV namespace under the key "stack:high".
//
// GET  /api/high-score                 → { score }
// POST /api/high-score { score: int }  → { score, updated }

const KEY = 'stack:high';

// 20 million caps the most obvious devtools fakes (typing 9999999999...) while
// leaving plenty of headroom — the real-world NES Tetris record is ~13.6M.
const MAX_SCORE = 20_000_000;

export async function onRequestGet({ env }) {
  const stored = await env.SESSION.get(KEY);
  const score = parseInt(stored || '0', 10);
  return Response.json({ score }, {
    // Always serve the live value; the CDN must not cache this endpoint.
    headers: { 'Cache-Control': 'no-store' },
  });
}

export async function onRequestPost({ env, request }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: 'invalid json' }, { status: 400 });
  }

  const score = Number.isFinite(body?.score) ? Math.floor(body.score) : -1;
  if (score < 0 || score > MAX_SCORE) {
    return Response.json({ error: 'invalid score' }, { status: 400 });
  }

  // Read-then-write. KV is eventually consistent and concurrent writes can
  // race; for a single global high score this is acceptable (worst case: two
  // players tie a new high simultaneously and one of the writes is the kept
  // one). The if-check ensures we never go backwards.
  const current = parseInt((await env.SESSION.get(KEY)) || '0', 10);
  if (score > current) {
    await env.SESSION.put(KEY, String(score));
    return Response.json({ score, updated: true });
  }
  return Response.json({ score: current, updated: false });
}
