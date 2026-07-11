// src/scripts/play.js
// Multiplayer canvas client for /play — rendering, input, and the room socket.
// The server (game/src/room.ts) owns the truth; this draws it and asks to move.

const WORLD = { w: 640, h: 360 }; // mirrored in game/src/room.ts
const SIZE = 16; // sprite square, px
const SPEED = 160; // px per second
const SEND_HZ = 10; // position updates to the server, per second

// Side-view physics. Simulated only for your own sprite, in your own browser —
// the server just relays positions. Remote jumps arrive as positions and glide
// through the same interpolation as walking.
const GROUND_Y = 320; // top of the floor
const GRAVITY = 1400; // px/s^2
const JUMP_V = 520; // initial upward speed; ~96px apex, ~0.74s of air

const SOCKET_URL = import.meta.env.DEV
  ? 'ws://localhost:8787'
  : 'wss://peteramassih-play.peteramassih.workers.dev';

const KEYS = {
  ArrowLeft: 'left', a: 'left',
  ArrowRight: 'right', d: 'right',
};
const JUMP_KEYS = ['ArrowUp', 'w', ' '];

const BUBBLE_MS = 4000; // how long a chat bubble hangs over a sprite
const EMOTE_MS = 1200; // wave wiggle / heart float duration
const SMOOTH_RATE = 12; // per second: how fast remote sprites chase their target
const SNAP_DIST = 80; // farther than this is a spawn or teleport, not a walk: snap

const canvas = document.getElementById('play-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('play-status');
const countEl = document.getElementById('play-count');
const chatForm = document.getElementById('play-chat-form');
const chatInput = document.getElementById('play-chat');
const nameInput = document.getElementById('play-name');

let me = null; // my id, assigned by the server in init
// x/y are the network truth; rx/ry are where the sprite is drawn. Remote
// sprites ease rx/ry toward x/y so 10 Hz updates read as walking, not teleports.
const players = new Map(); // id -> { x, y, rx, ry, color, name }
const bubbles = new Map(); // id -> { text, until }
const effects = []; // running emotes: { id, kind, start }
const held = new Set(); // movement keys currently down
let vy = 0; // my vertical speed
let jumpQueued = false; // jump pressed, consumed by the next frame
let dirty = false; // my position changed since the last network send
let last = 0; // previous frame timestamp

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

// --- sprites ---
// Hand-authored pixel frames, one character per pixel: h hair, s skin, e eyes,
// t shirt (takes the player color), p pants, b boots, . transparent. Each
// (color, frame) pair is rendered once to an offscreen canvas and stamped
// with drawImage, so per-frame cost is one blit per player.
const SPRITE_PX = 2; // screen pixels per sprite pixel: 12x16 art drawn 24x32
const SPRITE_INK = { h: '#4a3320', s: '#f2c48a', e: '#1d1f24', p: '#2e3b57', b: '#3a2f28' };
const FRAMES = {
  idle: [
    '....hhhh....',
    '...hhhhhh...',
    '...hssssh...',
    '...sesses...',
    '...ssssss...',
    '....ssss....',
    '...tttttt...',
    '..tttttttt..',
    '..stttttts..',
    '..tttttttt..',
    '...tttttt...',
    '...pppppp...',
    '...pp..pp...',
    '...pp..pp...',
    '...bb..bb...',
    '...bb..bb...',
  ],
  walk0: [
    '....hhhh....',
    '...hhhhhh...',
    '...hssssh...',
    '...sesses...',
    '...ssssss...',
    '....ssss....',
    '...tttttt...',
    '..tttttttt..',
    '..stttttts..',
    '..tttttttt..',
    '...tttttt...',
    '...pppppp...',
    '....pppp....',
    '....pppp....',
    '....bbbb....',
    '...bb..bb...',
  ],
  walk1: [
    '....hhhh....',
    '...hhhhhh...',
    '...hssssh...',
    '...sesses...',
    '...ssssss...',
    '....ssss....',
    '...tttttt...',
    '..tttttttt..',
    '..stttttts..',
    '..tttttttt..',
    '...tttttt...',
    '...pppppp...',
    '..pp....pp..',
    '..pp....pp..',
    '.bb......bb.',
    '............',
  ],
  jump: [
    '....hhhh....',
    '...hhhhhh...',
    '...hssssh...',
    '...sesses...',
    '...ssssss...',
    '....ssss....',
    '...tttttt...',
    '.stttttttts.',
    '..tttttttt..',
    '..tttttttt..',
    '...tttttt...',
    '...pppppp...',
    '...pp..pp...',
    '..bb....bb..',
    '............',
    '............',
  ],
};
const SPR_W = 12 * SPRITE_PX;
const SPR_H = 16 * SPRITE_PX;

const spriteCache = new Map(); // "color|frame" -> offscreen canvas
function sprite(color, frame) {
  const key = color + '|' + frame;
  let c = spriteCache.get(key);
  if (c) return c;
  c = document.createElement('canvas');
  c.width = SPR_W;
  c.height = SPR_H;
  const g = c.getContext('2d');
  FRAMES[frame].forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      if (row[x] === '.') continue;
      g.fillStyle = row[x] === 't' ? color : SPRITE_INK[row[x]];
      g.fillRect(x * SPRITE_PX, y * SPRITE_PX, SPRITE_PX, SPRITE_PX);
    }
  });
  spriteCache.set(key, c);
  return c;
}

// --- socket ---

const socket = new WebSocket(SOCKET_URL);

socket.addEventListener('open', () => {
  statusEl.textContent = 'connected';
  // Reclaim the identity from the last visit, if any.
  const stored = nameInput.value.trim();
  if (stored) socket.send(JSON.stringify({ type: 'name', name: stored }));
});
socket.addEventListener('close', () => { statusEl.textContent = 'disconnected — refresh to rejoin'; });

socket.addEventListener('message', (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'init') {
    me = msg.id;
    players.clear();
    for (const p of msg.players) players.set(p.id, { ...p, rx: p.x, ry: p.y, facing: 1, moving: false });
  } else if (msg.type === 'joined') {
    const p = msg.player;
    players.set(p.id, { ...p, rx: p.x, ry: p.y, facing: 1, moving: false });
  } else if (msg.type === 'moved' && msg.id !== me) {
    const p = players.get(msg.id);
    if (p) { p.x = msg.x; p.y = msg.y; } // targets only; rx/ry ease there in frame()
  } else if (msg.type === 'emote') {
    effects.push({ id: msg.id, kind: msg.kind, start: performance.now() });
  } else if (msg.type === 'named') {
    const p = players.get(msg.id);
    if (p) p.name = msg.name;
  } else if (msg.type === 'chat') {
    bubbles.set(msg.id, { text: msg.text, until: performance.now() + BUBBLE_MS });
  } else if (msg.type === 'left') {
    players.delete(msg.id);
    bubbles.delete(msg.id);
  }
  countEl.textContent = players.size;
});

// Only send while actually moving: an idle tab sends nothing, so an idle
// room can hibernate — that is what keeps the server free.
setInterval(() => {
  if (!dirty || socket.readyState !== WebSocket.OPEN) return;
  const self = players.get(me);
  socket.send(JSON.stringify({ type: 'move', x: self.x, y: self.y }));
  dirty = false;
}, 1000 / SEND_HZ);

// --- input ---

window.addEventListener('keydown', (e) => {
  const el = document.activeElement;
  if (el && el.tagName === 'INPUT') {
    if (e.key === 'Escape') el.blur();
    return; // typing, not steering
  }
  if (e.key === 'Enter') {
    e.preventDefault();
    chatInput.focus();
    return;
  }
  if ((e.key === 'e' || e.key === 'q') && !e.repeat) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'emote', kind: e.key === 'e' ? 'wave' : 'heart' }));
    }
    return;
  }
  if (JUMP_KEYS.includes(e.key)) {
    e.preventDefault(); // space and up-arrow would scroll the page
    if (!e.repeat) jumpQueued = true;
    return;
  }
  if (e.key === 'ArrowDown' || e.key === 's') {
    e.preventDefault(); // nothing yet, but keep the page from scrolling
    return;
  }
  const dir = KEYS[e.key];
  if (!dir) return;
  e.preventDefault(); // arrows would scroll the page
  held.add(dir);
});
window.addEventListener('keyup', (e) => {
  const dir = KEYS[e.key];
  if (dir) held.delete(dir);
});

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (text && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'chat', text }));
  }
  chatInput.value = '';
  chatInput.blur(); // back to walking
});

// Name field: remembered locally, applied on change (Enter or blur). The
// server validates and broadcasts; the label updates from its echo.
nameInput.value = localStorage.getItem('play-name') ?? '';
nameInput.addEventListener('change', () => {
  const name = nameInput.value.trim();
  if (!/^[\w-]{2,16}$/.test(name)) return; // same rule the server enforces
  localStorage.setItem('play-name', name);
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'name', name }));
  }
});

// --- simulation + render ---

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05); // cap tab-switch jumps
  last = now;

  const self = me && players.get(me);
  if (self) {
    const floor = GROUND_Y - SIZE / 2; // sprite center when standing
    const grounded = self.y >= floor;

    const dx = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
    if (dx) {
      // Stricter than the server's clamp so the sprite never clips the edge.
      self.x = clamp(self.x + dx * SPEED * dt, SPR_W / 2, WORLD.w - SPR_W / 2);
      self.facing = dx > 0 ? 1 : -1;
      dirty = true;
    }
    self.moving = !!dx;

    if (jumpQueued) {
      if (grounded) vy = -JUMP_V;
      jumpQueued = false; // consume either way: no buffering jumps mid-air
    }
    if (!grounded || vy < 0) {
      vy += GRAVITY * dt;
      self.y = Math.max(self.y + vy * dt, SIZE / 2);
      if (self.y >= floor) { self.y = floor; vy = 0; } // landed
      dirty = true;
    }
  }

  // Your own sprite is drawn where you steered it; everyone else glides
  // toward their latest network position (framerate-independent easing).
  const k = 1 - Math.exp(-dt * SMOOTH_RATE);
  for (const [id, p] of players) {
    if (id === me) { p.rx = p.x; p.ry = p.y; continue; }
    const gx = p.x - p.rx;
    const gy = p.y - p.ry;
    if (Math.hypot(gx, gy) > SNAP_DIST) { p.rx = p.x; p.ry = p.y; continue; }
    p.rx += gx * k;
    p.ry += gy * k;
    // Remote facing and gait are inferred from how the sprite is gliding.
    if (Math.abs(gx) > 0.6) p.facing = gx > 0 ? 1 : -1;
    p.moving = Math.abs(gx) > 0.6;
  }

  draw();
  requestAnimationFrame(frame);
}

function draw() {
  ctx.fillStyle = '#16181d';
  ctx.fillRect(0, 0, WORLD.w, WORLD.h);

  // Faint grid so motion is legible against the flat backdrop.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.045)';
  ctx.beginPath();
  for (let x = 40; x < WORLD.w; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, WORLD.h); }
  for (let y = 40; y < GROUND_Y; y += 40) { ctx.moveTo(0, y); ctx.lineTo(WORLD.w, y); }
  ctx.stroke();

  // The floor.
  ctx.fillStyle = '#1c1f26';
  ctx.fillRect(0, GROUND_Y, WORLD.w, WORLD.h - GROUND_Y);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.beginPath();
  ctx.moveTo(0, GROUND_Y + 0.5);
  ctx.lineTo(WORLD.w, GROUND_Y + 0.5);
  ctx.stroke();

  const now = performance.now();
  for (let i = effects.length - 1; i >= 0; i--) {
    if (now - effects[i].start > EMOTE_MS) effects.splice(i, 1);
  }

  for (const [id, p] of players) {
    const feetY = p.ry + SIZE / 2;
    const airborne = p.ry < GROUND_Y - SIZE / 2 - 1;
    const frame = airborne ? 'jump'
      : p.moving ? (Math.floor(now / 140) % 2 ? 'walk1' : 'walk0')
      : 'idle';
    const img = sprite(p.color, frame);
    const wave = effects.find((f) => f.id === id && f.kind === 'wave');

    ctx.save();
    ctx.translate(p.rx, feetY - SPR_H / 2);
    if (wave) {
      // Wiggle: a decaying rock around the sprite's own center.
      const t = (now - wave.start) / EMOTE_MS;
      ctx.rotate(Math.sin(t * Math.PI * 6) * 0.35 * (1 - t));
    }
    if ((p.facing ?? 1) < 0) ctx.scale(-1, 1);
    ctx.drawImage(img, -SPR_W / 2, -SPR_H / 2);
    ctx.restore();

    // Your own name reads brighter; that is how you find yourself.
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = id === me ? 'rgba(255, 255, 255, 0.95)' : 'rgba(255, 255, 255, 0.55)';
    ctx.fillText(p.name, p.rx, feetY - SPR_H - 6);

    const bubble = bubbles.get(id);
    if (bubble) {
      if (now > bubble.until) bubbles.delete(id);
      else drawBubble(p, bubble.text);
    }
  }

  // Hearts float up over everything, fading as they rise.
  for (const f of effects) {
    if (f.kind !== 'heart') continue;
    const p = players.get(f.id);
    if (!p) continue;
    const t = (now - f.start) / EMOTE_MS;
    drawHeart(p.rx, p.ry + SIZE / 2 - SPR_H - 14 - t * 22, 1 - t);
  }
}

function drawHeart(x, y, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = '#e24b4a';
  ctx.beginPath();
  ctx.moveTo(x, y + 5);
  ctx.bezierCurveTo(x - 8, y - 3, x - 4, y - 10, x, y - 4);
  ctx.bezierCurveTo(x + 4, y - 10, x + 8, y - 3, x, y + 5);
  ctx.fill();
  ctx.restore();
}

function drawBubble(p, text) {
  ctx.font = '11px ui-monospace, monospace';
  const w = Math.min(ctx.measureText(text).width + 14, 240);
  const h = 20;
  // Keep the bubble on the canvas even at the edges of the world.
  const bx = clamp(p.rx - w / 2, 4, WORLD.w - w - 4);
  const by = clamp(p.ry + SIZE / 2 - SPR_H - 38, 4, WORLD.h - h - 4);

  ctx.fillStyle = 'rgba(244, 240, 230, 0.95)';
  ctx.beginPath();
  ctx.roundRect(bx, by, w, h, 6);
  ctx.fill();

  ctx.fillStyle = '#1d1f24';
  ctx.textAlign = 'center';
  ctx.fillText(text, bx + w / 2, by + 14, w - 10);
}

requestAnimationFrame(frame);
