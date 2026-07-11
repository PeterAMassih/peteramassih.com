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

// The brawl, client side. The server decides who got hit; the target's own
// browser applies the knockback (one movement authority per sprite), and
// everyone renders the tumble and the immunity blink. STUN/IMMUNE mirror
// game/src/room.ts.
const PUNCH_SHOW_MS = 220; // how long the extended-arm frame shows
const STUN_MS = 700;
const IMMUNE_MS = 1500;
const KB_VX = 260; // knockback launch speed, px/s, decaying
const KB_POP = -220; // the little upward pop on getting hit
const BURST_MS = 350; // impact starburst lifetime
const SHAKE_MS = 180; // screen shake after a hit
const SMOOTH_RATE = 12; // per second: how fast remote sprites chase their target
const SNAP_DIST = 80; // farther than this is a spawn or teleport, not a walk: snap

const canvas = document.getElementById('play-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('play-status');
const countEl = document.getElementById('play-count');
const chatForm = document.getElementById('play-chat-form');
const chatInput = document.getElementById('play-chat');
const nameInput = document.getElementById('play-name');
const touchPad = document.getElementById('play-touch');

let me = null; // my id, assigned by the server in init
// x/y are the network truth; rx/ry are where the sprite is drawn. Remote
// sprites ease rx/ry toward x/y so 10 Hz updates read as walking, not teleports.
const players = new Map(); // id -> { x, y, rx, ry, color, name }
const bubbles = new Map(); // id -> { text, until }
const effects = []; // running emotes: { id, kind, start }
const held = new Set(); // movement keys currently down
let vy = 0; // my vertical speed
let jumpQueued = false; // jump pressed, consumed by the next frame
let kbVx = 0; // horizontal knockback speed while stunned
let stunUntil = 0; // while stunned, input is ignored — you are tumbling
let shakeUntil = 0; // screen shake after a nearby hit
let shakeMag = 0;
let dirty = false; // my position changed since the last network send
let last = 0; // previous frame timestamp

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

// --- sound ---
// Synthesized with WebAudio: no files, no bundle weight. Each effect is one
// or two oscillators with a pitch slide and a fast decay — 8-bit flavored.
// The context can only start after a user gesture, so it is created (and
// resumed) from the keydown handler; events before that are just silent.
let audioCtx = null;
let muted = localStorage.getItem('play-muted') === '1';

function tone(shape, f0, f1, dur, vol, delay = 0) {
  const t = audioCtx.currentTime + delay;
  const o = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  o.type = shape;
  o.frequency.setValueAtTime(f0, t);
  o.frequency.exponentialRampToValueAtTime(Math.max(f1, 1), t + dur);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.connect(g).connect(audioCtx.destination);
  o.start(t);
  o.stop(t + dur);
}

// Browsers only allow audio after a user gesture — a keypress or a tap.
function unlockAudio() {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function sfx(kind) {
  if (muted || !audioCtx || audioCtx.state !== 'running') return;
  if (kind === 'hit') {
    tone('square', 150, 40, 0.14, 0.1);
    tone('sawtooth', 90, 28, 0.18, 0.06);
  } else if (kind === 'swing') {
    tone('sine', 480, 160, 0.06, 0.025);
  } else if (kind === 'jump') {
    tone('square', 200, 420, 0.09, 0.03);
  }
}

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
  punch: [
    '....hhhh....',
    '...hhhhhh...',
    '...hssssh...',
    '...sesses...',
    '...ssssss...',
    '....ssss....',
    '...tttttt...',
    '..tttttttt..',
    '..tttttttss.',
    '..tttttttt..',
    '...tttttt...',
    '...pppppp...',
    '...pp..pp...',
    '...pp..pp...',
    '...bb..bb...',
    '...bb..bb...',
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
  } else if (msg.type === 'swung') {
    const p = players.get(msg.id);
    if (p) { p.punchUntil = performance.now() + PUNCH_SHOW_MS; p.facing = msg.dir; }
    sfx('swing');
  } else if (msg.type === 'hit') {
    const now = performance.now();
    const t = players.get(msg.target);
    if (t) {
      t.hurtUntil = now + STUN_MS;
      t.immuneUntil = now + STUN_MS + IMMUNE_MS;
      // The impact burst lives at the moment of contact, not on a player.
      effects.push({ kind: 'burst', x: t.rx, y: t.ry - 4, start: now });
    }
    shakeUntil = now + SHAKE_MS;
    shakeMag = msg.target === me ? 5 : 2.5;
    sfx('hit');
    if (msg.target === me) {
      // I got punched: my own physics takes the impulse.
      stunUntil = now + STUN_MS;
      kbVx = msg.dir * KB_VX;
      vy = KB_POP;
    }
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
  unlockAudio();
  const el = document.activeElement;
  if (el && el.tagName === 'INPUT') {
    if (e.key === 'Escape') el.blur();
    return; // typing, not steering
  }
  if (e.key === 'm' && !e.repeat) {
    muted = !muted;
    localStorage.setItem('play-muted', muted ? '1' : '0');
    return;
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
  if (e.key === 'z' && !e.repeat) {
    sendPunch();
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

function sendPunch() {
  const self = me && players.get(me);
  if (self && performance.now() >= stunUntil && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'punch', facing: self.facing ?? 1 }));
  }
}

// Touch pad (rendered only on coarse-pointer devices). Each finger is
// tracked by pointerId so holding a walk button while tapping jump with
// the other thumb releases the right thing.
const touchActs = new Map(); // pointerId -> act
touchPad.addEventListener('pointerdown', (e) => {
  const act = e.target.dataset.act;
  if (!act) return;
  e.preventDefault();
  unlockAudio();
  if (act === 'left' || act === 'right') { held.add(act); touchActs.set(e.pointerId, act); }
  else if (act === 'jump') jumpQueued = true;
  else if (act === 'punch') sendPunch();
  // Last, since it can throw on an already-gone pointer: keep receiving the
  // up event even when the finger slides off the button.
  e.target.setPointerCapture(e.pointerId);
});
const releaseTouch = (e) => {
  const act = touchActs.get(e.pointerId);
  if (act) { held.delete(act); touchActs.delete(e.pointerId); }
};
touchPad.addEventListener('pointerup', releaseTouch);
touchPad.addEventListener('pointercancel', releaseTouch);

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
    const stunned = performance.now() < stunUntil;

    const dx = stunned ? 0 : (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
    if (dx) {
      // Stricter than the server's clamp so the sprite never clips the edge.
      self.x = clamp(self.x + dx * SPEED * dt, SPR_W / 2, WORLD.w - SPR_W / 2);
      self.facing = dx > 0 ? 1 : -1;
      dirty = true;
    }
    self.moving = !!dx;

    // Knockback: steering is gone, the impulse carries you and bleeds off.
    if (kbVx) {
      self.x = clamp(self.x + kbVx * dt, SPR_W / 2, WORLD.w - SPR_W / 2);
      kbVx *= Math.exp(-dt * 4);
      if (!stunned || Math.abs(kbVx) < 4) kbVx = 0;
      dirty = true;
    }

    if (jumpQueued) {
      if (grounded && !stunned) { vy = -JUMP_V; sfx('jump'); }
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
  const now = performance.now();
  ctx.fillStyle = '#16181d';
  ctx.fillRect(0, 0, WORLD.w, WORLD.h);

  // A hit rattles the whole room, briefly. Skipped for reduced-motion users.
  ctx.save();
  if (now < shakeUntil && !reducedMotion) {
    ctx.translate((Math.random() - 0.5) * 2 * shakeMag, (Math.random() - 0.5) * 2 * shakeMag);
  }

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

  for (let i = effects.length - 1; i >= 0; i--) {
    const life = effects[i].kind === 'burst' ? BURST_MS : EMOTE_MS;
    if (now - effects[i].start > life) effects.splice(i, 1);
  }

  for (const [id, p] of players) {
    const feetY = p.ry + SIZE / 2;
    const airborne = p.ry < GROUND_Y - SIZE / 2 - 1;
    const hurt = now < (p.hurtUntil ?? 0);
    const frame = now < (p.punchUntil ?? 0) ? 'punch'
      : airborne ? 'jump'
      : p.moving ? (Math.floor(now / 140) % 2 ? 'walk1' : 'walk0')
      : 'idle';
    const img = sprite(p.color, frame);
    const wave = effects.find((f) => f.id === id && f.kind === 'wave');

    ctx.save();
    ctx.translate(p.rx, feetY - SPR_H / 2);
    if (hurt) {
      // The tumble: a fast, hard wobble for as long as the stun lasts.
      ctx.rotate(Math.sin(now / 40) * 0.5);
    } else if (wave) {
      // Wiggle: a decaying rock around the sprite's own center.
      const t = (now - wave.start) / EMOTE_MS;
      ctx.rotate(Math.sin(t * Math.PI * 6) * 0.35 * (1 - t));
    }
    if ((p.facing ?? 1) < 0) ctx.scale(-1, 1);
    if (now < (p.immuneUntil ?? 0)) {
      // Immunity blink: untouchable, and visibly so.
      ctx.globalAlpha = 0.45 + 0.55 * Math.abs(Math.sin(now / 90));
    }
    ctx.drawImage(img, -SPR_W / 2, -SPR_H / 2);
    if (frame === 'punch') {
      // Speed lines off the fist, fading with the swing.
      const st = 1 - (p.punchUntil - now) / PUNCH_SHOW_MS;
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.5 * (1 - st)})`;
      ctx.beginPath();
      ctx.moveTo(SPR_W / 2 + 2, -6);
      ctx.lineTo(SPR_W / 2 + 8 + st * 6, -6);
      ctx.moveTo(SPR_W / 2 + 3, -1);
      ctx.lineTo(SPR_W / 2 + 11 + st * 6, -1);
      ctx.stroke();
    }
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
    if (f.kind === 'heart') {
      const p = players.get(f.id);
      if (!p) continue;
      const t = (now - f.start) / EMOTE_MS;
      drawHeart(p.rx, p.ry + SIZE / 2 - SPR_H - 14 - t * 22, 1 - t);
    } else if (f.kind === 'burst') {
      drawBurst(f.x, f.y, (now - f.start) / BURST_MS);
    }
  }

  ctx.restore(); // end of shake
}

function drawBurst(x, y, t) {
  const spikes = 8;
  const outer = 10 + t * 16;
  const inner = outer * 0.45;
  ctx.save();
  ctx.globalAlpha = 1 - t;
  ctx.translate(x, y);
  ctx.rotate(t * 0.8);
  ctx.beginPath();
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 ? inner : outer;
    const a = (i / (spikes * 2)) * Math.PI * 2;
    ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
  }
  ctx.closePath();
  ctx.fillStyle = '#ffd23f';
  ctx.fill();
  ctx.strokeStyle = '#e0930f';
  ctx.stroke();
  if (t < 0.6) {
    ctx.fillStyle = '#a3291f';
    ctx.font = 'bold 10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('pow!', 0, 3.5);
  }
  ctx.restore();
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
