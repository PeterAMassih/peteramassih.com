// src/scripts/play.js
// Multiplayer canvas client for /play — rendering, input, and the room socket.
// The server (game/src/room.ts) owns the truth; this draws it and asks to move.

const WORLD = { w: 640, h: 360 }; // mirrored in game/src/room.ts
const SIZE = 16; // sprite square, px
const SPEED = 160; // px per second
const SEND_HZ = 10; // position updates to the server, per second

const SOCKET_URL = import.meta.env.DEV
  ? 'ws://localhost:8787'
  : 'wss://peteramassih-play.peteramassih.workers.dev';

const KEYS = {
  ArrowUp: 'up', w: 'up',
  ArrowDown: 'down', s: 'down',
  ArrowLeft: 'left', a: 'left',
  ArrowRight: 'right', d: 'right',
};

const BUBBLE_MS = 4000; // how long a chat bubble hangs over a sprite

const canvas = document.getElementById('play-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('play-status');
const countEl = document.getElementById('play-count');
const chatForm = document.getElementById('play-chat-form');
const chatInput = document.getElementById('play-chat');

let me = null; // my id, assigned by the server in init
const players = new Map(); // id -> { x, y, color, name }
const bubbles = new Map(); // id -> { text, until }
const held = new Set(); // movement keys currently down
let dirty = false; // my position changed since the last network send
let last = 0; // previous frame timestamp

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

// --- socket ---

const socket = new WebSocket(SOCKET_URL);

socket.addEventListener('open', () => { statusEl.textContent = 'connected'; });
socket.addEventListener('close', () => { statusEl.textContent = 'disconnected — refresh to rejoin'; });

socket.addEventListener('message', (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'init') {
    me = msg.id;
    players.clear();
    for (const p of msg.players) players.set(p.id, p);
  } else if (msg.type === 'joined') {
    players.set(msg.player.id, msg.player);
  } else if (msg.type === 'moved' && msg.id !== me) {
    const p = players.get(msg.id);
    if (p) { p.x = msg.x; p.y = msg.y; }
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
  if (document.activeElement === chatInput) {
    if (e.key === 'Escape') chatInput.blur();
    return; // typing a message, not steering
  }
  if (e.key === 'Enter') {
    e.preventDefault();
    chatInput.focus();
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

// --- simulation + render ---

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05); // cap tab-switch jumps
  last = now;

  const self = me && players.get(me);
  if (self) {
    let dx = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
    let dy = (held.has('down') ? 1 : 0) - (held.has('up') ? 1 : 0);
    if (dx && dy) { dx *= Math.SQRT1_2; dy *= Math.SQRT1_2; } // diagonals no faster
    if (dx || dy) {
      // Stricter than the server's clamp so the sprite never clips the edge.
      self.x = clamp(self.x + dx * SPEED * dt, SIZE / 2, WORLD.w - SIZE / 2);
      self.y = clamp(self.y + dy * SPEED * dt, SIZE / 2, WORLD.h - SIZE / 2);
      dirty = true;
    }
  }

  draw();
  requestAnimationFrame(frame);
}

function draw() {
  ctx.fillStyle = '#16181d';
  ctx.fillRect(0, 0, WORLD.w, WORLD.h);

  // Faint grid so motion is legible against the flat floor.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.045)';
  ctx.beginPath();
  for (let x = 40; x < WORLD.w; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, WORLD.h); }
  for (let y = 40; y < WORLD.h; y += 40) { ctx.moveTo(0, y); ctx.lineTo(WORLD.w, y); }
  ctx.stroke();

  const now = performance.now();
  for (const [id, p] of players) {
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x - SIZE / 2, p.y - SIZE / 2, SIZE, SIZE);
    if (id === me) {
      // A ring with a gap marks your own square, whatever its color.
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeRect(p.x - SIZE / 2 - 3.5, p.y - SIZE / 2 - 3.5, SIZE + 7, SIZE + 7);
    }

    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fillText(p.name, p.x, p.y - SIZE / 2 - 8);

    const bubble = bubbles.get(id);
    if (bubble) {
      if (now > bubble.until) bubbles.delete(id);
      else drawBubble(p, bubble.text);
    }
  }
}

function drawBubble(p, text) {
  ctx.font = '11px ui-monospace, monospace';
  const w = Math.min(ctx.measureText(text).width + 14, 240);
  const h = 20;
  // Keep the bubble on the canvas even at the edges of the world.
  const bx = clamp(p.x - w / 2, 4, WORLD.w - w - 4);
  const by = clamp(p.y - SIZE / 2 - 42, 4, WORLD.h - h - 4);

  ctx.fillStyle = 'rgba(244, 240, 230, 0.95)';
  ctx.beginPath();
  ctx.roundRect(bx, by, w, h, 6);
  ctx.fill();

  ctx.fillStyle = '#1d1f24';
  ctx.textAlign = 'center';
  ctx.fillText(text, bx + w / 2, by + 14, w - 10);
}

requestAnimationFrame(frame);
