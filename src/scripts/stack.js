// src/scripts/stack.js
// Falling-tetromino game wired from src/pages/stack.astro.
// Sections: pieces, constants, board, state, moves, render, loop, input,
// controls, audio (Korobeiniki melody + bass + SFX), init.


/* ── Pieces ──────────────────────────────────────────────────────────────── */
// Each piece is a list of rotation states; each state is 4 [row, col] offsets
// inside a 4x4 bounding box. Spawn column is fixed at 3, which centers all
// pieces on a 10-wide board.

const PIECES = {
  I: [
    [[1,0],[1,1],[1,2],[1,3]],
    [[0,2],[1,2],[2,2],[3,2]],
    [[2,0],[2,1],[2,2],[2,3]],
    [[0,1],[1,1],[2,1],[3,1]],
  ],
  O: [
    [[0,1],[0,2],[1,1],[1,2]],
  ],
  T: [
    [[0,1],[1,0],[1,1],[1,2]],
    [[0,1],[1,1],[1,2],[2,1]],
    [[1,0],[1,1],[1,2],[2,1]],
    [[0,1],[1,0],[1,1],[2,1]],
  ],
  S: [
    [[0,1],[0,2],[1,0],[1,1]],
    [[0,1],[1,1],[1,2],[2,2]],
    [[1,1],[1,2],[2,0],[2,1]],
    [[0,0],[1,0],[1,1],[2,1]],
  ],
  Z: [
    [[0,0],[0,1],[1,1],[1,2]],
    [[0,2],[1,1],[1,2],[2,1]],
    [[1,0],[1,1],[2,1],[2,2]],
    [[0,1],[1,0],[1,1],[2,0]],
  ],
  J: [
    [[0,0],[1,0],[1,1],[1,2]],
    [[0,1],[0,2],[1,1],[2,1]],
    [[1,0],[1,1],[1,2],[2,2]],
    [[0,1],[1,1],[2,0],[2,1]],
  ],
  L: [
    [[0,2],[1,0],[1,1],[1,2]],
    [[0,1],[1,1],[2,1],[2,2]],
    [[1,0],[1,1],[1,2],[2,0]],
    [[0,0],[0,1],[1,1],[2,1]],
  ],
};

const PIECE_NAMES = Object.keys(PIECES);


/* ── Constants ───────────────────────────────────────────────────────────── */

const COLS = 10;
const ROWS = 20;
const CELL = 30;

// Standard Tetris scoring: single 40, double 100, triple 300, tetris 1200,
// each multiplied by current level.
const SCORE = [0, 40, 100, 300, 1200];

// Game feel. DAS = delay before auto-shift kicks in; ARR = interval between
// shifts once it does. SOFT_REPEAT is the held-down-arrow drop interval.
// LOCK_DELAY gives a window to slide a piece along the floor before it locks.
const DAS_DELAY = 150;
const DAS_REPEAT = 40;
const SOFT_REPEAT = 30;
const LOCK_DELAY = 500;
const CLEAR_DURATION = 280; // ms; how long the line-clear flash lasts

const HIGH_SCORE_KEY = 'stack:high';


/* ── Board ───────────────────────────────────────────────────────────────── */

function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

function collides(board, shape, row, col) {
  for (const [dr, dc] of shape) {
    const r = row + dr;
    const c = col + dc;
    if (c < 0 || c >= COLS || r >= ROWS) return true;
    if (r >= 0 && board[r][c]) return true;
  }
  return false;
}

function merge(board, shape, row, col) {
  for (const [dr, dc] of shape) {
    const r = row + dr;
    if (r >= 0) board[r][col + dc] = 1;
  }
}


/* ── State ───────────────────────────────────────────────────────────────── */

let board, bag, current;
let score, lines, level, dropInterval, dropCounter;
let paused, gameOver;
let clearing = null;        // { rows: number[], elapsed: number, duration: number }
let lockTimer = 0;
let highScore = 0;
let lastTime = 0;

// Held-key state for DAS / soft drop auto-repeat.
let dasDir = 0;
let dasSince = 0;
let lastDasShift = 0;
let softDropHeld = false;
let lastSoftDrop = 0;

function reset() {
  board = emptyBoard();
  bag = [];
  score = 0;
  lines = 0;
  level = 1;
  dropInterval = 1000;
  dropCounter = 0;
  paused = false;
  gameOver = false;
  clearing = null;
  lockTimer = 0;
  dasDir = 0;
  softDropHeld = false;
  spawn();
}

// 7-bag randomizer: deal each piece once before reshuffling. Prevents long
// droughts of any piece.
function nextName() {
  if (bag.length === 0) {
    bag = PIECE_NAMES.slice();
    for (let i = bag.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [bag[i], bag[j]] = [bag[j], bag[i]];
    }
  }
  return bag.shift();
}

function shapeOf(piece) {
  return PIECES[piece.name][piece.rotation];
}

function spawn() {
  current = { name: nextName(), rotation: 0, row: 0, col: 3 };
  lockTimer = 0;
  if (collides(board, shapeOf(current), current.row, current.col)) {
    gameOver = true;
    if (score > highScore) { highScore = score; saveHighScore(); }
    updateMeta();
    stopMusic();
    sfxGameOver();
  }
}

function loadHighScore() {
  const stored = localStorage.getItem(HIGH_SCORE_KEY);
  highScore = stored ? parseInt(stored, 10) || 0 : 0;
}

function saveHighScore() {
  try { localStorage.setItem(HIGH_SCORE_KEY, String(highScore)); } catch {}
}


/* ── Moves ───────────────────────────────────────────────────────────────── */

// Reset the lock timer if the piece is on the floor — gives players room to
// slide and rotate against the floor without insta-locking.
function maybeResetLock() {
  if (current && collides(board, shapeOf(current), current.row + 1, current.col)) {
    lockTimer = 0;
  }
}

function move(dx) {
  if (!current || clearing) return;
  if (!collides(board, shapeOf(current), current.row, current.col + dx)) {
    current.col += dx;
    maybeResetLock();
  }
}

function rotate() {
  if (!current || clearing) return;
  const rots = PIECES[current.name];
  const next = (current.rotation + 1) % rots.length;
  // Simple wall kicks: try same column, then nudge ±1 then ±2.
  for (const dx of [0, -1, 1, -2, 2]) {
    if (!collides(board, rots[next], current.row, current.col + dx)) {
      current.rotation = next;
      current.col += dx;
      maybeResetLock();
      return;
    }
  }
}

function softDrop() {
  if (!current || clearing) return;
  if (!collides(board, shapeOf(current), current.row + 1, current.col)) {
    current.row++;
    score += 1;
    updateMeta();
    lockTimer = 0;
  } else {
    lock();
  }
}

function hardDrop() {
  if (!current || clearing) return;
  let d = 0;
  while (!collides(board, shapeOf(current), current.row + d + 1, current.col)) d++;
  current.row += d;
  score += d * 2;
  lock();
}

function lock() {
  if (!current) return;
  merge(board, shapeOf(current), current.row, current.col);
  sfxLock();

  const fullRows = [];
  for (let r = 0; r < ROWS; r++) {
    if (board[r].every((v) => v)) fullRows.push(r);
  }

  if (fullRows.length > 0) {
    clearing = { rows: fullRows, elapsed: 0, duration: CLEAR_DURATION };
    sfxLineClear(fullRows.length);
    current = null;
  } else {
    spawn();
  }
}

function completeLineClear() {
  if (!clearing) return;
  const cleared = clearing.rows.length;
  // Remove from the highest row index down so splice doesn't shift the others.
  const sorted = clearing.rows.slice().sort((a, b) => b - a);
  for (const r of sorted) {
    board.splice(r, 1);
    board.unshift(Array(COLS).fill(0));
  }
  lines += cleared;
  score += SCORE[cleared] * level;
  if (score > highScore) { highScore = score; saveHighScore(); }
  const newLevel = Math.floor(lines / 10) + 1;
  if (newLevel !== level) {
    level = newLevel;
    dropInterval = Math.max(80, 1000 - (level - 1) * 80);
  }
  updateMeta();
  clearing = null;
  spawn();
}


/* ── Render ──────────────────────────────────────────────────────────────── */

let canvas, ctx;

// Read CSS custom properties live each frame so dark-mode toggles and system
// theme changes are picked up without any extra listener.
function color(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawCell(c, r) {
  ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
}

function ghostDrop() {
  let d = 0;
  while (!collides(board, shapeOf(current), current.row + d + 1, current.col)) d++;
  return d;
}

function render() {
  ctx.fillStyle = color('--color-code-bg');
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = color('--color-border');
  ctx.lineWidth = 1;
  for (let c = 1; c < COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * CELL + 0.5, 0);
    ctx.lineTo(c * CELL + 0.5, canvas.height);
    ctx.stroke();
  }
  for (let r = 1; r < ROWS; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * CELL + 0.5);
    ctx.lineTo(canvas.width, r * CELL + 0.5);
    ctx.stroke();
  }

  // Locked stack — skip rows being cleared so they don't double-draw under the flash.
  ctx.fillStyle = color('--color-text-muted');
  for (let r = 0; r < ROWS; r++) {
    if (clearing && clearing.rows.includes(r)) continue;
    for (let c = 0; c < COLS; c++) {
      if (board[r][c]) drawCell(c, r);
    }
  }

  // Line-clear flash: brief bright pop, then fade in accent.
  if (clearing) {
    const t = clearing.elapsed / clearing.duration;
    if (t < 0.3) {
      ctx.fillStyle = color('--color-text');
    } else {
      ctx.globalAlpha = 1 - (t - 0.3) / 0.7;
      ctx.fillStyle = color('--color-accent');
    }
    for (const r of clearing.rows) {
      for (let c = 0; c < COLS; c++) drawCell(c, r);
    }
    ctx.globalAlpha = 1;
  }

  if (current && !gameOver && !clearing) {
    const d = ghostDrop();
    ctx.fillStyle = color('--color-border');
    for (const [dr, dc] of shapeOf(current)) {
      const r = current.row + d + dr;
      if (r >= 0) drawCell(current.col + dc, r);
    }
    ctx.fillStyle = color('--color-accent');
    for (const [dr, dc] of shapeOf(current)) {
      const r = current.row + dr;
      if (r >= 0) drawCell(current.col + dc, r);
    }
  }

  if (gameOver) overlay('game over', 'press r to restart');
  else if (paused) overlay('paused', '');
}

function overlay(title, sub) {
  ctx.globalAlpha = 0.88;
  ctx.fillStyle = color('--color-bg');
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color('--color-text');
  ctx.font = '16px "JetBrains Mono", ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(title, canvas.width / 2, canvas.height / 2 - (sub ? 8 : 0));
  if (sub) {
    ctx.fillStyle = color('--color-text-subtle');
    ctx.font = '12px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillText(sub, canvas.width / 2, canvas.height / 2 + 14);
  }
}


/* ── Gravity loop ────────────────────────────────────────────────────────── */

function tick(now) {
  const dt = now - lastTime;
  lastTime = now;
  if (paused || gameOver) { render(); requestAnimationFrame(tick); return; }

  if (clearing) {
    clearing.elapsed += dt;
    if (clearing.elapsed >= clearing.duration) completeLineClear();
    render();
    requestAnimationFrame(tick);
    return;
  }

  // DAS auto-shift for held left/right.
  if (current && dasDir !== 0 && now - dasSince >= DAS_DELAY) {
    while (now - lastDasShift >= DAS_REPEAT) {
      move(dasDir);
      lastDasShift += DAS_REPEAT;
    }
  }

  // Soft-drop auto-repeat for held down.
  if (current && softDropHeld) {
    while (now - lastSoftDrop >= SOFT_REPEAT) {
      softDrop();
      lastSoftDrop += SOFT_REPEAT;
      if (!current || clearing) break;
    }
  }

  if (!current || clearing) { render(); requestAnimationFrame(tick); return; }

  // Gravity + lock delay. On the floor the piece doesn't fall but starts the
  // lock countdown; off the floor it falls on the regular drop interval.
  const onFloor = collides(board, shapeOf(current), current.row + 1, current.col);
  if (onFloor) {
    lockTimer += dt;
    if (lockTimer >= LOCK_DELAY) lock();
  } else {
    lockTimer = 0;
    dropCounter += dt;
    if (dropCounter >= dropInterval) {
      current.row++;
      dropCounter = 0;
    }
  }

  render();
  requestAnimationFrame(tick);
}


/* ── Input ───────────────────────────────────────────────────────────────── */

const GAME_KEYS = new Set([
  'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Space', 'KeyP', 'KeyR',
]);

function handleKey(e) {
  if (!GAME_KEYS.has(e.code)) return;
  // Buttons handle their own Space/Enter; don't hijack when one is focused.
  const focused = document.activeElement;
  if (focused && focused.tagName === 'BUTTON' && (e.code === 'Space' || e.code === 'Enter')) return;
  e.preventDefault();

  if (e.code === 'KeyR' && gameOver) { resetAndStart(); return; }
  if (e.code === 'KeyP') { togglePause(); return; }
  if (paused || gameOver) return;
  // Skip OS-level repeats — we run our own DAS/ARR loop in tick().
  if (e.repeat) return;

  const now = performance.now();
  switch (e.code) {
    case 'ArrowLeft':
      move(-1);
      dasDir = -1; dasSince = now; lastDasShift = now;
      break;
    case 'ArrowRight':
      move(1);
      dasDir = 1; dasSince = now; lastDasShift = now;
      break;
    case 'ArrowDown':
      softDrop();
      softDropHeld = true; lastSoftDrop = now;
      break;
    case 'ArrowUp':
      rotate();
      break;
    case 'Space':
      hardDrop();
      break;
  }
}

function handleKeyUp(e) {
  if (e.code === 'ArrowLeft' && dasDir === -1) dasDir = 0;
  else if (e.code === 'ArrowRight' && dasDir === 1) dasDir = 0;
  else if (e.code === 'ArrowDown') softDropHeld = false;
}


/* ── Touch input ─────────────────────────────────────────────────────────── */

// Drag horizontally to shift the piece (one column per SWIPE_CELL of travel).
// Drag downward to soft-drop. A short stationary tap rotates. An upward flick
// on release is a hard drop. Tapping the canvas while game-over restarts.
const SWIPE_CELL = 28;
const SWIPE_HARD = 60;
const TAP_SLOP = 10;

let touchStartX = 0;
let touchStartY = 0;
let touchLastX = 0;
let touchLastY = 0;
let touchMoved = false;

function handleTouchStart(e) {
  if (e.touches.length !== 1) return;
  e.preventDefault();
  if (gameOver) { resetAndStart(); return; }
  const t = e.touches[0];
  touchStartX = touchLastX = t.clientX;
  touchStartY = touchLastY = t.clientY;
  touchMoved = false;
}

function handleTouchMove(e) {
  if (e.touches.length !== 1) return;
  e.preventDefault();
  if (paused || gameOver || clearing || !current) return;
  const t = e.touches[0];

  const dx = t.clientX - touchLastX;
  if (Math.abs(dx) >= SWIPE_CELL) {
    const cells = Math.trunc(dx / SWIPE_CELL);
    const dir = Math.sign(cells);
    for (let i = 0; i < Math.abs(cells); i++) move(dir);
    touchLastX += cells * SWIPE_CELL;
    touchMoved = true;
  }

  const dy = t.clientY - touchLastY;
  if (dy >= SWIPE_CELL) {
    const drops = Math.trunc(dy / SWIPE_CELL);
    for (let i = 0; i < drops; i++) {
      softDrop();
      if (!current || clearing) break;
    }
    touchLastY += drops * SWIPE_CELL;
    touchMoved = true;
  }
}

function handleTouchEnd(e) {
  e.preventDefault();
  if (paused || gameOver || clearing) return;

  const totalDx = touchLastX - touchStartX;
  const totalDy = touchLastY - touchStartY;

  if (!touchMoved && Math.abs(totalDx) < TAP_SLOP && Math.abs(totalDy) < TAP_SLOP) {
    rotate();
  } else if (totalDy < -SWIPE_HARD) {
    hardDrop();
  }
}


/* ── Controls + meta ─────────────────────────────────────────────────────── */

let scoreEl, highEl, linesEl, levelEl, toggleBtn, restartBtn, soundBtn;

function updateMeta() {
  scoreEl.textContent = String(score);
  highEl.textContent = String(highScore);
  linesEl.textContent = String(lines);
  levelEl.textContent = String(level);
}

function togglePause() {
  if (gameOver) return;
  paused = !paused;
  toggleBtn.textContent = paused ? 'resume' : 'pause';
  if (paused) {
    stopMusic();
    // Clear held-key state so resume doesn't apply queued shifts.
    dasDir = 0;
    softDropHeld = false;
  } else {
    lastTime = performance.now();
    if (soundOn) startMusic();
  }
}

function resetAndStart() {
  reset();
  toggleBtn.textContent = 'pause';
  updateMeta();
  if (soundOn) { melodyIndex = 0; bassIndex = 0; startMusic(); }
}


/* ── Audio: Korobeiniki melody + bass via Web Audio ──────────────────────── */

const FREQ = {
  E2: 82.41, A2: 110.00, B2: 123.47, D3: 146.83, E3: 164.81, 'F#3': 185.00,
  A3: 220.00, B3: 246.94, C4: 261.63, D4: 293.66, E4: 329.63, F4: 349.23, G4: 392.00,
  A4: 440.00, B4: 493.88, C5: 523.25, D5: 587.33, E5: 659.25, F5: 698.46, G5: 783.99,
  A5: 880.00, B5: 987.77, C6: 1046.50, D6: 1174.66, E6: 1318.51,
};

// Durations in sixteenths: quarter = 4, eighth = 2, dotted-quarter = 6.
// null = rest. Plays the canonical AABA folk form twice — once in the high
// register, once an octave down — giving the song the proper verse / verse /
// bridge / verse arc instead of a short loop. ~52 seconds at 144 BPM.
const A_HIGH = [
  ['E5',4],['B4',2],['C5',2], ['D5',4],['C5',2],['B4',2],
  ['A4',4],['A4',2],['C5',2], ['E5',4],['D5',2],['C5',2],
  ['B4',6],['C5',2],          ['D5',4],['E5',4],
  ['C5',4],['A4',4],          ['A4',4],[null,4],
];
const B_HIGH = [
  ['D5',4],['F5',2],['A5',2], ['G5',2],['F5',2],['E5',6],['C5',2],
  ['E5',4],['D5',2],['C5',2], ['B4',6],['C5',2],
  ['D5',4],['E5',4],          ['C5',4],['A4',4],
  ['A4',4],[null,4],
];
const A_LOW = [
  ['E4',4],['B3',2],['C4',2], ['D4',4],['C4',2],['B3',2],
  ['A3',4],['A3',2],['C4',2], ['E4',4],['D4',2],['C4',2],
  ['B3',6],['C4',2],          ['D4',4],['E4',4],
  ['C4',4],['A3',4],          ['A3',4],[null,4],
];
const B_LOW = [
  ['D4',4],['F4',2],['A4',2], ['G4',2],['F4',2],['E4',6],['C4',2],
  ['E4',4],['D4',2],['C4',2], ['B3',6],['C4',2],
  ['D4',4],['E4',4],          ['C4',4],['A3',4],
  ['A3',4],[null,4],
];
// Climactic statement of A one octave above A_HIGH — the "ff" peak that
// concert arrangements use as the dynamic high point before recapitulation.
const A_VHIGH = [
  ['E6',4],['B5',2],['C6',2], ['D6',4],['C6',2],['B5',2],
  ['A5',4],['A5',2],['C6',2], ['E6',4],['D6',2],['C6',2],
  ['B5',6],['C6',2],          ['D6',4],['E6',4],
  ['C6',4],['A5',4],          ['A5',4],[null,4],
];
// Verse / verse / bridge / verse in the home register, mirrored an octave
// below, then a climactic octave-up statement, a full recapitulation, and a
// single A-statement coda. ~94 seconds at 144 BPM — the length the long
// concert arrangements settle into.
const KOROBEINIKI = [
  ...A_HIGH,  ...A_HIGH, ...B_HIGH, ...A_HIGH,   // first pass, home register
  ...A_LOW,   ...A_LOW,  ...B_LOW,  ...A_LOW,    // second pass, octave down
  ...A_VHIGH,                                     // climax, octave up
  ...A_HIGH,  ...A_HIGH, ...B_HIGH, ...A_HIGH,   // recapitulation
  ...A_HIGH,                                      // coda
];

// Root-fifth boom-chick bass under each implied chord. Eighth notes (2 sixteenths).
// Each 4-note group = one bar = 8 sixteenths; 16 bars total = 128 sixteenths,
// which matches the melody total above.
const KOROBEINIKI_BASS = (() => {
  const Em  = [['E2',2],['B2',2],['E2',2],['B2',2]];
  const Am  = [['A2',2],['E3',2],['A2',2],['E3',2]];
  const B7  = [['B2',2],['F#3',2],['B2',2],['F#3',2]];
  const AmB = [['A2',2],['E3',2],['B2',2],['F#3',2]];
  const verse  = [...Em,...Em, ...Am,...Em, ...B7,...Em, ...AmB,...Em];
  const bridge = [...Em,...Em, ...Em,...Em, ...B7,...Em, ...AmB,...Em];
  // Mirrors the melody: AABA high, AABA low, climax (verse), AABA return, coda.
  return [
    ...verse, ...verse, ...bridge, ...verse,
    ...verse, ...verse, ...bridge, ...verse,
    ...verse,
    ...verse, ...verse, ...bridge, ...verse,
    ...verse,
  ];
})();

const BPM = 144;

let audioCtx = null;
let soundOn = false;
let nextNoteTime = 0;
let nextBassTime = 0;
let melodyIndex = 0;
let bassIndex = 0;
let schedulerInterval = null;

function sixteenthSeconds() { return 60 / BPM / 4; }

function initAudio() {
  if (audioCtx) return;
  // Bracket access on the legacy Safari prefix to keep TS strict-mode quiet
  // without a JSDoc cast.
  const Ctor = window.AudioContext || window['webkitAudioContext'];
  audioCtx = new Ctor();
}

// Short attack, brief sustain, fast decay. Keeps it crisp and prevents
// click artifacts from raw gate signals.
function envelope(gain, startTime, durSeconds, peak) {
  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(peak, startTime + 0.005);
  gain.gain.setValueAtTime(peak, startTime + durSeconds * 0.6);
  gain.gain.exponentialRampToValueAtTime(0.0001, startTime + durSeconds);
}

function playMelodyNote(freq, startTime, durSeconds) {
  if (freq == null) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'square';
  osc.frequency.value = freq;
  envelope(gain, startTime, durSeconds, 0.05);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(startTime);
  osc.stop(startTime + durSeconds + 0.02);
}

// Triangle wave is warmer at low frequencies than square — keeps the bass
// from competing with the melody for harmonic space.
function playBassNote(freq, startTime, durSeconds) {
  if (freq == null) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'triangle';
  osc.frequency.value = freq;
  envelope(gain, startTime, durSeconds, 0.06);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(startTime);
  osc.stop(startTime + durSeconds + 0.02);
}

// Schedule a sliding 250ms window of upcoming notes (melody + bass independently).
// More reliable than queueing the whole loop, and survives pause/resume cleanly.
function scheduler() {
  while (nextNoteTime < audioCtx.currentTime + 0.25) {
    const [note, dur] = KOROBEINIKI[melodyIndex];
    const seconds = dur * sixteenthSeconds();
    if (note) playMelodyNote(FREQ[note], nextNoteTime, seconds);
    nextNoteTime += seconds;
    melodyIndex = (melodyIndex + 1) % KOROBEINIKI.length;
  }
  while (nextBassTime < audioCtx.currentTime + 0.25) {
    const [note, dur] = KOROBEINIKI_BASS[bassIndex];
    const seconds = dur * sixteenthSeconds();
    if (note) playBassNote(FREQ[note], nextBassTime, seconds);
    nextBassTime += seconds;
    bassIndex = (bassIndex + 1) % KOROBEINIKI_BASS.length;
  }
}

function startMusic() {
  initAudio();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (schedulerInterval) return;
  const start = audioCtx.currentTime + 0.05;
  nextNoteTime = start;
  nextBassTime = start;
  scheduler();
  schedulerInterval = setInterval(scheduler, 50);
}

function stopMusic() {
  if (schedulerInterval) { clearInterval(schedulerInterval); schedulerInterval = null; }
  if (audioCtx && audioCtx.state === 'running') audioCtx.suspend();
}

// iOS Safari runs Web Audio in the "ambient" audio session by default, which
// the hardware silent switch mutes. Playing any HTMLAudioElement flips the
// session to "playback" (volume-only, ignores the silent switch). Run on the
// first sound-on so visitors don't have to know about the iOS quirk.
let silentModeUnlocked = false;
function unlockSilentMode() {
  if (silentModeUnlocked) return;
  silentModeUnlocked = true;
  try {
    const a = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=');
    a.volume = 0;
    const p = a.play();
    if (p && p.catch) p.catch(() => {});
  } catch (_) {}
}

function toggleSound() {
  soundOn = !soundOn;
  soundBtn.setAttribute('aria-pressed', soundOn ? 'true' : 'false');
  soundBtn.textContent = soundOn ? 'sound: on' : 'sound: off';
  if (soundOn) {
    unlockSilentMode();
    if (!paused && !gameOver) startMusic();
  } else {
    stopMusic();
  }
}


/* ── Audio: SFX ──────────────────────────────────────────────────────────── */

// All SFX no-op silently if the user hasn't enabled sound yet.

function sfxLock() {
  if (!audioCtx || !soundOn) return;
  const t = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'square';
  osc.frequency.value = 110;
  envelope(gain, t, 0.06, 0.04);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(t);
  osc.stop(t + 0.08);
}

function sfxLineClear(count) {
  if (!audioCtx || !soundOn) return;
  // Single → 2-note rise; Tetris (4) → 4-note rise. Scale steps in between.
  const steps = [659.25, 880.00, 1318.51, 1760.00].slice(0, Math.min(4, count + 1));
  const peak = count >= 4 ? 0.07 : 0.05;
  const t = audioCtx.currentTime;
  steps.forEach((freq, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'square';
    osc.frequency.value = freq;
    envelope(gain, t + i * 0.05, 0.12, peak);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t + i * 0.05);
    osc.stop(t + i * 0.05 + 0.15);
  });
}

function sfxGameOver() {
  if (!audioCtx || !soundOn) return;
  const t = audioCtx.currentTime;
  [440, 349.23, 277.18, 220].forEach((freq, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'square';
    osc.frequency.value = freq;
    envelope(gain, t + i * 0.15, 0.16, 0.05);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t + i * 0.15);
    osc.stop(t + i * 0.15 + 0.2);
  });
}


/* ── Init ────────────────────────────────────────────────────────────────── */

function bindButton(el, fn) {
  el.addEventListener('click', () => { fn(); el.blur(); });
}

function init() {
  canvas = document.getElementById('stack-canvas');
  ctx = canvas.getContext('2d');
  scoreEl = document.getElementById('score');
  highEl = document.getElementById('high');
  linesEl = document.getElementById('lines');
  levelEl = document.getElementById('level');
  toggleBtn = document.getElementById('toggle');
  restartBtn = document.getElementById('restart');
  soundBtn = document.getElementById('sound');

  loadHighScore();
  reset();
  updateMeta();

  bindButton(toggleBtn, togglePause);
  bindButton(restartBtn, resetAndStart);
  bindButton(soundBtn, toggleSound);
  window.addEventListener('keydown', handleKey);
  window.addEventListener('keyup', handleKeyUp);
  canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
  canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
  canvas.addEventListener('touchend', handleTouchEnd, { passive: false });
  canvas.addEventListener('touchcancel', handleTouchEnd, { passive: false });

  // Pause music + drop held-key state when the tab is hidden so it doesn't keep
  // playing in the background and the game doesn't snap on resume.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopMusic();
      dasDir = 0;
      softDropHeld = false;
    } else {
      lastTime = performance.now();
      if (soundOn && !paused && !gameOver) startMusic();
    }
  });

  lastTime = performance.now();
  requestAnimationFrame(tick);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
