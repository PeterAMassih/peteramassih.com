

// src/scripts/life.js
// Conway's Game of Life — state, rules, rendering, UI bindings.
// Toroidal grid (edges wrap). Auto-bootstraps when imported.

const PATTERNS = {
  glider: [[0,1,0],[0,0,1],[1,1,1]],
  blinker: [[1,1,1]],
  beacon: [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]],
  pulsar: [
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,0,1,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,1,1,1,0,0],
  ],
  gun: [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  ],
};

const CELL = 12;
let canvas, ctx, cols, rows, grid;
let running = true, generation = 0, fps = 10, placingPattern = null, last = 0;
let genEl, popEl, toggleBtn;

const empty = () => Array.from({ length: rows }, () => new Uint8Array(cols));

function fit() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cols = Math.floor(w / CELL);
  rows = Math.floor(h / CELL);
}

function randomize() {
  grid = empty();
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (Math.random() < 0.28) grid[r][c] = 1;
  generation = 0;
}

function step() {
  const next = empty();
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let n = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          n += grid[(r + dr + rows) % rows][(c + dc + cols) % cols];
        }
      next[r][c] = grid[r][c] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
    }
  }
  grid = next;
  generation++;
}

function place(pattern, startR, startC) {
  for (let r = 0; r < pattern.length; r++)
    for (let c = 0; c < pattern[r].length; c++)
      if (pattern[r][c]) grid[(startR + r) % rows][(startC + c) % cols] = 1;
}

function draw() {
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue('--color-code-bg').trim();
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = css.getPropertyValue('--color-text').trim();
  let pop = 0;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (grid[r][c]) { ctx.fillRect(c * CELL, r * CELL, CELL - 1, CELL - 1); pop++; }
  if (genEl) genEl.textContent = generation;
  if (popEl) popEl.textContent = pop;
}

function loop(now) {
  if (running && now - last > 1000 / fps) { step(); draw(); last = now; }
  requestAnimationFrame(loop);
}

function clearPatternSelection() {
  document.querySelectorAll('[data-pattern]').forEach((b) => b.classList.remove('active'));
  placingPattern = null;
}

function init() {
  canvas = document.getElementById('life-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');
  genEl = document.getElementById('gen');
  popEl = document.getElementById('pop');
  toggleBtn = document.getElementById('toggle');

  fit(); randomize(); draw();
  requestAnimationFrame(loop);

  toggleBtn.addEventListener('click', () => {
    running = !running;
    toggleBtn.textContent = running ? 'Pause' : 'Play';
  });
  document.getElementById('step').addEventListener('click', () => { step(); draw(); });
  document.getElementById('random').addEventListener('click', () => { randomize(); draw(); });
  document.getElementById('clear').addEventListener('click', () => { grid = empty(); generation = 0; draw(); });

  document.querySelectorAll('[data-fps]').forEach((btn) => {
    btn.addEventListener('click', () => {
      fps = Number(btn.dataset.fps);
      document.querySelectorAll('[data-fps]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  document.querySelectorAll('[data-pattern]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.pattern;
      if (placingPattern === name) { clearPatternSelection(); return; }
      clearPatternSelection();
      placingPattern = name;
      btn.classList.add('active');
    });
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / CELL);
    const r = Math.floor((e.clientY - rect.top) / CELL);
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (placingPattern) {
      place(PATTERNS[placingPattern], r, c);
      clearPatternSelection();
    } else {
      grid[r][c] = grid[r][c] ? 0 : 1;
    }
    draw();
  });

  window.addEventListener('resize', () => { fit(); draw(); });
}

init();
