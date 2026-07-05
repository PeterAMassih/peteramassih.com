// src/scripts/gradient.js
// Interactive gradient descent on a Gaussian-mixture loss landscape.
// Loss = -Σᵢ Aᵢ·exp(-[(x-xᵢ)² + (y-yᵢ)²] / 2σᵢ²). Three shallow wells are
// local minima and one deeper well is the global minimum. Plain gradient
// descent often gets stuck in a shallow well from a common start. Momentum
// and Adam tend to escape.

// WELLS holds hand-tuned parameters for this demo, not from any benchmark.
// There is no canonical "Gaussian mixture for gradient descent" test
// function in the literature. Common ones (Rosenbrock, Himmelblau, Beale)
// don't tell the local-vs-global trap story we want.
//
// Design constraints these four numbers satisfy:
//   1. World canvas spans [-5, 5]² (see WORLD constant). Centers at
//      roughly (±2, ±2.5) sit comfortably inside with room for contour
//      rings on every side and no clipping.
//   2. Three shallow wells of equal depth (A = 2.0) plus one clearly
//      deeper well (A = 3.2). The 60% depth gap makes the global minimum
//      visually dominant while keeping local minima deep enough to trap
//      plain gradient descent from a nearby start.
//   3. Slightly wider sigma on the deep well (1.6 vs 1.3) so its basin
//      of attraction is the largest, and most starting points converge there
//      unless dropped close to a shallow well.
//   4. Deep well at (1.5, 1.5), deliberately not in a corner, so the
//      dominant feature sits in the visual center-right of the canvas.
const WELLS = [
  { x: -2.5, y: -2.5, A: 2.0, s: 1.3 },
  { x:  2.5, y: -2.0, A: 2.0, s: 1.3 },
  { x: -2.0, y:  2.5, A: 2.0, s: 1.3 },
  { x:  1.5, y:  1.5, A: 3.2, s: 1.6 }, // deepest well, the global minimum
];
const WORLD = 5;
const F_MIN = -4.5;
const STEP = 1;
const PATH_MAX = 120;
const NUM_CONTOURS = 8;
const FPS = 30;
const COLOR_GAMMA = 0.55;
const CONVERGE_GRAD = 0.001;

/**
 * Loss at a point: the negated sum of all Gaussian wells.
 * @param {number} x World x-coordinate.
 * @param {number} y World y-coordinate.
 * @return {number} Loss value; more negative is deeper.
 */
function f(x, y) {
  let s = 0;
  for (const w of WELLS) {
    const dx = x - w.x, dy = y - w.y;
    s += w.A * Math.exp(-(dx*dx + dy*dy) / (2 * w.s * w.s));
  }
  return -s;
}

/**
 * Analytic gradient of f.
 * @param {number} x World x-coordinate.
 * @param {number} y World y-coordinate.
 * @return {number[]} [df/dx, df/dy].
 */
function grad(x, y) {
  let gx = 0, gy = 0;
  for (const w of WELLS) {
    const dx = x - w.x, dy = y - w.y;
    const e = w.A * Math.exp(-(dx*dx + dy*dy) / (2 * w.s * w.s));
    gx += e * dx / (w.s * w.s);
    gy += e * dy / (w.s * w.s);
  }
  return [gx, gy];
}

/**
 * Descends from a starting point to the nearby local minimum.
 * Well centers are NOT the local minima when wells overlap — the tails of
 * other wells shift each minimum slightly, so the true minima are found
 * numerically once at load time.
 * @param {number} startX Starting x in world coordinates.
 * @param {number} startY Starting y in world coordinates.
 * @return {number[]} [x, y] of the local minimum.
 */
function findMin(startX, startY) {
  let x = startX, y = startY;
  for (let i = 0; i < 500; i++) {
    const [gx, gy] = grad(x, y);
    if (gx * gx + gy * gy < 1e-14) break;
    x -= 0.05 * gx;
    y -= 0.05 * gy;
  }
  return [x, y];
}
const MINIMA = WELLS.map((w) => findMin(w.x, w.y));

const state = {
  x: 0, y: 0,
  vx: 0, vy: 0, mx: 0, my: 0, t: 0,
  step: 0,
  path: [],
  active: false,
  running: false,
};
let lr = 0.1;
let optimizer = 'gd';

/**
 * Restarts the descent from a point, clearing all optimizer state
 * (momentum velocity, Adam moments, step count, path history).
 * @param {number} x Starting x in world coordinates.
 * @param {number} y Starting y in world coordinates.
 */
function reset(x, y) {
  state.x = x; state.y = y;
  state.vx = state.vy = state.mx = state.my = 0;
  state.t = 0;
  state.step = 0;
  state.path = [[x, y]];
  state.active = true;
}

/** One plain gradient descent update: θ ← θ − lr·∇f. */
function stepGd() {
  const [gx, gy] = grad(state.x, state.y);
  state.x -= lr * gx;
  state.y -= lr * gy;
}

/** One momentum update: v ← βv − lr·∇f, θ ← θ + v, with β = 0.9. */
function stepMomentum() {
  const beta = 0.9;
  const [gx, gy] = grad(state.x, state.y);
  state.vx = beta * state.vx - lr * gx;
  state.vy = beta * state.vy - lr * gy;
  state.x += state.vx;
  state.y += state.vy;
}

/** One Adam update with bias correction (β₁ = 0.9, β₂ = 0.999). */
function stepAdam() {
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const [gx, gy] = grad(state.x, state.y);
  state.t++;
  state.mx = b1 * state.mx + (1 - b1) * gx;
  state.my = b1 * state.my + (1 - b1) * gy;
  state.vx = b2 * state.vx + (1 - b2) * gx * gx;
  state.vy = b2 * state.vy + (1 - b2) * gy * gy;
  const bc1 = 1 - Math.pow(b1, state.t);
  const bc2 = 1 - Math.pow(b2, state.t);
  state.x -= lr * (state.mx / bc1) / (Math.sqrt(state.vx / bc2) + eps);
  state.y -= lr * (state.my / bc1) / (Math.sqrt(state.vy / bc2) + eps);
}

/** Applies one update of the selected optimizer and records the path. */
function doStep() {
  if (optimizer === 'gd') stepGd();
  else if (optimizer === 'momentum') stepMomentum();
  else stepAdam();
  state.step++;
  state.path.push([state.x, state.y]);
  if (state.path.length > PATH_MAX) state.path.shift();
}

// Canvas, offscreen heatmap, cached theme colors, cached DOM refs.
let canvas, ctx, heatmapCanvas;
let colors = null;
let els = null;

/**
 * Converts canvas pixel coordinates to world coordinates (y flipped).
 * @param {number} px Pixel x.
 * @param {number} py Pixel y.
 * @return {number[]} [x, y] in world space.
 */
function pixToWorld(px, py) {
  return [
    (px / canvas.clientWidth) * 2 * WORLD - WORLD,
    -((py / canvas.clientHeight) * 2 * WORLD - WORLD),
  ];
}

/**
 * Converts world coordinates to canvas pixel coordinates (y flipped).
 * @param {number} x World x.
 * @param {number} y World y.
 * @return {number[]} [px, py] in CSS pixels.
 */
function worldToPix(x, y) {
  return [
    ((x + WORLD) / (2 * WORLD)) * canvas.clientWidth,
    ((WORLD - y) / (2 * WORLD)) * canvas.clientHeight,
  ];
}

/**
 * Parses a 6-digit hex color into RGB components.
 * @param {string} hex Color like "#1a1a1a".
 * @return {number[]} [r, g, b] in 0–255.
 */
function hexToRgb(hex) {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0,2), 16), parseInt(v.slice(2,4), 16), parseInt(v.slice(4,6), 16)];
}

/** Caches the current theme's colors from CSS custom properties. */
function readColors() {
  const css = getComputedStyle(document.documentElement);
  colors = {
    bg: hexToRgb(css.getPropertyValue('--color-code-bg').trim()),
    fg: hexToRgb(css.getPropertyValue('--color-text').trim()),
    contour: hexToRgb(css.getPropertyValue('--color-text-muted').trim()),
    accent: css.getPropertyValue('--color-accent').trim(),
  };
}

/** Sizes the main canvas backing store to device resolution for a crisp path + ball. */
function fit() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/**
 * Renders the loss heatmap, contour lines, and minima markers to an
 * offscreen CSS-pixel canvas, so each frame can GPU-blit it with drawImage
 * instead of a slow per-frame putImageData on a device-sized canvas.
 */
function renderHeatmap() {
  const { bg, fg, contour } = colors;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  heatmapCanvas = document.createElement('canvas');
  heatmapCanvas.width = w;
  heatmapCanvas.height = h;
  const hctx = heatmapCanvas.getContext('2d');

  const gridW = Math.ceil(w / STEP) + 1;
  const gridH = Math.ceil(h / STEP) + 1;
  const fGrid = new Float32Array(gridW * gridH);
  for (let i = 0; i < gridH; i++)
    for (let j = 0; j < gridW; j++) {
      const [wx, wy] = pixToWorld(j * STEP, i * STEP);
      fGrid[i * gridW + j] = f(wx, wy);
    }
  const lvl = (v) => Math.floor(Math.min(1, Math.max(0, v / F_MIN)) * NUM_CONTOURS);

  const data = hctx.createImageData(w, h);
  for (let py = 0; py < h; py += STEP) {
    for (let px = 0; px < w; px += STEP) {
      const gi = (py / STEP) | 0, gj = (px / STEP) | 0;
      const fv = fGrid[gi * gridW + gj];
      const t = Math.pow(Math.min(1, Math.max(0, fv / F_MIN)), COLOR_GAMMA);
      let r = (bg[0] * (1 - t) + fg[0] * t) | 0;
      let g = (bg[1] * (1 - t) + fg[1] * t) | 0;
      let b = (bg[2] * (1 - t) + fg[2] * t) | 0;
      const my = lvl(fv);
      const right = gj + 1 < gridW ? lvl(fGrid[gi * gridW + gj + 1]) : my;
      const down = gi + 1 < gridH ? lvl(fGrid[(gi + 1) * gridW + gj]) : my;
      if (my !== right || my !== down) { r = contour[0]; g = contour[1]; b = contour[2]; }
      for (let dy = 0; dy < STEP && py + dy < h; dy++)
        for (let dx = 0; dx < STEP && px + dx < w; dx++) {
          const idx = ((py + dy) * w + (px + dx)) * 4;
          data.data[idx] = r; data.data[idx + 1] = g; data.data[idx + 2] = b; data.data[idx + 3] = 255;
        }
    }
  }
  hctx.putImageData(data, 0, 0);

  // Mark each actual local minimum (not the well centers, which differ
  // by a few pixels because of inter-well gradient contributions).
  hctx.strokeStyle = colors.accent;
  hctx.lineWidth = 1;
  for (const m of MINIMA) {
    const [px, py] = worldToPix(m[0], m[1]);
    hctx.beginPath();
    hctx.moveTo(px - 5, py); hctx.lineTo(px + 5, py);
    hctx.moveTo(px, py - 5); hctx.lineTo(px, py + 5);
    hctx.stroke();
  }
}

/**
 * Draws one frame: blits the cached heatmap, then overlays the descent
 * path and ball, and refreshes the step/loss/gradient-norm readouts.
 */
function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  // GPU-accelerated blit. Respects the dpr transform so it fills the whole canvas.
  ctx.drawImage(heatmapCanvas, 0, 0, w, h);

  if (state.path.length > 1) {
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const [px0, py0] = worldToPix(state.path[0][0], state.path[0][1]);
    ctx.moveTo(px0, py0);
    for (let i = 1; i < state.path.length; i++) {
      const [pxi, pyi] = worldToPix(state.path[i][0], state.path[i][1]);
      ctx.lineTo(pxi, pyi);
    }
    ctx.stroke();
  }
  if (state.active) {
    const [px, py] = worldToPix(state.x, state.y);
    ctx.fillStyle = colors.accent;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
  }
  const [gx, gy] = grad(state.x, state.y);
  els.step.textContent = state.step;
  els.loss.textContent = state.active ? f(state.x, state.y).toFixed(3) : '—';
  els.gnorm.textContent = state.active ? Math.sqrt(gx * gx + gy * gy).toFixed(3) : '—';
}

let last = 0;

/**
 * requestAnimationFrame loop, throttled to FPS. Pauses itself once the
 * gradient norm drops below CONVERGE_GRAD so we don't burn cycles
 * redrawing a frozen frame.
 * @param {number} now Timestamp supplied by requestAnimationFrame.
 */
function loop(now) {
  if (state.running && state.active && now - last > 1000 / FPS) {
    doStep();
    draw();
    last = now;
    const [gx, gy] = grad(state.x, state.y);
    if (Math.sqrt(gx*gx + gy*gy) < CONVERGE_GRAD && state.step > 5) {
      state.running = false;
      els.toggle.textContent = 'Play';
    }
  }
  requestAnimationFrame(loop);
}

/** Re-reads theme colors and re-renders everything after a theme change. */
function refreshTheme() { readColors(); renderHeatmap(); draw(); }

/**
 * Finds the DOM, renders the landscape, starts the loop, and wires all
 * controls (reset, play/pause, single-step, learning rate, optimizer,
 * click-to-drop, theme changes). Bails silently on pages without the canvas.
 */
function init() {
  canvas = document.getElementById('grad-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');
  els = {
    step: document.getElementById('step'),
    loss: document.getElementById('loss'),
    gnorm: document.getElementById('gnorm'),
    toggle: document.getElementById('toggle'),
  };
  readColors();
  fit(); renderHeatmap(); draw();
  requestAnimationFrame(loop);

  document.getElementById('reset').addEventListener('click', () => {
    state.active = false; state.running = false; state.path = []; state.step = 0;
    els.toggle.textContent = 'Play';
    draw();
  });
  els.toggle.addEventListener('click', () => {
    if (!state.active) return;
    state.running = !state.running;
    els.toggle.textContent = state.running ? 'Pause' : 'Play';
  });
  document.getElementById('step-btn').addEventListener('click', () => {
    if (!state.active) return;
    doStep(); draw();
  });
  document.querySelectorAll('[data-lr]').forEach((btn) => {
    btn.addEventListener('click', () => {
      lr = Number(btn.dataset.lr);
      document.querySelectorAll('[data-lr]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
  document.querySelectorAll('[data-opt]').forEach((btn) => {
    btn.addEventListener('click', () => {
      optimizer = btn.dataset.opt;
      state.vx = state.vy = state.mx = state.my = 0; state.t = 0;
      document.querySelectorAll('[data-opt]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      // Reveal the matching update rule, hide the others.
      document.querySelectorAll('[data-opt-formula]').forEach((el) => {
        el.hidden = el.dataset.optFormula !== optimizer;
      });
      // Show only the glossary entries that apply to this optimizer.
      document.querySelectorAll('[data-opt-gloss]').forEach((el) => {
        el.hidden = !el.dataset.optGloss.split(' ').includes(optimizer);
      });
    });
  });
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = pixToWorld(e.clientX - rect.left, e.clientY - rect.top);
    reset(wx, wy);
    state.running = true;
    els.toggle.textContent = 'Pause';
    draw();
  });
  window.addEventListener('resize', () => { fit(); renderHeatmap(); draw(); });

  new MutationObserver(refreshTheme).observe(document.documentElement, {
    attributes: true, attributeFilter: ['data-theme'],
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshTheme);
}

init();
