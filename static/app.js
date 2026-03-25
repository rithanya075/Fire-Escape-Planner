/**
 * app.js — AI Fire Escape Planner Frontend
 * =========================================
 * Handles:
 *   - Canvas grid rendering (cell types + risk overlay + A* path animation)
 *   - Interactive grid editor (click/drag to place walls, fire, exits, person)
 *   - API calls to Flask backend (init, step, set_cell, reset)
 *   - Live Chart.js graphs (path cost + fire coverage)
 *   - Auto-play loop for fire spread animation
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let state       = null;   // latest server state JSON
let activeTool  = 0;      // currently selected editor tool type
let isDrawing   = false;  // mouse-down drag state
let autoTimer   = null;   // setInterval handle for auto-play
let isPlaying   = false;
let costChart   = null;
let fireChart   = null;

// ── Cell colours (matching Python constants) ──────────────────────────────────
const CELL_COLORS = {
  0: "#e8e4dc",   // EMPTY
  1: "#1a2332",   // WALL
  2: "#ff4500",   // FIRE
  3: "#00c87a",   // EXIT
  4: "#3d9eff",   // PERSON
  5: "#ff6b35",   // ACTIVE FIRE
};

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById("grid-canvas");
const ctx    = canvas.getContext("2d");

// Animation vars for path pulse
let pathPulse = 0;
let rafHandle = null;

// ── Tool selection ────────────────────────────────────────────────────────────
function setTool(type) {
  activeTool = type;
  document.querySelectorAll(".tool-btn").forEach(b => {
    b.classList.toggle("active", parseInt(b.dataset.type) === type);
  });
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function post(url, body = {}) {
  const res = await fetch(url, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  return res.json();
}

// ── INIT ──────────────────────────────────────────────────────────────────────
async function initSim() {
  const rows = parseInt(document.getElementById("grid-size").value);
  const seed = parseInt(document.getElementById("fire-seed").value) || 42;

  stopAuto();

  const btn = document.getElementById("btn-init");
  btn.innerHTML = '<span class="spinner"></span>TRAINING ML…';
  btn.disabled = true;

  try {
    const data = await post("/api/init", { rows, cols: rows, seed });
    if (!data.ok) { showToast("Init failed", "warn"); return; }

    state = data.state;
    applyState();
    hideOverlay();
    showToast("Ready — ML trained + A* path found", "ok");
    setBadgeOnline(true);
  } catch(e) {
    showToast("Server error — is Flask running?", "warn");
  } finally {
    btn.innerHTML = "▶ INITIALISE";
    btn.disabled = false;
  }
}

// ── STEP ──────────────────────────────────────────────────────────────────────
async function stepOnce() {
  if (!state) return;
  const data = await post("/api/step");
  if (!data.ok) { showToast("Step failed", "warn"); return; }
  state = data.state;
  applyState();

  if (!state.success) {
    showToast("⚠ No path found — fire blocked all exits!", "warn");
    stopAuto();
  }
}

// ── AUTO PLAY ─────────────────────────────────────────────────────────────────
function toggleAuto() {
  isPlaying ? stopAuto() : startAuto();
}

function startAuto() {
  if (isPlaying) return;
  isPlaying = true;
  const btn = document.getElementById("btn-auto");
  btn.textContent = "⏸ PAUSE";
  btn.classList.add("playing");

  const ms = parseInt(document.getElementById("speed").value);
  autoTimer = setInterval(async () => {
    await stepOnce();
    if (!state || !state.success) stopAuto();
  }, ms);
}

function stopAuto() {
  if (!isPlaying) return;
  isPlaying = false;
  clearInterval(autoTimer);
  autoTimer = null;
  const btn = document.getElementById("btn-auto");
  if (btn) { btn.textContent = "▶ AUTO PLAY"; btn.classList.remove("playing"); }
}

// ── RESET ─────────────────────────────────────────────────────────────────────
async function resetSim() {
  stopAuto();
  await post("/api/reset");
  state = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  showOverlay();
  setBadgeOnline(false);
  updateStats(null);
  resetCharts();
  showToast("Reset — click INITIALISE", "");
}

// ── APPLY STATE (render everything) ──────────────────────────────────────────
function applyState() {
  if (!state) return;
  updateStats(state);
  renderGrid(state);
  updateCharts(state);
  document.getElementById("btn-step").disabled = false;
  document.getElementById("btn-auto").disabled = false;
}

// ── GRID RENDERER ─────────────────────────────────────────────────────────────
function renderGrid(s) {
  if (!s || !s.cells) return;

  const rows = s.rows;
  const cols = s.cols;

  // Size canvas to fit the centre panel
  const wrap   = canvas.parentElement;
  const wrapW  = wrap.clientWidth  - 0;
  const wrapH  = wrap.clientHeight - 36; // header height

  const cellSize = Math.floor(Math.min(wrapW / cols, wrapH / rows));
  const totalW   = cellSize * cols;
  const totalH   = cellSize * rows;

  canvas.width  = totalW;
  canvas.height = totalH;

  const pathSet = new Set((s.path || []).map(p => `${p[0]},${p[1]}`));

  // ── Draw cells ──
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ctype = s.cells[r][c];
      const risk  = s.risk[r][c];
      const x = c * cellSize;
      const y = r * cellSize;

      // Base cell colour
      ctx.fillStyle = CELL_COLORS[ctype] || "#e8e4dc";
      ctx.fillRect(x, y, cellSize, cellSize);

      // Risk overlay (red tint for non-fire, non-wall cells)
      if (ctype === 0 && risk > 0.05) {
        ctx.fillStyle = `rgba(180,30,0,${risk * 0.55})`;
        ctx.fillRect(x, y, cellSize, cellSize);
      }

      // Fire shimmer animation
      if (ctype === 2 || ctype === 5) {
        const flicker = 0.3 + 0.25 * Math.sin(pathPulse * 0.08 + r * 1.3 + c * 0.9);
        ctx.fillStyle = `rgba(255,200,0,${flicker})`;
        ctx.fillRect(x, y, cellSize, cellSize);
      }

      // Cell border
      ctx.strokeStyle = "rgba(0,0,0,0.18)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
    }
  }

  // ── Draw ALL exit paths (dimmed) ──
  if (s.all_paths && s.all_paths.length > 1) {
    const altColors = ["#3d9eff", "#00c87a", "#a371f7", "#d29922"];
    s.all_paths.forEach((ap, idx) => {
      if (ap.is_best) return; // skip best — drawn separately below
      if (!ap.path || ap.path.length < 2) return;
      const col = altColors[idx % altColors.length];
      ctx.strokeStyle = col + "55"; // transparent
      ctx.lineWidth   = Math.max(1.5, cellSize * 0.15);
      ctx.lineCap     = "round";
      ctx.lineJoin    = "round";
      ctx.setLineDash([cellSize * 0.4, cellSize * 0.3]);
      ctx.beginPath();
      const [r0,c0] = ap.path[0];
      ctx.moveTo(c0*cellSize+cellSize/2, r0*cellSize+cellSize/2);
      ap.path.slice(1).forEach(([r,c]) =>
        ctx.lineTo(c*cellSize+cellSize/2, r*cellSize+cellSize/2));
      ctx.stroke();
      ctx.setLineDash([]);
      // Small exit marker
      const [er,ec] = ap.path[ap.path.length-1];
      ctx.fillStyle = col + "99";
      ctx.beginPath();
      ctx.arc(ec*cellSize+cellSize/2, er*cellSize+cellSize/2, cellSize*0.22, 0, Math.PI*2);
      ctx.fill();
      // Cost label
      if (cellSize > 14) {
        ctx.fillStyle = col + "cc";
        ctx.font = `bold ${Math.max(8, cellSize*0.35)}px Courier New`;
        ctx.textAlign = "center";
        ctx.fillText(ap.cost.toFixed(1), ec*cellSize+cellSize/2, er*cellSize+cellSize/2 - cellSize*0.6);
      }
    });
    ctx.textAlign = "left";
  }

  // ── Draw BEST A* path (gold, pulsing) ──
  if (s.path && s.path.length > 1) {
    const pulse = 0.65 + 0.35 * Math.sin(pathPulse * 0.06);

    ctx.strokeStyle = `rgba(255,215,0,${pulse})`;
    ctx.lineWidth   = Math.max(2, cellSize * 0.28);
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
    ctx.shadowColor = "#ffd700";
    ctx.shadowBlur  = cellSize * 0.6;

    ctx.beginPath();
    const [pr0, pc0] = s.path[0];
    ctx.moveTo(pc0 * cellSize + cellSize / 2, pr0 * cellSize + cellSize / 2);
    for (let i = 1; i < s.path.length; i++) {
      const [pr, pc] = s.path[i];
      ctx.lineTo(pc * cellSize + cellSize / 2, pr * cellSize + cellSize / 2);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Start dot (blue circle)
    const [sr, sc] = s.path[0];
    ctx.fillStyle = "#3d9eff";
    ctx.beginPath();
    ctx.arc(sc * cellSize + cellSize/2, sr * cellSize + cellSize/2, cellSize*0.32, 0, Math.PI*2);
    ctx.fill();

    // Goal star
    const [gr, gc] = s.path[s.path.length - 1];
    drawStar(ctx,
      gc * cellSize + cellSize/2,
      gr * cellSize + cellSize/2,
      cellSize * 0.35, 5, "#ffd700"
    );
  }

  pathPulse++;
}

function drawStar(ctx, cx, cy, r, pts, color) {
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  for (let i = 0; i < pts * 2; i++) {
    const angle  = (i * Math.PI) / pts - Math.PI / 2;
    const radius = i % 2 === 0 ? r : r * 0.42;
    i === 0
      ? ctx.moveTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius)
      : ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
  }
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
}

// ── Animation loop (path pulse) ───────────────────────────────────────────────
function startRenderLoop() {
  function loop() {
    if (state) renderGrid(state);
    rafHandle = requestAnimationFrame(loop);
  }
  loop();
}

// ── INTERACTIVE EDITOR ────────────────────────────────────────────────────────
function canvasCell(e) {
  if (!state) return null;
  const rect     = canvas.getBoundingClientRect();
  const cellSize = canvas.width / state.cols;
  const c = Math.floor((e.clientX - rect.left)  / cellSize);
  const r = Math.floor((e.clientY - rect.top)   / cellSize);
  if (r < 0 || r >= state.rows || c < 0 || c >= state.cols) return null;
  return { r, c };
}

async function handleCellEdit(e) {
  const pos = canvasCell(e);
  if (!pos) return;
  const data = await post("/api/set_cell", { row: pos.r, col: pos.c, type: activeTool });
  if (data.ok) {
    state = data.state;
    updateStats(state);
    updateCharts(state);
  }
}

canvas.addEventListener("mousedown", e => {
  if (!state) return;
  isDrawing = true;
  handleCellEdit(e);
});
canvas.addEventListener("mousemove", e => { if (isDrawing) handleCellEdit(e); });
canvas.addEventListener("mouseup",   () => { isDrawing = false; });
canvas.addEventListener("mouseleave",() => { isDrawing = false; });

// Touch support
canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  isDrawing = true;
  handleCellEdit(e.touches[0]);
}, { passive: false });
canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  if (isDrawing) handleCellEdit(e.touches[0]);
}, { passive: false });
canvas.addEventListener("touchend", () => { isDrawing = false; });

// ── STATS UPDATE ──────────────────────────────────────────────────────────────
function updateStats(s) {
  const set = (id, val) => { document.getElementById(id).textContent = val; };
  if (!s) {
    ["stat-step","stat-cost","stat-len","stat-goal","stat-ms","stat-nodes","stat-fire","stat-cov"]
      .forEach(id => set(id, "—"));
    return;
  }
  set("stat-step",  s.step);
  set("stat-cost",  s.success ? s.path_cost.toFixed(2) : "NO PATH");
  set("stat-len",   s.success ? `${s.path_len} cells` : "—");
  set("stat-goal",  s.success ? `(${s.goal[0]}, ${s.goal[1]})` : "—");
  const reachable = s.all_paths ? s.all_paths.length : (s.success ? 1 : 0);
  set("stat-exits", s.success ? `${reachable} / ${(s.exits||[]).length}` : "0 / 4");
  set("stat-ms",    s.astar_ms.toFixed(2) + " ms");
  set("stat-nodes", s.nodes_exp);
  set("stat-fire",  s.fire_count);
  set("stat-cov",   s.coverage_pct + "%");
}

// ── CHART.JS CHARTS ───────────────────────────────────────────────────────────
const CHART_OPTS = (label, color) => ({
  type: "line",
  data: {
    labels:   [],
    datasets: [{
      label,
      data:            [],
      borderColor:     color,
      backgroundColor: color + "22",
      borderWidth:     2,
      pointRadius:     3,
      pointBackgroundColor: color,
      fill:            true,
      tension:         0.35,
    }],
  },
  options: {
    responsive: true,
    animation:  { duration: 200 },
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: "#6e7681", font: { family: "'Share Tech Mono'", size: 9 } },
        grid:  { color: "#1c2433" },
      },
      y: {
        ticks: { color: "#6e7681", font: { family: "'Share Tech Mono'", size: 9 } },
        grid:  { color: "#1c2433" },
      },
    },
  },
});

function initCharts() {
  Chart.defaults.color = "#8b949e";
  Chart.defaults.borderColor = "#30363d";

  if (costChart) costChart.destroy();
  if (fireChart) fireChart.destroy();

  costChart = new Chart(
    document.getElementById("chart-cost"),
    CHART_OPTS("Path Cost", "#3d9eff")
  );
  fireChart = new Chart(
    document.getElementById("chart-fire"),
    CHART_OPTS("Fire %", "#ff4500")
  );
}

function updateCharts(s) {
  if (!costChart || !fireChart) return;

  const labels = s.history.map((_, i) => `S${i}`);

  costChart.data.labels            = labels;
  costChart.data.datasets[0].data  = s.history.map(v => v ?? null);
  costChart.update("none");

  fireChart.data.labels            = labels;
  fireChart.data.datasets[0].data  = s.fire_history;
  fireChart.update("none");
}

function resetCharts() {
  if (!costChart || !fireChart) return;
  costChart.data.labels = [];
  costChart.data.datasets[0].data = [];
  costChart.update();
  fireChart.data.labels = [];
  fireChart.data.datasets[0].data = [];
  fireChart.update();
}

// ── OVERLAY / TOAST / BADGE ───────────────────────────────────────────────────
function hideOverlay() {
  document.getElementById("grid-overlay").classList.add("hidden");
}
function showOverlay() {
  document.getElementById("grid-overlay").classList.remove("hidden");
}

let toastTimer = null;
function showToast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show " + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2800);
}

function setBadgeOnline(on) {
  const el = document.getElementById("status-badge");
  el.textContent = on ? "● LIVE" : "● OFFLINE";
  el.classList.toggle("online", on);
}

// ── CANVAS RESIZE ─────────────────────────────────────────────────────────────
window.addEventListener("resize", () => { if (state) renderGrid(state); });

// ── BOOT ──────────────────────────────────────────────────────────────────────
initCharts();
startRenderLoop();
