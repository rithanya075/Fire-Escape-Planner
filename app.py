"""
app.py — Flask Web Server for AI Fire Escape Planner
=====================================================
Exposes REST API endpoints consumed by the browser frontend.

Routes
------
GET  /                  → Serve the main HTML page
POST /api/init          → Initialise grid + train ML model
POST /api/step          → Advance fire one step + replan A*
POST /api/reset         → Reset to fresh grid
POST /api/set_cell      → Interactive editor: change a cell type
GET  /api/state         → Return current full grid state as JSON
"""

from flask import Flask, jsonify, request, render_template
import numpy as np
import sys, os, threading

# ── Add project root to path so we can import existing modules ──
sys.path.insert(0, os.path.dirname(__file__))

from grid import Grid, EMPTY, WALL, FIRE, EXIT, PERSON, ACTIVE_FIRE
from fire_simulation import FireSimulation
from ml_model import FireRiskModel
from astar import astar

app = Flask(__name__)

# ── Global simulation state (single session) ─────────────────────────────────
_state = {
    "grid": None,
    "fire_sim": None,
    "model": None,
    "result": None,
    "step": 0,
    "ready": False,
    "history": [],        # list of path costs per step
    "fire_history": [],   # fire coverage per step
}
_lock = threading.Lock()


# ── Helper: serialise grid to JSON-friendly dict ─────────────────────────────

def _serialise_state():
    """Convert current simulation state to a JSON-serialisable dict."""
    g = _state["grid"]
    result = _state["result"]
    fire_sim = _state["fire_sim"]

    if g is None:
        return {"ready": False}

    rows, cols = g.rows, g.cols

    # Cell types as flat list
    cells = g.cells.tolist()

    # Risk scores rounded to 2dp
    risk = np.round(g.risk_scores, 2).tolist()

    # A* path as list of [r,c]
    path = [list(p) for p in result.path] if (result and result.success) else []
    path_cost = round(result.total_cost, 2) if (result and result.success) else 0
    path_len  = len(path)
    goal      = list(result.goal) if (result and result.success) else []
    astar_ms  = round(result.elapsed_ms, 2) if result else 0
    nodes_exp = result.nodes_expanded if result else 0
    success   = result.success if result else False
    # All exit paths for multi-exit display
    all_paths = []
    if result and result.success and hasattr(result, "all_paths"):
        for cost, p, ex in result.all_paths:
            all_paths.append({
                "cost": round(cost, 2),
                "path": [list(c) for c in p],
                "exit": list(ex),
                "is_best": (ex == result.goal)
            })

    fire_count   = len(g.fire_cells)
    coverage_pct = round(fire_sim.fire_coverage() * 100, 1) if fire_sim else 0

    return {
        "ready":        True,
        "step":         _state["step"],
        "rows":         rows,
        "cols":         cols,
        "cells":        cells,
        "risk":         risk,
        "path":         path,
        "path_cost":    path_cost,
        "path_len":     path_len,
        "goal":         goal,
        "astar_ms":     astar_ms,
        "nodes_exp":    nodes_exp,
        "success":      success,
        "fire_count":   fire_count,
        "coverage_pct": coverage_pct,
        "exits":        [list(e) for e in g.exits],
        "start":        list(g.start),
        "history":      _state["history"],
        "fire_history": _state["fire_history"],
        "all_paths":    all_paths,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def api_init():
    """
    Initialise (or reinitialise) the simulation.
    Body (JSON, all optional):
        rows    : int (default 20)
        cols    : int (default 20)
        seed    : int (default 42)
    """
    body = request.get_json(silent=True) or {}
    rows = int(body.get("rows", 20))
    cols = int(body.get("cols", 20))
    seed = int(body.get("seed", 42))

    with _lock:
        # Build grid
        grid = Grid.create_default(rows=rows, cols=cols)

        # Train model (fast — ~500 ms)
        model = FireRiskModel(seed=seed)
        model.train(verbose=False)

        # Fire simulation
        fire_sim = FireSimulation(grid, seed=seed)

        # Initial ML prediction
        risk_grid = model.predict_risk_scores(fire_sim)
        grid.update_risk_scores(risk_grid)

        # Initial A*
        result = astar(grid)

        _state.update({
            "grid":         grid,
            "fire_sim":     fire_sim,
            "model":        model,
            "result":       result,
            "step":         0,
            "ready":        True,
            "history":      [round(result.total_cost, 2)] if result.success else [0],
            "fire_history": [0.0],
        })

    return jsonify({"ok": True, "state": _serialise_state()})


@app.route("/api/step", methods=["POST"])
def api_step():
    """Advance fire simulation by one step and replan A*."""
    with _lock:
        if not _state["ready"]:
            return jsonify({"ok": False, "error": "Not initialised"}), 400

        grid     = _state["grid"]
        fire_sim = _state["fire_sim"]
        model    = _state["model"]

        # 1. Spread fire
        fire_sim.step()

        # 2. Recompute ML risk scores
        risk_grid = model.predict_risk_scores(fire_sim)
        grid.update_risk_scores(risk_grid)

        # 3. Replan A*
        result = astar(grid)

        _state["result"] = result
        _state["step"]  += 1

        cost = round(result.total_cost, 2) if result.success else None
        _state["history"].append(cost)
        _state["fire_history"].append(round(fire_sim.fire_coverage() * 100, 1))

    return jsonify({"ok": True, "state": _serialise_state()})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset to a fresh grid (re-uses last init params via re-init)."""
    with _lock:
        _state["ready"] = False
        _state["grid"]  = None

    return jsonify({"ok": True})


@app.route("/api/set_cell", methods=["POST"])
def api_set_cell():
    """
    Interactive editor: change a single cell's type.
    Body: { row: int, col: int, type: int }
    Types: 0=Empty, 1=Wall, 2=Fire, 3=Exit, 4=Person
    """
    body = request.get_json(silent=True) or {}
    r    = int(body.get("row", -1))
    c    = int(body.get("col", -1))
    t    = int(body.get("type", 0))

    with _lock:
        if not _state["ready"]:
            return jsonify({"ok": False, "error": "Not initialised"}), 400

        grid     = _state["grid"]
        fire_sim = _state["fire_sim"]
        model    = _state["model"]

        if not grid.in_bounds(r, c):
            return jsonify({"ok": False, "error": "Out of bounds"}), 400

        # Don't allow editing outer border (keep it walls)
        if r == 0 or r == grid.rows-1 or c == 0 or c == grid.cols-1:
            if t not in (EXIT,):
                return jsonify({"ok": False, "error": "Border cell"}), 400

        old_type = grid.cells[r, c]

        # Update fire tracking set
        if old_type in (FIRE, ACTIVE_FIRE):
            grid.fire_cells.discard((r, c))
        if t in (FIRE, ACTIVE_FIRE):
            grid.fire_cells.add((r, c))

        # Update person start if placing a person
        if t == PERSON:
            # Clear old person cell
            old_r, old_c = grid.start
            if grid.cells[old_r, old_c] == PERSON:
                grid.cells[old_r, old_c] = EMPTY
            grid.start = (r, c)

        # Update exits list
        if old_type == EXIT and (r, c) in grid.exits:
            grid.exits.remove((r, c))
        if t == EXIT:
            grid.exits.append((r, c))

        grid.cells[r, c] = t

        # Recompute ML + A* after edit
        risk_grid = model.predict_risk_scores(fire_sim)
        grid.update_risk_scores(risk_grid)
        result = astar(grid)
        _state["result"] = result

    return jsonify({"ok": True, "state": _serialise_state()})


@app.route("/api/state", methods=["GET"])
def api_state():
    with _lock:
        return jsonify(_serialise_state())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔥 AI Fire Escape Planner — Flask Web App")
    print("   Open your browser at: http://127.0.0.1:5000\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
