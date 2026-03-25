"""
visualization.py — Matplotlib Grid Visualizer
===============================================
Renders the building grid, fire zones, ML risk heatmap, A* paths,
person positions, and exits in a clear, colour-coded matplotlib figure.

Colour scheme
-------------
    White (#FFFFFF)   Empty walkable cell
    #222222           Wall / obstacle
    #FF4500 (red-orange) Fire cell (FIRE or ACTIVE_FIRE)
    #27ae60 (green)   Exit cell
    #3498db (blue)    Person (start)
    Viridis gradient  Risk overlay (0=purple=safe → yellow=danger)
    #FFD700 (gold)    A* path
    ★ star marker     Exit reached by path
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

from grid import Grid, EMPTY, WALL, FIRE, EXIT, PERSON, ACTIVE_FIRE
from astar import AStarResult

# ── Colour map for cell types (base layer) ───────────────────────────────────
_CELL_COLORS = {
    EMPTY:       "#F5F5F5",
    WALL:        "#2C3E50",
    FIRE:        "#FF4500",
    ACTIVE_FIRE: "#FF6B35",
    EXIT:        "#27AE60",
    PERSON:      "#3498DB",
}


def _cell_color(cell_type: int) -> str:
    return _CELL_COLORS.get(cell_type, "#F5F5F5")


# ── Main render function ─────────────────────────────────────────────────────

def render_frame(
    grid: Grid,
    result: AStarResult | None,
    title: str = "AI Fire Escape Planner",
    show_risk: bool = True,
    save_path: str | None = None,
    persons: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """
    Render one frame of the simulation.

    Parameters
    ----------
    grid      : Current Grid state.
    result    : A* result (path, goal). Pass None if no path yet.
    title     : Figure title string.
    show_risk : Overlay ML risk heatmap if True.
    save_path : If given, save figure to this PNG path instead of showing.
    persons   : List of additional person positions (multi-person mode).

    Returns
    -------
    fig : matplotlib Figure object.
    """
    rows, cols = grid.rows, grid.cols

    fig, axes = plt.subplots(1, 2 if show_risk else 1,
                             figsize=(14 if show_risk else 8, 8))
    if not show_risk:
        axes = [axes]

    # ── Left panel: cell-type grid ──
    ax_grid = axes[0]
    _draw_cell_grid(ax_grid, grid, result, persons)
    ax_grid.set_title("Building Layout & Path", fontsize=13, fontweight="bold")

    # ── Right panel: ML risk heatmap ──
    if show_risk:
        ax_risk = axes[1]
        _draw_risk_heatmap(ax_risk, grid, result)
        ax_risk.set_title("ML Risk Score Heatmap", fontsize=13, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#2C3E50")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[Viz] Saved → {save_path}")

    return fig


def _draw_cell_grid(ax, grid: Grid, result, persons):
    rows, cols = grid.rows, grid.cols

    # ── Background colour matrix ──
    rgb = np.ones((rows, cols, 3))
    color_map = {
        "#F5F5F5": [0.96, 0.96, 0.96],
        "#2C3E50": [0.17, 0.24, 0.31],
        "#FF4500": [1.00, 0.27, 0.00],
        "#FF6B35": [1.00, 0.42, 0.21],
        "#27AE60": [0.15, 0.68, 0.38],
        "#3498DB": [0.20, 0.60, 0.86],
    }
    for r in range(rows):
        for c in range(cols):
            hex_c = _cell_color(grid.cells[r, c])
            rgb[r, c] = color_map.get(hex_c, [0.96, 0.96, 0.96])

    ax.imshow(rgb, origin="upper", aspect="equal")

    # ── Risk score overlay (transparency gradient) ──
    risk = grid.risk_scores
    risk_rgba = np.zeros((rows, cols, 4))
    for r in range(rows):
        for c in range(cols):
            if grid.cells[r, c] not in (WALL, FIRE, ACTIVE_FIRE, EXIT):
                sc = risk[r, c]
                risk_rgba[r, c] = [sc, 0.0, 0.0, sc * 0.45]   # red overlay
    ax.imshow(risk_rgba, origin="upper", aspect="equal")

    # ── A* path overlay ──
    if result and result.success and result.path:
        path_rows = [p[0] for p in result.path]
        path_cols = [p[1] for p in result.path]
        ax.plot(path_cols, path_rows, color="#FFD700", linewidth=2.5,
                zorder=5, label="Safe Path")
        # Mark start
        ax.plot(path_cols[0], path_rows[0], "o", color="#3498DB",
                markersize=10, zorder=6)
        # Mark goal
        ax.plot(path_cols[-1], path_rows[-1], "*", color="#F39C12",
                markersize=14, zorder=6)

    # ── Extra person markers ──
    if persons:
        for pr, pc in persons:
            ax.plot(pc, pr, "D", color="#9B59B6", markersize=9, zorder=7)

    # ── Grid lines ──
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#CCCCCC", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # ── Legend ──
    legend_items = [
        mpatches.Patch(color=[0.96, 0.96, 0.96], label="Empty"),
        mpatches.Patch(color=[0.17, 0.24, 0.31], label="Wall"),
        mpatches.Patch(color=[1.00, 0.27, 0.00], label="Fire"),
        mpatches.Patch(color=[0.15, 0.68, 0.38], label="Exit"),
        mpatches.Patch(color=[0.20, 0.60, 0.86], label="Person"),
        mpatches.Patch(color=[1.00, 0.85, 0.00], label="Path"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              framealpha=0.85)


def _draw_risk_heatmap(ax, grid: Grid, result):
    rows, cols = grid.rows, grid.cols
    risk = grid.risk_scores.copy()

    # Mask walls as NaN so they appear grey
    masked = np.where(grid.cells == WALL, np.nan, risk)

    im = ax.imshow(masked, origin="upper", aspect="equal",
                   cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Risk Score")

    # Draw path on risk map too
    if result and result.success and result.path:
        path_rows = [p[0] for p in result.path]
        path_cols = [p[1] for p in result.path]
        ax.plot(path_cols, path_rows, color="white", linewidth=2.5, zorder=5)
        ax.plot(path_cols[-1], path_rows[-1], "*", color="white",
                markersize=14, zorder=6)

    # Overlay wall cells as dark rectangles
    for r in range(rows):
        for c in range(cols):
            if grid.cells[r, c] == WALL:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#2C3E50", edgecolor="none", zorder=3
                ))

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.2)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)


# ── Multi-step animation saver ────────────────────────────────────────────────

def save_simulation_frames(
    frames: list[dict],
    output_dir: str = "output/frames",
) -> list[str]:
    """
    Save a list of frame-dicts to PNG files.

    Each frame-dict must contain:
        "grid"    : Grid
        "result"  : AStarResult
        "step"    : int
        "persons" : list[tuple] (optional)

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for frame in frames:
        step = frame["step"]
        path = os.path.join(output_dir, f"frame_{step:03d}.png")
        render_frame(
            grid=frame["grid"],
            result=frame["result"],
            title=f"Fire Escape Simulation — Step {step}",
            save_path=path,
            persons=frame.get("persons"),
        )
        paths.append(path)
        plt.close("all")
    return paths


# ── Summary statistics plot ───────────────────────────────────────────────────

def plot_summary(stats: list[dict], save_path: str = "output/summary.png") -> None:
    """
    Plot path cost and fire coverage over simulation steps.

    stats : list of {"step": int, "cost": float, "fire_coverage": float,
                      "path_len": int}
    """
    steps = [s["step"] for s in stats]
    costs = [s["cost"] for s in stats]
    coverage = [s["fire_coverage"] * 100 for s in stats]
    lengths = [s["path_len"] for s in stats]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle("Simulation Statistics", fontsize=14, fontweight="bold")

    ax1.plot(steps, costs, marker="o", color="#E74C3C", linewidth=2)
    ax1.set_ylabel("Path Cost (A*)")
    ax1.set_title("Path Cost over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, coverage, marker="s", color="#E67E22", linewidth=2)
    ax2.set_ylabel("Fire Coverage (%)")
    ax2.set_title("Fire Spread Progress")
    ax2.grid(True, alpha=0.3)

    ax3.plot(steps, lengths, marker="^", color="#2980B9", linewidth=2)
    ax3.set_xlabel("Simulation Step")
    ax3.set_ylabel("Path Length (cells)")
    ax3.set_title("Path Length over Time")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[Viz] Summary chart saved → {save_path}")
    plt.close(fig)
