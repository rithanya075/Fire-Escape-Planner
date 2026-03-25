"""
grid.py — Building Grid Representation
=======================================
Defines the 2D grid environment for the fire escape simulation.

Cell Types:
    0 = Empty walkable cell
    1 = Wall / obstacle
    2 = Fire source (initial fire)
    3 = Exit point
    4 = Person (start position)
    5 = Active fire (spread fire)

Each cell also carries a 'risk_score' (0.0–1.0) injected by the ML model,
which A* uses as an additional cost weight.
"""

import numpy as np

# ── Cell-type constants ──────────────────────────────────────────────────────
EMPTY = 0
WALL = 1
FIRE = 2
EXIT = 3
PERSON = 4
ACTIVE_FIRE = 5


class Grid:
    """
    Manages the building layout, cell types, and per-cell risk scores.

    Attributes
    ----------
    rows, cols : int
        Dimensions of the grid.
    cells : np.ndarray[int], shape (rows, cols)
        Cell-type values (EMPTY, WALL, FIRE, EXIT, PERSON, ACTIVE_FIRE).
    risk_scores : np.ndarray[float], shape (rows, cols)
        ML-predicted risk score for each cell (0.0 = safe, 1.0 = max risk).
    start : tuple[int, int]
        (row, col) of the person's starting position.
    exits : list[tuple[int, int]]
        All exit positions on the grid.
    fire_cells : set[tuple[int, int]]
        All cells currently on fire.
    """

    def __init__(self, rows: int = 20, cols: int = 20):
        self.rows = rows
        self.cols = cols
        self.cells = np.zeros((rows, cols), dtype=int)
        self.risk_scores = np.zeros((rows, cols), dtype=float)
        self.start: tuple[int, int] = (0, 0)
        self.exits: list[tuple[int, int]] = []
        self.fire_cells: set[tuple[int, int]] = set()

    # ── Factory: default office-like building layout ─────────────────────────

    @classmethod
    def create_default(cls, rows: int = 20, cols: int = 20) -> "Grid":
        """
        Build a realistic office-style grid with corridors, rooms, multiple
        exits, a person, and initial fire sources.
        Scales to any grid size ≥ 10×10.
        """
        g = cls(rows, cols)

        # ── Outer walls ──
        g.cells[0, :] = WALL
        g.cells[-1, :] = WALL
        g.cells[:, 0] = WALL
        g.cells[:, -1] = WALL

        # ── Scale divider positions to grid size ──
        r1 = max(3, rows // 3)          # first horizontal divider row
        r2 = max(6, 2 * rows // 3)      # second horizontal divider row
        c1 = max(3, cols // 3)          # first vertical divider col
        c2 = max(6, 2 * cols // 3)      # second vertical divider col

        # Gap positions (doorways)
        gap_r1 = r1 // 2                # doorway in first horizontal wall
        gap_r2 = (r1 + r2) // 2        # doorway in second horizontal wall
        gap_c1 = c1 // 2               # doorway in first vertical wall
        gap_c2 = (c1 + c2) // 2        # doorway in second vertical wall

        # Horizontal divider 1 (with two doorways)
        for c in range(1, cols - 1):
            if c not in (gap_r1, c2):
                g.cells[r1, c] = WALL

        # Horizontal divider 2 (with two doorways)
        for c in range(1, cols - 1):
            if c not in (gap_r2, c1):
                g.cells[r2, c] = WALL

        # Vertical divider 1 (rows 1 to r1, doorway at middle)
        mid1 = max(1, r1 // 2)
        for r in range(1, r1):
            if r != mid1:
                g.cells[r, c1] = WALL

        # Vertical divider 2 (rows r1+1 to r2, doorway at middle)
        mid2 = r1 + max(1, (r2 - r1) // 2)
        for r in range(r1 + 1, r2):
            if r != mid2:
                g.cells[r, c2] = WALL

        # ── Exits on outer walls ──
        exit_col = cols // 2
        exit_row = rows // 2
        exits = [(0, exit_col), (rows - 1, exit_col),
                 (exit_row, 0), (exit_row, cols - 1)]
        for r, c in exits:
            g.cells[r, c] = EXIT
            g.exits.append((r, c))

        # ── Person start (top-left room, safely inside) ──
        sr, sc = max(2, r1 // 2), max(2, c1 // 2)
        g.cells[sr, sc] = PERSON
        g.start = (sr, sc)

        # ── Initial fire sources (bottom-right quadrant) ──
        fr = max(r1 + 1, rows * 3 // 5)
        fc = max(c1 + 1, cols * 3 // 5)
        fire_starts = [(fr, fc), (fr, fc + 1), (fr + 1, fc)]
        for r, c in fire_starts:
            if g.in_bounds(r, c) and g.cells[r, c] == EMPTY:
                g.cells[r, c] = FIRE
                g.fire_cells.add((r, c))

        return g

    # ── Accessors ────────────────────────────────────────────────────────────

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r: int, c: int) -> bool:
        """True if A* can step on this cell (not a wall or fire)."""
        if not self.in_bounds(r, c):
            return False
        return self.cells[r, c] not in (WALL, FIRE, ACTIVE_FIRE)

    def is_exit(self, r: int, c: int) -> bool:
        return self.cells[r, c] == EXIT

    def is_on_fire(self, r: int, c: int) -> bool:
        return self.cells[r, c] in (FIRE, ACTIVE_FIRE)

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        """Return walkable 4-connected neighbors."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [
            (r + dr, c + dc)
            for dr, dc in dirs
            if self.in_bounds(r + dr, c + dc) and self.is_walkable(r + dr, c + dc)
        ]

    def cell_cost(self, r: int, c: int) -> float:
        """
        Base movement cost for entering a cell, boosted by the ML risk score.

        Formula:
            cost = 1.0 + (risk_score * RISK_WEIGHT)

        A completely safe cell costs 1.0; a maximum-risk cell costs 6.0.
        This makes A* prefer safer corridors even when they are longer.
        """
        RISK_WEIGHT = 5.0
        return 1.0 + self.risk_scores[r, c] * RISK_WEIGHT

    def update_risk_scores(self, scores: np.ndarray) -> None:
        """Inject ML-predicted risk scores into the grid."""
        assert scores.shape == (self.rows, self.cols), "Shape mismatch"
        self.risk_scores = scores.copy()

    def reset_person(self) -> None:
        """Remove the PERSON marker (used after pathfinding starts)."""
        r, c = self.start
        if self.cells[r, c] == PERSON:
            self.cells[r, c] = EMPTY

    def clone(self) -> "Grid":
        """Deep-copy the grid (for safe replanning snapshots)."""
        g = Grid(self.rows, self.cols)
        g.cells = self.cells.copy()
        g.risk_scores = self.risk_scores.copy()
        g.start = self.start
        g.exits = list(self.exits)
        g.fire_cells = set(self.fire_cells)
        return g
