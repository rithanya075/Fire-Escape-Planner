"""
fire_simulation.py — Dynamic Fire Spread Simulation
=====================================================
Simulates realistic fire propagation across the building grid over time.

Fire Spread Model
-----------------
At each time step, every cell adjacent to a burning cell has a probability
of catching fire:

    P(ignite) = BASE_PROB * intensity_factor * distance_decay

Where:
    BASE_PROB        : Base ignition probability per step (~0.35)
    intensity_factor : Scales with how many burning neighbours a cell has
                       (more surrounding fire → higher chance)
    distance_decay   : Cells farther from fire sources ignite more slowly
                       (not used in immediate-neighbour spread, but affects
                        the simulated temperature used by the ML model)

Walls block fire completely. Exits can burn but remain passable
(the building structure holds; exits don't collapse).

The module also maintains simulated physical properties used as ML features:
    - temperature[r,c]  : Heat in °C (fire=700+, adjacent=200–400, ambient=25)
    - smoke_level[r,c]  : 0.0–1.0 smoke concentration
    - time_step         : Elapsed simulation steps
"""

import numpy as np
from grid import Grid, WALL, EXIT, FIRE, ACTIVE_FIRE, EMPTY, PERSON

# ── Simulation constants ──────────────────────────────────────────────────────
BASE_IGNITION_PROB = 0.35    # chance a neighbour catches fire each step
INTENSITY_BOOST = 0.15       # extra probability per additional burning neighbour
AMBIENT_TEMP = 25.0          # °C base room temperature
FIRE_TEMP = 750.0            # °C temperature at fire cell
HEAT_DECAY = 0.6             # temperature drop per cell of distance from fire
SMOKE_SPREAD = 0.4           # smoke diffusion rate per step


class FireSimulation:
    """
    Manages fire spread and physical environment state over time.

    Attributes
    ----------
    grid        : Reference to the live Grid object (mutated in-place).
    time_step   : Current simulation step number.
    temperature : 2D array of cell temperatures (°C).
    smoke_level : 2D array of smoke concentrations (0.0–1.0).
    rng         : NumPy random generator (seeded for reproducibility).
    """

    def __init__(self, grid: Grid, seed: int = 42):
        self.grid = grid
        self.time_step = 0
        self.rng = np.random.default_rng(seed)

        rows, cols = grid.rows, grid.cols
        self.temperature = np.full((rows, cols), AMBIENT_TEMP, dtype=float)
        self.smoke_level = np.zeros((rows, cols), dtype=float)

        # Initialise temperature at existing fire cells
        for r, c in grid.fire_cells:
            self.temperature[r, c] = FIRE_TEMP

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self) -> set[tuple[int, int]]:
        """
        Advance the simulation by one time step.

        Returns
        -------
        newly_ignited : Set of (row, col) cells that caught fire this step.
        """
        self.time_step += 1
        grid = self.grid
        newly_ignited: set[tuple[int, int]] = set()

        # Collect cells to potentially ignite (don't modify while iterating)
        candidates: dict[tuple[int, int], int] = {}  # cell → burning neighbour count

        for (fr, fc) in grid.fire_cells:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in dirs:
                nr, nc = fr + dr, fc + dc
                if not grid.in_bounds(nr, nc):
                    continue
                cell_type = grid.cells[nr, nc]
                # Walls block fire; already-burning cells skip
                if cell_type in (WALL, FIRE, ACTIVE_FIRE):
                    continue
                candidates[(nr, nc)] = candidates.get((nr, nc), 0) + 1

        # ── Probabilistic ignition ──
        for (nr, nc), burning_neighbours in candidates.items():
            prob = BASE_IGNITION_PROB + (burning_neighbours - 1) * INTENSITY_BOOST
            prob = min(prob, 0.90)  # cap at 90%
            if self.rng.random() < prob:
                grid.cells[nr, nc] = ACTIVE_FIRE
                grid.fire_cells.add((nr, nc))
                newly_ignited.add((nr, nc))
                self.temperature[nr, nc] = FIRE_TEMP

        # ── Update temperature field (heat diffusion) ──
        self._update_temperature()

        # ── Update smoke field ──
        self._update_smoke()

        return newly_ignited

    # ── Physical environment updates ─────────────────────────────────────────

    def _update_temperature(self) -> None:
        """
        Diffuse heat from fire cells outward.
        Each non-fire cell's temperature rises toward a weighted average of
        its neighbours' temperatures, with fire cells anchored at FIRE_TEMP.
        """
        temp = self.temperature
        new_temp = temp.copy()
        rows, cols = self.grid.rows, self.grid.cols

        for r in range(rows):
            for c in range(cols):
                if self.grid.is_on_fire(r, c):
                    new_temp[r, c] = FIRE_TEMP
                    continue
                if self.grid.cells[r, c] == WALL:
                    continue
                # Average temperature of accessible neighbours
                nb_temps = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if self.grid.in_bounds(nr, nc):
                        nb_temps.append(temp[nr, nc])
                if nb_temps:
                    avg_nb = np.mean(nb_temps)
                    # Blend current temp toward neighbour average
                    new_temp[r, c] = temp[r, c] * 0.7 + avg_nb * 0.3

        self.temperature = new_temp

    def _update_smoke(self) -> None:
        """
        Smoke rises from fire cells and diffuses outward, decaying over time.
        """
        smoke = self.smoke_level
        new_smoke = smoke * 0.85  # natural decay (ventilation)
        rows, cols = self.grid.rows, self.grid.cols

        for r in range(rows):
            for c in range(cols):
                if self.grid.is_on_fire(r, c):
                    new_smoke[r, c] = 1.0
                elif self.grid.cells[r, c] != WALL:
                    # Receive smoke from burning neighbours
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if self.grid.in_bounds(nr, nc) and self.grid.is_on_fire(nr, nc):
                            new_smoke[r, c] = min(1.0, new_smoke[r, c] + SMOKE_SPREAD)

        self.smoke_level = np.clip(new_smoke, 0.0, 1.0)

    # ── Feature extraction for ML model ─────────────────────────────────────

    def get_ml_features(self) -> np.ndarray:
        """
        Build a feature matrix for the ML risk-prediction model.

        Returns
        -------
        features : np.ndarray, shape (rows*cols, 4)
            Columns: [dist_from_fire, temperature_norm, smoke_level, time_norm]
        """
        rows, cols = self.grid.rows, self.grid.cols
        dist_map = self._distance_from_fire()

        # Normalise distance (max possible = rows + cols)
        max_dist = rows + cols
        dist_norm = np.clip(dist_map / max_dist, 0.0, 1.0)

        # Normalise temperature to [0, 1]
        temp_norm = np.clip(
            (self.temperature - AMBIENT_TEMP) / (FIRE_TEMP - AMBIENT_TEMP),
            0.0, 1.0
        )

        # Time normalised (saturates at step 20)
        time_norm = np.full((rows, cols), min(self.time_step / 20.0, 1.0))

        # Stack into (rows*cols, 4)
        features = np.stack(
            [dist_norm.ravel(), temp_norm.ravel(),
             self.smoke_level.ravel(), time_norm.ravel()],
            axis=1
        )
        return features

    def _distance_from_fire(self) -> np.ndarray:
        """
        BFS from all current fire cells to compute shortest grid distance
        (in steps) for every cell. Walls are traversable for distance calc
        (fire heat travels through walls, just not as pathfinding moves).
        """
        from collections import deque
        rows, cols = self.grid.rows, self.grid.cols
        dist = np.full((rows, cols), float(rows + cols))

        queue = deque()
        for (r, c) in self.grid.fire_cells:
            dist[r, c] = 0.0
            queue.append((r, c, 0))

        while queue:
            r, c, d = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc, d + 1))

        return dist

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def fire_coverage(self) -> float:
        """Fraction of non-wall cells currently on fire."""
        total = np.sum(self.grid.cells != WALL)
        return len(self.grid.fire_cells) / max(total, 1)

    def is_path_threatened(self, path: list[tuple]) -> bool:
        """Return True if any cell on the path is now on fire."""
        return any(self.grid.is_on_fire(r, c) for r, c in path)
