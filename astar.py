"""
astar.py — A* Pathfinding Algorithm (from scratch)
====================================================
Core pathfinding engine for the AI Fire Escape Planner.

Algorithm Details
-----------------
A* finds the safest path from start to the BEST exit by running a separate
A* search to EACH exit and returning the one with the lowest actual cost.

    g(n) : Accumulated cost from start to node n.
           = sum of cell_cost() for each cell visited.
           cell_cost = 1.0 + risk_score * 5.0
           High-risk cells are expensive, so A* routes around them.

    h(n) : Manhattan distance to the TARGET exit (single exit per run).
           This makes each run optimally directed at one exit.

    f(n) : g(n) + h(n) — priority in the min-heap.

Multi-exit strategy:
    Run one A* per exit → compare all successful paths → return cheapest.
    This guarantees the globally safest exit is always chosen, even if
    a farther exit has a cheaper (safer) path due to fire blocking the closer one.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

from grid import Grid

HEURISTIC_WEIGHT = 1.0   # 1.0 = fully optimal A*


# ── Internal heap node ────────────────────────────────────────────────────────

@dataclass
class _Node:
    f:   float
    g:   float
    pos: tuple

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.f == other.f


# ── Public result ─────────────────────────────────────────────────────────────

@dataclass
class AStarResult:
    path:           list        # [(row,col), ...] from start → exit
    total_cost:     float       # g-value at goal
    goal:           tuple       # which exit was reached
    nodes_expanded: int
    elapsed_ms:     float
    success:        bool
    # NEW: paths and costs to ALL reachable exits (for multi-exit display)
    all_paths:      list = field(default_factory=list)  # list of (cost, path, exit)


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _manhattan(a, b) -> float:
    return HEURISTIC_WEIGHT * (abs(a[0] - b[0]) + abs(a[1] - b[1]))


# ── Single-target A* ──────────────────────────────────────────────────────────

def _astar_to_exit(grid: Grid, start: tuple, goal: tuple):
    """
    Run A* from start to one specific exit (goal).
    Returns (path, cost, nodes_expanded) or (None, inf, n) if unreachable.
    """
    open_heap = []
    came_from = {start: None}
    g_score   = {start: 0.0}
    closed    = set()
    nodes_exp = 0

    h0 = _manhattan(start, goal)
    heapq.heappush(open_heap, _Node(f=h0, g=0.0, pos=start))

    while open_heap:
        node    = heapq.heappop(open_heap)
        current = node.pos

        if current in closed:
            continue
        closed.add(current)
        nodes_exp += 1

        # Goal reached
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
            return path, g_score[goal], nodes_exp

        for nb in grid.neighbors(*current):
            if nb in closed:
                continue
            tentative_g = g_score[current] + grid.cell_cost(*nb)
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb]   = tentative_g
                f = tentative_g + _manhattan(nb, goal)
                heapq.heappush(open_heap, _Node(f=f, g=tentative_g, pos=nb))

    return None, float("inf"), nodes_exp


# ── Main A* (tries ALL exits, picks best) ────────────────────────────────────

def astar(grid: Grid, start: Optional[tuple] = None) -> AStarResult:
    """
    Run A* from `start` to EVERY exit, return the path with the lowest cost.

    This guarantees the globally safest exit is selected — not just the
    geometrically nearest one. If fire blocks the closest exit, A* will
    automatically route to the next best one.
    """
    t0    = time.perf_counter()
    start = start or grid.start
    exits = [e for e in grid.exits if grid.is_exit(*e)]  # only valid exits

    if not exits:
        return AStarResult([], 0.0, (-1,-1), 0, 0.0, False)

    best_path  = None
    best_cost  = float("inf")
    best_goal  = (-1, -1)
    total_exp  = 0
    all_paths  = []   # (cost, path, exit) for every reachable exit

    # ── Run one A* per exit ──
    for exit_cell in exits:
        path, cost, exp = _astar_to_exit(grid, start, exit_cell)
        total_exp += exp
        if path is not None:
            all_paths.append((cost, path, exit_cell))
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_goal = exit_cell

    elapsed = (time.perf_counter() - t0) * 1000

    if best_path is None:
        return AStarResult([], 0.0, (-1,-1), total_exp, elapsed, False,
                           all_paths=[])

    # Sort all_paths by cost so frontend can display them ranked
    all_paths.sort(key=lambda x: x[0])

    return AStarResult(
        path           = best_path,
        total_cost     = best_cost,
        goal           = best_goal,
        nodes_expanded = total_exp,
        elapsed_ms     = elapsed,
        success        = True,
        all_paths      = all_paths,
    )


# ── Multi-person ──────────────────────────────────────────────────────────────

def astar_multi(grid: Grid, starts: list) -> list:
    return [astar(grid, start=s) for s in starts]
