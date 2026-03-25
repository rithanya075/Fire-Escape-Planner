"""
utils.py — Shared Utilities
============================
Helper functions used across multiple modules.
"""

import time
import os
from colorama import Fore, Style, init

init(autoreset=True)  # colorama cross-platform colour support


# ── Console printing helpers ──────────────────────────────────────────────────

def print_header(text: str) -> None:
    bar = "═" * (len(text) + 4)
    print(f"\n{Fore.CYAN}╔{bar}╗")
    print(f"║  {text}  ║")
    print(f"╚{bar}╝{Style.RESET_ALL}\n")


def print_step(step: int, msg: str) -> None:
    print(f"{Fore.YELLOW}[Step {step:02d}]{Style.RESET_ALL} {msg}")


def print_success(msg: str) -> None:
    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")


def print_warning(msg: str) -> None:
    print(f"{Fore.RED}⚠ {msg}{Style.RESET_ALL}")


def print_info(msg: str) -> None:
    print(f"{Fore.BLUE}ℹ {msg}{Style.RESET_ALL}")


def print_path_summary(result, step: int = 0) -> None:
    """Print a formatted A* result summary."""
    from astar import AStarResult
    if not result.success:
        print_warning(f"Step {step}: NO PATH FOUND — all exits blocked!")
        return

    path_len = len(result.path)
    cost = result.total_cost
    goal = result.goal
    expanded = result.nodes_expanded
    ms = result.elapsed_ms

    print(f"  {Fore.GREEN}Path found:{Style.RESET_ALL}")
    print(f"    • Length     : {path_len} cells")
    print(f"    • Total cost : {cost:.2f}  (includes ML risk weights)")
    print(f"    • Exit cell  : {goal}")
    print(f"    • Nodes exp. : {expanded}")
    print(f"    • A* time    : {ms:.2f} ms")


# ── Grid ASCII dump (fallback / debug) ───────────────────────────────────────

_GLYPH = {
    0: "·",    # EMPTY
    1: "█",    # WALL
    2: "F",    # FIRE
    3: "E",    # EXIT
    4: "P",    # PERSON
    5: "f",    # ACTIVE FIRE
}

def print_grid_ascii(grid, path: list | None = None, label: str = "") -> None:
    """
    Print a compact ASCII representation of the grid to the console.
    Path cells are shown as '*'.
    """
    path_set = set(path) if path else set()
    if label:
        print(f"\n{Fore.CYAN}{label}{Style.RESET_ALL}")
    for r in range(grid.rows):
        row_str = ""
        for c in range(grid.cols):
            if (r, c) in path_set:
                row_str += Fore.YELLOW + "*" + Style.RESET_ALL
            elif grid.cells[r, c] in (2, 5):  # fire
                row_str += Fore.RED + _GLYPH.get(grid.cells[r, c], "?") + Style.RESET_ALL
            elif grid.cells[r, c] == 3:        # exit
                row_str += Fore.GREEN + "E" + Style.RESET_ALL
            else:
                row_str += _GLYPH.get(grid.cells[r, c], "?")
        print(row_str)


# ── Timing context manager ────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0
        if self.label:
            print_info(f"{self.label}: {self.elapsed*1000:.1f} ms")


# ── Output directory helper ───────────────────────────────────────────────────

def ensure_output_dir(path: str = "output") -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ── Multi-person start positions ─────────────────────────────────────────────

def get_multi_person_starts(grid, n: int = 3) -> list[tuple[int, int]]:
    """
    Return n walkable cells spread across the grid as person starting
    positions (avoids fire cells and walls).  Used for multi-person mode.
    """
    from grid import EMPTY
    import random
    random.seed(99)

    candidates = [
        (r, c)
        for r in range(1, grid.rows - 1)
        for c in range(1, grid.cols - 1)
        if grid.cells[r, c] == EMPTY
    ]
    random.shuffle(candidates)

    # Spread people out — pick positions with at least 3-cell separation
    selected = []
    for pos in candidates:
        if len(selected) >= n:
            break
        if all(abs(pos[0] - s[0]) + abs(pos[1] - s[1]) > 5 for s in selected):
            selected.append(pos)

    return selected
