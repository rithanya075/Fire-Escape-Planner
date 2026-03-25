"""
main.py — AI Fire Escape Planner — Entry Point
===============================================
Orchestrates the complete simulation:

    1. Build grid (office building layout)
    2. Train ML risk model on synthetic fire data
    3. Run initial A* pathfinding
    4. Simulate fire spread step by step
    5. After each fire step:
         a. Recompute ML risk scores → inject into grid
         b. Replan path with A* (dynamic replanning)
         c. Render and save frame
    6. Save summary statistics chart
    7. (Optional) multi-person mode

CLI Usage
---------
    python main.py                          # default single-person run
    python main.py --steps 8               # run for 8 fire steps
    python main.py --multi                 # multi-person mode (3 people)
    python main.py --steps 5 --no-frames   # skip frame images (fast)
    python main.py --seed 7                # different random fire spread
    python main.py --rows 15 --cols 15    # smaller grid
"""

import argparse
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project modules
from grid import Grid
from fire_simulation import FireSimulation
from ml_model import FireRiskModel
from astar import astar, astar_multi
from visualization import render_frame, save_simulation_frames, plot_summary
from utils import (
    print_header, print_step, print_success, print_warning, print_info,
    print_path_summary, print_grid_ascii, ensure_output_dir,
    get_multi_person_starts, Timer
)


# ── CLI argument parser ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Fire Escape Planner — A* + ML pathfinding simulation"
    )
    parser.add_argument("--steps", type=int, default=6,
                        help="Number of fire-spread steps to simulate (default: 6)")
    parser.add_argument("--rows", type=int, default=20,
                        help="Grid rows (default: 20)")
    parser.add_argument("--cols", type=int, default=20,
                        help="Grid columns (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fire spread (default: 42)")
    parser.add_argument("--multi", action="store_true",
                        help="Enable multi-person evacuation mode")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip saving per-step frame images (faster)")
    parser.add_argument("--no-risk", action="store_true",
                        help="Disable ML risk heatmap panel")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: 'output')")
    parser.add_argument("--ascii", action="store_true",
                        help="Print ASCII grid to console each step")
    return parser.parse_args()


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(args):
    ensure_output_dir(args.output)
    frames_dir = os.path.join(args.output, "frames")

    print_header("AI FIRE ESCAPE PLANNER  —  A* + Machine Learning")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Build grid
    # ─────────────────────────────────────────────────────────────────────────
    print_info("Building grid environment …")
    grid = Grid.create_default(rows=args.rows, cols=args.cols)
    print_success(f"Grid created: {grid.rows}×{grid.cols}  |  "
                  f"Exits: {len(grid.exits)}  |  "
                  f"Fire sources: {len(grid.fire_cells)}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Train ML model
    # ─────────────────────────────────────────────────────────────────────────
    print_info("Training ML risk-prediction model …")
    with Timer("ML training"):
        model = FireRiskModel(seed=args.seed)
        model.train(verbose=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Initialise fire simulation
    # ─────────────────────────────────────────────────────────────────────────
    fire_sim = FireSimulation(grid, seed=args.seed)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Multi-person mode setup
    # ─────────────────────────────────────────────────────────────────────────
    if args.multi:
        extra_persons = get_multi_person_starts(grid, n=2)
        persons = [grid.start] + extra_persons
        print_success(f"Multi-person mode: {len(persons)} evacuees at {persons}")
    else:
        persons = [grid.start]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Initial ML prediction + A* planning (step 0)
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Initial Planning (Step 0)")
    risk_grid = model.predict_risk_scores(fire_sim)
    grid.update_risk_scores(risk_grid)

    if args.multi:
        results = astar_multi(grid, persons)
        result = results[0]  # primary person for display
    else:
        result = astar(grid)

    print_path_summary(result, step=0)

    if args.ascii:
        print_grid_ascii(grid, path=result.path if result.success else None,
                         label="Step 0 — Initial")

    stats = []
    if result.success:
        stats.append({
            "step": 0,
            "cost": result.total_cost,
            "fire_coverage": fire_sim.fire_coverage(),
            "path_len": len(result.path),
        })

    # Save initial frame
    if not args.no_frames:
        frame_path = os.path.join(frames_dir, "frame_000.png")
        render_frame(
            grid=grid,
            result=result,
            title="Step 0 — Initial Path (A* + ML Risk)",
            show_risk=not args.no_risk,
            save_path=frame_path,
            persons=persons if args.multi else None,
        )
        plt.close("all")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Dynamic simulation loop
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Dynamic Fire Spread + Replanning")

    for step in range(1, args.steps + 1):
        print_step(step, f"Fire spread step {step} …")

        # ── a) Spread fire ──
        newly_ignited = fire_sim.step()
        coverage_pct = fire_sim.fire_coverage() * 100
        print_info(f"  New cells ignited: {len(newly_ignited)}  |  "
                   f"Coverage: {coverage_pct:.1f}%")

        # ── b) Check if previous path is now threatened ──
        if result.success and fire_sim.is_path_threatened(result.path):
            print_warning("  Previous path is ON FIRE — replanning required!")

        # ── c) Recompute ML risk scores ──
        risk_grid = model.predict_risk_scores(fire_sim)
        grid.update_risk_scores(risk_grid)

        # ── d) Replan with A* ──
        if args.multi:
            results = astar_multi(grid, persons)
            result = results[0]
        else:
            result = astar(grid)

        print_path_summary(result, step=step)

        if args.ascii:
            print_grid_ascii(grid, path=result.path if result.success else None,
                             label=f"Step {step}")

        # ── e) Record stats ──
        stats.append({
            "step": step,
            "cost": result.total_cost if result.success else float("nan"),
            "fire_coverage": fire_sim.fire_coverage(),
            "path_len": len(result.path) if result.success else 0,
        })

        # ── f) Save frame ──
        if not args.no_frames:
            frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
            render_frame(
                grid=grid,
                result=result,
                title=f"Step {step} — Dynamic Replanning (A* + ML)",
                show_risk=not args.no_risk,
                save_path=frame_path,
                persons=persons if args.multi else None,
            )
            plt.close("all")

        # Stop early if no path exists at all
        if not result.success and step > 1:
            print_warning("All paths blocked — simulation terminated early.")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Summary
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Simulation Complete")

    summary_path = os.path.join(args.output, "summary.png")
    plot_summary(stats, save_path=summary_path)

    print_success(f"Frames saved in   : {frames_dir}/")
    print_success(f"Summary chart     : {summary_path}")
    print_success(f"Total steps run   : {len(stats) - 1}")

    if result.success:
        print_success(
            f"Final path cost   : {result.total_cost:.2f}  "
            f"({len(result.path)} cells, exit at {result.goal})"
        )
    else:
        print_warning("Final state: No evacuation path available.")

    print_info("\nOpen the 'output/frames/' folder to view the simulation frames.")
    print_info("Run 'python make_gif.py' to stitch frames into an animated GIF.")


# ── GIF helper (bonus) ────────────────────────────────────────────────────────

def _write_gif_script(output_dir: str) -> None:
    script = f"""\
\"\"\"make_gif.py — Stitch simulation frames into an animated GIF.\"\"\"
import glob, os
from PIL import Image

frames_dir = "{output_dir}/frames"
out_path   = "{output_dir}/simulation.gif"

paths = sorted(glob.glob(f"{{frames_dir}}/frame_*.png"))
if not paths:
    print("No frames found — run main.py first.")
else:
    imgs = [Image.open(p) for p in paths]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=800, loop=0)
    print(f"GIF saved → {{out_path}}")
"""
    with open("make_gif.py", "w") as f:
        f.write(script)
    print_info("make_gif.py written — install Pillow and run it to get a GIF.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
    _write_gif_script(args.output)
