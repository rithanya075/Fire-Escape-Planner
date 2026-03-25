# 🔥 AI Fire Escape Planner

A production-quality Python simulation that finds the **safest evacuation path** inside a burning building using **A\* pathfinding** and a **Machine Learning fire-risk prediction pipeline**.

---

## 🎯 What It Does

| Feature | Detail |
|---|---|
| **A\* Algorithm** | Custom implementation with g(n) + h(n), weighted nodes, multi-exit support |
| **ML Risk Model** | Random Forest trained on fire physics; predicts risk score 0–1 per cell |
| **Fire Simulation** | Probabilistic BFS-based spread; temperature & smoke fields |
| **Dynamic Replanning** | Path recomputed every step as fire spreads |
| **Visualization** | Matplotlib frames: cell grid + ML risk heatmap side-by-side |
| **Multi-person** | Independent A\* for each evacuee |
| **CLI** | Fully configurable via command-line flags |

---

## 📁 Project Structure

```
ai_fire_escape/
├── main.py              # Orchestrator + CLI
├── grid.py              # 2D building grid (cell types, costs, walkability)
├── astar.py             # A* from scratch (heapq, g/h/f, multi-exit)
├── fire_simulation.py   # Fire spread + temperature + smoke fields
├── ml_model.py          # RandomForest pipeline (train → predict → inject)
├── visualization.py     # Matplotlib renderer (grid + risk heatmap)
├── utils.py             # Console helpers, timer, ASCII grid
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-fire-escape-planner.git
cd ai-fire-escape-planner

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

### Basic run (default settings)
```bash
python main.py
```

### With options
```bash
# Run 10 fire-spread steps
python main.py --steps 10

# Multi-person evacuation (3 people)
python main.py --multi

# Smaller grid, more steps, different fire pattern
python main.py --rows 15 --cols 15 --steps 8 --seed 7

# Skip saving image frames (fast terminal-only mode)
python main.py --steps 5 --no-frames

# Print ASCII grid to console each step
python main.py --ascii

# Custom output folder
python main.py --output my_results
```

### Make animated GIF (after running main.py)
```bash
pip install Pillow
python make_gif.py
```

---

## 📊 Output Files

After running, the `output/` folder contains:

```
output/
├── frames/
│   ├── frame_000.png    # Initial path
│   ├── frame_001.png    # After step 1
│   ├── frame_002.png    # After step 2
│   └── ...
├── summary.png          # Path cost / fire coverage over time
└── simulation.gif       # (if make_gif.py is run)
```

---

## 🧠 Architecture

### A\* Algorithm (`astar.py`)

```
f(n) = g(n) + h(n)

g(n) = accumulated cost from start
       = sum of cell_cost(r,c) for each cell visited
       where cell_cost = 1.0 + risk_score * 5.0

h(n) = weighted Manhattan distance to nearest exit
       = 1.2 * (|Δrow| + |Δcol|)

Open set  = min-heap keyed by f(n)
Closed set = set of expanded nodes
```

ML risk scores directly inflate `g(n)`, so A\* **actively avoids** high-risk corridors.

### ML Pipeline (`ml_model.py`)

```
Features (per cell):
  [dist_from_fire, temperature_norm, smoke_level, time_norm]

Model: RandomForestClassifier (100 trees, max_depth=8)
Output: P(high_risk) → injected as risk_score into grid

Feature importance (approximate):
  dist_from_fire   ████████████████ 0.40
  temperature_norm ██████████ 0.25
  smoke_level      ████████ 0.20
  time_norm        ██████ 0.15
```

### Fire Spread (`fire_simulation.py`)

```
P(cell ignites) = BASE_PROB(0.35) + (burning_neighbours - 1) * 0.15
                  capped at 0.90
Walls block fire completely.
Temperature diffuses outward from fire cells.
Smoke spreads with 0.4 coefficient + 0.85 decay per step.
```

---

## 📋 Example Console Output

```
╔══════════════════════════════════════════════╗
║  AI FIRE ESCAPE PLANNER  —  A* + Machine Learning  ║
╚══════════════════════════════════════════════╝

ℹ Building grid environment …
✓ Grid created: 20×20 | Exits: 4 | Fire sources: 3

ℹ Training ML risk-prediction model …
[ML] Random Forest training complete.
[ML] Training samples : 4,000
[ML] Test samples     : 1,000
[ML] Classification report:
              precision  recall  f1-score
  Low Risk       0.912   0.934     0.923
  High Risk      0.921   0.897     0.909

╔═══════════════════════════════╗
║  Initial Planning (Step 0)   ║
╚═══════════════════════════════╝
  Path found:
    • Length     : 14 cells
    • Total cost : 21.34  (includes ML risk weights)
    • Exit cell  : (10, 0)
    • Nodes exp. : 127
    • A* time    : 2.41 ms

[Step 01] Fire spread step 1 …
ℹ   New cells ignited: 3  |  Coverage: 2.1%
  Path found:
    • Length     : 17 cells
    • Total cost : 27.88
...
```

---

## 🛠 VS Code Tips

1. Open the folder: `File → Open Folder → ai_fire_escape/`
2. Select Python interpreter: `Ctrl+Shift+P → Python: Select Interpreter → venv`
3. Run main.py: Right-click → `Run Python File in Terminal`
4. Or use the integrated terminal: `Ctrl+\`` → `python main.py --ascii`

---

## 🐙 GitHub Setup (Step by Step)

1. Create a new repo at https://github.com/new
   - Name: `ai-fire-escape-planner`
   - Public, no README (we have one)

2. In your project folder:
```bash
git init
git add .
git commit -m "Initial commit: AI Fire Escape Planner"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-fire-escape-planner.git
git push -u origin main
```

3. Add a `.gitignore`:
```
venv/
output/
__pycache__/
*.pyc
.DS_Store
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Array operations, grid math |
| `scikit-learn` | RandomForest ML pipeline |
| `matplotlib` | Grid and heatmap visualization |
| `colorama` | Cross-platform coloured terminal output |

---

## 📜 License

MIT — free to use, modify, and distribute.
