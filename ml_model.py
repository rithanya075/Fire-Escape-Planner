"""
ml_model.py — Machine Learning Risk Prediction Pipeline
=========================================================
Trains and applies a Random Forest classifier to predict the fire-risk
score (0.0–1.0) for every cell in the grid.

Pipeline
--------
1. GENERATE synthetic training data from fire physics rules.
   (In a real deployment this would come from IoT sensor logs.)

2. TRAIN a RandomForestClassifier on 4 features:
       • dist_from_fire   (normalised BFS distance, 0=on fire, 1=far)
       • temperature_norm (0=ambient 25°C, 1=fire 750°C)
       • smoke_level      (0=clear, 1=max smoke)
       • time_norm        (0=start, 1=step 20+)

3. PREDICT risk for the current state of the grid by calling
   predict_risk_scores(), which returns a (rows, cols) array of
   probabilities in [0.0, 1.0].

4. INJECT those scores into the Grid object.  A* then reads them
   via grid.cell_cost() and routes around high-risk zones.

Design Choices
--------------
• RandomForest over LogisticRegression: handles non-linear risk surfaces
  (e.g. smoke can be high even when distance from fire is moderate).
• Synthetic training: physics-based labels → model generalises correctly
  even without real data.
• Probability output (predict_proba) rather than hard classes gives A*
  a smooth cost gradient to optimise over.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Training data generation ──────────────────────────────────────────────────

def _generate_training_data(n_samples: int = 5000, seed: int = 0) -> tuple:
    """
    Produce synthetic (features, labels) for training.

    Features (4 columns)
    --------------------
    dist_from_fire   : float in [0, 1]   0 = at fire, 1 = far away
    temperature_norm : float in [0, 1]   0 = cold, 1 = hot
    smoke_level      : float in [0, 1]
    time_norm        : float in [0, 1]   how long fire has been burning

    Label
    -----
    1 = HIGH RISK (score threshold ≥ 0.5)
    0 = LOW RISK

    The labelling rule encodes physical intuition:
        risk = w1*(1-dist) + w2*temp + w3*smoke + w4*time
    """
    rng = np.random.default_rng(seed)

    dist  = rng.uniform(0.0, 1.0, n_samples)
    temp  = rng.uniform(0.0, 1.0, n_samples)
    smoke = rng.uniform(0.0, 1.0, n_samples)
    time_ = rng.uniform(0.0, 1.0, n_samples)

    # Physics-inspired risk score (ground truth)
    raw_risk = (
        0.40 * (1.0 - dist)   # proximity to fire is the biggest factor
        + 0.25 * temp          # temperature
        + 0.20 * smoke         # smoke inhalation danger
        + 0.15 * time_         # fire age (spread has had more time)
    )
    # Add small noise to avoid overfitting perfect synthetic data
    raw_risk += rng.normal(0, 0.05, n_samples)
    raw_risk = np.clip(raw_risk, 0.0, 1.0)

    labels = (raw_risk >= 0.45).astype(int)   # binary threshold

    X = np.column_stack([dist, temp, smoke, time_])
    return X, labels, raw_risk


# ── Model class ───────────────────────────────────────────────────────────────

class FireRiskModel:
    """
    Wraps a scikit-learn Pipeline (Scaler → RandomForest) for fire-risk
    prediction.

    Usage
    -----
        model = FireRiskModel()
        model.train(verbose=True)
        risk_grid = model.predict_risk_scores(fire_sim)   # (rows, cols)
        grid.update_risk_scores(risk_grid)
    """

    FEATURE_NAMES = ["dist_from_fire", "temperature_norm", "smoke_level", "time_norm"]

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.seed = seed
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            )),
        ])
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_samples: int = 5000, verbose: bool = True) -> None:
        """Generate synthetic data and fit the pipeline."""
        X, y, _ = _generate_training_data(n_samples, seed=self.seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed, stratify=y
        )

        self._pipeline.fit(X_train, y_train)
        self._trained = True

        if verbose:
            y_pred = self._pipeline.predict(X_test)
            print("\n[ML] Random Forest training complete.")
            print(f"[ML] Training samples : {len(X_train):,}")
            print(f"[ML] Test samples     : {len(X_test):,}")
            print("[ML] Classification report (test set):")
            print(classification_report(y_test, y_pred,
                                        target_names=["Low Risk", "High Risk"],
                                        digits=3))
            self._print_feature_importance()

    def _print_feature_importance(self) -> None:
        clf = self._pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        print("[ML] Feature importances:")
        for name, imp in zip(self.FEATURE_NAMES, importances):
            bar = "█" * int(imp * 40)
            print(f"       {name:<22} {imp:.3f}  {bar}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_risk_scores(self, fire_sim) -> np.ndarray:
        """
        Predict per-cell risk scores for the current simulation state.

        Parameters
        ----------
        fire_sim : FireSimulation
            Provides the feature matrix via get_ml_features().

        Returns
        -------
        risk_grid : np.ndarray, shape (rows, cols)
            Float risk scores in [0.0, 1.0].
            Cells on fire are forced to 1.0.
            Walls are forced to 0.0 (irrelevant; A* skips them).
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        grid = fire_sim.grid
        rows, cols = grid.rows, grid.cols

        # Feature matrix: (rows*cols, 4)
        X = fire_sim.get_ml_features()

        # Predict probability of HIGH RISK class (index 1)
        proba = self._pipeline.predict_proba(X)[:, 1]  # shape: (rows*cols,)
        risk_grid = proba.reshape(rows, cols)

        # Override: fire cells = 1.0, walls = 0.0
        from grid import WALL, FIRE, ACTIVE_FIRE
        for r in range(rows):
            for c in range(cols):
                if grid.cells[r, c] == WALL:
                    risk_grid[r, c] = 0.0
                elif grid.cells[r, c] in (FIRE, ACTIVE_FIRE):
                    risk_grid[r, c] = 1.0

        return risk_grid

    # ── Utility ───────────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        return self._trained

    def risk_category(self, score: float) -> str:
        """Human-readable risk label for a score."""
        if score < 0.25:
            return "SAFE"
        elif score < 0.50:
            return "LOW"
        elif score < 0.75:
            return "MEDIUM"
        else:
            return "HIGH"
