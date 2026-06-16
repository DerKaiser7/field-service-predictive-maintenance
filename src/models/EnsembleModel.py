"""
Ensemble Model: Stacked LR-over-XGBoost with cost-optimised threshold.

Wraps the two base models and a meta-learner into a single object
that the FastAPI serving layer can load and call without needing to
manage three separate artifacts.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix, f1_score
)

from src.models.MachineLogisticRegression import MachineLogisticRegression
from src.models.MachineXGBoost import MachineXGBoost


class EnsembleModel:
    """
    Stacked ensemble: LR + XGBoost base models, LogisticRegression meta-learner.

    Attributes:
        lr: Fitted MachineLogisticRegression base model
        xgb: Fitted MachineXGBoost base model
        meta_learner: Fitted LogisticRegression trained on stacked val predictions
        threshold: Optimised decision threshold (cost-based)
        metrics: Evaluation metrics from training
    """

    COST_FN = 5
    COST_FP = 1

    def __init__(self) -> None:
        self.lr: MachineLogisticRegression | None = None
        self.xgb: MachineXGBoost | None = None
        self.meta_learner: LogisticRegression | None = None
        self.threshold: float = 0.5
        self.metrics: dict = {}

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.lr is None or self.xgb is None:
            raise ValueError("Base models must be set before building meta-features")
        lr_proba = self.lr.predict_proba(X)[:, 1]
        xgb_proba = self.xgb.predict_proba(X)[:, 1]
        return np.column_stack([lr_proba, xgb_proba])

    def fit_meta_learner(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train the meta-learner on val-set stacked predictions."""
        X_meta = self._build_meta_features(X_val)
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_learner.fit(X_meta, y_val)

    def optimise_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Sweep thresholds on val set; pick the one minimising business cost."""
        val_proba = self.predict_proba(X_val)
        thresholds = np.linspace(0.01, 0.99, 200)
        best_cost = float("inf")
        best_t = 0.5

        for t in thresholds:
            y_pred = (val_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            cost = fn * self.COST_FN + fp * self.COST_FP
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)

        self.threshold = best_t
        return best_t

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return failure probability from the meta-learner."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner is not fitted")
        X_meta = self._build_meta_features(X)
        return self.meta_learner.predict_proba(X_meta)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using the optimised threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate on test set and store metrics."""
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= self.threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)

        self.metrics = {
            "pr_auc": float(pr_auc),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1": float(f1_score(y_test, y_pred)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "threshold": self.threshold,
            "business_cost": int(fn * self.COST_FN + fp * self.COST_FP),
        }
        return self.metrics

    def save(self, model_dir: Path) -> None:
        """Save meta-learner, threshold, and metrics to model_dir."""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)

        with open(model_dir / "ensemble_meta_learner.pkl", "wb") as f:
            pickle.dump(self.meta_learner, f)

        threshold_payload = {
            "optimal_threshold": self.threshold,
            "cost_fn": self.COST_FN,
            "cost_fp": self.COST_FP,
            "rationale": (
                f"Threshold swept from 0.01 to 0.99 on val set. "
                f"Selected {self.threshold:.3f} to minimise ({self.COST_FN}*FN + {self.COST_FP}*FP). "
                f"Missed failures penalised {self.COST_FN}x over false alarms to reflect "
                f"production maintenance cost asymmetry."
            ),
        }
        with open(model_dir / "optimal_threshold.json", "w") as f:
            json.dump(threshold_payload, f, indent=2)

        with open(model_dir / "ensemble_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    @staticmethod
    def load(model_dir: Path) -> "EnsembleModel":
        """Load the full ensemble (base models + meta-learner + threshold)."""
        model_dir = Path(model_dir)
        instance = EnsembleModel()

        instance.lr = MachineLogisticRegression.load(model_dir)
        instance.xgb = MachineXGBoost.load(model_dir)

        with open(model_dir / "ensemble_meta_learner.pkl", "rb") as f:
            instance.meta_learner = pickle.load(f)

        with open(model_dir / "optimal_threshold.json", "r") as f:
            data = json.load(f)
            instance.threshold = data["optimal_threshold"]

        with open(model_dir / "ensemble_metrics.json", "r") as f:
            instance.metrics = json.load(f)

        return instance
