"""
Shared base class for MachineLogisticRegression and MachineXGBoost.

Owns the preprocessing pipeline, evaluation metrics, predict/predict_proba,
and serialisation helpers so subclasses contain only model-specific logic.
"""

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler


class BaseMachineModel(ABC):

    def __init__(self, random_state: int = 42, use_scaling: bool = True) -> None:
        self.random_state = random_state
        self._use_scaling = use_scaling
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names: list[str] | None = None
        self.metrics: dict[str, Any] = {}
        self._is_fitted = False
        self.model: Any = None

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before use")

    def _ohe_feature_names(self) -> list[str] | None:
        """Return expected post-OHE feature names for inference-time alignment.

        Subclasses override this to read from their fitted model's stored feature
        list. Returning None skips alignment (safe during fit).
        """
        return None

    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        X_processed = X.copy()

        if fit:
            X_processed[numeric_cols] = np.asarray(self.imputer.fit_transform(X[numeric_cols]))  # type: ignore[index]
        else:
            X_processed[numeric_cols] = np.asarray(self.imputer.transform(X[numeric_cols]))  # type: ignore[index]

        if self._use_scaling:
            if fit:
                X_processed[numeric_cols] = np.asarray(self.scaler.fit_transform(X_processed[numeric_cols]))  # type: ignore[index]
            else:
                X_processed[numeric_cols] = np.asarray(self.scaler.transform(X_processed[numeric_cols]))  # type: ignore[index]

        if categorical_cols:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
            expected = self._ohe_feature_names()
            if expected:
                for col in expected:
                    if col not in X_processed.columns:
                        X_processed[col] = 0.0
                X_processed = X_processed[expected]

        return X_processed

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        self._check_fitted()
        X_scaled = self.preprocess_features(X_test, fit=False)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = self.model.predict(X_scaled)

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        test_metrics: dict[str, float] = {
            'test_pr_auc':      float(pr_auc),
            'test_precision':   float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'test_recall':      float(sensitivity),
            'test_specificity': float(specificity),
            'test_f1':          float(f1_score(y_test, y_pred)),
            'test_tp':          float(tp),
            'test_fp':          float(fp),
            'test_tn':          float(tn),
            'test_fn':          float(fn),
        }
        self.metrics['test'] = test_metrics
        return test_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(self.preprocess_features(X, fit=False))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(self.preprocess_features(X, fit=False))

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame: ...

    # ------------------------------------------------------------------
    # Serialisation helpers — subclasses call these then handle their own
    # model file and any extra artifacts (hyperparams, feature importance).
    # ------------------------------------------------------------------

    def _save_preprocessors(self, model_dir: Path, prefix: str) -> None:
        with open(model_dir / f"{prefix}_imputer.pkl", 'wb') as f:
            pickle.dump(self.imputer, f)
        if self._use_scaling:
            with open(model_dir / f"{prefix}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
        with open(model_dir / f"{prefix}_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _load_preprocessors(self, model_dir: Path, prefix: str) -> None:
        with open(model_dir / f"{prefix}_imputer.pkl", 'rb') as f:
            self.imputer = pickle.load(f)
        scaler_path = model_dir / f"{prefix}_scaler.pkl"
        if self._use_scaling and scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        with open(model_dir / f"{prefix}_metrics.json", 'r') as f:
            self.metrics = json.load(f)
