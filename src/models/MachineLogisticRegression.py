import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.BaseMachineModel import BaseMachineModel


class MachineLogisticRegression(BaseMachineModel):

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        super().__init__(random_state=random_state, use_scaling=True)
        self.max_iter = max_iter
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver='lbfgs',
            class_weight='balanced',
        )

    def _ohe_feature_names(self) -> list[str] | None:
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            return list(self.model.feature_names_in_)
        return None

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        """K-fold cross-validation using a Pipeline so preprocessing is fitted
        independently on each train fold — no data leakage from the val fold."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
        ])

        transformers: list[tuple[str, Any, list[str]]] = [('num', numeric_pipe, numeric_cols)]
        if categorical_cols:
            transformers.append((
                'cat',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                categorical_cols,
            ))

        pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(transformers)),
            ('model', LogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
                solver='lbfgs',
                class_weight='balanced',
            )),
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=cv, scoring='average_precision', n_jobs=-1,
        )

        cv_metrics = {
            'cv_scores':   cv_scores.tolist(),
            'mean_pr_auc': float(cv_scores.mean()),
            'std_pr_auc':  float(cv_scores.std()),
        }
        self.metrics['cross_validation'] = cv_metrics
        return cv_metrics

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_scaled = self.preprocess_features(X_train, fit=True)
        self.feature_names = X_scaled.columns.tolist()
        self.model.fit(X_scaled, y_train)
        self._is_fitted = True

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        if self.feature_names is None:
            raise ValueError("feature_names not set — call fit() first")
        return pd.DataFrame({
            'feature':     self.feature_names,
            'coefficient': self.model.coef_[0],
        }).sort_values('coefficient', ascending=False)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / "baseline_lr.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        self._save_preprocessors(model_dir, prefix="baseline")
        self.get_feature_importance().to_csv(
            model_dir / "baseline_feature_importance.csv", index=False
        )

    @staticmethod
    def load(model_dir: Path) -> 'MachineLogisticRegression':
        model_dir = Path(model_dir)
        instance = MachineLogisticRegression()
        with open(model_dir / "baseline_lr.pkl", 'rb') as f:
            instance.model = pickle.load(f)
        instance._load_preprocessors(model_dir, prefix="baseline")
        instance._is_fitted = True
        return instance
