import json
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve

from src.models.BaseMachineModel import BaseMachineModel


class MachineXGBoost(BaseMachineModel):

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        scale_pos_weight: float = 1.0,
    ) -> None:
        super().__init__(random_state=random_state, use_scaling=False)
        self.scale_pos_weight = scale_pos_weight
        self.best_params: dict = {}
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='aucpr',
            verbosity=0,
        )

    def _ohe_feature_names(self) -> list[str] | None:
        if not self._is_fitted:
            return None
        return self.model.get_booster().feature_names or None

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        param_grid: dict | None = None,
    ) -> dict:
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_val_scaled = self.preprocess_features(X_val, fit=False)

        if param_grid is None:
            param_grid = {
                'max_depth':     [5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators':  [100, 200, 300],
            }

        best_pr_auc = 0.0
        best_params: dict = {}

        for max_depth in param_grid.get('max_depth', [6]):
            for lr in param_grid.get('learning_rate', [0.05]):
                for n_est in param_grid.get('n_estimators', [100]):
                    params = {
                        'max_depth':        max_depth,
                        'learning_rate':    lr,
                        'n_estimators':     n_est,
                        'scale_pos_weight': self.scale_pos_weight,
                        'random_state':     self.random_state,
                        'objective':        'binary:logistic',
                        'eval_metric':      'aucpr',
                        'verbosity':        0,
                    }
                    model = xgb.XGBClassifier(**params, early_stopping_rounds=10)
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False,
                    )
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                    pr_auc = auc(recall, precision)
                    if pr_auc > best_pr_auc:
                        best_pr_auc = pr_auc
                        best_params = params

        self.best_params = best_params
        self.model = xgb.XGBClassifier(**best_params)
        return best_params

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_scaled = self.preprocess_features(X_train, fit=True)
        self.feature_names = X_scaled.columns.tolist()
        self.model.fit(X_scaled, y_train, verbose=False)
        self._is_fitted = True

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        if self.feature_names is None:
            raise ValueError("feature_names not set — call fit() first")
        return pd.DataFrame({
            'feature': self.feature_names,
            'gain':    self.model.feature_importances_,
        }).sort_values('gain', ascending=False)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        with open(model_dir / "xgboost_main.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        self._save_preprocessors(model_dir, prefix="xgboost")
        json.dump(
            {k: v for k, v in self.best_params.items()
             if k not in ('objective', 'eval_metric', 'verbosity')},
            open(model_dir / "xgboost_hyperparams.json", 'w'),
            indent=2,
        )
        self.get_feature_importance().to_csv(
            model_dir / "xgboost_feature_importance.csv", index=False
        )

    @staticmethod
    def load(model_dir: Path) -> 'MachineXGBoost':
        model_dir = Path(model_dir)
        instance = MachineXGBoost()
        with open(model_dir / "xgboost_main.pkl", 'rb') as f:
            instance.model = pickle.load(f)
        instance._load_preprocessors(model_dir, prefix="xgboost")
        with open(model_dir / "xgboost_hyperparams.json", 'r') as f:
            instance.best_params = json.load(f)
        instance._is_fitted = True
        return instance