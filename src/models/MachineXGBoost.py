"""
XGBoost Model for Predictive Maintenance

Class-based wrapper around xgboost.XGBClassifier that provides:
- Hyperparameter tuning (grid search)
- Class imbalance handling
- Feature importance computation
- Model serialization
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler


class MachineXGBoost:
    """
    XGBoost model for machine failure prediction.
    
    Attributes:
        model: Fitted xgboost.XGBClassifier instance
        scaler: Fitted StandardScaler for feature normalization
        imputer: Fitted SimpleImputer for missing value handling
        feature_names: List of feature names
        best_params: Best hyperparameters found during tuning
        metrics: Dictionary of evaluation metrics
    """
    
    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        scale_pos_weight: float = 1.0,
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            random_state: Random seed
            max_depth: Tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            scale_pos_weight: Weight for positive class (for imbalance)
        """
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        
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
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        self.best_params = {}
        self.metrics = {}
        self._is_fitted = False
    
    def preprocess_features(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """Preprocess features: impute and standardize."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        X_processed = X.copy()
        
        # Impute
        if fit:
            X_processed[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
        else:
            X_processed[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Standardize
        if fit:
            X_processed[numeric_cols] = self.scaler.fit_transform(X_processed[numeric_cols])
        else:
            X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])
        
        # One-hot encode
        if categorical_cols:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        return X_processed
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        param_grid: Dict = None,
    ) -> Dict:
        """
        Grid search for best hyperparameters.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
            param_grid: Parameter grid for search
        
        Returns:
            Best hyperparameters
        """
        X_train_scaled = self.preprocess_features(X_train, fit=True)
        X_val_scaled = self.preprocess_features(X_val, fit=False)
        
        if param_grid is None:
            param_grid = {
                'max_depth': [5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
            }
        
        best_pr_auc = 0
        best_params = {}
        
        for max_depth in param_grid.get('max_depth', [6]):
            for lr in param_grid.get('learning_rate', [0.05]):
                for n_est in param_grid.get('n_estimators', [100]):
                    
                    params = {
                        'max_depth': max_depth,
                        'learning_rate': lr,
                        'n_estimators': n_est,
                        'scale_pos_weight': self.scale_pos_weight,
                        'random_state': self.random_state,
                        'objective': 'binary:logistic',
                        'eval_metric': 'aucpr',
                        'verbosity': 0,
                    }
                    
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        early_stopping_rounds=10,
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
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train the model."""
        X_scaled = self.preprocess_features(X_train, fit=True)
        self.feature_names = X_scaled.columns.tolist()
        self.model.fit(X_scaled, y_train, verbose=False)
        self._is_fitted = True
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_scaled = self.preprocess_features(X_test, fit=False)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = self.model.predict(X_scaled)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        test_metrics = {
            'test_pr_auc': float(pr_auc),
            'test_precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'test_recall': float(sensitivity),
            'test_specificity': float(specificity),
            'test_f1': float(f1_score(y_test, y_pred)),
            'test_tp': int(tp),
            'test_fp': int(fp),
            'test_tn': int(tn),
            'test_fn': int(fn),
        }
        
        self.metrics['test'] = test_metrics
        return test_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (gain-based)."""
        if not self._is_fitted or self.feature_names is None:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'gain': self.model.feature_importances_,
        }).sort_values('gain', ascending=False)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def save(self, model_dir: Path) -> None:
        """Save model artifacts."""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        with open(model_dir / "xgboost_main.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessors
        with open(model_dir / "xgboost_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / "xgboost_imputer.pkl", 'wb') as f:
            pickle.dump(self.imputer, f)
        
        # Save hyperparameters
        with open(model_dir / "xgboost_hyperparams.json", 'w') as f:
            json.dump(
                {k: v for k, v in self.best_params.items() 
                 if k not in ['objective', 'eval_metric', 'verbosity']},
                f, indent=2
            )
        
        # Save metrics
        with open(model_dir / "xgboost_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save feature importance
        self.get_feature_importance().to_csv(
            model_dir / "xgboost_feature_importance.csv", index=False
        )
    
    @staticmethod
    def load(model_dir: Path) -> 'MachineXGBoost':
        """Load a saved model."""
        model_dir = Path(model_dir)
        
        instance = MachineXGBoost()
        
        with open(model_dir / "xgboost_main.pkl", 'rb') as f:
            instance.model = pickle.load(f)
        
        with open(model_dir / "xgboost_scaler.pkl", 'rb') as f:
            instance.scaler = pickle.load(f)
        
        with open(model_dir / "xgboost_imputer.pkl", 'rb') as f:
            instance.imputer = pickle.load(f)
        
        with open(model_dir / "xgboost_hyperparams.json", 'r') as f:
            instance.best_params = json.load(f)
        
        with open(model_dir / "xgboost_metrics.json", 'r') as f:
            instance.metrics = json.load(f)
        
        instance._is_fitted = True
        return instance
