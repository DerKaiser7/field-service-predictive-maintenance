"""
Logistic Regression Model for Predictive Maintenance

Class-based wrapper around scikit-learn's LogisticRegression that provides:
- Training with cross-validation
- Feature preprocessing (imputation, standardization)
- Evaluation metrics (PR-AUC, confusion matrix, etc.)
- Model serialization and feature importance
"""

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


class MachineLogisticRegression:
    """
    Logistic Regression model for machine failure prediction.
    
    Attributes:
        model: Fitted scikit-learn LogisticRegression instance
        scaler: Fitted StandardScaler for feature normalization
        imputer: Fitted SimpleImputer for missing value handling
        feature_names: List of feature names (for interpretation)
        metrics: Dictionary of evaluation metrics (CV, test)
    """
    
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        """
        Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for optimization
        """
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver='lbfgs',
            class_weight='balanced'
        )
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        self.metrics = {}
        self._is_fitted = False
    
    def preprocess_features(
        self, 
        X: pd.DataFrame, 
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Preprocess features: impute missing values and standardize.
        
        Args:
            X: Feature matrix
            fit: If True, fit the transformers on this data
        
        Returns:
            Preprocessed feature matrix
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        X_processed = X.copy()
        
        # Impute missing values
        if fit:
            X_processed[numeric_cols] = np.asarray(self.imputer.fit_transform(X[numeric_cols]))  # type: ignore[index]
        else:
            X_processed[numeric_cols] = np.asarray(self.imputer.transform(X[numeric_cols]))  # type: ignore[index]

        # Standardize numeric features
        if fit:
            X_processed[numeric_cols] = np.asarray(self.scaler.fit_transform(X_processed[numeric_cols]))  # type: ignore[index]
        else:
            X_processed[numeric_cols] = np.asarray(self.scaler.transform(X_processed[numeric_cols]))  # type: ignore[index]
        
        # One-hot encode categorical features
        if categorical_cols:
            X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        
        return X_processed
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of folds
        
        Returns:
            Dictionary with CV metrics (PR-AUC scores, mean, std)
        """
        X_scaled = self.preprocess_features(X, fit=True)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=cv,
            scoring='average_precision',  # PR-AUC
            n_jobs=-1
        )
        
        cv_metrics = {
            'cv_scores': cv_scores.tolist(),
            'mean_pr_auc': float(cv_scores.mean()),
            'std_pr_auc': float(cv_scores.std()),
        }
        
        self.metrics['cross_validation'] = cv_metrics
        return cv_metrics
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> None:
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        X_scaled = self.preprocess_features(X_train, fit=True)
        self.feature_names = X_scaled.columns.tolist()
        self.model.fit(X_scaled, y_train)
        self._is_fitted = True
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary with test metrics
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_scaled = self.preprocess_features(X_test, fit=False)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = self.model.predict(X_scaled)
        
        # Compute PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Confusion matrix metrics
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
        """
        Get feature importance (coefficients).
        
        Returns:
            DataFrame with features and their coefficients
        """
        if not self._is_fitted or self.feature_names is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
        }).sort_values('coefficient', ascending=False)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Binary predictions (0 or 1)
        """
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for both classes.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probability matrix (n_samples, 2)
        """
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def save(self, model_dir: Path) -> None:
        """
        Save model artifacts to disk.
        
        Args:
            model_dir: Directory to save artifacts
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        with open(model_dir / "baseline_lr.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessors
        with open(model_dir / "baseline_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / "baseline_imputer.pkl", 'wb') as f:
            pickle.dump(self.imputer, f)
        
        # Save metrics
        with open(model_dir / "baseline_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save feature importance
        self.get_feature_importance().to_csv(
            model_dir / "baseline_feature_importance.csv", index=False
        )
    
    @staticmethod
    def load(model_dir: Path) -> 'MachineLogisticRegression':
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory containing saved artifacts
        
        Returns:
            Loaded MachineLogisticRegression instance
        """
        model_dir = Path(model_dir)
        
        instance = MachineLogisticRegression()
        
        with open(model_dir / "baseline_lr.pkl", 'rb') as f:
            instance.model = pickle.load(f)
        
        with open(model_dir / "baseline_scaler.pkl", 'rb') as f:
            instance.scaler = pickle.load(f)
        
        with open(model_dir / "baseline_imputer.pkl", 'rb') as f:
            instance.imputer = pickle.load(f)
        
        with open(model_dir / "baseline_metrics.json", 'r') as f:
            instance.metrics = json.load(f)
        
        instance._is_fitted = True
        return instance