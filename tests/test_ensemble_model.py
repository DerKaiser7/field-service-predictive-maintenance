import numpy as np
import pytest

from src.models.EnsembleModel import EnsembleModel
from src.models.MachineLogisticRegression import MachineLogisticRegression
from src.models.MachineXGBoost import MachineXGBoost


def _fitted_ensemble(train_val_split) -> tuple[EnsembleModel, "pd.DataFrame", "pd.Series"]:
    X_train, X_val, y_train, y_val = train_val_split

    lr = MachineLogisticRegression()
    lr.fit(X_train, y_train)
    xgb = MachineXGBoost(n_estimators=20, max_depth=3)
    xgb.fit(X_train, y_train)

    ensemble = EnsembleModel()
    ensemble.lr = lr
    ensemble.xgb = xgb
    ensemble.fit_meta_learner(X_val, y_val)
    ensemble.optimise_threshold(X_val, y_val)
    return ensemble, X_val, y_val


def test_predict_proba_without_meta_learner_raises():
    ensemble = EnsembleModel()
    ensemble.lr = None  # type: ignore[assignment]
    ensemble.xgb = None  # type: ignore[assignment]
    ensemble.meta_learner = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="not fitted"):
        ensemble.predict_proba(None)


def test_threshold_optimisation_stays_in_bounds(train_val_split):
    ensemble, X_val, y_val = _fitted_ensemble(train_val_split)
    assert 0.0 < ensemble.threshold < 1.0


def test_predict_uses_optimised_threshold(train_val_split):
    ensemble, X_val, _ = _fitted_ensemble(train_val_split)
    proba = ensemble.predict_proba(X_val)
    preds = ensemble.predict(X_val)
    np.testing.assert_array_equal(preds, (proba >= ensemble.threshold).astype(int))


def test_evaluate_business_cost_matches_confusion_matrix(train_val_split):
    ensemble, X_val, y_val = _fitted_ensemble(train_val_split)
    metrics = ensemble.evaluate(X_val, y_val)

    expected_cost = metrics["fn"] * EnsembleModel.COST_FN + metrics["fp"] * EnsembleModel.COST_FP
    assert metrics["business_cost"] == expected_cost
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_val)


def test_save_load_roundtrip_predictions_match(train_val_split, tmp_path):
    ensemble, X_val, y_val = _fitted_ensemble(train_val_split)
    ensemble.evaluate(X_val, y_val)

    ensemble.lr.save(tmp_path)
    ensemble.xgb.save(tmp_path)
    ensemble.save(tmp_path)

    reloaded = EnsembleModel.load(tmp_path)
    np.testing.assert_allclose(
        ensemble.predict_proba(X_val), reloaded.predict_proba(X_val), rtol=1e-6,
    )
    assert reloaded.threshold == ensemble.threshold
