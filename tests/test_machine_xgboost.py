import numpy as np

from src.models.MachineXGBoost import MachineXGBoost


def test_fit_predict_roundtrip(training_frame):
    X, y = training_frame
    model = MachineXGBoost(n_estimators=20, max_depth=3)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_ohe_feature_names_only_available_after_fit(training_frame):
    X, y = training_frame
    model = MachineXGBoost(n_estimators=20, max_depth=3)
    assert model._ohe_feature_names() is None

    model.fit(X, y)
    assert model._ohe_feature_names() == model.model.get_booster().feature_names


def test_get_feature_importance_shape(training_frame):
    X, y = training_frame
    model = MachineXGBoost(n_estimators=20, max_depth=3)
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert len(importance) == len(model.feature_names)
    assert list(importance.columns) == ["feature", "gain"]


def test_save_load_roundtrip_predictions_match(training_frame, tmp_path):
    X, y = training_frame
    model = MachineXGBoost(n_estimators=20, max_depth=3)
    model.fit(X, y)
    model.save(tmp_path)

    reloaded = MachineXGBoost.load(tmp_path)
    np.testing.assert_allclose(
        model.predict_proba(X), reloaded.predict_proba(X), rtol=1e-6,
    )
