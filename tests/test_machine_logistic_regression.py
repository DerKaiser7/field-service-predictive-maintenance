import pandas as pd

from src.models.MachineLogisticRegression import MachineLogisticRegression


def test_fit_predict_roundtrip(training_frame):
    X, y = training_frame
    model = MachineLogisticRegression()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()

    preds = model.predict(X)
    assert set(preds.tolist()) <= {0, 1}


def test_get_feature_importance_matches_coefficients(training_frame):
    X, y = training_frame
    model = MachineLogisticRegression()
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert len(importance) == len(model.feature_names)
    assert list(importance.columns) == ["feature", "coefficient"]
    # Sorted descending by coefficient.
    assert importance["coefficient"].is_monotonic_decreasing


def test_evaluate_returns_expected_keys(train_val_split):
    X_train, X_test, y_train, y_test = train_val_split
    model = MachineLogisticRegression()
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    expected_keys = {
        "test_pr_auc", "test_precision", "test_recall", "test_specificity",
        "test_f1", "test_tp", "test_fp", "test_tn", "test_fn",
    }
    assert expected_keys <= metrics.keys()
    assert 0.0 <= metrics["test_pr_auc"] <= 1.0


def test_save_load_roundtrip_predictions_match(training_frame, tmp_path):
    X, y = training_frame
    model = MachineLogisticRegression()
    model.fit(X, y)
    model.save(tmp_path)

    reloaded = MachineLogisticRegression.load(tmp_path)
    pd.testing.assert_frame_equal(
        pd.DataFrame(model.predict_proba(X)),
        pd.DataFrame(reloaded.predict_proba(X)),
    )
