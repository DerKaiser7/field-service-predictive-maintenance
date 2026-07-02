import numpy as np
import pandas as pd
import pytest

from src.models.MachineLogisticRegression import MachineLogisticRegression


def test_predict_before_fit_raises(training_frame):
    X, _ = training_frame
    model = MachineLogisticRegression()
    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(X)


def test_preprocess_features_imputes_missing_numeric(training_frame):
    X, y = training_frame
    model = MachineLogisticRegression()
    model.fit(X, y)

    X_with_gap = X.copy()
    X_with_gap.loc[0, "voltage_mean_24h"] = np.nan

    processed = model.preprocess_features(X_with_gap, fit=False)
    assert not processed.filter(like="voltage_mean_24h").isna().any().any()


def test_preprocess_features_one_hot_aligns_to_training_columns(training_frame):
    X, y = training_frame
    model = MachineLogisticRegression()
    model.fit(X, y)

    # A category never seen at fit time must not blow up inference — it
    # should simply fail to activate any of the fitted dummy columns.
    X_unseen = X.copy()
    X_unseen.loc[0, "model"] = "model_never_seen"
    processed = model.preprocess_features(X_unseen, fit=False)

    assert list(processed.columns) == model._ohe_feature_names()


def test_preprocess_features_scales_numeric_columns(training_frame):
    X, y = training_frame
    model = MachineLogisticRegression(random_state=0)
    model.fit(X, y)

    processed = model.preprocess_features(X, fit=False)
    numeric_processed = processed[[c for c in processed.columns if c in X.select_dtypes("number").columns]]
    # StandardScaler output should be roughly zero-mean per column.
    assert numeric_processed.mean().abs().max() < 1.0
