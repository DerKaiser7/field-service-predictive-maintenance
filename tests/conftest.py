"""Shared fixtures for the model/API test suite.

These tests exercise the model classes directly rather than hitting
PostgreSQL — they synthesize a small feature frame shaped exactly like
the real `model_input_features` table (same columns MachineLogisticRegression
and MachineXGBoost expect) so the pipeline can be fit/evaluated without a
database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

NUMERIC_COLS = [
    "age",
    "voltage_mean_3h", "voltage_std_3h", "voltage_min_3h", "voltage_max_3h",
    "rotation_mean_3h", "rotation_std_3h", "rotation_min_3h", "rotation_max_3h",
    "pressure_mean_3h", "pressure_std_3h", "pressure_min_3h", "pressure_max_3h",
    "vibration_mean_3h", "vibration_std_3h", "vibration_min_3h", "vibration_max_3h",
    "voltage_mean_12h", "voltage_std_12h",
    "rotation_mean_12h", "rotation_std_12h",
    "pressure_mean_12h", "pressure_std_12h",
    "vibration_mean_12h", "vibration_std_12h",
    "voltage_mean_24h", "voltage_std_24h",
    "rotation_mean_24h", "rotation_std_24h",
    "pressure_mean_24h", "pressure_std_24h",
    "vibration_mean_24h", "vibration_std_24h",
    "error_count_24h", "hours_since_last_error", "distinct_error_types",
    "days_since_last_maintenance", "component_diversity",
    "total_prior_failures", "days_since_last_failure", "distinct_failure_types",
]
CATEGORICAL_COLS = ["model", "age_category"]
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def synthesize_features(n: int, seed: int, positive_rate: float = 0.15) -> tuple[pd.DataFrame, pd.Series]:
    """Build an (X, y) pair shaped like `model_input_features`.

    The label is deterministically correlated with `vibration_mean_24h`
    and `error_count_24h` so model classes have real signal to fit against
    (a constant/random label would make PR-AUC undefined or degenerate).
    """
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {"age": rng.integers(1, 21, size=n).astype(float)}
    for sensor, base in [("voltage", 170.0), ("rotation", 450.0), ("pressure", 100.0), ("vibration", 40.0)]:
        for window in ("3h", "12h", "24h"):
            data[f"{sensor}_mean_{window}"] = rng.normal(base, base * 0.02, size=n)
            data[f"{sensor}_std_{window}"] = np.abs(rng.normal(2.0, 0.5, size=n))
        data[f"{sensor}_min_3h"] = data[f"{sensor}_mean_3h"] - np.abs(rng.normal(3.0, 1.0, size=n))
        data[f"{sensor}_max_3h"] = data[f"{sensor}_mean_3h"] + np.abs(rng.normal(3.0, 1.0, size=n))

    data["error_count_24h"] = rng.poisson(1.0, size=n).astype(float)
    data["hours_since_last_error"] = rng.uniform(0, 400, size=n)
    data["distinct_error_types"] = np.clip(data["error_count_24h"], 0, 3)
    data["days_since_last_maintenance"] = rng.uniform(0, 200, size=n)
    data["component_diversity"] = rng.integers(1, 4, size=n).astype(float)
    data["total_prior_failures"] = rng.poisson(1.0, size=n).astype(float)
    data["days_since_last_failure"] = rng.uniform(0, 300, size=n)
    data["distinct_failure_types"] = np.clip(data["total_prior_failures"], 0, 3)

    X = pd.DataFrame(data)
    X["model"] = rng.choice(["model1", "model2", "model3", "model4"], size=n)
    X["age_category"] = pd.cut(
        X["age"], bins=[0, 3, 7, 21], labels=["new", "mid_life", "aged"]
    ).astype(str)

    # Deterministic-ish signal: elevated vibration + errors raise failure odds.
    risk_score = (
        0.08 * (X["vibration_mean_24h"] - 40.0)
        + 0.6 * X["error_count_24h"]
        - 0.01 * X["days_since_last_maintenance"]
    )
    prob = 1 / (1 + np.exp(-(risk_score - risk_score.quantile(1 - positive_rate))))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="label")

    # Guarantee both classes are present regardless of the random draw above.
    if y.sum() == 0:
        y.iloc[0] = 1
    if y.sum() == len(y):
        y.iloc[0] = 0

    return X[ALL_FEATURE_COLS], y


@pytest.fixture
def training_frame() -> tuple[pd.DataFrame, pd.Series]:
    return synthesize_features(n=300, seed=42, positive_rate=0.2)


@pytest.fixture
def train_val_split(training_frame):
    X, y = training_frame
    split = int(len(X) * 0.7)
    return X.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True), \
        y.iloc[:split].reset_index(drop=True), y.iloc[split:].reset_index(drop=True)
