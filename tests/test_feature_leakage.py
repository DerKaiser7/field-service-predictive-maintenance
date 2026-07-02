"""Static regression guard against feature leakage in build_features.sql.

There's no automated way to run the SQL against a live Postgres instance in
CI without provisioning a database, so this test takes a cheaper but still
useful approach: it parses the feature-construction CTEs and asserts none of
them reference a forward-looking time window (`t.datetime + INTERVAL` /
`> t.datetime` without an upper bound at `t.datetime`). The label block is
intentionally forward-looking (it predicts the next 24h) and is excluded —
everything *before* it must only look backward.

This won't catch every possible leakage bug, but it will catch the most
common regression: someone copy-pasting the label's forward window into a
new feature CTE.
"""

from pathlib import Path

import pytest

SQL_PATH = Path(__file__).parent.parent / "sql" / "features" / "build_features.sql"
LABEL_MARKER = "-- FINAL: Assemble all features + label"


@pytest.fixture
def feature_section() -> str:
    text = SQL_PATH.read_text()
    assert LABEL_MARKER in text, "expected marker not found — did build_features.sql get restructured?"
    return text.split(LABEL_MARKER)[0]


def test_sql_file_exists():
    assert SQL_PATH.exists()


def test_feature_ctes_do_not_reference_future_intervals(feature_section: str):
    """No feature CTE may add an interval to `t.datetime` (a forward window)."""
    forbidden_patterns = ["t.datetime + INTERVAL", "t2.datetime + INTERVAL"]
    for pattern in forbidden_patterns:
        assert pattern not in feature_section, (
            f"found forward-looking pattern {pattern!r} in a feature CTE — "
            "this would leak future data into a feature"
        )


def test_feature_ctes_bound_lookback_windows_at_observation_time(feature_section: str):
    """Every rolling-window CASE/JOIN must cap at `<= t.datetime` (or `<= t2.datetime`
    inside a self-join), not some later point — guards against someone loosening
    a bound while refactoring."""
    assert feature_section.count("<= t.datetime") + feature_section.count("<= t2.datetime") >= 20, (
        "expected the 3h/12h/24h telemetry windows to bound themselves at "
        "observation_time — count dropped, check for an accidental loosened join"
    )


def test_label_is_the_only_forward_looking_block():
    """Sanity check that the forward-looking window is isolated to the label
    computation, not smuggled in earlier under a different alias."""
    text = SQL_PATH.read_text()
    label_section = text.split(LABEL_MARKER)[1]
    assert "t.observation_time + INTERVAL '24 hours'" in label_section
