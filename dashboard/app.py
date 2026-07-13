"""
Predictive Maintenance — Azure PdM Dataset
Interactive demo: live ensemble predictions with SHAP explanations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.models.EnsembleModel import EnsembleModel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance — Azure PdM",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ARTIFACTS = ROOT / "model_artifacts"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ensemble model…")
def load_model():
    ensemble = EnsembleModel.load(ARTIFACTS)
    explainer = shap.TreeExplainer(ensemble.xgb.model)
    numeric_cols = list(ensemble.lr.imputer.feature_names_in_)
    xgb_features = ensemble.xgb.model.get_booster().feature_names or []
    return ensemble, explainer, numeric_cols, xgb_features

ensemble, explainer, NUMERIC_COLS, XGB_FEATURES = load_model()
THRESHOLD = ensemble.threshold

# ---------------------------------------------------------------------------
# Machine catalogue with realistic per-machine sensor profiles.
#
# Each machine has feature values derived from its age, model type, and
# maintenance history — producing a spread of LOW / MEDIUM / HIGH predictions
# that demonstrates the model working across the full risk spectrum.
# Values loosely reflect dataset statistics for each machine class.
# ---------------------------------------------------------------------------
MACHINES = [
    # id   model      age  age_cat    fails  errs  hrs_err  maint  vibr_24  volt_24  rot_24  pres_24  vibr_std
    ("1",  "model3",  18,  "aged",    3,     0,    96.0,    6.0,   40.1,    170.2,   449.0,  100.1,   1.8),
    ("2",  "model4",  7,   "mid_life",1,     1,    36.0,    28.0,  41.5,    169.4,   446.8,  99.5,    2.4),
    ("3",  "model3",  12,  "aged",    2,     2,    18.0,    52.0,  45.2,    167.1,   441.3,  97.8,    3.6),
    ("4",  "model1",  5,   "mid_life",0,     0,    180.0,   4.0,   38.8,    171.0,   451.2,  100.8,   1.5),
    ("5",  "model3",  20,  "aged",    4,     4,    2.5,     180.0, 53.1,    162.4,   401.5,  92.1,    6.2),
    ("6",  "model4",  8,   "mid_life",1,     1,    22.0,    35.0,  42.3,    168.8,   444.7,  99.2,    2.8),
    ("7",  "model1",  2,   "new",     0,     0,    240.0,   3.0,   38.4,    171.3,   452.1,  101.0,   1.3),
    ("8",  "model2",  9,   "aged",    2,     2,    14.0,    61.0,  46.8,    166.5,   439.2,  97.1,    4.1),
    ("9",  "model3",  15,  "aged",    3,     4,    3.0,     155.0, 51.8,    163.2,   404.1,  93.0,    5.8),
    ("10", "model1",  4,   "mid_life",0,     0,    300.0,   5.0,   39.0,    170.8,   450.5,  100.5,   1.6),
    ("11", "model4",  11,  "aged",    2,     3,    8.0,     90.0,  48.4,    165.0,   436.8,  96.3,    4.9),
    ("12", "model2",  3,   "new",     0,     0,    360.0,   2.0,   38.6,    171.1,   451.8,  100.9,   1.4),
    ("13", "model3",  16,  "aged",    4,     4,    1.5,     190.0, 54.2,    161.8,   399.3,  91.4,    6.8),
    ("14", "model1",  6,   "mid_life",1,     1,    30.0,    32.0,  41.0,    169.7,   447.5,  99.8,    2.2),
    ("15", "model4",  14,  "aged",    3,     3,    6.0,     120.0, 49.7,    164.3,   433.5,  95.7,    5.3),
    ("16", "model2",  1,   "new",     0,     0,    480.0,   1.0,   38.2,    171.5,   452.8,  101.2,   1.2),
    ("17", "model3",  10,  "aged",    2,     2,    20.0,    74.0,  44.6,    167.6,   442.0,  98.2,    3.3),
    ("18", "model1",  7,   "mid_life",1,     1,    42.0,    24.0,  40.8,    169.9,   448.2,  100.0,   2.1),
    ("19", "model4",  19,  "aged",    5,     4,    1.8,     210.0, 55.3,    161.2,   397.8,  90.8,    7.1),
    ("20", "model2",  5,   "mid_life",0,     0,    150.0,   10.0,  39.3,    170.5,   450.0,  100.3,   1.7),
]

COLS = ["id","model","age","age_category","prior_failures",
        "error_count_24h","hours_since_last_error","days_since_last_maintenance",
        "vibration_mean_24h","voltage_mean_24h","rotation_mean_24h","pressure_mean_24h",
        "vibration_std_3h"]
MACHINES_DF = pd.DataFrame(MACHINES, columns=COLS)
# Pre-built for the selectbox format_func — avoids .loc[mask, col] which
# collapses to a scalar union type Pylance can't resolve .iloc/.values on.
_machine_labels: dict[str, str] = {
    str(row["id"]): f"Machine {str(row['id']):>2}  ·  {row['model']}  ·  age {int(row['age'])}y"
    for _, row in MACHINES_DF.iterrows()
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_feature_row(m: pd.Series) -> pd.DataFrame:
    """Build a model-ready single-row DataFrame from a machine profile row."""
    v   = float(m["voltage_mean_24h"])
    r   = float(m["rotation_mean_24h"])
    p   = float(m["pressure_mean_24h"])
    vb  = float(m["vibration_mean_24h"])
    vs  = float(m["vibration_std_3h"])
    errs = int(m["error_count_24h"])

    features = {
        "model":        m["model"],
        "age_category": m["age_category"],
        "age":          int(m["age"]),
        # 3h windows (short-term snapshot; std elevated for machines under stress)
        "voltage_mean_3h":    v - 0.5,   "voltage_std_3h":   2.1 + vs * 0.3,
        "voltage_min_3h":     v - 3.0,   "voltage_max_3h":   v + 2.5,
        "rotation_mean_3h":   r - 1.0,   "rotation_std_3h":  4.8 + vs * 1.5,
        "rotation_min_3h":    r - 6.0,   "rotation_max_3h":  r + 5.0,
        "pressure_mean_3h":   p - 0.2,   "pressure_std_3h":  1.4 + vs * 0.2,
        "pressure_min_3h":    p - 2.0,   "pressure_max_3h":  p + 2.0,
        "vibration_mean_3h":  vb - 0.5,  "vibration_std_3h": vs,
        "vibration_min_3h":   vb - vs,   "vibration_max_3h": vb + vs * 1.2,
        # 12h windows
        "voltage_mean_12h":   v - 0.2,   "voltage_std_12h":  2.2 + vs * 0.3,
        "rotation_mean_12h":  r - 0.5,   "rotation_std_12h": 5.0 + vs * 1.2,
        "pressure_mean_12h":  p + 0.1,   "pressure_std_12h": 1.5 + vs * 0.2,
        "vibration_mean_12h": vb + 0.2,  "vibration_std_12h":vs * 1.1,
        # 24h windows (stored directly from profile)
        "voltage_mean_24h":   v,          "voltage_std_24h":  2.0 + vs * 0.3,
        "rotation_mean_24h":  r,          "rotation_std_24h": 4.9 + vs * 1.0,
        "pressure_mean_24h":  p,          "pressure_std_24h": 1.4 + vs * 0.2,
        "vibration_mean_24h": vb,         "vibration_std_24h":vs * 1.05,
        # Error / maintenance / failure history
        "error_count_24h":           errs,
        "hours_since_last_error":    float(m["hours_since_last_error"]),
        "distinct_error_types":      min(errs, 3),
        "days_since_last_maintenance": float(m["days_since_last_maintenance"]),
        "component_diversity":       max(1, errs),
        "total_prior_failures":      int(m["prior_failures"]),
        "days_since_last_failure":   np.nan if m["prior_failures"] == 0 else 30.0 * m["age"] / max(m["prior_failures"], 1),
        "distinct_failure_types":    min(int(m["prior_failures"]), 3),
    }

    row = {k: (np.nan if v is None else v) for k, v in features.items()}
    ordered = {c: row[c] for c in NUMERIC_COLS if c in row}
    ordered["model"] = features["model"]
    ordered["age_category"] = features["age_category"]
    return pd.DataFrame([ordered])


def run_prediction(X: pd.DataFrame):
    prob = float(ensemble.predict_proba(X)[0])
    label = int(prob >= THRESHOLD)
    risk = "HIGH" if prob >= THRESHOLD else ("MEDIUM" if prob >= THRESHOLD * 0.5 else "LOW")

    X_proc = ensemble.xgb.preprocess_features(X, fit=False)
    for col in XGB_FEATURES:
        if col not in X_proc.columns:
            X_proc[col] = 0.0
    X_proc = X_proc[XGB_FEATURES]
    sv = explainer(X_proc)
    shap_vals = sv.values[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:7]
    top_features = [(XGB_FEATURES[i], float(shap_vals[i])) for i in top_idx]

    return prob, label, risk, top_features


RISK_COLOR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}


def gauge_fig(prob: float, risk: str) -> go.Figure:
    color = RISK_COLOR[risk]
    pct = round(prob * 100, 1)
    threshold_pct = round(THRESHOLD * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={
            "suffix": "%",
            "font": {"size": 48, "color": color},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#4a5278",
                "tickfont": {"size": 10, "color": "#8b92b8"},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold_pct * 0.5],  "color": "rgba(34,197,94,0.18)"},
                {"range": [threshold_pct * 0.5, threshold_pct], "color": "rgba(245,158,11,0.18)"},
                {"range": [threshold_pct, 100],       "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": threshold_pct,
            },
        },
    ))

    # Annotate the threshold
    fig.add_annotation(
        x=0.5, y=-0.08, xref="paper", yref="paper",
        text=f"Decision threshold: {threshold_pct}%",
        showarrow=False,
        font={"size": 11, "color": "#8b92b8"},
    )

    fig.update_layout(
        height=270,
        margin=dict(t=10, b=30, l=30, r=30),
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font_color="#e8eaf0",
    )
    return fig


def shap_fig(top_features: list) -> go.Figure:
    names  = [f.replace("_", " ") for f, _ in reversed(top_features)]
    values = [v for _, v in reversed(top_features)]
    colors = [RISK_COLOR["HIGH"] if v > 0 else RISK_COLOR["LOW"] for v in values]

    max_abs = max(abs(v) for v in values) if values else 1.0
    x_pad   = max_abs * 0.25
    x_range = [-(max_abs + x_pad), max_abs + x_pad]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont={"size": 10, "color": "#8b92b8"},
    ))
    fig.add_vline(x=0, line_color="#4a5278", line_width=1)
    fig.update_layout(
        height=290,
        margin=dict(t=10, b=10, l=10, r=70),
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        xaxis={"range": x_range, "gridcolor": "#2d3148", "zerolinecolor": "#2d3148", "tickfont": {"size": 10}, "zeroline": False},
        yaxis={"tickfont": {"size": 11}, "gridcolor": "rgba(0,0,0,0)", "automargin": True},
        font_color="#e8eaf0",
    )
    return fig


def telemetry_fig(m: pd.Series) -> go.Figure:
    seed = int(m["id"])
    vb   = float(m["vibration_mean_24h"])
    v    = float(m["voltage_mean_24h"])
    r    = float(m["rotation_mean_24h"])
    p    = float(m["pressure_mean_24h"])
    vs   = float(m["vibration_std_3h"])
    hours = [f"{h:02d}:00" for h in range(24)]

    def wave(base, amp, i):
        return round(base + amp * np.sin(seed * 0.5 + i * 0.8) + amp * 0.3 * np.cos(seed + i * 1.6), 2)

    traces = [
        ("Voltage",   [wave(v,  vs * 0.5 + 1.0, i) for i in range(24)], "#4f7cff"),
        ("Rotation",  [wave(r,  vs * 2.5 + 2.5, i) for i in range(24)], "#22c55e"),
        ("Pressure",  [wave(p,  vs * 0.4 + 0.8, i) for i in range(24)], "#f59e0b"),
        ("Vibration", [wave(vb, vs * 0.8 + 0.5, i) for i in range(24)], "#ef4444"),
    ]

    fig = go.Figure()
    for name, vals, color in traces:
        fig.add_trace(go.Scatter(
            x=hours, y=vals, name=name,
            line={"color": color, "width": 1.8},
            mode="lines",
        ))
    fig.update_layout(
        height=230,
        margin=dict(t=10, b=40, l=50, r=20),
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        xaxis={"gridcolor": "#2d3148", "tickfont": {"size": 10}, "dtick": 4},
        yaxis={"gridcolor": "#2d3148", "tickfont": {"size": 10}},
        legend={"font": {"size": 10}, "orientation": "h", "y": -0.3},
        font_color="#e8eaf0",
    )
    return fig

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1 { font-size: 1.65rem !important; margin-bottom: 0.25rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    [data-testid="metric-container"] { background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; padding: 12px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Predictive Maintenance — Azure PdM Dataset")
st.caption(
    "Binary ensemble classifier · predicts machine failure within 24 hours · "
    "PR-AUC 0.9992 · 876,100 training observations"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_demo, tab_overview, tab_methodology = st.tabs(["Live Demo", "Overview", "Methodology"])

# ============================================================
# TAB 1 — LIVE DEMO
# ============================================================
with tab_demo:
    st.subheader("Machine Failure Risk Prediction")
    st.caption(
        "Select a machine. The ensemble model runs a live prediction and returns "
        "the failure probability for the next 24 hours with SHAP feature attributions. "
        "Sensor readings reflect each machine's age, model type, and service history."
    )

    col_sel, col_result = st.columns([1, 2.4], gap="large")

    with col_sel:
        machine_id = st.selectbox(
            "Select machine",
            options=[m[0] for m in MACHINES],
            format_func=lambda mid: _machine_labels[mid],
        )
        m = MACHINES_DF[MACHINES_DF["id"] == machine_id].iloc[0]

        st.markdown("---")
        st.markdown(f"**Model:** {m['model']}")
        st.markdown(f"**Age:** {m['age']} years")
        st.markdown(f"**Prior failures:** {int(m['prior_failures'])}")
        st.markdown(f"**Errors (last 24h):** {int(m['error_count_24h'])}")
        st.markdown(f"**Hours since last error:** {m['hours_since_last_error']:.0f}h")
        st.markdown(f"**Days since maintenance:** {m['days_since_last_maintenance']:.0f}d")

    with col_result:
        X = build_feature_row(m)
        prob, label, risk, top_features = run_prediction(X)
        color = RISK_COLOR[risk]

        c_gauge, c_info = st.columns([1.1, 1])

        with c_gauge:
            st.plotly_chart(gauge_fig(prob, risk), use_container_width=True, config={"displayModeBar": False})

        with c_info:
            st.markdown("<br>", unsafe_allow_html=True)
            badge = (
                f'<div style="display:inline-block;background:{color}22;'
                f'color:{color};padding:6px 18px;border-radius:20px;'
                f'font-size:1.05rem;font-weight:700;border:1px solid {color}55;">'
                f'{risk} RISK</div>'
            )
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Failure probability:** `{prob:.2%}`")
            st.markdown(f"**Decision:** {'⚠️ FAILURE PREDICTED' if label else '✅ NO FAILURE'}")
            st.markdown(f"**Threshold:** `{THRESHOLD:.3f}` (cost-optimised)")
            st.markdown(f"**Model:** v1.0 ensemble (LR + XGBoost)")

        st.markdown("**Top 7 SHAP Feature Attributions**")
        st.caption(
            "Red bars push probability up (increase failure risk).  "
            "Green bars push probability down.  "
            "Values are SHAP contributions in log-odds space."
        )
        st.plotly_chart(shap_fig(top_features), use_container_width=True, config={"displayModeBar": False})

    st.markdown("**24-Hour Sensor Telemetry — Machine {}**".format(machine_id))
    st.caption("Simulated from the machine's operational profile. Amplitude reflects sensor variability observed in the dataset.")
    st.plotly_chart(telemetry_fig(m), use_container_width=True, config={"displayModeBar": False})

# ============================================================
# TAB 2 — OVERVIEW
# ============================================================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC",    "0.9992", "Ensemble model")
    c2.metric("Recall",    "99.85%", "Failures caught")
    c3.metric("Precision", "98.88%", "Alerts correct")
    c4.metric("Business cost", "64", "5×FN + 1×FP")

    st.markdown("---")
    st.subheader("What it does")
    st.markdown(
        """
        Given hourly sensor readings for an industrial machine, this system predicts whether
        that machine will **fail within the next 24 hours** — giving maintenance teams time to
        intervene before an unplanned shutdown.

        **Dataset:** [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
        100 machines · 876,100 hourly observations · 1.96% positive class (17,184 failures)
        """
    )

    st.subheader("Pipeline")
    st.code(
        """Raw CSVs  →  PostgreSQL staging  →  Base tables
    ↓  sql/features/build_features.sql
model_input_features  (876,100 rows × 42 features + label)
    ├─  train_baseline.py   →  Logistic Regression  (PR-AUC 0.8285)
    ├─  train_xgboost.py    →  XGBoost             (PR-AUC 0.9999)
    └─  ensemble.py         →  Stacked ensemble    (PR-AUC 0.9992)
         ↓
    FastAPI  /predict  +  Streamlit demo""",
        language="text",
    )

    st.subheader("Model comparison")
    results = pd.DataFrame([
        {"Model": "Logistic Regression (baseline)", "PR-AUC": 0.8285, "Precision": 0.4735, "Recall": 0.9895, "F1": 0.6405},
        {"Model": "XGBoost",                        "PR-AUC": 0.9999, "Precision": 0.9840, "Recall": 1.0000, "F1": 0.9919},
        {"Model": "Ensemble (threshold=0.729) ✓",   "PR-AUC": 0.9992, "Precision": 0.9888, "Recall": 0.9985, "F1": 0.9936},
    ]).set_index("Model")
    st.dataframe(results.style.format("{:.4f}").highlight_max(color="#1a2a1a"), use_container_width=True)
    st.caption("60/20/20 stratified split · PR-AUC preferred over ROC-AUC for 1.96% class imbalance")

    st.subheader("Confusion matrix — held-out test set")
    cm = pd.DataFrame(
        {"Actual FAILURE": [3432, 5], "Actual NO FAILURE": [39, 171744]},
        index=["Predicted FAILURE", "Predicted NO FAILURE"],
    )
    st.dataframe(cm, use_container_width=True)
    st.caption("Business cost: 5 × 5 FN + 1 × 39 FP = **64** at threshold 0.729")

# ============================================================
# TAB 3 — METHODOLOGY
# ============================================================
with tab_methodology:
    st.subheader("Feature Engineering — 42 Features, Zero Leakage")
    st.markdown(
        "All features computed strictly from data at or before `observation_time`. "
        "Label (`failure_within_24h`) is computed separately after features are locked."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Telemetry rolling windows (32 features)**")
        st.markdown(
            "Self-join on the telemetry table for three lookback windows per sensor "
            "(voltage, rotation, pressure, vibration):"
        )
        st.markdown(
            "- **3h** — mean, std, min, max  *(transient spikes)*\n"
            "- **12h** — mean, std  *(medium-term trends)*\n"
            "- **24h** — mean, std  *(daily cycles, degradation)*"
        )
        st.code(
            """AVG(CASE
    WHEN t2.datetime > t.datetime - INTERVAL '3 hours'
     AND t2.datetime <= t.datetime
    THEN t2.volt
END) AS voltage_mean_3h""",
            language="sql",
        )

    with col_b:
        st.markdown("**Event & history features (10 features)**")
        st.markdown(
            "- `error_count_24h` · `hours_since_last_error` · `distinct_error_types`\n"
            "- `days_since_last_maintenance` · `component_diversity`\n"
            "- `total_prior_failures` · `days_since_last_failure` · `distinct_failure_types`\n"
            "- `model` · `age` · `age_category`"
        )
        st.markdown("**Class imbalance handling**")
        st.markdown(
            "- XGBoost: `scale_pos_weight = 49.98` (n_neg / n_pos)\n"
            "- Logistic Regression: `class_weight = 'balanced'`\n"
            "- No resampling — preserves original class distribution"
        )

    st.markdown("---")
    st.subheader("Ensemble Stacking")
    st.markdown(
        """
**Layer 1 — Base models** (trained on 60% training set):
- Logistic Regression with balanced class weights → captures linear signal, well-calibrated probabilities
- XGBoost with `scale_pos_weight` → captures non-linear sensor interactions

**Layer 2 — Meta-learner** (trained on 20% validation set predictions):
- Logistic Regression on `[lr_proba, xgb_proba]` stacked columns
- Learns optimal weighting of base models without seeing test data

**Threshold optimisation** (swept on validation set):
"""
    )
    st.code("cost = 5 × FN + 1 × FP    →    optimal threshold: 0.729", language="text")
    st.markdown(
        "Missed failures (FN) cost 5× more than false alarms (FP), "
        "reflecting real maintenance economics: unplanned shutdown >> unnecessary inspection."
    )

    st.markdown("---")
    st.subheader("SHAP Explainability")
    st.markdown(
        "Every prediction in the Live Demo runs `shap.TreeExplainer` on the XGBoost base model. "
        "SHAP values decompose the prediction into per-feature contributions that sum to the "
        "log-odds difference from the base rate:\n\n"
        "- **Positive SHAP** → feature increases failure probability\n"
        "- **Negative SHAP** → feature decreases failure probability\n\n"
        "This makes individual predictions auditable — "
        "a maintenance manager can see *exactly* which sensor reading or history event drove the alert."
    )
