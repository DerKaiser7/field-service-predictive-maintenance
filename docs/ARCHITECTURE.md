# Architecture

## Pipeline

```mermaid
flowchart TD
    A["Raw CSVs<br/>telemetry, errors, maintenance, failures, machines"] --> B["Staging Tables<br/>(PostgreSQL)"]
    B -->|"src/data_operations/promote_to_base.py"| C["Base Tables<br/>machines, telemetry, errors, maintenance, failures"]
    C -->|"sql/features/build_features.sql"| D["model_input_features<br/>876,100 rows × 42 features + label"]
    D --> E["Logistic Regression<br/>train_baseline.py"]
    D --> F["XGBoost<br/>train_xgboost.py"]
    E --> G["Stacked Ensemble + threshold optimisation<br/>ensemble.py"]
    F --> G
    G --> H["model_artifacts/<br/>committed to git"]
    H --> I["FastAPI Prediction Service<br/>src/api/main.py"]
    H --> J["Streamlit Demo<br/>dashboard/app.py"]
```

Every feature in `model_input_features` is computed strictly from data at or before `observation_time` — see [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) for the window definitions and [`tests/test_feature_leakage.py`](../tests/test_feature_leakage.py) for the automated guard against regressions.

## Deployment topology

Two independent runtimes consume `model_artifacts/`, neither depends on the other:

```mermaid
flowchart LR
    subgraph Local / Docker Compose
        PG["PostgreSQL<br/>(kaiser_postgres)"]
        API["FastAPI<br/>(kaiser_api, port 8000)"]
        API -->|"prediction_logs writes"| PG
    end

    subgraph Streamlit Community Cloud
        ST["Streamlit dashboard<br/>(dashboard/app.py)"]
    end

    MA["model_artifacts/<br/>(committed to git)"]
    MA --> API
    MA --> ST
```

The API's database dependency is limited to writing `prediction_logs` in a background task — model loading and inference do not require Postgres to be reachable. The Streamlit demo has no database dependency at all, which is why its "Live Demo" tab runs on 20 hardcoded machine profiles rather than live rows (Streamlit Community Cloud has no database access) — see the README's [Design Notes](../README.MD#design-notes).
