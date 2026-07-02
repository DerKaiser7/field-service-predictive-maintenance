"""Shared MLflow setup for the training scripts.

Points at the mlflow-server docker-compose service by default; override
MLFLOW_TRACKING_URI to point at a different tracking server (e.g. a local
`mlflow ui` instance) without touching the training scripts.
"""

import os

import mlflow

EXPERIMENT_NAME = "predictive-maintenance"


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
