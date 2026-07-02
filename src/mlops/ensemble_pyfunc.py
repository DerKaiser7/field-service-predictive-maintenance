"""MLflow pyfunc wrapper for EnsembleModel.

The ensemble is three artifacts (LR base model, XGBoost base model,
meta-learner + threshold) that only make sense together. Registering just
the meta-learner in MLflow would be misleading — it's not a usable model on
its own. This wrapper lets MLflow log/register/serve the whole thing as one
unit by pointing it at the same model_artifacts/ directory EnsembleModel.load
already reads.
"""

from pathlib import Path

import mlflow.pyfunc

from src.models.EnsembleModel import EnsembleModel


class EnsemblePyfuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.ensemble = EnsembleModel.load(Path(context.artifacts["model_artifacts"]))

    def predict(self, context, model_input):
        return self.ensemble.predict_proba(model_input)
