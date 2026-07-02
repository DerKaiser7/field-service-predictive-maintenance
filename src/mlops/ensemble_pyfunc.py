"""MLflow pyfunc wrapper for EnsembleModel.

The ensemble is three artifacts (LR base model, XGBoost base model,
meta-learner + threshold) that only make sense together. Registering just
the meta-learner in MLflow would be misleading — it's not a usable model on
its own. This wrapper lets MLflow log/register/serve the whole thing as one
unit by pointing it at the same model_artifacts/ directory EnsembleModel.load
already reads.
"""

from pathlib import Path

from mlflow.pyfunc.model import PythonModel

from src.models.EnsembleModel import EnsembleModel


class EnsemblePyfuncModel(PythonModel):
    """MLflow wraps this in its own PythonModelContext/params machinery, so
    the base class's `predict`/`load_context` signatures are intentionally
    left untyped here too — annotating them tighter than the (also untyped)
    base method trips Pyright's override-compatibility check without fixing
    anything real."""

    def load_context(self, context):
        self.ensemble = EnsembleModel.load(Path(context.artifacts["model_artifacts"]))

    def predict(self, context, model_input, params=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        # context/params are part of the PythonModel interface but unused here —
        # the ensemble was already loaded onto self in load_context.
        return self.ensemble.predict_proba(model_input)
