"""Core data and forward-model utilities for the paper SE analysis.

The public API deliberately contains no optimizer or scientific model-selection
policy.  It provides validated observations, SRIM deposition profiles, exact
escape integrals, and the three forward models used in the paper comparison.
"""

from .data import ObservationData, load_observations, load_srim
from .deposition import IonizationProfile
from .models import (
    PAPER_MODEL_SPECS,
    ModelPrediction,
    ModelSpec,
    components,
    paper_model_specs,
    predict_model,
    prediction,
)

__all__ = [
    "IonizationProfile",
    "ModelPrediction",
    "ModelSpec",
    "ObservationData",
    "PAPER_MODEL_SPECS",
    "components",
    "load_observations",
    "load_srim",
    "paper_model_specs",
    "predict_model",
    "prediction",
]
