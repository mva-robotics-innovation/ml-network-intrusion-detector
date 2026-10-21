"""Model pipeline: preprocessing + RandomForest in one serializable object."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE)


def build_pipeline(n_estimators: int = 150,
                   random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the production pipeline.

    Trees are scale-invariant, so numeric features pass through untouched;
    categorical features are one-hot encoded and unseen categories at
    inference time are ignored instead of crashing the service.
    """
    prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=random_state)
    return Pipeline([("prep", prep), ("clf", clf)])


def _origin_feature(encoded_name: str) -> str:
    kind, _, rest = encoded_name.partition("__")
    if kind == "cat":
        for col in CATEGORICAL_FEATURES:
            if rest.startswith(col + "_"):
                return col
    return rest


def aggregate_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Feature importances summed back to the original 41 input features."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    imps = pipeline.named_steps["clf"].feature_importances_
    df = pd.DataFrame({"feature": [_origin_feature(n) for n in names],
                       "importance": imps})
    return (df.groupby("feature", as_index=False)["importance"].sum()
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))
