"""Inference layer shared by the dashboard, the REST API and scripts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from . import __version__, log
from .config import (ARTIFACT_DIR, CATEGORICAL_FEATURES, DEFAULT_FLOW,
                     FEATURE_NAMES, METRICS_FILE, MODEL_FILE, NUMERIC_FEATURES,
                     PRODUCT_NAME, SEVERITY_SCORE, THREAT_LEVELS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NIDSPredictor:
    """Loads the trained pipeline (integrity-checked) and scores flows."""

    def __init__(self, artifact_dir: Optional[Path] = None,
                 verify_integrity: bool = True) -> None:
        self.dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
        model_path = self.dir / MODEL_FILE
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} not found. Run `python ml_nids.py` first.")

        self.metrics: Dict[str, Any] = {}
        metrics_path = self.dir / METRICS_FILE
        if metrics_path.is_file():
            with open(metrics_path, encoding="utf-8") as fh:
                self.metrics = json.load(fh)

        # joblib/pickle can execute code on load: only load models you trained.
        expected = self.metrics.get("model_sha256")
        if verify_integrity:
            if expected and _sha256(model_path) != expected:
                raise ValueError(
                    "Model file hash does not match metrics.json. "
                    "Refusing to load a possibly tampered model.")
            if not expected:
                log.warn("No model hash recorded; integrity check skipped.")

        self.pipeline = joblib.load(model_path)
        self.classes: List[str] = [str(c) for c in self.pipeline.classes_]

    # -- schema helpers -----------------------------------------------------
    def categories(self) -> Dict[str, List[str]]:
        """Categorical values seen during training (for UI dropdowns)."""
        enc = self.pipeline.named_steps["prep"].named_transformers_["cat"]
        return {col: [str(v) for v in cats]
                for col, cats in zip(CATEGORICAL_FEATURES, enc.categories_)}

    @staticmethod
    def missing_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in FEATURE_NAMES if c not in df.columns]

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Return a clean 41-column frame: add defaults, coerce types."""
        out = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            default = DEFAULT_FLOW[col]
            if col in df.columns:
                if col in NUMERIC_FEATURES:
                    out[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
                else:
                    out[col] = df[col].astype(str).str.strip()
            else:
                out[col] = default
        return out[FEATURE_NAMES]

    # -- inference ----------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score many flows. Returns prediction, confidence, threat level, probs."""
        X = self.prepare(df)
        proba = self.pipeline.predict_proba(X)
        best = proba.argmax(axis=1)
        out = pd.DataFrame({
            "prediction": [self.classes[i] for i in best],
            "confidence": proba.max(axis=1),
        }, index=df.index)
        out["threat_level"] = out["prediction"].map(THREAT_LEVELS)
        for j, cls in enumerate(self.classes):
            out[f"prob_{cls}"] = proba[:, j]
        return out

    def predict_one(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Score one flow (missing features get defaults)."""
        row = self.predict(pd.DataFrame([flow])).iloc[0]
        return {
            "prediction": row["prediction"],
            "confidence": float(row["confidence"]),
            "threat_level": row["threat_level"],
            "is_attack": row["prediction"] != "Normal",
            "probabilities": {c: float(row[f"prob_{c}"]) for c in self.classes},
        }

    def to_siem_event(self, result: Dict[str, Any],
                      flow: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ECS-style JSON event for Wazuh / Splunk / Elastic ingestion."""
        flow = flow or {}
        return {
            "@timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": {
                "kind": "alert" if result["is_attack"] else "event",
                "category": ["intrusion_detection"],
                "severity": SEVERITY_SCORE.get(result["threat_level"], 0),
                "module": "ml-nids",
            },
            "nids": {
                "label": result["prediction"],
                "confidence": round(result["confidence"], 4),
                "threat_level": result["threat_level"],
                "probabilities": {k: round(v, 4)
                                  for k, v in result["probabilities"].items()},
                "model_version": self.metrics.get("version", __version__),
                "trained_on_synthetic": self.metrics.get(
                    "dataset", {}).get("synthetic"),
            },
            "network": {"protocol": flow.get("protocol_type"),
                        "service": flow.get("service")},
            "observer": {"product": PRODUCT_NAME},
        }
