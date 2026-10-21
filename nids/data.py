"""Dataset loading: real NSL-KDD when available, labeled synthetic otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from . import log
from .config import (ATTACK_CATEGORY, CATEGORICAL_FEATURES, CLASS_NAMES,
                     DATA_DIR, FEATURE_NAMES, NUMERIC_FEATURES, RANDOM_STATE,
                     ROOT_DIR, SYNTH_FLAGS, SYNTH_PROTOCOLS, SYNTH_SERVICES,
                     TEST_FILE, TRAIN_FILE)


@dataclass
class Dataset:
    """Container for training data plus optional official external test set."""

    X: pd.DataFrame
    y: pd.Series
    source: str
    synthetic: bool
    X_ext: Optional[pd.DataFrame] = None
    y_ext: Optional[pd.Series] = None


def find_file(filename: str, data_dir: Optional[Path] = None) -> Optional[Path]:
    """Look for a dataset file in data_dir, ./data, project root and cwd."""
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir) / filename)
    candidates += [DATA_DIR / filename, ROOT_DIR / filename, Path.cwd() / filename]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_nsl_kdd(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Parse an NSL-KDD text file into (features, 5-class labels)."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 43:
        df.columns = FEATURE_NAMES + ["label", "difficulty"]
        df = df.drop(columns="difficulty")
    elif df.shape[1] == 42:
        df.columns = FEATURE_NAMES + ["label"]
    else:
        raise ValueError(
            f"{path.name}: expected 42 or 43 columns, found {df.shape[1]}")

    raw = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    category = raw.map(ATTACK_CATEGORY)
    unknown = category.isna()
    if unknown.any():
        names = sorted(raw[unknown].unique())
        log.warn(f"Dropping {int(unknown.sum())} rows with unmapped labels: {names}")
        df, category = df[~unknown], category[~unknown]

    X = df[FEATURE_NAMES].reset_index(drop=True).copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    y = category.reset_index(drop=True).rename("label")
    return X, y


def _bin_to_categories(values: np.ndarray, categories: list) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    codes = pd.cut(ranks, bins=len(categories), labels=False).astype(int)
    return np.array(categories, dtype=object)[codes.to_numpy()]


def generate_synthetic(n_samples: int = 15000,
                       seed: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic stand-in with the NSL-KDD schema (DEMO ONLY, not real traffic)."""
    Xr, yr = make_classification(
        n_samples=n_samples, n_features=len(FEATURE_NAMES), n_informative=30,
        n_redundant=5, n_classes=5, n_clusters_per_class=2,
        weights=[0.5, 0.2, 0.15, 0.1, 0.05], flip_y=0.01, random_state=seed)
    X = pd.DataFrame(Xr, columns=FEATURE_NAMES)

    # Rescale numeric columns into plausible non-negative ranges.
    scale = {"duration": 10000, "src_bytes": 100000, "dst_bytes": 100000,
             "count": 500, "srv_count": 500, "dst_host_count": 255,
             "dst_host_srv_count": 255}
    for col in NUMERIC_FEATURES:
        v = X[col].to_numpy()
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        X[col] = v * scale.get(col, 1.0)

    X["protocol_type"] = _bin_to_categories(Xr[:, 1], SYNTH_PROTOCOLS)
    X["service"] = _bin_to_categories(Xr[:, 2], SYNTH_SERVICES)
    X["flag"] = _bin_to_categories(Xr[:, 3], SYNTH_FLAGS)

    y = pd.Series(np.array(CLASS_NAMES, dtype=object)[yr], name="label")
    return X, y


def load_dataset(data_dir: Optional[Path] = None, allow_synthetic: bool = True,
                 n_samples: int = 15000) -> Dataset:
    """Load NSL-KDD if present, else fall back to synthetic data (if allowed)."""
    train_path = find_file(TRAIN_FILE, data_dir)
    if train_path is not None:
        X, y = load_nsl_kdd(train_path)
        log.ok(f"NSL-KDD training file loaded: {train_path} {X.shape}")
        ds = Dataset(X, y, source=f"NSL-KDD ({train_path.name})", synthetic=False)
        test_path = find_file(TEST_FILE, data_dir)
        if test_path is not None:
            ds.X_ext, ds.y_ext = load_nsl_kdd(test_path)
            log.ok(f"NSL-KDD official test file loaded: {test_path} {ds.X_ext.shape}")
        return ds

    if not allow_synthetic:
        raise FileNotFoundError(
            f"{TRAIN_FILE} not found. Place it in {DATA_DIR} or drop --require-real-data.")

    log.warn("NSL-KDD not found. Using SYNTHETIC data with the NSL-KDD schema.")
    log.warn("Metrics on synthetic data are NOT real-world detection results.")
    X, y = generate_synthetic(n_samples)
    return Dataset(X, y, source="synthetic (make_classification)", synthetic=True)
