"""Training / evaluation / benchmarking pipeline (CLI entry point)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import (StratifiedKFold, cross_validate,  # noqa: E402
                                     train_test_split)

from . import __version__, log
from .config import (ARTIFACT_DIR, CLASS_NAMES, CONFUSION_PNG, IMPORTANCE_CSV,
                     METRICS_FILE, MODEL_FILE, PRODUCT_NAME, RANDOM_STATE,
                     DEVELOPER, AFFILIATION, ORGANIZATION, PROGRAM)
from .data import load_dataset
from .pipeline import aggregate_importances, build_pipeline


def detection_metrics(cm: np.ndarray, labels: List[str]) -> Dict[str, Optional[float]]:
    """Binary IDS view: detection rate (attacks flagged) and false-alarm rate."""
    if "Normal" not in labels:
        return {}
    n = labels.index("Normal")
    normal_total = cm[n].sum()
    attack_total = cm.sum() - normal_total
    false_alarms = normal_total - cm[n, n]
    missed = cm[:, n].sum() - cm[n, n]
    return {
        "detection_rate": float((attack_total - missed) / attack_total) if attack_total else None,
        "false_alarm_rate": float(false_alarms / normal_total) if normal_total else None,
    }


def evaluate(pipeline, X: pd.DataFrame, y: pd.Series, labels: List[str]) -> dict:
    """Compute the full metric set for one evaluation split."""
    y_pred = pipeline.predict(X)
    p, r, f, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted", zero_division=0)
    report = classification_report(y, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels)
    per_class = {
        lab: {"precision": float(report[lab]["precision"]),
              "recall": float(report[lab]["recall"]),
              "f1": float(report[lab]["f1-score"]),
              "support": int(report[lab]["support"])}
        for lab in labels
    }
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "security": detection_metrics(cm, labels),
    }


def benchmark(pipeline, X: pd.DataFrame, n_single: int = 200) -> dict:
    """Measure real single-flow latency and batch throughput (1 CPU thread)."""
    pipeline.set_params(clf__n_jobs=1)
    sample = X.sample(min(n_single, len(X)), random_state=RANDOM_STATE)
    pipeline.predict(sample.iloc[[0]])  # warm-up
    times = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        pipeline.predict(sample.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000.0)
    t0 = time.perf_counter()
    pipeline.predict(X)
    batch_s = time.perf_counter() - t0
    return {
        "single_flow_latency_ms_mean": float(np.mean(times)),
        "single_flow_latency_ms_p95": float(np.percentile(times, 95)),
        "batch_throughput_flows_per_sec": float(len(X) / batch_s),
        "batch_size": int(len(X)),
        "cpu_threads": 1,
    }


def plot_confusion(cm: np.ndarray, labels: List[str], accuracy: float,
                   path: Path, synthetic: bool) -> None:
    """Counts + row-normalised confusion matrices (recall per class)."""
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title("Counts")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
    axes[1].set_title("Row-normalised (per-class recall)")
    for ax in axes:
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    tag = " [SYNTHETIC DATA]" if synthetic else ""
    fig.suptitle(f"{PRODUCT_NAME} - Hold-out Confusion Matrix "
                 f"(accuracy {accuracy * 100:.2f}%){tag}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_training(data_dir: Optional[Path] = None,
                 artifact_dir: Optional[Path] = None,
                 n_samples: int = 15000, run_cv: bool = True,
                 n_estimators: int = 150, allow_synthetic: bool = True,
                 verbose: bool = True) -> dict:
    """Train, evaluate, benchmark and persist everything. Returns metrics dict."""
    log.set_verbose(verbose)
    out = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"{PRODUCT_NAME} v{__version__} - training pipeline starting")

    ds = load_dataset(data_dir, allow_synthetic, n_samples)
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=RANDOM_STATE, stratify=ds.y)
    labels = [c for c in CLASS_NAMES if c in set(ds.y)]
    log.info(f"Train {X_tr.shape} | Test {X_te.shape} | Classes {labels}")

    pipeline = build_pipeline(n_estimators)

    cv_result = None
    if run_cv:
        log.info("Running 5-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(pipeline, X_tr, y_tr, cv=cv,
                                scoring={"acc": "accuracy", "f1m": "f1_macro"})
        cv_result = {
            "folds": 5,
            "accuracy_mean": float(scores["test_acc"].mean()),
            "accuracy_std": float(scores["test_acc"].std()),
            "f1_macro_mean": float(scores["test_f1m"].mean()),
            "f1_macro_std": float(scores["test_f1m"].std()),
        }
        log.ok(f"CV accuracy: {cv_result['accuracy_mean'] * 100:.2f}% "
               f"(+/- {cv_result['accuracy_std'] * 100:.2f}%)")

    log.info("Fitting final model...")
    pipeline.fit(X_tr, y_tr)
    labels = [c for c in CLASS_NAMES if c in set(pipeline.classes_)]

    holdout = evaluate(pipeline, X_te, y_te, labels)
    log.ok(f"Hold-out accuracy: {holdout['accuracy'] * 100:.2f}% | "
           f"weighted F1: {holdout['f1_weighted'] * 100:.2f}% | "
           f"macro F1: {holdout['f1_macro'] * 100:.2f}%")
    sec = holdout["security"]
    if sec:
        log.ok(f"Detection rate: {sec['detection_rate'] * 100:.2f}% | "
               f"False-alarm rate: {sec['false_alarm_rate'] * 100:.2f}%")
    log.info("\n" + classification_report(y_te, pipeline.predict(X_te),
                                          labels=labels, zero_division=0))

    external = None
    if ds.X_ext is not None:
        external = evaluate(pipeline, ds.X_ext, ds.y_ext, labels)
        log.ok(f"Official KDDTest+ accuracy: {external['accuracy'] * 100:.2f}% "
               f"(expect lower than hold-out: the official set is harder)")

    imp = aggregate_importances(pipeline)
    imp.to_csv(out / IMPORTANCE_CSV, index=False)
    log.info("Top 10 features:")
    for _, row in imp.head(10).iterrows():
        log.info(f"   {row['feature']:<32}{row['importance'] * 100:6.2f}%")

    plot_confusion(np.array(holdout["confusion_matrix"]), labels,
                   holdout["accuracy"], out / CONFUSION_PNG, ds.synthetic)

    bench = benchmark(pipeline, X_te)
    log.ok(f"Latency: {bench['single_flow_latency_ms_mean']:.2f} ms/flow (mean), "
           f"{bench['single_flow_latency_ms_p95']:.2f} ms (p95) | "
           f"Throughput: {bench['batch_throughput_flows_per_sec']:,.0f} flows/s "
           f"on {bench['cpu_threads']} thread")

    model_path = out / MODEL_FILE
    joblib.dump(pipeline, model_path, compress=3)

    metrics = {
        "product": PRODUCT_NAME, "version": __version__,
        "developer": {"name": DEVELOPER, "affiliation": AFFILIATION,
                      "organization": ORGANIZATION, "program": PROGRAM},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "source": ds.source, "synthetic": ds.synthetic,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "n_features": int(ds.X.shape[1]),
            "class_distribution": {k: int(v) for k, v in ds.y.value_counts().items()},
        },
        "labels": labels,
        "holdout": holdout,
        "external_test": external,
        "cross_validation": cv_result,
        "benchmark": bench,
        "feature_importance": [
            {"feature": r.feature, "importance": float(r.importance)}
            for r in imp.itertuples()],
        "environment": {"python": platform.python_version(),
                        "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "model_sha256": _sha256(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 3),
    }
    with open(out / METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    log.ok(f"Artifacts saved to {out} "
           f"({MODEL_FILE}: {metrics['model_size_mb']} MB, {METRICS_FILE}, "
           f"{CONFUSION_PNG}, {IMPORTANCE_CSV})")
    if ds.synthetic:
        log.warn("Model trained on SYNTHETIC data: demo only, not for real traffic.")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PRODUCT_NAME} training pipeline")
    ap.add_argument("--data-dir", type=Path, help="folder with KDDTrain+.txt")
    ap.add_argument("--artifact-dir", type=Path, help="output folder")
    ap.add_argument("--samples", type=int, default=15000,
                    help="synthetic sample count (fallback only)")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    ap.add_argument("--require-real-data", action="store_true",
                    help="fail instead of falling back to synthetic data")
    args = ap.parse_args(argv)
    try:
        run_training(args.data_dir, args.artifact_dir, args.samples,
                     not args.no_cv, args.trees, not args.require_real_data)
    except FileNotFoundError as exc:
        log.warn(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
