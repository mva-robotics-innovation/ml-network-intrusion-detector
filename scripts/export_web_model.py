"""Export a compact RandomForest to site/model.json for in-browser inference."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nids.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, THREAT_LEVELS
from nids.data import generate_synthetic

X, y = generate_synthetic(12000)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
prep = ColumnTransformer([("n", "passthrough", NUMERIC_FEATURES),
                          ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)])
A, B = prep.fit_transform(Xtr), prep.transform(Xte)
rf = RandomForestClassifier(n_estimators=30, max_depth=9, min_samples_leaf=15,
                            class_weight="balanced", random_state=42, n_jobs=-1).fit(A, ytr)
labels, pred = list(rf.classes_), rf.predict(B)
cm = confusion_matrix(yte, pred, labels=labels)
n = labels.index("Normal")
rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
cats = {c: [str(v) for v in k] for c, k in zip(CATEGORICAL_FEATURES, prep.named_transformers_["c"].categories_)}

imp, i = {}, len(NUMERIC_FEATURES)
for f, v in zip(NUMERIC_FEATURES, rf.feature_importances_[:i]):
    imp[f] = float(v)
for c in CATEGORICAL_FEATURES:
    imp[c] = float(rf.feature_importances_[i:i + len(cats[c])].sum()); i += len(cats[c])

trees = []
for e in rf.estimators_:
    t = e.tree_; v = t.value[:, 0, :]; v = v / v.sum(1, keepdims=True)
    trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(),
                  "f": t.feature.tolist(), "t": [round(float(x), 4) for x in t.threshold],
                  "v": [[round(float(q), 3) for q in v[k]] if t.children_left[k] == -1 else 0
                        for k in range(t.node_count)]})
rng = np.random.RandomState(7)
samples = []
for lab in labels:
    for idx in rng.choice(np.where(yte.to_numpy() == lab)[0], 4, replace=False):
        r = Xte.iloc[idx]
        samples.append({"label": lab, "num": [round(float(r[f]), 3) for f in NUMERIC_FEATURES],
                        "cat": {c: str(r[c]) for c in CATEGORICAL_FEATURES}})
out = {"classes": labels, "numeric": NUMERIC_FEATURES, "cat_cols": CATEGORICAL_FEATURES,
       "categories": cats, "threat": THREAT_LEVELS, "trees": trees, "samples": samples,
       "synthetic": True, "n_train": len(Xtr), "n_test": len(Xte),
       "metrics": {"accuracy": float((pred == yte).mean()),
                   "f1_macro": float(f1_score(yte, pred, average="macro")),
                   "detection_rate": float(1 - (cm[:, n].sum() - cm[n, n]) / (cm.sum() - cm[n].sum())),
                   "false_alarm_rate": float((cm[n].sum() - cm[n, n]) / cm[n].sum())},
       "per_class": {l: {k: round(rep[l][k], 3) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[l]["support"])} for l in labels},
       "confusion": cm.tolist(), "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])}
Path(__file__).resolve().parent.parent.joinpath("site", "model.json").write_text(json.dumps(out, separators=(",", ":")))
print("exported", len(trees), "trees")
