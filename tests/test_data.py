import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nids.config import (ATTACK_CATEGORY, CATEGORICAL_FEATURES, CLASS_NAMES,
                         DEFAULT_FLOW, FEATURE_NAMES)
from nids.data import generate_synthetic, load_nsl_kdd


class DataTests(unittest.TestCase):
    def test_schema_has_41_features(self):
        self.assertEqual(len(FEATURE_NAMES), 41)

    def test_label_taxonomy(self):
        self.assertEqual(ATTACK_CATEGORY["neptune"], "DoS")
        self.assertEqual(ATTACK_CATEGORY["nmap"], "Probe")
        self.assertEqual(ATTACK_CATEGORY["guess_passwd"], "R2L")
        self.assertEqual(ATTACK_CATEGORY["rootkit"], "U2R")

    def test_synthetic_shape_and_classes(self):
        X, y = generate_synthetic(600)
        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(set(y), set(CLASS_NAMES))
        for col in CATEGORICAL_FEATURES:
            self.assertTrue(X[col].map(lambda v: isinstance(v, str)).all())

    def test_nsl_loader_43_columns(self):
        rows = []
        for label in ("normal", "neptune", "satan", "bogus_attack"):
            row = dict(DEFAULT_FLOW)
            row["service"] = "http"
            rows.append([row[f] for f in FEATURE_NAMES] + [label, 20])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "KDDTrain+.txt"
            pd.DataFrame(rows).to_csv(path, header=False, index=False)
            X, y = load_nsl_kdd(path)
        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(list(y), ["Normal", "DoS", "Probe"])  # unknown dropped


if __name__ == "__main__":
    unittest.main()
