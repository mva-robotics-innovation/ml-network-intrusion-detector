import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nids.config import MODEL_FILE, METRICS_FILE
from nids.predict import NIDSPredictor
from nids.train import run_training


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.metrics = run_training(artifact_dir=cls.dir, n_samples=1500,
                                   run_cv=False, n_estimators=20, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        for name in (MODEL_FILE, METRICS_FILE, "confusion_matrix.png",
                     "feature_importance.csv"):
            self.assertTrue((self.dir / name).is_file(), name)

    def test_metrics_are_measured_not_hardcoded(self):
        h = self.metrics["holdout"]
        self.assertGreater(h["accuracy"], 0.3)
        self.assertLessEqual(h["accuracy"], 1.0)
        self.assertGreater(self.metrics["benchmark"]["batch_throughput_flows_per_sec"], 0)
        self.assertTrue(self.metrics["dataset"]["synthetic"])

    def test_predict_one_and_siem(self):
        p = NIDSPredictor(self.dir)
        r = p.predict_one({"src_bytes": 100, "count": 300})
        self.assertIn(r["prediction"], p.classes)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=5)
        ev = p.to_siem_event(r, {"protocol_type": "tcp"})
        self.assertIn(ev["event"]["kind"], ("alert", "event"))
        self.assertEqual(ev["nids"]["label"], r["prediction"])

    def test_batch_and_unseen_category(self):
        p = NIDSPredictor(self.dir)
        res = p.predict(pd.DataFrame([{"service": "never_seen"}, {"src_bytes": "abc"}]))
        self.assertEqual(len(res), 2)

    def test_tampered_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as t2:
            t2 = Path(t2)
            for name in (MODEL_FILE, METRICS_FILE):
                (t2 / name).write_bytes((self.dir / name).read_bytes())
            with open(t2 / MODEL_FILE, "ab") as fh:
                fh.write(b"tamper")
            with self.assertRaises(ValueError):
                NIDSPredictor(t2)


if __name__ == "__main__":
    unittest.main()
