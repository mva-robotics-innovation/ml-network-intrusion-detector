import os
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except ImportError:  # fastapi/httpx not installed
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi + httpx required")
class ApiTests(unittest.TestCase):
    def test_health_and_predict(self):
        from nids.train import run_training
        with tempfile.TemporaryDirectory() as tmp:
            run_training(artifact_dir=Path(tmp), n_samples=1500, run_cv=False,
                         n_estimators=20, verbose=False)
            os.environ["NIDS_ARTIFACT_DIR"] = tmp
            import importlib
            import nids.config
            importlib.reload(nids.config)
            import nids.predict
            importlib.reload(nids.predict)
            import api
            importlib.reload(api)
            client = TestClient(api.app)
            self.assertEqual(client.get("/health").json()["status"], "ok")
            r = client.post("/predict", json={"features": {"src_bytes": 10}})
            self.assertEqual(r.status_code, 200)
            self.assertIn("siem_event", r.json())
            b = client.post("/predict/batch", json={"flows": []})
            self.assertEqual(b.status_code, 422)
            os.environ.pop("NIDS_ARTIFACT_DIR", None)


if __name__ == "__main__":
    unittest.main()
