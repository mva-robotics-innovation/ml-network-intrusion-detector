# 🛡️ ML-NIDS Enterprise v1.0
# ML Network Intrusion Detector

A Scikit-learn-based machine learning project for classifying network traffic anomalies using public intrusion detection datasets.
**Machine Learning Network Intrusion Detection System**
IBM Cybersecurity Bootcamp 2026 · Project Option 6

Classifies network flows into five categories (**Normal, DoS, Probe, R2L, U2R**)
using a RandomForest trained on the NSL-KDD 41-feature schema. Ships with a
training pipeline, an analyst dashboard, a REST API, SIEM-ready JSON output,
automated tests, CI and Docker.

## Highlights

- **One serialisable pipeline**: one-hot encoding + RandomForest in a single
  scikit-learn `Pipeline`, so training and serving cannot drift apart.
- **Real NSL-KDD support**: parses the 43-column files, maps ~40 attack names
  to 5 classes, optionally evaluates on the official `KDDTest+`.
- **Measured, not claimed**: accuracy, macro-F1, detection rate, false-alarm
  rate, latency and throughput are computed at training time and stored in
  `artifacts/metrics.json`.
- **Security-minded**: SHA-256 model integrity check before unpickling,
  optional API-key auth (constant-time compare), non-root Docker user,
  batch-size limits, unseen categories handled safely.
- **SIEM output**: ECS-style JSON events for Wazuh / Splunk / Elastic.

## Architecture

```
KDDTrain+.txt ─► nids/data.py ─► nids/pipeline.py ─► nids/train.py ─► artifacts/
 (or synthetic)   load + map      OneHot + RF          CV, eval, bench   model + metrics
                                                              │
                         ┌────────────────────────────────────┘
                         ▼
                 nids/predict.py (NIDSPredictor, integrity-checked)
                    ├──► app.py   Streamlit dashboard (live + batch CSV)
                    └──► api.py   FastAPI REST service (/predict, /predict/batch)
```

## Quick start

Python **3.9–3.11** is required for the pinned dependency versions.

```bash
pip install -r requirements.txt
python ml_nids.py              # train (uses data/KDDTrain+.txt if present)
streamlit run app.py           # dashboard  -> http://localhost:8501
uvicorn api:app --port 8000    # REST API   -> http://localhost:8000/docs
```

Docker:

```bash
docker compose --profile train run --rm train
docker compose up --build
```

### Using real data

1. Download NSL-KDD and put `KDDTrain+.txt` (and optionally `KDDTest+.txt`) in `data/`.
2. `python ml_nids.py --require-real-data`

Without the files, training falls back to a **synthetic** dataset. It is
labelled as such everywhere (console, dashboard banner, SIEM events, charts)
and must not be used to claim real-world performance.

## REST API

```bash
export NIDS_API_KEY=change-me        # optional; enables X-API-Key auth
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $NIDS_API_KEY" -H "Content-Type: application/json" \
  -d '{"features": {"src_bytes": 500, "count": 10, "serror_rate": 0.0}}'
```

Endpoints: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch` (max 10,000 flows).

## Python usage

```python
from nids.predict import NIDSPredictor
p = NIDSPredictor()
result = p.predict_one({"src_bytes": 500, "count": 10})
event = p.to_siem_event(result)
```

## Results

Run `python ml_nids.py` and copy the numbers from `artifacts/metrics.json`
(or the dashboard's Analytics tab) here. Report hold-out **and** official
`KDDTest+` results, plus macro-F1, detection rate and false-alarm rate;
accuracy alone is misleading on imbalanced data (R2L and U2R are rare).

## Testing

```bash
python -m unittest discover -s tests -t . -v      # or: pytest
```

## Limitations (read before deploying)

- NSL-KDD is a 2009-era benchmark; it does not represent modern encrypted or
  cloud traffic. Validate on your own labelled data before any production use.
- Hold-out accuracy on NSL-KDD is optimistic; the official `KDDTest+` set is
  harder. R2L/U2R recall is typically the weakest part.
- Input is the 41 pre-computed flow features, not raw packets; a feature
  extractor (e.g. from PCAP/Zeek/CICFlowMeter) is needed for live traffic.
- Supervised models detect known attack patterns; they do not guarantee
  zero-day detection.
- Threat levels in `nids/config.py` are an editable policy, not ground truth.

## Roadmap

- **v2.0** LSTM/sequence models for multi-flow attack patterns
- **v2.1** Real-time PCAP ingestion and flow feature extraction
- **v3.0** Federated learning across organisations

## About the developer

**Ashish Kumar**
Student-led startup of Jharkhand Rai University
Director & Founder, MVA Robotics Innovation Org
IBM Cybersecurity Bootcamp 2026

Licensed under MIT.

## Live demo site

The `site/` folder is a static page with in-browser inference (`site/model.json`, re-exported by `scripts/export_web_model.py`). The `Deploy site` workflow publishes it to GitHub Pages on every push to `main`.
