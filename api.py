"""FastAPI service: `uvicorn api:app --host 0.0.0.0 --port 8000`.

Set NIDS_API_KEY to require an `X-API-Key` header on prediction endpoints.
"""

from __future__ import annotations

import hmac
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from nids import __version__
from nids.config import PRODUCT_NAME
from nids.predict import NIDSPredictor

MAX_BATCH = 10_000

app = FastAPI(
    title=f"{PRODUCT_NAME} API", version=__version__,
    description="Network intrusion detection over the NSL-KDD 41-feature schema. "
                "Developer: Ashish Kumar, Student-led startup of Jharkhand Rai "
                "University; Director & Founder, MVA Robotics Innovation Org.")


class FlowRequest(BaseModel):
    features: Dict[str, Any]


class BatchRequest(BaseModel):
    flows: List[Dict[str, Any]]


@lru_cache(maxsize=1)
def _load() -> NIDSPredictor:
    return NIDSPredictor()


def get_predictor() -> NIDSPredictor:
    try:
        return _load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("NIDS_API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _load()
        return {"status": "ok", "model_loaded": True, "version": __version__}
    except (FileNotFoundError, ValueError):
        return {"status": "degraded", "model_loaded": False, "version": __version__}


@app.get("/model/info", dependencies=[Depends(require_api_key)])
def model_info(p: NIDSPredictor = Depends(get_predictor)) -> Dict[str, Any]:
    m = p.metrics
    return {"version": m.get("version"), "trained_at": m.get("trained_at"),
            "dataset": m.get("dataset"), "labels": p.classes,
            "holdout": {k: m.get("holdout", {}).get(k)
                        for k in ("accuracy", "f1_weighted", "f1_macro", "security")},
            "benchmark": m.get("benchmark")}


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: FlowRequest,
            p: NIDSPredictor = Depends(get_predictor)) -> Dict[str, Any]:
    result = p.predict_one(req.features)
    return {"result": result, "siem_event": p.to_siem_event(result, req.features)}


@app.post("/predict/batch", dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchRequest,
                  p: NIDSPredictor = Depends(get_predictor)) -> Dict[str, Any]:
    if not req.flows:
        raise HTTPException(status_code=422, detail="`flows` must not be empty")
    if len(req.flows) > MAX_BATCH:
        raise HTTPException(status_code=413,
                            detail=f"Batch too large (max {MAX_BATCH})")
    res = p.predict(pd.DataFrame(req.flows))
    return {"count": len(res),
            "attacks_detected": int((res["prediction"] != "Normal").sum()),
            "predictions": res.to_dict(orient="records")}
