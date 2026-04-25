"""
api.py — FastAPI REST Endpoint
================================
Production-ready HTTP API for serving predictions.

Start server:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health          → health check
    GET  /model/info      → model metadata
    POST /predict         → single employee prediction
    POST /predict/batch   → multiple employees at once

Install:
    pip install fastapi uvicorn
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

from config import CFG
from predict import predict_single, predict_batch, load_pipeline


# ── Request / Response schemas ───────────────────────────────────────
class EmployeeInput(BaseModel):
    age                 : int   = Field(..., ge=18, le=70,  example=32)
    gender              : str   = Field(...,                example="Female")
    education           : str   = Field(...,                example="Master")
    department          : str   = Field(...,                example="Engineering")
    experience_years    : int   = Field(..., ge=0,  le=40,  example=8)
    salary              : float = Field(..., ge=0,          example=75000)
    training_hours      : int   = Field(..., ge=0,  le=200, example=60)
    projects_completed  : int   = Field(..., ge=0,          example=12)
    avg_monthly_hours   : float = Field(..., ge=0,          example=180)
    satisfaction_score  : float = Field(..., ge=1,  le=5,   example=4.2)
    last_promotion_years: int   = Field(..., ge=0,          example=2)
    absenteeism_days    : int   = Field(..., ge=0,  le=365, example=3)
    peer_review_score   : float = Field(..., ge=1,  le=5,   example=4.0)
    manager_rating      : float = Field(..., ge=1,  le=5,   example=4.5)
    performance_score   : Optional[float] = Field(default=0.0)


class PredictionResponse(BaseModel):
    label           : str
    confidence      : float
    probabilities   : dict
    risk_score      : float
    recommendations : List[str]


class BatchInput(BaseModel):
    employees: List[EmployeeInput]


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title       = "Employee Performance Predictor API",
        description = "AI-powered HR analytics — predicts High/Medium/Low performance",
        version     = "2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        pipeline_exists = os.path.exists(CFG.PIPELINE_PKL)
        return {
            "status"         : "ok" if pipeline_exists else "model_not_found",
            "pipeline_exists": pipeline_exists,
            "version"        : "2.0.0",
        }

    @app.get("/model/info")
    def model_info():
        if not os.path.exists(CFG.METADATA_JSON):
            raise HTTPException(status_code=404, detail="Model metadata not found. Train the model first.")
        with open(CFG.METADATA_JSON) as f:
            return json.load(f)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(employee: EmployeeInput):
        try:
            result = predict_single(employee.model_dump())
            return PredictionResponse(**result)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @app.post("/predict/batch")
    def predict_batch_endpoint(batch: BatchInput):
        try:
            pipe    = load_pipeline()
            records = [e.model_dump() for e in batch.employees]
            df      = pd.DataFrame(records)
            preds   = pipe.predict(df)
            probas  = pipe.predict_proba(df).max(axis=1)
            return {
                "count"      : len(records),
                "predictions": [
                    {"label": p, "confidence": round(c * 100, 1)}
                    for p, c in zip(preds, probas)
                ],
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Stub so the file can be imported without crashing
    app = None
    print("FastAPI not available. Install with: pip install fastapi uvicorn")


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run("api:app", host=CFG.API_HOST, port=CFG.API_PORT, reload=True)
    else:
        print("Install FastAPI first: pip install fastapi uvicorn")
