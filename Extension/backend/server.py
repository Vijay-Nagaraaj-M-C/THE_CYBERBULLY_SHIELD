"""
server.py — FastAPI backend for The Cyberbully Shield browser extension.
Run with:  python server.py
Serves on: http://localhost:5000
"""

import os
import base64
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import ShieldPredictor
from database import init_db, log_detection, get_history, get_stats, clear_history

# ── Pydantic models ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = ""
    image_base64: str | None = None
    source_url: str = ""

class BatchPredictRequest(BaseModel):
    texts: list[str]
    source_url: str = ""

# ── Global predictor ─────────────────────────────────────────────────────
predictor: ShieldPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    init_db()
    predictor = ShieldPredictor()
    yield

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Cyberbully Shield API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Endpoints ────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "online", "model_loaded": predictor is not None}

@app.post("/api/predict")
async def predict(req: PredictRequest):
    image_bytes = None
    content_type = "text"
    if req.image_base64:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            content_type = "multimodal"
        except Exception:
            pass

    result = predictor.predict(req.text, image_bytes)

    # Log if bullying detected
    if result["is_bullying"]:
        log_detection(
            text_snippet=req.text,
            source_url=req.source_url,
            prediction=result["label"],
            confidence=result["confidence"],
            content_type=content_type,
            was_censored=True
        )

    return result

@app.post("/api/predict_batch")
async def predict_batch(req: BatchPredictRequest):
    results = predictor.predict_batch(req.texts)

    # Log any detections
    for text, result in zip(req.texts, results):
        if result["is_bullying"]:
            log_detection(
                text_snippet=text,
                source_url=req.source_url,
                prediction=result["label"],
                confidence=result["confidence"],
                content_type="text",
                was_censored=True
            )

    return {"results": results}

@app.get("/api/stats")
async def stats():
    return get_stats()

@app.get("/api/history")
async def history(limit: int = 100):
    return get_history(limit)

@app.post("/api/clear")
async def clear():
    clear_history()
    return {"status": "cleared"}

# ── Dashboard ────────────────────────────────────────────────────────────
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), 'dashboard')

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(DASHBOARD_DIR, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  THE CYBERBULLY SHIELD — Local Inference Server")
    print("  Dashboard: http://localhost:5000/dashboard")
    print("  API:       http://localhost:5000/api/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5000)
