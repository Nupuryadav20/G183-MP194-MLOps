import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict


# --- Logging ---
# The handbook requires one JSON line per log event written to stdout.

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        skip = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                payload[key] = value
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(_JSONFormatter())
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().handlers = [handler]
logger = logging.getLogger("mp194.api")


# --- Model ---
# Loaded once at import time so TestClient and production both work.

MODEL_PATH = Path("models/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")

_model = joblib.load(MODEL_PATH)
logger.info("Model loaded", extra={"model_type": type(_model).__name__})


# --- Lifespan ---

@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("API starting up")
    yield
    logger.info("API shutting down")


# --- App ---

app = FastAPI(
    title="MP194 Household Electricity Prediction API",
    description="Predict monthly household electricity consumption (kWh) for Kochi.",
    version="1.0.0",
    lifespan=_lifespan,
)


# --- Request schema ---
# Fields from PROBLEM.md section 5. Validation rules reject bad input with 422.

class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    rooms: int
    members: int
    ac_hours_per_day: float = Field(le=25)
    avg_temp_c: float
    house_area_sqft: float = Field(ge=150)
    solar_kw: float
    connection: str
    feeder: str


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "message": "MP194 Electricity Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    logger.info(
        "Prediction request",
        extra={
            "rooms": request.rooms,
            "members": request.members,
            "connection": request.connection,
            "feeder": request.feeder,
        },
    )

    input_df = pd.DataFrame([request.model_dump()])
    prediction = round(float(_model.predict(input_df)[0]), 2)

    logger.info("Prediction done", extra={"prediction_kwh": prediction})

    return {"prediction": prediction, "unit": "kWh", "status": "success"}


# --- Error handler ---

@app.exception_handler(Exception)
async def _error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error", extra={"error": type(exc).__name__})
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )
