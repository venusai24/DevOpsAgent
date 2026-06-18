"""
airs_ml_sidecar/main.py
========================

FastAPI service for AIRS Isolation Forest anomaly scoring.

Endpoints
---------
GET  /healthz          — Liveness probe (always 200 OK)
GET  /readyz           — Readiness probe (200 once model warm-up complete)
GET  /metrics          — Prometheus metrics (scraped by PodMonitor)
GET  /scores           — All current anomaly scores (JSON, for debugging)
GET  /scores/{service} — Score for a specific service
POST /ingest           — Internal: ingest a batch of feature vectors

Background loops
----------------
  ScrapeLoop: Every SCRAPE_INTERVAL_SECONDS, calls PrometheusFeatureExtractor
              to pull the latest RED metrics for all services, adds them to
              the model's training buffer, and updates Prometheus gauges.

Deployment
----------
Runs as a single-replica Deployment in the observability namespace.
The main PrometheusAnomalyEngine queries this sidecar via:
  http://airs-ml-sidecar.observability.svc.cluster.local:8080/scores
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from features import PrometheusFeatureExtractor
from model import IsolationForestModel, AnomalyScore

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus.monitoring.svc.cluster.local:9090",
)
NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "15"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("airs_ml_sidecar")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: IsolationForestModel | None = None
_extractor: PrometheusFeatureExtractor | None = None
_latest_scores: dict[str, AnomalyScore] = {}
_scrape_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Background scrape loop
# ---------------------------------------------------------------------------

async def _scrape_loop() -> None:
    """
    Continuously scrapes Prometheus metrics, updates the IF model, and
    refreshes the anomaly score cache for all services.
    """
    global _latest_scores

    while True:
        try:
            t_start = time.monotonic()
            vectors = _extractor.extract_all_services()

            for vec in vectors:
                feat = vec.to_array()
                _model.add_observation(feat)
                score = _model.score(
                    feat,
                    service_name=vec.service_name,
                    namespace=vec.namespace,
                )
                _latest_scores[vec.service_name] = score

            elapsed = (time.monotonic() - t_start) * 1000
            logger.info(
                "[ScrapeLoop] Scored %d services in %.0fms (model_ready=%s)",
                len(vectors), elapsed, _model.is_ready,
            )
        except asyncio.CancelledError:
            logger.info("[ScrapeLoop] Cancelled — exiting")
            return
        except Exception as exc:
            logger.warning("[ScrapeLoop] Error: %s", exc)

        await asyncio.sleep(SCRAPE_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _extractor, _scrape_task

    logger.info("AIRS ML Sidecar starting: namespace=%s, prom=%s", NAMESPACE, PROMETHEUS_URL)

    _extractor = PrometheusFeatureExtractor(
        prometheus_url=PROMETHEUS_URL,
        namespace=NAMESPACE,
    )
    _model = IsolationForestModel()

    _scrape_task = asyncio.create_task(_scrape_loop())
    logger.info("Scrape loop started (interval=%ds)", SCRAPE_INTERVAL_SECONDS)

    yield  # Application runs here

    if _scrape_task and not _scrape_task.done():
        _scrape_task.cancel()
        try:
            await _scrape_task
        except asyncio.CancelledError:
            pass
    logger.info("AIRS ML Sidecar shut down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AIRS ML Sidecar — Isolation Forest Anomaly Detector",
    description=(
        "Exposes Isolation Forest anomaly scores for Kubernetes services "
        "as Prometheus metrics and a REST API. Integrated with the AIRS v2 "
        "PrometheusAnomalyEngine for composite Z-Score + IF detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", summary="Liveness probe")
async def healthz():
    """Always 200 OK — confirms the process is running."""
    return {"status": "ok"}


@app.get("/readyz", summary="Readiness probe")
async def readyz():
    """
    200 OK once the Isolation Forest has completed its warm-up window.
    503 Service Unavailable during warm-up (model not yet ready).
    """
    if _model is None or not _model.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model warming up — not yet ready to produce scores",
        )
    return {"status": "ready", "model_ready": True}


@app.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics():
    """
    Expose Prometheus metrics in text format.
    Scraped by the PodMonitor in 11-airs-ml-sidecar.yaml.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        content = generate_latest()
        return PlainTextResponse(content=content, media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return PlainTextResponse(
            content="# prometheus-client not installed\n",
            status_code=503,
        )


@app.get("/scores", summary="All current anomaly scores")
async def get_all_scores():
    """
    Return the latest anomaly scores for all discovered services.
    Used by PrometheusAnomalyEngine for composite decision fusion.
    """
    if not _latest_scores:
        return JSONResponse(
            content={"scores": [], "model_ready": _model.is_ready if _model else False},
        )
    scores = [
        {
            "service_name": s.service_name,
            "namespace": s.namespace,
            "score": s.score,
            "is_anomalous": s.is_anomalous,
            "model_ready": s.model_ready,
            "timestamp": s.timestamp,
        }
        for s in _latest_scores.values()
    ]
    return JSONResponse(content={"scores": scores, "model_ready": _model.is_ready if _model else False})


@app.get("/scores/{service_name}", summary="Score for a specific service")
async def get_service_score(service_name: str):
    """
    Return the anomaly score for a specific service.
    Returns 404 if the service has no score yet (not seen by Prometheus).
    """
    score = _latest_scores.get(service_name)
    if score is None:
        raise HTTPException(
            status_code=404,
            detail=f"No score available for service '{service_name}'. "
                   f"Is it visible in Prometheus?",
        )
    return {
        "service_name": score.service_name,
        "namespace": score.namespace,
        "score": score.score,
        "is_anomalous": score.is_anomalous,
        "model_ready": score.model_ready,
        "timestamp": score.timestamp,
    }
