FROM python:3.12-slim

LABEL maintainer="AIRS v2 Team"
LABEL description="AIRS ML Sidecar — Isolation Forest Anomaly Detection"
LABEL version="1.0.0"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN groupadd --gid 1001 airs && \
    useradd --uid 1001 --gid airs --shell /bin/bash --create-home airs

WORKDIR /app

# Install dependencies — separate layer for better layer caching.
# scikit-learn wheels are pre-built for slim, so this is fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py features.py model.py ./

# Switch to non-root user
USER airs

# FastAPI served by uvicorn on port 8080
EXPOSE 8080

# Uvicorn with 1 worker (single-process — model state is in-process).
# Do NOT use multiple workers: the model training buffer is not shared
# across processes without additional coordination (Redis, etc.).
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
