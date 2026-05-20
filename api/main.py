from fastapi import FastAPI
from api.routers import pagerduty
from config import settings

app = FastAPI(
    title="AIRS Production API",
    description="Production ingestion and ChatOps server for the Autonomous Incident Response System.",
    version="1.0.0",
)

app.include_router(pagerduty.router)
# Future ChatOps routers will be included here
# app.include_router(slack.router)

@app.get("/health", tags=["Meta"])
async def health_check():
    """Healthcheck endpoint for Kubernetes or Load Balancers."""
    return {"status": "ok", "environment": "production"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
