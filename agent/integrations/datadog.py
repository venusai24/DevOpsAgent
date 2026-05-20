import time
import json
import logging
from config import settings
try:
    from datadog_api_client import ApiClient, Configuration
    from datadog_api_client.v1.api.metrics_api import MetricsApi
except ImportError:
    ApiClient = None
    Configuration = None
    MetricsApi = None

logger = logging.getLogger(__name__)

async def fetch_datadog_metrics(query: str, time_range: str = "last_15m") -> str:
    """
    Fetches real metrics from Datadog API.
    If the API client is not installed or keys are missing, returns an error string.
    """
    if not settings.DATADOG_API_KEY or not settings.DATADOG_APP_KEY:
        raise ValueError("Datadog API keys not configured.")
        
    if not ApiClient:
        raise ImportError("datadog_api_client not installed")

    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = settings.DATADOG_API_KEY
    configuration.api_key["appKeyAuth"] = settings.DATADOG_APP_KEY
    
    # Calculate timestamps (mock logic for demo)
    now = int(time.time())
    start = now - (15 * 60) # 15 minutes ago
    
    try:
        with ApiClient(configuration) as api_client:
            api_instance = MetricsApi(api_client)
            # Query the active metrics
            response = api_instance.query_metrics(
                _from=start,
                to=now,
                query=query
            )
            # Serialize for LLM digestion
            return json.dumps(response.to_dict(), indent=2)
    except Exception as e:
        logger.error(f"Datadog API error: {e}")
        raise
