"""Reusable Pydantic validators for tool input schemas."""

from datetime import datetime

FORMAT_HINT = (
    "Use ISO 8601 (e.g. '2026-07-22T14:30:00Z') or a Unix epoch "
    "timestamp in seconds (e.g. 1753185000)."
)

def parse_timestamp(value: str | int | float, field_name: str) -> float:
    """Parse a string or float into a Unix timestamp (seconds)."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        raise ValueError(f"'{field_name}'={value!r} is not a recognized timestamp. {FORMAT_HINT}")
