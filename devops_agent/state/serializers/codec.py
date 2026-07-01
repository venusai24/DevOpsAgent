"""Type codec registry for safe JSON serialisation of all state model types.

Design constraints (from Phase 3 §7.4):
- datetime → ISO 8601 string (always UTC-aware).
- UUID → lowercase hex string without dashes.
- Enum → .value string.
- tuple → JSON array (preserves list round-trip with explicit decoding).
- Frozen dataclasses → recursive dict serialisation.
- Optional[T] → None or T.

No business logic belongs here.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


def encode(obj: Any) -> Any:
    """Recursively encode a Python object to a JSON-safe representation."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [encode(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    # Frozen dataclass / BaseState — use __dataclass_fields__ or to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {name: encode(getattr(obj, name)) for name in obj.__dataclass_fields__}
    raise TypeError(f"Cannot encode type {type(obj)!r}: {obj!r}")


def decode_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def decode_uuid(value: Any) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def decode_enum(enum_class: type[Enum], value: Any) -> Enum | None:
    if value is None:
        return None
    if isinstance(value, enum_class):
        return value
    return enum_class(value)


def to_json(obj: Any) -> str:
    """Serialise any state object to a compact JSON string."""
    return json.dumps(encode(obj), separators=(",", ":"))


def from_json(s: str) -> Any:
    """Parse a JSON string back to Python primitives."""
    return json.loads(s)


def to_jsonb_dict(obj: Any) -> dict[str, Any]:
    """Serialise to a JSON-safe dict suitable for insertion into a JSONB column."""
    encoded = encode(obj)
    if not isinstance(encoded, dict):
        raise TypeError(f"Expected dict, got {type(encoded)}")
    return encoded
