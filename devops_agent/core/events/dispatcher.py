"""Event Bus for Telemetry and State Transitions."""

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    investigation_id: str

class StageTransitionEvent(BaseEvent):
    event_type: str = "stage_transition"
    from_node: str
    to_node: str
    reason: str = "normal"

class GuardrailTriggerEvent(BaseEvent):
    event_type: str = "guardrail_trigger"
    guardrail_type: str
    message: str

class EventDispatcher:
    """Asynchronous event bus for system telemetry."""
    
    _instance = None
    
    def __init__(self):
        self.handlers: dict[str, list[Callable[[BaseEvent], None]]] = {}
        
    @classmethod
    def get_instance(cls) -> 'EventDispatcher':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def dispatch(self, event: BaseEvent) -> None:
        """Dispatches an event to all registered handlers for its type."""
        for handler in self.handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                pass
