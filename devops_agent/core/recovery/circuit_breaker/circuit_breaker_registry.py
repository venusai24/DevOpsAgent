"""Global CB Registry."""

from typing import Optional

from ...config.guardrails_config import CircuitBreakerConfig
from .circuit_breaker import CircuitBreaker


class CircuitBreakerRegistry:
    _instance: Optional['CircuitBreakerRegistry'] = None
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.breakers: dict[str, CircuitBreaker] = {}
        
    @classmethod
    def get_instance(cls, config: CircuitBreakerConfig = None) -> 'CircuitBreakerRegistry':
        if cls._instance is None:
            cls._instance = cls(config or CircuitBreakerConfig())
        return cls._instance

    def get_breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self.breakers:
            self.breakers[tool_name] = CircuitBreaker(tool_name, self.config)
        return self.breakers[tool_name]
