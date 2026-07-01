"""Dependency Injection Container and Application Wiring."""

import logging
from typing import Optional

from ..orchestrator.graph import build_investigation_graph
from ..state.config import PersistenceConfig

# Import Persistence layer 
from ..state.factories import RepositoryFactory
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from .config.guardrails_config import GuardrailsConfig
from .events.dispatcher import EventDispatcher
from .orchestrator.budget.investigation_clock import InvestigationClock
from .recovery.circuit_breaker.circuit_breaker_registry import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

class AppContainer:
    _instance: Optional['AppContainer'] = None

    def __init__(self):
        # 1. Configuration
        self.config = GuardrailsConfig()
        
        # 2. Event Bus
        self.event_bus = EventDispatcher.get_instance()
        
        # 3. Guardrails & Infrastructure
        self.circuit_breakers = CircuitBreakerRegistry.get_instance(self.config.circuit_breaker)
        self.clock = InvestigationClock(self.config.timeouts)
        
        # 4. Persistence
        # Initialize the state persistence subsystem
        self.persistence_config = PersistenceConfig.from_env()
        self.repository_factory = RepositoryFactory(self.persistence_config)
        
        # 5. Tools
        self.tool_registry = ToolRegistry.get_instance()
        self.tool_executor = ToolExecutor(self.tool_registry)
        
        # 6. Orchestration Graph
        # Checkpointer would be passed here (e.g., MemorySaver or PostgresSaver)
        self.graph = build_investigation_graph(checkpointer=None)

    @classmethod
    def get_instance(cls) -> 'AppContainer':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Run startup hooks, database migrations, and tool discovery."""
        logger.info("Initializing Application Container...")
        self.tool_registry.discover_tools("devops_agent.tools.interfaces")
        logger.info(f"Discovered {len(self.tool_registry.list_tools())} tools.")
