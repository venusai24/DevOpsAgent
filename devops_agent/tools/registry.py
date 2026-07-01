"""Tool registry and dependency injection."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .base import BaseTool
from .exceptions import ToolNotFoundError

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Central registry for discovering and instantiating tools."""
    
    _instance: ToolRegistry | None = None
    
    @classmethod
    def get_instance(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        name = tool.metadata.name
        if name in self._tools:
            logger.warning("Overwriting existing tool registration for '%s'", name)
        self._tools[name] = tool
        logger.info("Registered tool: %s (v%s)", name, tool.metadata.version)
        
    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name, raising ToolNotFoundError if missing."""
        tool = self._tools.get(name)
        if not tool:
            raise ToolNotFoundError(name)
        return tool

    def get_all(self) -> Sequence[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_tools(self) -> Sequence[BaseTool]:
        """Alias for get_all."""
        return self.get_all()

    def discover_tools(self, package_path: str) -> None:
        """Discover and register all tool classes in a package."""
        import importlib
        import inspect
        import pkgutil

        try:
            package = importlib.import_module(package_path)
            prefix = package.__name__ + "."
            for _, name, is_pkg in pkgutil.walk_packages(package.__path__, prefix):
                module = importlib.import_module(name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTool) and obj is not BaseTool:
                        try:
                            # Instantiate and register the tool
                            self.register(obj())
                        except Exception as e:
                            logger.debug("Could not instantiate tool %s: %s", obj.__name__, e)
        except Exception as e:
            logger.error("Failed to discover tools in %s: %s", package_path, e)

    def get_by_tags(self, tags: set[str]) -> Sequence[BaseTool]:
        """Return tools matching all provided tags."""
        return [
            tool for tool in self._tools.values()
            if tags.issubset(tool.metadata.tags)
        ]

def get_registry() -> ToolRegistry:
    """Get the global tool registry singleton."""
    return ToolRegistry.get_instance()
