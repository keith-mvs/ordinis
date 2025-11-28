"""
Plugin registry for managing plugin lifecycle.
"""

from typing import Dict, List, Optional, Type
import logging
from .base import Plugin, PluginConfig, PluginStatus, PluginHealth

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for managing plugins.

    Handles plugin registration, initialization, and lifecycle management.
    """

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}

    def register_class(self, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin class for later instantiation.

        Args:
            plugin_class: Plugin class to register.
        """
        name = plugin_class.name
        if name in self._plugin_classes:
            logger.warning(f"Overwriting plugin class: {name}")
        self._plugin_classes[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def create_plugin(
        self,
        name: str,
        config: PluginConfig
    ) -> Optional[Plugin]:
        """
        Create a plugin instance from registered class.

        Args:
            name: Plugin class name.
            config: Plugin configuration.

        Returns:
            Plugin instance or None if not found.
        """
        if name not in self._plugin_classes:
            logger.error(f"Plugin class not found: {name}")
            return None

        plugin_class = self._plugin_classes[name]
        plugin = plugin_class(config)
        self._plugins[config.name] = plugin
        logger.info(f"Created plugin instance: {config.name}")
        return plugin

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugin instances."""
        return list(self._plugins.keys())

    def list_available_classes(self) -> List[str]:
        """List all registered plugin classes."""
        return list(self._plugin_classes.keys())

    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered plugins.

        Returns:
            Dict of plugin name to initialization success.
        """
        results = {}
        for name, plugin in self._plugins.items():
            try:
                success = await plugin.initialize()
                results[name] = success
                logger.info(f"Plugin {name} initialized: {success}")
            except Exception as e:
                results[name] = False
                logger.error(f"Plugin {name} initialization failed: {e}")
        return results

    async def shutdown_all(self) -> None:
        """Shutdown all plugins gracefully."""
        for name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
                logger.info(f"Plugin {name} shutdown complete")
            except Exception as e:
                logger.error(f"Plugin {name} shutdown error: {e}")

    async def health_check_all(self) -> Dict[str, PluginHealth]:
        """
        Run health checks on all plugins.

        Returns:
            Dict of plugin name to health status.
        """
        results = {}
        for name, plugin in self._plugins.items():
            try:
                health = await plugin.health_check()
                results[name] = health
            except Exception as e:
                results[name] = PluginHealth(
                    status=PluginStatus.ERROR,
                    last_check=datetime.utcnow(),
                    latency_ms=0,
                    last_error=str(e)
                )
        return results

    def get_plugins_by_status(self, status: PluginStatus) -> List[Plugin]:
        """Get all plugins with a specific status."""
        return [p for p in self._plugins.values() if p.status == status]

    def get_plugins_by_capability(self, capability: str) -> List[Plugin]:
        """Get all plugins with a specific capability."""
        from .base import PluginCapability
        cap = PluginCapability(capability)
        return [p for p in self._plugins.values() if cap in p.capabilities]


# Global registry instance
from datetime import datetime
registry = PluginRegistry()
