"""
Plugin system for data providers, brokers, and integrations.
"""

from .base import Plugin, PluginConfig, PluginStatus
from .massive import MassivePlugin, MassivePluginConfig, NewsArticle
from .registry import PluginRegistry

__all__ = [
    "MassivePlugin",
    "MassivePluginConfig",
    "NewsArticle",
    "Plugin",
    "PluginConfig",
    "PluginRegistry",
    "PluginStatus",
]
