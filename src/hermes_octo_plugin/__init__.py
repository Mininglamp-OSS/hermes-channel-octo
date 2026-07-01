"""
Octo (WuKongIM) platform adapter for Hermes Agent.

Provides a BasePlatformAdapter subclass that connects to Octo's
WuKongIM-based messaging infrastructure via WebSocket binary protocol,
enabling bot-to-user and bot-to-group messaging.
"""

from .adapter import OctoAdapter, register
from .types import ChannelType, MessageType

__all__ = ["OctoAdapter", "register", "ChannelType", "MessageType"]
__version__ = "0.1.2"
