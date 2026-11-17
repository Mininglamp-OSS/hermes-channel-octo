"""
Octo (WuKongIM) platform adapter for Hermes Agent.

Provides a BasePlatformAdapter subclass that connects to Octo's
WuKongIM-based messaging infrastructure via WebSocket binary protocol,
enabling bot-to-user and bot-to-group messaging.
"""

from .types import ChannelType, MessageType

__all__ = ["OctoAdapter", "register", "ChannelType", "MessageType"]
__version__ = "0.2.0"


def register(ctx) -> None:
    from .adapter import register as register_adapter

    register_adapter(ctx)


def __getattr__(name: str):
    if name == "OctoAdapter":
        from .adapter import OctoAdapter

        return OctoAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
