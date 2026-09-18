"""Transport layer: WebSocket broadcast of HandEvent JSON."""

from __future__ import annotations

from hand_interaction.transport.serialize import event_to_dict, event_to_json
from hand_interaction.transport.websocket_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WebSocketServer,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "WebSocketServer",
    "event_to_dict",
    "event_to_json",
]
