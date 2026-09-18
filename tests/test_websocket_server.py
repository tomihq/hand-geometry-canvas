"""Camera-free WebSocketServer integration tests."""

from __future__ import annotations

import asyncio
import json
import socket
import time

import pytest

from hand_interaction.transport.serialize import event_to_dict
from hand_interaction.transport.websocket_server import WebSocketServer
from hand_interaction.types import HandMove, PinchStart, Vector2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def server() -> WebSocketServer:
    srv = WebSocketServer(host="127.0.0.1", port=_free_port())
    srv.start()
    yield srv
    srv.stop()


async def _recv_one(url: str, timeout: float = 2.0) -> dict:
    import websockets

    async with websockets.connect(url) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(raw)


async def _recv_n(url: str, n: int, timeout: float = 2.0) -> list[dict]:
    import websockets

    async with websockets.connect(url) as ws:
        out: list[dict] = []
        for _ in range(n):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            out.append(json.loads(raw))
        return out


def test_publish_before_start_is_noop() -> None:
    srv = WebSocketServer(host="127.0.0.1", port=_free_port())
    srv.publish(PinchStart("Right", Vector2(0.1, 0.2)))  # must not raise


def test_double_start_raises(server: WebSocketServer) -> None:
    with pytest.raises(RuntimeError, match="already started"):
        server.start()


def test_double_stop_is_noop(server: WebSocketServer) -> None:
    server.stop()
    server.stop()


def test_single_client_receives_event(server: WebSocketServer) -> None:
    event = PinchStart("Right", Vector2(0.2, 0.3))
    url = server.url

    async def run() -> dict:
        import websockets

        async with websockets.connect(url) as ws:
            await asyncio.sleep(0.05)
            server.publish(event)
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            return json.loads(raw)

    payload = asyncio.run(run())
    assert payload == event_to_dict(event)


def test_multiple_clients_receive_same_event(server: WebSocketServer) -> None:
    event = HandMove("Left", Vector2(0.5, 0.5), Vector2(0.01, 0.0))
    url = server.url
    barrier = asyncio.Event()

    async def client() -> dict:
        import websockets

        async with websockets.connect(url) as ws:
            barrier.set()
            await asyncio.sleep(0.05)
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            return json.loads(raw)

    async def run() -> tuple[dict, dict]:
        t1 = asyncio.create_task(client())
        t2 = asyncio.create_task(client())
        # Wait until at least one connected, then give both time to join.
        await asyncio.wait_for(barrier.wait(), timeout=2.0)
        await asyncio.sleep(0.1)
        server.publish(event)
        return await t1, await t2

    a, b = asyncio.run(run())
    expected = event_to_dict(event)
    assert a == expected
    assert b == expected


def test_client_disconnect_does_not_break_others(server: WebSocketServer) -> None:
    event = PinchStart("Right", Vector2(0.4, 0.6))
    url = server.url

    async def run() -> dict:
        import websockets

        ws_a = await websockets.connect(url)
        ws_b = await websockets.connect(url)
        await asyncio.sleep(0.05)
        await ws_a.close()
        await asyncio.sleep(0.05)
        server.publish(event)
        raw = await asyncio.wait_for(ws_b.recv(), timeout=2.0)
        await ws_b.close()
        return json.loads(raw)

    payload = asyncio.run(run())
    assert payload == event_to_dict(event)


def test_start_stop_releases_port() -> None:
    port = _free_port()
    srv = WebSocketServer(host="127.0.0.1", port=port)
    srv.start()
    assert srv.port == port
    srv.stop()

    # Allow OS to fully release the socket.
    time.sleep(0.05)
    again = WebSocketServer(host="127.0.0.1", port=port)
    again.start()
    try:
        assert again.port == port
    finally:
        again.stop()


def test_fake_source_wires_to_server(server: WebSocketServer) -> None:
    from hand_interaction import FakeHandSource, create_hand_gesture

    fake = FakeHandSource()
    gesture = create_hand_gesture(source=fake)
    gesture.on_event(server.publish)
    gesture.start()

    event = PinchStart("Right", Vector2(0.7, 0.2))
    url = server.url

    async def run() -> dict:
        import websockets

        async with websockets.connect(url) as ws:
            await asyncio.sleep(0.05)
            fake.emit(event)
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            return json.loads(raw)

    try:
        payload = asyncio.run(run())
    finally:
        gesture.stop()

    assert payload == event_to_dict(event)
