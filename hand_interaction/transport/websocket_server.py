"""Thread-safe WebSocket broadcaster for HandEvent."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from hand_interaction.transport.serialize import event_to_json
from hand_interaction.types import HandEvent

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


class WebSocketServer:
    """Broadcast HandEvents as JSON to all connected WebSocket clients.

    ``publish`` is safe to call from any thread (including the camera loop).
    Dead or failing clients are dropped without affecting other clients or
    the hand-tracking pipeline.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._clients: set[Any] = set()
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._started = False
        self._lock = threading.Lock()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        """Bound port once started; configured port before start."""
        if self._server is not None:
            sockets = getattr(self._server, "sockets", None) or []
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self._port

    @property
    def url(self) -> str:
        return f"ws://{self._host}:{self.port}"

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("WebSocketServer already started")
            self._started = True
            self._ready.clear()
            self._start_error = None
            self._thread = threading.Thread(
                target=self._run_loop,
                name="hand-interaction-ws",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("WebSocketServer failed to start")
        if self._start_error is not None:
            err = self._start_error
            self.stop()
            raise RuntimeError(f"WebSocketServer failed to start: {err}") from err

    def stop(self) -> None:
        with self._lock:
            if not self._started and self._thread is None:
                return
            loop = self._loop
            thread = self._thread
            self._started = False

        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            try:
                fut.result(timeout=5.0)
            except Exception:
                logger.exception("error during WebSocketServer shutdown")
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None:
            thread.join(timeout=5.0)

        with self._lock:
            self._loop = None
            self._thread = None
            self._server = None
            self._clients.clear()

    def publish(self, event: HandEvent) -> None:
        """Serialize and broadcast ``event``. No-op if the server is not running."""
        loop = self._loop
        if loop is None or not loop.is_running() or not self._started:
            return
        try:
            payload = event_to_json(event)
        except Exception:
            logger.exception("failed to serialize HandEvent")
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
        except RuntimeError:
            # Loop closed between check and schedule.
            return

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
            self._ready.set()
            loop.run_forever()
        except Exception as exc:
            logger.exception("WebSocketServer loop crashed")
            self._start_error = exc
            self._ready.set()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()

    async def _serve(self) -> None:
        import websockets

        async def handler(websocket: Any) -> None:
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self._clients.discard(websocket)

        self._server = await websockets.serve(
            handler,
            self._host,
            self._port,
        )

    async def _broadcast(self, payload: str) -> None:
        clients = list(self._clients)
        if not clients:
            return
        dead: list[Any] = []
        for client in clients:
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)
            try:
                await client.close()
            except Exception:
                pass

    async def _shutdown(self) -> None:
        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
