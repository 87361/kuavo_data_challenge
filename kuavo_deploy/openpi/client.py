from __future__ import annotations

import time
from typing import Any

import websockets.sync.client

from kuavo_deploy.openpi import msgpack_numpy


class OpenPIWebsocketClient:
    """Minimal synchronous client for OpenPI's websocket policy server."""

    def __init__(self, host: str, port: int, timeout_s: float = 180.0) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._wait_for_server(timeout_s)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _wait_for_server(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(f"Timed out waiting for OpenPI server at {self._uri}") from last_error

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in OpenPI inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()
