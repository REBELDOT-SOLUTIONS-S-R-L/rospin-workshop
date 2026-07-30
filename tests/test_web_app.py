from __future__ import annotations

import time

import numpy as np
from fastapi.testclient import TestClient

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.web_app import create_app


def test_browser_websocket_keys_move_and_rotate_eef(tmp_path) -> None:
    app = create_app(
        RuntimeConfig(
            data_root=tmp_path,
            control_hz=20,
            camera_hz=5,
            image_width=96,
            image_height=72,
        )
    )
    with TestClient(app) as client:
        initial = client.get("/api/status").json()
        assert initial["error"] is None

        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "w", "pressed": True})
            socket.receive_json()
            time.sleep(0.5)
            socket.send_json({"type": "key", "key": "w", "pressed": False})
            socket.receive_json()
        translated = client.get("/api/status").json()
        assert translated["eef_position"][0] > initial["eef_position"][0] + 0.005

        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "u", "pressed": True})
            socket.receive_json()
            time.sleep(0.5)
            socket.send_json({"type": "key", "key": "u", "pressed": False})
            socket.receive_json()
        rotated = client.get("/api/status").json()
        before = np.asarray(translated["eef_orientation"])
        after = np.asarray(rotated["eef_orientation"])
        angle = 2 * np.arccos(np.clip(abs(np.dot(before, after)), 0.0, 1.0))
        assert angle > 0.05
