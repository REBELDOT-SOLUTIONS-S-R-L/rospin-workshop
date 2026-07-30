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
        assert translated["eef_position"][1] > initial["eef_position"][1] + 0.005

        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "a", "pressed": True})
            socket.receive_json()
            time.sleep(0.5)
            socket.send_json({"type": "key", "key": "a", "pressed": False})
            socket.receive_json()
        translated_x = client.get("/api/status").json()
        assert translated_x["eef_position"][0] > translated["eef_position"][0] + 0.005

        time.sleep(0.1)
        translated = client.get("/api/status").json()
        before_joints = np.asarray(translated["joint_positions"])
        before_targets = np.asarray(translated["joint_targets"])
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "j", "pressed": True})
            socket.receive_json()
            time.sleep(0.5)
            active = client.get("/api/status").json()
            socket.send_json({"type": "key", "key": "j", "pressed": False})
            socket.receive_json()
        rotated = client.get("/api/status").json()
        after_joints = np.asarray(rotated["joint_positions"])
        assert after_joints[4] > before_joints[4] + 0.01
        active_targets = np.asarray(active["joint_targets"])
        target_changes = np.flatnonzero(
            np.abs(active_targets[:5] - before_targets[:5]) > 0.01
        )
        assert target_changes.tolist() == [4]

        gripper_start = rotated["joint_positions"][5]
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "[", "pressed": True})
            socket.receive_json()
            socket.send_json({"type": "key", "key": "[", "pressed": False})
            socket.receive_json()
        time.sleep(0.6)
        closed = client.get("/api/status").json()
        assert closed["joint_targets"][5] < -0.17
        assert closed["joint_positions"][5] < gripper_start - 0.2
