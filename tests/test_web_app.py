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
        page = client.get("/").text
        for key in ("r", "f", "t", "g"):
            assert f'data-key="{key}"' in page
        assert '"r", "f", "t", "g"' in page
        assert 'id="perspectiveViewport"' in page
        assert 'id="cameraResetButton"' in page
        assert "drag orbit · Shift+drag pan · wheel zoom" in page
        assert "Perspective · viewer only" in page
        assert "Gripper camera · recorded" in page
        assert 'id="keyboardToggle"' in page
        assert "Turn off to follow the physical SO-101 remote." in page

        initial = client.get("/api/status").json()
        assert initial["error"] is None
        assert initial["keyboard_enabled"] is True
        assert initial["active_control_source"] == "keyboard"
        assert initial["remote"]["configured"] is False
        initial_camera = initial["perspective_camera"]

        with client.websocket_connect("/ws") as socket:
            socket.send_json(
                {"type": "camera", "action": "orbit", "dx": 80, "dy": -20}
            )
            orbit = socket.receive_json()["data"]["perspective_camera"]
            assert orbit["azimuth_degrees"] != initial_camera["azimuth_degrees"]
            assert orbit["elevation_degrees"] != initial_camera["elevation_degrees"]

            socket.send_json(
                {"type": "camera", "action": "pan", "dx": 25, "dy": -10}
            )
            pan = socket.receive_json()["data"]["perspective_camera"]
            assert pan["lookat"] != orbit["lookat"]

            socket.send_json({"type": "camera", "action": "zoom", "delta": -100})
            zoom = socket.receive_json()["data"]["perspective_camera"]
            assert zoom["distance"] < pan["distance"]

            socket.send_json({"type": "camera", "action": "reset"})
            reset = socket.receive_json()["data"]["perspective_camera"]
            assert reset == initial_camera

        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "w", "pressed": True})
            socket.receive_json()
            time.sleep(0.5)
            socket.send_json({"type": "key", "key": "w", "pressed": False})
            socket.receive_json()
        translated = client.get("/api/status").json()
        assert translated["eef_position"][1] < initial["eef_position"][1] - 0.005

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

        for key, joint_index in (("r", 1), ("t", 2)):
            time.sleep(0.1)
            before = client.get("/api/status").json()
            before_positions = np.asarray(before["joint_positions"])
            before_targets = np.asarray(before["joint_targets"])
            with client.websocket_connect("/ws") as socket:
                socket.send_json({"type": "key", "key": key, "pressed": True})
                socket.receive_json()
                time.sleep(0.4)
                active = client.get("/api/status").json()
                socket.send_json({"type": "key", "key": key, "pressed": False})
                socket.receive_json()
            after = client.get("/api/status").json()
            assert after["joint_positions"][joint_index] > (
                before_positions[joint_index] + 0.005
            )
            active_targets = np.asarray(active["joint_targets"])
            target_changes = np.flatnonzero(
                np.abs(active_targets[:5] - before_targets[:5]) > 0.01
            )
            assert target_changes.tolist() == [joint_index]

        gripper_start = client.get("/api/status").json()["joint_positions"][5]
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "[", "pressed": True})
            socket.receive_json()
            socket.send_json({"type": "key", "key": "[", "pressed": False})
            socket.receive_json()
        time.sleep(0.6)
        closed = client.get("/api/status").json()
        assert closed["joint_targets"][5] < -0.17
        assert closed["joint_positions"][5] < gripper_start - 0.2


def test_websocket_can_disable_and_reenable_keyboard_controls(tmp_path) -> None:
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
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "keyboard_control", "enabled": False})
            disabled = socket.receive_json()["data"]
            assert disabled["keyboard_enabled"] is False
            assert disabled["keys"] == []

            socket.send_json({"type": "key", "key": "w", "pressed": True})
            ignored = socket.receive_json()["data"]
            assert ignored["keys"] == []
            time.sleep(0.25)
            held = client.get("/api/status").json()
            assert held["active_control_source"] == "hold"
            assert np.isclose(
                held["eef_position"][1],
                initial["eef_position"][1],
                atol=0.005,
            )

            socket.send_json({"type": "keyboard_control", "enabled": True})
            enabled = socket.receive_json()["data"]
            assert enabled["keyboard_enabled"] is True
            socket.send_json({"type": "key", "key": "w", "pressed": True})
            socket.receive_json()
            time.sleep(0.4)
            socket.send_json({"type": "key", "key": "w", "pressed": False})
            socket.receive_json()
        moved = client.get("/api/status").json()
        assert moved["active_control_source"] == "keyboard"
        assert moved["eef_position"][1] < held["eef_position"][1] - 0.005
