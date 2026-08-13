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
        assert ".perspective-card { width: 642px;" in page
        assert "grid-template-columns: repeat(2, 642px)" in page
        assert "Wrist camera · recorded" in page
        assert "640 × 480 · dataset source" in page
        assert 'id="wristCamera" class="camera wrist-camera"' in page
        assert 'data-src="/camera/wrist.mjpg"' in page
        assert 'id="taskTitle"' in page
        assert 'id="successProgress"' in page
        assert 'id="policyCheckpoint"' in page
        assert 'id="policyStartButton"' in page
        assert "ACT policy deployment" in page
        assert 'width="640" height="480"' in page
        views_start = page.index('<section class="views">')
        views_end = page.index("</section>", views_start)
        controls_start = page.index('<aside id="controlRail"')
        teleop_start = page.index('id="teleopPanel"')
        recording_start = page.index("Local recording")
        status_start = page.index("Session status")
        assert views_end < controls_start
        assert controls_start < teleop_start < recording_start < status_start
        assert 'id="controlRail" class="bottom-panels"' in page
        assert "max-height: none" in page
        assert "overflow: visible" in page
        assert 'id="keyboardToggle"' not in page
        assert "physical SO-101 remote" not in page
        assert "Initializing simulation" in page
        assert "Waiting for simulation" in page
        assert "retryDelay = Math.min(3000, retryDelay * 2)" in page
        assert "if (!next.task_ready) ensureTask()" in page
        assert 'cameraDrag.action === "orbit" ? -1 : 1' in page
        assert "direction * dx, direction * dy" in page

        ready = client.get("/api/status").json()
        assert ready["task_ready"] is True
        assert ready["task_id"] == "cube_in_bowl"
        tasks = client.get("/api/tasks").json()
        assert [task["id"] for task in tasks] == ["cube_in_bowl"]
        trajectory_status = client.get("/api/trajectory/status")
        assert trajectory_status.status_code == 200
        assert trajectory_status.json()["running"] is False
        policy_status = client.get("/api/policy/status")
        assert policy_status.status_code == 200
        assert policy_status.json()["running"] is False
        assert client.post("/api/policy/start", json={}).status_code == 422
        assert client.post("/api/policy/stop", json={}).status_code == 200
        assert client.post("/api/trajectory/start", json={}).status_code == 422
        assert (
            client.post(
                "/api/trajectory/start",
                json={"program": "missing.py", "preview": True},
            ).status_code
            == 422
        )
        selected = client.post(
            "/api/session/task", json={"task_id": "cube_in_bowl"}
        )
        assert selected.status_code == 200
        assert client.post(
            "/api/session/task", json={"task_id": "cube_in_bowl"}
        ).status_code == 200
        assert client.post(
            "/api/session/task", json={"task_id": "missing_task"}
        ).status_code == 404
        initial = selected.json()
        assert initial["error"] is None
        assert initial["task_ready"] is True
        assert initial["task_id"] == "cube_in_bowl"
        assert initial["active_control_source"] == "keyboard"
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

        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "key", "key": "[", "pressed": True})
            socket.receive_json()
            socket.send_json({"type": "key", "key": "[", "pressed": False})
            socket.receive_json()
            time.sleep(0.6)
            closed = client.get("/api/status").json()
            assert closed["joint_targets"][5] < -0.17
            assert abs(closed["gripper_force_nm"]) <= 0.0801
