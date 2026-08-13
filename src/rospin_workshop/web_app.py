from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import TaskSessionConflictError, WorkshopController
from rospin_workshop.deployment import PolicyDeploymentManager
from rospin_workshop.trajectory.runner import TrajectoryManager


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime = config or RuntimeConfig()
    controller = WorkshopController(runtime)
    trajectory_manager = TrajectoryManager(controller, runtime.trajectories_root)
    policy_manager = PolicyDeploymentManager(controller, runtime.outputs_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        controller.start()
        try:
            # A single-task workshop has no selection ambiguity. Initialize its
            # MuJoCo environments and camera contexts before Uvicorn reports
            # application startup complete, so that log line means the whole
            # workshop—not only the HTTP socket—is ready.
            only_task = controller.task_registry.only()
            if only_task is not None:
                await asyncio.to_thread(controller.select_task, only_task.id)
            yield
        finally:
            policy_manager.close()
            trajectory_manager.close()
            controller.close()

    app = FastAPI(title="ROSpin SO-101 Workshop", lifespan=lifespan)
    app.state.controller = controller
    app.state.trajectory_manager = trajectory_manager
    app.state.policy_manager = policy_manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = (
            files("rospin_workshop")
            .joinpath("web/index.html")
            .read_text(encoding="utf-8")
        )
        return HTMLResponse(page)

    async def mjpeg(camera: str):
        while True:
            frame = controller.camera_jpeg(camera)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame
                + b"\r\n"
            )
            await asyncio.sleep(1.0 / runtime.camera_hz)

    @app.get("/camera/{camera}.mjpg")
    async def camera_stream(camera: str) -> StreamingResponse:
        if camera not in ("wrist", "perspective"):
            raise HTTPException(status_code=404, detail="Unknown camera")
        if not controller.status().get("task_ready"):
            raise HTTPException(status_code=503, detail="Select a task first")
        return StreamingResponse(
            mjpeg(camera), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return controller.status()

    @app.get("/api/tasks")
    async def tasks() -> list[dict[str, Any]]:
        return controller.tasks()

    @app.get("/api/trajectory/status")
    async def trajectory_status() -> dict[str, Any]:
        return trajectory_manager.status()

    @app.post("/api/trajectory/start")
    async def start_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
        program = payload.get("program")
        if not isinstance(program, str) or not program.strip():
            raise HTTPException(status_code=422, detail="program is required")
        if policy_manager.status()["running"]:
            raise HTTPException(
                status_code=409,
                detail="A policy deployment is running",
            )
        try:
            return await asyncio.to_thread(
                trajectory_manager.start,
                program_path=program.strip(),
                episodes=int(payload.get("episodes", 1)),
                seed=int(payload.get("seed", 0)),
                preview=bool(payload.get("preview", False)),
                dataset_name=str(
                    payload.get("dataset_name", "synthetic_trajectory")
                ),
                preflight=bool(payload.get("preflight", True)),
                resume_dataset=(
                    str(payload["resume_dataset"])
                    if payload.get("resume_dataset")
                    else None
                ),
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/trajectory/stop")
    async def stop_trajectory() -> dict[str, Any]:
        return trajectory_manager.stop()

    @app.get("/api/policy/status")
    async def policy_status() -> dict[str, Any]:
        return policy_manager.status()

    @app.post("/api/policy/start")
    async def start_policy(payload: dict[str, Any]) -> dict[str, Any]:
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise HTTPException(status_code=422, detail="checkpoint is required")
        if trajectory_manager.status()["running"]:
            raise HTTPException(
                status_code=409,
                detail="A trajectory program is running",
            )
        try:
            return await asyncio.to_thread(
                policy_manager.start,
                checkpoint=checkpoint.strip(),
                episodes=int(payload.get("episodes", 1)),
                seed=int(payload.get("seed", 0)),
                device=str(payload.get("device", "cpu")),
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/policy/stop")
    async def stop_policy() -> dict[str, Any]:
        return policy_manager.stop()

    @app.post("/api/session/task")
    async def select_task(payload: dict[str, Any]) -> dict[str, Any]:
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise HTTPException(status_code=422, detail="task_id is required")
        try:
            await asyncio.to_thread(controller.select_task, task_id.strip())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TaskSessionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return controller.status()

    @app.websocket("/ws")
    async def control_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "key":
                    controller.set_key(
                        str(message.get("key", "")), bool(message.get("pressed"))
                    )
                elif message_type == "clear_keys":
                    controller.clear_keys()
                elif message_type == "camera":
                    try:
                        controller.control_perspective_camera(
                            str(message.get("action", "")),
                            dict(message),
                        )
                    except Exception as exc:  # noqa: BLE001 - report input errors
                        await websocket.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                elif message_type == "command":
                    try:
                        await asyncio.to_thread(
                            controller.command,
                            str(message.get("command", "")),
                            dict(message.get("payload", {})),
                        )
                    except Exception as exc:  # noqa: BLE001 - report command errors
                        await websocket.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                await websocket.send_json(
                    {"type": "status", "data": controller.status()}
                )
        except WebSocketDisconnect:
            controller.clear_keys()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run browser-based SO-101 simulation and teleoperation"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    config = RuntimeConfig()
    uvicorn.run(
        create_app(config),
        host=args.host or config.host,
        port=args.port or config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
