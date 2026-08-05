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


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime = config or RuntimeConfig()
    controller = WorkshopController(runtime)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        controller.start()
        try:
            yield
        finally:
            controller.close()

    app = FastAPI(title="ROSpin SO-101 Workshop", lifespan=lifespan)
    app.state.controller = controller

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
                elif message_type == "keyboard_control":
                    controller.set_keyboard_enabled(bool(message.get("enabled")))
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
