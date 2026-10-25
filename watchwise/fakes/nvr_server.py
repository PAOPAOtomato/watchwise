"""A fake NVR that speaks the subset of Frigate's API watchwise depends on.

Not a mock object -- a real ASGI app. Tests drive the actual ``NVRClient``
against it over an httpx ASGI transport, so URL shapes, query parameters and
error codes are exercised for real. When the client is later pointed at a live
NVR, only the base URL changes.

Endpoints implemented (matching Frigate's HTTP API):
    GET /api/version
    GET /api/config
    GET /api/stats
    GET /api/review
    GET /api/review/{id}
    GET /api/events
    GET /api/events/search
    GET /api/events/{id}
    GET /api/events/{id}/snapshot.jpg
    GET /api/events/{id}/thumbnail.{ext}
    GET /api/{camera}/latest.jpg
    GET /api/{camera}/recordings/{ts}/snapshot.{fmt}
    POST /api/faces/{name}/register
"""

from __future__ import annotations

import io
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response


def make_jpeg(width: int = 64, height: int = 48, shade: int = 128) -> bytes:
    """A tiny valid JPEG, so image handling code paths are real."""
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), (shade, shade, shade))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return buf.getvalue()
    except ImportError:
        # Minimal JPEG header + EOI; enough to be non-empty bytes.
        return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


class FakeNVRState:
    """Mutable fixture state a test can shape before making requests."""

    def __init__(self) -> None:
        now = time.time()
        self.version = "0.17.0-fake"
        self.cameras: dict[str, dict[str, Any]] = {
            "cam_drive": {"enabled": True},
            "cam_door": {"enabled": True},
            "cam_yard": {"enabled": True},
        }
        self.camera_stats: dict[str, dict[str, Any]] = {
            name: {
                "camera_fps": 5.0,
                "process_fps": 5.0,
                "detection_fps": 1.2,
                "skipped_fps": 0.0,
                "pid": 1000 + i,
            }
            for i, name in enumerate(self.cameras)
        }
        # Face Library: name -> images registered; whether a face is in view.
        self.faces: dict[str, int] = {}
        self.face_in_view = True
        self.detector_stats: dict[str, dict[str, Any]] = {
            "ov": {"inference_speed": 8.5, "detection_start": 0.0}
        }
        self.service_stats: dict[str, Any] = {
            "uptime": 3600,
            "version": self.version,
            "storage": {"/media/nvr/recordings": {"free": 100000, "total": 500000}},
        }
        self.reviews: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.recordings_available: bool = True
        self.now = now

    # -------- fixture helpers --------

    def add_event(
        self,
        *,
        event_id: str,
        camera: str,
        label: str = "person",
        start_time: float | None = None,
        duration: float = 8.0,
        zones: list[str] | None = None,
        score: float = 0.82,
        description: str | None = None,
    ) -> dict[str, Any]:
        start = start_time if start_time is not None else self.now
        event = {
            "id": event_id,
            "camera": camera,
            "label": label,
            "sub_label": None,
            "start_time": start,
            "end_time": start + duration,
            "zones": zones or [],
            "score": score,
            "top_score": score,
            "false_positive": False,
            "has_clip": True,
            "has_snapshot": True,
            "data": {"description": description} if description else {},
        }
        self.events.append(event)
        return event

    def add_review(
        self,
        *,
        review_id: str,
        camera: str,
        severity: str = "alert",
        start_time: float | None = None,
        duration: float = 8.0,
        objects: list[str] | None = None,
        zones: list[str] | None = None,
        detections: list[str] | None = None,
    ) -> dict[str, Any]:
        start = start_time if start_time is not None else self.now
        review = {
            "id": review_id,
            "camera": camera,
            "start_time": start,
            "end_time": start + duration,
            "severity": severity,
            "thumb_path": f"/media/nvr/clips/review/{review_id}.webp",
            "data": {
                "objects": objects or ["person"],
                "zones": zones or [],
                "detections": detections or [],
                "sub_labels": [],
            },
        }
        self.reviews.append(review)
        return review


def build_fake_nvr(state: FakeNVRState | None = None) -> tuple[FastAPI, FakeNVRState]:
    state = state or FakeNVRState()
    app = FastAPI(title="fake-nvr")

    @app.get("/api/version", response_class=PlainTextResponse)
    async def version() -> str:
        return state.version

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {"cameras": state.cameras, "mqtt": {"topic_prefix": "nvr"}}

    @app.get("/api/stats")
    async def stats() -> dict[str, Any]:
        return {
            "cameras": state.camera_stats,
            "detectors": state.detector_stats,
            "service": state.service_stats,
        }

    @app.get("/api/review")
    async def list_review(
        cameras: str | None = None,
        after: float | None = None,
        before: float | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        out = state.reviews
        if cameras:
            wanted = set(cameras.split(","))
            out = [r for r in out if r["camera"] in wanted]
        if severity:
            out = [r for r in out if r["severity"] == severity]
        if after is not None:
            out = [r for r in out if r["start_time"] >= after]
        if before is not None:
            out = [r for r in out if r["start_time"] <= before]
        return out[:limit]

    @app.get("/api/review/{review_id}")
    async def get_review(review_id: str) -> dict[str, Any]:
        for r in state.reviews:
            if r["id"] == review_id:
                return r
        raise HTTPException(status_code=404, detail="review not found")

    @app.get("/api/events")
    async def list_events(
        cameras: str | None = None,
        labels: str | None = None,
        zones: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
        include_thumbnails: int = 0,
    ) -> list[dict[str, Any]]:
        out = state.events
        if cameras:
            wanted = set(cameras.split(","))
            out = [e for e in out if e["camera"] in wanted]
        if labels:
            wanted_l = set(labels.split(","))
            out = [e for e in out if e["label"] in wanted_l]
        if zones:
            wanted_z = set(zones.split(","))
            out = [e for e in out if set(e.get("zones") or []) & wanted_z]
        if after is not None:
            out = [e for e in out if e["start_time"] >= after]
        if before is not None:
            out = [e for e in out if e["start_time"] <= before]
        return sorted(out, key=lambda e: e["start_time"])[:limit]

    @app.get("/api/events/search")
    async def search_events(
        query: str = Query(...),
        cameras: str | None = None,
        limit: int = 20,
        search_type: str = "description,thumbnail",
    ) -> list[dict[str, Any]]:
        # Crude lexical stand-in for the CLIP index: enough for the client
        # and tool code paths to be exercised honestly.
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for e in state.events:
            if cameras and e["camera"] not in set(cameras.split(",")):
                continue
            hay = " ".join(
                [
                    e["camera"],
                    e["label"],
                    " ".join(e.get("zones") or []),
                    str((e.get("data") or {}).get("description") or ""),
                ]
            ).lower()
            overlap = len([t for t in terms if t in hay])
            if overlap:
                scored.append((overlap, e))
        scored.sort(key=lambda pair: -pair[0])
        return [
            {**e, "search_distance": round(1.0 / (1 + n), 3)} for n, e in scored[:limit]
        ]

    @app.get("/api/events/{event_id}")
    async def get_event(event_id: str) -> dict[str, Any]:
        for e in state.events:
            if e["id"] == event_id:
                return e
        raise HTTPException(status_code=404, detail="event not found")

    @app.get("/api/events/{event_id}/snapshot.jpg")
    async def event_snapshot(event_id: str, height: int | None = None) -> Response:
        if not any(e["id"] == event_id for e in state.events):
            raise HTTPException(status_code=404, detail="event not found")
        return Response(content=make_jpeg(shade=100), media_type="image/jpeg")

    @app.get("/api/events/{event_id}/thumbnail.{ext}")
    async def event_thumbnail(event_id: str, ext: str) -> Response:
        return Response(content=make_jpeg(32, 24, shade=90), media_type="image/jpeg")

    @app.get("/api/{camera_name}/latest.jpg")
    async def latest(camera_name: str, height: int | None = None) -> Response:
        if camera_name not in state.cameras:
            raise HTTPException(status_code=404, detail="camera not found")
        return Response(content=make_jpeg(shade=160), media_type="image/jpeg")

    @app.get("/api/{camera_name}/recordings/{frame_time}/snapshot.{fmt}")
    async def recording_frame(
        camera_name: str, frame_time: float, fmt: str, height: int | None = None
    ) -> Response:
        if camera_name not in state.cameras:
            raise HTTPException(status_code=404, detail="camera not found")
        if not state.recordings_available:
            raise HTTPException(status_code=404, detail="no recordings for that time")
        shade = int(abs(frame_time) % 200)
        return Response(content=make_jpeg(shade=shade), media_type="image/jpeg")

    @app.post("/api/faces/{name}/register")
    async def register_face(name: str, request: Request) -> JSONResponse:
        if b"filename=" not in await request.body():
            raise HTTPException(status_code=422, detail="file required")
        if not state.face_in_view:
            return JSONResponse({"success": False, "message": "No face was detected."}, status_code=400)
        state.faces[name] = state.faces.get(name, 0) + 1
        return JSONResponse({"success": True, "message": "Successfully registered face."})

    return app, state
