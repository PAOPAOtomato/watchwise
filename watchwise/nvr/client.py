"""Async HTTP client for the NVR (Frigate) API.

This is the *only* place that knows NVR's URL shapes. Everything the agent
does forensically -- pulling a frame from five minutes ago, listing what the
neighbouring camera saw, running a semantic search over past events -- goes
through here, so swapping NVR versions is a one-file change.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from watchwise.config import NVRSettings
from watchwise.types import Frame

logger = logging.getLogger(__name__)


class NVRError(RuntimeError):
    pass


class NVRClient:
    def __init__(self, settings: NVRSettings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            timeout=settings.timeout_s,
            headers=settings.headers or None,
            verify=settings.verify_tls,
        )

    @property
    def api(self) -> str:
        return self.settings.api_prefix.rstrip("/")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "NVRClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ---------------- low level ----------------

    async def _get_json(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = await self._client.get(f"{self.api}{path}", params=params)
        except httpx.HTTPError as exc:
            raise NVRError(f"GET {path} failed: {exc}") from exc
        if r.status_code >= 400:
            raise NVRError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    async def _get_bytes(self, path: str, **params: Any) -> bytes:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = await self._client.get(f"{self.api}{path}", params=params)
        except httpx.HTTPError as exc:
            raise NVRError(f"GET {path} failed: {exc}") from exc
        if r.status_code >= 400:
            raise NVRError(f"GET {path} -> {r.status_code}")
        return r.content

    # ---------------- system ----------------

    async def version(self) -> str:
        r = await self._client.get(f"{self.api}/version")
        return r.text.strip()

    async def config(self) -> dict[str, Any]:
        return await self._get_json("/config")

    async def stats(self) -> dict[str, Any]:
        return await self._get_json("/stats")

    async def cameras(self) -> list[str]:
        """The NVR's cameras, without those switched off in its config."""
        cfg = await self.config()
        cameras = cfg.get("cameras") or {}
        return sorted(name for name, cam in cameras.items() if (cam or {}).get("enabled", True) is not False)

    # ---------------- reviews ----------------

    async def get_review(self, review_id: str) -> dict[str, Any]:
        return await self._get_json(f"/review/{review_id}")

    async def list_reviews(
        self,
        *,
        cameras: list[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return await self._get_json(
            "/review",
            cameras=",".join(cameras) if cameras else None,
            after=after,
            before=before,
            severity=severity,
            limit=limit,
        )

    # ---------------- events ----------------

    async def get_event(self, event_id: str) -> dict[str, Any]:
        return await self._get_json(f"/events/{event_id}")

    async def list_events(
        self,
        *,
        cameras: list[str] | None = None,
        labels: list[str] | None = None,
        zones: list[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
        include_thumbnails: int = 0,
    ) -> list[dict[str, Any]]:
        return await self._get_json(
            "/events",
            cameras=",".join(cameras) if cameras else None,
            labels=",".join(labels) if labels else None,
            zones=",".join(zones) if zones else None,
            after=after,
            before=before,
            limit=limit,
            include_thumbnails=include_thumbnails,
        )

    async def search_events(
        self,
        query: str,
        *,
        cameras: list[str] | None = None,
        limit: int = 20,
        search_type: str = "description,thumbnail",
    ) -> list[dict[str, Any]]:
        """NVR's CLIP/embedding semantic search over past events.

        We reuse this rather than building a second video index -- it is
        already maintained, already backfilled, and already on the same DB.
        """
        return await self._get_json(
            "/events/search",
            query=query,
            cameras=",".join(cameras) if cameras else None,
            limit=limit,
            search_type=search_type,
        )

    # ---------------- media ----------------

    async def recording_frame(
        self, camera: str, timestamp: float, *, fmt: str = "jpg", height: int | None = None
    ) -> Frame:
        """Pull an arbitrary historical frame straight out of the recordings.

        This is the primitive behind "retrieve additional historical frames":
        the agent can ask for t-30s or t+10s without us keeping any extra
        video buffer of our own.
        """
        data = await self._get_bytes(
            f"/{camera}/recordings/{timestamp}/snapshot.{fmt}", height=height
        )
        return Frame(camera=camera, timestamp=timestamp, jpeg=data, source="recording")

    async def event_snapshot(self, event_id: str, *, height: int | None = None) -> bytes:
        return await self._get_bytes(f"/events/{event_id}/snapshot.jpg", height=height)

    async def event_thumbnail(self, event_id: str, *, extension: str = "jpg") -> bytes:
        return await self._get_bytes(f"/events/{event_id}/thumbnail.{extension}")

    async def latest_frame(self, camera: str, *, height: int | None = None) -> Frame:
        import time as _time

        data = await self._get_bytes(f"/{camera}/latest.jpg", height=height)
        return Frame(camera=camera, timestamp=_time.time(), jpeg=data, source="snapshot")

    # ---------------- faces ----------------

    async def faces(self) -> dict[str, int]:
        """The NVR's Face Library: name -> number of photos. Frigate's
        ``train`` folder (its recent recognition attempts) is not a person."""
        library = await self._get_json("/faces")
        return {
            str(name): len(files or [])
            for name, files in (library or {}).items()
            if name != "train" and isinstance(files, list)
        }

    async def face_attempts(self) -> list[str]:
        """File names of the NVR's recent face-recognition attempts (Frigate's
        ``train`` folder): ``<event id>-<time>-<name or unknown>-<score>.webp``."""
        library = await self._get_json("/faces")
        return [str(f) for f in (library or {}).get("train") or []]

    async def face_attempt(self, file: str) -> bytes:
        """One attempt's face crop (served outside /api, like other clips)."""
        path = f"/clips/faces/train/{quote(file, safe='')}"
        try:
            r = await self._client.get(path)
        except httpx.HTTPError as exc:
            raise NVRError(f"GET {path} failed: {exc}") from exc
        if r.status_code >= 400:
            raise NVRError(f"GET {path} -> {r.status_code}")
        return r.content

    async def train_face(self, name: str, training_file: str) -> dict[str, Any]:
        """Move a recognition attempt's crop into ``name``'s Face Library
        folder (D-071); the NVR then rebuilds its recognizer."""
        path = f"/faces/train/{quote(name, safe='')}/classify"
        try:
            r = await self._client.post(f"{self.api}{path}", json={"training_file": training_file})
        except httpx.HTTPError as exc:
            raise NVRError(f"POST {path} failed: {exc}") from exc
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 500 or not isinstance(body, dict):
            raise NVRError(f"POST {path} -> {r.status_code}: {r.text[:200]}")
        ok = r.status_code < 400 and body.get("success", True) is not False
        return {"success": ok, "message": str(body.get("message") or "")}

    async def register_face(self, name: str, image: bytes) -> dict[str, Any]:
        """Add a face to the NVR's Face Library from a whole image; the NVR
        finds the face in it. Returns its answer: {"success", "message"}."""
        path = f"/faces/{name}/register"
        try:
            r = await self._client.post(f"{self.api}{path}", files={"file": ("face.jpg", image, "image/jpeg")})
        except httpx.HTTPError as exc:
            raise NVRError(f"POST {path} failed: {exc}") from exc
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 500 or not isinstance(body, dict):
            raise NVRError(f"POST {path} -> {r.status_code}: {r.text[:200]}")
        ok = r.status_code < 400 and body.get("success", True) is not False
        return {"success": ok, "message": str(body.get("message") or "")}

    def clip_url(self, camera: str, start: float, end: float) -> str:
        return f"{self.settings.base_url.rstrip('/')}{self.api}/{camera}/start/{start}/end/{end}/clip.mp4"

    def review_clip_url(self, review_id: str) -> str:
        return f"{self.settings.base_url.rstrip('/')}{self.api}/review/{review_id}/clip.mp4"
