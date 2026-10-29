"""Add faces to Frigate's Face Library (D-058, D-071).

Frigate keeps one library for all cameras: each person is a folder of face
crops, averaged into one embedding that every camera's faces are compared
with. So a photo from any camera enrolls a person everywhere; more clear
photos with some variety (angle, light) make recognition steadier.

Two ways in, both asking before anything changes:
- `/face <name> [camera]`: the camera's current frame; Frigate finds the face.
- "👤 That was <name>" on a card for a face nobody matched: the face crop
  Frigate itself tried on that event, so the camera that missed teaches it.

Like the button actions, this never sends anything itself, so any channel
can use it.
"""

from __future__ import annotations

import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from watchwise.config import FaceSettings
from watchwise.nvr.client import NVRClient, NVRError
from watchwise.setup.home import NAME

HOLD_S = 300.0  # how long a capture waits for its confirmation
MAX_PENDING = 16
ENOUGH = 5  # Frigate's guide: 5-10 clear photos at least, 20-30 for robust recognition
ATTEMPT = re.compile(r"^(?P<event>\d+\.\d+-[a-z0-9]+)-(?P<at>\d+(?:\.\d+)?)-(?P<label>.+)-(?P<score>[0-9.]+)\.webp$")


@dataclass
class Capture:
    name: str
    camera: str
    jpeg: bytes
    taken_at: float
    training_file: str = ""  # an NVR recognition attempt, instead of a whole frame


class FaceEnroller:
    def __init__(self, nvr: NVRClient, faces: FaceSettings, *, clock: Callable[[], float] = time.time):
        self.nvr = nvr
        self.faces = faces
        self.clock = clock
        self._pending: OrderedDict[str, Capture] = OrderedDict()

    async def capture(self, name: str, camera: str | None = None) -> tuple[str | None, Capture | None, str]:
        """Take the frame to confirm: (token, capture, message), or (None, None, why not)."""
        if not NAME.match(name):
            return None, None, "Use a simple name: letters, digits or _ (e.g. alex), up to 32 characters."
        try:
            cameras = await self.nvr.cameras()
        except NVRError as exc:
            return None, None, f"Couldn't reach Frigate ({exc})."
        if camera is None and len(cameras) == 1:
            camera = cameras[0]
        if camera not in cameras:
            return None, None, f"Which camera? Send /face {name} <camera>, one of: {', '.join(cameras) or 'none'}."
        try:
            frame = await self.nvr.latest_frame(camera)
        except NVRError as exc:
            return None, None, f"Couldn't get a frame from {camera} ({exc})."
        capture = Capture(name=name, camera=camera, jpeg=frame.jpeg, taken_at=self.clock())
        token = self._hold(capture)
        return token, capture, (
            f"Add the face in this frame from {camera} to Frigate's Face Library as {name}? "
            "Face the camera, in good light, alone in view."
        )

    async def from_event(self, name: str, review_ids: list[str]) -> tuple[str | None, Capture | None, str]:
        """The face crop Frigate tried on these events (a card's), to add to
        ``name``: (token, capture, message), or (None, None, why not)."""
        if not NAME.match(name):
            return None, None, "That name isn't a simple one; use /face <name> <camera> instead."
        try:
            events: dict[str, str] = {}
            for review_id in review_ids:
                review = await self.nvr.get_review(review_id)
                for event_id in (review.get("data") or {}).get("detections") or []:
                    events[str(event_id)] = str(review.get("camera") or "")
            attempts = await self.nvr.face_attempts()
        except NVRError as exc:
            return None, None, f"Couldn't reach Frigate ({exc})."
        found = [(m, f) for f in attempts if (m := ATTEMPT.match(f)) and m["event"] in events]
        if not found:
            return None, None, (
                "Frigate kept no face crop from this event (it keeps only its latest attempts). "
                f"Use /face {name} <camera> while facing that camera."
            )
        # The largest crop is usually the closest, clearest face.
        best, best_file, best_bytes = None, "", b""
        for match, file in sorted(found, key=lambda x: -float(x[0]["at"]))[:6]:
            try:
                crop = await self.nvr.face_attempt(file)
            except NVRError:
                continue
            if len(crop) > len(best_bytes):
                best, best_file, best_bytes = match, file, crop
        if best is None:
            return None, None, "Couldn't fetch the face crops from Frigate; try again, or use /face."
        camera = events[best["event"]]
        capture = Capture(name=name, camera=camera, jpeg=_as_jpeg(best_bytes), taken_at=self.clock(),
                          training_file=best_file)
        token = self._hold(capture)
        seen = "matched nobody" if best["label"] == "unknown" else f"was closest to {best['label']}"
        return token, capture, (
            f"Add this face from {camera} to {name} in Frigate's Face Library? It {seen} "
            f"(score {float(best['score']):.2f}). Only if it is clearly {name}'s face: a blurred or "
            "turned one teaches the wrong thing."
        )

    async def status(self) -> str:
        """What the Face Library holds, for `/face` on its own."""
        try:
            library = await self.nvr.faces()
        except NVRError as exc:
            return f"Couldn't reach Frigate ({exc})."
        residents = list(self.faces.household)
        lines = ["Frigate's Face Library (one library for all cameras: a photo from any camera counts for every camera):"]
        for name in residents:
            count = next((n for person, n in library.items() if person.lower() == name.lower()), 0)
            hint = "" if count >= ENOUGH else f"; add {ENOUGH - count}+ more clear photos for steadier recognition"
            lines.append(f"• {name}: {count} photo(s){hint}")
        others = [f"{person} ({n})" for person, n in sorted(library.items()) if person.lower() not in {r.lower() for r in residents}]
        if others:
            lines.append("Also in the library (not residents): " + ", ".join(others))
        if not residents:
            lines.append("No residents yet: name them in /setup.")
        lines.append("Add photos with /face <name> [camera], or with 👤 on a card for a face nobody matched.")
        return "\n".join(lines)

    def _hold(self, capture: Capture) -> str:
        token = secrets.token_hex(4)
        self._pending[token] = capture
        while len(self._pending) > MAX_PENDING:
            self._pending.popitem(last=False)
        return token

    async def confirm(self, token: str) -> str:
        capture = self._pending.pop(token, None)
        if capture is None or self.clock() - capture.taken_at > HOLD_S:
            return "That capture has expired or was already used; send /face again."
        try:
            if capture.training_file:
                result = await self.nvr.train_face(capture.name, capture.training_file)
            else:
                result = await self.nvr.register_face(capture.name, capture.jpeg)
        except NVRError as exc:
            return f"Frigate didn't take it ({exc})."
        if not result["success"]:
            return (
                f"Frigate couldn't use that frame: {result['message'] or 'no reason given'} "
                "Face the camera in good light and send /face again."
            )
        message = f"✅ Added a face for {capture.name} to Frigate's Face Library; every camera uses it."
        if capture.name.lower() not in {n.lower() for n in self.faces.household}:
            message += (
                f" To have Watchwise hold {capture.name} back while you're home, add them as a "
                "resident with /setup."
            )
        return message

    def cancel(self, token: str) -> str:
        self._pending.pop(token, None)
        return "Cancelled; nothing was added."


def _as_jpeg(image: bytes) -> bytes:
    """Frigate stores attempts as WebP; a JPEG preview is safest for every
    channel. Without Pillow the WebP goes as it is."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image)) as im:
            out = io.BytesIO()
            im.convert("RGB").save(out, "JPEG", quality=90)
            return out.getvalue()
    except Exception:
        return image
