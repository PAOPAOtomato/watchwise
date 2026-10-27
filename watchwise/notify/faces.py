"""Add a face to Frigate's Face Library from what a camera sees now (D-058).

`/face <name> [camera]` takes the camera's current frame and asks before
anything changes; on confirmation the frame goes to Frigate, which finds the
face in it and stores the crop. Like the button actions, this never sends
anything itself, so any channel can use it.
"""

from __future__ import annotations

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


@dataclass
class Capture:
    name: str
    camera: str
    jpeg: bytes
    taken_at: float


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
        token = secrets.token_hex(4)
        self._pending[token] = capture
        while len(self._pending) > MAX_PENDING:
            self._pending.popitem(last=False)
        return token, capture, (
            f"Add the face in this frame from {camera} to Frigate's Face Library as {name}? "
            "Face the camera, in good light, alone in view."
        )

    async def confirm(self, token: str) -> str:
        capture = self._pending.pop(token, None)
        if capture is None or self.clock() - capture.taken_at > HOLD_S:
            return "That capture has expired or was already used; send /face again."
        try:
            result = await self.nvr.register_face(capture.name, capture.jpeg)
        except NVRError as exc:
            return f"Frigate didn't take it ({exc})."
        if not result["success"]:
            return (
                f"Frigate couldn't use that frame: {result['message'] or 'no reason given'} "
                "Face the camera in good light and send /face again."
            )
        message = f"✅ Added a face for {capture.name} to Frigate's Face Library."
        if capture.name.lower() not in {n.lower() for n in self.faces.household}:
            message += (
                f" To have Watchwise hold {capture.name} back while you're home, add them as a "
                "resident with /setup."
            )
        return message

    def cancel(self, token: str) -> str:
        self._pending.pop(token, None)
        return "Cancelled; nothing was added."
