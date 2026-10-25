"""Choosing and shrinking the frames the VLM actually sees.

Two decisions matter for cost and latency:

**Which frames.** Evenly spaced across the review segment, biased slightly
toward the start (the interesting thing usually happens as the object enters).
We pull them from NVR's recordings endpoint rather than buffering video
ourselves -- NVR already wrote those segments to disk.

**How big.** A 1080p JPEG is ~150-250 kB and costs far more to process than a
640px one, for no gain on a screening question ("is there a person"). We
downscale before the model ever sees it.
"""

from __future__ import annotations

import io
import logging

from watchwise.nvr.client import NVRClient, NVRError
from watchwise.types import AlertContext, Frame

logger = logging.getLogger(__name__)


def sample_timestamps(ctx: AlertContext, count: int, *, front_bias: bool = True) -> list[float]:
    """Evenly spaced timestamps inside the segment, front-biased unless asked
    to cover it all (a summary of the whole event)."""
    if count <= 0:
        return []
    end = ctx.end_time or (ctx.start_time + 5.0)
    span = max(0.0, end - ctx.start_time)
    if span <= 0.5 or count == 1:
        return [ctx.start_time + span / 2.0]

    # Fractions across the segment: skip the very first and last instants,
    # which are often the object half-entering the frame.
    step = 1.0 / (count + 1)
    fractions = [step * (i + 1) for i in range(count)]
    # Front bias: an exponent above 1 pulls fractions in (0,1) toward zero,
    # so samples cluster nearer the start of the segment.
    if front_bias:
        fractions = [f**1.35 for f in fractions]
    return [ctx.start_time + f * span for f in fractions]


def downscale_jpeg(data: bytes, *, max_edge: int, quality: int) -> bytes:
    """Shrink a JPEG's long edge. Returns the original on any failure."""
    try:
        from PIL import Image
    except ImportError:  # Pillow is optional; full-size frames still work
        return data

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            longest = max(img.size)
            if longest <= max_edge:
                return data
            scale = max_edge / float(longest)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            resized = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception as exc:
        logger.debug("downscale failed, using original: %s", exc)
        return data


class FrameSampler:
    def __init__(self, client: NVRClient, *, max_edge: int = 640, quality: int = 70):
        self.client = client
        self.max_edge = max_edge
        self.quality = quality

    async def for_alert(self, ctx: AlertContext, count: int, *, front_bias: bool = True) -> list[Frame]:
        """Best-effort frame set for a review segment.

        Falls back through recordings -> event snapshot -> live latest frame.
        Screening on two frames beats not screening at all, so a partial
        result is returned rather than raising.
        """
        frames: list[Frame] = []
        for ts in sample_timestamps(ctx, count, front_bias=front_bias):
            try:
                frame = await self.client.recording_frame(ctx.camera, ts, height=self.max_edge)
            except NVRError as exc:
                logger.debug("no recording frame at %.2f on %s: %s", ts, ctx.camera, exc)
                continue
            frame.jpeg = downscale_jpeg(
                frame.jpeg, max_edge=self.max_edge, quality=self.quality
            )
            frames.append(frame)

        if frames:
            return frames

        for event_id in ctx.event_ids[:1]:
            try:
                data = await self.client.event_snapshot(event_id, height=self.max_edge)
            except NVRError:
                continue
            frames.append(
                Frame(
                    camera=ctx.camera,
                    timestamp=ctx.start_time,
                    jpeg=downscale_jpeg(data, max_edge=self.max_edge, quality=self.quality),
                    source="snapshot",
                )
            )

        if frames:
            return frames

        try:
            frame = await self.client.latest_frame(ctx.camera, height=self.max_edge)
        except NVRError as exc:
            logger.warning("no frames available for %s: %s", ctx.review_id, exc)
            return []
        frame.jpeg = downscale_jpeg(frame.jpeg, max_edge=self.max_edge, quality=self.quality)
        return [frame]
