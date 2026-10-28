"""/setup: one conversation that tells Watchwise about this home (D-060).

Four questions: what each camera looks at (and, with several cameras, which
room it is in), who lives here, what should always alert, and what to assume
while no arming state is set. Then the
plan: every answer in plain words, and which blocks of the config file
change. Nothing is written until the owner taps Apply; the file is backed
up first and Undo puts it back (``setup/home.py`` does the writing, and the
running service uses the answers from the next event on). Afterwards it
offers to add photos to Frigate's Face Library (the ``/face`` flow) and a
recognition check: "walk past; I'll say if I recognized you".

Like the button actions, it never sends anything itself: each step returns
the messages for the channel to send. Its buttons carry ``su:<step>:...``,
so a press on an earlier question is recognized as such.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import html
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watchwise.config import WatchwiseConfig
from watchwise.nvr.client import NVRClient
from watchwise.pipeline import FaceFeed, FaceSeen
from watchwise.setup.home import (
    ARMING_LABELS,
    FileChanged,
    HomeAnswers,
    HomePlan,
    SetupError,
    apply_home,
    clean_note,
    clean_rules,
    plan_home,
    read_config,
    restore,
    room_from_reply,
    split_names,
    update_live,
)
from watchwise.tier2.sampling import downscale_jpeg

logger = logging.getLogger(__name__)

SESSION_S = 1800.0  # an untouched setup is forgotten after this long
CHECK_S = 120.0  # how long the recognition check waits for a face
QUIET_S = 20.0  # ...and how long after the last face result
GOOD_PHOTOS = 3  # photos of a person in the Face Library that recognize well
DIFF_LIMIT = 3500
CLOUD_PROVIDERS = ("openai_compat", "anthropic")
# A diff line whose value may be a secret (a hand-written mqtt.password, say):
# the diff goes through Telegram, so the value is masked.
SECRET_LINE = re.compile(r"^([-+ ]\s*[\w-]*(?:password|passwd|secret|token|api_key)[\w-]*\s*:\s*)\S.*$", re.I)
# Callback step names -> the step they answer.
STEP_KEYS = {"cam": "camera", "room": "room", "who": "who", "alw": "always", "arm": "arming", "plan": "plan"}

Button = tuple[str, str]  # (label, callback data)


def esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


@dataclass
class Reply:
    """One message for the channel to send (Telegram HTML)."""

    text: str
    buttons: list[list[Button]] = field(default_factory=list)
    photo: bytes | None = None
    edit: bool = False  # replace the message whose button was pressed
    check: bool = False  # then run check() and send its result
    face: str | None = None  # then start the /face flow for this name...
    face_camera: str | None = None  # ...with this camera's frame


@dataclass
class _Session:
    cameras: list[str]
    current: HomeAnswers  # what the file says
    answers: HomeAnswers  # what the owner has said so far
    names: list[str]  # offered as buttons on the "who lives here" step
    touched: float
    step: str = "camera"
    camera: int = 0
    plan: HomePlan | None = None
    room_choices: list[str] = field(default_factory=list)  # the room buttons last offered


class _Watch:
    """The face results that arrive during one recognition check."""

    def __init__(self) -> None:
        self.attempts: list[FaceSeen] = []
        self.confirmed: FaceSeen | None = None
        self._last = 0.0
        self._news = asyncio.Event()

    def observe(self, seen: FaceSeen) -> None:
        if seen.confirmed:
            self.confirmed = self.confirmed or seen
        else:
            self.attempts.append(seen)
        self._last = time.monotonic()
        self._news.set()

    async def wait(self, window_s: float, quiet_s: float) -> None:
        """Until a resident is recognized, the window ends, or no new face
        result has come for ``quiet_s`` (the person has walked past)."""
        deadline = time.monotonic() + window_s
        while self.confirmed is None:
            now = time.monotonic()
            left = deadline - now
            if self.attempts:
                left = min(left, self._last + quiet_s - now)
            if left <= 0:
                return
            self._news.clear()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(left):
                    await self._news.wait()


class SetupChat:
    def __init__(
        self,
        config: WatchwiseConfig,
        path: Path | None,
        nvr: NVRClient,
        *,
        arming: Any = None,
        face_feed: FaceFeed | None = None,
        send_images: bool = True,
        clock: Callable[[], float] = time.time,
        check_s: float = CHECK_S,
        quiet_s: float = QUIET_S,
    ):
        # The running service's config: Apply updates it in place.
        self.config = config
        self.path = path
        self.nvr = nvr
        self.arming = arming
        self.face_feed = face_feed
        self.send_images = send_images
        self.clock = clock
        self.check_s = check_s
        self.quiet_s = quiet_s
        self._sessions: dict[int, _Session] = {}
        # chat -> (backup, the file as /setup wrote it), for Undo.
        self._undo: dict[int, tuple[Path, str]] = {}
        self._checking: set[int] = set()

    # ---------------- entry points ----------------

    async def start(self, chat_id: int) -> list[Reply]:
        self._sessions.pop(chat_id, None)
        try:
            _, config = read_config(self.path)
        except SetupError as exc:
            return [Reply(esc(exc))]
        current = HomeAnswers.from_config(config)
        intro = (
            "<b>Let's set up Watchwise for your home.</b> Four short questions; nothing "
            "changes until you confirm at the end. /cancel stops at any time."
        )
        try:
            cameras = await self.nvr.cameras()
        except Exception as exc:
            cameras = []
            intro += f"\nI couldn't reach Frigate ({esc(exc)}), so I'll skip the camera questions."
        library = await self._library()
        names: list[str] = []
        for name in [*current.household, *library]:
            if name.lower() not in {n.lower() for n in names}:
                names.append(name)
        session = _Session(cameras, current, copy.deepcopy(current), names, touched=self.clock())
        self._sessions[chat_id] = session
        return [Reply(intro), *(await self._ask(session))]

    def waiting(self, chat_id: int) -> bool:
        """Is a setup in progress in this chat? Then what the owner types
        goes to it (an answer, or a reminder to use the buttons)."""
        session = self._sessions.get(chat_id)
        return session is not None and not self._expired(session)

    async def answer(self, chat_id: int, text: str) -> list[Reply]:
        session = self._session(chat_id)
        if session is None:
            return [_ended()]
        if session.step == "camera":
            name = session.cameras[session.camera]
            note = clean_note(text)
            if not note:
                return [Reply("That was empty; reply in a few words, or tap Skip.")]
            session.answers.notes[name] = note
            return [Reply(f"📷 Noted for {esc(name)}: “{esc(note)}”"), *(await self._after_note(session))]
        if session.step == "room":
            name = session.cameras[session.camera]
            room = room_from_reply(text)
            if not room:
                return [Reply("Type the room's name, or tap a button.")]
            session.answers.rooms[name] = room
            session.camera += 1
            session.step = "camera"
            return [Reply(f"🏠 {esc(name)} is in {esc(room)}."), *(await self._ask(session))]
        if session.step == "who":
            usable, rejected = split_names(text)
            for name in usable:
                known = next((n for n in session.names if n.lower() == name.lower()), None)
                if known is None:
                    session.names.append(name)
                    known = name
                if known.lower() not in {h.lower() for h in session.answers.household}:
                    session.answers.household.append(known)
            session.answers.household = self._ordered(session)
            lead = ""
            if rejected:
                lead = (
                    f"I can't use {esc(', '.join(rejected))}: a name is letters, digits or _, "
                    "as in Frigate's Face Library.\n\n"
                )
            return [self._who_question(session, lead)]
        if session.step == "always":
            rules = clean_rules(text.splitlines())
            if not rules:
                return [Reply("I didn't catch anything there; one item per line, please.")]
            session.answers.always_alert = rules
            session.step = "arming"
            return await self._ask(session)
        return [Reply("Please use the buttons above, or /cancel.")]

    async def press(self, chat_id: int, data: str) -> list[Reply]:
        """A button: ``su:<step>[:<arg>[:<index>]]``."""
        parts = data.split(":")
        step = parts[1] if len(parts) > 1 else ""
        arg = parts[2] if len(parts) > 2 else ""
        if step == "face" and arg:
            return await self._face_photo(arg, parts[3] if len(parts) > 3 else None)
        if step == "check":
            return self.check_reply(chat_id)
        if step == "undo":
            return self._undo_reply(chat_id)
        if step == "done":
            self._sessions.pop(chat_id, None)
            return [Reply(
                "👍 All set. /setup again any time; /arm switches arming; /help explains the "
                "alert buttons."
            )]
        session = self._session(chat_id)
        if session is None:
            return [_ended()]
        if STEP_KEYS.get(step) != session.step:
            return [Reply("That question has been answered already; use the latest message.")]
        if step == "cam":
            if arg not in ("keep", "clear") or parts[3:4] != [str(session.camera)]:
                return [Reply("That question has been answered already; use the latest message.")]
            if arg == "clear":
                session.answers.notes.pop(session.cameras[session.camera], None)
            return await self._after_note(session)
        if step == "room":
            choice = parts[3] if len(parts) > 3 else ""
            if arg != str(session.camera):
                return [Reply("That question has been answered already; use the latest message.")]
            name = session.cameras[session.camera]
            if choice == "own":
                session.answers.rooms.pop(name, None)
            elif choice.isdigit() and int(choice) < len(session.room_choices):
                session.answers.rooms[name] = session.room_choices[int(choice)]
            else:
                return [Reply("Unknown button.")]
            session.camera += 1
            session.step = "camera"
            return await self._ask(session)
        if step == "who":
            if arg == "done":
                session.step = "always"
                return await self._ask(session)
            if not arg.isdigit() or int(arg) >= len(session.names):
                return [Reply("Unknown button.")]
            name = session.names[int(arg)]
            chosen = {h.lower() for h in session.answers.household}
            if name.lower() in chosen:
                session.answers.household = [h for h in session.answers.household if h.lower() != name.lower()]
            else:
                session.answers.household.append(name)
            session.answers.household = self._ordered(session)
            reply = self._who_question(session)
            reply.edit = True
            return [reply]
        if step == "alw":
            if arg == "clear":
                session.answers.always_alert = []
            session.step = "arming"
            return await self._ask(session)
        if step == "arm":
            if arg not in ARMING_LABELS:
                return [Reply("Unknown button.")]
            session.answers.arming_default = arg
            session.step = "plan"
            return await self._ask(session)
        if step == "plan":
            if arg == "apply":
                return await self._apply(chat_id, session)
            if arg == "diff":
                return [self._diff(session)]
            if arg == "restart":
                return await self.start(chat_id)
            if arg == "cancel":
                return self.cancel(chat_id)
        return [Reply("Unknown button.")]

    def cancel(self, chat_id: int) -> list[Reply]:
        if self._sessions.pop(chat_id, None) is None:
            return [Reply("There's no setup in progress.")]
        return [Reply("Cancelled; nothing was changed.")]

    # ---------------- the questions ----------------

    async def _ask(self, session: _Session) -> list[Reply]:
        if session.step == "camera":
            if session.camera < len(session.cameras):
                return [await self._camera_question(session)]
            session.step = "who"
        if session.step == "who":
            return [self._who_question(session)]
        if session.step == "always":
            return [self._always_question(session)]
        if session.step == "arming":
            return [self._arming_question(session)]
        return self._plan(session)

    async def _camera_question(self, session: _Session) -> Reply:
        index = session.camera
        name = session.cameras[index]
        of = f" ({index + 1} of {len(session.cameras)})" if len(session.cameras) > 1 else ""
        text = (
            f"<b>1/4 · Camera {esc(name)}</b>{of}: what does it look at, and what's normal "
            "there? Reply in a few words, e.g. “my bedroom workspace; I sit at the desk most "
            "of the day”."
        )
        note = session.answers.notes.get(name, "")
        if note:
            text += f"\nNow: “{esc(note)}”"
            buttons = [[("Keep it", f"su:cam:keep:{index}"), ("Clear it", f"su:cam:clear:{index}")]]
        else:
            buttons = [[("Skip", f"su:cam:keep:{index}")]]
        photo = await self._frame(name) if self.send_images else None
        return Reply(text, buttons, photo=photo)

    async def _after_note(self, session: _Session) -> list[Reply]:
        """With several cameras, ask which room this one is in (D-062)."""
        if len(session.cameras) > 1:
            session.step = "room"
            return [self._room_question(session)]
        session.camera += 1
        return await self._ask(session)

    def _room_question(self, session: _Session) -> Reply:
        index = session.camera
        name = session.cameras[index]
        known: list[str] = []
        for room in (room_from_reply(r) for r in session.answers.rooms.values()):
            if room and room.lower() not in {k.lower() for k in known}:
                known.append(room)
        session.room_choices = known
        text = (
            f"Which room is <b>{esc(name)}</b> in? Cameras in one room share one card per "
            "incident, and a tap on it teaches all of them. Tap a room, or type just its "
            "name, e.g. “bedroom”."
        )
        current = session.answers.rooms.get(name, "")
        if current:
            text += f"\nNow: {esc(current)}"
        tiles = [(f"🏠 {room}", f"su:room:{index}:{j}") for j, room in enumerate(known)]
        buttons = [tiles[i : i + 2] for i in range(0, len(tiles), 2)]
        buttons.append([("Its own room", f"su:room:{index}:own")])
        return Reply(text, buttons)

    def _who_question(self, session: _Session, lead: str = "") -> Reply:
        chosen = {h.lower() for h in session.answers.household}
        text = (
            f"{lead}<b>2/4 · Who lives here?</b> Tap each resident, or type names separated "
            "by commas, then tap Done. While you're home, an event in which Frigate "
            "recognizes a resident is held back."
        )
        if session.names:
            text += " The buttons are the names in Frigate's Face Library and your config."
        tiles = [
            (f"{'✅' if n.lower() in chosen else '▫️'} {n}", f"su:who:{i}") for i, n in enumerate(session.names)
        ]
        buttons = [tiles[i : i + 3] for i in range(0, len(tiles), 3)]
        buttons.append([("Done", "su:who:done")])
        return Reply(text, buttons)

    def _always_question(self, session: _Session) -> Reply:
        rules = session.answers.always_alert
        text = (
            "<b>3/4 · What should always alert you</b>, even when you're home? One per line, "
            "e.g. “anyone at the window” or “someone carrying a ladder”. The vision model "
            "checks every event for these."
        )
        if rules:
            text += "\nNow:\n" + "\n".join(f"• {esc(r)}" for r in rules) + "\n(A reply replaces these.)"
            buttons = [[("Keep these", "su:alw:keep"), ("None", "su:alw:clear")]]
        else:
            buttons = [[("Nothing extra", "su:alw:keep")]]
        return Reply(text, buttons)

    def _arming_question(self, session: _Session) -> Reply:
        default = session.answers.arming_default
        text = (
            "<b>4/4 · When no arming state is set</b>, what should Watchwise assume? "
            f"Now: {esc(ARMING_LABELS.get(default, default))}. (Away or Night: any person in "
            "view alerts unless the duty roster or a recognized resident explains them.)"
        )
        now = self._arming_now()
        if now is not None and now.get("source") != "default":
            label = ARMING_LABELS.get(str(now.get("state")), str(now.get("state")))
            text += (
                f"\nArming is set to {esc(label)} right now (/arm switches it), so this "
                "applies only while no state is set."
            )
        buttons = [
            [("🏠 Home", "su:arm:armed_home"), ("🚪 Away", "su:arm:armed_away")],
            [("🌙 Night", "su:arm:armed_night"), ("🔓 Off", "su:arm:disarmed")],
        ]
        return Reply(text, buttons)

    # ---------------- plan, apply, undo ----------------

    def _plan(self, session: _Session, lead: str = "") -> list[Reply]:
        session.step = "plan"
        try:
            session.plan = plan_home(self.path, session.answers)
        except SetupError as exc:
            return [Reply(f"{lead}{esc(exc)}", [[("↺ Start over", "su:plan:restart")]])]
        plan = session.plan
        lines = "\n".join(esc(line) for line in plan.lines)
        if not plan.changes_anything:
            return [Reply(
                f"{lead}<b>Nothing to change</b>: {esc(plan.path.name)} already says this.\n{lines}",
                [[("✖️ Close", "su:plan:cancel"), ("↺ Start over", "su:plan:restart")]],
            )]
        text = (
            f"{lead}<b>The plan</b>\n{lines}\n\n"
            f"✅ Apply changes {esc(', '.join(plan.blocks))} in <code>{esc(plan.path.name)}</code>. "
            "Nothing else in the file changes, and the current version is backed up first. "
            f"{self._privacy()}"
        )
        return [Reply(text, [
            [("✅ Apply", "su:plan:apply"), ("📄 Show the diff", "su:plan:diff")],
            [("↺ Start over", "su:plan:restart"), ("✖️ Cancel", "su:plan:cancel")],
        ])]

    def _privacy(self) -> str:
        ep = self.config.endpoint("screener")
        model = f"{ep.provider} {ep.model}" if ep is not None else f"{self.config.vlm.backend} {self.config.vlm.model}"
        line = f"Camera notes and always-alert rules go into the model prompts ({model})"
        clouds = sorted(
            {e.provider for role in ("screener", "investigator") if (e := self.config.endpoint(role)) and e.provider in CLOUD_PROVIDERS}
        )
        if clouds:
            line += f"; with {', '.join(clouds)} they leave this machine"
        return esc(line) + "."

    def _diff(self, session: _Session) -> Reply:
        diff = session.plan.change.diff if session.plan is not None else ""
        if not diff:
            return Reply("There's no diff to show.")
        diff = "\n".join(SECRET_LINE.sub(r"\1•••", line) for line in diff.splitlines())
        clipped = diff[:DIFF_LIMIT] + ("\n…" if len(diff) > DIFF_LIMIT else "")
        return Reply(f"<pre>{esc(clipped)}</pre>")

    async def _apply(self, chat_id: int, session: _Session) -> list[Reply]:
        plan = session.plan
        if plan is None:
            return self._plan(session)
        try:
            new = apply_home(plan, now=self.clock())
        except FileChanged:
            return self._plan(
                session,
                lead=(
                    "The config file changed after I showed you the plan, so I wrote nothing. "
                    "Here it is again, against the file as it is now.\n\n"
                ),
            )
        except (SetupError, OSError, ValueError) as exc:
            return [Reply(f"Nothing was written: {esc(exc)}")]
        update_live(self.config, new)
        self._sessions.pop(chat_id, None)
        change = plan.change
        logger.info("/setup saved %s (%s)", plan.path, ", ".join(plan.blocks))
        text = f"✅ Saved in <code>{esc(plan.path.name)}</code>"
        buttons: list[list[Button]] = []
        if change.backup is not None and change.content is not None:
            self._undo[chat_id] = (change.backup, change.content)
            text += f"; the previous version is <code>{esc(change.backup.name)}</code>"
            buttons = [[("↩️ Undo", "su:undo")]]
        text += ". Watchwise uses it from the next event on, with no restart."
        return [Reply(text, buttons), await self._faces_reply()]

    def _undo_reply(self, chat_id: int) -> list[Reply]:
        saved = self._undo.pop(chat_id, None)
        if saved is None or self.path is None:
            return [Reply("Nothing to undo.")]
        backup, written = saved
        try:
            change, config = restore(self.path, backup, written, now=self.clock())
        except SetupError as exc:
            return [Reply(esc(exc))]
        update_live(self.config, config)
        logger.info("/setup undone: %s restored from %s", self.path, backup)
        kept = f" The /setup version is kept as <code>{esc(change.backup.name)}</code>." if change.backup else ""
        return [Reply(
            f"↩️ Undone: <code>{esc(self.path.name)}</code> is back to how it was before /setup, "
            f"and Watchwise uses that again.{kept}"
        )]

    # ---------------- faces and the recognition check ----------------

    async def _faces_reply(self) -> Reply:
        household = list(self.config.faces.household)
        if not household:
            return Reply(
                "No residents are set, so faces change nothing for now. /setup again any time.",
                [[("Done", "su:done")]],
            )
        counts = {name.lower(): count for name, count in (await self._library()).items()}
        lines: list[str] = []
        buttons: list[list[Button]] = []
        for name in household:
            count = counts.get(name.lower(), 0)
            lines.append(f"• {esc(name)}: {count} photo{'' if count == 1 else 's'}" + (" ✅" if count >= GOOD_PHOTOS else ""))
            if count < GOOD_PHOTOS:
                buttons.append([(f"📸 Add a photo of {name}", f"su:face:{name}")])
        buttons.append([("👋 Check recognition", "su:check"), ("Done", "su:done")])
        return Reply(
            "<b>Faces.</b> Frigate recognizes residents from the photos in its Face Library; "
            f"{GOOD_PHOTOS} or more, facing the camera in good light, work best.\n"
            + "\n".join(lines)
            + "\nThen tap 👋 and walk past a camera: I'll tell you whether Frigate recognized you.",
            buttons,
        )

    async def _face_photo(self, name: str, camera: str | None) -> list[Reply]:
        """📸: with several cameras, ask which one takes the photo. A
        resident is recognized best by a camera that has a photo from its
        own angle (live: laptopcam knew Ray only after one)."""
        try:
            cameras = await self.nvr.cameras()
        except Exception:
            cameras = []
        if camera is not None and camera.isdigit() and int(camera) < len(cameras):
            return [Reply("", face=name, face_camera=cameras[int(camera)])]
        if len(cameras) <= 1:
            return [Reply("", face=name)]
        tiles = [(f"📷 {c}", f"su:face:{name}:{i}") for i, c in enumerate(cameras)]
        return [Reply(
            f"📸 Which camera should take the photo of {esc(name)}? Face that camera, alone "
            "and in good light, then tap it. Each camera recognizes you best with a photo "
            "from its own angle.",
            [tiles[i : i + 2] for i in range(0, len(tiles), 2)],
        )]

    def check_reply(self, chat_id: int) -> list[Reply]:
        if self.face_feed is None:
            return [Reply("The recognition check needs the running service (watchwise serve).")]
        if chat_id in self._checking:
            return [Reply("A check is already running: walk past a camera, facing it.")]
        return [Reply(
            f"👋 <b>Walk past a camera, facing it</b>, within the next {self.check_s / 60:g} "
            "minutes. I'll tell you what Frigate recognized. If it doesn't recognize you, an "
            "ordinary alert card may come too.",
            check=True,
        )]

    async def check(self, chat_id: int) -> Reply:
        """Listen to Frigate's face results until a resident is recognized,
        or the person has walked past, or the window ends."""
        if self.face_feed is None:
            return Reply("The recognition check needs the running service (watchwise serve).")
        self._checking.add(chat_id)
        watch = _Watch()
        unsubscribe = self.face_feed.subscribe(watch.observe)
        try:
            await watch.wait(self.check_s, self.quiet_s)
        finally:
            unsubscribe()
            self._checking.discard(chat_id)
        return self._check_result(watch, await self._threshold())

    def _check_result(self, watch: _Watch, threshold: float | None) -> Reply:
        again: list[list[Button]] = [[("👋 Check again", "su:check"), ("Done", "su:done")]]
        if watch.confirmed is not None:
            name = watch.confirmed.name or "?"
            scores = [a.score for a in watch.attempts if a.name == name]
            how = f" ({max(scores):.2f})" if scores else ""
            return Reply(
                f"✅ <b>Recognized {esc(name)}</b>{how} on {esc(watch.confirmed.camera)}. While "
                "you're home (arming Home or Off), events in which Frigate recognizes "
                f"{esc(name)} are held back with no model call; while armed Away or Night, "
                "they're still checked.",
                again,
            )
        count = len(watch.attempts)
        times = f"{count} time{'' if count == 1 else 's'}"
        named = [a for a in watch.attempts if a.name]
        if named:
            best = max(named, key=lambda a: a.score)
            name = best.name or "?"
            below = (
                f", below the {threshold:.2f} Frigate needs to name someone"
                if threshold is not None and best.score < threshold
                else ""
            )
            text = (
                f"🤔 Frigate saw a face {times}; its best guess was <b>{esc(name)}</b> at "
                f"{best.score:.2f}{below}."
            )
            if name.lower() in {h.lower() for h in self.config.faces.household}:
                text += " More photos help: add one facing the camera in good light, then check again."
                return Reply(text, [[(f"📸 Add a photo of {name}", f"su:face:{name}")], *again])
            return Reply(text + f" {esc(name)} isn't a resident in Watchwise's config; /setup adds them.", again)
        if watch.attempts:
            return Reply(
                f"⚠️ Frigate saw a face {times} but matched nobody. Add photos of the residents "
                "(/face &lt;name&gt; while facing the camera), then check again.",
                again,
            )
        return Reply(
            f"No face reached Frigate's face recognition in {self.check_s / 60:g} minutes. It needs "
            "to see a face from the front, close enough to the camera. If you did walk past, "
            "check that face recognition is on in Frigate (and that Frigate was restarted once "
            "after it was turned on).",
            again,
        )

    # ---------------- helpers ----------------

    def _session(self, chat_id: int) -> _Session | None:
        session = self._sessions.get(chat_id)
        if session is None:
            return None
        if self._expired(session):
            self._sessions.pop(chat_id, None)
            return None
        session.touched = self.clock()
        return session

    def _expired(self, session: _Session) -> bool:
        return self.clock() - session.touched > SESSION_S

    @staticmethod
    def _ordered(session: _Session) -> list[str]:
        chosen = {h.lower() for h in session.answers.household}
        return [n for n in session.names if n.lower() in chosen]

    async def _library(self) -> dict[str, int]:
        try:
            return await self.nvr.faces()
        except Exception as exc:
            logger.info("no Face Library from the NVR (%s)", exc)
            return {}

    async def _frame(self, camera: str) -> bytes | None:
        try:
            frame = await self.nvr.latest_frame(camera)
            return downscale_jpeg(frame.jpeg, max_edge=1280, quality=80)
        except Exception as exc:
            logger.info("no frame from %s for /setup (%s)", camera, exc)
            return None

    async def _threshold(self) -> float | None:
        """The score Frigate needs before it names someone."""
        try:
            face = (await self.nvr.config()).get("face_recognition") or {}
            value = face.get("recognition_threshold")
            return float(value) if value is not None else None
        except Exception:
            return None

    def _arming_now(self) -> dict[str, Any] | None:
        try:
            return self.arming.get() if self.arming is not None else None
        except Exception:
            return None


def _ended() -> Reply:
    return Reply("This setup has ended or expired; send /setup to start again.")
