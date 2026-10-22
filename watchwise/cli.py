"""The ``watchwise`` command.

    watchwise doctor      what is this machine, and which profile fits it? (read-only)
    watchwise init        write config/watchwise.yml for a profile (plan -> confirm -> apply)
    watchwise check       talk to every configured model once; report latency and tool use
    watchwise demo        a simulated day through the full cascade, no cameras needed
    watchwise profiles    list the hardware profiles
    watchwise telegram    link a Telegram chat for phone alerts, or send a test card
    watchwise serve       run the service (the default; old-style flags still work)

Output is plain ASCII so it renders on any console.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

COMMANDS = ("serve", "doctor", "init", "check", "demo", "profiles", "telegram")
DEFAULT_CONFIG = Path("config/watchwise.yml")


# ---------------------------------------------------------------- helpers


def load_dotenv(path: Path = Path(".env")) -> list[str]:
    """Load KEY=VALUE lines from .env without overriding the real environment."""
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def resolve_config_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("WATCHWISE_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None


def _yes_no(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# ---------------------------------------------------------------- doctor


def format_doctor(report: Any, rec: Any, catalog: Any) -> str:
    out = io.StringIO()
    w = out.write
    w("Watchwise doctor (read-only: nothing on this machine is changed)\n\n")
    w(f"  Machine   {report.os} {report.os_version} ({report.arch})\n")
    w(f"  CPU       {report.cpu} - {report.cpu_threads} threads\n")
    w(f"  Memory    {report.ram_gb:g} GB\n")
    if report.gpus:
        for gpu in report.gpus:
            size = f", {gpu.vram_gb:g} GB" if gpu.vram_gb else ""
            kind = " (integrated)" if gpu.integrated and gpu.vendor != "apple" else ""
            w(f"  GPU       {gpu.name}{size}{kind}\n")
    else:
        w("  GPU       none found\n")
    if report.npu:
        w(f"  NPU       {report.npu} (reported only; not used for local LLMs)\n")
    docker = report.docker or {}
    w(
        "  Docker    "
        + (
            "running" if docker.get("running") else "installed, not running"
            if docker.get("installed") else "not installed"
        )
        + "\n"
    )
    accel = report.accelerator
    w(f"  => accelerator for local models: {accel}\n\n")

    if report.services:
        w("  Services\n")
        for s in report.services:
            mark = "[ok]" if s.reachable else "[--]"
            extra = f" ({', '.join(s.models[:4])}{'...' if len(s.models) > 4 else ''})" if s.models else ""
            w(f"    {mark} {s.name:<10} {s.url}  {s.detail}{extra}\n")
        w("\n")
    keys = report.api_keys or {}
    found = []
    for provider in catalog.cloud_providers.values():
        for env in provider.envs:
            if keys.get(env):
                scope = "Watchwise only" if env.startswith("WATCHWISE_") else "shared"
                found.append(f"{provider.label}: {env} ({scope})")
    w("  API keys  " + ("\n            ".join(found) if found else "none set") + "  [presence only]\n")
    w("            A key just for Watchwise goes in .env as WATCHWISE_<PROVIDER>_API_KEY;\n")
    w("            it takes precedence over the shared variable.\n\n")

    profile = rec.profile
    w(f"Recommended profile: {profile.name} - {profile.title}\n")
    for reason in rec.reasons:
        w(f"  why: {reason}\n")
    w(f"  {profile.summary}\n")
    try:
        from watchwise.hw.profiles import resolve

        models, roles = resolve(profile, rec.cloud)
        for role, name in roles.items():
            spec = models.get(name, {})
            w(f"  {role:<13}-> {spec.get('provider')} {spec.get('model')}\n")
    except ValueError as exc:
        w(f"  ({exc})\n")
    w(f"  leaves this machine: {profile.leaves_machine}\n")
    for note in profile.notes:
        w(f"  note: {note}\n")
    others = ", ".join(rec.alternatives)
    w(f"\nOther profiles: {others}  (see: watchwise profiles)\n")
    w(f"Next: watchwise init --profile {profile.name}\n")
    return out.getvalue()


def cmd_doctor(args: argparse.Namespace) -> int:
    from watchwise.hw.detect import detect_hardware, probe_services
    from watchwise.hw.profiles import load_catalog, recommend

    report = detect_hardware()
    if not args.no_probe:
        report.services = asyncio.run(
            probe_services(frigate_url=args.frigate_url, mqtt_host=args.mqtt_host)
        )
    catalog = load_catalog()
    rec = recommend(catalog, report, cloud=args.cloud)
    if args.json:
        print(
            json.dumps(
                {
                    "report": report.to_dict(),
                    "recommended_profile": rec.profile.name,
                    "cloud_provider": rec.cloud.name if rec.cloud else None,
                    "cloud_key_env": rec.cloud.env if rec.cloud else None,
                    "reasons": rec.reasons,
                    "alternatives": rec.alternatives,
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(format_doctor(report, rec, catalog), end="")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    from watchwise.hw.profiles import load_catalog

    catalog = load_catalog()
    for profile in catalog.profiles.values():
        print(f"{profile.name:<20} {profile.title}")
        print(f"{'':<20} leaves this machine: {profile.leaves_machine}")
    print("\nCloud providers (picked by which API key is set):")
    for p in catalog.cloud_providers.values():
        print(f"  {p.name:<12} {' or '.join(p.envs):<56} {p.endpoint.get('model')}")
    return 0


# ---------------------------------------------------------------- init


def compose_env(profile: Any) -> str:
    files = ["docker-compose.yml", *[f"docker-compose.{c}.yml" for c in profile.compose]]
    return (
        "# Written by `watchwise init --deploy`. Selects the compose files for this machine.\n"
        f"# Profile: {profile.name}\n"
        "COMPOSE_PROJECT_NAME=watchwise\n"
        "COMPOSE_PATH_SEPARATOR=:\n"
        f"COMPOSE_FILE={':'.join(files)}\n"
    )


def cmd_init(args: argparse.Namespace) -> int:
    from watchwise.config import WatchwiseConfig, interpolate_env
    from watchwise.hw.detect import detect_hardware
    from watchwise.hw.profiles import available_clouds, load_catalog, recommend, resolve
    from watchwise.setup.plan import apply, plan_file
    from watchwise.setup.render import build_config_dict, render_config

    root = Path(args.root)
    catalog = load_catalog()
    report = detect_hardware()

    if args.profile in (None, "auto"):
        rec = recommend(catalog, report, cloud=args.cloud)
        profile, cloud = rec.profile, rec.cloud
        print(f"Profile: {profile.name} (auto: {'; '.join(rec.reasons)})")
    else:
        if args.profile not in catalog.profiles:
            print(f"error: unknown profile {args.profile!r}. Try: watchwise profiles", file=sys.stderr)
            return 2
        profile = catalog.profiles[args.profile]
        cloud = None
        if profile.needs_cloud:
            if args.cloud:
                cloud = catalog.cloud_providers.get(args.cloud)
                if cloud is None:
                    print(f"error: unknown cloud provider {args.cloud!r}", file=sys.stderr)
                    return 2
            else:
                found = available_clouds(catalog, report)
                cloud = found[0] if found else None
        print(f"Profile: {profile.name} - {profile.title}")

    cloud_model = args.cloud_model
    if profile.needs_cloud and cloud is None:
        names = ", ".join(p.envs[0] for p in catalog.cloud_providers.values())
        print(
            f"error: profile {profile.name!r} needs a cloud API key. Put one of {names} "
            "in .env, or choose a local profile (watchwise profiles).",
            file=sys.stderr,
        )
        return 2
    if cloud is not None:
        print(f"Cloud:   {cloud.label} via {cloud.env}")
        if cloud.verify_model and not cloud_model:
            suggested = cloud.endpoint.get("model")
            if _interactive() and not args.yes:
                answer = input(f"Model for {cloud.label} [{suggested}]: ").strip()
                cloud_model = answer or None
            else:
                print(f"         model {suggested} is a suggestion; `watchwise check` verifies it.")

    models, roles = resolve(profile, cloud, cloud_model=cloud_model)
    config_dict = build_config_dict(profile, models, roles)
    text = render_config(profile, config_dict)
    # Never write a config the service would refuse to load.
    WatchwiseConfig.model_validate(interpolate_env(yaml.safe_load(text)))

    config_path = Path(args.config) if args.config else root / DEFAULT_CONFIG
    changes = [plan_file(config_path, text, reason=f"profile {profile.name}")]
    for example, target, why in (
        (root / ".env.example", root / ".env", "secrets file (API keys), from .env.example"),
        (
            root / "config" / "duty_roster.example.yml",
            root / "config" / "duty_roster.yml",
            "duty roster to edit, from the example",
        ),
    ):
        if example.is_file():
            changes.append(
                plan_file(target, example.read_text(encoding="utf-8"), reason=why, overwrite=False)
            )
    if args.deploy:
        changes.append(
            plan_file(root / "deploy" / ".env", compose_env(profile), reason="compose file selection")
        )
        frigate_example = root / "deploy" / "frigate" / "config.example.yml"
        if frigate_example.is_file():
            changes.append(
                plan_file(
                    root / "deploy" / "frigate" / "config.yml",
                    frigate_example.read_text(encoding="utf-8"),
                    reason="Frigate config to edit (add your cameras)",
                    overwrite=False,
                )
            )

    print(f"\nPrivacy: with this profile, {profile.leaves_machine} leaves this machine.\n")
    print("Plan:")
    for change in changes:
        print(f"  {change.action:<9} {change.path}  ({change.reason})")
        if change.diff:
            print("".join(f"      {line}" for line in change.diff.splitlines(keepends=True)))
    writes = [c for c in changes if c.writes]
    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0
    if not writes:
        print("\nNothing to change.")
        return 0
    if not args.yes and not _yes_no("\nApply these changes?"):
        print("Cancelled; nothing written.")
        return 1
    apply(changes)
    for change in writes:
        note = f" (previous version saved as {change.backup.name})" if change.backup else ""
        print(f"  wrote {change.path}{note}")

    pulls = sorted({m["model"] for m in models.values() if m.get("provider") == "ollama"})
    print("\nNext steps:")
    step = 1
    if pulls:
        print(f"  {step}. Pull the local model(s): " + " && ".join(f"ollama pull {m}" for m in pulls))
        step += 1
    if cloud is not None and not os.environ.get(cloud.env):
        print(f"  {step}. Put your key in .env: {cloud.env}=...")
        step += 1
    print(f"  {step}. watchwise check      # talk to each model once")
    print(f"  {step + 1}. watchwise demo       # a simulated day, no cameras needed")
    print(f"  {step + 2}. watchwise serve      # run against your NVR")
    return 0


# ---------------------------------------------------------------- check


@dataclass
class CheckResult:
    role: str
    target: str
    status: str  # ok | warn | fail | skip
    detail: str
    latency_ms: float = 0.0


def synthetic_frame() -> bytes:
    """A small drawn scene (a door and a figure-ish block), as JPEG."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), (118, 128, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 60, 120, 230), fill=(92, 64, 40))  # door
    draw.rectangle((200, 110, 232, 220), fill=(40, 40, 60))  # figure
    draw.ellipse((204, 84, 228, 108), fill=(196, 160, 130))  # head
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def _check_screener(config: Any) -> CheckResult:
    import time

    from watchwise.tier2 import prompts
    from watchwise.tier2.vlm.base import REPLY_PROBLEMS, TIMEOUT, TRUNCATED
    from watchwise.tier2.vlm.registry import build_vlm
    from watchwise.types import AlertContext, Frame

    settings = config.screener_settings()
    target = f"{settings.backend}/{settings.model}"
    try:
        vlm = build_vlm(settings)
    except Exception as exc:
        return CheckResult("screener", target, "fail", str(exc))
    try:
        now = time.time()
        ctx = AlertContext(review_id="check", camera="check_cam", start_time=now - 8, end_time=now, labels=["person"])
        frame = Frame(camera="check_cam", timestamp=now, jpeg=synthetic_frame())
        result = await vlm.screen([frame], prompts.build_prompt(ctx, 1), prompts.SYSTEM)
    finally:
        await vlm.aclose()
    if result.error == TIMEOUT:
        detail = (
            f"no reply within vlm.timeout_s = {settings.timeout_s:g}s. Loading a model for the "
            "first time can take longer: run check again, and raise vlm.timeout_s if it persists"
        )
        return CheckResult("screener", target, "fail", detail, result.latency_ms)
    if result.error in REPLY_PROBLEMS:
        # It answered, but every real alert would screen as "uncertain" and
        # escalate to the investigator.
        detail = result.reason[:200]
        if result.error == TRUNCATED:
            detail += (
                f". Raise {config.screener_max_tokens_key()} (thinking models need about 1000), "
                "or use a non-thinking model"
            )
        return CheckResult("screener", target, "warn", detail, result.latency_ms)
    if result.error:
        detail = result.reason[:300]
        if settings.backend == "ollama" and "connect" in str(result.error).lower():
            where = settings.base_url or "http://localhost:11434"
            detail = (
                f"Ollama is not reachable at {where}. Install it from https://ollama.com, "
                f"then run: ollama pull {settings.model}"
            )
        return CheckResult("screener", target, "fail", detail, result.latency_ms)
    tokens = f", {result.tokens['output']} output tokens" if result.tokens.get("output") else ""
    return CheckResult(
        "screener",
        target,
        "ok",
        f"answered {result.verdict.value} @ {result.confidence:.2f} (a drawn test image{tokens})",
        result.latency_ms,
    )


async def probe_tool_calling(chat: Any) -> tuple[str, str, float]:
    """One tool call and one follow-up. Returns (status, detail, latency_ms)."""
    from watchwise.llm.base import Message, ToolSpec

    tool = ToolSpec(
        "get_current_time",
        "Returns the current local time.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    )
    messages = [
        Message.system("You are checking that tool use works. Use the tool when asked."),
        Message.user("What time is it? Call get_current_time, then answer in one short sentence."),
    ]
    first = await chat.complete(messages, tools=[tool])
    if not first.tool_calls:
        if first.stop_reason == "length":
            return (
                "warn",
                f"the reply was cut off at the output cap ({first.usage.output_tokens} tokens) "
                "before a tool call; raise this model's max_tokens",
                first.latency_ms,
            )
        return (
            "warn",
            "replied without calling the tool; tool calling looks unreliable with this model",
            first.latency_ms,
        )
    messages += [first.as_message(), Message.tool_result(first.tool_calls[0], '{"time": "12:00"}')]
    second = await chat.complete(messages, tools=[tool])
    latency = first.latency_ms + second.latency_ms
    if not second.content.strip():
        if second.stop_reason == "length":
            return "warn", "called the tool, but the answer was cut off at the output cap", latency
        return "warn", "called the tool but gave no final answer", latency
    tokens = first.usage.input_tokens + second.usage.input_tokens
    return "ok", f"tool call + answer ({tokens} input tokens)", latency


async def _check_chat_role(config: Any, role: str) -> CheckResult:
    from watchwise.llm.registry import build_chat_model

    endpoint = config.endpoint(role)
    if role == "investigator":
        backend = config.investigator_backend()
        if backend == "scripted":
            return CheckResult(role, "scripted", "ok", "deterministic loop over the real tools (no model)")
        if backend == "claude_sdk":
            from watchwise.tier3.backends.claude_sdk import ClaudeSDKBackend

            ready, why = ClaudeSDKBackend.ensure_available()
            return CheckResult(role, f"claude_sdk/{config.agent.model}", "ok" if ready else "fail", why)
    if endpoint is None:
        return CheckResult(role, "-", "skip", "not configured")
    target = f"{endpoint.provider}/{endpoint.model}"
    try:
        chat = build_chat_model(endpoint)
    except Exception as exc:
        return CheckResult(role, target, "fail", str(exc))
    try:
        status, detail, latency = await probe_tool_calling(chat)
    except Exception as exc:
        return CheckResult(role, target, "fail", (str(exc) or type(exc).__name__)[:300])
    finally:
        await chat.aclose()
    return CheckResult(role, target, status, detail, latency)


async def run_checks(config: Any, roles: list[str] | None = None) -> list[CheckResult]:
    wanted = roles or ["screener", "investigator", "chat"]
    results = []
    for role in wanted:
        if role == "screener":
            results.append(await _check_screener(config))
        else:
            results.append(await _check_chat_role(config, role))
    return results


def cmd_check(args: argparse.Namespace) -> int:
    from watchwise.config import WatchwiseConfig

    path = resolve_config_path(args.config)
    config = WatchwiseConfig.load(path)
    # With --json, stdout carries only the JSON; the header goes to stderr.
    print(
        f"Checking models from {path or 'built-in defaults'} (profile: {config.profile or '-'})\n",
        file=sys.stderr if args.json else sys.stdout,
    )
    results = asyncio.run(run_checks(config, [args.role] if args.role else None))
    failed = [r for r in results if r.status == "fail"]
    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return 1 if failed else 0
    for r in results:
        latency = f"{r.latency_ms / 1000:.1f}s" if r.latency_ms else "-"
        print(f"  [{r.status:<4}] {r.role:<13} {r.target:<40} {latency:>7}  {r.detail}")
    if failed:
        print("\nFix the failures above, then run `watchwise check` again.")
    return 1 if failed else 0


# ---------------------------------------------------------------- telegram


def cmd_telegram(args: argparse.Namespace) -> int:
    from watchwise.config import WatchwiseConfig

    path = resolve_config_path(args.config)
    config = WatchwiseConfig.load(path)
    settings = config.notify.telegram
    token = os.environ.get(settings.bot_token_env, "")
    if not token:
        print(
            f"error: {settings.bot_token_env} is not set. In Telegram, talk to @BotFather, "
            f"send /newbot, and put the token it gives you in .env as "
            f"{settings.bot_token_env}=<token>.",
            file=sys.stderr,
        )
        return 2
    if args.telegram_command == "link":
        return asyncio.run(_telegram_link(args, path, config, token))
    return asyncio.run(_telegram_test(config, token))


async def _telegram_link(args: argparse.Namespace, path: Path | None, config: Any, token: str) -> int:
    import time

    from watchwise.config import WatchwiseConfig, interpolate_env
    from watchwise.notify.telegram import TelegramAPI, TelegramError
    from watchwise.setup.plan import apply, plan_file
    from watchwise.setup.render import replace_top_level_block

    if path is None or not path.is_file():
        print("error: no config file yet; run `watchwise init` first.", file=sys.stderr)
        return 2
    api = TelegramAPI(token, base_url=config.notify.telegram.api_base_url)
    try:
        me = await api.get_me()
        username = me.get("username", "your_bot")
        print(f"Bot: @{username}. In Telegram, open https://t.me/{username} and send /start.")
        print(f"Waiting up to {args.wait:g}s (stop `watchwise serve` first if it is running)...")
        deadline = time.monotonic() + args.wait
        offset: int | None = None
        chat: dict[str, Any] | None = None
        while chat is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for update in await api.get_updates(offset=offset, timeout=int(min(10, remaining))):
                offset = int(update["update_id"]) + 1
                message = update.get("message") or {}
                if chat is None and (message.get("text") or "").startswith("/start"):
                    chat = message.get("chat") or None
        if offset is not None:
            await api.get_updates(offset=offset, timeout=0)  # confirm, so they are not replayed
        if not chat:
            print("No /start received. Run this again when you are ready.")
            return 1
        chat_id = int(chat["id"])
        name = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
        print(f"Found chat {chat_id} {name}".rstrip())

        text = path.read_text(encoding="utf-8")
        notify = dict((yaml.safe_load(text) or {}).get("notify") or {})
        telegram = dict(notify.get("telegram") or {})
        chat_ids = [int(c) for c in telegram.get("chat_ids") or []]
        if chat_id not in chat_ids:
            chat_ids.append(chat_id)
        telegram.update({"enabled": True, "chat_ids": chat_ids})
        notify["telegram"] = telegram
        new_text = replace_top_level_block(text, "notify", notify)
        WatchwiseConfig.model_validate(interpolate_env(yaml.safe_load(new_text)))
        change = plan_file(path, new_text, reason="connect this Telegram chat")
        if change.writes:
            print(change.diff, end="")
            if not args.yes and not _yes_no("Apply?"):
                print("Cancelled; nothing written.")
                return 1
            apply([change])
            backup = f" (previous version saved as {change.backup.name})" if change.backup else ""
            print(f"  wrote {path}{backup}")
        else:
            print("This chat is already connected.")
        await api.send_message(
            chat_id,
            "✅ Linked. Watchwise alerts will come to this chat while "
            "<code>watchwise serve</code> is running.",
        )
        print("Next: watchwise telegram test   # sends a sample card to this chat")
        return 0
    except TelegramError as exc:
        print(f"error: Telegram said: {exc}", file=sys.stderr)
        return 1
    finally:
        await api.aclose()


async def _telegram_test(config: Any, token: str) -> int:
    import time

    from watchwise.notify.presenter import (
        CAPTION_LIMIT,
        TEXT_LIMIT,
        build_card,
        keyboard_markup,
    )
    from watchwise.notify.telegram import TelegramAPI, TelegramError
    from watchwise.types import AlertContext, Decision, Disposition, Severity, Stage

    settings = config.notify.telegram
    if not settings.chat_ids:
        print("error: no chat connected yet; run `watchwise telegram link` first.", file=sys.stderr)
        return 2
    now = time.time()
    ctx = AlertContext(
        review_id="test", camera="test_camera", start_time=now, end_time=now + 5,
        labels=["person"], zones=["driveway"],
    )
    decision = Decision(
        review_id="test", disposition=Disposition.ALERTED, severity=Severity.ALERT,
        confidence=0.82, rationale="test", stage=Stage.GATED,
    )
    card = build_card(ctx, decision)
    card.lines = [
        "This is a test card from Watchwise (a drawn image, not a camera).",
        "On real alerts the buttons work while `watchwise serve` is running.",
    ]
    markup = keyboard_markup("test", "standard")
    api = TelegramAPI(token, base_url=settings.api_base_url)
    try:
        for chat_id in settings.chat_ids:
            if settings.send_images:
                await api.send_photo(chat_id, synthetic_frame(), card.html(CAPTION_LIMIT), reply_markup=markup)
            else:
                await api.send_message(chat_id, card.html(TEXT_LIMIT), reply_markup=markup)
            print(f"  sent a test card to chat {chat_id}")
        return 0
    except TelegramError as exc:
        print(f"error: Telegram said: {exc}", file=sys.stderr)
        return 1
    finally:
        await api.aclose()


# ---------------------------------------------------------------- demo / serve


def cmd_demo(args: argparse.Namespace) -> int:
    from watchwise.config import WatchwiseConfig
    from watchwise.demo import run_demo

    models_from = None
    if args.real_models:
        path = resolve_config_path(args.config)
        models_from = WatchwiseConfig.load(path)
        print(
            f"Using the models from {path or 'built-in defaults'}. The fake NVR serves plain "
            "test images, so this checks the wiring under real latency, not judgement.\n"
        )
    result = asyncio.run(run_demo(models_from))
    return 1 if result.genuine_hidden else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from watchwise.config import WatchwiseConfig
    from watchwise.service import create_app

    config = WatchwiseConfig.load(resolve_config_path(args.config))
    if args.no_mqtt:
        config.mqtt.enabled = False
    host = args.host or config.service.host
    port = args.port or config.service.port
    logging.basicConfig(
        level=getattr(logging, config.service.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    uvicorn.run(create_app(config), host=host, port=port, log_level=config.service.log_level.lower())
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watchwise",
        description="An AI guard for your NVR: explainable alerts, on your own hardware.",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the service (default)")
    serve.add_argument("--config", default=None, help="config YAML (default: config/watchwise.yml)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--no-mqtt", action="store_true", help="accept alerts over HTTP only")
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="detect hardware and recommend a profile (read-only)")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--no-probe", action="store_true", help="skip probing local services")
    doctor.add_argument("--frigate-url", default=os.environ.get("FRIGATE_URL", "http://localhost:5000"))
    doctor.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "localhost"))
    doctor.add_argument("--cloud", default=None, help="prefer this cloud provider")
    doctor.set_defaults(func=cmd_doctor)

    init = sub.add_parser("init", help="write config/watchwise.yml for a hardware profile")
    init.add_argument("--profile", default="auto", help="profile name, or auto (default)")
    init.add_argument("--cloud", default=None, help="cloud provider for @cloud roles")
    init.add_argument("--cloud-model", default=None, help="override the cloud model")
    init.add_argument("--config", default=None, help="where to write the config")
    init.add_argument("--root", default=".", help="repository root (default: .)")
    init.add_argument("--deploy", action="store_true", help="also write deploy/.env for docker compose")
    init.add_argument("--dry-run", action="store_true", help="show the plan only")
    init.add_argument("--yes", "-y", action="store_true", help="apply without asking")
    init.set_defaults(func=cmd_init)

    check = sub.add_parser("check", help="talk to each configured model once")
    check.add_argument("--config", default=None)
    check.add_argument("--role", choices=["screener", "investigator", "chat"], default=None)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    demo = sub.add_parser("demo", help="simulate a day of alerts, no cameras needed")
    demo.add_argument("--real-models", action="store_true", help="use the configured models")
    demo.add_argument("--config", default=None)
    demo.set_defaults(func=cmd_demo)

    profiles = sub.add_parser("profiles", help="list hardware profiles")
    profiles.set_defaults(func=cmd_profiles)

    telegram = sub.add_parser("telegram", help="connect phone alerts (Telegram)")
    tsub = telegram.add_subparsers(dest="telegram_command", required=True)
    link = tsub.add_parser("link", help="connect a Telegram chat: send /start to your bot")
    link.add_argument("--config", default=None)
    link.add_argument("--wait", type=float, default=120.0, help="seconds to wait for /start")
    link.add_argument("--yes", "-y", action="store_true", help="apply without asking")
    link.set_defaults(func=cmd_telegram)
    test = tsub.add_parser("test", help="send a sample alert card to the connected chats")
    test.add_argument("--config", default=None)
    test.set_defaults(func=cmd_telegram)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatible: `watchwise --config x --no-mqtt` still means serve.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["serve", *argv]
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
