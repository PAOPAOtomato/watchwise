"""Choose a hardware profile and turn it into concrete ``models`` + ``roles``.

The catalog (``profiles.yml``) is data. This module only does three things:
match a profile to a ``HardwareReport``, pick a cloud provider when a profile
needs one, and resolve the ``"@cloud"`` placeholders into named endpoints.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from importlib import resources
from typing import Any

import yaml

from watchwise.hw.detect import HardwareReport

CLOUD = "@cloud"
PROJECT_KEY_PREFIX = "WATCHWISE_"


@dataclass
class CloudProvider:
    name: str
    # Variables that may hold the key, in precedence order: the WATCHWISE_
    # project key first, then the provider's usual (shared) variable.
    envs: list[str]
    label: str
    endpoint: dict[str, Any]
    role_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    verify_model: bool = False
    # Which of ``envs`` is actually set on this machine, once detected.
    key_env: str | None = None

    @property
    def env(self) -> str:
        """The variable the generated config names: the one found, else the
        project-scoped name, so a key added later goes where it should."""
        return self.key_env or self.envs[0]

    @property
    def project_scoped(self) -> bool:
        return self.env.startswith(PROJECT_KEY_PREFIX)


@dataclass
class Profile:
    name: str
    title: str
    summary: str
    models: dict[str, dict[str, Any]]
    roles: dict[str, str]
    when: dict[str, Any] | None = None
    agent_backend: str | None = None
    tuning: dict[str, dict[str, Any]] = field(default_factory=dict)
    compose: list[str] = field(default_factory=list)
    leaves_machine: str = "nothing"
    notes: list[str] = field(default_factory=list)

    @property
    def needs_cloud(self) -> bool:
        return CLOUD in self.roles.values()


@dataclass
class Catalog:
    profiles: dict[str, Profile]
    cloud_providers: dict[str, CloudProvider]
    auto_order: list[str]


@dataclass
class Recommendation:
    profile: Profile
    cloud: CloudProvider | None
    reasons: list[str]
    alternatives: list[str]


def load_catalog(text: str | None = None) -> Catalog:
    if text is None:
        text = resources.files("watchwise.hw").joinpath("profiles.yml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    providers = {
        name: CloudProvider(
            name=name,
            envs=list(spec["env"]) if isinstance(spec["env"], list) else [spec["env"]],
            label=spec.get("label", name),
            endpoint=dict(spec["endpoint"]),
            role_overrides=dict(spec.get("role_overrides") or {}),
            verify_model=bool(spec.get("verify_model", False)),
        )
        for name, spec in (raw.get("cloud_providers") or {}).items()
    }
    profiles = {
        name: Profile(
            name=name,
            title=spec.get("title", name),
            summary=" ".join(str(spec.get("summary", "")).split()),
            models={k: dict(v) for k, v in (spec.get("models") or {}).items()},
            roles={k: str(v) for k, v in (spec.get("roles") or {}).items()},
            when=spec.get("when"),
            agent_backend=spec.get("agent_backend"),
            tuning={k: dict(v) for k, v in (spec.get("tuning") or {}).items()},
            compose=list(spec.get("compose") or []),
            leaves_machine=str(spec.get("leaves_machine", "nothing")),
            notes=list(spec.get("notes") or []),
        )
        for name, spec in (raw.get("profiles") or {}).items()
    }
    order = [n for n in raw.get("auto_order") or [] if n in profiles]
    return Catalog(profiles=profiles, cloud_providers=providers, auto_order=order)


def available_clouds(catalog: Catalog, report: HardwareReport) -> list[CloudProvider]:
    """Cloud providers whose key is present, in catalog preference order.

    Each returned provider records which of its variables holds the key, the
    project-scoped one winning over the shared one.
    """
    found = []
    for provider in catalog.cloud_providers.values():
        present = next((e for e in provider.envs if report.api_keys.get(e)), None)
        if present:
            found.append(replace(provider, key_env=present))
    return found


def matches(
    profile: Profile,
    report: HardwareReport,
    *,
    cloud_key: bool,
    accelerator: str | None = None,
) -> bool:
    when = profile.when
    if not when:
        return False
    acc = accelerator or report.accelerator
    if "accelerator" in when and when["accelerator"] != acc:
        return False
    if "min_accelerator_gb" in when:
        memory = report.accelerator_memory_gb
        if memory is None or memory < float(when["min_accelerator_gb"]):
            return False
    if "cloud_key" in when and bool(when["cloud_key"]) != cloud_key:
        return False
    return True


def recommend(
    catalog: Catalog, report: HardwareReport, *, cloud: str | None = None
) -> Recommendation:
    """Pick the best profile for this machine, and explain why."""
    clouds = available_clouds(catalog, report)
    chosen_cloud = _pick_cloud(catalog, clouds, cloud)
    cloud_key = chosen_cloud is not None

    # A GPU too small for any GPU profile (or of unknown size) is treated as
    # no GPU at all: the CPU profiles still work, the GPU ones would not.
    attempts = [report.accelerator] + (["none"] if report.accelerator != "none" else [])
    for acc in attempts:
        for name in catalog.auto_order:
            profile = catalog.profiles[name]
            if not matches(profile, report, cloud_key=cloud_key, accelerator=acc):
                continue
            reasons = _reasons(report, profile, chosen_cloud)
            if acc != report.accelerator:
                memory = report.accelerator_memory_gb
                size = f"has {memory:g} GB" if memory else "has an unknown amount of memory"
                reasons[0] = (
                    f"the {report.accelerator} GPU {size}, below what the GPU profiles "
                    "need, so models run on the CPU"
                )
            alternatives = [n for n in catalog.profiles if n != name]
            return Recommendation(
                profile, chosen_cloud if profile.needs_cloud else None, reasons, alternatives
            )
    fallback = catalog.profiles["demo"]
    return Recommendation(
        fallback,
        None,
        ["no profile matched this machine; the demo profile needs nothing"],
        [n for n in catalog.profiles if n != "demo"],
    )


def _pick_cloud(
    catalog: Catalog, clouds: list[CloudProvider], requested: str | None
) -> CloudProvider | None:
    if requested:
        if requested not in catalog.cloud_providers:
            raise ValueError(
                f"unknown cloud provider {requested!r}; known: {sorted(catalog.cloud_providers)}"
            )
        detected = next((c for c in clouds if c.name == requested), None)
        return detected or catalog.cloud_providers[requested]
    return clouds[0] if clouds else None


def _reasons(report: HardwareReport, profile: Profile, cloud: CloudProvider | None) -> list[str]:
    acc = report.accelerator
    memory = report.accelerator_memory_gb
    if acc == "none":
        reasons = ["no NVIDIA, AMD (ROCm) or Apple Silicon GPU was found for local models"]
    else:
        size = f" with {memory:g} GB" if memory else ""
        reasons = [f"{acc} accelerator found{size}"]
    if profile.needs_cloud and cloud is not None:
        scope = "a key just for Watchwise" if cloud.project_scoped else "a key shared with other tools"
        reasons.append(f"{cloud.env} ({scope}) is set, so conversation can use {cloud.label}")
    elif acc == "none" and not report.cloud_keys:
        reasons.append("no cloud API key found, so everything stays local")
    return reasons


def resolve(
    profile: Profile,
    cloud: CloudProvider | None,
    *,
    cloud_model: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Concrete ``models`` and ``roles`` for a profile.

    ``"@cloud"`` roles become endpoints named ``cloud`` (or ``cloud_<role>``
    when a role needs different tuning, such as a lower effort for chat).
    """
    models = copy.deepcopy(profile.models)
    roles: dict[str, str] = {}
    variants: dict[str, str] = {}  # serialized endpoint -> name, to share identical ones
    for role, target in profile.roles.items():
        if target != CLOUD:
            roles[role] = target
            continue
        if cloud is None:
            raise ValueError(
                f"profile {profile.name!r} needs a cloud API key for the {role} role; "
                "add one to .env or choose another profile"
            )
        endpoint = dict(cloud.endpoint)
        endpoint["api_key_env"] = cloud.env
        if cloud_model:
            endpoint["model"] = cloud_model
        endpoint.update(cloud.role_overrides.get(role, {}))
        key = repr(sorted(endpoint.items()))
        if key not in variants:
            variants[key] = "cloud" if "cloud" not in models else f"cloud_{role}"
            models[variants[key]] = endpoint
        roles[role] = variants[key]
    return models, roles
