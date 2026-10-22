"""Layer 9: hardware detection, profiles, and guided setup.

The detection tests replay recorded command output from machines the suite
never runs on (NVIDIA, AMD, Apple Silicon, Windows, Linux), because the whole
point of the profile system is to behave correctly on hardware the developer
does not have. The catalog tests make sure every shipped profile produces a
config the service will actually load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from watchwise import cli
from watchwise.config import WatchwiseConfig, interpolate_env
from watchwise.hw.detect import GPUInfo, HardwareReport, detect_hardware
from watchwise.hw.profiles import CloudProvider, load_catalog, recommend, resolve
from watchwise.setup.plan import apply, plan_file
from watchwise.setup.render import build_config_dict, render_config

REPO = Path(__file__).resolve().parents[1]

NVIDIA_SMI = "NVIDIA GeForce RTX 3060, 12288, 555.85\nNVIDIA RTX A4000, 16376, 555.85\n"
ROCM_SMI = json.dumps(
    {"card0": {"Card Series": "Radeon RX 7900 XTX", "VRAM Total Memory (B)": str(24 * 1024**3)}}
)
WMI_INTEL_AND_NVIDIA = json.dumps(
    [
        {"Name": "Intel(R) Graphics", "AdapterRAM": 2147479552, "DriverVersion": "32.0"},
        {"Name": "NVIDIA GeForce RTX 3060", "AdapterRAM": 4293918720, "DriverVersion": "555"},
    ]
)


def runner_for(outputs: dict[str, tuple[int, str]]):
    """A fake command runner keyed by the command's first word."""

    def run(args, timeout=5.0):
        return outputs.get(args[0], (127, ""))

    return run


def report(accelerator="none", memory=None, ram=16.0, keys=()):
    gpus = []
    if accelerator == "nvidia":
        gpus = [GPUInfo("nvidia", "RTX", vram_gb=memory)]
    elif accelerator == "amd":
        gpus = [GPUInfo("amd", "Radeon", vram_gb=memory)]
    elif accelerator == "apple":
        gpus = [GPUInfo("apple", "Apple M2", vram_gb=ram, integrated=True)]
    else:
        gpus = [GPUInfo("intel", "Intel(R) Graphics", integrated=True)]
    from watchwise.hw.detect import API_KEY_VARS

    return HardwareReport(
        os="Linux",
        os_version="",
        arch="x86_64",
        cpu="cpu",
        cpu_threads=8,
        ram_gb=ram,
        gpus=gpus,
        api_keys={k: k in keys for k in API_KEY_VARS},
    )


# ---------------------------------------------------------------- detection


def test_nvidia_gpus_are_read_from_nvidia_smi():
    rep = detect_hardware(
        runner=runner_for({"nvidia-smi": (0, NVIDIA_SMI)}),
        read_text=lambda p: None,
        system="Linux",
        machine="x86_64",
        env={},
        which=lambda name: None,
    )
    assert [(g.name, g.vram_gb) for g in rep.gpus] == [
        ("NVIDIA GeForce RTX 3060", 12.0),
        ("NVIDIA RTX A4000", 16.0),
    ]
    assert rep.accelerator == "nvidia" and rep.accelerator_memory_gb == 16.0
    assert rep.docker == {"installed": False, "running": False}


def test_amd_gpus_are_read_from_rocm_smi():
    rep = detect_hardware(
        runner=runner_for({"rocm-smi": (0, ROCM_SMI)}),
        read_text=lambda p: None,
        system="Linux",
        machine="x86_64",
        env={},
        which=lambda name: None,
    )
    assert rep.accelerator == "amd"
    assert rep.gpus[0].name == "Radeon RX 7900 XTX" and rep.gpus[0].vram_gb == 24.0


def test_apple_silicon_uses_unified_memory():
    outputs = {"sysctl": (0, str(16 * 1024**3))}
    rep = detect_hardware(
        runner=runner_for(outputs), read_text=lambda p: None, system="Darwin",
        machine="arm64", env={}, which=lambda name: None,
    )
    assert rep.accelerator == "apple"
    assert rep.ram_gb == 16.0 and rep.accelerator_memory_gb == 16.0
    assert rep.npu == "Apple Neural Engine"


def test_windows_integrated_intel_graphics_is_not_an_accelerator():
    outputs = {
        "nvidia-smi": (127, ""),
        "powershell": (0, WMI_INTEL_AND_NVIDIA),
    }
    rep = detect_hardware(
        runner=runner_for(outputs), read_text=lambda p: None, system="Windows",
        machine="amd64", env={}, which=lambda name: None,
    )
    names = [(g.vendor, g.integrated) for g in rep.gpus]
    assert ("intel", True) in names
    # Without nvidia-smi the NVIDIA card is still listed, but VRAM is unknown,
    # so it cannot satisfy a profile that needs a minimum size.
    assert rep.accelerator == "nvidia" and rep.accelerator_memory_gb is None


def test_linux_gpus_found_through_sysfs(tmp_path):
    for card, vendor in (("card0", "0x8086"), ("card1", "0x1002")):
        device = tmp_path / "sys/class/drm" / card / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
    (tmp_path / "sys/class/drm/card0-HDMI-A-1").mkdir()
    rep = detect_hardware(
        runner=runner_for({}),
        read_text=lambda p: Path(p).read_text() if Path(p).exists() else None,
        system="Linux", machine="x86_64", env={}, sys_root=str(tmp_path), which=lambda n: None,
    )
    assert sorted(g.vendor for g in rep.gpus) == ["amd", "intel"]


def test_api_keys_are_reported_as_presence_only():
    rep = detect_hardware(
        runner=runner_for({}), read_text=lambda p: None, system="Linux", machine="x86_64",
        env={"OPENAI_API_KEY": "sk-secret"}, which=lambda n: None,
    )
    assert rep.api_keys["OPENAI_API_KEY"] is True
    assert rep.api_keys["ANTHROPIC_API_KEY"] is False
    assert "sk-secret" not in json.dumps(rep.to_dict())


# ---------------------------------------------------------------- profiles


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.mark.parametrize(
    "rep, expected",
    [
        (report("none"), "cpu-local"),
        (report("none", keys=("OPENAI_API_KEY",)), "cpu-hybrid"),
        (report("nvidia", memory=12.0), "nvidia"),
        (report("nvidia", memory=24.0), "nvidia-16gb"),
        (report("nvidia", memory=4.0), "cpu-local"),  # too small for local models
        (report("apple", ram=16.0), "apple-silicon"),
        (report("apple", ram=8.0), "apple-silicon-8gb"),
        (report("amd", memory=16.0), "amd-rocm"),
    ],
)
def test_profile_recommendation(catalog, rep, expected):
    assert recommend(catalog, rep).profile.name == expected


def test_cloud_providers_are_preferred_in_catalog_order(catalog):
    rec = recommend(catalog, report("none", keys=("OPENAI_API_KEY", "ANTHROPIC_API_KEY")))
    assert rec.profile.name == "cpu-hybrid" and rec.cloud.name == "anthropic"
    assert any("ANTHROPIC_API_KEY" in r for r in rec.reasons)
    with pytest.raises(ValueError, match="unknown cloud provider"):
        recommend(catalog, report("none"), cloud="nope")


def test_cloud_roles_get_per_role_tuning(catalog):
    models, roles = resolve(catalog.profiles["cpu-hybrid"], catalog.cloud_providers["anthropic"])
    assert roles == {"screener": "local", "investigator": "cloud", "chat": "cloud_chat"}
    assert models["cloud"]["model"] == "claude-opus-5-5"
    assert (models["cloud"]["effort"], models["cloud_chat"]["effort"]) == ("medium", "low")
    with pytest.raises(ValueError, match="needs a cloud API key"):
        resolve(catalog.profiles["cpu-hybrid"], None)


def _fake_cloud() -> CloudProvider:
    return CloudProvider(
        name="fake", envs=["FAKE_KEY"], label="Fake",
        endpoint={"provider": "openai_compat", "base_url": "https://example.invalid/v1",
                  "model": "fake"},
    )


def test_a_key_just_for_watchwise_wins_over_the_shared_one(catalog):
    both = report("none", keys=("WATCHWISE_OPENAI_API_KEY", "OPENAI_API_KEY"))
    rec = recommend(catalog, both)
    assert rec.cloud.env == "WATCHWISE_OPENAI_API_KEY" and rec.cloud.project_scoped
    models, roles = resolve(rec.profile, rec.cloud)
    assert models[roles["chat"]]["api_key_env"] == "WATCHWISE_OPENAI_API_KEY"
    assert any("just for Watchwise" in r for r in rec.reasons)

    shared = recommend(catalog, report("none", keys=("OPENAI_API_KEY",)))
    assert shared.cloud.env == "OPENAI_API_KEY" and not shared.cloud.project_scoped


def test_an_explicit_provider_without_a_key_points_at_the_project_variable(catalog):
    rec = recommend(catalog, report("none"), cloud="anthropic")
    models, roles = resolve(catalog.profiles["cpu-hybrid"], rec.cloud)
    assert models[roles["investigator"]]["api_key_env"] == "WATCHWISE_ANTHROPIC_API_KEY"


def test_every_key_variable_in_the_catalog_is_detected(catalog):
    from watchwise.hw.detect import API_KEY_VARS

    for provider in catalog.cloud_providers.values():
        assert provider.envs[0].startswith("WATCHWISE_"), provider.name
        assert set(provider.envs) <= set(API_KEY_VARS), provider.name


def test_doctor_says_which_key_it_found_without_showing_it(monkeypatch, capsys):
    rep = report("none", keys=("WATCHWISE_OPENAI_API_KEY",))
    monkeypatch.setattr("watchwise.hw.detect.detect_hardware", lambda **kw: rep)
    assert cli.main(["doctor", "--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "OpenAI: WATCHWISE_OPENAI_API_KEY (Watchwise only)" in out
    assert "cpu-hybrid" in out


def test_every_shipped_profile_renders_a_config_the_service_loads(catalog):
    for profile in catalog.profiles.values():
        models, roles = resolve(profile, _fake_cloud() if profile.needs_cloud else None)
        text = render_config(profile, build_config_dict(profile, models, roles))
        cfg = WatchwiseConfig.model_validate(interpolate_env(yaml.safe_load(text), env={}))
        assert cfg.profile == profile.name
        for role in ("screener", "investigator", "chat"):
            assert cfg.endpoint(role) is not None, f"{profile.name}: {role} has no model"
        for name, endpoint in cfg.models.items():
            if endpoint.provider == "ollama":
                assert endpoint.context_window, f"{profile.name}/{name} needs context_window"
        for overlay in profile.compose:
            assert (REPO / "deploy" / f"docker-compose.{overlay}.yml").is_file(), overlay


def test_rendering_is_deterministic(catalog):
    profile = catalog.profiles["cpu-local"]
    models, roles = resolve(profile, None)
    first = render_config(profile, build_config_dict(profile, models, roles))
    second = render_config(profile, build_config_dict(profile, models, roles))
    assert first == second
    assert "leaves this machine" in first


# ---------------------------------------------------------------- plan / apply


def test_plan_apply_is_idempotent_and_backs_up(tmp_path):
    target = tmp_path / "config" / "watchwise.yml"
    created = apply([plan_file(target, "a: 1\n")])[0]
    assert created.action == "create" and target.read_text() == "a: 1\n"

    assert plan_file(target, "a: 1\n").action == "unchanged"

    update = plan_file(target, "a: 2\n")
    assert update.action == "update" and "-a: 1" in update.diff and "+a: 2" in update.diff
    apply([update], now=0)
    assert target.read_text() == "a: 2\n"
    assert update.backup is not None and update.backup.read_text() == "a: 1\n"


def test_existing_secrets_are_never_overwritten(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-mine\n")
    change = plan_file(env, "OPENAI_API_KEY=\n", overwrite=False)
    assert change.action == "keep" and not change.writes
    apply([change])
    assert env.read_text() == "OPENAI_API_KEY=sk-mine\n"


# ---------------------------------------------------------------- CLI


@pytest.fixture
def fixed_machine(monkeypatch):
    rep = report("none")
    monkeypatch.setattr("watchwise.hw.detect.detect_hardware", lambda **kw: rep)
    return rep


def test_cli_profiles_lists_the_catalog(capsys):
    assert cli.main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "cpu-hybrid" in out and "leaves this machine" in out


def test_cli_doctor_json(fixed_machine, capsys):
    assert cli.main(["doctor", "--json", "--no-probe"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["recommended_profile"] == "cpu-local"
    assert data["report"]["accelerator"] == "none"


def test_cli_doctor_text_is_ascii(fixed_machine, capsys):
    assert cli.main(["doctor", "--no-probe"]) == 0
    out = capsys.readouterr().out
    assert "Recommended profile: cpu-local" in out
    out.encode("ascii")  # renders on any console code page


def test_cli_init_writes_then_is_a_no_op(fixed_machine, tmp_path, capsys):
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n")
    args = ["init", "--profile", "demo", "--root", str(tmp_path), "--yes"]
    assert cli.main(args) == 0
    cfg = WatchwiseConfig.load(tmp_path / "config" / "watchwise.yml")
    assert cfg.profile == "demo" and cfg.investigator_backend() == "scripted"
    assert (tmp_path / ".env").read_text() == "OPENAI_API_KEY=\n"

    capsys.readouterr()
    assert cli.main(args) == 0
    assert "Nothing to change." in capsys.readouterr().out


def test_cli_init_dry_run_writes_nothing(fixed_machine, tmp_path):
    assert cli.main(["init", "--profile", "cpu-local", "--root", str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "config").exists()


def test_cli_init_explains_a_missing_cloud_key(fixed_machine, tmp_path, capsys):
    assert cli.main(["init", "--profile", "cpu-hybrid", "--root", str(tmp_path), "--yes"]) == 2
    assert "needs a cloud API key" in capsys.readouterr().err


def test_cli_init_deploy_selects_compose_files(fixed_machine, tmp_path):
    assert cli.main(["init", "--profile", "cpu-local", "--root", str(tmp_path), "--deploy", "--yes"]) == 0
    env = (tmp_path / "deploy" / ".env").read_text()
    assert "COMPOSE_FILE=docker-compose.yml:docker-compose.ollama.yml" in env


def test_cli_check_with_stub_models(tmp_path, capsys):
    path = tmp_path / "watchwise.yml"
    path.write_text(
        "models: {stub: {provider: stub}}\n"
        "roles: {screener: stub, investigator: stub, chat: stub}\n"
    )
    assert cli.main(["check", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok  ]") == 3


def test_cli_check_json_prints_only_json(tmp_path, capsys):
    """TD-4: the header line used to precede the JSON on stdout."""
    path = tmp_path / "watchwise.yml"
    path.write_text("models: {stub: {provider: stub}}\nroles: {screener: stub, investigator: stub, chat: stub}\n")
    assert cli.main(["check", "--config", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    assert [r["status"] for r in json.loads(captured.out)] == ["ok", "ok", "ok"]
    assert "Checking models from" in captured.err


@pytest.fixture
def local_screener(tmp_path):
    path = tmp_path / "watchwise.yml"
    path.write_text(
        "models: {local: {provider: ollama, model: qwen3-vl:2b, max_tokens: 512}}\n"
        "roles: {screener: local}\n"
        "vlm: {timeout_s: 60}\n"
    )
    return path


def screener_answering(monkeypatch, behaviour, *, needs_loading=False):
    from watchwise.tier2.vlm.base import VLMBackend

    class Scripted(VLMBackend):
        name = "ollama"

        async def load(self):
            return needs_loading

        async def infer(self, frames, prompt, system):
            return behaviour()

    monkeypatch.setattr("watchwise.tier2.vlm.registry.build_vlm", lambda settings: Scripted(settings.model))


def test_cli_check_loads_the_model_before_timing_the_screen(local_screener, monkeypatch, capsys):
    reply = '{"verdict": "benign", "confidence": 0.9, "reason": "a cat"}'
    screener_answering(monkeypatch, lambda: reply, needs_loading=True)
    assert cli.main(["check", "--config", str(local_screener), "--role", "screener"]) == 0
    out = capsys.readouterr().out
    assert "[ok  ] screener" in out and "answered benign @ 0.90" in out and "; model load 0.0s" in out


def test_cli_check_fails_on_a_screener_timeout(local_screener, monkeypatch, capsys):
    """B-010: a timeout used to print [ok]."""
    import httpx

    def time_out():
        raise httpx.ReadTimeout("")

    screener_answering(monkeypatch, time_out)
    assert cli.main(["check", "--config", str(local_screener), "--role", "screener"]) == 1
    out = capsys.readouterr().out
    assert "[fail] screener" in out and "no reply within vlm.timeout_s = 60s" in out


def test_cli_check_warns_when_the_screener_is_cut_off(local_screener, monkeypatch, capsys):
    """B-011 seen from the CLI: the model answers, but never with a verdict."""
    from watchwise.tier2.vlm.base import VLMReply

    screener_answering(
        monkeypatch, lambda: VLMReply("", truncated=True, max_tokens=512, output_tokens=512, thinking=True)
    )
    assert cli.main(["check", "--config", str(local_screener), "--role", "screener"]) == 0
    out = capsys.readouterr().out
    assert "[warn] screener" in out
    assert "512-token output cap while it was still thinking" in out
    assert "Raise models.local.max_tokens" in out

    screener_answering(monkeypatch, lambda: "I think it is a cat")
    cli.main(["check", "--config", str(local_screener), "--role", "screener"])
    assert "[warn] screener" in capsys.readouterr().out


def test_no_profile_starves_a_thinking_screener(catalog):
    """Every shipped screener has room for qwen3-vl's reasoning (B-011)."""
    for profile in catalog.profiles.values():
        models, roles = resolve(profile, _fake_cloud() if profile.needs_cloud else None)
        text = render_config(profile, build_config_dict(profile, models, roles))
        cfg = WatchwiseConfig.model_validate(interpolate_env(yaml.safe_load(text), env={}))
        if cfg.screener_settings().backend != "stub":
            assert cfg.screener_settings().max_tokens >= 1024, profile.name


def test_cli_demo_runs_the_simulated_day(capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "PASS 2" in out and "Genuine alerts hidden     : 0" in out


def test_old_style_flags_still_mean_serve(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "cmd_serve", lambda args: seen.setdefault("args", args) and 0)
    cli.main(["--no-mqtt", "--port", "5999"])
    assert seen["args"].no_mqtt is True and seen["args"].port == 5999


def test_dotenv_loading_never_overrides_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport A_KEY="from-file"\nB_KEY=from-file\nC_KEY=\n')
    monkeypatch.setenv("B_KEY", "from-shell")
    monkeypatch.delenv("A_KEY", raising=False)
    loaded = cli.load_dotenv(env)
    import os

    assert os.environ["A_KEY"] == "from-file"
    assert os.environ["B_KEY"] == "from-shell"
    assert "B_KEY" not in loaded
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("C_KEY", raising=False)


def test_the_annotated_reference_config_loads():
    cfg = WatchwiseConfig.model_validate(
        interpolate_env(yaml.safe_load((REPO / "config.example.yml").read_text()), env={})
    )
    assert cfg.endpoint("investigator").provider == "ollama"
    assert cfg.investigator_backend() == "tools"
    assert cfg.screener_settings().max_tokens >= 1024


def test_synthetic_check_frame_is_a_jpeg():
    assert cli.synthetic_frame()[:2] == b"\xff\xd8"


# ---------------------------------------------------------------- deploy files


def test_compose_files_parse_and_wire_the_services():
    base = yaml.safe_load((REPO / "deploy" / "docker-compose.yml").read_text())
    assert {"mqtt", "frigate", "watchwise"} <= set(base["services"])
    watchwise = base["services"]["watchwise"]
    assert watchwise["environment"]["FRIGATE_URL"] == "http://frigate:5000"
    for overlay in sorted((REPO / "deploy").glob("docker-compose.*.yml")):
        data = yaml.safe_load(overlay.read_text())
        assert "services" in data, overlay.name
    frigate_cfg = yaml.safe_load((REPO / "deploy" / "frigate" / "config.example.yml").read_text())
    assert frigate_cfg["mqtt"]["host"] == "mqtt"
    assert frigate_cfg["mqtt"]["topic_prefix"] == WatchwiseConfig().mqtt.topic_prefix
