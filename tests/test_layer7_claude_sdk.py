"""The Claude Agent SDK backend: wiring, guard rails, and graceful fallback.

No API calls are made here. What is verified is everything that must be right
*before* a call is made -- that the agent is confined to the forensic tools,
that the in-process MCP server is attached, that a missing credential degrades
to the scripted backend instead of failing at the first alert, and that the
SDK's message stream is mirrored faithfully into our own audit trace.

The trace-mirroring test uses stand-in message objects shaped like the SDK's,
so the parsing logic is exercised without a subprocess or a model.
"""

from __future__ import annotations

import pytest

from sentinel.config import AgentSettings, SentinelConfig
from sentinel.memory.short_term import ShortTermMemory
from sentinel.tier3.backends.base import ToolRunner
from sentinel.tier3.backends.claude_sdk import ClaudeSDKBackend, credentials_present
from sentinel.tier3.backends.scripted import ScriptedBackend
from sentinel.tools.base import ToolContext
from sentinel.tools.registry import ALL_TOOLS
from sentinel.types import Investigation
from sentinel.wiring import build_agent_backend
from tests.conftest import make_alert


@pytest.fixture
def tool_ctx(nvr_client, store, patterns, tmp_path):
    from sentinel.tier2.vlm.stub import StubVLM

    return ToolContext(
        nvr=nvr_client,
        store=store,
        patterns=patterns,
        config_cache={"vlm": StubVLM(), "arming_path": str(tmp_path / "arming.json")},
    )


# ---------------- availability & fallback ----------------


def test_backend_reports_why_it_cannot_run(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    ok, why = ClaudeSDKBackend.ensure_available()
    assert ok is False
    assert "ANTHROPIC_API_KEY" in why
    assert "CLI profile" in why, "must not claim absence of credentials it cannot see"


def test_backend_reports_ready_with_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert credentials_present() is True
    ok, why = ClaudeSDKBackend.ensure_available()
    assert ok is True and why == "ready"


def test_wiring_falls_back_loudly_when_no_credentials(tool_ctx, monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    config = SentinelConfig(agent=AgentSettings(backend="claude_sdk"))

    with caplog.at_level("WARNING"):
        backend, status = build_agent_backend(config, tool_ctx)

    assert isinstance(backend, ScriptedBackend)
    assert status.startswith("fell back to scripted")
    assert any("falling back" in r.message for r in caplog.records)


def test_wiring_selects_the_sdk_backend_when_credentials_exist(tool_ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config = SentinelConfig(
        agent=AgentSettings(backend="claude_sdk", model="claude-opus-5-5")
    )
    backend, status = build_agent_backend(config, tool_ctx)
    assert isinstance(backend, ClaudeSDKBackend)
    assert backend.model == "claude-opus-5-5"
    assert status == "ready"


def test_scripted_is_the_default(tool_ctx):
    backend, status = build_agent_backend(SentinelConfig(), tool_ctx)
    assert isinstance(backend, ScriptedBackend)
    assert status == "configured"


# ---------------- options & tool confinement ----------------


def test_options_attach_the_in_process_mcp_server(tool_ctx):
    backend = ClaudeSDKBackend(tool_ctx, model="claude-opus-5-5")
    options = backend._options()
    assert "forensics" in options.mcp_servers
    assert options.mcp_servers["forensics"]["type"] == "sdk"


def test_the_agent_is_confined_to_the_forensic_tools(tool_ctx):
    """No filesystem, no bash, no web. An investigator reads evidence only."""
    options = ClaudeSDKBackend(tool_ctx)._options()
    assert options.tools == []
    assert len(options.allowed_tools) == len(ALL_TOOLS)
    assert all(name.startswith("mcp__forensics__") for name in options.allowed_tools)
    for forbidden in ("Bash", "Write", "Edit", "Read", "WebFetch", "WebSearch"):
        assert forbidden not in options.allowed_tools


def test_options_carry_the_budget_and_iteration_caps(tool_ctx):
    backend = ClaudeSDKBackend(
        tool_ctx, model="claude-opus-5-5", max_iterations=4, max_budget_usd=0.25,
        effort="low",
    )
    options = backend._options()
    assert options.max_turns == 4
    assert options.max_budget_usd == 0.25
    assert options.effort == "low"
    assert options.thinking == {"type": "adaptive"}
    assert options.setting_sources is None, (
        "a service agent must not inherit the developer's local Claude settings"
    )


def test_default_model_is_current(tool_ctx):
    assert ClaudeSDKBackend(tool_ctx).model == "claude-opus-5-5"


# ---------------- trace mirroring ----------------


class _Text:
    def __init__(self, text):
        self.text = text


class _Thinking:
    def __init__(self, thinking):
        self.thinking = thinking


class _ToolUse:
    def __init__(self, name, input):
        self.name = name
        self.input = input


class _ToolResult:
    def __init__(self, content):
        self.content = content


class _Assistant:
    def __init__(self, content):
        self.content = content


class _Result:
    def __init__(self, total_cost_usd):
        self.total_cost_usd = total_cost_usd


def _blocks():
    return (_Assistant, _Result, _Text, _Thinking, _ToolUse, _ToolResult)


def test_assistant_turns_become_react_iterations(tool_ctx):
    backend = ClaudeSDKBackend(tool_ctx)
    inv = Investigation(id="i", review_id="r", camera="cam_drive")
    memory = ShortTermMemory()
    runner = ToolRunner(tool_ctx, memory, inv)

    final = ""
    for turn in range(3):
        final = backend._absorb(
            _Assistant([_Text(f"turn {turn}")]), runner, memory, final, blocks=_blocks()
        )
    assert inv.iterations == 3
    assert final == "turn 2"


def test_tool_use_blocks_are_recorded_with_the_namespace_stripped(tool_ctx):
    backend = ClaudeSDKBackend(tool_ctx)
    inv = Investigation(id="i", review_id="r", camera="cam_drive")
    memory = ShortTermMemory()
    runner = ToolRunner(tool_ctx, memory, inv)

    backend._absorb(
        _Assistant(
            [
                _Thinking("I should check prior judgements first."),
                _ToolUse("mcp__forensics__search_similar_alerts", {"query": "car"}),
            ]
        ),
        runner,
        memory,
        "",
        blocks=_blocks(),
    )
    assert len(inv.tool_calls) == 1
    call = inv.tool_calls[0]
    assert call.name == "search_similar_alerts", "namespace must be stripped for the trace"
    assert call.arguments == {"query": "car"}
    assert call.iteration == 1
    assert inv.thoughts == ["I should check prior judgements first."]


def test_tool_results_land_in_short_term_memory(tool_ctx):
    backend = ClaudeSDKBackend(tool_ctx)
    inv = Investigation(id="i", review_id="r", camera="cam_drive")
    memory = ShortTermMemory()
    runner = ToolRunner(tool_ctx, memory, inv)

    backend._absorb(
        _Assistant([_ToolResult('{"ok": true, "cameras": {}}')]),
        runner,
        memory,
        "",
        blocks=_blocks(),
    )
    assert "tool_result" in memory.transcript()


def test_cost_is_captured_from_the_result_message(tool_ctx):
    backend = ClaudeSDKBackend(tool_ctx)
    inv = Investigation(id="i", review_id="r", camera="cam_drive")
    runner = ToolRunner(tool_ctx, ShortTermMemory(), inv)

    backend._absorb(_Result(0.0431), runner, ShortTermMemory(), "", blocks=_blocks())
    assert inv.cost_usd == pytest.approx(0.0431)
    assert inv.iterations == 0, "a result message is not a ReAct iteration"


async def test_a_broken_sdk_session_yields_uncertain_not_an_exception(tool_ctx, monkeypatch):
    """If the CLI or the credentials are wrong, the alert still gets handled."""
    backend = ClaudeSDKBackend(tool_ctx, model="claude-opus-5-5")

    def explode(*args, **kwargs):
        raise RuntimeError("claude CLI not found")

    monkeypatch.setattr(
        "claude_agent_sdk.ClaudeSDKClient", explode, raising=False
    )

    inv = Investigation(id="i", review_id="r", camera="cam_drive")
    memory = ShortTermMemory()
    memory.seed(make_alert())
    conclusion = await backend.run(make_alert(), None, memory, ToolRunner(tool_ctx, memory, inv))

    from sentinel.types import Verdict

    assert conclusion.verdict is Verdict.UNCERTAIN
    assert conclusion.confidence == 0.0
    assert "could not complete" in conclusion.conclusion


# ---------------- the prompt contract ----------------


def test_system_prompt_states_the_anti_guessing_rule():
    from sentinel.tier3 import prompts

    assert "Do not guess" in prompts.SYSTEM
    assert "2-3 rounds" in prompts.SYSTEM
    assert "uncertain" in prompts.SYSTEM


def test_task_prompt_carries_the_alert_and_the_screening_reason():
    from sentinel.tier3 import prompts
    from sentinel.types import ScreenResult, Verdict

    screen = ScreenResult(
        verdict=Verdict.UNCERTAIN, confidence=0.4, reason="too dark to tell",
        backend="stub", model="m",
    )
    memory = ShortTermMemory()
    alert = make_alert(review_id="rev-77", camera="cam_yard", labels=["person"])
    memory.seed(alert, screen)

    prompt = prompts.build_task_prompt(alert, screen, memory.briefing())
    assert "rev-77" in prompt
    assert "cam_yard" in prompt
    assert "too dark to tell" in prompt
    assert "WORKING MEMORY" in prompt


def test_tool_manifest_covers_every_tool():
    from sentinel.tier3 import prompts

    manifest = prompts.tool_manifest()
    for tool in ALL_TOOLS:
        assert tool.name in manifest
