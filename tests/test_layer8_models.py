"""Layer 8: the provider-neutral model layer and the tool-calling investigator.

What is asserted here is what lets a user swap hardware and models by config:

* every provider turns the same neutral messages into its own wire format and
  normalizes the reply back (tested over real HTTP against a fake server, or
  the SDK's own transport for Anthropic)
* the untidy habits of small local models -- tool calls written into prose,
  ``<think>`` blocks, malformed arguments -- are absorbed in one place
* the ``tools`` investigation backend reaches a conclusion on any provider,
  and every way a model can misbehave ends in "uncertain", never in an
  exception on the alert path or a silent suppression
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import httpx
import pytest

from watchwise.config import (
    AgentSettings,
    MemorySettings,
    ModelEndpoint,
    RolesSettings,
    WatchwiseConfig,
    VLMSettings,
    interpolate_env,
)
from watchwise.fakes.llm_server import FakeLLMState, build_fake_llm
from watchwise.llm.base import (
    INVALID_ARGUMENTS,
    ChatModelError,
    ChatResult,
    ImagePart,
    Message,
    ToolCallRequest,
    ToolSpec,
    extract_text_tool_calls,
    openai_url,
    parse_arguments,
    split_reasoning,
)
from watchwise.llm.ollama import OllamaChat
from watchwise.llm.openai_compat import OpenAICompatChat
from watchwise.llm.registry import build_chat_model
from watchwise.llm.stub import StubChat
from watchwise.tier2.vlm.chat_adapter import ChatModelVLM
from watchwise.tier3.backends.scripted import ScriptedBackend
from watchwise.tier3.backends.tool_loop import NUDGE, ToolLoopBackend
from watchwise.tier3.investigator import Investigator
from watchwise.tools.base import ToolContext
from watchwise.types import Frame, ScreenResult, Verdict
from watchwise.wiring import build_agent_backend, build_system
from tests.conftest import NIGHT_TS, make_alert

TOOL = ToolSpec(
    "get_device_status",
    "Camera and system health.",
    {"type": "object", "properties": {"camera": {"type": "string"}}, "required": [], "additionalProperties": False},
)
UNCERTAIN = ScreenResult(
    verdict=Verdict.UNCERTAIN, confidence=0.4, reason="could not tell", backend="stub", model="stub-vlm"
)


@pytest.fixture
def fake_llm():
    app, state = build_fake_llm(FakeLLMState())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fake-llm")
    return state, client


@pytest.fixture
def tool_ctx(nvr_client, store, patterns, tmp_path):
    from watchwise.tier2.vlm.stub import StubVLM

    return ToolContext(
        nvr=nvr_client,
        store=store,
        patterns=patterns,
        config_cache={"vlm": StubVLM(), "arming_path": str(tmp_path / "arming.json")},
    )


def investigator_for(chat, tool_ctx, store, *, max_iterations: int = 4) -> Investigator:
    return Investigator(
        ToolLoopBackend(chat, max_iterations=max_iterations),
        tool_ctx,
        store,
        AgentSettings(timeout_s=30.0),
        MemorySettings(),
    )


def conclusion(verdict: str = "benign", confidence: float = 0.9, text: str = "settled") -> ChatResult:
    return ChatResult(
        content=json.dumps(
            {"verdict": verdict, "confidence": confidence, "conclusion": text, "evidence": ["x"]}
        )
    )


def call(name: str = "get_device_status", **arguments) -> ChatResult:
    return ChatResult(
        content="",
        tool_calls=[ToolCallRequest(id=f"c-{name}", name=name, arguments=arguments)],
        stop_reason="tool_calls",
    )


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://localhost:11434", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/", "http://localhost:11434/v1/chat/completions"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/chat/completions"),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
    ],
)
def test_openai_url_accepts_both_base_url_conventions(base, expected):
    assert openai_url(base, "chat/completions") == expected


def test_think_blocks_are_separated_from_the_answer():
    visible, reasoning = split_reasoning("<think>check the roster first</think>\n{\"verdict\": \"benign\"}")
    assert visible == '{"verdict": "benign"}'
    assert reasoning == "check the roster first"
    # A reply cut off mid-thought has no answer, only reasoning.
    assert split_reasoning("<think>still going") == ("", "still going")


def test_arguments_are_parsed_forgivingly_but_never_guessed():
    assert parse_arguments('{"camera": "cam_door"}') == {"camera": "cam_door"}
    assert parse_arguments('sure: {"camera": "cam_door"}') == {"camera": "cam_door"}
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments(None) == {}
    assert INVALID_ARGUMENTS in parse_arguments("{camera: cam_door")


def test_tool_calls_written_into_text_are_recovered():
    names = {"get_device_status", "check_duty_roster"}
    tagged = 'Let me look.\n<tool_call>{"name": "get_device_status", "arguments": {"camera": "cam_yard"}}</tool_call>'
    calls, rest = extract_text_tool_calls(tagged, names)
    assert [(c.name, c.arguments) for c in calls] == [("get_device_status", {"camera": "cam_yard"})]
    assert rest == "Let me look."

    fenced = '```json\n{"tool": "check_duty_roster", "args": {"camera": "cam_door"}}\n```'
    calls, rest = extract_text_tool_calls(fenced, names)
    assert calls[0].name == "check_duty_roster" and rest == ""


def test_a_final_answer_is_never_mistaken_for_a_tool_call():
    answer = '{"verdict": "benign", "confidence": 0.9, "conclusion": "ok"}'
    assert extract_text_tool_calls(answer, {"get_device_status"}) == ([], answer)
    unknown = '{"name": "rm_rf", "arguments": {}}'
    assert extract_text_tool_calls(unknown, {"get_device_status"})[0] == []


# ---------------------------------------------------------------- OpenAI-compatible


async def test_openai_compat_round_trip_with_native_tool_calls(fake_llm, monkeypatch):
    state, client = fake_llm
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    state.replies.append(
        {"tool_calls": [{"id": "call_7", "name": "get_device_status", "arguments": {"camera": "cam_door"}}]}
    )
    chat = OpenAICompatChat("fake-model", "http://fake-llm", api_key_env="FAKE_KEY", client=client)
    image = ImagePart(b"\xff\xd8jpeg")
    result = await chat.complete(
        [Message.system("sys"), Message.user("look", images=[image])], tools=[TOOL]
    )
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls == [ToolCallRequest("call_7", "get_device_status", {"camera": "cam_door"})]
    assert result.usage.input_tokens == 11

    sent = state.requests[0]
    assert state.headers[0]["authorization"] == "Bearer sk-test"
    assert sent["tools"][0]["function"]["name"] == "get_device_status"
    assert sent["tools"][0]["function"]["parameters"] == TOOL.parameters
    image_part = sent["messages"][1]["content"][1]
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()

    # The follow-up carries the assistant call and the tool result in OpenAI's shape.
    await chat.complete(
        [
            Message.user("look"),
            result.as_message(),
            Message.tool_result(result.tool_calls[0], '{"ok": true}'),
        ],
        tools=[TOOL],
    )
    followup = state.requests[1]["messages"]
    assert followup[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"camera": "cam_door"})
    assert followup[2] == {"role": "tool", "tool_call_id": "call_7", "content": '{"ok": true}'}


async def test_openai_compat_recovers_tool_calls_and_strips_thinking(fake_llm):
    state, client = fake_llm
    state.replies.append(
        {"content": '<think>hmm</think><tool_call>{"name": "get_device_status", "arguments": {}}</tool_call>'}
    )
    chat = OpenAICompatChat("fake-model", "http://fake-llm", client=client)
    result = await chat.complete([Message.user("go")], tools=[TOOL])
    assert [c.name for c in result.tool_calls] == ["get_device_status"]
    assert result.content == "" and result.reasoning == "hmm"


def test_a_configured_key_that_is_missing_fails_clearly(monkeypatch):
    monkeypatch.delenv("WATCHWISE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ChatModelError, match="WATCHWISE_OPENAI_API_KEY is not set"):
        OpenAICompatChat("gpt", "https://api.openai.com/v1", api_key_env="WATCHWISE_OPENAI_API_KEY")


def test_a_missing_project_key_never_falls_back_to_another_claude_credential(monkeypatch):
    pytest.importorskip("anthropic")
    from watchwise.llm.anthropic import AnthropicChat

    monkeypatch.delenv("WATCHWISE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-key")  # must not be used instead
    with pytest.raises(ChatModelError, match="WATCHWISE_ANTHROPIC_API_KEY is not set"):
        AnthropicChat("claude-opus-5-5", api_key_env="WATCHWISE_ANTHROPIC_API_KEY")


async def test_openai_compat_errors_become_chat_model_errors(fake_llm):
    state, client = fake_llm
    state.status = 500
    chat = OpenAICompatChat("fake-model", "http://fake-llm", client=client)
    with pytest.raises(ChatModelError, match="HTTP 500: fake failure$"):
        await chat.complete([Message.user("go")])
    assert await chat.list_models() == ["fake-model"]


async def test_a_rejected_token_parameter_is_switched_once_and_remembered(fake_llm):
    """OpenAI's GPT-5 models refuse max_tokens and want max_completion_tokens."""
    state, client = fake_llm
    state.reject_params = {
        "max_tokens": "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    }
    chat = OpenAICompatChat("fake-model", "http://fake-llm", client=client, max_tokens=321)
    first = await chat.complete([Message.user("one")])
    second = await chat.complete([Message.user("two")])
    assert first.content and second.content
    assert len(state.requests) == 3  # one rejected attempt, then remembered
    assert "max_tokens" in state.requests[0]
    assert state.requests[1]["max_completion_tokens"] == 321 and "max_tokens" not in state.requests[1]
    assert "max_tokens" not in state.requests[2]


async def test_rejected_temperature_and_effort_are_dropped(fake_llm):
    state, client = fake_llm
    state.reject_params = {
        "temperature": "Unsupported value: 'temperature' does not support 0 with this model.",
        "reasoning_effort": "Unrecognized request argument supplied: reasoning_effort",
    }
    chat = OpenAICompatChat("fake-model", "http://fake-llm", client=client, effort="low")
    result = await chat.complete([Message.user("go")])
    assert result.content
    assert state.requests[0]["reasoning_effort"] == "low"
    final = state.requests[-1]
    assert "temperature" not in final and "reasoning_effort" not in final


def test_openai_hosts_start_with_the_parameter_they_require():
    from watchwise.llm.openai_compat import RequestTuning

    assert RequestTuning("https://api.openai.com/v1").token_param == "max_completion_tokens"
    assert RequestTuning("http://localhost:1234").token_param == "max_tokens"
    assert RequestTuning("http://x", effort="max").reasoning_effort == "high"


async def test_openai_compat_screener_adapts_too(fake_llm):
    from watchwise.tier2.vlm.remote import OpenAICompatVLM

    state, client = fake_llm
    state.reject_params = {"max_tokens": "Use 'max_completion_tokens' instead.", "temperature": "x"}
    state.replies.append({"content": '{"verdict": "benign", "confidence": 0.9, "reason": "a cat"}'})
    vlm = OpenAICompatVLM("fake-model", "http://fake-llm", client=client, effort="low")
    result = await vlm.screen([Frame(camera="c", timestamp=0.0, jpeg=b"jpg")], "prompt", "system")
    assert (result.verdict, result.confidence, result.error) == (Verdict.BENIGN, 0.9, None)
    final = state.requests[-1]
    assert final["max_completion_tokens"] and "temperature" not in final
    assert final["response_format"] == {"type": "json_object"}


THINKING_VERDICT = {
    "content": '{"verdict": "benign", "confidence": 0.9, "reason": "a cat on the path"}',
    "thinking": "The frame shows a small animal near the door...",
    "tokens": 700,  # what qwen3-vl spends: 430-660 measured on the L4, mostly thinking
}


async def test_a_thinking_screener_answers_within_its_models_cap(fake_llm):
    """B-011: the screener used a fixed 512 (256 on CPU profiles), which a
    thinking model fills before it answers, so every alert came back empty."""
    from watchwise.tier2.vlm.remote import OllamaVLM

    state, client = fake_llm
    state.models = ["qwen3-vl:2b"]
    state.replies = [dict(THINKING_VERDICT), dict(THINKING_VERDICT)]
    cfg = WatchwiseConfig.model_validate(
        {
            "models": {"local": {"provider": "ollama", "model": "qwen3-vl:2b", "max_tokens": 1024}},
            "roles": {"screener": "local"},
            "vlm": {"frames": 1, "timeout_s": 120},
        }
    )
    settings = cfg.screener_settings()
    frame = Frame(camera="c", timestamp=0.0, jpeg=b"jpg")

    vlm = OllamaVLM(settings.model, "http://fake-llm", max_tokens=settings.max_tokens, client=client)
    result = await vlm.screen([frame], "prompt", "system")
    assert (result.verdict, result.confidence, result.error) == (Verdict.BENIGN, 0.9, None)
    assert result.tokens == {"input": 13, "output": 700}
    assert state.requests[0]["options"]["num_predict"] == 1024

    old_cap = OllamaVLM("qwen3-vl:2b", "http://fake-llm", max_tokens=512, client=client)
    cut = await old_cap.screen([frame], "prompt", "system")
    assert (cut.verdict, cut.confidence, cut.error) == (Verdict.UNCERTAIN, 0.0, "truncated")
    assert "cut off at the 512-token output cap while it was still thinking" in cut.reason


async def test_a_cut_off_reply_is_reported_by_openai_compatible_screeners_too(fake_llm):
    from watchwise.tier2.vlm.remote import OpenAICompatVLM

    state, client = fake_llm
    state.replies = [dict(THINKING_VERDICT)]
    vlm = OpenAICompatVLM("fake-model", "http://fake-llm", client=client, max_tokens=400)
    result = await vlm.screen([Frame(camera="c", timestamp=0.0, jpeg=b"jpg")], "prompt", "system")
    assert (result.verdict, result.error) == (Verdict.UNCERTAIN, "truncated")
    assert "400-token output cap while it was still thinking" in result.reason
    assert result.tokens["output"] == 400


async def test_a_screener_model_that_is_not_pulled_says_how_to_fix_it(fake_llm):
    from watchwise.tier2.vlm.remote import OllamaVLM

    _, client = fake_llm
    vlm = OllamaVLM("qwen3-vl:8b-instruct", "http://fake-llm", client=client)
    result = await vlm.screen([Frame(camera="c", timestamp=0.0, jpeg=b"jpg")], "prompt", "system")
    assert result.verdict is Verdict.UNCERTAIN
    assert "ollama pull qwen3-vl:8b-instruct" in result.error


# ---------------------------------------------------------------- Ollama native


async def test_ollama_sends_context_length_and_parses_tool_calls(fake_llm):
    state, client = fake_llm
    state.models = ["qwen3-vl:4b"]
    state.replies.append(
        {"thinking": "roster first", "tool_calls": [{"name": "get_device_status", "arguments": {"camera": "cam_yard"}}]}
    )
    chat = OllamaChat("qwen3-vl:4b", "http://fake-llm", context_window=8192, keep_alive="30m", client=client)
    result = await chat.complete(
        [Message.system("sys"), Message.user("look", images=[ImagePart(b"img")])], tools=[TOOL]
    )
    sent = state.requests[0]
    assert sent["options"]["num_ctx"] == 8192
    assert sent["keep_alive"] == "30m"
    assert sent["stream"] is False
    assert sent["messages"][1]["images"] == [base64.b64encode(b"img").decode()]
    assert result.tool_calls[0].arguments == {"camera": "cam_yard"}
    assert result.reasoning == "roster first"
    assert result.usage.input_tokens == 13

    await chat.complete(
        [result.as_message(), Message.tool_result(result.tool_calls[0], "{}")], tools=[TOOL]
    )
    tool_msg = state.requests[1]["messages"][1]
    assert tool_msg == {"role": "tool", "content": "{}", "tool_name": "get_device_status"}


async def test_a_thinking_model_can_be_told_not_to_think(fake_llm):
    """M2-14: hidden reasoning was most of the investigator's time per call."""
    state, client = fake_llm
    state.models = ["qwen3:14b"]
    quiet = build_chat_model(ModelEndpoint(provider="ollama", model="qwen3:14b", think=False))
    quiet._client = client
    await quiet.complete([Message.user("hi")])
    assert state.requests[-1]["think"] is False
    default = OllamaChat("qwen3:14b", "http://fake-llm", client=client)
    await default.complete([Message.user("hi")])
    assert "think" not in state.requests[-1]


async def test_ollama_missing_model_says_how_to_fix_it(fake_llm):
    _, client = fake_llm
    chat = OllamaChat("not-pulled:7b", "http://fake-llm", client=client)
    with pytest.raises(ChatModelError, match="ollama pull not-pulled:7b"):
        await chat.complete([Message.user("hi")])


# ---------------------------------------------------------------- warm-up


async def test_warm_up_loads_each_local_model_with_the_context_it_will_use(fake_llm, caplog):
    from watchwise.tier2.vlm.remote import OllamaVLM
    from watchwise.warmup import ModelWarmer

    state, client = fake_llm
    state.models = ["qwen3-vl:8b", "qwen3:14b"]
    state.load_delay_s = 0.05
    vlm = OllamaVLM("qwen3-vl:8b", "http://fake-llm", context_window=8192, keep_alive="30m", client=client)
    chat = OllamaChat("qwen3:14b", "http://fake-llm", context_window=16384, keep_alive="30m", client=client)
    warmer = ModelWarmer(vlm, [("investigator", chat)])
    with caplog.at_level(logging.INFO, logger="watchwise.warmup"):
        await warmer.warm()

    # A different num_ctx would make Ollama load the model again on the first
    # real request. The screener comes last as well as first, so if the two
    # models don't fit in memory together, the alert path keeps its model.
    sent = [(r["model"], r["options"]["num_ctx"], r["keep_alive"]) for r in state.requests]
    assert sent == [("qwen3-vl:8b", 8192, "30m"), ("qwen3:14b", 16384, "30m"), ("qwen3-vl:8b", 8192, "30m")]
    assert "prompt" not in state.requests[0] and state.requests[1]["messages"] == []
    assert warmer.status["screener"]["ready"] and warmer.status["investigator"]["ready"]
    assert "warm-up: screener model ollama/qwen3-vl:8b ready after" in caplog.text
    assert "investigator model ollama/qwen3:14b ready after" in caplog.text


async def test_a_screen_while_the_model_is_loading_times_out_and_says_why(fake_llm):
    from watchwise.tier2.vlm.remote import OllamaVLM
    from watchwise.warmup import ModelWarmer

    state, client = fake_llm
    state.models = ["qwen3-vl:8b"]
    state.load_delay_s = 0.5  # a cold model takes longer to load than the screener waits
    state.replies.append({"content": '{"verdict": "benign", "confidence": 0.9, "reason": "a cat"}'})
    vlm = OllamaVLM("qwen3-vl:8b", "http://fake-llm", timeout_s=0.1, client=client)
    warming = asyncio.create_task(ModelWarmer(vlm).warm())
    await asyncio.sleep(0.02)
    frame = Frame(camera="c", timestamp=0.0, jpeg=b"jpg")

    early = await vlm.screen([frame], "prompt", "system")
    assert (early.verdict, early.confidence, early.error) == (Verdict.UNCERTAIN, 0.0, "timeout")
    assert "did not answer within 0.1s" in early.reason
    assert "the model was still being loaded" in early.reason

    await warming
    later = await vlm.screen([frame], "prompt", "system")
    assert (later.verdict, later.error) == (Verdict.BENIGN, None)


async def test_the_warmer_keeps_the_screener_loaded_until_stopped(fake_llm):
    from watchwise.tier2.vlm.remote import OllamaVLM
    from watchwise.tier2.vlm.stub import StubVLM
    from watchwise.warmup import ModelWarmer

    state, client = fake_llm
    state.models = ["qwen3-vl:2b"]
    warmer = ModelWarmer(OllamaVLM("qwen3-vl:2b", "http://fake-llm", client=client), refresh_s=0.02)
    await warmer.start()
    await asyncio.sleep(0.15)
    await warmer.stop()
    touched = len(state.requests)
    assert touched >= 3
    await asyncio.sleep(0.05)
    assert len(state.requests) == touched

    nothing_to_load = ModelWarmer(StubVLM(), refresh_s=0.01)
    await nothing_to_load.start()
    await asyncio.sleep(0.03)
    assert nothing_to_load.status == {} and nothing_to_load._task.done()
    await nothing_to_load.stop()


@pytest.mark.parametrize(
    "keep_alive, every",
    [("30m", 900), ("1h30m", 2700), ("300", 150), (None, 150), ("10s", 30), ("-1", None), ("0", None), ("soon", 150)],
)
def test_the_screener_is_touched_well_before_ollama_would_unload_it(keep_alive, every):
    from watchwise.warmup import refresh_interval

    assert refresh_interval(keep_alive) == every


# ---------------------------------------------------------------- Anthropic


def _anthropic_chat(monkeypatch, responses, model="claude-opus-5-5", **kwargs):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from watchwise.llm.anthropic import AnthropicChat

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: list[dict] = []

    def handler(request):
        captured.append(
            {"headers": dict(request.headers), "body": json.loads(request.content), "path": request.url.path}
        )
        return httpx2.Response(200, json=responses.pop(0))

    client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicChat(model, http_client=client, max_retries=0, **kwargs), captured


def _message(content, stop_reason="end_turn", **extra):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 9, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 0},
        **extra,
    }


async def test_anthropic_request_shape_and_tool_use_parsing(monkeypatch):
    tool_turn = _message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "text", "text": "Checking the camera."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_device_status", "input": {"camera": "cam_door"}},
        ],
        stop_reason="tool_use",
    )
    chat, captured = _anthropic_chat(monkeypatch, [tool_turn, _message([{"type": "text", "text": "done"}])], effort="medium")
    result = await chat.complete(
        [Message.system("You investigate."), Message.user("look", images=[ImagePart(b"jpg")])], tools=[TOOL]
    )
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls == [ToolCallRequest("toolu_1", "get_device_status", {"camera": "cam_door"})]
    assert result.content == "Checking the camera."
    assert result.usage.cache_read_tokens == 5

    first = captured[0]
    body = first["body"]
    assert first["path"].endswith("/v1/messages")
    # anthropic 1.x sends the key as x-api-key; 2.x as a bearer token.
    headers = first["headers"]
    assert headers.get("x-api-key") == "test-key" or headers.get("authorization") == "Bearer test-key"
    assert "server-side-fallback-2026-07-01" in first["headers"]["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["system"] == "You investigate."
    assert "temperature" not in body and "thinking" not in body
    assert body["tools"] == [{"name": TOOL.name, "description": TOOL.description, "input_schema": TOOL.parameters}]
    image_block, text_block = body["messages"][0]["content"]
    assert image_block["source"] == {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(b"jpg").decode()}
    assert text_block == {"type": "text", "text": "look"}

    # Next turn: the assistant content is replayed verbatim (thinking block and
    # signature included) and parallel tool results share one user message.
    second_call = ToolCallRequest("toolu_2", "get_device_status", {})
    await chat.complete(
        [
            Message.user("look"),
            result.as_message(),
            Message.tool_result(result.tool_calls[0], '{"ok": true}'),
            Message.tool_result(second_call, "boom", is_error=True),
        ],
        tools=[TOOL],
    )
    replay = captured[1]["body"]["messages"]
    assert replay[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert [b["tool_use_id"] for b in replay[2]["content"]] == ["toolu_1", "toolu_2"]
    assert replay[2]["content"][1]["is_error"] is True
    assert len(replay) == 3


async def test_anthropic_refusal_is_reported_not_raised(monkeypatch):
    refusal = _message([], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "x"})
    chat, _ = _anthropic_chat(monkeypatch, [refusal])
    result = await chat.complete([Message.user("hi")])
    assert result.stop_reason == "refusal"
    assert result.content == "" and result.tool_calls == []
    assert "cyber" in result.reasoning


async def test_anthropic_fallback_is_only_sent_to_models_that_accept_it(monkeypatch):
    chat, captured = _anthropic_chat(monkeypatch, [_message([{"type": "text", "text": "ok"}])], model="claude-haiku-5-5")
    await chat.complete([Message.user("hi")])
    assert "fallbacks" not in captured[0]["body"]
    assert "anthropic-beta" not in captured[0]["headers"]


# ---------------------------------------------------------------- registry & config


def test_registry_builds_each_provider():
    assert isinstance(build_chat_model(ModelEndpoint(provider="stub")), StubChat)
    assert isinstance(build_chat_model(ModelEndpoint(provider="ollama", model="m")), OllamaChat)
    compat = build_chat_model(ModelEndpoint(provider="openai_compat", model="m", base_url="http://x"))
    assert isinstance(compat, OpenAICompatChat)
    with pytest.raises(ChatModelError, match="base_url"):
        build_chat_model(ModelEndpoint(provider="openai_compat", model="m"))


def test_roles_must_name_defined_models():
    with pytest.raises(ValueError, match="roles.chat refers to model 'ghost'"):
        WatchwiseConfig(roles=RolesSettings(chat="ghost"))


def test_screener_role_overrides_the_vlm_model_but_keeps_tuning():
    cfg = WatchwiseConfig(
        models={"local": ModelEndpoint(provider="ollama", model="qwen3-vl:2b", base_url="http://o:11434", context_window=4096)},
        roles=RolesSettings(screener="local"),
        vlm=VLMSettings(frames=1, max_image_edge=448),
    )
    screening = cfg.screener_settings()
    assert (screening.backend, screening.model, screening.base_url) == ("ollama", "qwen3-vl:2b", "http://o:11434")
    assert (screening.frames, screening.max_image_edge, screening.context_window) == (1, 448, 4096)


def test_the_screener_cap_comes_from_its_model_unless_set_on_purpose():
    models = {"local": {"provider": "ollama", "model": "qwen3-vl:2b", "max_tokens": 1500}}
    cfg = WatchwiseConfig.model_validate({"models": models, "roles": {"screener": "local"}, "vlm": {"frames": 1}})
    assert cfg.screener_settings().max_tokens == 1500
    assert cfg.screener_max_tokens_key() == "models.local.max_tokens"

    pinned = WatchwiseConfig.model_validate(
        {"models": models, "roles": {"screener": "local"}, "vlm": {"max_tokens": 600}}
    )
    assert pinned.screener_settings().max_tokens == 600
    assert pinned.screener_max_tokens_key() == "vlm.max_tokens"

    legacy = WatchwiseConfig.model_validate({"vlm": {"backend": "ollama", "model": "qwen3-vl:2b"}})
    assert legacy.screener_settings().max_tokens == 1024


def test_old_configs_keep_their_behaviour():
    legacy = WatchwiseConfig.model_validate({"vlm": {"backend": "stub"}, "agent": {"max_iterations": 5}})
    assert legacy.investigator_backend() == "scripted"
    assert legacy.screener_settings() == legacy.vlm


def test_auto_backend_resolution():
    real = WatchwiseConfig(models={"m": ModelEndpoint(provider="ollama", model="x")}, roles=RolesSettings(investigator="m"))
    stub = WatchwiseConfig(models={"m": ModelEndpoint(provider="stub")}, roles=RolesSettings(investigator="m"))
    assert real.investigator_backend() == "tools"
    assert stub.investigator_backend() == "scripted"
    pinned = real.model_copy(update={"agent": AgentSettings(backend="scripted")})
    assert pinned.investigator_backend() == "scripted"


def test_env_interpolation_with_defaults():
    env = {"SET": "http://ollama:11434"}
    data = {"a": "${SET:-http://localhost:11434}", "b": "${UNSET:-fallback}", "c": ["${UNSET}"]}
    assert interpolate_env(data, env) == {"a": "http://ollama:11434", "b": "fallback", "c": [""]}


async def test_wiring_picks_the_tool_loop_for_a_real_investigator(tool_ctx):
    cfg = WatchwiseConfig(
        models={"inv": ModelEndpoint(provider="ollama", model="qwen3-vl:4b")},
        roles=RolesSettings(investigator="inv"),
    )
    backend, status = build_agent_backend(cfg, tool_ctx)
    assert isinstance(backend, ToolLoopBackend) and status == "ready"
    assert backend.model == "qwen3-vl:4b"
    await backend.aclose()

    stub_cfg = WatchwiseConfig(models={"s": ModelEndpoint()}, roles=RolesSettings(investigator="s"))
    assert isinstance(build_agent_backend(stub_cfg, tool_ctx)[0], ScriptedBackend)


# ---------------------------------------------------------------- the tool loop


async def test_tool_loop_investigates_with_real_tools_and_concludes(tool_ctx, store):
    chat = StubChat(script=[call(camera="cam_drive"), conclusion("benign", 0.9, "camera healthy")])
    inv = await investigator_for(chat, tool_ctx, store).investigate(make_alert(), UNCERTAIN)

    assert inv.status == "done"
    assert (inv.verdict, inv.confidence) == (Verdict.BENIGN, 0.9)
    assert [c.name for c in inv.tool_calls] == ["get_device_status"]
    assert any("model usage" in t for t in inv.thoughts)
    # The model saw the real tool output, attributed to its call.
    tool_message = chat.calls[1][0][-1]
    assert tool_message.role == "tool" and tool_message.tool_call_id == "c-get_device_status"
    assert json.loads(tool_message.content)["ok"] is True


async def test_tool_loop_nudges_once_for_a_structured_answer(tool_ctx, store):
    chat = StubChat(script=[ChatResult(content="It looks fine to me."), conclusion("benign", 0.7)])
    inv = await investigator_for(chat, tool_ctx, store).investigate(make_alert(), UNCERTAIN)
    assert inv.verdict is Verdict.BENIGN
    assert chat.calls[1][0][-1].content == NUDGE


async def test_prose_after_the_nudge_fails_safe(tool_ctx, store):
    chat = StubChat(script=[ChatResult(content="fine"), ChatResult(content="still fine")])
    inv = await investigator_for(chat, tool_ctx, store).investigate(make_alert(), UNCERTAIN)
    assert inv.verdict is Verdict.UNCERTAIN and inv.confidence == 0.0


async def test_bad_arguments_and_unknown_tools_are_reported_back_not_run(tool_ctx, store):
    bad = ChatResult(
        content="",
        tool_calls=[
            ToolCallRequest("c1", "get_device_status", {INVALID_ARGUMENTS: "{camera:"}),
            ToolCallRequest("c2", "delete_all_recordings", {}),
        ],
        stop_reason="tool_calls",
    )
    chat = StubChat(script=[bad, conclusion()])
    inv = await investigator_for(chat, tool_ctx, store).investigate(make_alert(), UNCERTAIN)
    assert inv.tool_calls == []  # neither call reached a tool
    results = [m for m in chat.calls[1][0] if m.role == "tool"]
    assert all(m.is_error for m in results)
    assert "not valid JSON" in results[0].content
    assert "unknown tool" in results[1].content and "get_device_status" in results[1].content


async def test_a_refusal_escalates_to_a_human(tool_ctx, store):
    chat = StubChat(script=[ChatResult(content="", stop_reason="refusal")])
    inv = await investigator_for(chat, tool_ctx, store).investigate(make_alert(), UNCERTAIN)
    assert inv.verdict is Verdict.UNCERTAIN and "declined" in inv.conclusion


async def test_a_model_that_never_concludes_is_cut_off(tool_ctx, store):
    chat = StubChat(script=[lambda msgs, tools: call(camera="cam_drive")] * 10)
    inv = await investigator_for(chat, tool_ctx, store, max_iterations=2).investigate(make_alert(), UNCERTAIN)
    assert inv.verdict is Verdict.UNCERTAIN and inv.confidence == 0.0
    assert len(chat.calls) == 3  # two rounds plus one "conclude now" turn
    assert inv.iterations == 2


async def test_a_transport_failure_marks_the_investigation_failed(tool_ctx, store):
    class Down(StubChat):
        async def complete(self, messages, **kwargs):
            raise ChatModelError("connection refused")

    inv = await investigator_for(Down(), tool_ctx, store).investigate(make_alert(), UNCERTAIN)
    assert inv.status == "failed" and inv.verdict is Verdict.UNCERTAIN
    assert "connection refused" in inv.error


async def test_the_pipeline_runs_tier_three_on_the_configured_model(config, nvr_client, monkeypatch):
    chat = StubChat(script=[call(camera="cam_yard"), conclusion("suspicious", 0.88, "unknown person, no roster match")])
    monkeypatch.setattr("watchwise.llm.registry.build_chat_model", lambda endpoint: chat)
    config.models = {"inv": ModelEndpoint(provider="openai_compat", model="any", base_url="http://unused")}
    config.roles = RolesSettings(investigator="inv")
    config.agent = AgentSettings(max_concurrency=1, timeout_s=30.0)

    system = build_system(config, nvr_client=nvr_client, start_sources=False)
    await system.start()
    try:
        alert = make_alert(review_id="rev-t3", camera="cam_yard", labels=[], start_time=NIGHT_TS)
        await system.pipeline.handle(alert)
        await system.investigations.drain(timeout=10)
        final = system.memory_sink.latest("rev-t3")
    finally:
        await system.stop()
    assert final.investigation_id is not None
    assert final.disposition.value == "alerted"
    assert "unknown person" in final.rationale


# ---------------------------------------------------------------- VLM adapter


async def test_any_chat_model_can_screen_frames():
    reply = ChatResult(content='{"verdict": "benign", "confidence": 0.85, "reason": "a cat", "tags": ["animal"]}')
    chat = StubChat("vision", script=[reply])
    vlm = ChatModelVLM(chat)
    frame = Frame(camera="c", timestamp=0.0, jpeg=b"jpg")
    result = await vlm.screen([frame], "prompt", "system")
    assert (result.verdict, result.confidence) == (Verdict.BENIGN, 0.85)
    sent = chat.calls[0][0][1]
    assert sent.images[0].data == b"jpg"

    refused = await ChatModelVLM(StubChat(script=[ChatResult(content="", stop_reason="refusal")])).screen(
        [frame], "prompt", "system"
    )
    assert refused.verdict is Verdict.UNCERTAIN and refused.error
