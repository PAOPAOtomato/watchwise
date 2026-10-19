"""Investigation backend built on the Claude Agent SDK.

The SDK supplies the ReAct harness -- the think -> tool_use -> tool_result
loop, context management, and the MCP client -- so this module only has to:

  * hand it the forensic tools as an in-process MCP server
  * constrain it to those tools only (no filesystem, no bash, no web)
  * mirror its message stream into our own trace and short-term memory, so an
    investigation is auditable after the fact
  * parse the final JSON conclusion

Note the tool handlers here are the *same* ``ForensicTool`` objects the
scripted backend and the stdio MCP server use, so a conclusion reached with a
model is reproducible against the same evidence.

Requires credentials for Claude. With none configured the service should be
left on the scripted backend; ``ensure_available`` reports that clearly rather
than failing at the first alert.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sentinel.mcp.sdk_server import SERVER_NAME, allowed_tool_names, build_sdk_mcp_server
from sentinel.memory.short_term import ShortTermMemory
from sentinel.tier3 import prompts
from sentinel.tier3.backends.base import (
    AgentBackend,
    Conclusion,
    ToolRunner,
    parse_conclusion,
)
from sentinel.tools.base import ToolContext
from sentinel.types import AlertContext, ScreenResult, Verdict

logger = logging.getLogger(__name__)


def credentials_present() -> bool:
    """True if the SDK is likely to be able to authenticate.

    The SDK resolves credentials in order: ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, then a logged-in CLI profile. We can only observe
    the first two directly; the profile case is reported as "maybe" and left
    to fail loudly on the first call rather than being guessed at here.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClaudeSDKBackend(AgentBackend):
    name = "claude_sdk"

    def __init__(
        self,
        tool_ctx: ToolContext,
        model: str = "claude-opus-5-5",
        *,
        max_iterations: int = 6,
        effort: str = "medium",
        max_budget_usd: float | None = 0.50,
    ):
        super().__init__(model)
        self.tool_ctx = tool_ctx
        self.max_iterations = max_iterations
        self.effort = effort
        self.max_budget_usd = max_budget_usd

    def _options(self) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=prompts.SYSTEM,
            mcp_servers={SERVER_NAME: build_sdk_mcp_server(self.tool_ctx)},
            # Only the forensic tools. An investigation agent has no business
            # touching the filesystem or the network directly.
            allowed_tools=allowed_tool_names(),
            tools=[],
            permission_mode="bypassPermissions",
            max_turns=self.max_iterations,
            max_budget_usd=self.max_budget_usd,
            effort=self.effort,  # type: ignore[arg-type]
            thinking={"type": "adaptive"},
            setting_sources=None,
        )

    async def run(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        memory: ShortTermMemory,
        runner: ToolRunner,
    ) -> Conclusion:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        task = prompts.build_task_prompt(ctx, screen, memory.briefing())
        final_text = ""

        try:
            async with ClaudeSDKClient(options=self._options()) as client:
                await client.query(task)
                async for message in client.receive_response():
                    final_text = self._absorb(
                        message,
                        runner,
                        memory,
                        final_text,
                        blocks=(
                            AssistantMessage,
                            ResultMessage,
                            TextBlock,
                            ThinkingBlock,
                            ToolUseBlock,
                            ToolResultBlock,
                        ),
                    )
        except Exception as exc:
            logger.exception("Claude Agent SDK investigation failed")
            return Conclusion(
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                conclusion=f"Investigation could not complete: {type(exc).__name__}: {exc}",
                recommended_action="Human review needed; the investigation backend errored.",
                evidence=[],
            )

        return parse_conclusion(final_text)

    def _absorb(
        self,
        message: Any,
        runner: ToolRunner,
        memory: ShortTermMemory,
        final_text: str,
        *,
        blocks: tuple[type, ...],
    ) -> str:
        """Mirror one SDK message into our trace. Returns the latest text."""
        (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
        ) = blocks

        if isinstance(message, ResultMessage):
            runner.investigation.cost_usd = float(
                getattr(message, "total_cost_usd", 0.0) or 0.0
            )
            return final_text

        if not isinstance(message, AssistantMessage):
            return final_text

        # One assistant turn == one ReAct iteration.
        runner.next_iteration()

        for block in getattr(message, "content", []) or []:
            if isinstance(block, ThinkingBlock):
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    runner.record_thought(text[:1500])
            elif isinstance(block, TextBlock):
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    final_text = text
                    runner.record_thought(text[:1500])
            elif isinstance(block, ToolUseBlock):
                # The SDK executes the tool itself against our in-process MCP
                # server; we record the call so the trace is complete.
                name = str(getattr(block, "name", "")).replace(
                    f"mcp__{SERVER_NAME}__", ""
                )
                args = dict(getattr(block, "input", {}) or {})
                from sentinel.types import ToolCall

                runner.investigation.tool_calls.append(
                    ToolCall(
                        iteration=runner.iteration,
                        name=name,
                        arguments=args,
                        result_summary="(executed by the Agent SDK via in-process MCP)",
                    )
                )
                memory.add_thought(f"calling {name} with {args}", runner.iteration)
            elif isinstance(block, ToolResultBlock):
                content = getattr(block, "content", None)
                memory.add_observation("tool_result", str(content)[:1200], runner.iteration)

        return final_text

    @staticmethod
    def ensure_available() -> tuple[bool, str]:
        """Report whether this backend can actually run, and why not."""
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            return False, "claude-agent-sdk is not installed"
        if not credentials_present():
            return (
                False,
                "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment; "
                "the SDK may still authenticate via a logged-in CLI profile, but "
                "that cannot be verified without making a call",
            )
        return True, "ready"
