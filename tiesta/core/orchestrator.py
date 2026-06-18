"""
core/orchestrator.py
────────────────────
The agentic execution loop — drives the conversation between the user,
the LLM, and the tool layer.

Key behaviours (per spec §3 Engineering Directives)
────────────────────────────────────────────────────
• **Autonomous error recovery:**  When a tool call fails (bad JSON,
  runtime error, file-not-found, etc.) the orchestrator injects the
  error into the conversation context and asks the model to self-correct
  — up to ``max_correction_attempts`` times before surfacing a failure.
• **Modular decoupling:**  The orchestrator knows nothing about *how*
  tools run.  It dispatches through a ``ToolRegistry`` interface that
  maps tool names → async callables.
• **Context budget:**  Large tool outputs are automatically truncated
  to ``tool_result_budget`` characters before being appended to the
  conversation, preventing context-window exhaustion on small models.
• **Fully async:**  Every I/O-bound operation uses ``await``.

Changelog
─────────
• FIX: PermissionManager now injected via constructor instead of being
  instantiated fresh on every tool call — TUI callback now fires correctly.
• FIX: tool_result_budget reduced from 12_000 → 3_000 to prevent
  context exhaustion on small models (e.g. qwen2.5-coder:3b).
• FIX: run() and followup() now share a single _run_loop() to ensure
  consistent error recovery and AUTO_RECOVERY behaviour in both paths.
• FIX: followup() no longer silently re-initialises _messages when
  called on an empty history — delegates cleanly to run() instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from openai.types.chat import ChatCompletionToolParam

from tiesta.core.llm_client import LLMClient, LLMClientConfig, LLMError, LLMErrorKind, LLMResult
from tiesta.core.permissions import PermissionManager

logger = logging.getLogger(__name__)

# ─────────────────────── Tool abstractions ─────────────────────────

# A tool handler is any async callable: (arguments: dict) → Any
ToolHandler = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]


class ToolRegistry:
    """Maps tool names → (schema, handler).

    The orchestrator uses this to:
    1. Build the ``tools`` array for the LLM request.
    2. Dispatch tool calls to the correct handler.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, _RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a tool with its JSON-Schema parameter spec and
        async handler function."""
        schema: ChatCompletionToolParam = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._tools[name] = _RegisteredTool(
            name=name, schema=schema, handler=handler
        )
        logger.info("Registered tool: %s", name)

    def get_handler(self, name: str) -> Optional[ToolHandler]:
        entry = self._tools.get(name)
        return entry.handler if entry else None

    def get_schemas(self) -> List[ChatCompletionToolParam]:
        return [t.schema for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())


@dataclass
class _RegisteredTool:
    name: str
    schema: ChatCompletionToolParam
    handler: ToolHandler


# ─────────────────── Orchestrator configuration ───────────────────


@dataclass
class OrchestratorConfig:
    """Knobs for the agentic loop — all overridable at construction."""

    max_turns: int = 25
    """Hard ceiling on LLM round-trips per task to prevent runaways."""

    max_correction_attempts: int = 3
    """How many times to retry after a tool-call error before giving up."""

    # FIX: Reduced from 12_000 → 3_000 to prevent context exhaustion
    # on small models such as qwen2.5-coder:3b (~4k context window).
    tool_result_budget: int = 3_000
    """Max characters kept per tool result before truncation (§2.3)."""

    system_prompt: str = (
        "You are Tiesta, an expert autonomous coding assistant running "
        "locally. You have access to tools for file operations, shell "
        "commands, and code intelligence. Think step-by-step and use the "
        "most specific tool available. CRITICAL: Never output raw JSON "
        "tool calls in your conversational response. Always invoke tools "
        "using the proper function-calling API. When responding to the user, "
        "use friendly natural language to summarize your actions concisely."
    )

    inject_tool_errors: bool = True
    """When True, tool execution errors are fed back to the model as a
    ``tool`` role message so it can attempt self-correction."""


# ────────────────── Turn / event data structures ──────────────────


class TurnKind(Enum):
    ASSISTANT_TEXT = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    ERROR = auto()
    CORRECTION = auto()
    AUTO_RECOVERY = auto()


@dataclass
class TurnRecord:
    """An immutable log entry for one orchestrator turn.

    The TUI / caller can inspect these to render progress in real-time.
    """

    kind: TurnKind
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# Callback the TUI can hook into to render turns as they happen.
TurnCallback = Callable[[TurnRecord], Coroutine[Any, Any, None]]


# ────────────────────── Orchestrator class ────────────────────────


class Orchestrator:
    """Drives the LLM ↔ Tool loop.

    Typical usage
    -------------
    ```python
    registry = ToolRegistry()
    registry.register("read_file", "Read a file", {...}, read_file_handler)

    async with LLMClient() as client:
        orch = Orchestrator(client, registry)
        result = await orch.run("Refactor utils.py to use dataclasses")
        print(result.final_text)
    ```
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        config: Optional[OrchestratorConfig] = None,
        on_turn: Optional[TurnCallback] = None,
        # FIX: Accept a shared PermissionManager so TUI callbacks fire correctly.
        # Previously a fresh PermissionManager() was created on every tool call,
        # discarding the callback registered in main.py.
        permission_manager: Optional[PermissionManager] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.cfg = config or OrchestratorConfig()
        self._on_turn = on_turn
        # FIX: Store the injected permission manager; fall back to a plain one
        # only when none is supplied (e.g. in unit tests).
        self._permission_manager: PermissionManager = permission_manager or PermissionManager()

        # The live conversation (mutated during `run` / `followup`)
        self._messages: List[Dict[str, Any]] = []
        self._turn_log: List[TurnRecord] = []

    # ── public entry point ─────────────────────────────────────────

    async def run(self, user_message: str) -> OrchestratorResult:
        """Execute a full agentic task from a fresh conversation.

        Always starts with a clean slate — system prompt + user message.
        Returns an ``OrchestratorResult`` with the final assistant text,
        full turn log, and aggregate stats.
        """
        self._messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._run_loop()

    # ── public: continue an existing conversation ──────────────────

    async def followup(self, user_message: str) -> OrchestratorResult:
        """Send a follow-up message in the current conversation.

        Re-uses the existing ``_messages`` history so the model retains
        context from previous turns.

        If the conversation history is empty (e.g. after a reset), this
        delegates cleanly to ``run()`` so the system prompt is always
        present at the top of the message list.
        """
        # FIX: If history is empty, run() sets up the system prompt correctly.
        # Previously this check existed but the branch was also missing the
        # system prompt in the inline path — now the shared _run_loop handles both.
        if not self._messages:
            return await self.run(user_message)

        self._messages.append({"role": "user", "content": user_message})
        return await self._run_loop()

    # ── shared agentic loop ────────────────────────────────────────

    async def _run_loop(self) -> OrchestratorResult:
        """Single agentic loop shared by both ``run()`` and ``followup()``.

        FIX: Previously run() and followup() each contained a full copy of
        the loop.  This caused subtle divergence — in particular, the
        AUTO_RECOVERY branch was only reachable via run(), not followup().
        A single shared implementation eliminates that class of bug.
        """
        self._turn_log = []
        total_turns = 0
        final_text = ""

        while total_turns < self.cfg.max_turns:
            total_turns += 1
            logger.debug("─── Orchestrator turn %d ───", total_turns)

            # ① Call the LLM
            result = await self.llm.chat(
                messages=self._messages,
                tools=self.tools.get_schemas() or None,
                tool_choice="auto" if self.tools.get_schemas() else None,
            )

            # ② Handle LLM-level errors (connection, timeout, etc.)
            if isinstance(result, LLMError):
                record = TurnRecord(
                    kind=TurnKind.ERROR,
                    error=result.message,
                    latency_ms=0,
                )
                await self._emit(record)

                if result.retryable and total_turns < self.cfg.max_turns:
                    logger.warning(
                        "LLM error (%s) — will retry after back-off",
                        result.kind.name,
                    )
                    await asyncio.sleep(2)
                    continue
                else:
                    final_text = f"[Error] {result.message}"
                    break

            assert isinstance(result, LLMResult)

            # ③ If the model produced text content, record it
            if result.content:
                final_text = result.content
                record = TurnRecord(
                    kind=TurnKind.ASSISTANT_TEXT,
                    content=result.content,
                    latency_ms=result.latency_ms,
                )
                await self._emit(record)

            # ④ If no tool calls, the model is done
            if not result.tool_calls:
                self._messages.append(
                    {"role": "assistant", "content": result.content or ""}
                )
                break

            # ⑤ Process each tool call
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            self._messages.append(assistant_msg)

            for tc in result.tool_calls:
                tool_result_str = await self._execute_tool_call(
                    tc, correction_depth=0
                )
                if tool_result_str != "__HANDLED_BY_CORRECTION__":
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result_str,
                        }
                    )

        return OrchestratorResult(
            final_text=final_text,
            turns=list(self._turn_log),
            total_llm_calls=total_turns,
            messages=list(self._messages),
        )

    # ── internal: tool execution + self-correction ─────────────────

    async def _execute_tool_call(
        self, tc: Dict[str, Any], *, correction_depth: int
    ) -> str:
        """Execute a single tool call with layered error recovery.

        Recovery hierarchy
        ──────────────────
        1. **Parse error** — the model's JSON was malformed.  Feed the
           raw output + error back and ask for a corrected call.
        2. **Unknown tool** — model hallucinated a tool name.  Tell it
           which tools are actually available.
        3. **Handler exception** — the tool itself threw.  Capture the
           traceback and feed it back.
        4. **Max corrections exceeded** — give up and return a failure
           message as the tool result.
        """
        name = tc["name"]
        arguments = tc["arguments"]
        call_id = tc["id"]
        parse_error = tc.get("parse_error", False)

        # ── Case 1: JSON parse failure ─────────────────────────────
        if parse_error:
            error_msg = (
                f"[TOOL CALL ERROR] Failed to parse the arguments for "
                f"tool '{name}'.  The raw output was:\n"
                f"```\n{arguments}\n```\n"
                f"Please provide valid JSON arguments for this tool call."
            )
            logger.warning("Parse error on tool '%s' — depth %d", name, correction_depth)
            return await self._attempt_correction(
                call_id=call_id,
                tool_name=name,
                error_msg=error_msg,
                correction_depth=correction_depth,
            )

        # ── Case 2: Unknown tool ───────────────────────────────────
        if not self.tools.has(name):
            available = ", ".join(self.tools.names) or "(none)"
            error_msg = (
                f"[TOOL CALL ERROR] Tool '{name}' does not exist.  "
                f"Available tools: {available}.  "
                f"Please use one of the available tools."
            )
            logger.warning("Unknown tool '%s' — depth %d", name, correction_depth)
            return await self._attempt_correction(
                call_id=call_id,
                tool_name=name,
                error_msg=error_msg,
                correction_depth=correction_depth,
            )

        # ── Case 3: Permission Control Plane ───────────────────────
        # FIX: Previously called PermissionManager() here, creating a new
        # instance on every tool execution and discarding the TUI callback
        # registered in main.py.  Now uses self._permission_manager which
        # is the shared instance injected at construction time.
        needs_permission = False
        if name in ("execute_bash", "execute_git_commit", "read_serial_monitor"):
            needs_permission = True
        elif name == "patch_file":
            try:
                search_text = arguments.get("search", "")
                replace_text = arguments.get("replace", "")
                if len(search_text) > 500 and len(replace_text) < (len(search_text) * 0.3):
                    needs_permission = True
            except Exception:
                pass

        if needs_permission:
            # FIX: Use the injected shared instance instead of PermissionManager()
            approved = await self._permission_manager.request_permission(name, arguments)
            if not approved:
                error_record = TurnRecord(
                    kind=TurnKind.ERROR,
                    tool_name=name,
                    error="User denied the execution of this command.",
                    latency_ms=0.0,
                )
                await self._emit(error_record)
                return (
                    "User denied the execution of this command. "
                    "The user has reviewed the proposed action and chose not to allow it. "
                    "Please propose an alternative approach or ask the user for guidance."
                )

        # ── Case 4: Execute the handler ────────────────────────────
        handler = self.tools.get_handler(name)
        assert handler is not None

        record = TurnRecord(
            kind=TurnKind.TOOL_CALL,
            tool_name=name,
            tool_args=arguments if isinstance(arguments, dict) else {},
        )
        await self._emit(record)

        t0 = time.perf_counter()
        try:
            raw_result = await handler(arguments)
            elapsed = (time.perf_counter() - t0) * 1000

            result_str = self._stringify_tool_result(raw_result)
            result_str = self._truncate(result_str)

            # ── Autonomous Bash Recovery ───────────────────────────
            if name == "execute_bash" and isinstance(raw_result, dict):
                is_bash_error = (
                    raw_result.get("status") == "error"
                    or raw_result.get("exit_code", 0) != 0
                )
                if is_bash_error:
                    return await self._attempt_bash_recovery(
                        call_id=call_id,
                        tool_name=name,
                        raw_result=raw_result,
                        result_str=result_str,
                        correction_depth=correction_depth,
                    )

            result_record = TurnRecord(
                kind=TurnKind.TOOL_RESULT,
                tool_name=name,
                tool_result=result_str,
                latency_ms=elapsed,
            )
            await self._emit(result_record)
            return result_str

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            error_msg = (
                f"[TOOL EXECUTION ERROR] Tool '{name}' raised an exception:\n"
                f"{type(exc).__name__}: {exc}\n"
                f"Arguments were: {json.dumps(arguments, default=str)}\n"
                f"Please review and correct your approach."
            )
            logger.error(
                "Tool '%s' raised %s: %s (%.0fms)",
                name,
                type(exc).__name__,
                exc,
                elapsed,
            )

            error_record = TurnRecord(
                kind=TurnKind.ERROR,
                tool_name=name,
                error=str(exc),
                latency_ms=elapsed,
            )
            await self._emit(error_record)

            if self.cfg.inject_tool_errors:
                return await self._attempt_correction(
                    call_id=call_id,
                    tool_name=name,
                    error_msg=error_msg,
                    correction_depth=correction_depth,
                )
            return error_msg

    # ── internal: self-correction loop ─────────────────────────────

    async def _attempt_correction(
        self,
        *,
        call_id: str,
        tool_name: str,
        error_msg: str,
        correction_depth: int,
    ) -> str:
        """Feed an error back to the model and ask it to retry.

        If ``correction_depth`` exceeds ``max_correction_attempts``,
        gives up and returns the error as the tool result.
        """
        if correction_depth >= self.cfg.max_correction_attempts:
            logger.error(
                "Max correction attempts (%d) reached for tool '%s'",
                self.cfg.max_correction_attempts,
                tool_name,
            )
            return (
                f"[UNRECOVERABLE ERROR] After {correction_depth} correction "
                f"attempts, tool '{tool_name}' still cannot execute "
                f"successfully.  Last error:\n{error_msg}"
            )

        correction_record = TurnRecord(
            kind=TurnKind.CORRECTION,
            tool_name=tool_name,
            error=error_msg,
        )
        await self._emit(correction_record)

        # Inject the error as a tool-role message so the model sees it
        # as the "result" of its failed call and can self-correct.
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": error_msg,
            }
        )

        # Ask the model to try again
        result = await self.llm.chat(
            messages=self._messages,
            tools=self.tools.get_schemas() or None,
            tool_choice="auto",
        )

        if isinstance(result, LLMError):
            return f"[CORRECTION FAILED — LLM ERROR] {result.message}"

        assert isinstance(result, LLMResult)

        # If the model produced a corrected tool call, execute it
        if result.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            self._messages.append(assistant_msg)

            for tc in result.tool_calls:
                res = await self._execute_tool_call(
                    tc, correction_depth=correction_depth + 1
                )
                if res != "__HANDLED_BY_CORRECTION__":
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": res,
                    })

            return "__HANDLED_BY_CORRECTION__"

        # Model responded with text instead of a corrected call
        if result.content:
            self._messages.append(
                {"role": "assistant", "content": result.content}
            )
            return result.content

        return error_msg

    async def _attempt_bash_recovery(
        self,
        *,
        call_id: str,
        tool_name: str,
        raw_result: Dict[str, Any],
        result_str: str,
        correction_depth: int,
    ) -> str:
        """Autonomously attempt to recover from a failed execute_bash command."""
        max_retries = 3
        if correction_depth >= max_retries:
            msg = (
                f"I have attempted to fix this error {max_retries} times "
                f"autonomously but failed. I need your guidance."
            )
            logger.error("Max bash recovery attempts (%d) reached.", max_retries)
            return msg

        recovery_record = TurnRecord(
            kind=TurnKind.AUTO_RECOVERY,
            tool_name=tool_name,
            error=f"Attempt {correction_depth + 1}/{max_retries}",
        )
        await self._emit(recovery_record)

        injection = (
            f"{result_str}\n\n"
            f"[SYSTEM INTERVENTION]: The command failed. Do not ask the user for help yet. "
            f"Analyze the error trace, identify the bug in the code or environment, use file_ops to apply a fix, "
            f"and then execute the bash command again to verify."
        )

        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": injection,
            }
        )

        result = await self.llm.chat(
            messages=self._messages,
            tools=self.tools.get_schemas() or None,
            tool_choice="auto",
        )

        if isinstance(result, LLMError):
            return f"[BASH RECOVERY FAILED — LLM ERROR] {result.message}"

        assert isinstance(result, LLMResult)

        if result.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            self._messages.append(assistant_msg)

            for tc in result.tool_calls:
                res = await self._execute_tool_call(
                    tc, correction_depth=correction_depth + 1
                )
                if res != "__HANDLED_BY_CORRECTION__":
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": res,
                    })

            return "__HANDLED_BY_CORRECTION__"

        if result.content:
            self._messages.append({"role": "assistant", "content": result.content})
            return result.content

        return result_str

    # ── internal: helpers ──────────────────────────────────────────

    @staticmethod
    def _stringify_tool_result(raw: Any) -> str:
        """Normalise any tool return value to a string."""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (dict, list)):
            try:
                return json.dumps(raw, indent=2, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(raw)
        return str(raw)

    def _truncate(self, text: str) -> str:
        """Budget-aware truncation (§2.3 Tool Result Budgeting)."""
        budget = self.cfg.tool_result_budget
        if len(text) <= budget:
            return text
        half = budget // 2
        truncated_notice = (
            f"\n\n... [TRUNCATED — {len(text):,} chars total, "
            f"showing first {half:,} and last {half:,}] ...\n\n"
        )
        return text[:half] + truncated_notice + text[-half:]

    async def _emit(self, record: TurnRecord) -> None:
        """Send a turn record to the registered callback (if any)."""
        self._turn_log.append(record)
        if self._on_turn:
            try:
                await self._on_turn(record)
            except Exception:
                logger.exception("Turn callback raised — ignoring")

    # ── read-only accessors ────────────────────────────────────────

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Current conversation history (read-only copy)."""
        return list(self._messages)

    @property
    def turn_log(self) -> List[TurnRecord]:
        """Full log of orchestrator turns (read-only copy)."""
        return list(self._turn_log)

    def reset(self) -> None:
        """Clear all conversation history and turn logs.

        After calling this, the next ``followup()`` call delegates to
        ``run()`` — ensuring the system prompt is always re-injected.
        """
        self._messages.clear()
        self._turn_log.clear()
        logger.info("Orchestrator conversation history reset")


# ──────────────────── Result container ─────────────────────────────


@dataclass
class OrchestratorResult:
    """Returned by ``Orchestrator.run()`` — everything the TUI needs."""

    final_text: str
    """The last assistant text message."""

    turns: List[TurnRecord]
    """Chronological log of every turn."""

    total_llm_calls: int
    """Number of LLM round-trips consumed."""

    messages: List[Dict[str, Any]]
    """The full conversation array (for inspection / serialisation)."""