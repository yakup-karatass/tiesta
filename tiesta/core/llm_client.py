"""
core/llm_client.py
──────────────────
Async LLM client that talks to a **local** Ollama instance via the
OpenAI-compatible `/v1` endpoint.  Zero external API calls — all traffic
stays on localhost.

Design decisions
────────────────
• Uses the official `openai` Python SDK (AsyncOpenAI) so the transport
  layer is battle-tested and the orchestrator can swap between Ollama,
  llama.cpp, LM Studio, or a remote API by just changing the base URL.
• Every public method is `async` (§3 Engineering Directives — asyncio).
• Retry / back-off / timeout are configurable at construction time.
• Structured error types so the orchestrator can decide how to react
  (e.g. feed an error back to the model for self-correction).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

logger = logging.getLogger(__name__)

# ───────────────────────── Error taxonomy ──────────────────────────


class LLMErrorKind(Enum):
    """Categorical bucket so the orchestrator can branch on error *type*
    rather than parsing exception strings."""

    CONNECTION = auto()       # Ollama not running / unreachable
    TIMEOUT = auto()          # Generation took too long
    RATE_LIMIT = auto()       # Shouldn't happen locally, but handled
    BAD_REQUEST = auto()      # Malformed request (prompt too long, etc.)
    MALFORMED_RESPONSE = auto()  # Model returned unparseable output
    SERVER_ERROR = auto()     # 5xx from the inference server
    UNKNOWN = auto()


@dataclass(frozen=True)
class LLMError:
    """Structured error object returned instead of raising.

    The orchestrator inspects `.kind` and `.retryable` to decide whether
    to retry, feed the error back, or surface it to the user.
    """

    kind: LLMErrorKind
    message: str
    retryable: bool
    raw_exception: Optional[BaseException] = field(default=None, repr=False)

    def for_context_injection(self) -> str:
        """Human-readable summary safe to inject back into the LLM
        conversation so the model can attempt self-correction."""
        return (
            f"[SYSTEM ERROR — {self.kind.name}] {self.message}\n"
            "Please review and correct your previous response."
        )


# ──────────────────────── Result wrapper ───────────────────────────


@dataclass
class LLMResult:
    """Unified result container for both streaming and non-streaming calls.

    Attributes
    ----------
    content : str | None
        The assistant's text reply (may be ``None`` if the model only
        produced tool calls).
    tool_calls : list[dict]
        Parsed tool-call dicts ready for the orchestrator's dispatcher.
        Each dict has keys ``id``, ``name``, ``arguments`` (already a
        Python dict, not a raw JSON string).
    finish_reason : str | None
        ``"stop"`` / ``"tool_calls"`` / ``"length"`` etc.
    usage : dict
        Token usage stats (prompt / completion / total).
    model : str
        Which model actually served the request.
    latency_ms : float
        Wall-clock milliseconds for the full round-trip.
    """

    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0


# ──────────────────────── Client config ────────────────────────────


@dataclass
class LLMClientConfig:
    """All knobs in one place — no magic globals."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"                       # Ollama ignores this but SDK requires it
    default_model: str = "qwen2.5-coder:7b"
    temperature: float = 0.1                      # Low temp ⟶ more deterministic tool calls
    max_tokens: int = 4096
    request_timeout_s: float = 120.0              # Per-request hard timeout
    max_retries: int = 3                          # SDK-level retries for transient errors
    backoff_base_s: float = 1.0                   # Exponential back-off base
    backoff_max_s: float = 16.0                   # Cap on back-off sleep
    stream: bool = False                          # Default streaming preference


# ──────────────────────── Client class ─────────────────────────────


class LLMClient:
    """Async wrapper around a local Ollama / OpenAI-compat server.

    Usage
    -----
    ```python
    client = LLMClient()              # default config → localhost Ollama
    result = await client.chat([
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user",   "content": "Write a hello world in Rust."},
    ])
    if isinstance(result, LLMError):
        print(result.for_context_injection())
    else:
        print(result.content)
    ```
    """

    def __init__(self, config: Optional[LLMClientConfig] = None) -> None:
        self.cfg = config or LLMClientConfig()
        self._client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.request_timeout_s,
            max_retries=0,  # We handle retries ourselves for finer control
        )
        logger.info(
            "LLMClient initialised — base_url=%s  model=%s",
            self.cfg.base_url,
            self.cfg.default_model,
        )

    # ── public: single-shot completion ─────────────────────────────

    async def chat(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Optional[Sequence[ChatCompletionToolParam]] = None,
        tool_choice: Union[Literal["auto", "none", "required"], None] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> Union[LLMResult, LLMError]:
        """Send a chat completion request with full retry + error wrapping.

        Returns either an ``LLMResult`` on success or an ``LLMError``
        on failure — **never raises** into the orchestrator.
        """
        effective_stream = stream if stream is not None else self.cfg.stream
        kwargs = self._build_kwargs(
            messages, tools, tool_choice, model, temperature, max_tokens
        )

        if effective_stream:
            return await self._chat_streaming(kwargs)
        return await self._chat_non_streaming(kwargs)

    # ── public: streaming iterator ─────────────────────────────────

    async def chat_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        tools: Optional[Sequence[ChatCompletionToolParam]] = None,
        tool_choice: Union[Literal["auto", "none", "required"], None] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Union[str, LLMResult, LLMError]]:
        """Yield content deltas as ``str``, then a final ``LLMResult``.

        On error yields a single ``LLMError`` and stops.
        """
        kwargs = self._build_kwargs(
            messages, tools, tool_choice, model, temperature, max_tokens
        )
        kwargs["stream"] = True

        attempt = 0
        while attempt <= self.cfg.max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                stream = await self._client.chat.completions.create(**kwargs)

                collected_content: list[str] = []
                collected_tool_calls: dict[int, dict[str, Any]] = {}
                finish_reason: Optional[str] = None
                model_used = ""

                async for chunk in stream:  # type: ChatCompletionChunk
                    model_used = chunk.model or model_used
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason or finish_reason

                    # ── text content ───────────────────────────────
                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

                    # ── tool-call deltas ───────────────────────────
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in collected_tool_calls:
                                collected_tool_calls[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments_raw": "",
                                }
                            entry = collected_tool_calls[idx]
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    entry["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    entry["arguments_raw"] += (
                                        tc_delta.function.arguments
                                    )

                elapsed = (time.perf_counter() - t0) * 1000
                parsed_tc = self._parse_tool_calls_safe(collected_tool_calls)

                res = LLMResult(
                    content="".join(collected_content) or None,
                    tool_calls=parsed_tc,
                    finish_reason=finish_reason,
                    usage={},  # streaming doesn't return usage by default
                    model=model_used,
                    latency_ms=elapsed,
                )
                self._apply_fuzzy_tool_parsing(res)
                yield res
                return  # success — stop retrying

            except Exception as exc:
                err = self._classify_exception(exc)
                if err.retryable and attempt <= self.cfg.max_retries:
                    sleep = min(
                        self.cfg.backoff_base_s * (2 ** (attempt - 1)),
                        self.cfg.backoff_max_s,
                    )
                    logger.warning(
                        "Stream attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt,
                        self.cfg.max_retries,
                        err.kind.name,
                        sleep,
                    )
                    await asyncio.sleep(sleep)
                    continue
                yield err
                return

    # ── public: health check ───────────────────────────────────────

    async def ping(self) -> Union[bool, LLMError]:
        """Quick connectivity check against the Ollama server."""
        try:
            await self._client.models.list()
            return True
        except Exception as exc:
            return self._classify_exception(exc)

    # ── public: list available models ──────────────────────────────

    async def list_models(self) -> Union[List[str], LLMError]:
        """Return model IDs available on the local Ollama instance."""
        try:
            resp = await self._client.models.list()
            return [m.id for m in resp.data]
        except Exception as exc:
            return self._classify_exception(exc)

    # ── internal: build kwargs dict ────────────────────────────────

    def _build_kwargs(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Optional[Sequence[ChatCompletionToolParam]],
        tool_choice: Union[str, None],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model or self.cfg.default_model,
            "messages": list(messages),
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        return kwargs

    # ── internal: non-streaming completion ─────────────────────────

    async def _chat_non_streaming(
        self, kwargs: Dict[str, Any]
    ) -> Union[LLMResult, LLMError]:
        kwargs["stream"] = False
        attempt = 0
        last_error: Optional[LLMError] = None

        while attempt <= self.cfg.max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                resp: ChatCompletion = (
                    await self._client.chat.completions.create(**kwargs)
                )
                elapsed = (time.perf_counter() - t0) * 1000

                choice = resp.choices[0]
                raw_tc = (
                    {
                        i: {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments_raw": tc.function.arguments,
                        }
                        for i, tc in enumerate(choice.message.tool_calls)
                    }
                    if choice.message.tool_calls
                    else {}
                )
                parsed_tc = self._parse_tool_calls_safe(raw_tc)

                usage_info = {}
                if resp.usage:
                    usage_info = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }

                res = LLMResult(
                    content=choice.message.content,
                    tool_calls=parsed_tc,
                    finish_reason=choice.finish_reason,
                    usage=usage_info,
                    model=resp.model,
                    latency_ms=elapsed,
                )
                self._apply_fuzzy_tool_parsing(res)
                return res

            except Exception as exc:
                last_error = self._classify_exception(exc)
                if last_error.retryable and attempt <= self.cfg.max_retries:
                    sleep = min(
                        self.cfg.backoff_base_s * (2 ** (attempt - 1)),
                        self.cfg.backoff_max_s,
                    )
                    logger.warning(
                        "Attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt,
                        self.cfg.max_retries,
                        last_error.kind.name,
                        sleep,
                    )
                    await asyncio.sleep(sleep)
                else:
                    return last_error

        # Should be unreachable, but satisfy type-checker
        assert last_error is not None
        return last_error

    # ── internal: streaming completion (collected) ─────────────────

    async def _chat_streaming(
        self, kwargs: Dict[str, Any]
    ) -> Union[LLMResult, LLMError]:
        """Streaming call that collects the full response into an
        ``LLMResult`` (used when the caller wants streaming transport
        but doesn't need per-delta callbacks)."""
        final: Union[LLMResult, LLMError, None] = None
        async for item in self.chat_stream(
            messages=kwargs.pop("messages"),
            tools=kwargs.pop("tools", None),
            tool_choice=kwargs.pop("tool_choice", None),
            model=kwargs.pop("model", None),
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
        ):
            if isinstance(item, (LLMResult, LLMError)):
                final = item
        if final is None:
            return LLMError(
                kind=LLMErrorKind.UNKNOWN,
                message="Streaming completed without producing a result.",
                retryable=False,
            )
        return final

    def _apply_fuzzy_tool_parsing(self, result: LLMResult) -> None:
        """Fuzzy interceptor: if content is a raw JSON tool call, parse it into tool_calls."""
        if not result.content or result.tool_calls:
            return

        content = result.content
        # Fast check
        if '"name"' not in content or '"arguments"' not in content:
            return

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            return
            
        json_str = match.group(0)

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                # Construct internal tool_calls schema
                tool_call = {
                    "id": "fuzzy_call_0",
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                    "parse_error": not isinstance(parsed["arguments"], dict),
                }
                result.tool_calls.append(tool_call)
                # Clear content since it was just a tool call
                result.content = None
                logger.info("Fuzzy tool parser recovered a hallucinated tool call: %s", parsed["name"])
        except json.JSONDecodeError:
            pass

    # ── internal: parse tool calls (the critical robustness layer) ─

    def _parse_tool_calls_safe(
        self, raw: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse tool-call argument JSON with aggressive recovery.

        Local models (especially smaller quantisations) frequently
        produce:
        • Trailing commas in JSON objects
        • Single-quoted strings
        • Unquoted keys
        • Truncated JSON (ran out of tokens)
        • Extra text surrounding the JSON

        We attempt multiple repair strategies before giving up.
        """
        parsed: List[Dict[str, Any]] = []
        for idx in sorted(raw.keys()):
            entry = raw[idx]
            name = entry.get("name", "unknown")
            raw_args = entry.get("arguments_raw", "")
            arguments = self._try_parse_json(raw_args, tool_name=name)
            parsed.append(
                {
                    "id": entry.get("id", f"call_{idx}"),
                    "name": name,
                    "arguments": arguments,  # dict on success, str on failure
                    "parse_error": not isinstance(arguments, dict),
                }
            )
        return parsed

    def _try_parse_json(
        self, raw: str, *, tool_name: str = ""
    ) -> Union[Dict[str, Any], str]:
        """Multi-strategy JSON parser for unreliable local-model output.

        Returns a ``dict`` on success or the *original raw string* on
        total failure (so the orchestrator can feed it back for
        correction).
        """
        raw = raw.strip()
        if not raw:
            return {}

        # Strategy 1: direct parse — works for well-behaved models
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract the first { … } block (model may have
        # wrapped the JSON in markdown fences or added commentary)
        try:
            start = raw.index("{")
            depth, end = 0, start
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            candidate = raw[start : end + 1]
            result = json.loads(candidate)
            if isinstance(result, dict):
                logger.debug(
                    "Recovered JSON for tool '%s' via brace extraction", tool_name
                )
                return result
        except (ValueError, json.JSONDecodeError):
            pass

        # Strategy 3: fixup common local-model mistakes
        try:
            fixed = raw
            # Replace single quotes with double quotes (crude but effective)
            fixed = fixed.replace("'", '"')
            # Remove trailing commas before } or ]
            import re

            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            result = json.loads(fixed)
            if isinstance(result, dict):
                logger.debug(
                    "Recovered JSON for tool '%s' via fixup heuristics", tool_name
                )
                return result
        except (json.JSONDecodeError, Exception):
            pass

        # Strategy 4: try to find JSON within markdown code fences
        try:
            import re

            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if fence_match:
                result = json.loads(fence_match.group(1))
                if isinstance(result, dict):
                    logger.debug(
                        "Recovered JSON for tool '%s' from code fence", tool_name
                    )
                    return result
        except (json.JSONDecodeError, Exception):
            pass

        # All strategies exhausted — return the raw string so the
        # orchestrator can feed it back for self-correction.
        logger.warning(
            "Failed to parse tool-call arguments for '%s': %.200s",
            tool_name,
            raw,
        )
        return raw

    # ── internal: exception → structured error ─────────────────────

    @staticmethod
    def _classify_exception(exc: BaseException) -> LLMError:
        """Map SDK / network exceptions to our ``LLMError`` taxonomy."""
        if isinstance(exc, APIConnectionError):
            return LLMError(
                kind=LLMErrorKind.CONNECTION,
                message=(
                    "Cannot reach the Ollama server. "
                    "Is `ollama serve` running on localhost:11434?"
                ),
                retryable=True,
                raw_exception=exc,
            )
        if isinstance(exc, APITimeoutError):
            return LLMError(
                kind=LLMErrorKind.TIMEOUT,
                message=f"Request timed out: {exc}",
                retryable=True,
                raw_exception=exc,
            )
        if isinstance(exc, RateLimitError):
            return LLMError(
                kind=LLMErrorKind.RATE_LIMIT,
                message=f"Rate limited (unexpected for local server): {exc}",
                retryable=True,
                raw_exception=exc,
            )
        if isinstance(exc, BadRequestError):
            return LLMError(
                kind=LLMErrorKind.BAD_REQUEST,
                message=f"Bad request — possibly prompt too long or invalid tool schema: {exc}",
                retryable=False,
                raw_exception=exc,
            )
        if isinstance(exc, APIStatusError) and exc.status_code >= 500:
            return LLMError(
                kind=LLMErrorKind.SERVER_ERROR,
                message=f"Server error from inference backend: {exc}",
                retryable=True,
                raw_exception=exc,
            )
        # Catch-all
        return LLMError(
            kind=LLMErrorKind.UNKNOWN,
            message=f"Unexpected error: {type(exc).__name__}: {exc}",
            retryable=False,
            raw_exception=exc,
        )

    # ── lifecycle ──────────────────────────────────────────────────

    async def close(self) -> None:
        """Cleanly shut down the underlying HTTP transport."""
        await self._client.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
