"""
tools/bash_executor.py
──────────────────────
Observable Bash Execution tool for the Tiesta agent.

Gives the LLM the ability to run shell commands — tests, scripts,
dependency installs, git operations — with strict safety boundaries:

• **Hard timeout** on every subprocess (default 30s) via
  ``asyncio.wait_for``.  Prevents the agent from hanging on blocking
  processes (web servers, interactive prompts, infinite loops).
• **Output truncation** — stdout and stderr are each capped to a
  configurable character budget (default 3 000 chars, keeping the
  *tail* so the most recent / relevant output is preserved).  This
  prevents massive logs (``npm install``, ``cargo build``) from
  exhausting the local LLM's context window.
• **Workspace CWD** — every subprocess runs with ``cwd`` set to the
  active workspace, and inherits a sanitised environment.
• **Structured return** — clean dict with ``exit_code``, ``stdout``,
  ``stderr``, ``status``, ``timed_out``, and ``duration_s``.

Security note
─────────────
Command-level permission gating (blocking ``rm -rf /``, ``git reset
--hard``, etc.) belongs in the **Permission Control Plane** layer that
sits *above* tools — it intercepts tool calls in the orchestrator
before they reach this handler.  This module intentionally does NOT
second-guess which commands are "safe"; it focuses on reliable,
observable execution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)

# ────────────────────── Configuration ─────────────────────────────


@dataclass
class BashExecutorConfig:
    """All knobs for shell execution — overridable at construction."""

    default_timeout_s: float = 30.0
    """Per-command hard timeout in seconds."""

    max_timeout_s: float = 300.0
    """Ceiling the LLM can request via the ``timeout`` parameter."""

    stdout_budget: int = 3_000
    """Max characters kept from stdout (tail-truncated)."""

    stderr_budget: int = 3_000
    """Max characters kept from stderr (tail-truncated)."""

    env_overrides: Dict[str, str] | None = None
    """Extra environment variables merged into the subprocess env.
    Useful for injecting ``PYTHONDONTWRITEBYTECODE=1`` etc."""

    shell_executable: Optional[str] = None
    """Explicit shell binary.  ``None`` → OS default
    (``cmd`` on Windows, ``/bin/sh`` on Unix)."""


# ────────────────────── Tool class ────────────────────────────────


class BashExecutor(BaseTool):
    """Observable shell execution tool — sandboxed to the workspace."""

    def __init__(
        self,
        workspace_root: str,
        config: Optional[BashExecutorConfig] = None,
    ) -> None:
        super().__init__(workspace_root)
        self.cfg = config or BashExecutorConfig()

    # ── tool definition ────────────────────────────────────────────

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="execute_bash",
                description=(
                    "Execute a shell command in the project workspace "
                    "directory.  Returns the exit code, stdout, and stderr. "
                    "Use this for running tests, installing dependencies, "
                    "git operations, builds, and any other CLI tasks. "
                    "Output is automatically truncated to avoid context "
                    "overflow.  A timeout is enforced to prevent hangs — "
                    "do NOT start long-running servers with this tool."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The shell command to execute "
                                "(e.g. 'python -m pytest tests/ -v')."
                            ),
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                f"Optional timeout in seconds.  "
                                f"Defaults to {self.cfg.default_timeout_s}s, "
                                f"max {self.cfg.max_timeout_s}s."
                            ),
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=self._handle_execute_bash,
            ),
        ]

    # ── handler ────────────────────────────────────────────────────

    async def _handle_execute_bash(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a shell command with timeout and output truncation."""
        command = args.get("command", "")
        timeout_raw = args.get("timeout")

        # ── Validate ───────────────────────────────────────────────
        if not command or not command.strip():
            return _error("Parameter 'command' is required and must not be empty.")

        # Resolve timeout
        timeout_s = self.cfg.default_timeout_s
        if timeout_raw is not None:
            try:
                timeout_s = float(timeout_raw)
            except (TypeError, ValueError):
                timeout_s = self.cfg.default_timeout_s
        timeout_s = max(1.0, min(timeout_s, self.cfg.max_timeout_s))

        # ── Build environment ──────────────────────────────────────
        env = os.environ.copy()
        # Prevent interactive pagers from blocking the subprocess
        env["PAGER"] = "cat"
        env["GIT_PAGER"] = "cat"
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Merge user overrides
        if self.cfg.env_overrides:
            env.update(self.cfg.env_overrides)

        logger.info(
            "execute_bash → %.1fs timeout | %s",
            timeout_s,
            command[:120] + ("…" if len(command) > 120 else ""),
        )

        # ── Execute ────────────────────────────────────────────────
        t0 = time.perf_counter()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace_root,
                env=env,
            )

            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                timed_out = True
                # Kill the entire process tree — critical on Windows
                # where proc.kill() only kills the top-level shell.
                await self._kill_process_tree(proc)
                # Drain whatever was buffered before the kill
                try:
                    raw_stdout, raw_stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=3.0
                    )
                except (asyncio.TimeoutError, Exception):
                    raw_stdout = b""
                    raw_stderr = b""

            elapsed = time.perf_counter() - t0
            exit_code = proc.returncode if proc.returncode is not None else -1

        except OSError as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Failed to spawn subprocess: %s", exc)
            return _error(
                f"Failed to start command: {type(exc).__name__}: {exc}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Unexpected error during execution: %s", exc)
            return _error(
                f"Unexpected execution error: {type(exc).__name__}: {exc}"
            )

        # ── Decode & truncate ──────────────────────────────────────
        stdout_str = _safe_decode(raw_stdout)
        stderr_str = _safe_decode(raw_stderr)

        stdout_truncated, stdout_was_truncated = _tail_truncate(
            stdout_str, self.cfg.stdout_budget
        )
        stderr_truncated, stderr_was_truncated = _tail_truncate(
            stderr_str, self.cfg.stderr_budget
        )

        # ── Determine status ───────────────────────────────────────
        if timed_out:
            status = "timeout"
        elif exit_code == 0:
            status = "success"
        else:
            status = "error"

        logger.info(
            "execute_bash ← %s (exit=%d, %.2fs, stdout=%d→%d, stderr=%d→%d)",
            status,
            exit_code,
            elapsed,
            len(stdout_str),
            len(stdout_truncated),
            len(stderr_str),
            len(stderr_truncated),
        )

        result: Dict[str, Any] = {
            "status": status,
            "exit_code": exit_code,
            "stdout": stdout_truncated,
            "stderr": stderr_truncated,
            "timed_out": timed_out,
            "duration_s": round(elapsed, 2),
        }

        # Only add truncation notices if they actually happened — keeps
        # the result compact for the LLM.
        if stdout_was_truncated:
            result["stdout_truncated_from"] = len(stdout_str)
        if stderr_was_truncated:
            result["stderr_truncated_from"] = len(stderr_str)

        return result

    # ── internal: process-tree kill ─────────────────────────────────

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
        """Kill a subprocess and all its children.

        On Windows, ``proc.kill()`` only terminates the top-level
        ``cmd.exe`` shell — child processes (ping, npm, cargo, etc.)
        keep running.  We use ``taskkill /F /T /PID`` to nuke the
        entire tree.

        On Unix we send ``SIGTERM`` to the process group, wait briefly,
        then ``SIGKILL`` if it's still alive.
        """
        pid = proc.pid
        if pid is None:
            return

        try:
            if sys.platform == "win32":
                # /F = force  /T = tree (children)
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            else:
                # Try graceful SIGTERM to the process group first
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                # Give it a moment, then force-kill
                await asyncio.sleep(0.5)
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except Exception as exc:
            logger.debug("Process tree kill failed (pid=%s): %s", pid, exc)
            # Last resort — plain kill
            try:
                proc.kill()
            except ProcessLookupError:
                pass


# ── module-level helpers ──────────────────────────────────────────


def _safe_decode(raw: bytes) -> str:
    """Decode bytes with fallback — never crash on encoding issues."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _tail_truncate(text: str, budget: int) -> tuple[str, bool]:
    """Keep the **tail** of ``text`` (most recent output is usually
    the most relevant for error diagnosis).

    Returns ``(truncated_text, was_truncated)``.
    """
    if len(text) <= budget:
        return text, False

    notice = (
        f"[… truncated — showing last {budget:,} of "
        f"{len(text):,} chars …]\n"
    )
    # Reserve space for the notice itself
    usable = budget - len(notice)
    if usable < 100:
        # Budget is tiny — just hard-cut
        return text[-budget:], True

    return notice + text[-usable:], True


def _error(message: str) -> Dict[str, Any]:
    """Uniform error response dict."""
    return {"status": "error", "error": message}
