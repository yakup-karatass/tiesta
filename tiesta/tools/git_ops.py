"""
tools/git_ops.py
────────────────
Git operations tool for Tiesta.

Provides specialized tools for checking git status, viewing diffs (with
context limits to prevent token exhaustion), and committing changes.
Gracefully handles non-git repositories.
"""

import asyncio
import logging
from typing import Any, Dict, List

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class GitOpsTool(BaseTool):
    """Provides tools for interacting with Git repositories."""

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_git_status",
                description=(
                    "Get the current git status, showing modified, untracked, "
                    "and staged files. Returns an empty or graceful message "
                    "if the workspace is not a git repository."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._handle_get_git_status,
            ),
            ToolDefinition(
                name="get_git_diff",
                description=(
                    "Get the current git diff for unstaged changes. Output is "
                    "truncated if it exceeds the context budget to prevent "
                    "overwhelming the LLM."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "staged": {
                            "type": "boolean",
                            "description": "If true, show diff for staged changes instead.",
                            "default": False,
                        }
                    },
                    "additionalProperties": False,
                },
                handler=self._handle_get_git_diff,
            ),
            ToolDefinition(
                name="execute_git_commit",
                description=(
                    "Stage all changes (git add .) and create a new commit "
                    "with the provided message."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The commit message.",
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
                handler=self._handle_execute_git_commit,
            ),
        ]

    async def _run_git(self, *args: str) -> tuple[int, str, str]:
        """Helper to run a git command in the workspace asynchronously."""
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self._workspace_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _handle_get_git_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get git status."""
        code, stdout, stderr = await self._run_git("status", "--short", "--branch")
        
        if code != 0:
            if "not a git repository" in stderr.lower():
                return {
                    "status": "ok",
                    "is_git_repo": False,
                    "output": "Not a git repository.",
                }
            return {"status": "error", "error": f"Git status failed: {stderr.strip()}"}

        return {
            "status": "ok",
            "is_git_repo": True,
            "output": stdout.strip() or "Working tree clean.",
        }

    async def _handle_get_git_diff(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get git diff."""
        staged = args.get("staged", False)
        
        cmd_args = ["diff"]
        if staged:
            cmd_args.append("--staged")
            
        code, stdout, stderr = await self._run_git(*cmd_args)
        
        if code != 0:
            if "not a git repository" in stderr.lower():
                return {
                    "status": "error",
                    "error": "Not a git repository.",
                }
            return {"status": "error", "error": f"Git diff failed: {stderr.strip()}"}

        output = stdout.strip()
        if not output:
            return {
                "status": "ok",
                "output": "No diff found (working tree clean or changes not staged).",
            }

        # Truncate output to prevent context bloat (e.g., max 4000 chars)
        max_length = 4000
        if len(output) > max_length:
            half = max_length // 2
            output = (
                output[:half]
                + f"\n\n... [DIFF TRUNCATED - {len(output)} chars total] ...\n\n"
                + output[-half:]
            )

        return {
            "status": "ok",
            "output": output,
        }

    async def _handle_execute_git_commit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stage all changes and commit."""
        message = args.get("message")
        if not message:
            return {"status": "error", "error": "Commit message is required."}

        # First, add all changes
        code, out, err = await self._run_git("add", ".")
        if code != 0:
            return {"status": "error", "error": f"Failed to stage changes (git add): {err.strip()}"}

        # Check if there's anything to commit
        code, out, err = await self._run_git("status", "--porcelain")
        if not out.strip():
            return {"status": "error", "error": "Nothing to commit (working tree clean)."}

        # Commit
        code, out, err = await self._run_git("commit", "-m", message)
        if code != 0:
            return {"status": "error", "error": f"Failed to commit: {err.strip()}"}

        return {
            "status": "ok",
            "action": "committed",
            "output": out.strip(),
        }
