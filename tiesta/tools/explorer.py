"""
tools/explorer.py
─────────────────
Exploration tools for navigating the workspace efficiently.

Provides a compressed tree-view of the directory structure while
automatically ignoring noisy/bloated directories like .git, node_modules,
or virtual environments. This ensures the local LLM can understand the
project architecture without exhausting its context window.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from tiesta.tools.base import BaseTool, SandboxViolationError, ToolDefinition


class ExplorerTool(BaseTool):
    """Provides workspace exploration and architecture mapping tools."""

    IGNORE_DIRS = {
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".env",
        ".git",
        "__pycache__",
        "build",
        "dist",
        ".tiesta",
        ".pytest_cache",
        ".mypy_cache",
        ".idea",
        ".vscode",
    }

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="list_directory_tree",
                description=(
                    "Get a compressed, tree-like string representation of the "
                    "project structure. Automatically ignores noisy directories "
                    "like node_modules, .git, and venvs. Use this to quickly "
                    "understand the codebase architecture."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Workspace-relative directory path to scan. "
                                "Use '.' or omit to scan the workspace root."
                            ),
                            "default": ".",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth to recurse. Defaults to 4.",
                            "default": 4,
                        },
                    },
                    "additionalProperties": False,
                },
                handler=self._handle_list_directory_tree,
            ),
        ]

    async def _handle_list_directory_tree(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scan and return a tree representation of a directory."""
        raw_path = args.get("path", ".")
        max_depth = int(args.get("max_depth", 4))

        try:
            abs_path = self._resolve_and_guard(raw_path)
        except SandboxViolationError as exc:
            return {"status": "error", "error": str(exc)}

        target = Path(abs_path)

        if not target.exists():
            return {
                "status": "error",
                "error": f"Directory '{self._relative_display(abs_path)}' does not exist.",
            }

        if not target.is_dir():
            return {
                "status": "error",
                "error": f"'{self._relative_display(abs_path)}' is a file, not a directory.",
            }

        tree_lines = []
        self._generate_tree(target, "", 0, max_depth, tree_lines)

        tree_str = "\n".join(tree_lines)
        if not tree_str:
            tree_str = "(empty directory)"

        return {
            "status": "ok",
            "path": self._relative_display(abs_path),
            "tree": tree_str,
        }

    def _generate_tree(
        self,
        current_path: Path,
        prefix: str,
        current_depth: int,
        max_depth: int,
        lines: List[str],
    ) -> None:
        if current_depth > max_depth:
            return

        try:
            # Sort directories first, then files, both alphabetically
            entries = sorted(
                list(current_path.iterdir()),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}└── [Permission Denied]")
            return
        except OSError:
            return

        # Filter out ignored directories
        valid_entries = []
        for e in entries:
            if e.is_dir() and e.name in self.IGNORE_DIRS:
                continue
            if e.name.startswith(".DS_Store"):
                continue
            valid_entries.append(e)

        for i, entry in enumerate(valid_entries):
            is_last = i == len(valid_entries) - 1
            connector = "└── " if is_last else "├── "
            
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")

            if entry.is_dir() and current_depth < max_depth:
                extension = "    " if is_last else "│   "
                self._generate_tree(
                    entry,
                    prefix + extension,
                    current_depth + 1,
                    max_depth,
                    lines,
                )
