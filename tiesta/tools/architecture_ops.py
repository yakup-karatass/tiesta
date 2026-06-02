"""
tiesta/tools/architecture_ops.py
────────────────────────────────
Tools for autonomously generating visual Mermaid.js architectural maps.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from tiesta.tools.base import BaseTool, ToolDefinition


class ArchitectureOpsTool(BaseTool):
    """Generates Mermaid graphs of the workspace architecture."""

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
        "tiesta.egg-info",
    }

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="generate_architecture_map",
                description=(
                    "Scan the workspace and generate a Mermaid.js graph of the project "
                    "architecture and core file relationships. Saves to TIESTA_ARCHITECTURE.md."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._handle_generate_architecture_map,
            ),
        ]

    async def _handle_generate_architecture_map(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Scan workspace and write Mermaid graph to TIESTA_ARCHITECTURE.md."""
        root = Path(self.workspace_root)
        
        lines = ["```mermaid", "graph TD"]
        
        # We need unique node IDs. We'll use a simple counter mapped to paths.
        path_to_id: Dict[Path, str] = {}
        node_counter = 0

        def get_id(p: Path) -> str:
            nonlocal node_counter
            if p not in path_to_id:
                path_to_id[p] = f"N{node_counter}"
                node_counter += 1
            return path_to_id[p]

        # Register root
        root_id = get_id(root)
        lines.append(f'    {root_id}["{root.name} (Workspace)"]')

        # Recursively build graph
        self._build_graph(root, lines, get_id, max_depth=5, current_depth=0)

        lines.append("```")
        
        mermaid_content = "\n".join(lines)
        output_file = root / "TIESTA_ARCHITECTURE.md"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# Project Architecture\n\n")
                f.write(mermaid_content)
                f.write("\n")
            
            return {
                "status": "ok",
                "message": f"Successfully mapped architecture to {self._relative_display(str(output_file))}",
            }
        except Exception as exc:
            return {"status": "error", "error": f"Failed to write map: {exc}"}

    def _build_graph(self, current_dir: Path, lines: List[str], get_id, max_depth: int, current_depth: int) -> None:
        if current_depth > max_depth:
            return

        try:
            # Sort: dirs first, then files
            entries = sorted(
                list(current_dir.iterdir()),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except OSError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name not in {".tiesta"}:
                continue
            if entry.name in self.IGNORE_DIRS:
                continue
            if entry.name.endswith(".pyc") or entry.name.endswith(".md") and entry.name == "TIESTA_ARCHITECTURE.md":
                continue

            entry_id = get_id(entry)
            parent_id = get_id(current_dir)
            
            # Add node definition with label
            # e.g., N1["src"]
            lines.append(f'    {entry_id}["{entry.name}"]')
            # Add edge
            # e.g., N0 --> N1
            lines.append(f"    {parent_id} --> {entry_id}")

            if entry.is_dir():
                self._build_graph(entry, lines, get_id, max_depth, current_depth + 1)
