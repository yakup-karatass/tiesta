"""
tiesta/tools/lsp_tools.py
─────────────────────────
LSP Tools exposed to the LLM for semantic code navigation using Jedi.
"""

from typing import Any, Dict, List

from tiesta.core.lsp_client import LSPClient
from tiesta.tools.base import BaseTool, ToolDefinition


class LSPTools(BaseTool):
    """Provides semantic code navigation via Jedi."""

    def __init__(self, workspace_root: str) -> None:
        super().__init__(workspace_root)
        self.client = LSPClient(self.workspace_root)

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="goto_definition",
                description=(
                    "Jump to the exact file, line, and column where a class, function, "
                    "or variable is defined. Returns the target file destination."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to the file containing the symbol usage.",
                        },
                        "line": {
                            "type": "integer",
                            "description": "The 1-indexed line number of the symbol usage.",
                        },
                        "column": {
                            "type": "integer",
                            "description": "The 0-indexed column number of the symbol usage.",
                        },
                    },
                    "required": ["path", "line", "column"],
                    "additionalProperties": False,
                },
                handler=self._handle_goto_definition,
            ),
            ToolDefinition(
                name="find_usages",
                description=(
                    "Find all references and usages of a class, function, or variable "
                    "across the entire workspace."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to the file containing the symbol.",
                        },
                        "line": {
                            "type": "integer",
                            "description": "The 1-indexed line number of the symbol.",
                        },
                        "column": {
                            "type": "integer",
                            "description": "The 0-indexed column number of the symbol.",
                        },
                    },
                    "required": ["path", "line", "column"],
                    "additionalProperties": False,
                },
                handler=self._handle_find_usages,
            ),
        ]

    # ── handlers ───────────────────────────────────────────────────

    async def _handle_goto_definition(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goto definition request."""
        path = args.get("path", "")
        line = args.get("line")
        column = args.get("column")

        if not path or line is None or column is None:
            return {"error": "Missing required parameters: path, line, column."}

        try:
            resolved_path = self._resolve_and_guard(path)
        except Exception as exc:
            return {"error": str(exc)}

        result = self.client.get_definition(resolved_path, int(line), int(column))
        
        # If it returns a file, display it relative to workspace for cleaner LLM context
        if "file" in result:
            result["file"] = self._relative_display(result["file"])

        return result

    async def _handle_find_usages(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find usages request."""
        path = args.get("path", "")
        line = args.get("line")
        column = args.get("column")

        if not path or line is None or column is None:
            return {"error": "Missing required parameters: path, line, column."}

        try:
            resolved_path = self._resolve_and_guard(path)
        except Exception as exc:
            return {"error": str(exc)}

        result = self.client.get_references(resolved_path, int(line), int(column))
        
        # If it returns a list of references, display relative paths
        if "references" in result:
            for ref in result["references"]:
                if "file" in ref:
                    ref["file"] = self._relative_display(ref["file"])

        return result
