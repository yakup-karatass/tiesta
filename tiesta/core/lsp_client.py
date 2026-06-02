"""
tiesta/core/lsp_client.py
─────────────────────────
Lightweight Language Server Protocol (LSP) capabilities for Tiesta using Jedi.

This class provides static analysis, definitions, and references for Python code,
giving the LLM semantic code navigation without the overhead of a full daemon.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import jedi
except ImportError:
    jedi = None  # Handled gracefully if not installed

logger = logging.getLogger(__name__)


class LSPClient:
    """Wrapper around Jedi to provide LSP-like features for Python files."""

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if jedi is None:
            logger.warning("Jedi is not installed. LSP features will be disabled.")

    def _is_python_file(self, file_path: str) -> bool:
        return str(file_path).endswith(".py")

    def get_definition(self, file_path: str, line: int, column: int) -> Dict[str, Any]:
        """Find the definition of the symbol at the given line and column.
        
        Args:
            file_path: Absolute path to the file.
            line: 1-indexed line number.
            column: 0-indexed column number.
            
        Returns:
            Dictionary with destination file, line, and column, or an error.
        """
        if jedi is None:
            return {"error": "LSP engine (jedi) is not installed."}

        if not self._is_python_file(file_path):
            return {"error": "LSP currently only supports Python (.py) files."}

        try:
            script = jedi.Script(path=file_path)
            # Jedi uses 1-indexed lines and 0-indexed columns
            definitions = script.goto(line=line, column=column, follow_imports=True)
            
            if not definitions:
                return {"error": f"No definition found at line {line}, column {column}."}

            # Usually we want the first destination
            target = definitions[0]
            
            if target.module_path is None:
                return {"error": "Definition is a built-in or has no source file."}

            return {
                "file": str(target.module_path),
                "line": target.line,
                "column": target.column,
                "name": target.name,
                "description": target.description,
            }
        except Exception as exc:
            logger.error("Jedi goto failed: %s", exc)
            return {"error": f"Failed to get definition: {exc}"}

    def get_references(self, file_path: str, line: int, column: int) -> Dict[str, Any]:
        """Find all usages/references of the symbol at the given line and column.
        
        Args:
            file_path: Absolute path to the file.
            line: 1-indexed line number.
            column: 0-indexed column number.
            
        Returns:
            Dictionary containing a list of reference locations, or an error.
        """
        if jedi is None:
            return {"error": "LSP engine (jedi) is not installed."}

        if not self._is_python_file(file_path):
            return {"error": "LSP currently only supports Python (.py) files."}

        try:
            script = jedi.Script(path=file_path)
            references = script.get_references(line=line, column=column)
            
            if not references:
                return {"error": f"No references found for symbol at line {line}, column {column}."}

            results = []
            for ref in references:
                if ref.module_path:
                    results.append({
                        "file": str(ref.module_path),
                        "line": ref.line,
                        "column": ref.column,
                        "name": ref.name,
                        "description": getattr(ref, "description", ""),
                    })

            return {"references": results}
        except Exception as exc:
            logger.error("Jedi get_references failed: %s", exc)
            return {"error": f"Failed to get references: {exc}"}
