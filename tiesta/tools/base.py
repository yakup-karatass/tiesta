"""
tools/base.py
─────────────
Abstract base class for all Tiesta tools.

Every tool in the system inherits from ``BaseTool`` and must implement:
1. A **JSON Schema** describing its parameters (for the LLM).
2. An async **execute** method that performs the actual work.

The base class provides a uniform ``register_all()`` bridge that
plugs an arbitrary number of tools into the orchestrator's
``ToolRegistry`` in a single call — keeping the wiring code minimal
and consistent.

Design rationale
────────────────
• Tools are grouped by *capability domain* (file ops, shell, LSP, …).
  A single ``BaseTool`` subclass can expose **multiple** named tool
  functions (e.g. ``FileOps`` registers ``create_file``, ``read_file``,
  ``patch_file`` as separate tools the LLM can call).
• The base class enforces **directory sandboxing** at the interface
  level: every subclass receives the ``workspace_root`` and uses the
  provided ``_resolve_and_guard()`` helper to validate paths before
  touching the filesystem.
• All public methods are async (§3 Engineering Directives).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tiesta.core.orchestrator import ToolHandler, ToolRegistry

logger = logging.getLogger(__name__)


# ────────────────────── Sandbox violation ──────────────────────────


class SandboxViolationError(Exception):
    """Raised when a tool attempts to operate outside the workspace."""

    def __init__(self, requested_path: str, workspace_root: str) -> None:
        self.requested_path = requested_path
        self.workspace_root = workspace_root
        super().__init__(
            f"SANDBOX VIOLATION: Path '{requested_path}' resolves outside "
            f"the allowed workspace '{workspace_root}'. Operation blocked."
        )


# ────────────────────── Tool definition ───────────────────────────


@dataclass(frozen=True)
class ToolDefinition:
    """Describes a single named tool that a ``BaseTool`` exposes.

    Attributes
    ----------
    name : str
        The function name the LLM will call (e.g. ``"read_file"``).
    description : str
        Natural-language description injected into the system prompt.
    parameters : dict
        JSON-Schema object describing the expected arguments.
    handler : ToolHandler
        The async callable ``(args: dict) → Any``.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler


# ────────────────────── Abstract base ─────────────────────────────


class BaseTool(ABC):
    """Abstract base for all Tiesta tools.

    Subclasses must implement ``definitions()`` which returns the list
    of ``ToolDefinition`` objects the tool exposes.

    Usage
    -----
    ```python
    file_ops = FileOps(workspace_root="/home/user/project")
    file_ops.register(registry)
    ```
    """

    def __init__(self, workspace_root: str) -> None:
        # Resolve once at construction so every guard check is
        # comparing fully-resolved absolute paths.
        self._workspace_root = str(Path(workspace_root).resolve())
        logger.info(
            "%s initialised — workspace: %s",
            type(self).__name__,
            self._workspace_root,
        )

    @property
    def workspace_root(self) -> str:
        return self._workspace_root

    # ── subclass contract ──────────────────────────────────────────

    @abstractmethod
    def definitions(self) -> List[ToolDefinition]:
        """Return all tool definitions this class exposes."""
        ...

    # ── registration bridge ────────────────────────────────────────

    def register(self, registry: ToolRegistry) -> None:
        """Bulk-register every definition into the orchestrator's
        ``ToolRegistry``."""
        for defn in self.definitions():
            registry.register(
                name=defn.name,
                description=defn.description,
                parameters=defn.parameters,
                handler=defn.handler,
            )

    # ── sandboxing primitives ──────────────────────────────────────

    def _resolve_and_guard(self, raw_path: str) -> str:
        """Resolve *raw_path* relative to the workspace and ensure it
        stays inside.

        Guards against:
        • ``../`` traversal
        • Absolute paths outside the workspace
        • Symlinks that escape (resolved via ``Path.resolve()``)
        • Writes to Tiesta's own core source code

        Returns the fully-resolved absolute path as a string.

        Raises
        ------
        SandboxViolationError
            If the resolved path is outside ``workspace_root`` or targets Tiesta's core files.
        """
        import os
        
        # If raw_path is relative, anchor it to the workspace
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path(self._workspace_root) / candidate

        resolved = str(candidate.resolve())

        # Normalise separators for reliable prefix comparison on Windows
        resolved_norm = resolved.replace("\\", "/").rstrip("/")
        root_norm = self._workspace_root.replace("\\", "/").rstrip("/")

        # 1. Ensure path is within the workspace
        if not (
            resolved_norm == root_norm
            or resolved_norm.startswith(root_norm + "/")
        ):
            raise SandboxViolationError(raw_path, self._workspace_root)

        # 2. Ensure path does not target Tiesta's own source code
        tiesta_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkg_norm = tiesta_pkg_dir.replace("\\", "/").rstrip("/")
        if resolved_norm == pkg_norm or resolved_norm.startswith(pkg_norm + "/"):
            raise SandboxViolationError(
                raw_path,
                f"{self._workspace_root} (ERROR: Attempted to overwrite Tiesta core files at {pkg_norm})"
            )

        return resolved

    def _relative_display(self, absolute_path: str) -> str:
        """Return a human-readable workspace-relative path for tool
        output (keeps LLM context clean)."""
        try:
            return str(
                Path(absolute_path).relative_to(self._workspace_root)
            )
        except ValueError:
            return absolute_path
