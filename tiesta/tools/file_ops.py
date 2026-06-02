"""
tools/file_ops.py
─────────────────
High-performance, sandboxed file management tools for the Tiesta agent.

Exposes four tools to the LLM
─────────────────────────────
• **create_file**      — write content to a new or existing file
                         (auto-creates parent directories).
• **read_file**        — read file contents with optional line-range
                         targeting for context-efficient reads.
• **patch_file**       — surgical search-and-replace.
• **undo_last_edit**   — revert the most recent destructive file
                         operation from the automatic backup stack.

Safety
──────
• Every path is routed through ``_resolve_and_guard()`` (sandbox).
• Destructive ops (patch_file, overwrite) automatically snapshot the
  file into ``.tiesta/backups/`` *before* writing.  The backup stack
  is capped at ``max_backups`` (default 10) to prevent bloat.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tiesta.tools.base import BaseTool, SandboxViolationError, ToolDefinition

logger = logging.getLogger(__name__)


# ────────────────────── Backup Manager ────────────────────────────


@dataclass
class BackupRecord:
    """Metadata for a single file backup."""

    original_path: str
    """Absolute path of the original file."""

    backup_path: str
    """Absolute path of the backup copy."""

    timestamp: float
    """Unix timestamp when the backup was created."""

    operation: str
    """Which operation triggered this backup (patch_file, create_file)."""

    relative_path: str
    """Workspace-relative path for display."""


class BackupManager:
    """Manages automatic file backups in ``.tiesta/backups/``.

    • Creates a timestamped copy before every destructive write.
    • Maintains a LIFO stack (most recent first) for undo.
    • Prunes old backups beyond ``max_backups``.
    • Persists the manifest as JSON so state survives process restarts.
    """

    def __init__(self, workspace_root: str, max_backups: int = 10) -> None:
        self._workspace = workspace_root
        self._max_backups = max_backups
        self._backup_dir = Path(workspace_root) / ".tiesta" / "backups"
        self._manifest_path = self._backup_dir / "_manifest.json"
        self._stack: List[BackupRecord] = []

        # Ensure the backup directory exists
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest if present
        self._load_manifest()

    # ── public API ─────────────────────────────────────────────────

    def snapshot(
        self,
        abs_path: str,
        relative_path: str,
        operation: str,
    ) -> bool:
        """Create a backup of ``abs_path`` before it gets modified.

        Returns ``True`` on success, ``False`` if the file doesn't
        exist (nothing to back up — e.g. creating a new file).
        """
        source = Path(abs_path)
        if not source.exists() or not source.is_file():
            return False

        import uuid
        ts = time.time()
        # Build a unique backup filename:
        # e.g.  src__utils.py__1717341234.567__a1b2c3.bak
        safe_name = relative_path.replace("/", "__").replace("\\", "__")
        uid = uuid.uuid4().hex[:6]
        backup_name = f"{safe_name}__{ts:.3f}__{uid}.bak"
        backup_path = self._backup_dir / backup_name

        try:
            shutil.copy2(str(source), str(backup_path))
        except OSError as exc:
            logger.error("Backup failed for %s: %s", abs_path, exc)
            return False

        record = BackupRecord(
            original_path=abs_path,
            backup_path=str(backup_path),
            timestamp=ts,
            operation=operation,
            relative_path=relative_path,
        )
        self._stack.append(record)

        logger.info(
            "Backup created: %s → %s (%s)",
            relative_path,
            backup_name,
            operation,
        )

        # Prune old backups
        self._prune()
        self._save_manifest()
        return True

    def undo(self) -> Optional[BackupRecord]:
        """Restore the most recent backup.

        Pops the top of the stack, copies the backup back to the
        original location, and deletes the backup file.

        Returns the ``BackupRecord`` on success, or ``None`` if the
        stack is empty.
        """
        if not self._stack:
            return None

        record = self._stack.pop()
        backup = Path(record.backup_path)
        original = Path(record.original_path)

        if not backup.exists():
            logger.warning(
                "Backup file missing: %s — cannot undo", record.backup_path
            )
            self._save_manifest()
            return None

        try:
            # Ensure parent dir still exists (in case it was deleted)
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(backup), str(original))
            backup.unlink()
        except OSError as exc:
            logger.error("Undo failed: %s", exc)
            # Push it back so the user can retry
            self._stack.append(record)
            return None

        logger.info("Undo: restored %s", record.relative_path)
        self._save_manifest()
        return record

    def peek(self) -> Optional[BackupRecord]:
        """Return the most recent backup record without removing it."""
        return self._stack[-1] if self._stack else None

    @property
    def stack_size(self) -> int:
        return len(self._stack)

    @property
    def stack(self) -> List[BackupRecord]:
        """Read-only copy of the backup stack (oldest first)."""
        return list(self._stack)

    # ── internal ───────────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove the oldest backups beyond ``max_backups``."""
        while len(self._stack) > self._max_backups:
            old = self._stack.pop(0)
            old_path = Path(old.backup_path)
            if old_path.exists():
                try:
                    old_path.unlink()
                    logger.debug("Pruned old backup: %s", old.backup_path)
                except OSError:
                    pass

    def _save_manifest(self) -> None:
        """Persist the stack to disk as JSON."""
        try:
            data = [asdict(r) for r in self._stack]
            self._manifest_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Failed to save backup manifest: %s", exc)

    def _load_manifest(self) -> None:
        """Load the stack from disk if the manifest exists."""
        if not self._manifest_path.exists():
            return
        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._stack = [BackupRecord(**entry) for entry in data]
            # Clean up entries whose backup files no longer exist
            self._stack = [
                r for r in self._stack if Path(r.backup_path).exists()
            ]
            logger.info(
                "Loaded %d backup records from manifest", len(self._stack)
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load backup manifest: %s", exc)
            self._stack = []
# ────────────────────── FileOps tool ──────────────────────────────


class FileOps(BaseTool):
    """File-operation tools — sandboxed to the workspace."""

    def __init__(
        self,
        workspace_root: str,
        max_backups: int = 10,
    ) -> None:
        super().__init__(workspace_root)
        self.backups = BackupManager(
            workspace_root=self._workspace_root,
            max_backups=max_backups,
        )

    async def _validate_syntax(self, filepath: Path) -> Optional[str]:
        """Check syntax. Returns an error message if invalid, or None if valid/skipped."""
        ext = filepath.suffix.lower()
        if ext == ".py":
            import ast
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                ast.parse(content)
                return None
            except SyntaxError as e:
                return f"Python SyntaxError on line {e.lineno}: {e.msg}\n{e.text}"
            except Exception as e:
                return f"Validation error: {e}"
        elif ext in {".js", ".ts", ".jsx", ".tsx"}:
            import asyncio
            try:
                process = await asyncio.create_subprocess_exec(
                    "node", "--check", str(filepath),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2.0)
                    if process.returncode != 0:
                        err_text = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
                        return f"JS/TS Syntax Error:\n{err_text}"
                    return None
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    return None
            except FileNotFoundError:
                return None  # node not installed, skip validation
        else:
            import asyncio
            cmd = []
            if ext in {".c", ".cpp"}:
                cmd = ["clang", "-fsyntax-only", str(filepath)]
            elif ext == ".dart":
                cmd = ["dart", "analyze", str(filepath)]
            elif ext == ".rs":
                cmd = ["rustc", "--emit", "metadata", str(filepath)]
            
            if cmd:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2.0)
                        if process.returncode != 0:
                            err_text = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
                            return f"Syntax Error ({ext}):\n{err_text}"
                        return None
                    except asyncio.TimeoutError:
                        try:
                            process.kill()
                        except OSError:
                            pass
                        return None
                except FileNotFoundError:
                    return None  # Compiler not installed, skip validation

        return None

    # ── tool definitions (schemas for the LLM) ─────────────────────

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="create_file",
                description=(
                    "Create or overwrite a file with the given content. "
                    "Parent directories are created automatically. "
                    "The path must be relative to the project workspace. "
                    "If overwriting, the previous version is automatically "
                    "backed up and can be restored with undo_last_edit."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Workspace-relative file path "
                                "(e.g. 'src/utils.py')."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "The full content to write.",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": (
                                "If true, overwrite the file if it already "
                                "exists.  Defaults to false for safety."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self._handle_create_file,
            ),
            ToolDefinition(
                name="read_file",
                description=(
                    "Read the contents of a file.  Supports optional "
                    "line-range targeting (start_line / end_line) to "
                    "reduce context usage when only a portion is needed. "
                    "Lines are 1-indexed."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Workspace-relative file path to read."
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "description": (
                                "First line to return (1-indexed, inclusive). "
                                "Omit to start from the beginning."
                            ),
                        },
                        "end_line": {
                            "type": "integer",
                            "description": (
                                "Last line to return (1-indexed, inclusive). "
                                "Omit to read to the end."
                            ),
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self._handle_read_file,
            ),
            ToolDefinition(
                name="patch_file",
                description=(
                    "Apply a precise search-and-replace patch to a file. "
                    "You provide the exact block of text to find (search) "
                    "and its replacement.  This is far more efficient than "
                    "rewriting the entire file — use it for targeted edits. "
                    "The search text must match the file content EXACTLY "
                    "(including whitespace and indentation). "
                    "The file is automatically backed up before patching. "
                    "Use undo_last_edit to revert if the patch is wrong."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Workspace-relative file path to patch."
                            ),
                        },
                        "search": {
                            "type": "string",
                            "description": (
                                "The exact text block to find in the file. "
                                "Must match verbatim."
                            ),
                        },
                        "replace": {
                            "type": "string",
                            "description": (
                                "The replacement text that will take the "
                                "place of the search block."
                            ),
                        },
                        "occurrence": {
                            "type": "integer",
                            "description": (
                                "Which occurrence to replace (1-indexed). "
                                "Use 0 or omit to replace ALL occurrences. "
                                "Defaults to 1 (first match only) for safety."
                            ),
                            "default": 1,
                        },
                    },
                    "required": ["path", "search", "replace"],
                    "additionalProperties": False,
                },
                handler=self._handle_patch_file,
            ),
            ToolDefinition(
                name="undo_last_edit",
                description=(
                    "Revert the most recent file modification (patch_file "
                    "or create_file overwrite) by restoring the automatic "
                    "backup.  Use this when a previous edit corrupted a "
                    "file or introduced bugs.  Can be called multiple "
                    "times to undo further back."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=self._handle_undo_last_edit,
            ),
        ]

    # ── handlers ───────────────────────────────────────────────────

    async def _handle_create_file(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create or overwrite a file inside the workspace."""
        raw_path = args.get("path", "")
        content = args.get("content", "")
        overwrite = args.get("overwrite", False)

        # ── Validate ───────────────────────────────────────────────
        if not raw_path:
            return _error("Parameter 'path' is required and must not be empty.")

        try:
            abs_path = self._resolve_and_guard(raw_path)
        except SandboxViolationError as exc:
            return _error(str(exc))

        target = Path(abs_path)

        if target.exists() and not overwrite:
            return _error(
                f"File '{self._relative_display(abs_path)}' already exists. "
                f"Set overwrite=true to replace it, or use patch_file for "
                f"targeted edits."
            )

        if target.exists() and target.is_dir():
            return _error(
                f"Path '{self._relative_display(abs_path)}' is a directory, "
                f"not a file."
            )

        # ── Backup before overwrite ────────────────────────────────
        is_overwrite = target.exists()
        if is_overwrite:
            self.backups.snapshot(
                abs_path,
                self._relative_display(abs_path),
                operation="create_file(overwrite)",
            )

        # ── Write ──────────────────────────────────────────────────
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return _error(f"OS error writing file: {exc}")

        # ── Autonomous Syntax Validation ───────────────────────────
        syntax_err = await self._validate_syntax(target)
        if syntax_err:
            if is_overwrite:
                self.backups.undo()
            else:
                try:
                    target.unlink()
                except OSError:
                    pass
            return {
                "status": "error",
                "error": "Syntax validation failed. The change has been automatically reverted.",
                "details": syntax_err,
            }

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        byte_count = len(content.encode("utf-8"))
        rel = self._relative_display(abs_path)

        logger.info("create_file → %s (%d lines, %d bytes)", rel, line_count, byte_count)

        result: Dict[str, Any] = {
            "status": "ok",
            "action": "overwritten" if is_overwrite else "created",
            "path": rel,
            "lines": line_count,
            "bytes": byte_count,
        }
        if is_overwrite:
            result["backed_up"] = True
        return result

    async def _handle_read_file(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Read file contents with optional line-range slicing."""
        raw_path = args.get("path", "")
        start_line: Optional[int] = args.get("start_line")
        end_line: Optional[int] = args.get("end_line")

        # ── Validate ───────────────────────────────────────────────
        if not raw_path:
            return _error("Parameter 'path' is required and must not be empty.")

        try:
            abs_path = self._resolve_and_guard(raw_path)
        except SandboxViolationError as exc:
            return _error(str(exc))

        target = Path(abs_path)

        if not target.exists():
            return _error(
                f"File '{self._relative_display(abs_path)}' does not exist."
            )
        if target.is_dir():
            return _error(
                f"Path '{self._relative_display(abs_path)}' is a directory. "
                f"Use a different tool to list directory contents."
            )

        # ── Read ───────────────────────────────────────────────────
        try:
            raw_content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error(f"OS error reading file: {exc}")

        all_lines = raw_content.splitlines(keepends=True)
        total_lines = len(all_lines)

        # ── Line-range targeting ───────────────────────────────────
        effective_start = 1
        effective_end = total_lines

        if start_line is not None:
            effective_start = max(1, int(start_line))
        if end_line is not None:
            effective_end = min(total_lines, int(end_line))

        if effective_start > total_lines:
            return _error(
                f"start_line ({effective_start}) exceeds total lines "
                f"({total_lines}) in '{self._relative_display(abs_path)}'."
            )

        if effective_start > effective_end:
            return _error(
                f"start_line ({effective_start}) is greater than "
                f"end_line ({effective_end})."
            )

        # Slice (convert 1-indexed to 0-indexed)
        selected = all_lines[effective_start - 1 : effective_end]

        # Prepend line numbers for LLM clarity
        numbered_lines: List[str] = []
        for i, line in enumerate(selected, start=effective_start):
            # Strip trailing newline for clean display, then re-add
            numbered_lines.append(f"{i:>6} | {line.rstrip()}")

        content_block = "\n".join(numbered_lines)
        rel = self._relative_display(abs_path)
        is_partial = (effective_start > 1) or (effective_end < total_lines)

        logger.info(
            "read_file → %s [%d–%d of %d lines]",
            rel,
            effective_start,
            effective_end,
            total_lines,
        )

        return {
            "status": "ok",
            "path": rel,
            "total_lines": total_lines,
            "showing_lines": f"{effective_start}-{effective_end}",
            "is_partial": is_partial,
            "content": content_block,
        }

    async def _handle_patch_file(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a surgical search-and-replace patch."""
        raw_path = args.get("path", "")
        search = args.get("search", "")
        replace = args.get("replace", "")
        occurrence = args.get("occurrence", 1)

        # ── Validate ───────────────────────────────────────────────
        if not raw_path:
            return _error("Parameter 'path' is required and must not be empty.")
        if not search:
            return _error(
                "Parameter 'search' is required and must not be empty. "
                "Provide the exact text block to find."
            )

        try:
            abs_path = self._resolve_and_guard(raw_path)
        except SandboxViolationError as exc:
            return _error(str(exc))

        target = Path(abs_path)

        if not target.exists():
            return _error(
                f"File '{self._relative_display(abs_path)}' does not exist."
            )
        if target.is_dir():
            return _error(
                f"Path '{self._relative_display(abs_path)}' is a directory."
            )

        # ── Read current content ───────────────────────────────────
        try:
            original = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _error(f"OS error reading file: {exc}")

        # ── Search ─────────────────────────────────────────────────
        count = original.count(search)
        if count == 0:
            # Provide a helpful hint — show nearby lines
            hint = self._fuzzy_hint(original, search)
            return _error(
                f"Search text not found in '{self._relative_display(abs_path)}'. "
                f"Ensure the search string matches the file content exactly, "
                f"including whitespace and indentation.\n{hint}"
            )

        # ── Backup BEFORE modifying ────────────────────────────────
        self.backups.snapshot(
            abs_path,
            self._relative_display(abs_path),
            operation="patch_file",
        )

        # ── Replace ───────────────────────────────────────────────
        occurrence = int(occurrence) if occurrence is not None else 1

        if occurrence == 0:
            # Replace ALL occurrences
            new_content = original.replace(search, replace)
            replaced_count = count
        elif 1 <= occurrence <= count:
            # Replace the Nth occurrence
            new_content = self._replace_nth(original, search, replace, occurrence)
            replaced_count = 1
        else:
            return _error(
                f"Requested occurrence {occurrence} but only {count} "
                f"occurrence(s) found."
            )

        # ── Write back ─────────────────────────────────────────────
        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return _error(f"OS error writing file: {exc}")

        # ── Autonomous Syntax Validation ───────────────────────────
        syntax_err = await self._validate_syntax(target)
        if syntax_err:
            self.backups.undo()
            return {
                "status": "error",
                "error": "Syntax validation failed. The change has been automatically reverted.",
                "details": syntax_err,
            }

        # ── Build a compact unified diff for LLM context ───────────
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{self._relative_display(abs_path)}",
                tofile=f"b/{self._relative_display(abs_path)}",
                n=3,  # 3 lines of context
            )
        )
        # Cap diff length to avoid bloating the context window
        max_diff_lines = 60
        diff_text = "".join(diff_lines[:max_diff_lines])
        if len(diff_lines) > max_diff_lines:
            diff_text += f"\n... ({len(diff_lines) - max_diff_lines} more diff lines omitted)\n"

        rel = self._relative_display(abs_path)
        logger.info(
            "patch_file → %s (%d replacement(s))",
            rel,
            replaced_count,
        )

        return {
            "status": "ok",
            "path": rel,
            "occurrences_found": count,
            "occurrences_replaced": replaced_count,
            "backed_up": True,
            "diff": diff_text,
        }

    async def _handle_undo_last_edit(
        self, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Revert the most recent file modification."""
        if self.backups.stack_size == 0:
            return _error(
                "No backups available.  There are no recent file "
                "modifications to undo."
            )

        # Show what we're about to undo
        top = self.backups.peek()
        assert top is not None

        record = self.backups.undo()
        if record is None:
            return _error(
                "Undo failed — the backup file may have been deleted."
            )

        logger.info("undo_last_edit → restored %s", record.relative_path)

        return {
            "status": "ok",
            "action": "restored",
            "path": record.relative_path,
            "operation_undone": record.operation,
            "remaining_backups": self.backups.stack_size,
        }

    # ── private helpers ────────────────────────────────────────────

    @staticmethod
    def _replace_nth(text: str, search: str, replace: str, n: int) -> str:
        """Replace only the *n*-th occurrence (1-indexed) of *search*
        in *text*."""
        idx = -1
        for _ in range(n):
            idx = text.find(search, idx + 1)
            if idx == -1:
                return text  # shouldn't happen — caller already checked
        return text[:idx] + replace + text[idx + len(search) :]

    @staticmethod
    def _fuzzy_hint(content: str, search: str) -> str:
        """When an exact match fails, produce a hint showing the closest
        lines in the file to help the model self-correct.

        Uses ``difflib.get_close_matches`` on individual lines for
        speed, then builds a short snippet.
        """
        search_lines = search.strip().splitlines()
        if not search_lines:
            return ""

        content_lines = content.splitlines()
        if not content_lines:
            return ""

        # Find closest matches for the first non-empty search line
        first_search = search_lines[0].strip()
        if not first_search:
            return ""

        stripped_content = [l.strip() for l in content_lines]
        matches = difflib.get_close_matches(
            first_search, stripped_content, n=3, cutoff=0.5
        )

        if not matches:
            return "Hint: No similar lines found. Double-check the file path and content."

        hints: List[str] = ["Hint — similar lines found in the file:"]
        for match in matches:
            # Find the original (with indentation) line number
            for line_no, line in enumerate(content_lines, 1):
                if line.strip() == match:
                    hints.append(f"  Line {line_no}: {line.rstrip()}")
                    break

        return "\n".join(hints)


# ── module-level helper ───────────────────────────────────────────


def _error(message: str) -> Dict[str, Any]:
    """Uniform error response dict."""
    return {"status": "error", "error": message}
