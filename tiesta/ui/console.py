"""
ui/console.py
─────────────
Rich-powered Terminal User Interface for the Tiesta agent.

Responsibilities
────────────────
1. **Render orchestrator turns** in real-time via the ``on_turn``
   callback — assistant text as syntax-highlighted Markdown, tool calls
   in visually distinct panels, errors in red.
2. **Permission Control Plane** — intercept destructive tool calls
   (``execute_bash``) and prompt the user for approval before execution.
   Denied commands inject a structured error back into the LLM context.
3. **Interactive chat loop** — pretty prompt, streaming output, graceful
   Ctrl-C handling.

Uses the ``rich`` library for all rendering.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from tiesta.core.orchestrator import TurnKind, TurnRecord, ToolRegistry

logger = logging.getLogger(__name__)

# ────────────────────── Theme ─────────────────────────────────────

TIESTA_THEME = Theme(
    {
        "tiesta.brand": "bold bright_cyan",
        "tiesta.prompt": "bold bright_green",
        "tiesta.tool_name": "bold bright_yellow",
        "tiesta.tool_result": "dim white",
        "tiesta.error": "bold red",
        "tiesta.warning": "bold yellow",
        "tiesta.info": "dim cyan",
        "tiesta.success": "bold green",
        "tiesta.muted": "dim white",
        "tiesta.permission": "bold bright_magenta",
    }
)


# ────────────────────── TUI Renderer ──────────────────────────────


class TiestaConsole:
    """Rich-powered console renderer and user interaction manager.

    Usage
    -----
    ```python
    tui = TiestaConsole()
    orchestrator = Orchestrator(llm, tools, on_turn=tui.on_turn)
    await tui.chat_loop(orchestrator)
    ```
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console(theme=TIESTA_THEME, highlight=False)
        self._last_kind: Optional[TurnKind] = None

    # ── orchestrator callback ──────────────────────────────────────

    async def on_turn(self, record: TurnRecord) -> None:
        """Async callback wired into ``Orchestrator(on_turn=...)``.

        Renders each turn event in real-time as the agent works.
        """
        if record.kind == TurnKind.ASSISTANT_TEXT:
            self._render_assistant_text(record.content)

        elif record.kind == TurnKind.TOOL_CALL:
            self._render_tool_call(record)

        elif record.kind == TurnKind.TOOL_RESULT:
            self._render_tool_result(record)

        elif record.kind == TurnKind.ERROR:
            self._render_error(record)

        elif record.kind == TurnKind.CORRECTION:
            self._render_correction(record)

        elif record.kind == TurnKind.AUTO_RECOVERY:
            self._render_auto_recovery(record)

        self._last_kind = record.kind

    # ── rendering helpers ──────────────────────────────────────────

    def _render_assistant_text(self, content: str) -> None:
        """Render the assistant's text response as highlighted Markdown."""
        if not content:
            return
        self.console.print()
        md = Markdown(content)
        self.console.print(md)

    def _render_tool_call(self, record: TurnRecord) -> None:
        """Render a tool invocation in a distinct visual panel."""
        name = record.tool_name or "unknown"
        args = record.tool_args or {}

        # Build a compact args display
        args_display = self._format_args(args, name)

        header = Text()
        header.append("⚡ ", style="bright_yellow")
        header.append(name, style="tiesta.tool_name")

        panel = Panel(
            args_display,
            title=header,
            title_align="left",
            border_style="bright_yellow",
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)

    def _render_tool_result(self, record: TurnRecord) -> None:
        """Render a tool's return value compactly."""
        name = record.tool_name or ""
        result = record.tool_result or ""
        latency = record.latency_ms

        # Try to parse as JSON for pretty display
        status = ""
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                status = parsed.get("status", "")
        except (json.JSONDecodeError, TypeError):
            pass

        # Status indicator
        if status == "ok" or status == "success":
            icon = "✓"
            style = "tiesta.success"
        elif status == "error" or status == "timeout":
            icon = "✗"
            style = "tiesta.error"
        else:
            icon = "→"
            style = "tiesta.muted"

        header = Text()
        header.append(f"  {icon} ", style=style)
        header.append(name, style="tiesta.muted")
        if latency > 0:
            header.append(f"  ({latency:.0f}ms)", style="dim")
        self.console.print(header)

        # Show result content — truncate for display
        display_result = result
        if len(display_result) > 1500:
            display_result = display_result[:750] + "\n  … [truncated for display] …\n" + display_result[-750:]

        if display_result.strip():
            # Use a dimmed, indented block
            for line in display_result.splitlines()[:30]:
                self.console.print(f"  │ {line}", style="tiesta.tool_result")
            total_lines = display_result.count("\n") + 1
            if total_lines > 30:
                self.console.print(
                    f"  │ … ({total_lines - 30} more lines)",
                    style="dim",
                )

    def _render_error(self, record: TurnRecord) -> None:
        """Render an error prominently."""
        error = record.error or "Unknown error"
        tool = record.tool_name or ""
        label = f"Error in {tool}" if tool else "Error"

        panel = Panel(
            Text(error, style="red"),
            title=f"✗ {label}",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)

    def _render_correction(self, record: TurnRecord) -> None:
        """Render a self-correction attempt."""
        tool = record.tool_name or ""
        self.console.print()
        self.console.print(
            f"  ↻ Self-correcting [tiesta.tool_name]{tool}[/]…",
            style="tiesta.warning",
        )

    def _render_auto_recovery(self, record: TurnRecord) -> None:
        """Render an autonomous bash recovery attempt."""
        attempt = record.error or ""
        self.console.print()
        self.console.print(
            f"  [Auto-Recovery {attempt}] Analyzing terminal error...",
            style="tiesta.warning",
        )

    def _format_args(self, args: Dict[str, Any], tool_name: str) -> Any:
        """Format tool arguments for panel display.

        Special-cases common tools for better readability.
        """
        if tool_name == "execute_bash":
            cmd = args.get("command", "")
            timeout = args.get("timeout")
            content = Syntax(cmd, "bash", theme="monokai", line_numbers=False)
            if timeout:
                # Can't easily append to Syntax, so we wrap
                group = Text()
                group.append(cmd, style="bright_white")
                if timeout:
                    group.append(f"\n⏱  timeout: {timeout}s", style="dim")
                return group
            return content

        if tool_name in ("create_file", "read_file", "patch_file"):
            path = args.get("path", "")
            lines: List[str] = [f"📄 {path}"]

            if tool_name == "read_file":
                s = args.get("start_line")
                e = args.get("end_line")
                if s or e:
                    lines.append(f"   lines {s or '1'}–{e or 'EOF'}")

            if tool_name == "patch_file":
                search = args.get("search", "")
                replace = args.get("replace", "")
                occ = args.get("occurrence", 1)
                lines.append(f"   search ({len(search)} chars) → replace ({len(replace)} chars)")
                if occ != 1:
                    lines.append(f"   occurrence: {occ}")

            if tool_name == "create_file":
                content = args.get("content", "")
                overwrite = args.get("overwrite", False)
                lines.append(f"   {len(content)} chars, overwrite={overwrite}")

            return Text("\n".join(lines))

        # Generic: compact JSON
        try:
            formatted = json.dumps(args, indent=2, default=str, ensure_ascii=False)
            if len(formatted) > 500:
                formatted = formatted[:500] + "\n…"
            return Syntax(formatted, "json", theme="monokai", line_numbers=False)
        except (TypeError, ValueError):
            return Text(str(args))

    # ── branding & chrome ──────────────────────────────────────────

    def print_banner(self) -> None:
        """Print the startup banner."""
        banner = Text()
        banner.append("\n  ╔══════════════════════════════════════╗\n", style="bright_cyan")
        banner.append("  ║", style="bright_cyan")
        banner.append("          ⚡ T I E S T A ⚡          ", style="bold bright_white on dark_blue")
        banner.append("║\n", style="bright_cyan")
        banner.append("  ║", style="bright_cyan")
        banner.append("    Local AI Coding Assistant v0.1   ", style="dim white")
        banner.append("║\n", style="bright_cyan")
        banner.append("  ╚══════════════════════════════════════╝\n", style="bright_cyan")
        self.console.print(banner)

    def print_status(self, model: str, workspace: str) -> None:
        """Print connection status info."""
        self.console.print(f"  Model     : [bold]{model}[/]", style="tiesta.info")
        self.console.print(f"  Workspace : [bold]{workspace}[/]", style="tiesta.info")
        self.console.print(f"  Type [bold green]/help[/] for commands, [bold green]Ctrl+C[/] to interrupt.\n", style="tiesta.muted")
        self.console.print(Rule(style="dim"))

    def print_separator(self) -> None:
        self.console.print()
        self.console.print(Rule(style="dim"))

    def print_goodbye(self) -> None:
        self.console.print("\n  👋 Goodbye!\n", style="tiesta.brand")

    def print_thinking(self) -> None:
        self.console.print("\n  ◐ Thinking…", style="dim bright_cyan")

    def print_warning(self, msg: str) -> None:
        self.console.print(f"\n  ⚠ {msg}", style="tiesta.warning")

    def print_info(self, msg: str) -> None:
        self.console.print(f"  {msg}", style="tiesta.info")

    def print_connection_error(self, error_msg: str) -> None:
        panel = Panel(
            Text(
                f"{error_msg}\n\n"
                "Make sure Ollama is running:\n"
                "  1. Install Ollama from https://ollama.com\n"
                "  2. Run: ollama serve\n"
                "  3. Pull a model: ollama pull qwen2.5-coder:7b\n"
                "  4. Restart Tiesta",
                style="red",
            ),
            title="✗ Connection Failed",
            title_align="left",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_help(self) -> None:
        """Print a styled help panel with slash commands and tips."""
        help_text = Text()

        help_text.append("  Slash Commands\n", style="bold bright_cyan underline")
        help_text.append("  ─────────────────────────────────────\n", style="dim")

        commands = [
            ("/help", "Show this help panel"),
            ("/clear, /reset", "Clear screen & reset conversation memory"),
            ("/undo", "Revert the most recent file modification"),
            ("/exit, /quit", "Exit Tiesta gracefully"),
        ]
        for cmd, desc in commands:
            help_text.append(f"  {cmd:<18}", style="bold bright_green")
            help_text.append(f"{desc}\n", style="white")

        help_text.append("\n")
        help_text.append("  Usage Tips\n", style="bold bright_cyan underline")
        help_text.append("  ─────────────────────────────────────\n", style="dim")

        tips = [
            ("Natural chat", "Just type your request — Tiesta will use tools automatically."),
            ("One-shot mode", 'Run  tiesta "your prompt"  from any terminal.'),
            ("Permissions", "Bash commands require [Y/n/a] approval. Press 'a' to auto-approve all."),
            ("Ctrl+C", "Interrupt the current LLM generation."),
        ]
        for label, tip in tips:
            help_text.append(f"  {label:<18}", style="bold bright_yellow")
            help_text.append(f"{tip}\n", style="white")

        help_text.append("\n")
        help_text.append("  Available Tools\n", style="bold bright_cyan underline")
        help_text.append("  ─────────────────────────────────────\n", style="dim")

        tools = [
            ("create_file", "Create/overwrite files (auto-creates directories)"),
            ("read_file", "Read files with optional line-range targeting"),
            ("patch_file", "Surgical search-and-replace edits"),
            ("undo_last_edit", "Revert last file modification from backup"),
            ("execute_bash", "Run shell commands with timeout & truncation"),
            ("list_directory_tree", "View workspace tree (auto-ignores bloat)"),
            ("get_git_status", "Check git status (tracked/untracked/staged)"),
            ("get_git_diff", "View git diff with automatic length truncation"),
            ("execute_git_commit", "Stage all changes and commit"),
            ("search_codebase", "Semantic code search using local vector DB"),
            ("goto_definition", "Jump to the exact definition of a symbol"),
            ("find_usages", "Find all references to a symbol across workspace"),
            ("generate_architecture_map", "Generate visual Mermaid graph of codebase"),
            ("list_serial_ports", "List connected hardware devices (IoT/Robotics)"),
            ("read_serial_monitor", "Read physical serial port streaming logs"),
        ]
        for name, desc in tools:
            help_text.append(f"  {name:<20}", style="bold bright_magenta")
            help_text.append(f"{desc}\n", style="white")

        panel = Panel(
            help_text,
            title="📖 Tiesta Help",
            title_align="left",
            border_style="bright_cyan",
            padding=(1, 1),
        )
        self.console.print()
        self.console.print(panel)

    def print_reset_confirmation(self) -> None:
        """Print confirmation after a conversation reset."""
        self.console.print(
            "\n  ✓ Conversation history cleared. Starting fresh.\n",
            style="tiesta.success",
        )

    def print_undo_success(
        self, path: str, operation: str, remaining: int
    ) -> None:
        """Print confirmation after a successful undo."""
        self.console.print()
        self.console.print(
            f"  ✓ Restored [bold]{path}[/] "
            f"(undid [italic]{operation}[/])",
            style="tiesta.success",
        )
        if remaining > 0:
            self.console.print(
                f"    {remaining} more backup(s) available — "
                f"type /undo again to keep reverting.",
                style="tiesta.muted",
            )
        else:
            self.console.print(
                "    No more backups remaining.",
                style="tiesta.muted",
            )

    async def prompt_permission(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, bool]:
        """Display a dangerous action and prompt for approval.
        Returns: (approved, auto_approve_all)
        """
        self.console.print()

        import json
        if tool_name == "execute_bash":
            command = args.get("command", "<empty>")
            detail = Syntax(command, "bash", theme="monokai", line_numbers=False)
        elif tool_name == "patch_file":
            detail = Text(f"File: {args.get('path', 'unknown')}\n\nSearch:\n{args.get('search', '')}\n\nReplace:\n{args.get('replace', '')}")
        else:
            detail = Text(json.dumps(args, indent=2, default=str))

        panel = Panel(
            detail,
            title="[bold red]⚠️ DANGEROUS ACTION[/]",
            title_align="center",
            subtitle=f"Tool: {tool_name}",
            subtitle_align="right",
            border_style="red",
            padding=(1, 1),
        )
        self.console.print(panel)

        self.console.print(
            "  [tiesta.permission]Allow this action?[/]  "
            "[bold green]Y[/]es / [bold red]n[/]o / [bold cyan]a[/]llow all  ",
            end="",
        )

        import asyncio
        loop = asyncio.get_event_loop()
        try:
            def _read() -> str:
                try: return input()
                except EOFError: return "n"
            response = await loop.run_in_executor(None, _read)
        except (EOFError, KeyboardInterrupt):
            self.console.print("[red]denied[/]")
            return False, False

        choice = response.strip().lower()
        if choice in ("a", "allow", "all", "always"):
            self.console.print("  ✓ Auto-approving all dangerous actions this session.", style="tiesta.success")
            return True, True
        elif choice in ("", "y", "yes"):
            self.console.print("  ✓ Approved", style="tiesta.success")
            return True, False
        else:
            self.console.print("  ✗ Denied", style="tiesta.error")
            return False, False



