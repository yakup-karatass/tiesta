"""
main.py
───────
Tiesta — Local AI Coding Assistant.

Entrypoint that wires everything together:
1. Initialises the LLM client (→ local Ollama server).
2. Registers all tools (file_ops, bash_executor) into the ToolRegistry.
3. Wraps the registry with the Permission Control Plane.
4. Starts the interactive TUI chat loop OR runs a one-shot prompt.

Usage:
    tiesta                                   # interactive chat loop
    tiesta "create a basic express server"   # one-shot prompt
    tiesta -m qwen2.5-coder:3b -w /project
    python main.py                           # same as `tiesta`
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from tiesta.core.config import ConfigManager
from tiesta.core.llm_client import LLMClient, LLMClientConfig, LLMError
from tiesta.core.orchestrator import Orchestrator, OrchestratorConfig, ToolRegistry
from tiesta.core.permissions import PermissionManager
from tiesta.core.skill_loader import load_skills
from tiesta.tools.architecture_ops import ArchitectureOpsTool
from tiesta.tools.bash_executor import BashExecutor, BashExecutorConfig
from tiesta.tools.explorer import ExplorerTool
from tiesta.tools.file_ops import FileOps
from tiesta.tools.git_ops import GitOpsTool
from tiesta.tools.hardware_ops import HardwareOpsTool
from tiesta.tools.lsp_tools import LSPTools
from tiesta.tools.semantic_search import SemanticSearchTool
from tiesta.ui.console import TiestaConsole
from tiesta.ui.wizard import run_wizard

# ────────────────────── Logging setup ─────────────────────────────


def _setup_logging(verbose: bool, workspace: str) -> None:
    """Configure logging — file for debug, console only for warnings+."""
    log_dir = Path(workspace) / ".tiesta"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "tiesta.log"

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=handlers,
    )


# ────────────────────── CLI arguments ─────────────────────────────


def _parse_args() -> argparse.Namespace:
    config = ConfigManager().load()

    parser = argparse.ArgumentParser(
        prog="tiesta",
        description="Tiesta — Local AI Coding Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tiesta                                 # interactive mode\n"
            '  tiesta "refactor utils.py"             # one-shot prompt\n'
            "  tiesta -m qwen2.5-coder:3b             # specify model\n"
            "  tiesta -w /path/to/project             # specify workspace\n"
        ),
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help=(
            "One-shot prompt.  If provided, Tiesta processes this "
            "prompt and exits.  If omitted, starts interactive chat."
        ),
    )

    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=os.getcwd(),
        help="Project workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=config.default_model,
        help=f"Ollama model name (default: {config.default_model})",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="[localhost](http://localhost:11434/v1)",
        help="Ollama API base URL (default: [localhost](http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="LLM request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        # FIX: Reduced from 25 → 12. Small models (≤3B) lose track of the
        # original goal after ~10 turns. 12 is a safe ceiling that still
        # allows multi-step tasks but prevents runaway drift.
        default=12,
        help="Max orchestrator turns per task (default: 12)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging to stderr",
    )
    parser.add_argument(
        "--no-permission",
        action="store_true",
        help="Disable permission prompts (auto-approve everything)",
    )

    return parser.parse_args()


# ────────────────────── Tool registration ─────────────────────────


def _build_tool_registry(workspace: str) -> tuple[ToolRegistry, FileOps]:
    """Create and populate the tool registry with all available tools."""
    registry = ToolRegistry()

    file_ops = FileOps(workspace_root=workspace)
    file_ops.register(registry)

    bash_config = BashExecutorConfig(
        default_timeout_s=30.0,
        max_timeout_s=300.0,
        stdout_budget=3_000,
        stderr_budget=3_000,
    )
    bash_executor = BashExecutor(workspace_root=workspace, config=bash_config)
    bash_executor.register(registry)

    explorer = ExplorerTool(workspace_root=workspace)
    explorer.register(registry)

    git_ops = GitOpsTool(workspace_root=workspace)
    git_ops.register(registry)

    semantic_search = SemanticSearchTool(workspace_root=workspace)
    semantic_search.register(registry)

    lsp_tools = LSPTools(workspace_root=workspace)
    lsp_tools.register(registry)

    arch_tools = ArchitectureOpsTool(workspace_root=workspace)
    arch_tools.register(registry)

    hw_tools = HardwareOpsTool(workspace_root=workspace)
    hw_tools.register(registry)

    load_skills(registry, workspace)

    return registry, file_ops


# ────────────────── System prompt builder ─────────────────────────


async def _build_system_prompt(workspace: str, registry_names: list[str]) -> str:
    """Generate the system prompt optimized for small models (≤3B params).

    Design principles for small-model reliability:
    ──────────────────────────────────────────────
    1. **Workspace context FIRST** — small models exhibit strong recency
       bias, so volatile context (tree, git) goes at the top where it
       can be skimmed and discarded. Rules go at the BOTTOM where the
       model's attention is freshest when generating the next token.
    2. **Imperative, numbered protocols** — small models follow
       step-by-step checklists far better than abstract principles.
    3. **TASK CHECKLIST protocol** — forces the model to enumerate every
       requirement of the user's request before acting, which prevents
       silent omissions like "forgot to print Tiesta".
    4. **Self-verification gate** — explicit instruction to re-read the
       original user request before declaring completion.
    5. **No filler, no examples in prompt** — every token in the system
       prompt competes with the user's task for attention budget.
    """

    # ── Gather workspace awareness (goes at TOP of prompt) ─────────
    explorer = ExplorerTool(workspace_root=workspace)
    tree_result = await explorer._handle_list_directory_tree(
        {"path": ".", "max_depth": 3}
    )
    tree_str = tree_result.get("tree", "(unavailable)")

    git_ops = GitOpsTool(workspace_root=workspace)
    git_result = await git_ops._handle_get_git_status({})
    git_str = git_result.get("output", "(unavailable)")

    arch_map_path = Path(workspace) / "TIESTA_ARCHITECTURE.md"
    arch_str = ""
    if arch_map_path.exists():
        try:
            with open(arch_map_path, "r", encoding="utf-8") as f:
                arch_str = f.read()
        except Exception:
            pass

    # ── Tool list (compact) ────────────────────────────────────────
    tools_csv = ", ".join(registry_names)

    # ── Build prompt: CONTEXT first, RULES last ────────────────────
    # This ordering is critical for small models. Rules at the bottom
    # stay in the active attention window during generation.

    context_block = (
        f"# WORKSPACE CONTEXT\n"
        f"Directory: {workspace}\n"
        f"Available tools: {tools_csv}\n\n"
        f"## File tree (depth=3)\n"
        f"{tree_str}\n\n"
        f"## Git status\n"
        f"{git_str}\n"
    )

    if arch_str:
        context_block += f"\n## Architecture map\n{arch_str}\n"

    rules_block = (
        f"\n---\n\n"
        f"# YOU ARE TIESTA\n"
        f"An autonomous local coding assistant. You execute tool calls "
        f"to complete coding tasks. Follow the protocol below EXACTLY.\n\n"

        f"# PROTOCOL\n\n"

        f"## STEP 1 — DECOMPOSE THE TASK\n"
        f"Before doing anything else, internally enumerate EVERY discrete "
        f"requirement in the user's request as a numbered checklist.\n"
        f"Example: User says 'write a counter from 1 to 10 and then print Tiesta'.\n"
        f"  Checklist:\n"
        f"    [1] Write a loop counting 1 through 10\n"
        f"    [2] Print each number\n"
        f"    [3] After the loop, print the literal word 'Tiesta'\n"
        f"You MUST address every item. Missing one is a hard failure.\n\n"

        f"## STEP 2 — INVESTIGATE BEFORE EDITING\n"
        f"- Editing an existing file? → call `read_file` FIRST.\n"
        f"- Unknown symbol or import? → call `goto_definition` or `find_usages`.\n"
        f"- Unfamiliar library/API? → call `search_web` then `fetch_webpage`.\n"
        f"- Never patch a file you have not read in this session.\n\n"

        f"## STEP 3 — WRITE COMPLETE CODE\n"
        f"- Deliver the FULL solution. No placeholders, no '...', no TODO comments.\n"
        f"- Small targeted change → `patch_file`.\n"
        f"- New file or full rewrite → `write_file`.\n"
        f"- Code must run as-is. If imports are needed, include them.\n\n"

        f"## STEP 4 — VERIFY EVERY CHECKLIST ITEM\n"
        f"After writing code, before responding to the user:\n"
        f"  (a) Re-read your checklist from Step 1.\n"
        f"  (b) For each item, confirm it appears in the final code.\n"
        f"  (c) If any item is missing, fix it NOW with another tool call.\n"
        f"  (d) Optionally `read_file` the result to double-check.\n\n"

        f"# TOOL CALL RULES\n"
        f"- Output ONLY valid JSON for tool calls. Keys: 'name', 'arguments'.\n"
        f"- No prose mixed into tool call JSON.\n"
        f"- Wrong file edited → `undo_last_edit` immediately, do not patch over it.\n"
        f"- Tool error → analyze the error, change your approach, do not retry "
        f"the same call.\n\n"

        f"# RESPONSE RULES\n"
        f"- No filler ('Sure!', 'Of course!', 'Here you go').\n"
        f"- No restating the user's request.\n"
        f"- After completing a task, give a one-line confirmation of what was done.\n"
        f"- If you could not complete an item, say so explicitly with the reason.\n\n"

        f"# FINAL CHECK (mandatory before every user-facing reply)\n"
        f"Ask yourself: 'Did I address every numbered item from my Step 1 checklist?'\n"
        f"If the answer is not a confident YES, go back and fix it before replying.\n"
    )

    return context_block + rules_block


# ────────────────── Shared bootstrap logic ────────────────────────


async def _bootstrap(
    args: argparse.Namespace,
) -> tuple[TiestaConsole, Orchestrator, LLMClient, FileOps] | None:
    """Common init sequence for both interactive and one-shot modes."""
    workspace = str(Path(args.workspace).resolve())
    _setup_logging(args.verbose, workspace)
    tui = TiestaConsole()

    tui.print_banner()
    tui.print_status(model=args.model, workspace=workspace)

    llm_config = LLMClientConfig(
        base_url=args.base_url,
        default_model=args.model,
        request_timeout_s=args.timeout,
    )
    llm = LLMClient(config=llm_config)

    tui.print_info("Connecting to Ollama…")
    ping_result = await llm.ping()
    if isinstance(ping_result, LLMError):
        tui.print_connection_error(ping_result.message)
        await llm.close()
        return None

    models_result = await llm.list_models()
    if isinstance(models_result, list):
        tui.print_info(f"Available models: {', '.join(models_result[:8])}")
        if args.model not in models_result:
            tui.print_warning(
                f"Model '{args.model}' not found locally.  "
                f"Run: ollama pull {args.model}"
            )
    else:
        tui.print_info("Connected to Ollama ✓")

    tui.console.print()

    registry, file_ops = _build_tool_registry(workspace)
    tui.print_info(f"Tools loaded: {', '.join(registry.names)}")
    tui.print_info(
        f"Backup system: {file_ops.backups.stack_size} backup(s) from previous session"
    )

    # ── Permission Control Plane ───────────────────────────────────
    # FIX: Build the PermissionManager BEFORE the Orchestrator so we can
    # inject it. Previously the Orchestrator created its own internal
    # PermissionManager which never received the TUI callback.
    pm = PermissionManager()
    if args.no_permission:
        pm.disabled = True
        tui.print_warning("Permission prompts DISABLED (--no-permission)")
    else:
        pm.register_callback(tui.prompt_permission)
        tui.print_info("Permission Control Plane active for dangerous actions")

    # ── Orchestrator ───────────────────────────────────────────────
    system_prompt = await _build_system_prompt(workspace, registry.names)

    orch_config = OrchestratorConfig(
        max_turns=args.max_turns,
        system_prompt=system_prompt,
        # FIX: For small models, tighter correction budget prevents
        # endless self-correction loops on the same broken call.
        max_correction_attempts=2,
        # FIX: Slightly higher result budget — 3000 was too small once
        # workspace context is also competing for tokens. 4500 strikes
        # a balance between truncation and context exhaustion.
        tool_result_budget=4_500,
    )
    orchestrator = Orchestrator(
        llm=llm,
        tools=registry,
        config=orch_config,
        on_turn=tui.on_turn,
        # FIX: Inject the shared permission manager so the TUI callback
        # actually fires when a dangerous tool is invoked.
        permission_manager=pm,
    )

    return tui, orchestrator, llm, file_ops


# ────────────────────── Interactive mode ──────────────────────────


async def _run_interactive(
    tui: TiestaConsole,
    orchestrator: Orchestrator,
    file_ops: FileOps,
) -> None:
    """Asynchronous interactive chat loop — the default mode."""
    loop = asyncio.get_event_loop()

    tui.console.print()
    tui.console.print(
        "  Ready!  Ask me anything about your code.\n",
        style="tiesta.success",
    )

    while True:
        tui.console.print()
        try:
            user_input = await loop.run_in_executor(
                None,
                lambda: input("  You ❯ "),
            )
        except (EOFError, KeyboardInterrupt):
            tui.print_goodbye()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            handled = await _handle_slash_command(
                cmd, tui, orchestrator, file_ops
            )
            if handled == "exit":
                break
            if handled:
                continue
            tui.print_warning(
                f"Unknown command: {cmd}\n"
                "  Type /help to see available commands."
            )
            continue

        tui.print_thinking()

        try:
            result = await orchestrator.followup(user_input)
        except KeyboardInterrupt:
            tui.print_warning("Interrupted by user.")
            continue
        except Exception as exc:
            tui.print_warning(f"Unexpected error: {type(exc).__name__}: {exc}")
            logging.exception("Unhandled exception in chat loop")
            continue

        tui.print_separator()
        tui.print_info(
            f"  ── {result.total_llm_calls} LLM call(s), "
            f"{len([t for t in result.turns if t.tool_name])} tool action(s)"
        )


# ────────────────── Slash command handler ─────────────────────────


async def _handle_slash_command(
    cmd: str,
    tui: TiestaConsole,
    orchestrator: Orchestrator,
    file_ops: FileOps,
) -> str | None:
    """Route a slash command to its handler."""
    if cmd in ("/help", "/h", "/?"):
        tui.print_help()
        return "handled"

    if cmd in ("/clear", "/reset"):
        tui.console.clear()
        tui.print_banner()
        # FIX: Refresh zero-shot awareness on reset. We need a reliable
        # way to find the workspace root — use file_ops, not bash_executor,
        # since bash_executor's attribute name was inconsistent.
        workspace = str(file_ops._workspace_root)
        new_prompt = await _build_system_prompt(
            workspace, orchestrator.tools.names
        )
        orchestrator.cfg.system_prompt = new_prompt
        orchestrator.reset()
        tui.print_reset_confirmation()
        return "handled"

    if cmd in ("/exit", "/quit", "/q"):
        tui.print_goodbye()
        return "exit"

    if cmd == "/undo":
        result = await file_ops._handle_undo_last_edit({})
        if result["status"] == "ok":
            tui.print_undo_success(
                result["path"],
                result["operation_undone"],
                result["remaining_backups"],
            )
        else:
            tui.print_warning(result["error"])
        return "handled"

    return None


# ────────────────────── One-shot mode ─────────────────────────────


async def _run_oneshot(
    tui: TiestaConsole,
    orchestrator: Orchestrator,
    prompt: str,
) -> None:
    """Process a single prompt and exit."""
    tui.console.print()
    tui.console.print(f"  [bold]Prompt:[/] {prompt}\n", style="tiesta.info")

    tui.print_thinking()

    try:
        result = await orchestrator.run(prompt)
    except KeyboardInterrupt:
        tui.print_warning("Interrupted by user.")
        return
    except Exception as exc:
        tui.print_warning(f"Unexpected error: {type(exc).__name__}: {exc}")
        logging.exception("Unhandled exception in one-shot mode")
        return

    tui.print_separator()

    tool_count = len([t for t in result.turns if t.tool_name])
    tui.print_info(
        f"  ── Completed: {result.total_llm_calls} LLM call(s), "
        f"{tool_count} tool action(s)"
    )
    tui.console.print()


# ────────────────────── Entrypoint ────────────────────────────────


async def _async_main() -> None:
    """Async entrypoint — bootstraps and routes to the correct mode."""
    args = _parse_args()

    result = await _bootstrap(args)
    if result is None:
        sys.exit(1)

    tui, orchestrator, llm, file_ops = result

    try:
        if args.prompt:
            await _run_oneshot(tui, orchestrator, args.prompt)
        else:
            await _run_interactive(tui, orchestrator, file_ops)
    finally:
        await llm.close()


def cli_entry() -> None:
    """Synchronous entry point for ``console_scripts``."""
    manager = ConfigManager()
    if not manager.exists():
        run_wizard()

    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\n  Goodbye!")


if __name__ == "__main__":
    cli_entry()
