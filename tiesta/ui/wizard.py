"""
tiesta/ui/wizard.py
───────────────────
Interactive setup wizard for first-time users.
"""

import os
import shutil
from pathlib import Path

import urllib.request
import urllib.error

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from tiesta.core.config import Config, ConfigManager


def run_wizard() -> None:
    """Run the interactive onboarding sequence and save preferences."""
    console = Console()
    console.clear()
    
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to Tiesta[/]\n\n"
            "Tiesta is a fully autonomous local coding assistant.\n"
            "Let's get you set up for the first time.",
            title="Setup Wizard",
            border_style="cyan"
        )
    )
    console.print()

    # 1. Fetch available models
    models = _fetch_ollama_models()
    
    if not models:
        console.print("[bold red]No Ollama models detected. Please run 'ollama pull qwen2.5-coder:3b' first.[/]")
        model = "qwen2.5-coder:3b"
    else:
        default_model = "qwen2.5-coder:3b" if "qwen2.5-coder:3b" in models else models[0]
        model = Prompt.ask(
            "[bold green]Which Ollama model would you like to use by default?[/]",
            choices=models,
            default=default_model,
            console=console,
        )

    # 2. Ask for default skills
    install_skills = Confirm.ask(
        "[bold green]Do you want to install the default skills (Web Search, Web Scraper)?[/]",
        default=True,
        console=console,
    )

    config = Config(default_model=model)

    if install_skills:
        _install_default_skills(console)
        config.enabled_skills = ["web_search", "web_scraper"]

    # Save to global config
    manager = ConfigManager()
    manager.save(config)

    console.print()
    console.print("[bold cyan]Setup complete! Tiesta is ready to code.[/]")
    console.print()


def _fetch_ollama_models() -> list[str]:
    """Fetch the list of available models from local Ollama instance."""
    import json
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return models
    except (urllib.error.URLError, json.JSONDecodeError):
        return []

def _install_default_skills(console: Console) -> None:
    """Copy default skills from the package to the user's home directory."""
    home_skills_dir = Path.home() / ".tiesta" / "skills"
    home_skills_dir.mkdir(parents=True, exist_ok=True)

    # Determine where the skills_template folder is in the installed package
    package_dir = Path(__file__).parent.parent
    template_dir = package_dir / "skills_template"

    if not template_dir.exists():
        console.print(f"[bold red]Error:[/] Could not locate skills template at {template_dir}")
        return

    for filename in ["web_search.py", "web_scraper.py"]:
        src = template_dir / filename
        dst = home_skills_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            console.print(f"  [dim]Installed {filename}[/]")
        else:
            console.print(f"  [bold red]Error:[/] {filename} not found in template directory.")
