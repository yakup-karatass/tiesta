"""
core/skill_loader.py
────────────────────
Dynamic Plugin/Skill Engine for Tiesta.

Scans local and global `.tiesta/skills/` directories for `.py` files.
Dynamically imports them and calls `register_skill(registry, workspace_root)` 
if defined, allowing the user to extend Tiesta with custom tools without
modifying the core orchestrator codebase.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

from tiesta.core.orchestrator import ToolRegistry

logger = logging.getLogger(__name__)


def load_skills(registry: ToolRegistry, workspace_root: str) -> None:
    """Scan and dynamically load all skills into the registry."""
    
    # 1. Determine skill directories
    home_dir = Path.home()
    global_skills_dir = home_dir / ".tiesta" / "skills"
    local_skills_dir = Path(workspace_root) / ".tiesta" / "skills"

    # Create directories if they don't exist
    global_skills_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_skills_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # In case workspace is read-only or similar

    directories_to_scan = [
        ("Global", global_skills_dir),
        ("Local", local_skills_dir),
    ]

    for scope, directory in directories_to_scan:
        if not directory.exists() or not directory.is_dir():
            continue

        for filename in os.listdir(directory):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            filepath = directory / filename
            module_name = f"tiesta_skill_{scope.lower()}_{filename[:-3]}"

            try:
                # Dynamically load the module
                spec = importlib.util.spec_from_file_location(module_name, str(filepath))
                if spec is None or spec.loader is None:
                    logger.warning("Could not load skill spec for %s", filename)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Check for the registration hook
                if hasattr(module, "register_skill"):
                    module.register_skill(registry, workspace_root)
                    logger.info("Loaded %s skill: %s", scope, filename)
                else:
                    logger.debug("Skill %s has no 'register_skill' function, skipping.", filename)

            except Exception as exc:
                logger.error("Failed to load skill '%s': %s", filename, exc, exc_info=True)
