"""
tiesta/core/config.py
─────────────────────
Global configuration manager for Tiesta.

Manages user preferences stored in `~/.tiesta/config.json`.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Config:
    default_model: str = "qwen2.5-coder:7b"
    enabled_skills: List[str] = field(default_factory=list)


class ConfigManager:
    """Handles reading and writing the global Tiesta config."""

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".tiesta"
        self.config_path = self.config_dir / "config.json"

    def exists(self) -> bool:
        """Check if the global config file exists (indicates completed onboarding)."""
        return self.config_path.exists()

    def load(self) -> Config:
        """Load the global config, returning default if not found."""
        if not self.exists():
            return Config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Config(**data)
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", self.config_path, exc)
            return Config()

    def save(self, config: Config) -> None:
        """Save the config to disk."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(config), f, indent=4)
        except Exception as exc:
            logger.error("Failed to save config to %s: %s", self.config_path, exc)
