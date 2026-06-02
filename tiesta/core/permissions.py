"""
tiesta/core/permissions.py
──────────────────────────
Unified Permission Control Plane.

Centralizes the human-in-the-loop permission logic to ensure dangerous actions
(like executing shell commands or doing major file deletions) are approved
by the user before execution.
"""

from typing import Any, Callable, Coroutine, Dict, Optional, Tuple


class PermissionManager:
    """Singleton managing permission states for the session."""

    _instance = None

    def __new__(cls) -> "PermissionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.auto_approve_all = False
            cls._instance.prompt_callback = None
            cls._instance.disabled = False
        return cls._instance

    def register_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Tuple[bool, bool]]]
    ) -> None:
        """Register the async UI callback that prompts the user.
        Callback signature: async (tool_name, args) -> (approved, auto_approve_all)
        """
        self.prompt_callback = callback

    async def request_permission(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Request permission to execute a dangerous tool.
        Returns True if allowed, False if denied.
        """
        if self.disabled or self.auto_approve_all:
            return True

        if self.prompt_callback:
            approved, auto_approve_all = await self.prompt_callback(tool_name, arguments)
            if auto_approve_all:
                self.auto_approve_all = True
            return approved

        # Fallback if no callback registered (fail closed)
        return False
