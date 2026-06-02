"""
Web Search Skill
────────────────
A skill demonstrating the Tiesta Plugin Engine and Autonomous Research.

To use this skill, copy this file into your global or local skills directory:
- Global: `~/.tiesta/skills/web_search.py`
- Local: `<workspace>/.tiesta/skills/web_search.py`
"""

import logging
from typing import Any, Dict, List

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """A custom tool for performing web searches via DuckDuckGo."""

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_web",
                description=(
                    "Perform a web search using DuckDuckGo to find information, "
                    "documentation, or answers to errors. Returns the top "
                    "search results including their URLs and snippets."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'python asyncio subprocess timeout')",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Defaults to 5.",
                            "default": 5,
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self._handle_search_web,
            ),
        ]

    async def _handle_search_web(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a DuckDuckGo text search."""
        query = args.get("query", "")
        if not query:
            return {"status": "error", "error": "Query is required."}

        max_results = int(args.get("max_results", 5))

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return {
                "status": "error",
                "error": "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
            }

        try:
            results = []
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(query, max_results=max_results)
                for r in ddgs_gen:
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    })
                    
            if not results:
                return {
                    "status": "ok",
                    "query": query,
                    "results": "No results found."
                }

            # Return cleanly formatted JSON-serializable list
            return {
                "status": "ok",
                "query": query,
                "results": results
            }

        except Exception as e:
            logger.error("Web search failed: %s", e, exc_info=True)
            return {"status": "error", "error": f"Search failed: {e}"}


# =====================================================================
# Registration Hook
# Tiesta's Plugin Engine looks for this exact function signature.
# =====================================================================

def register_skill(registry, workspace_root: str) -> None:
    """Injects the WebSearchTool into Tiesta's ToolRegistry."""
    searcher = WebSearchTool(workspace_root)
    searcher.register(registry)
