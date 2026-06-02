"""
Web Scraper Skill
─────────────────
A template skill demonstrating the Tiesta Plugin Engine.

To use this skill, copy this file into your global or local skills directory:
- Global: `~/.tiesta/skills/web_scraper.py`
- Local: `<workspace>/.tiesta/skills/web_scraper.py`
"""

import html.parser
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class HTMLTextExtractor(html.parser.HTMLParser):
    """Simple built-in HTML parser to extract clean text."""
    def __init__(self):
        super().__init__()
        self.text_chunks = []
        self.ignore_tags = {'script', 'style', 'head', 'meta', 'title'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.ignore_tags:
            clean = data.strip()
            if clean:
                self.text_chunks.append(clean)


class WebScraperTool(BaseTool):
    """A custom tool for fetching and reading web pages."""

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="fetch_webpage",
                description=(
                    "Fetch a webpage from a given URL and return its raw text content. "
                    "HTML tags and scripts are automatically stripped out. "
                    "Use this to read external documentation or guides."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch (e.g., https://docs.python.org/)",
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Max characters to return (to prevent context bloat). Defaults to 6000.",
                            "default": 6000,
                        }
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                handler=self._handle_fetch_webpage,
            ),
        ]

    async def _handle_fetch_webpage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch and parse the URL."""
        url = args.get("url", "")
        if not url:
            return {"status": "error", "error": "URL is required."}

        max_length = int(args.get("max_length", 6000))
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Tiesta/1.0 (Web Scraper Skill)'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                html_bytes = response.read()
                # Attempt to guess encoding from headers, default to utf-8
                encoding = response.headers.get_content_charset('utf-8')
                html_content = html_bytes.decode(encoding, errors='replace')
        except urllib.error.URLError as e:
            return {"status": "error", "error": f"Failed to fetch URL: {e.reason}"}
        except Exception as e:
            return {"status": "error", "error": f"Request failed: {e}"}

        # Parse text
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        
        # Join chunks and collapse multiple newlines
        raw_text = "\n".join(parser.text_chunks)
        
        # Truncate if necessary
        if len(raw_text) > max_length:
            half = max_length // 2
            raw_text = (
                raw_text[:half] + 
                f"\n\n... [TRUNCATED - Webpage was {len(raw_text)} chars] ...\n\n" + 
                raw_text[-half:]
            )

        return {
            "status": "ok",
            "url": url,
            "content": raw_text,
        }


# =====================================================================
# Registration Hook
# Tiesta's Plugin Engine looks for this exact function signature.
# =====================================================================

def register_skill(registry, workspace_root: str) -> None:
    """Injects the WebScraperTool into Tiesta's ToolRegistry."""
    scraper = WebScraperTool(workspace_root)
    scraper.register(registry)
