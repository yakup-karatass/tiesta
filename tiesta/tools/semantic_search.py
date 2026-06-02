"""
tools/semantic_search.py
────────────────────────
Semantic code search (Local RAG) using ChromaDB.

Provides the `search_codebase` tool, allowing the agent to locate
specific logic across the workspace using natural language queries.
Automatically maintains a local vector database inside `.tiesta/vector_db/`.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from tiesta.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class SemanticSearchTool(BaseTool):
    """Provides semantic code search using a local vector database."""

    # File extensions to index
    ALLOWED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".html", ".css",
        ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".json", ".toml"
    }

    # Directories to ignore
    IGNORE_DIRS = {
        "node_modules", "venv", ".venv", "env", ".env", ".git",
        "__pycache__", "build", "dist", ".tiesta", ".pytest_cache",
        ".mypy_cache", ".idea", ".vscode"
    }

    def __init__(self, workspace_root: str) -> None:
        super().__init__(workspace_root)
        self._db_path = Path(workspace_root) / ".tiesta" / "vector_db"
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._db_path / "_manifest.json"
        
        # Lazy initialization of chromadb client and collection
        self._client = None
        self._collection = None

    def _init_db(self) -> None:
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise RuntimeError("chromadb is not installed. Run: pip install chromadb")

            # Initialize persistent client
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            self._collection = self._client.get_or_create_collection(
                name="workspace_code",
                metadata={"hnsw:space": "cosine"}
            )

    def definitions(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_codebase",
                description=(
                    "Search the codebase semantically using natural language. "
                    "Use this to find where specific logic, classes, or functions "
                    "are implemented. Returns the top matching code snippets "
                    "with their file paths and line numbers. The index is "
                    "automatically kept up to date."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query (e.g. 'Where is the password hashing logic?')",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of top snippets to return. Defaults to 5.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self._handle_search_codebase,
            ),
        ]

    def _load_manifest(self) -> Dict[str, float]:
        """Load the tracking manifest mapping relative paths to mtimes."""
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_manifest(self, manifest: Dict[str, float]) -> None:
        """Save the tracking manifest."""
        try:
            self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to save vector db manifest: %s", exc)

    def _chunk_file(self, content: str, max_lines_per_chunk: int = 300) -> List[Dict[str, Any]]:
        """Split content into semantic structural chunks using Regex."""
        import re
        lines = content.splitlines(keepends=True)
        chunks = []
        total_lines = len(lines)
        
        if total_lines == 0:
            return chunks

        # Regex to detect major structural blocks (class, def, struct, fn, etc.)
        pattern = re.compile(
            r"^(?:export\s+|public\s+|private\s+|protected\s+|async\s+)?"
            r"(?:class|def|function|struct|enum|fn|interface)\s+[a-zA-Z0-9_]+"
        )

        boundaries = [0]
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if pattern.match(line):
                boundaries.append(i)
        
        boundaries.append(total_lines)
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            # If the block is too large, hard split it
            while end - start > max_lines_per_chunk:
                mid = start + max_lines_per_chunk
                chunk_text = "".join(lines[start:mid])
                chunks.append({
                    "text": chunk_text,
                    "start_line": start + 1,
                    "end_line": mid,
                })
                start = mid
                
            if start < end:
                chunk_text = "".join(lines[start:end])
                chunks.append({
                    "text": chunk_text,
                    "start_line": start + 1,
                    "end_line": end,
                })

        return chunks

    def _scan_workspace_files(self) -> Dict[str, float]:
        """Scan the workspace and return a dict of allowed files and their mtimes."""
        current_files: Dict[str, float] = {}
        workspace_path = Path(self._workspace_root)
        
        for root, dirs, files in os.walk(workspace_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.ALLOWED_EXTENSIONS:
                    abs_path = Path(root) / file
                    try:
                        mtime = abs_path.stat().st_mtime
                        rel_path = self._relative_display(str(abs_path))
                        current_files[rel_path] = mtime
                    except OSError:
                        pass
        return current_files

    def _sync_index(self) -> None:
        """Ensure the vector db is in sync with the file system."""
        self._init_db()
        assert self._collection is not None

        manifest = self._load_manifest()
        current_files = self._scan_workspace_files()
        workspace_path = Path(self._workspace_root)
        
        to_add_or_update = []
        to_delete = []

        # Find new or modified files
        for rel_path, mtime in current_files.items():
            if rel_path not in manifest or manifest[rel_path] < mtime:
                to_add_or_update.append(rel_path)

        # Find deleted files
        for rel_path in list(manifest.keys()):
            if rel_path not in current_files:
                to_delete.append(rel_path)

        if not to_add_or_update and not to_delete:
            return  # Index is up to date

        logger.info("SemanticSearch: Syncing index (%d to update, %d to delete)", len(to_add_or_update), len(to_delete))

        # Handle deletions
        if to_delete:
            # Delete chunks associated with deleted files
            # ChromaDB delete by metadata where condition
            for rel_path in to_delete:
                self._collection.delete(where={"file": rel_path})
                del manifest[rel_path]

        # Handle additions and updates
        for rel_path in to_add_or_update:
            abs_path = workspace_path / rel_path
            
            # If updating, first delete existing chunks for this file
            if rel_path in manifest:
                self._collection.delete(where={"file": rel_path})
            
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                chunks = self._chunk_file(content)
                
                if chunks:
                    ids = []
                    documents = []
                    metadatas = []
                    
                    for i, chunk in enumerate(chunks):
                        ids.append(f"{rel_path}::{i}")
                        # Prepend file info to the document text for better embedding context
                        header = f"File: {rel_path} (Lines {chunk['start_line']}-{chunk['end_line']})\n"
                        documents.append(header + chunk["text"])
                        metadatas.append({
                            "file": rel_path,
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                        })
                    
                    self._collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                
                manifest[rel_path] = current_files[rel_path]
            except Exception as exc:
                logger.warning("Failed to index %s: %s", rel_path, exc)

        self._save_manifest(manifest)

    async def _handle_search_codebase(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search the codebase semantically."""
        query = args.get("query", "")
        n_results = int(args.get("n_results", 5))

        if not query:
            return {"status": "error", "error": "Parameter 'query' is required."}

        try:
            # Sync index before searching to ensure freshness
            self._sync_index()
        except RuntimeError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": f"Failed to sync index: {e}"}

        assert self._collection is not None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
            )
        except Exception as e:
            return {"status": "error", "error": f"Query failed: {e}"}

        if not results["documents"] or not results["documents"][0]:
            return {
                "status": "ok",
                "output": "No matches found in the codebase.",
            }

        snippets = []
        for doc in results["documents"][0]:
            snippets.append(doc.strip())

        output_str = "\n\n" + ("=" * 50) + "\n\n".join(snippets) + "\n\n" + ("=" * 50)

        return {
            "status": "ok",
            "matches_found": len(snippets),
            "output": output_str,
        }
