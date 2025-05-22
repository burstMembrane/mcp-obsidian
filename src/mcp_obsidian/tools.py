import json
import os
from collections.abc import Sequence
from pathlib import Path

from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    Tool,
)

from mcp_obsidian.embeddings import _index_md, _query_chroma

from . import embeddings, obsidian

api_key = os.getenv("OBSIDIAN_API_KEY", "")
obsidian_host = os.getenv("OBSIDIAN_HOST", "127.0.0.1")

if api_key == "":
    raise ValueError(
        f"OBSIDIAN_API_KEY environment variable required. Working directory: {os.getcwd()}"
    )

TOOL_LIST_FILES_IN_VAULT = "obsidian_list_files_in_vault"
TOOL_LIST_FILES_IN_DIR = "obsidian_list_files_in_dir"


class ToolHandler:
    def __init__(self, tool_name: str):
        self.name = tool_name

    def get_tool_description(self) -> Tool:
        raise NotImplementedError()

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        raise NotImplementedError()


class ListFilesInVaultToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_VAULT)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories in the root directory of your Obsidian vault.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        files = api.list_files_in_vault()

        return [TextContent(type="text", text=json.dumps(files, indent=2))]


class ListFilesInDirToolHandler(ToolHandler):
    def __init__(self):
        super().__init__(TOOL_LIST_FILES_IN_DIR)

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Lists all files and directories that exist in a specific Obsidian directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Path to list files from (relative to your vault root). Note that empty directories will not be returned.",
                    },
                },
                "required": ["dirpath"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "dirpath" not in args:
            raise RuntimeError("dirpath argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        files = api.list_files_in_dir(args["dirpath"])

        return [TextContent(type="text", text=json.dumps(files, indent=2))]


class GetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the content of a single file in your vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the relevant file (relative to your vault root).",
                        "format": "path",
                    },
                },
                "required": ["filepath"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepath" not in args:
            raise RuntimeError("filepath argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)

        content = api.get_file_contents(args["filepath"])

        return [TextContent(type="text", text=json.dumps(content, indent=2))]


class LoadEmbeddingsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_load_embeddings")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="""Load Smart Connections embeddings from the vault's .smart-env/multi directory.
            Returns information about available embeddings and their metadata.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of embedding files to load (default: 10)",
                        "default": 10,
                    },
                    "show_vectors": {
                        "type": "boolean",
                        "description": "Whether to include the actual embedding vectors (default: false)",
                        "default": False,
                    },
                },
                "required": [],
            },
        )

    def get_embedding_path(self):
        # TODO: make this smarter
        from pathlib import Path

        vault_path = Path(
            os.getenv("VAULT_PATH", "/Users/liampower/obsidianvaults/home")
        )
        return vault_path / ".smart-env" / "multi"

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        import json
        import os
        from pathlib import Path

        limit = args.get("limit", 10)
        show_vectors = args.get("show_vectors", False)

        # Get vault path from environment or use default
        # TODO: make this smarter
        vault_path = Path(
            os.getenv("VAULT_PATH", "/Users/liampower/obsidianvaults/home")
        )
        smart_env_path = vault_path / ".smart-env" / "multi"
        if not smart_env_path.exists():
            return [
                TextContent(
                    type="text",
                    text=f"Smart Connections embeddings not found at: {smart_env_path}",
                )
            ]

        # Load embedding files
        embedding_files = [
            f for f in os.listdir(smart_env_path) if f.endswith(".ajson")
        ]
        embedding_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(smart_env_path, x))
        )

        if not embedding_files:
            return [
                TextContent(
                    type="text",
                    text=f"No .ajson embedding files found in: {smart_env_path}",
                )
            ]

        loaded_embeddings = []
        files_processed = 0

        for filename in embedding_files[:limit]:
            try:
                file_path = smart_env_path / filename
                with open(str(file_path.absolute()), "r", encoding="utf-8") as f:
                    content = f.read().strip()

                    # Handle .ajson format - wrap fragments in braces and remove trailing comma
                    if content.endswith(","):
                        content = content[:-1]  # Remove trailing comma

                    # Wrap in braces to make valid JSON
                    json_content = f"{{{content}}}"

                    data = json.loads(json_content)
                # Extract embedding info for each source in the file
                for source_key, source_data in data.items():
                    if "embeddings" in source_data:
                        embedding_info = {
                            "file": filename,
                            "source_key": source_key,
                            "path": source_data.get("path", ""),
                            "embedding_model": None,
                            "vector_dimensions": None,
                            "last_embed_hash": None,
                            "tokens": None,
                            "blocks": source_data.get("blocks", {}),
                            "outlinks_count": len(source_data.get("outlinks", [])),
                        }

                        # Get embedding details
                        embeddings = source_data.get("embeddings", {})
                        for model_name, embed_data in embeddings.items():
                            embedding_info["embedding_model"] = model_name
                            if "vec" in embed_data:
                                embedding_info["vector_dimensions"] = len(
                                    embed_data["vec"]
                                )
                                if show_vectors:
                                    embedding_info["vector"] = embed_data["vec"][
                                        :5
                                    ]  # Show first 5 values
                            if "last_embed" in embed_data:
                                embedding_info["last_embed_hash"] = embed_data[
                                    "last_embed"
                                ].get("hash", "")
                                embedding_info["tokens"] = embed_data["last_embed"].get(
                                    "tokens", 0
                                )
                            break  # Just take the first embedding model

                        loaded_embeddings.append(embedding_info)
                        files_processed += 1

            except Exception as e:
                loaded_embeddings.append(
                    {"file": filename, "error": f"Failed to load: {str(e)}"}
                )

        summary = {
            "smart_env_path": str(smart_env_path.absolute()),
            "total_ajson_files": len(embedding_files),
            "files_processed": files_processed,
            "embeddings_loaded": len(
                [e for e in loaded_embeddings if "error" not in e]
            ),
            "embeddings": loaded_embeddings,
        }

        return [TextContent(type="text", text=json.dumps(summary, indent=2))]


class SearchToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_simple_search")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="""Simple search for documents matching a specified text query across all files in the vault. 
            Use this tool when you want to do a simple text search""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to a simple search for in the vault.",
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "How much context to return around the matching string (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["query"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "query" not in args:
            raise RuntimeError("query argument missing in arguments")

        context_length = args.get("context_length", 100)

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.search(args["query"], context_length)

        formatted_results = []
        for result in results:
            formatted_matches = []
            for match in result.get("matches", []):
                context = match.get("context", "")
                match_pos = match.get("match", {})
                start = match_pos.get("start", 0)
                end = match_pos.get("end", 0)

                formatted_matches.append(
                    {"context": context, "match_position": {"start": start, "end": end}}
                )

            formatted_results.append(
                {
                    "filename": result.get("filename", ""),
                    "score": result.get("score", 0),
                    "matches": formatted_matches,
                }
            )

        return [TextContent(type="text", text=json.dumps(formatted_results, indent=2))]


class EchoToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_echo")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Echo a message back to the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo back to the user.",
                    }
                },
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "message" not in args:
            raise RuntimeError("message argument missing in arguments")

        return [TextContent(type="text", text=args["message"])]


class AppendContentToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_append_content")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Append content to a new or existing file in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file (relative to vault root)",
                        "format": "path",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append to the file",
                    },
                },
                "required": ["filepath", "content"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepath" not in args or "content" not in args:
            raise RuntimeError("filepath and content arguments required")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        api.append_content(args.get("filepath", ""), args["content"])

        return [
            TextContent(
                type="text", text=f"Successfully appended content to {args['filepath']}"
            )
        ]


class PatchContentToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_patch_content")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Insert content into an existing note relative to a heading, block reference, or frontmatter field.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file (relative to vault root)",
                        "format": "path",
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform (append, prepend, or replace)",
                        "enum": ["append", "prepend", "replace"],
                    },
                    "target_type": {
                        "type": "string",
                        "description": "Type of target to patch",
                        "enum": ["heading", "block", "frontmatter"],
                    },
                    "target": {
                        "type": "string",
                        "description": "Target identifier (heading path, block reference, or frontmatter field)",
                    },
                    "content": {"type": "string", "description": "Content to insert"},
                },
                "required": [
                    "filepath",
                    "operation",
                    "target_type",
                    "target",
                    "content",
                ],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if not all(
            k in args
            for k in ["filepath", "operation", "target_type", "target", "content"]
        ):
            raise RuntimeError(
                "filepath, operation, target_type, target and content arguments required"
            )

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        api.patch_content(
            args.get("filepath", ""),
            args.get("operation", ""),
            args.get("target_type", ""),
            args.get("target", ""),
            args.get("content", ""),
        )

        return [
            TextContent(
                type="text", text=f"Successfully patched content in {args['filepath']}"
            )
        ]


class DeleteFileToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_delete_file")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Delete a file or directory from the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file or directory to delete (relative to vault root)",
                        "format": "path",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirmation to delete the file (must be true)",
                        "default": False,
                    },
                },
                "required": ["filepath", "confirm"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepath" not in args:
            raise RuntimeError("filepath argument missing in arguments")

        if not args.get("confirm", False):
            raise RuntimeError("confirm must be set to true to delete a file")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        api.delete_file(args["filepath"])

        return [
            TextContent(type="text", text=f"Successfully deleted {args['filepath']}")
        ]


class ComplexSearchToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_complex_search")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="""Complex search for documents using a JsonLogic query. 
           Supports standard JsonLogic operators plus 'glob' and 'regexp' for pattern matching. Results must be non-falsy.

           Use this tool when you want to do a complex search, e.g. for all documents with certain tags etc.
           """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": 'JsonLogic query object. Example: {"glob": ["*.md", {"var": "path"}]} matches all markdown files',
                    }
                },
                "required": ["query"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "query" not in args:
            raise RuntimeError("query argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.search_json(args.get("query", ""))

        return [TextContent(type="text", text=json.dumps(results, indent=2))]


class BatchGetFileContentsToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_batch_get_file_contents")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Return the contents of multiple files in your vault, concatenated with headers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepaths": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Path to a file (relative to your vault root)",
                            "format": "path",
                        },
                        "description": "List of file paths to read",
                    },
                },
                "required": ["filepaths"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "filepaths" not in args:
            raise RuntimeError("filepaths argument missing in arguments")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        content = api.get_batch_file_contents(args["filepaths"])

        return [TextContent(type="text", text=content)]


class PeriodicNotesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_periodic_note")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get current periodic note for the specified period.",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "The period type (daily, weekly, monthly, quarterly, yearly)",
                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                    }
                },
                "required": ["period"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "period" not in args:
            raise RuntimeError("period argument missing in arguments")

        period = args["period"]
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if period not in valid_periods:
            raise RuntimeError(
                f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}"
            )

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        content = api.get_periodic_note(period)

        return [TextContent(type="text", text=content)]


class RecentPeriodicNotesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_recent_periodic_notes")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get most recent periodic notes for the specified period type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "The period type (daily, weekly, monthly, quarterly, yearly)",
                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of notes to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to include note content (default: false)",
                        "default": False,
                    },
                },
                "required": ["period"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        if "period" not in args:
            raise RuntimeError("period argument missing in arguments")

        period = args["period"]
        valid_periods = ["daily", "weekly", "monthly", "quarterly", "yearly"]
        if period not in valid_periods:
            raise RuntimeError(
                f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}"
            )

        limit = args.get("limit", 5)
        if not isinstance(limit, int) or limit < 1:
            raise RuntimeError(f"Invalid limit: {limit}. Must be a positive integer")

        include_content = args.get("include_content", False)
        if not isinstance(include_content, bool):
            raise RuntimeError(
                f"Invalid include_content: {include_content}. Must be a boolean"
            )

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.get_recent_periodic_notes(period, limit, include_content)

        return [TextContent(type="text", text=json.dumps(results, indent=2))]


class RecentChangesToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_get_recent_changes")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Get recently modified files in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Only include files modified within this many days (default: 90)",
                        "minimum": 1,
                        "default": 90,
                    },
                },
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        limit = args.get("limit", 10)
        if not isinstance(limit, int) or limit < 1:
            raise RuntimeError(f"Invalid limit: {limit}. Must be a positive integer")

        days = args.get("days", 90)
        if not isinstance(days, int) or days < 1:
            raise RuntimeError(f"Invalid days: {days}. Must be a positive integer")

        api = obsidian.Obsidian(api_key=api_key, host=obsidian_host)
        results = api.get_recent_changes(limit, days)

        return [TextContent(type="text", text=json.dumps(results, indent=2))]


class ChromaIndexToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_chroma_index")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Index markdown notes from your vault into ChromaDB for semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vault": {
                        "type": "string",
                        "description": "Vault subfolder name inside the Obsidian root directory",
                        "default": "home",
                    },
                    "db": {
                        "type": "string",
                        "description": "Path to the ChromaDB database directory (default: chroma.db)",
                        "default": "chroma.db",
                    },
                    "collection": {
                        "type": "string",
                        "description": "ChromaDB collection name (default: {vault_name})",
                        "default": "home",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Number of documents to process per batch (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["vault"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        obsidian_root = Path(os.getenv("VAULT_PATH", "~/obsidianvaults")).expanduser()
        vault = args.get("vault")
        db = args.get("db", "chroma.db")
        collection = args.get("collection", "vault_docs")
        batch_size = args.get("batch_size", 10)

        _index_md(obsidian_root, vault, db, collection, batch_size)

        return [
            TextContent(
                type="text",
                text=f"Indexed vault '{vault}' into collection '{collection}'",
            )
        ]


class ChromaQueryToolHandler(ToolHandler):
    def __init__(self):
        super().__init__("obsidian_chroma_query")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Query your indexed Obsidian notes using ChromaDB semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query to search for",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                    "db": {
                        "type": "string",
                        "description": "Path to the ChromaDB database directory (default: chroma.db)",
                        "default": "chroma.db",
                    },
                    "collection": {
                        "type": "string",
                        "description": "ChromaDB collection name (default: vault_docs)",
                        "default": "vault_docs",
                    },
                    "where": {
                        "type": "string",
                        "description": "Optional JSON metadata filter (as a string)",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        )

    def run_tool(
        self, args: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        query = args.get("query")
        n = args.get("n", 5)
        db = args.get("db", "chroma.db")
        collection = args.get("collection", "vault_docs")
        where = args.get("where") or None

        results = _query_chroma(query, n, db, collection, where)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        def escape_json_string(s: str) -> str:
            return (
                s.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\b", "\\b")
                .replace("\f", "\\f")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

        output = []
        for doc, meta in zip(documents, metadatas):
            title = meta.get("title", "Untitled")
            safe_text = escape_json_string(f"# {title}\n\n{doc}")
            output.append(TextContent(type="text", text=safe_text))

        return output
