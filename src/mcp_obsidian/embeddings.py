import bisect
import json
import re

# Fetch existing documents and stored mtimes
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
import frontmatter
import mistune
import typer
from chromadb.api.types import (
    Documents,
    Embeddings,
    IDs,
    Metadatas,
)
from mistune.core import BlockState
from mistune.renderers.markdown import MarkdownRenderer
from rich.console import Console
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

console = Console()

app = typer.Typer()


def strip_and_map(markdown_text, languages_to_remove, remove_all_fenced=False):
    """
    Removes YAML frontmatter, blockquote code blocks (with or without language), and
    fenced code blocks with specified language identifiers. Optionally removes all
    fenced code blocks.

    Returns the cleaned markdown text and a mapping list that maps each character
    index in the cleaned text back to the character index in the original text.
    """
    # Strip YAML frontmatter
    post = frontmatter.loads(markdown_text)
    content = post.content

    # Create a list of tuples (start, end) for ranges to remove
    remove_ranges = []

    def add_match_range(match):
        start, end = match.span()
        remove_ranges.append((start, end))

    # Find ranges to remove for specified languages
    for lang in languages_to_remove:
        # Blockquote code block: >   ```lang ... >   ```
        blockquote_pattern = rf"""
            (                           # Entire blockquote code block
                (?:^>.*\n)*?            # Optional leading lines
                ^>\s*```{lang}\s*\n     # Opening with optional spaces
                (?:^>.*\n)*?            # Content lines
                ^>\s*```\s*$            # Closing with optional spaces
            )
        """
        for match in re.finditer(
            blockquote_pattern, content, flags=re.MULTILINE | re.VERBOSE
        ):
            add_match_range(match)

        # Normal fenced code block: ```lang ... ```
        fenced_pattern = rf"^```{lang}\s*\n.*?\n```[ \t]*\n?"
        for match in re.finditer(
            fenced_pattern, content, flags=re.DOTALL | re.MULTILINE
        ):
            add_match_range(match)

    if remove_all_fenced:
        # Generic fenced blocks: ```...```, with or without language
        generic_pattern = r"^```[^\n]*\n.*?\n```[ \t]*\n?"
        for match in re.finditer(
            generic_pattern, content, flags=re.DOTALL | re.MULTILINE
        ):
            add_match_range(match)

    # Sort ranges and merge overlapping or adjacent ones
    remove_ranges.sort()
    merged_ranges = []
    for start, end in remove_ranges:
        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)

    cleaned_chars = []
    mapping = []
    idx = 0
    for start, end in merged_ranges:
        while idx < start:
            cleaned_chars.append(content[idx])
            mapping.append(idx)
            idx += 1
        idx = end
    while idx < len(content):
        cleaned_chars.append(content[idx])
        mapping.append(idx)
        idx += 1

    cleaned_text = "".join(cleaned_chars)
    return cleaned_text, mapping


def remove_specific_code_blocks(
    markdown_text, languages_to_remove, remove_all_fenced=False
):
    """
    Removes YAML frontmatter, blockquote code blocks (with or without language), and
    fenced code blocks with specified language identifiers. Optionally removes all
    fenced code blocks.

    Args:
        markdown_text (str): The markdown content to process.
        languages_to_remove (list): Language identifiers to target.
        remove_all_fenced (bool): Remove all fenced code blocks, regardless of language.

    Returns:
        str: Cleaned markdown.
    """
    # Strip YAML frontmatter
    post = frontmatter.loads(markdown_text)
    markdown_text = post.content

    for lang in languages_to_remove:
        # Blockquote code block: >   ```lang ... >   ```
        blockquote_pattern = rf"""
            (                           # Entire blockquote code block
                (?:^>.*\n)*?            # Optional leading lines
                ^>\s*```{lang}\s*\n     # Opening with optional spaces
                (?:^>.*\n)*?            # Content lines
                ^>\s*```\s*$            # Closing with optional spaces
            )
        """
        markdown_text = re.sub(
            blockquote_pattern, "", markdown_text, flags=re.MULTILINE | re.VERBOSE
        )

        # Normal fenced code block: ```lang ... ```
        fenced_pattern = rf"^```{lang}\s*\n.*?\n```[ \t]*\n?"
        markdown_text = re.sub(
            fenced_pattern, "", markdown_text, flags=re.DOTALL | re.MULTILINE
        )

    if remove_all_fenced:
        # Generic fenced blocks: ```...```, with or without language
        generic_pattern = r"^```[^\n]*\n.*?\n```[ \t]*\n?"
        markdown_text = re.sub(
            generic_pattern, "", markdown_text, flags=re.DOTALL | re.MULTILINE
        )

    return markdown_text


def chunk_markdown_blocks(markdown_text: str, max_chunk_chars: int = 1000):
    # Parse the markdown text into an AST
    markdown_parser = mistune.create_markdown(renderer=None)
    ast = markdown_parser(markdown_text)

    renderer = MarkdownRenderer()
    chunks = []
    current_chunk = ""
    chunk_start = 0
    char_cursor = 0

    for node in ast:
        block = renderer([node], state=BlockState()) + "\n\n"

        block_len = len(block)
        if not current_chunk:
            # start of a new chunk
            chunk_start = char_cursor
        if len(current_chunk) + block_len <= max_chunk_chars:
            current_chunk += block
        else:
            # flush existing chunk
            chunks.append(
                {
                    "text": current_chunk,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(current_chunk),
                }
            )
            # start new chunk
            chunk_start = char_cursor
            current_chunk = block
        char_cursor += block_len

    if current_chunk:
        chunks.append(
            {
                "text": current_chunk,
                "start_char": chunk_start,
                "end_char": chunk_start + len(current_chunk),
            }
        )
    return chunks


def get_client(path: str):
    return chromadb.PersistentClient(path=path)


def get_collection(client, name: str):
    return client.get_or_create_collection(name=name)


def create_batches(
    ids: IDs,
    embeddings: Optional[Embeddings] = None,
    metadatas: Optional[Metadatas] = None,
    documents: Optional[Documents] = None,
    batch_size: int = 10,
) -> List[Tuple[IDs, Embeddings, Optional[Metadatas], Optional[Documents]]]:
    _batches: List[
        Tuple[IDs, Embeddings, Optional[Metadatas], Optional[Documents]]
    ] = []
    if len(ids) > batch_size:
        # create split batches
        for i in range(0, len(ids), batch_size):
            _batches.append(
                (  # type: ignore
                    ids[i : i + batch_size],
                    embeddings[i : i + batch_size] if embeddings else None,
                    metadatas[i : i + batch_size] if metadatas else None,
                    documents[i : i + batch_size] if documents else None,
                )
            )
    else:
        _batches.append((ids, embeddings, metadatas, documents))  # type: ignore
    return _batches


def get_md_checkboxes(markdown_str: str):
    checkbox_pattern = re.compile(r"- \[\s?[xX]?\s?\] .+", re.MULTILINE)
    checkboxes = checkbox_pattern.findall(markdown_str)
    return checkboxes


def file_obj(md_file: Path):
    import re

    import frontmatter

    raw_text = md_file.read_text(encoding="utf-8", errors="ignore")
    # Strip YAML frontmatter only
    post = frontmatter.loads(raw_text)
    content = post.content
    tasks = get_md_checkboxes(content)
    # Use AST-based chunking BEFORE removing code blocks
    chunks = chunk_markdown_blocks(content)
    # Remove any chunks containing fenced code blocks
    filtered_chunks = []
    for chunk in chunks:
        if re.search(r"^```", chunk["text"], flags=re.MULTILINE):
            continue
        filtered_chunks.append(chunk)
    chunks = filtered_chunks
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            {
                "id": f"{md_file.absolute().as_uri()}_chunk_{i}",
                "document": chunk["text"],
                "metadata": {
                    "title": md_file.stem,
                    "path": str(md_file.absolute()),
                    "tasks": "\n".join(tasks),
                    "chunk_index": i,
                    "mtime": md_file.stat().st_mtime,
                },
            }
        )
    return docs


def _index_md(
    obsidian_root: Path,
    vault: str,
    db: str,
    collection: str,
    batch_size: int,
    force: bool = False,
):
    vault_path = obsidian_root / vault
    md_files = list(vault_path.glob("**/*.md"))
    # exclude files with a _underscore, they are templates
    md_files = [f for f in md_files if not f.stem.startswith("_")]
    if not md_files:
        raise ValueError(f"No markdown files found in {vault_path}")

    # Set up client and collection before processing files
    client = get_client(db)
    collection_obj = get_collection(client, collection)

    existing = collection_obj.get(include=["metadatas"], limit=999999)

    stored_mtimes = defaultdict(float)

    for meta in existing["metadatas"]:
        if "path" in meta and "mtime" in meta:
            path = meta["path"]
            stored_mtimes[path] = max(stored_mtimes[path], float(meta["mtime"]))
    # debug, print the difference between the stored mtimes and the current mtimes

    docs = []
    for p in tqdm(md_files):
        current_mtime = p.stat().st_mtime
        if (
            abs(stored_mtimes.get(str(p.absolute()), 0) - current_mtime) > 1e-4
            and not force
        ):
            continue
        docs.extend(file_obj(p))

    if not docs:
        print("No documents to update.")
        return

    # Pre-generate embeddings using the MiniLM model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["document"] for d in docs]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents"):
        embeddings.extend(
            model.encode(
                texts[i : i + batch_size],
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )
    ids = [d["id"] for d in docs]
    documents = [d["document"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    batches = create_batches(
        ids=ids,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        metadatas=metadatas,
        documents=documents,
        batch_size=batch_size,
    )

    for batch in tqdm(batches, desc="Upserting documents"):
        collection_obj.upsert(
            ids=batch[0],
            embeddings=batch[1],
            metadatas=batch[2],
            documents=batch[3],
        )

    print(f"Upserted {len(docs)} documents into '{collection}'")


@app.command("index")
def index_md(
    obsidian_root: Path = Path("~/obsidianvaults/").expanduser(),
    vault: str = typer.Option(..., help="Path to markdown vault"),
    db: str = "chroma.db",
    collection: Optional[str] = None,
    batch_size: int = 10,
    force: bool = False,
):
    _index_md(obsidian_root, vault, db, collection or vault, batch_size, force)


def _query_chroma(
    query: str,
    n: int,
    db: str,
    collection: str,
    where: Optional[str],
):
    client = get_client(db)
    collection_obj = get_collection(client, collection)

    where_dict = json.loads(where) if where else None

    result = collection_obj.query(
        query_texts=[query],
        n_results=n,
        where=where_dict,
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    for doc, metadata in zip(documents, metadatas):
        # debug: show original chunk location
        start_line = metadata.get("chunk_start_line", {})
        end_line = metadata.get("chunk_end_line", {})
        start_col = metadata.get("chunk_start_col", {})
        end_col = metadata.get("chunk_end_col", {})

        file_uri = f"{Path(metadata['path']).as_posix()}:{start_line}:{start_col}"

        console.print(f"# {metadata['title']}")
        console.print(file_uri)
        # build a file uri so that we can open the file in the editor e.g <file_abspath>:<line_start> \
        console.print(
            f"[debug] chunk at {start_line}:{start_col} → {end_line}:{end_col}"
        )
        console.print(Markdown(doc))
        console.print(f"\n\n")
    return result


@app.command("query")
def query_chroma(
    query: str = typer.Argument(...),
    n: int = 5,
    db: str = "chroma.db",
    collection: str = "vault_docs",
    where: Optional[str] = typer.Option(
        None, help='Metadata filter in JSON format (e.g., \'{"title": "foo"}\')'
    ),
):
    _query_chroma(query, n, db, collection, where)


if __name__ == "__main__":
    app()
