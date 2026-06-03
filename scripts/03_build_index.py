#!/usr/bin/env python3
"""Build LanceDB index with parent-child chunks and embeddings (Gemini or OpenRouter)."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import lancedb
import pyarrow as pa
import tiktoken
from rich.console import Console
from rich.progress import Progress

from shared import EMBED_DIMS, ROOT, make_embed_fn

MARKDOWN_DIR = ROOT / "markdown"
MAPS_DIR = ROOT / "maps"
INDEX_DIR = ROOT / "index"

console = Console()
PARENT_TOKENS = 800
CHILD_TOKENS = 256
OVERLAP_TOKENS = 50

enc = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(enc.encode(text))




def split_chapters(text: str) -> list[dict]:
    """Split markdown by H1 headers into chapters."""
    lines = text.split("\n")
    chapters = []
    current_title = None
    current_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# "):
            if current_title is not None or current_lines:
                chapters.append({
                    "title": current_title or "Untitled",
                    "text": "\n".join(current_lines),
                    "char_start": current_start,
                })
            current_title = line[2:].strip()
            current_lines = [line]
            current_start = sum(len(l) + 1 for l in lines[:i])
        else:
            current_lines.append(line)

    if current_title is not None or current_lines:
        chapters.append({
            "title": current_title or "Untitled",
            "text": "\n".join(current_lines),
            "char_start": current_start,
        })

    return chapters


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline separated)."""
    paras = re.split(r"\n\n+", text.strip())
    return [p.strip() for p in paras if p.strip()]


def chunk_by_tokens(paragraphs: list[str], target_tokens: int, overlap_tokens: int = 0) -> list[dict]:
    """Group paragraphs into chunks of approximately target_tokens."""
    chunks = []
    current_paras = []
    current_tokens = 0
    char_offset = 0

    para_positions = []
    pos = 0
    for p in paragraphs:
        para_positions.append(pos)
        pos += len(p) + 2

    for i, para in enumerate(paragraphs):
        para_tokens = token_len(para)

        if current_tokens + para_tokens > target_tokens and current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({"text": chunk_text, "char_start_offset": char_offset})

            overlap_paras = []
            overlap_count = 0
            for prev_para in reversed(current_paras):
                prev_tokens = token_len(prev_para)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_paras.insert(0, prev_para)
                overlap_count += prev_tokens

            current_paras = overlap_paras + [para]
            current_tokens = overlap_count + para_tokens
            char_offset = para_positions[i] - sum(len(p) + 2 for p in overlap_paras)
        else:
            if not current_paras:
                char_offset = para_positions[i]
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunk_text = "\n\n".join(current_paras)
        chunks.append({"text": chunk_text, "char_start_offset": char_offset})

    return chunks


def get_book_metadata(basename: str) -> dict:
    """Load author/title from JSON map if available, else infer from filename."""
    map_path = MAPS_DIR / f"{basename}.json"
    if map_path.exists():
        data = json.loads(map_path.read_text(encoding="utf-8"))
        return {
            "author": data.get("author", "Unknown"),
            "title": data.get("title", basename),
        }
    parts = basename.split("-", 1)
    author = parts[0].replace("-", " ").title() if parts else "Unknown"
    title = parts[1].replace("-", " ").title() if len(parts) > 1 else basename
    return {"author": author, "title": title}


PARENT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("book_id", pa.string()),
    pa.field("author", pa.string()),
    pa.field("title", pa.string()),
    pa.field("chapter_number", pa.int32()),
    pa.field("chapter_title", pa.string()),
    pa.field("parent_index", pa.int32()),
    pa.field("char_start", pa.int64()),
    pa.field("char_end", pa.int64()),
    pa.field("text", pa.string()),
])

CHILD_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("parent_id", pa.string()),
    pa.field("book_id", pa.string()),
    pa.field("author", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("context_text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIMS)),
])


def prepare_book(md_path: Path) -> dict:
    """Chunk a book into parent/child records (no embedding yet). Pure CPU work."""
    basename = md_path.stem
    text = md_path.read_text(encoding="utf-8")
    meta = get_book_metadata(basename)
    chapters = split_chapters(text)

    parent_records = []
    child_records = []
    texts_to_embed = []

    for ch_idx, chapter in enumerate(chapters):
        ch_num = ch_idx + 1
        ch_title = chapter["title"]
        ch_text = chapter["text"]
        ch_start = chapter["char_start"]
        paragraphs = split_paragraphs(ch_text)

        if not paragraphs:
            continue

        parent_chunks = chunk_by_tokens(paragraphs, PARENT_TOKENS, overlap_tokens=0)

        for p_idx, parent in enumerate(parent_chunks):
            parent_id = f"{basename}::ch{ch_num}::parent{p_idx}"
            parent_char_start = ch_start + max(0, parent["char_start_offset"])
            parent_char_end = parent_char_start + len(parent["text"])

            parent_records.append({
                "id": parent_id,
                "book_id": basename,
                "author": meta["author"],
                "title": meta["title"],
                "chapter_number": ch_num,
                "chapter_title": ch_title,
                "parent_index": p_idx,
                "char_start": parent_char_start,
                "char_end": parent_char_end,
                "text": parent["text"],
            })

            child_paragraphs = split_paragraphs(parent["text"])
            child_chunks = chunk_by_tokens(child_paragraphs, CHILD_TOKENS, overlap_tokens=OVERLAP_TOKENS)

            context_prefix = f"Author: {meta['author']} | Title: {meta['title']} | Chapter: {ch_title}\n\n"

            for c_idx, child in enumerate(child_chunks):
                child_id = f"{parent_id}::child{c_idx}"
                context_text = context_prefix + child["text"]
                child_records.append({
                    "id": child_id,
                    "parent_id": parent_id,
                    "book_id": basename,
                    "author": meta["author"],
                    "chunk_index": c_idx,
                    "text": child["text"],
                    "context_text": context_text,
                })
                texts_to_embed.append(context_text)

    return {
        "basename": basename,
        "parent_records": parent_records,
        "child_records": child_records,
        "texts_to_embed": texts_to_embed,
    }


def embed_and_store(
    embed_batch,
    parents_table,
    children_table,
    book_data: dict,
    db_lock: threading.Lock,
) -> dict:
    """Embed all child chunks for a book and write to DB. Returns stats."""
    basename = book_data["basename"]
    parent_records = book_data["parent_records"]
    child_records = book_data["child_records"]
    texts_to_embed = book_data["texts_to_embed"]

    # Embed in batches of 100
    all_vectors = []
    for batch_start in range(0, len(texts_to_embed), 100):
        batch = texts_to_embed[batch_start:batch_start + 100]
        vectors = embed_batch(batch)
        all_vectors.extend(vectors)

    # Attach vectors to child records
    for rec, vec in zip(child_records, all_vectors):
        rec["vector"] = vec

    # Bulk write to DB (locked for thread safety)
    with db_lock:
        if parent_records:
            parents_table.add(parent_records)
        if child_records:
            children_table.add(child_records)

    return {
        "basename": basename,
        "parents": len(parent_records),
        "children": len(child_records),
    }


@click.command()
@click.option("--force", is_flag=True, help="Re-index even if book already indexed")
@click.option("--provider", type=click.Choice(["gemini", "openrouter"]), default="gemini",
              help="Embedding provider (gemini=free, openrouter=paid fallback)")
@click.option("--workers", default=4, help="Number of parallel workers for embedding")
@click.option("--book", default=None, help="Index a single book by basename")
def main(force: bool, provider: str, workers: int, book: str):
    """Build LanceDB index from Markdown files."""
    embed_batch, _ = make_embed_fn(provider)
    console.print(f"Using [bold]{provider}[/bold] for embeddings ({EMBED_DIMS} dims, {workers} workers)")

    db = lancedb.connect(str(INDEX_DIR))

    # Create or open tables
    table_names = db.table_names()
    if "parents" in table_names:
        parents_table = db.open_table("parents")
    else:
        parents_table = db.create_table("parents", schema=PARENT_SCHEMA)

    if "children" in table_names:
        children_table = db.open_table("children")
    else:
        children_table = db.create_table("children", schema=CHILD_SCHEMA)

    # Find already-indexed books
    indexed_books = set()
    if not force:
        try:
            existing = parents_table.search().select(["book_id"]).limit(100_000).to_list()
            indexed_books = {r["book_id"] for r in existing}
        except Exception:
            pass

    if book:
        md_files = [MARKDOWN_DIR / f"{book}.md"]
        if not md_files[0].exists():
            raise click.ClickException(f"Not found: {md_files[0]}")
    else:
        md_files = sorted(MARKDOWN_DIR.glob("*.md"))

    if not md_files:
        console.print("[yellow]No .md files found in markdown/[/yellow]")
        return

    # Filter to books that need processing
    to_process = []
    skipped = 0
    for md_path in md_files:
        basename = md_path.stem
        if basename in indexed_books and not force:
            console.print(f"  [dim]Skip (indexed):[/dim] {basename}")
            skipped += 1
        else:
            to_process.append(md_path)

    if not to_process:
        console.print("[green]All books already indexed.[/green]")
        return

    # Phase 1: Chunk all books (fast, CPU-only)
    console.print(f"\nChunking {len(to_process)} books...")
    all_book_data = []
    for md_path in to_process:
        book_data = prepare_book(md_path)
        all_book_data.append(book_data)
        console.print(f"  {book_data['basename']}: {len(book_data['parent_records'])} parents, {len(book_data['child_records'])} children")

    total_children = sum(len(bd["child_records"]) for bd in all_book_data)
    console.print(f"Total chunks to embed: {total_children:,}")

    # Phase 2: Embed + store in parallel
    start_time = time.time()
    db_lock = threading.Lock()
    processed = 0
    errors = 0
    total_p = 0
    total_c = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Embedding & indexing...", total=len(all_book_data))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(embed_and_store, embed_batch, parents_table, children_table, bd, db_lock): bd["basename"]
                for bd in all_book_data
            }

            for future in as_completed(futures):
                basename = futures[future]
                try:
                    stats = future.result()
                    total_p += stats["parents"]
                    total_c += stats["children"]
                    processed += 1
                    progress.console.print(f"  [green]Done:[/green] {basename} ({stats['parents']} parents, {stats['children']} children)")
                except Exception as e:
                    progress.console.print(f"  [red]Error ({basename}):[/red] {e}")
                    errors += 1

                progress.advance(task)

    # Build FTS index on parents table for BM25 search
    if processed > 0:
        console.print("Building full-text search index...")
        try:
            parents_table.create_fts_index("text", replace=True)
            console.print("[green]FTS index built.[/green]")
        except Exception as e:
            console.print(f"[yellow]FTS index warning:[/yellow] {e}")

    elapsed = time.time() - start_time
    console.print(
        f"\n[green]Done.[/green] "
        f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}\n"
        f"Parents: {total_p:,}, Children: {total_c:,}\n"
        f"Time: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
