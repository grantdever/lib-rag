#!/usr/bin/env python3
"""Query the book corpus index with hybrid search."""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import click
import lancedb
from rich.console import Console

from shared import ROOT, make_embed_fn

INDEX_DIR = ROOT / "index"
# Obsidian vault queries directory (configurable via OBSIDIAN_VAULT env var)
_vault = Path(os.environ.get("OBSIDIAN_VAULT", str(Path.home() / "obsidian" / "vault")))
VAULT_QUERIES = _vault / "Queries"

console = Console()

RRF_K = 60  # Reciprocal rank fusion constant


def _escape_sql(value: str) -> str:
    """Escape single quotes for LanceDB SQL filter strings."""
    return value.replace("'", "''")


def _apply_filters(query, filters: dict):
    """Apply book_id and author filters to a LanceDB query."""
    if filters.get("book_id"):
        q = query.where(f"book_id = '{_escape_sql(filters['book_id'])}'")
    else:
        q = query
    if filters.get("author"):
        q = q.where(f"lower(author) LIKE '%{_escape_sql(filters['author'].lower())}%'")
    return q


def vector_search(children_table, query_vec: list[float], top_k: int, filters: dict) -> list[dict]:
    """Search children table by vector similarity."""
    q = children_table.search(query_vec).limit(top_k)
    q = _apply_filters(q, filters)
    return q.to_list()


def fts_search(parents_table, query_text: str, top_k: int, filters: dict) -> list[dict]:
    """Search parents table by full-text search (BM25)."""
    try:
        q = parents_table.search(query_text, query_type="fts").limit(top_k)
        q = _apply_filters(q, filters)
        return q.to_list()
    except Exception:
        return []


def reciprocal_rank_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    parents_table,
    top_k: int,
) -> list[dict]:
    """Merge vector (child) and FTS (parent) results via RRF, return parent records."""
    parent_scores = {}

    # Score from vector results (child → parent mapping)
    for rank, child in enumerate(vector_results):
        pid = child.get("parent_id", "")
        rrf_score = 1.0 / (RRF_K + rank + 1)
        parent_scores[pid] = parent_scores.get(pid, 0) + rrf_score

    # Score from FTS results (already parent records)
    for rank, parent in enumerate(fts_results):
        pid = parent.get("id", "")
        rrf_score = 1.0 / (RRF_K + rank + 1)
        parent_scores[pid] = parent_scores.get(pid, 0) + rrf_score

    # Sort by combined score
    sorted_pids = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Fetch parent records
    # Build a lookup from FTS results first
    fts_lookup = {r["id"]: r for r in fts_results if "id" in r}

    results = []
    for pid, score in sorted_pids:
        if pid in fts_lookup:
            record = dict(fts_lookup[pid])
            record["_rrf_score"] = score
            results.append(record)
        else:
            # Fetch from parents table
            try:
                rows = parents_table.search().where(f"id = '{_escape_sql(pid)}'").limit(1).to_list()
                if rows:
                    record = dict(rows[0])
                    record["_rrf_score"] = score
                    results.append(record)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to fetch parent {pid}: {e}[/yellow]", highlight=False)

    return results


def format_pretty(results: list[dict]) -> str:
    """Format results for terminal display."""
    lines = []
    for i, r in enumerate(results):
        score = r.get("_rrf_score") or r.get("_score", 0)
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        book = r.get("book_id", "?")
        chapter = r.get("chapter_title", "?")
        author = r.get("author", "?")
        text_preview = r.get("text", "")[:300].replace("\n", " ")

        lines.append(f"[{score_str}] {author} — {r.get('title', book)}, Ch. {r.get('chapter_number', '?')}: {chapter}")
        lines.append(f"  {text_preview}...")
        lines.append(f"  → markdown/{book}.md")
        lines.append("")
    return "\n".join(lines)


def format_json(results: list[dict]) -> str:
    """Format results as JSON for Claude Code."""
    clean = []
    for r in results:
        rec = {
            "score": r.get("_rrf_score") or r.get("_score", 0),
            "book_id": r.get("book_id"),
            "author": r.get("author"),
            "title": r.get("title"),
            "chapter_number": r.get("chapter_number"),
            "chapter_title": r.get("chapter_title"),
            "text": r.get("text"),
            "char_start": r.get("char_start"),
            "char_end": r.get("char_end"),
            "parent_id": r.get("id"),
        }
        clean.append(rec)
    return json.dumps(clean, indent=2, ensure_ascii=False)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60].rstrip("-")


def format_obsidian(query: str, results: list[dict]) -> tuple[str, str]:
    """Format results as Obsidian markdown. Returns (filename, content)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")
    slug = slugify(query)
    filename = f"{slug}-{now.strftime('%Y%m%d')}"

    lines = [
        f"# Query: {query}",
        f"*Generated {date_str} | {len(results)} results*",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results):
        score = r.get("_rrf_score") or r.get("_score", 0)
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        book_id = r.get("book_id", "?")
        chapter = r.get("chapter_title", "?")
        author = r.get("author", "?")
        title = r.get("title", book_id)
        ch_num = r.get("chapter_number", "?")
        text = r.get("text", "").strip()

        # Take first ~500 chars as preview
        preview = text[:500]
        if len(text) > 500:
            preview += "..."

        # Clean chapter title for wikilink anchor (Obsidian style)
        anchor = chapter.replace(" ", " ")

        lines.extend([
            f"## {i+1}. ({score_str}) {author} — *{title}*, Ch. {ch_num}",
            "",
            f"> {preview}",
            "",
            f"Source: [[{book_id}#{anchor}]]",
            "",
            "---",
            "",
        ])

    return filename, "\n".join(lines)


@click.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--book-filter", default=None, help="Filter by book_id")
@click.option("--author-filter", default=None, help="Filter by author name (partial match)")
@click.option("--mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="hybrid")
@click.option("--format", "fmt", type=click.Choice(["pretty", "json", "obsidian"]), default="pretty")
@click.option("--provider", type=click.Choice(["gemini", "openrouter"]), default="openrouter",
              help="Embedding provider for query vector")
def main(query: str, top_k: int, book_filter: str, author_filter: str, mode: str, fmt: str, provider: str):
    """Query the book corpus index."""
    db = lancedb.connect(str(INDEX_DIR))

    table_names = db.table_names()
    if "parents" not in table_names or "children" not in table_names:
        raise click.ClickException("Index not built yet. Run 03_build_index.py first.")

    parents_table = db.open_table("parents")
    children_table = db.open_table("children")

    filters = {"book_id": book_filter, "author": author_filter}
    fetch_k = top_k * 2  # Over-fetch for RRF merging

    if mode in ("hybrid", "semantic"):
        _, embed_query = make_embed_fn(provider)
        query_vec = embed_query(query)
        vector_results = vector_search(children_table, query_vec, fetch_k, filters)
    else:
        vector_results = []

    if mode in ("hybrid", "keyword"):
        fts_results = fts_search(parents_table, query, fetch_k, filters)
    else:
        fts_results = []

    if mode == "hybrid":
        results = reciprocal_rank_fusion(vector_results, fts_results, parents_table, top_k)
    elif mode == "semantic":
        # Map child results to parents
        parent_ids_seen = set()
        results = []
        for child in vector_results:
            pid = child.get("parent_id", "")
            if pid in parent_ids_seen:
                continue
            parent_ids_seen.add(pid)
            try:
                rows = parents_table.search().where(f"id = '{_escape_sql(pid)}'").limit(1).to_list()
                if rows:
                    record = dict(rows[0])
                    record["_score"] = child.get("_distance", 0)
                    results.append(record)
            except Exception as e:
                console.print(f"[yellow]Warning: failed to fetch parent {pid}: {e}[/yellow]", highlight=False)
            if len(results) >= top_k:
                break
    else:  # keyword
        results = fts_results[:top_k]
        for r in results:
            r["_score"] = r.get("_score", 0)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if fmt == "pretty":
        console.print(format_pretty(results))
    elif fmt == "json":
        print(format_json(results))
    elif fmt == "obsidian":
        filename, content = format_obsidian(query, results)
        VAULT_QUERIES.mkdir(parents=True, exist_ok=True)
        out_path = VAULT_QUERIES / f"{filename}.md"
        out_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Written:[/green] {out_path}")

        # Open in Obsidian
        vault_name = _vault.name
        obsidian_uri = f"obsidian://open?vault={quote(vault_name)}&file={quote(f'Queries/{filename}')}"
        subprocess.run(["open", obsidian_uri], check=False)
        console.print(f"[green]Opened in Obsidian[/green]")


if __name__ == "__main__":
    main()
