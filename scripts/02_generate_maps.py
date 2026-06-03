#!/usr/bin/env python3
"""Generate JSON book maps from Markdown files via LLM (OpenRouter or Gemini).

For long books (>100K chars), splits by chapter headers and processes each
chapter in parallel, then assembles the final map.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from shared import ROOT, is_retryable, get_llm_client

MARKDOWN_DIR = ROOT / "markdown"
MAPS_DIR = ROOT / "maps"
console = Console()

LONG_BOOK_THRESHOLD = 100_000  # characters

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a scholarly assistant that produces structured JSON summaries of books.

Given the full text of a book, produce a JSON object matching this schema EXACTLY. Output ONLY valid JSON.

Schema:
{
  "id": "<basename from filename>",
  "author": "<inferred from content or filename>",
  "title": "<inferred from content or filename>",
  "year": <publication year as integer if discoverable, else null>,
  "source_epub": "source/<basename>.epub",
  "source_md": "markdown/<basename>.md",
  "summary": "<3-5 sentences summarizing the whole book>",
  "key_themes": ["<5-10 short thematic tags>"],
  "chapters": [
    {
      "number": <int>,
      "title": "<chapter title from heading>",
      "summary": "<2-3 sentences>",
      "key_arguments": ["<substantive claims, not vague descriptions>"],
      "themes": ["<2-5 short lowercase thematic tags>"]
    }
  ]
}

Rules:
- Infer author and title from the text content and filename
- For year, use publication year if mentioned; otherwise null
- Chapter numbers should be sequential starting from 1
- key_arguments should be substantive claims, not vague descriptions
- themes should be short lowercase tags
- Output ONLY the JSON object. No preamble, no explanation."""

BOOK_META_PROMPT = """You are a scholarly assistant that produces structured JSON metadata for books.

Given the opening text of a book and a list of its chapter headings, produce a JSON object with ONLY the book-level metadata. Output ONLY valid JSON.

Schema:
{
  "author": "<inferred from content or filename>",
  "title": "<inferred from content or filename>",
  "year": <publication year as integer if discoverable, else null>,
  "summary": "<3-5 sentences summarizing the whole book based on the chapter titles and opening text>",
  "key_themes": ["<5-10 short thematic tags>"]
}

Rules:
- Infer author and title from the text content and filename
- For year, use publication year if mentioned; otherwise null
- key_themes should be short lowercase tags
- Output ONLY the JSON object. No preamble, no explanation."""

CHAPTER_PROMPT = """You are a scholarly assistant that produces structured JSON summaries of book chapters.

Given the text of a single chapter, produce a JSON object. Output ONLY valid JSON.

Schema:
{
  "number": <sequential chapter number provided below>,
  "title": "<chapter title from heading>",
  "summary": "<2-3 sentences summarizing this chapter>",
  "key_arguments": ["<3-5 substantive claims, not vague descriptions>"],
  "themes": ["<2-5 short lowercase thematic tags>"]
}

Rules:
- Use the chapter number provided in the user message
- key_arguments should be substantive claims, not vague descriptions
- themes should be short lowercase tags
- Output ONLY the JSON object. No preamble, no explanation."""

MAP_PROMPT_TEMPLATE = """Filename: {basename}

Below is the full text of the book. Produce the JSON map.

---
{text}
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(5),
)
def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 4000,
) -> tuple[dict, dict]:
    """Call the LLM and return (parsed_json, usage_dict)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")

    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    return data, usage


# ---------------------------------------------------------------------------
# Chapter splitting
# ---------------------------------------------------------------------------

def clean_heading(heading: str) -> str:
    """Remove markdown span syntax and attributes from a heading."""
    cleaned = re.sub(r"\{[^}]*\}", "", heading)
    cleaned = re.sub(r"\[([^\]]*)\]", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_chapter_level(text: str) -> int:
    """Detect whether chapters are at H1 (#) or H2 (##) level.

    Returns 1 for H1, 2 for H2.

    Heuristic: if H2 headers contain "chapter" or sequential numbering
    and H1 headers look like part dividers, use H2.
    """
    h1_headers = re.findall(r"^# (.+)$", text, re.MULTILINE)
    h2_headers = re.findall(r"^## (.+)$", text, re.MULTILINE)

    if not h2_headers:
        return 1

    chapter_pat = re.compile(r"chapter|ch\.\s*\d|^\s*\d+[\.\):]", re.IGNORECASE)
    h2_chapter_count = sum(1 for h in h2_headers if chapter_pat.search(clean_heading(h)))

    part_pat = re.compile(
        r"^[IVX]+[\.\s]|part\s|preliminar|appendix|glossary|reference|notes\s+for",
        re.IGNORECASE,
    )
    h1_part_count = sum(1 for h in h1_headers if part_pat.search(clean_heading(h)))

    if h2_chapter_count >= len(h2_headers) * 0.4 and h1_part_count >= len(h1_headers) * 0.3:
        return 2

    if len(h2_headers) > len(h1_headers) * 2 and h2_chapter_count >= 3:
        return 2

    return 1


def split_into_chapters(text: str, level: int) -> list[dict]:
    """Split markdown by heading level into chapters.

    Returns list of {"heading": str, "text": str} dicts.
    Filters out empty/spacer headers from EPUB conversion artifacts.
    """
    prefix = "#" * level + " "
    lines = text.split("\n")
    chapters: list[dict] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith(prefix) and not line.startswith("#" * (level + 1) + " "):
            if current_heading is not None:
                chapters.append({"heading": current_heading, "text": "\n".join(current_lines)})
            current_heading = line[len(prefix):].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_heading is not None:
        chapters.append({"heading": current_heading, "text": "\n".join(current_lines)})

    # Filter out empty/spacer headers (body must have >200 chars beyond the heading line)
    real = []
    for ch in chapters:
        clean = clean_heading(ch["heading"])
        body_len = len(ch["text"]) - len(ch["text"].split("\n")[0]) if ch["text"] else 0
        if clean and body_len > 200:
            real.append(ch)

    return real


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def validate_map(data: dict, basename: str) -> list[str]:
    """Return list of warnings for missing/bad fields."""
    warnings = []
    for field in ("author", "title", "summary", "key_themes", "chapters"):
        if field not in data:
            warnings.append(f"missing '{field}'")
    if "chapters" in data:
        if not isinstance(data["chapters"], list):
            warnings.append("'chapters' is not a list")
        elif len(data["chapters"]) == 0:
            warnings.append("'chapters' is empty")
        else:
            for i, ch in enumerate(data["chapters"]):
                if not ch.get("title"):
                    warnings.append(f"chapter {i+1} missing title")
                if not ch.get("key_arguments"):
                    warnings.append(f"chapter {i+1} missing key_arguments")
    return warnings


def process_short_book(client: OpenAI, model: str, basename: str, text: str) -> tuple[dict, dict]:
    """Process a short book with a single LLM call."""
    user_content = MAP_PROMPT_TEMPLATE.format(basename=basename, text=text)
    return call_llm(client, model, SYSTEM_PROMPT, user_content, max_tokens=12000)


def process_long_book(
    client: OpenAI,
    model: str,
    basename: str,
    text: str,
    workers: int,
    verbose: bool,
) -> tuple[dict, dict]:
    """Process a long book by splitting into chapters and processing in parallel."""
    level = detect_chapter_level(text)
    chapters = split_into_chapters(text, level)

    if verbose:
        console.print(f"  [dim]Heading level: H{level}, {len(chapters)} chapters[/dim]")
        for i, ch in enumerate(chapters):
            console.print(f"    [dim]{i+1}. {clean_heading(ch['heading'])[:80]}[/dim]")

    if not chapters:
        console.print("  [yellow]No chapters detected, falling back to truncated single-call[/yellow]")
        truncated = text[:LONG_BOOK_THRESHOLD] + "\n\n[...truncated...]"
        return process_short_book(client, model, basename, truncated)

    total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # Phase 1: Book-level metadata
    heading_list = "\n".join(
        f"  {i+1}. {clean_heading(ch['heading'])}" for i, ch in enumerate(chapters)
    )
    meta_prompt = (
        f"Filename: {basename}\n\n"
        f"Opening text (first 10,000 chars):\n---\n{text[:10_000]}\n---\n\n"
        f"Full chapter listing:\n{heading_list}\n\n"
        f"Produce the book-level metadata JSON."
    )

    if verbose:
        console.print("  [dim]Phase 1: book metadata...[/dim]")

    meta_data, meta_usage = call_llm(client, model, BOOK_META_PROMPT, meta_prompt, max_tokens=2000)
    total_usage["prompt_tokens"] += meta_usage.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += meta_usage.get("completion_tokens", 0)

    # Phase 2: Per-chapter detail in parallel
    if verbose:
        console.print(f"  [dim]Phase 2: {len(chapters)} chapters ({workers} workers)...[/dim]")

    chapter_results: dict[int, dict] = {}

    def map_one_chapter(ch_num: int, ch_text: str, ch_heading: str) -> tuple[int, dict, dict]:
        user_content = (
            f"Chapter number: {ch_num}\n"
            f"Chapter heading: {clean_heading(ch_heading)}\n\n"
            f"Chapter text:\n---\n{ch_text}\n---"
        )
        data, usage = call_llm(client, model, CHAPTER_PROMPT, user_content, max_tokens=2000)
        data["number"] = ch_num
        return ch_num, data, usage

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(map_one_chapter, i + 1, ch["text"], ch["heading"]): i + 1
            for i, ch in enumerate(chapters)
        }
        for future in as_completed(futures):
            ch_num = futures[future]
            try:
                num, data, usage = future.result()
                chapter_results[num] = data
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                if verbose:
                    console.print(f"    [green]Ch {num}:[/green] {clean_heading(chapters[num-1]['heading'])[:60]}")
            except Exception as e:
                console.print(f"    [red]Ch {ch_num} error:[/red] {e}")
                chapter_results[ch_num] = {
                    "number": ch_num,
                    "title": clean_heading(chapters[ch_num - 1]["heading"]),
                    "summary": f"[Error: {e}]",
                    "key_arguments": [],
                    "themes": [],
                }

    # Assemble
    assembled = {
        "id": basename,
        "author": meta_data.get("author", "Unknown"),
        "title": meta_data.get("title", basename),
        "year": meta_data.get("year"),
        "source_epub": f"source/{basename}.epub",
        "source_md": f"markdown/{basename}.md",
        "summary": meta_data.get("summary", ""),
        "key_themes": meta_data.get("key_themes", []),
        "chapters": [chapter_results[i] for i in sorted(chapter_results.keys())],
    }

    return assembled, total_usage


def process_one(
    client: OpenAI,
    model: str,
    md_path: Path,
    workers: int,
    verbose: bool,
) -> dict:
    """Process a single book. Returns result dict with status, usage, warnings."""
    basename = md_path.stem
    text = md_path.read_text(encoding="utf-8")

    if len(text) > LONG_BOOK_THRESHOLD:
        if verbose:
            console.print(f"  [bold]{basename}[/bold]: {len(text):,} chars — chapter-split mode")
        data, usage = process_long_book(client, model, basename, text, workers, verbose)
    else:
        data, usage = process_short_book(client, model, basename, text)

    data.setdefault("id", basename)
    data.setdefault("source_epub", f"source/{basename}.epub")
    data.setdefault("source_md", f"markdown/{basename}.md")

    warnings = validate_map(data, basename)

    map_path = MAPS_DIR / f"{basename}.json"
    map_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"basename": basename, "usage": usage, "warnings": warnings}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--provider", type=click.Choice(["openrouter", "gemini"]), default="openrouter")
@click.option("--force", is_flag=True, help="Regenerate existing maps")
@click.option("--workers", default=4, help="Number of parallel workers for chapter-split mode")
@click.option("--book", default=None, help="Process a single book by basename")
@click.option("--verbose", is_flag=True, help="Show per-chapter progress for long books")
def main(provider: str, force: bool, workers: int, book: str, verbose: bool):
    """Generate JSON book maps from Markdown files."""
    client, model = get_llm_client(provider)
    console.print(f"Using [bold]{provider}[/bold] ({model})")

    if book:
        md_files = [MARKDOWN_DIR / f"{book}.md"]
        if not md_files[0].exists():
            raise click.ClickException(f"Not found: {md_files[0]}")
    else:
        md_files = sorted(MARKDOWN_DIR.glob("*.md"))

    if not md_files:
        console.print("[yellow]No .md files found in markdown/[/yellow]")
        return

    to_process = []
    skipped = 0
    for md_path in md_files:
        basename = md_path.stem
        map_path = MAPS_DIR / f"{basename}.json"
        if map_path.exists() and not force:
            console.print(f"  [dim]Skip (exists):[/dim] {basename}")
            skipped += 1
        else:
            to_process.append(md_path)

    if not to_process:
        console.print("[green]All maps up to date.[/green]")
        return

    total_prompt = 0
    total_completion = 0
    generated = 0
    errors = 0

    # Process books sequentially — long books handle their own parallelization
    with Progress(console=console) as progress:
        task = progress.add_task("Generating maps...", total=len(to_process))

        for md_path in to_process:
            basename = md_path.stem
            try:
                result = process_one(client, model, md_path, workers, verbose)
                usage = result["usage"]
                total_prompt += usage.get("prompt_tokens", 0)
                total_completion += usage.get("completion_tokens", 0)
                generated += 1

                ch_count = ""
                map_path = MAPS_DIR / f"{basename}.json"
                if map_path.exists():
                    data = json.loads(map_path.read_text())
                    ch_count = f" ({len(data.get('chapters', []))} chapters)"

                if result["warnings"]:
                    progress.console.print(
                        f"  [yellow]Warning ({basename}{ch_count}):[/yellow] "
                        f"{', '.join(result['warnings'])}"
                    )
                else:
                    progress.console.print(f"  [green]Done:[/green] {basename}{ch_count}")

            except Exception as e:
                progress.console.print(f"  [red]Error ({basename}):[/red] {e}")
                errors += 1

            progress.advance(task)

    if provider == "openrouter":
        input_cost = total_prompt / 1_000_000 * 0.10
        output_cost = total_completion / 1_000_000 * 0.20
        total_cost = input_cost + output_cost
        console.print(f"\nTokens — input: {total_prompt:,}, output: {total_completion:,}")
        console.print(f"Estimated cost: ${total_cost:.4f}")

    console.print(
        f"\n[green]Done.[/green] Generated: {generated}, Skipped: {skipped}, Errors: {errors}"
    )


if __name__ == "__main__":
    main()
