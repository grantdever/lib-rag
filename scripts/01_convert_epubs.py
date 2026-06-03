#!/usr/bin/env python3
"""Convert EPUBs in source/ to Markdown in markdown/ via Pandoc, with cleanup."""

import re
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "source"
MARKDOWN_DIR = ROOT / "markdown"

console = Console()


def clean_markdown(text: str) -> str:
    """Clean Pandoc output: strip artifacts, fix headers, remove HTML."""

    # Line-level cleaning
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = re.sub(r"\[\]\{#[^}]*\}", "", line)            # empty anchors
        line = re.sub(r"\[\]\{(\.[a-zA-Z0-9_-]+\s*)+\}", "", line)  # empty class spans
        line = re.sub(r"</?div[^>]*>", "", line)
        line = re.sub(r"</?p[^>]*>", "", line)
        line = re.sub(r"</?span[^>]*>", "", line)

        if re.match(r"^:::", line):  # Pandoc fenced divs
            continue

        stripped = line.strip()
        if re.match(r"^!\[\]\(images/(cover|title|logo|image\d+)\.(jpg|png|svg)\)(\{[^}]*\})?\s*$", stripped, re.IGNORECASE):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Remove SVG blocks
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)

    # Clean headers: unwrap [text]{.class} spans, strip trailing {#anchor .class}
    def clean_header(m):
        prefix = m.group(1)
        rest = m.group(2)
        rest = re.sub(r"\[([^\]]*)\]\{[^}]*\}", r"\1", rest)
        rest = re.sub(r"\s*\{[^}]*\}\s*$", "", rest)
        rest = re.sub(r"\s+", " ", rest).strip()
        if not rest:
            return ""
        return f"{prefix} {rest}"

    text = re.sub(r"^(#{1,6})\s+(.+)$", clean_header, text, flags=re.MULTILINE)
    text = re.sub(r"^\s*$\n", "\n", text, flags=re.MULTILINE)

    # Strip inline class attributes: [visible text]{.class} → visible text
    text = re.sub(r"\[([^\]]+)\]\{(\.[a-zA-Z0-9_-]+\s*)+\}", r"\1", text)

    # Strip all remaining Pandoc attribute blocks: {#id}, {.class}, {#id .class}
    text = re.sub(r"\{[#.][^}]*\}", "", text)

    # Fix broken words in headers (Pandoc span artifacts: "COACHIN G" → "COACHING")
    def fix_broken_header(m):
        prefix = m.group(1)
        rest = m.group(2)
        rest = re.sub(r"(?<![A-Za-z])([A-Z]) ([a-z])", r"\1\2", rest)          # "C hapter" → "Chapter"
        rest = re.sub(r"([A-Z]{2,}) ([A-Z])(?=[\s.,;:]|$)", r"\1\2", rest)     # "COACHIN G" → "COACHING"
        rest = re.sub(r"(?<![A-Za-z])([A-Z]) ([A-Z]{2,})", r"\1\2", rest)      # "T ECHNICAL" → "TECHNICAL"
        rest = re.sub(r" +([.,;:])", r"\1", rest)                                # "V . TECH" → "V. TECH"
        rest = re.sub(r"  +", " ", rest).strip()
        return f"{prefix} {rest}" if rest else ""
    text = re.sub(r"^(#{1,6})\s+(.+)$", fix_broken_header, text, flags=re.MULTILINE)

    # Collapse excessive blank lines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Infer headers for books where Pandoc produced none
    has_h1 = any(line.startswith("# ") for line in text.splitlines())
    if not has_h1:
        text = infer_headers(text)

    return text.strip() + "\n"


def infer_headers(text: str) -> str:
    """For books with no H1 headers, try to detect chapter boundaries."""
    lines = text.split("\n")
    result = []
    prev_blank = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            prev_blank = True
            result.append(line)
            continue

        # "Chapter N" or "CHAPTER N" after a blank line
        ch_match = re.match(r"^(Chapter\s+\d+[\.:]?\s*.*|CHAPTER\s+[IVXLCDM\d]+[\.:]?\s*.*)$", stripped, re.IGNORECASE)
        if ch_match and prev_blank:
            result.append(f"# {stripped}")
            prev_blank = False
            continue

        # ALL CAPS title line (2+ words, >10 chars)
        if prev_blank and len(stripped) > 10 and stripped == stripped.upper() and re.match(r'^[A-Z\s\-:,\'"]+$', stripped):
            if len(stripped.split()) >= 2:
                result.append(f"# {stripped.title()}")
                prev_blank = False
                continue

        # Roman numeral sections: "I. Title" or "Part I"
        part_match = re.match(r"^(Part\s+[IVXLCDM]+[\.:]\s*.+|[IVXLCDM]+\.\s+[A-Z].+)$", stripped)
        if part_match and prev_blank:
            result.append(f"# {stripped}")
            prev_blank = False
            continue

        prev_blank = False
        result.append(line)

    return "\n".join(result)


def convert_one(epub_path: Path, md_path: Path) -> bool:
    """Convert a single EPUB to Markdown with cleanup. Returns True on success."""
    result = subprocess.run(
        ["pandoc", "-f", "epub", "-t", "markdown", "--wrap=none", str(epub_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"  [red]Pandoc error:[/red] {result.stderr.strip()}")
        return False

    md_text = clean_markdown(result.stdout)
    md_path.write_text(md_text, encoding="utf-8")

    has_h1 = any(line.startswith("# ") for line in md_text.splitlines())
    if not has_h1:
        console.print(f"  [yellow]Warning:[/yellow] No H1 (#) headers found after cleanup in {md_path.name}")
    return True


@click.command()
@click.option("--force", is_flag=True, help="Re-convert even if markdown already exists")
def main(force: bool):
    """Convert all EPUBs in source/ to Markdown."""
    epubs = sorted(SOURCE_DIR.glob("*.epub"))
    if not epubs:
        console.print("[yellow]No .epub files found in source/[/yellow]")
        return

    skipped = 0
    converted = 0
    errors = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Converting EPUBs...", total=len(epubs))

        for epub_path in epubs:
            slug = epub_path.stem
            md_path = MARKDOWN_DIR / f"{slug}.md"

            if md_path.exists() and not force:
                progress.console.print(f"  [dim]Skip (exists):[/dim] {slug}")
                skipped += 1
                progress.advance(task)
                continue

            progress.console.print(f"  Converting: {slug}")
            if convert_one(epub_path, md_path):
                converted += 1
            else:
                errors += 1
            progress.advance(task)

    console.print(
        f"\n[green]Done.[/green] Converted: {converted}, Skipped: {skipped}, Errors: {errors}"
    )


if __name__ == "__main__":
    main()
