#!/usr/bin/env python3
"""Convert EPUBs in source/ to Markdown in markdown/ via Pandoc, with cleanup."""

import re
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "source"
MARKDOWN_DIR = ROOT / "markdown"

# Add parent dir so pipeline imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cleanup import regex_cleanup  # shared with pipeline

console = Console()


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

    raw = result.stdout
    # Strip EPUB cover/title/logo images before general cleanup
    raw = re.sub(
        r"^!\[\]\(images/(cover|title|logo|image\d+)\.(jpg|png|svg)\)(\{[^}]*\})?\s*$",
        "", raw, flags=re.MULTILINE | re.IGNORECASE,
    )
    md_text = regex_cleanup(raw)
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
