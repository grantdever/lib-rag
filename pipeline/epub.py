"""EPUB → Markdown conversion via Pandoc subprocess.

Reuses the pandoc invocation pattern from the existing scripts/01_convert_epubs.py.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from pipeline.queue import write_meta

logger = logging.getLogger(__name__)


def convert_epub_to_markdown(epub_path: Path) -> str:
    """Convert an EPUB to raw markdown via pandoc.

    Raises RuntimeError if pandoc fails.
    """
    result = subprocess.run(
        ["pandoc", "-f", "epub", "-t", "markdown", "--wrap=none", str(epub_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pandoc error: {result.stderr.strip()}")

    return result.stdout


def process_epub(staging_folder: Path) -> str | None:
    """Convert an EPUB in a staging folder to raw.md.

    Returns "ok" on success, None on error.
    """
    source = staging_folder / "source.epub"
    if not source.exists():
        logger.error("No source.epub in %s", staging_folder)
        return None

    try:
        md_text = convert_epub_to_markdown(source)
    except Exception as e:
        logger.error("Pandoc failed for %s: %s", staging_folder.name, e)
        write_meta(staging_folder, conversion_error=str(e))
        return None

    raw_path = staging_folder / "raw.md"
    raw_path.write_text(md_text, encoding="utf-8")

    write_meta(
        staging_folder,
        raw_chars=len(md_text),
        conversion_method="pandoc",
    )

    logger.info("%s: converted EPUB, %d chars", staging_folder.name, len(md_text))
    return "ok"
