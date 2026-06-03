"""Local PDF → Markdown conversion via PyMuPDF4LLM.

Designed to run in a multiprocessing worker — PyMuPDF is not thread-safe.
Each worker holds its own Document instance (~150-250 MB peak).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.queue import write_meta
from pipeline.triage import get_pdf_page_count

logger = logging.getLogger(__name__)

# Quality gate: if raw.md has fewer than this many chars per source page,
# reclassify as scanned and move to cloud staging
MIN_CHARS_PER_PAGE_QUALITY = 500


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a native PDF to markdown using PyMuPDF4LLM.

    Must be called in a separate process (not thread) due to PyMuPDF constraints.
    Converts in 50-page batches with progress logging.
    OCR is disabled since these are native PDFs with a text layer — Tesseract
    would double the text and 10x the runtime.
    """
    import pymupdf
    import pymupdf4llm

    with pymupdf.open(str(pdf_path)) as doc:
        total = len(doc)

    BATCH = 50
    logger.info("Converting %d pages from %s (text-layer only, no OCR)", total, pdf_path.name)

    kwargs = dict(show_progress=False, use_ocr=False)

    if total <= BATCH:
        return pymupdf4llm.to_markdown(str(pdf_path), **kwargs)

    chunks = []
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        pages = list(range(start, end))
        chunk = pymupdf4llm.to_markdown(str(pdf_path), pages=pages, **kwargs)
        chunks.append(chunk)
        logger.info("  pages %d–%d / %d done", start + 1, end, total)

    return "\n\n".join(chunks)


def process_local_pdf(staging_folder: Path) -> str | None:
    """Convert a native PDF in a staging folder to raw.md.

    Returns "ok" on success, "reclassify" if quality gate fails, None on error.
    """
    source = staging_folder / "source.pdf"
    if not source.exists():
        logger.error("No source.pdf in %s", staging_folder)
        return None

    try:
        md_text = convert_pdf_to_markdown(source)
    except Exception as e:
        logger.error("PyMuPDF4LLM failed for %s: %s", staging_folder.name, e)
        write_meta(staging_folder, conversion_error=str(e))
        return None

    # Write raw markdown
    raw_path = staging_folder / "raw.md"
    raw_path.write_text(md_text, encoding="utf-8")

    # Quality gate
    page_count = get_pdf_page_count(source)
    chars_per_page = len(md_text) / max(page_count, 1)

    write_meta(
        staging_folder,
        raw_chars=len(md_text),
        page_count=page_count,
        chars_per_page=round(chars_per_page, 1),
        conversion_method="pymupdf4llm",
    )

    if chars_per_page < MIN_CHARS_PER_PAGE_QUALITY:
        logger.warning(
            "%s: only %.0f chars/page (threshold %d) — reclassifying as scanned",
            staging_folder.name,
            chars_per_page,
            MIN_CHARS_PER_PAGE_QUALITY,
        )
        write_meta(staging_folder, reclassified_to="scanned", quality_gate="failed")
        return "reclassify"

    logger.info(
        "%s: converted %d pages, %.0f chars/page",
        staging_folder.name,
        page_count,
        chars_per_page,
    )
    return "ok"
