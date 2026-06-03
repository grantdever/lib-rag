"""Cloud PDF → Markdown via Gemini 2.5 Flash on OpenRouter.

Uploads PDF chunks natively (not as images) for high-quality text extraction.
Produces reading-quality markdown suitable for EPUB/Kindle output —
substantially better than PyMuPDF4LLM for paragraph reconstruction,
dehyphenation, and header/footer removal.

Cost: ~$0.70–0.90 per 500 pages (mostly output tokens).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import time
from pathlib import Path

import requests

from pipeline.queue import write_meta
from pipeline.triage import get_pdf_page_count

logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # pages per API request
MAX_OUTPUT_TOKENS = 65000

SYSTEM_PROMPT = (
    "Convert these book pages to clean markdown. "
    "Preserve all text accurately. Use proper paragraph breaks. "
    "Remove headers, footers, and page numbers. "
    "Join hyphenated words split across line breaks. "
    "Preserve footnotes with their numbering. "
    "Use ## for chapter/section headings. "
    "Output ONLY the markdown text, no commentary or explanations."
)


def _get_api_key() -> str:
    """Get OpenRouter API key from environment (loaded by dotenv at startup)."""
    return os.getenv("OPENROUTER_API_KEY", "")


def _split_pdf(pdf_path: Path, start: int, end: int) -> bytes:
    """Extract pages [start, end) into a new PDF and return its bytes."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        with pymupdf.open() as out:
            out.insert_pdf(doc, from_page=start, to_page=end - 1)
            return out.tobytes()


def _convert_batch(pdf_bytes: bytes, filename: str, api_key: str) -> str | None:
    """Send a PDF chunk to Gemini 2.5 Flash via OpenRouter's native PDF support."""
    b64_pdf = base64.standard_b64encode(pdf_bytes).decode()

    content: list[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "file", "file": {
            "filename": filename,
            "file_data": f"data:application/pdf;base64,{b64_pdf}",
        }},
    ]

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": content}],
                    "plugins": [{"id": "file-parser", "pdf": {"engine": "native"}}],
                    "max_tokens": MAX_OUTPUT_TOKENS,
                },
                timeout=600,
            )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                logger.warning("JSON decode failed on attempt %d (status %d)", attempt + 1, resp.status_code)
                if attempt < 2:
                    time.sleep(5)
                    continue
                return None

            if "error" in data:
                err = data["error"]
                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                if resp.status_code == 429 or "rate" in err_msg.lower():
                    wait = 2 ** attempt * 10
                    logger.warning("Rate limited, waiting %ds: %s", wait, err_msg)
                    time.sleep(wait)
                    continue
                logger.error("API error: %s", err_msg)
                return None

            if "choices" in data and data["choices"]:
                text = data["choices"][0]["message"]["content"]
                # Strip markdown code fences if Gemini wraps output
                if text.startswith("```markdown"):
                    text = text[len("```markdown"):].strip()
                if text.startswith("```"):
                    text = text[3:].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                return text

            logger.error("Unexpected response: %s", json.dumps(data)[:500])
            return None

        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning("Network error on attempt %d: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(5)
        except Exception as e:
            logger.error("Request error on attempt %d: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(5)

    return None


def convert_pdf_cloud(pdf_path: Path, staging_folder: Path | None = None) -> str | None:
    """Convert a PDF to markdown using Gemini 2.5 Flash via OpenRouter.

    Splits the PDF into ~100-page chunks, uploads each natively, and joins
    the results. If staging_folder is provided, saves each batch to disk
    for resume support on interruption.

    Returns the full markdown text, or None on failure.
    """
    import pymupdf

    api_key = _get_api_key()
    if not api_key:
        logger.error("No OPENROUTER_API_KEY found")
        return None

    with pymupdf.open(str(pdf_path)) as doc:
        total = len(doc)

    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "Cloud-converting %d pages from %s (%d batches of up to %d pages)",
        total, pdf_path.name, num_batches, BATCH_SIZE,
    )

    # Set up batch cache directory for incremental saves
    batch_dir = None
    if staging_folder:
        batch_dir = staging_folder / ".batches"
        batch_dir.mkdir(exist_ok=True)

    chunks = []
    for batch_idx, start in enumerate(range(0, total, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, total)

        # Check for cached batch result
        if batch_dir:
            batch_file = batch_dir / f"batch_{batch_idx:03d}.md"
            if batch_file.exists():
                cached = batch_file.read_text(encoding="utf-8")
                chunks.append(cached)
                logger.info("  pages %d–%d / %d: cached (%d chars)", start + 1, end, total, len(cached))
                continue

        pdf_bytes = _split_pdf(pdf_path, start, end)
        filename = f"pages_{start + 1}-{end}.pdf"

        text = _convert_batch(pdf_bytes, filename, api_key)
        if text is None:
            logger.error("Failed on pages %d–%d", start + 1, end)
            return None

        # Save batch to disk immediately
        if batch_dir:
            batch_file = batch_dir / f"batch_{batch_idx:03d}.md"
            batch_file.write_text(text, encoding="utf-8")

        chunks.append(text)
        logger.info("  pages %d–%d / %d done (%d chars)", start + 1, end, total, len(text))

        if end < total:
            time.sleep(1)

    result = "\n\n".join(chunks)

    # Clean up batch cache on success
    if batch_dir and batch_dir.exists():
        shutil.rmtree(batch_dir)
        logger.debug("Cleaned up batch cache at %s", batch_dir)

    return result


def process_cloud_pdf(staging_folder: Path) -> str | None:
    """Convert a PDF in a staging folder using Gemini cloud conversion.

    Returns "ok" on success, None on error.
    """
    source = staging_folder / "source.pdf"
    if not source.exists():
        logger.error("No source.pdf in %s", staging_folder)
        return None

    try:
        md_text = convert_pdf_cloud(source, staging_folder=staging_folder)
    except Exception as e:
        logger.error("Cloud conversion failed for %s: %s", staging_folder.name, e)
        write_meta(staging_folder, conversion_error=str(e))
        return None

    if md_text is None:
        write_meta(staging_folder, conversion_error="Cloud conversion returned no text")
        return None

    raw_path = staging_folder / "raw.md"
    raw_path.write_text(md_text, encoding="utf-8")

    page_count = get_pdf_page_count(source)

    chars_per_page = len(md_text) / max(page_count, 1)

    write_meta(
        staging_folder,
        raw_chars=len(md_text),
        page_count=page_count,
        chars_per_page=round(chars_per_page, 1),
        conversion_method="gemini-2.5-flash",
    )

    logger.info(
        "%s: cloud-converted %d pages, %.0f chars/page",
        staging_folder.name, page_count, chars_per_page,
    )
    return "ok"
