"""Send to Kindle via Resend email API.

Amazon accepts EPUB natively — no MOBI conversion needed.
Uses Resend (resend.com) for email delivery with EPUB attachment.
"""

from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path

import resend

from pipeline.config import PipelineConfig
from pipeline.queue import read_meta

logger = logging.getLogger(__name__)


def create_epub_from_markdown(
    md_path: Path,
    epub_path: Path,
    title: str = "",
    author: str = "",
    subtitle: str = "",
    publisher: str = "",
    date: str = "",
    description: str = "",
    subjects: list[str] | None = None,
    lang: str = "en",
) -> bool:
    """Convert clean markdown to EPUB via pandoc with metadata."""
    cmd = ["pandoc", str(md_path), "-o", str(epub_path), "--wrap=none"]

    if title:
        cmd += ["--metadata", f"title={title}"]
    if author:
        cmd += ["--metadata", f"author={author}"]
    if subtitle:
        cmd += ["--metadata", f"subtitle={subtitle}"]
    if publisher:
        cmd += ["--metadata", f"publisher={publisher}"]
    if date:
        cmd += ["--metadata", f"date={date}"]
    if description:
        cmd += ["--metadata", f"description={description}"]
    if lang:
        cmd += ["--metadata", f"lang={lang}"]
    for subj in (subjects or []):
        cmd += ["--metadata", f"subject={subj}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("Pandoc md→epub failed: %s", result.stderr.strip())
            return False
        return True
    except Exception as e:
        logger.error("Pandoc md→epub error: %s", e)
        return False


def _extract_metadata(staging_folder: Path, cfg: PipelineConfig) -> dict:
    """Extract book metadata from meta.json and the book map if available."""
    meta = read_meta(staging_folder)
    original_name = meta.get("original_name", staging_folder.name)
    title = Path(original_name).stem

    result = {"title": title, "author": ""}

    book_id = meta.get("book_id", "")

    if book_id:
        map_path = cfg.paths.maps_dir / f"{book_id}.json"
        if map_path.exists():
            import json
            try:
                book_map = json.loads(map_path.read_text(encoding="utf-8"))
                result["title"] = book_map.get("title", title)
                result["author"] = book_map.get("author", "")
                result["description"] = book_map.get("summary", "")
                result["subjects"] = book_map.get("key_themes", [])[:5]
            except Exception as e:
                logger.warning("Failed to read book map %s: %s", map_path.name, e)

    return result


def send_to_kindle(epub_path: Path, cfg: PipelineConfig, book_title: str = "") -> bool:
    """Send an EPUB to Kindle via Resend API.

    Returns True on success, False on failure.
    """
    if not cfg.kindle.enabled:
        logger.info("Kindle delivery disabled — skipping")
        return False

    if not cfg.kindle.kindle_email or not cfg.kindle.sender_email:
        logger.warning("Kindle email or sender email not configured")
        return False

    if not cfg.kindle.resend_api_key:
        logger.warning("Resend API key not configured")
        return False

    resend.api_key = cfg.kindle.resend_api_key
    subject = book_title or epub_path.stem

    epub_data = epub_path.read_bytes()

    try:
        params: resend.Emails.SendParams = {
            "from": cfg.kindle.sender_email,
            "to": [cfg.kindle.kindle_email],
            "subject": subject,
            "text": "Sent via lib-rag",
            "attachments": [
                {
                    "filename": epub_path.name,
                    "content": base64.b64encode(epub_data).decode("ascii"),
                    "content_type": "application/epub+zip",
                }
            ],
        }
        result = resend.Emails.send(params)
        logger.info("Sent %s to Kindle (%s) — id: %s", epub_path.name, cfg.kindle.kindle_email, result.get("id"))
        return True
    except Exception as e:
        logger.error("Kindle send failed: %s", e)
        outbox = Path.home() / "outputs" / "kindle-outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(epub_path), str(outbox / epub_path.name))
        logger.info("EPUB saved to %s for manual sending", outbox / epub_path.name)
        return False


def process_kindle(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Handle Kindle delivery for a processed book.

    For EPUB sources: send the original EPUB directly.
    For PDF sources: convert clean.md → EPUB with metadata, then send.
    """
    if not cfg.kindle.enabled:
        return True  # not an error, just disabled

    meta = read_meta(staging_folder)
    original_name = meta.get("original_name", staging_folder.name)
    book_meta = _extract_metadata(staging_folder, cfg)

    # Check if source is EPUB (send original directly)
    source_epub = staging_folder / "source.epub"
    if source_epub.exists():
        display_title = book_meta.get("title", Path(original_name).stem)
        author = book_meta.get("author", "")
        label = f"{display_title} — {author}" if author else display_title
        return send_to_kindle(source_epub, cfg, label)

    # PDF path: convert clean.md → EPUB with metadata → send
    clean_md = staging_folder / "clean.md"
    if not clean_md.exists():
        logger.error("No clean.md for Kindle conversion in %s", staging_folder.name)
        return False

    # Name the EPUB file after the book so Kindle displays it correctly
    display_title = book_meta.get("title", Path(original_name).stem)
    author = book_meta.get("author", "")
    safe_name = "".join(c for c in display_title if c.isalnum() or c in " -_").strip()
    epub_filename = f"{safe_name}.epub" if safe_name else "output.epub"
    output_epub = staging_folder / epub_filename

    if not create_epub_from_markdown(
        clean_md,
        output_epub,
        title=display_title,
        author=author,
        description=book_meta.get("description", ""),
        subjects=book_meta.get("subjects"),
    ):
        return False

    label = f"{display_title} — {author}" if author else display_title
    return send_to_kindle(output_epub, cfg, label)
