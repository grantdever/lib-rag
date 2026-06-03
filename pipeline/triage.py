"""Triage incoming files: classify PDFs as native/scanned, route to staging.

Uses PyMuPDF to probe the text layer of PDFs. EPUBs are routed directly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pymupdf

from pipeline.config import PipelineConfig
from pipeline.queue import compute_sha256, create_staging_folder, write_meta

logger = logging.getLogger(__name__)

# Thresholds for native vs scanned classification
MIN_CHARS_PER_PAGE = 100
SAMPLE_PAGES = 5


def probe_pdf_text_layer(pdf_path: Path) -> dict:
    """Sample pages from a PDF and measure text content.

    Returns dict with:
        avg_chars_per_page: float
        sampled_pages: int
        total_pages: int
        classification: "native" | "scanned"
    """
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        return {
            "avg_chars_per_page": 0,
            "sampled_pages": 0,
            "total_pages": 0,
            "classification": "scanned",
        }

    # Sample up to SAMPLE_PAGES evenly distributed pages
    if total_pages <= SAMPLE_PAGES:
        sample_indices = list(range(total_pages))
    else:
        step = total_pages / SAMPLE_PAGES
        sample_indices = [int(i * step) for i in range(SAMPLE_PAGES)]

    total_chars = 0
    for idx in sample_indices:
        page = doc[idx]
        text = page.get_text("text")
        total_chars += len(text.strip())

    doc.close()

    avg_chars = total_chars / len(sample_indices) if sample_indices else 0
    classification = "native" if avg_chars >= MIN_CHARS_PER_PAGE else "scanned"

    return {
        "avg_chars_per_page": round(avg_chars, 1),
        "sampled_pages": len(sample_indices),
        "total_pages": total_pages,
        "classification": classification,
    }


def get_pdf_page_count(pdf_path: Path) -> int:
    """Quick page count without full text extraction."""
    doc = pymupdf.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def classify_file(file_path: Path) -> dict:
    """Classify a file as epub, native PDF, or scanned PDF.

    Returns dict with:
        file_type: "epub" | "pdf"
        classification: "epub" | "native" | "scanned"
        probe: dict (for PDFs only)
    """
    suffix = file_path.suffix.lower()

    if suffix == ".epub":
        return {
            "file_type": "epub",
            "classification": "epub",
            "probe": {},
        }

    if suffix == ".pdf":
        probe = probe_pdf_text_layer(file_path)
        return {
            "file_type": "pdf",
            "classification": probe["classification"],
            "probe": probe,
        }

    return {
        "file_type": suffix.lstrip("."),
        "classification": "unsupported",
        "probe": {},
    }


def triage_file(file_path: Path, cfg: PipelineConfig) -> Path | None:
    """Triage a single file: classify, dedup, route to staging.

    Returns the staging folder path, or None if skipped (duplicate/unsupported).
    """
    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return None

    # Compute SHA for dedup
    sha = compute_sha256(file_path)
    logger.info("Triaging %s (sha: %s...)", file_path.name, sha[:12])

    # Check for duplicates across all staging categories
    for category in ("local", "epub", "cloud"):
        cat_dir = cfg.paths.staging / category
        if not cat_dir.exists():
            continue
        for folder in cat_dir.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            meta_path = folder / "meta.json"
            if meta_path.exists():
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("sha256") == sha:
                    logger.info("Duplicate detected: %s already staged at %s", file_path.name, folder)
                    return None

    # Classify
    result = classify_file(file_path)

    if result["classification"] == "unsupported":
        logger.warning("Unsupported file type: %s", file_path.name)
        # Move to quarantine
        quarantine_dest = cfg.paths.quarantine / file_path.name
        shutil.move(str(file_path), str(quarantine_dest))
        return None

    # Route to staging
    category_map = {
        "epub": "epub",
        "native": "local",
        "scanned": "cloud",
    }
    category = category_map[result["classification"]]
    folder = create_staging_folder(cfg.paths.staging, category)

    # Move source file
    source_name = f"source{file_path.suffix.lower()}"
    dest = folder / source_name
    shutil.move(str(file_path), str(dest))

    # Write metadata
    write_meta(
        folder,
        sha256=sha,
        original_name=file_path.name,
        file_type=result["file_type"],
        classification=result["classification"],
        probe=result["probe"],
        staging_category=category,
    )

    logger.info(
        "Triaged %s → %s/%s (%s)",
        file_path.name,
        category,
        folder.name,
        result["classification"],
    )
    return folder
