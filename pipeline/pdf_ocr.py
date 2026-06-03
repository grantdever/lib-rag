"""Mistral OCR batch API client for scanned PDFs.

Uses the Mistral direct API (not OpenRouter) for batch mode with 50% discount.
Batch API: upload file → create batch → poll → download results.

Ref: https://docs.mistral.ai/capabilities/document/
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx

from pipeline.config import PipelineConfig
from pipeline.queue import write_meta

logger = logging.getLogger(__name__)

MISTRAL_API_BASE = "https://api.mistral.ai/v1"
OCR_MODEL = "mistral-ocr-latest"

# Batch API endpoints
BATCH_API_BASE = "https://api.mistral.ai/v1/batch"
FILES_API_BASE = "https://api.mistral.ai/v1/files"


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def ocr_single_pdf(pdf_path: Path, api_key: str) -> str:
    """OCR a single PDF via Mistral's sync API (for testing/small files).

    Returns markdown text.
    """
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    data_url = f"data:application/pdf;base64,{b64}"

    payload = {
        "model": OCR_MODEL,
        "document": {
            "type": "document_url",
            "document_url": data_url,
        },
    }

    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{MISTRAL_API_BASE}/ocr",
            headers=_headers(api_key),
            json=payload,
        )
        resp.raise_for_status()

    result = resp.json()
    # Extract markdown from pages
    pages = result.get("pages", [])
    md_parts = []
    for page in pages:
        md = page.get("markdown", "")
        if md:
            md_parts.append(md)

    return "\n\n---\n\n".join(md_parts)


def upload_file_for_batch(pdf_path: Path, api_key: str) -> str:
    """Upload a PDF to Mistral's files API for batch processing.

    Returns the file ID.
    """
    with httpx.Client(timeout=120) as client:
        with open(pdf_path, "rb") as f:
            resp = client.post(
                FILES_API_BASE,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"purpose": "ocr"},
            )
            resp.raise_for_status()

    data = resp.json()
    file_id = data["id"]
    logger.info("Uploaded %s → file_id=%s", pdf_path.name, file_id)
    return file_id


def create_batch_request_file(
    staging_folders: list[Path],
    api_key: str,
) -> tuple[str, dict[str, Path]]:
    """Upload PDFs and create a JSONL batch request file.

    Returns (request_file_id, {custom_id: staging_folder} mapping).
    """
    lines = []
    folder_map: dict[str, Path] = {}

    for folder in staging_folders:
        source = folder / "source.pdf"
        if not source.exists():
            logger.warning("No source.pdf in %s, skipping", folder.name)
            continue

        file_id = upload_file_for_batch(source, api_key)
        custom_id = folder.name

        line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/ocr",
            "body": {
                "model": OCR_MODEL,
                "document": {
                    "type": "file_id",
                    "file_id": file_id,
                },
            },
        }
        lines.append(json.dumps(line))
        folder_map[custom_id] = folder

        write_meta(folder, mistral_file_id=file_id, batch_status="uploaded")

    # Upload JSONL request file
    jsonl_content = "\n".join(lines)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.write(jsonl_content)
        tmp_path = Path(tmp.name)

    with httpx.Client(timeout=60) as client:
        with open(tmp_path, "rb") as f:
            resp = client.post(
                FILES_API_BASE,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (f"batch_request.jsonl", f, "application/jsonl")},
                data={"purpose": "batch"},
            )
            resp.raise_for_status()

    tmp_path.unlink()
    request_file_id = resp.json()["id"]
    logger.info("Created batch request file: %s (%d items)", request_file_id, len(lines))
    return request_file_id, folder_map


def submit_batch(request_file_id: str, api_key: str) -> dict:
    """Submit a batch job to Mistral.

    Returns the batch job metadata dict.
    """
    payload = {
        "input_files": [request_file_id],
        "endpoint": "/v1/ocr",
        "model": OCR_MODEL,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{BATCH_API_BASE}/jobs",
            headers=_headers(api_key),
            json=payload,
        )
        resp.raise_for_status()

    batch_data = resp.json()
    logger.info("Batch submitted: %s (status: %s)", batch_data["id"], batch_data.get("status"))
    return batch_data


def check_batch_status(batch_id: str, api_key: str) -> dict:
    """Check the status of a batch job.

    Returns the batch job metadata dict.
    """
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{BATCH_API_BASE}/jobs/{batch_id}",
            headers=_headers(api_key),
        )
        resp.raise_for_status()

    return resp.json()


def download_batch_results(batch_data: dict, api_key: str) -> list[dict]:
    """Download results from a completed batch.

    Returns list of result dicts with custom_id and response body.
    """
    output_file_id = batch_data.get("output_file")
    if not output_file_id:
        raise ValueError("Batch has no output_file — may not be complete")

    with httpx.Client(timeout=120) as client:
        resp = client.get(
            f"{FILES_API_BASE}/{output_file_id}/content",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()

    results = []
    for line in resp.text.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


def extract_markdown_from_result(result: dict) -> str:
    """Extract markdown text from a batch result item."""
    response = result.get("response", {})
    body = response.get("body", {})
    pages = body.get("pages", [])

    md_parts = []
    for page in pages:
        md = page.get("markdown", "")
        if md:
            md_parts.append(md)

    return "\n\n".join(md_parts)


def cancel_batch(batch_id: str, api_key: str) -> dict:
    """Cancel a queued batch job."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{BATCH_API_BASE}/jobs/{batch_id}/cancel",
            headers=_headers(api_key),
        )
        resp.raise_for_status()

    return resp.json()
