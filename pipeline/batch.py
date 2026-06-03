"""Batch CLI for Mistral OCR: submit, status, cancel, and poll.

Usage:
    python -m pipeline.batch submit    # preview cost, submit scanned PDFs
    python -m pipeline.batch status    # list open batches
    python -m pipeline.batch cancel ID # cancel a queued batch
    python -m pipeline.batch poll      # check completed batches, download results
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from pipeline.config import ensure_config, ensure_directories
from pipeline.pdf_ocr import (
    cancel_batch,
    check_batch_status,
    create_batch_request_file,
    download_batch_results,
    extract_markdown_from_result,
    submit_batch,
)
from pipeline.queue import read_meta, write_meta
from pipeline.triage import get_pdf_page_count

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Manage Mistral OCR batch processing for scanned PDFs."""
    pass


@cli.command()
def submit():
    """Survey pending scanned PDFs, preview cost, and submit batch."""
    cfg = ensure_config()
    ensure_directories(cfg)

    if not cfg.api_keys.mistral:
        click.echo("Error: MISTRAL_API_KEY not configured. Add to .env or config.toml.")
        sys.exit(1)

    cloud_dir = cfg.paths.staging_cloud
    pending = []

    for folder in sorted(cloud_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta = read_meta(folder)
        if meta.get("batch_status") in ("submitted", "complete"):
            continue
        source = folder / "source.pdf"
        if not source.exists():
            continue
        page_count = get_pdf_page_count(source)
        pending.append((folder, meta, page_count))

    if not pending:
        click.echo("No scanned PDFs pending in cloud staging.")
        return

    click.echo(f"\n📚 Scanned PDFs pending in {cloud_dir}:\n")
    total_pages = 0
    for folder, meta, pages in pending:
        name = meta.get("original_name", folder.name)
        click.echo(f"  {name:<50} {pages:>6} pages")
        total_pages += pages

    cost_estimate = total_pages / 1000 * 1.0  # $1/1k pages batch rate
    click.echo(f"\n  {'Total:':<50} {total_pages:>6} pages")
    click.echo(f"  Estimated cost (batch, $1/1k pages):  ${cost_estimate:.2f}")
    click.echo(f"  Estimated turnaround:                 up to 24 hours")

    if not click.confirm("\nContinue?", default=False):
        click.echo("Aborted.")
        return

    # Create batch request
    folders = [f for f, _, _ in pending]
    try:
        request_file_id, folder_map = create_batch_request_file(
            folders, cfg.api_keys.mistral
        )
    except Exception as e:
        click.echo(f"Error uploading files: {e}")
        sys.exit(1)

    # Submit batch
    try:
        batch_data = submit_batch(request_file_id, cfg.api_keys.mistral)
    except Exception as e:
        click.echo(f"Error submitting batch: {e}")
        sys.exit(1)

    batch_id = batch_data["id"]

    # Save tracking file
    batches_dir = cfg.paths.batches_dir
    batches_dir.mkdir(parents=True, exist_ok=True)
    tracking = {
        "batch_id": batch_id,
        "request_file_id": request_file_id,
        "folders": {cid: str(f) for cid, f in folder_map.items()},
        "total_pages": total_pages,
        "cost_estimate": cost_estimate,
        "status": batch_data.get("status", "queued"),
    }
    tracking_path = batches_dir / f"{batch_id}.json"
    tracking_path.write_text(json.dumps(tracking, indent=2) + "\n", encoding="utf-8")

    # Update folder metas
    for folder in folders:
        write_meta(folder, batch_id=batch_id, batch_status="submitted")

    click.echo(f"\n✅ Batch submitted: {batch_id}")
    click.echo(f"   Tracked: {tracking_path}")
    click.echo(f"   Check progress: python -m pipeline.batch status")


@cli.command()
def status():
    """List open batches with their status."""
    cfg = ensure_config()
    batches_dir = cfg.paths.batches_dir

    if not batches_dir.exists():
        click.echo("No batches directory found.")
        return

    tracking_files = sorted(batches_dir.glob("*.json"))
    if not tracking_files:
        click.echo("No open batches.")
        return

    for tf in tracking_files:
        tracking = json.loads(tf.read_text(encoding="utf-8"))
        batch_id = tracking["batch_id"]

        # Poll current status
        try:
            batch_data = check_batch_status(batch_id, cfg.api_keys.mistral)
            current_status = batch_data.get("status", "unknown")
        except Exception as e:
            current_status = f"error: {e}"

        pages = tracking.get("total_pages", "?")
        cost = tracking.get("cost_estimate", 0)
        n_books = len(tracking.get("folders", {}))

        click.echo(
            f"  {batch_id}  {current_status:<12}  "
            f"{n_books} books, {pages} pages, ~${cost:.2f}"
        )


@cli.command()
@click.argument("batch_id")
def cancel(batch_id: str):
    """Cancel a queued batch job."""
    cfg = ensure_config()

    try:
        result = cancel_batch(batch_id, cfg.api_keys.mistral)
        click.echo(f"Cancelled: {batch_id} (status: {result.get('status')})")
    except Exception as e:
        click.echo(f"Error cancelling {batch_id}: {e}")
        sys.exit(1)

    # Update tracking file
    tracking_path = cfg.paths.batches_dir / f"{batch_id}.json"
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        tracking["status"] = "cancelled"
        tracking_path.write_text(json.dumps(tracking, indent=2) + "\n", encoding="utf-8")


@cli.command()
def poll():
    """Check completed batches and download results.

    This is the entry point for the hourly launchd poller.
    """
    cfg = ensure_config()
    batches_dir = cfg.paths.batches_dir

    if not batches_dir.exists():
        return

    tracking_files = sorted(batches_dir.glob("*.json"))
    if not tracking_files:
        return

    for tf in tracking_files:
        tracking = json.loads(tf.read_text(encoding="utf-8"))
        batch_id = tracking["batch_id"]

        if tracking.get("status") in ("complete", "cancelled", "failed"):
            continue

        try:
            batch_data = check_batch_status(batch_id, cfg.api_keys.mistral)
        except Exception as e:
            logger.error("Error checking batch %s: %s", batch_id, e)
            continue

        current_status = batch_data.get("status", "unknown")
        tracking["status"] = current_status
        tf.write_text(json.dumps(tracking, indent=2) + "\n", encoding="utf-8")

        if current_status not in ("SUCCESS", "COMPLETED", "complete"):
            logger.info("Batch %s: %s", batch_id, current_status)
            continue

        # Download results
        logger.info("Batch %s complete — downloading results", batch_id)
        try:
            results = download_batch_results(batch_data, cfg.api_keys.mistral)
        except Exception as e:
            logger.error("Error downloading batch %s results: %s", batch_id, e)
            continue

        folder_map = {
            cid: Path(p) for cid, p in tracking.get("folders", {}).items()
        }

        for item in results:
            custom_id = item.get("custom_id", "")
            folder = folder_map.get(custom_id)
            if not folder or not folder.exists():
                logger.warning("No folder for custom_id %s", custom_id)
                continue

            md_text = extract_markdown_from_result(item)
            if md_text:
                raw_path = folder / "raw.md"
                raw_path.write_text(md_text, encoding="utf-8")
                write_meta(folder, batch_status="complete", raw_chars=len(md_text))
                logger.info("Downloaded results for %s (%d chars)", custom_id, len(md_text))
            else:
                write_meta(folder, batch_status="failed", batch_error="empty result")
                logger.warning("Empty result for %s", custom_id)

        tracking["status"] = "complete"
        tf.write_text(json.dumps(tracking, indent=2) + "\n", encoding="utf-8")
        logger.info("Batch %s fully processed", batch_id)


@cli.command(name="list")
def list_pending():
    """List scanned PDFs pending in cloud staging."""
    cfg = ensure_config()
    cloud_dir = cfg.paths.staging_cloud

    if not cloud_dir.exists():
        click.echo("No cloud staging directory.")
        return

    for folder in sorted(cloud_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta = read_meta(folder)
        name = meta.get("original_name", folder.name)
        status = meta.get("batch_status", "pending")
        click.echo(f"  {folder.name}  {name:<40}  {status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cli()
