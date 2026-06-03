"""Watcher daemon: monitors ~/inbox/books/, triages files, runs local processing.

Startup behavior: process backlog sorted by mtime, then watch for new drops.
Local conversion uses multiprocessing.Pool for PyMuPDF (not thread-safe).
After conversion + cleanup, kicks off async fan-out.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import shutil
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pipeline.cleanup import process_cleanup
from pipeline.config import PipelineConfig, ensure_config, ensure_directories
from pipeline.epub import process_epub
from pipeline.fanout import fanout
from pipeline.pdf_local import process_local_pdf
from pipeline.queue import acquire_lock, read_meta, release_lock, write_meta
from pipeline.triage import triage_file

logger = logging.getLogger(__name__)


def _process_local_worker(staging_folder_str: str, pdf_engine: str = "local") -> tuple[str, str | None]:
    """Worker function for multiprocessing pool.

    Takes a string path (pickle-safe), returns (folder_name, result).
    Must import inside the function for multiprocessing safety.
    """
    from pathlib import Path
    from pipeline.pdf_local import process_local_pdf
    from pipeline.epub import process_epub
    from pipeline.queue import read_meta

    folder = Path(staging_folder_str)
    meta = read_meta(folder)
    classification = meta.get("classification", "")

    if classification == "epub":
        result = process_epub(folder)
    elif classification == "native":
        if pdf_engine == "cloud":
            from pipeline.pdf_cloud import process_cloud_pdf
            result = process_cloud_pdf(folder)
        else:
            result = process_local_pdf(folder)
    else:
        return folder.name, None

    return folder.name, result


def handle_reclassification(staging_folder: Path, cfg: PipelineConfig) -> None:
    """Move a reclassified-as-scanned folder from local/ to cloud/."""
    meta = read_meta(staging_folder)
    new_folder = cfg.paths.staging_cloud / staging_folder.name
    shutil.move(str(staging_folder), str(new_folder))
    write_meta(new_folder, reclassified_from="local", staging_category="cloud")
    logger.info("Reclassified %s → cloud staging", staging_folder.name)


class InboxHandler(FileSystemEventHandler):
    """Watchdog handler for new files in ~/inbox/books/."""

    def __init__(self, cfg: PipelineConfig, work_queue: asyncio.Queue):
        self.cfg = cfg
        self.work_queue = work_queue

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        # Skip hidden files and partial downloads
        if path.name.startswith(".") or path.suffix.lower() in (".crdownload", ".part", ".tmp"):
            return
        # Small delay to let file finish writing
        time.sleep(1)
        if path.exists():
            logger.info("New file detected: %s", path.name)
            try:
                self.work_queue.put_nowait(path)
            except asyncio.QueueFull:
                logger.warning("Work queue full, dropping %s", path.name)


async def process_one_book(staging_folder: Path, cfg: PipelineConfig) -> None:
    """Process a single book through conversion → cleanup → fan-out."""
    if not acquire_lock(staging_folder):
        logger.debug("Skipping locked folder: %s", staging_folder.name)
        return

    try:
        meta = read_meta(staging_folder)
        classification = meta.get("classification", "")
        original_name = meta.get("original_name", staging_folder.name)

        # Step 1: Convert (in process pool for PyMuPDF safety)
        if not (staging_folder / "raw.md").exists():
            loop = asyncio.get_event_loop()
            pool = multiprocessing.Pool(processes=1)
            try:
                pdf_engine = cfg.conversion.pdf_engine
                name, result = await loop.run_in_executor(
                    None,
                    lambda: pool.apply(_process_local_worker, (str(staging_folder), pdf_engine)),
                )
            finally:
                pool.close()
                pool.join()

            if result == "reclassify":
                release_lock(staging_folder)
                handle_reclassification(staging_folder, cfg)
                return
            elif result is None:
                logger.error("Conversion failed for %s — quarantining", original_name)
                release_lock(staging_folder)
                quarantine_dest = cfg.paths.quarantine / staging_folder.name
                shutil.move(str(staging_folder), str(quarantine_dest))
                return

        # Step 2: Cleanup
        if not (staging_folder / "clean.md").exists():
            success = process_cleanup(staging_folder, cfg)
            if not success:
                logger.error("Cleanup failed for %s", original_name)
                release_lock(staging_folder)
                return

        # Step 3: Fan-out
        results = await fanout(staging_folder, cfg)
        logger.info("Book complete: %s → %s", original_name, results)

    except Exception as e:
        logger.error("Error processing %s: %s", staging_folder.name, e, exc_info=True)
    finally:
        release_lock(staging_folder)


async def process_backlog(cfg: PipelineConfig) -> None:
    """Process any existing files in staging folders."""
    sem = asyncio.Semaphore(cfg.workers.cleanup_concurrency)

    async def _bounded(folder):
        async with sem:
            await process_one_book(folder, cfg)

    tasks = []
    for category in ("local", "epub"):
        cat_dir = cfg.paths.staging / category
        if not cat_dir.exists():
            continue
        for folder in sorted(cat_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            if (folder / ".done").exists():
                continue
            if (folder / "raw.md").exists() or (folder / "source.pdf").exists() or (folder / "source.epub").exists():
                tasks.append(_bounded(folder))

    if tasks:
        logger.info("Processing backlog: %d items", len(tasks))
        await asyncio.gather(*tasks)


async def run_watcher(cfg: PipelineConfig) -> None:
    """Main watcher loop: process backlog, then watch for new files."""
    ensure_directories(cfg)

    # Process any existing files in inbox
    inbox_files = sorted(cfg.paths.inbox.glob("*"), key=lambda p: p.stat().st_mtime)
    for f in inbox_files:
        if f.is_file() and not f.name.startswith("."):
            logger.info("Backlog: triaging %s", f.name)
            triage_file(f, cfg)

    # Process any staging backlog
    await process_backlog(cfg)

    # Watch for new files
    work_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    handler = InboxHandler(cfg, work_queue)
    observer = Observer()
    observer.schedule(handler, str(cfg.paths.inbox), recursive=False)
    observer.start()
    logger.info("Watching %s for new files...", cfg.paths.inbox)

    sem = asyncio.Semaphore(cfg.workers.cleanup_concurrency)

    try:
        while True:
            try:
                file_path = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(2)
                continue

            # Triage the new file
            staging_folder = triage_file(file_path, cfg)
            if staging_folder is None:
                continue

            meta = read_meta(staging_folder)
            if meta.get("staging_category") == "cloud":
                logger.info("%s routed to cloud staging — awaiting batch submit", file_path.name)
                continue

            # Process local/epub files
            async def _bounded(folder):
                async with sem:
                    await process_one_book(folder, cfg)

            asyncio.create_task(_bounded(staging_folder))

    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
    finally:
        observer.stop()
        observer.join()


def main():
    """Entry point for the watcher daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = ensure_config()
    asyncio.run(run_watcher(cfg))


if __name__ == "__main__":
    main()
