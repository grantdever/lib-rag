"""Fan-out: run Kindle delivery, RAG ingestion, and archive in parallel.

After clean.md is produced, three branches run via asyncio.gather:
A. Kindle delivery (SMTP)
B. RAG ingestion (DeepSeek map + Gemini embeddings + LanceDB)
C. Archive (move source to done, remove staging)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.embed import process_rag_ingest
from pipeline.kindle import process_kindle
from pipeline.queue import mark_done, read_meta, write_meta

logger = logging.getLogger(__name__)


async def _run_kindle(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Run Kindle delivery in a thread (SMTP is blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_kindle, staging_folder, cfg)


async def _run_rag(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Run RAG ingestion in a thread (embeds are blocking API calls)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_rag_ingest, staging_folder, cfg)


async def _run_archive(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Archive source file and clean up staging folder."""
    try:
        meta = read_meta(staging_folder)
        original_name = meta.get("original_name", "unknown")

        # Copy source to done/
        for source_file in staging_folder.glob("source.*"):
            done_dir = cfg.paths.done / staging_folder.name
            done_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_file), str(done_dir / original_name))
            # Also copy meta for provenance
            meta_src = staging_folder / "meta.json"
            if meta_src.exists():
                shutil.copy2(str(meta_src), str(done_dir / "meta.json"))
            break

        mark_done(staging_folder)
        logger.info("Archived %s → %s", original_name, cfg.paths.done / staging_folder.name)
        return True

    except Exception as e:
        logger.error("Archive failed for %s: %s", staging_folder.name, e)
        return False


async def fanout(staging_folder: Path, cfg: PipelineConfig) -> dict:
    """Run all three fan-out branches in parallel.

    Returns dict with results of each branch.
    """
    meta = read_meta(staging_folder)
    original_name = meta.get("original_name", staging_folder.name)
    logger.info("Fan-out starting for %s", original_name)

    kindle_task = _run_kindle(staging_folder, cfg)
    rag_task = _run_rag(staging_folder, cfg)
    archive_task = _run_archive(staging_folder, cfg)

    kindle_ok, rag_ok, archive_ok = await asyncio.gather(
        kindle_task, rag_task, archive_task,
        return_exceptions=True,
    )

    # Handle exceptions from gather
    results = {}
    for name, result in [("kindle", kindle_ok), ("rag", rag_ok), ("archive", archive_ok)]:
        if isinstance(result, Exception):
            logger.error("Fan-out %s failed: %s", name, result)
            results[name] = False
        else:
            results[name] = result

    write_meta(
        staging_folder,
        fanout_kindle=results.get("kindle", False),
        fanout_rag=results.get("rag", False),
        fanout_archive=results.get("archive", False),
    )

    logger.info("Fan-out complete for %s: %s", original_name, results)
    return results
