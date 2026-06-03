"""Adapter to the existing RAG ingestion pipeline.

Copies clean.md to the repo's markdown/ dir, generates a JSON book map
via DeepSeek, then chunks + embeds via Gemini into LanceDB.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.queue import read_meta, write_meta

logger = logging.getLogger(__name__)


def slugify_book_name(original_name: str) -> str:
    """Convert an original filename to a book_id slug.

    'Nisbet - Quest for Community.pdf' → 'nisbet-quest-for-community'
    """
    import re

    name = Path(original_name).stem
    # Remove common prefixes/suffixes
    name = re.sub(r"\s*\(.*?\)\s*$", "", name)  # trailing (year) etc
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def copy_to_library(staging_folder: Path, cfg: PipelineConfig) -> tuple[str, Path]:
    """Copy clean.md to the repo's markdown/ dir.

    Returns (book_id, md_path).
    """
    meta = read_meta(staging_folder)
    original_name = meta.get("original_name", staging_folder.name)
    book_id = slugify_book_name(original_name)

    clean_md = staging_folder / "clean.md"
    if not clean_md.exists():
        raise FileNotFoundError(f"No clean.md in {staging_folder}")

    md_path = cfg.paths.markdown_dir / f"{book_id}.md"
    shutil.copy2(str(clean_md), str(md_path))
    logger.info("Copied clean.md → %s", md_path)

    return book_id, md_path


def generate_book_map(book_id: str, md_path: Path, cfg: PipelineConfig) -> bool:
    """Generate a JSON book map using the existing 02_generate_maps.py logic.

    Imports from the existing scripts/ directory.
    """
    # Add scripts dir to path so we can import
    scripts_dir = cfg.paths.repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from shared import get_llm_client
        # Import the map generation functions
        # Use importlib to handle the numeric filename
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_maps", scripts_dir / "02_generate_maps.py"
        )
        maps_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maps_module)

        client, model = get_llm_client("openrouter")
        result = maps_module.process_one(client, model, md_path, workers=4, verbose=False)

        if result.get("warnings"):
            logger.warning("Map warnings for %s: %s", book_id, result["warnings"])

        logger.info("Generated map for %s", book_id)
        return True

    except Exception as e:
        logger.error("Map generation failed for %s: %s", book_id, e)
        return False


def index_book(book_id: str, md_path: Path, cfg: PipelineConfig) -> bool:
    """Index a book into LanceDB using the existing 03_build_index.py logic.

    Imports from the existing scripts/ directory.
    """
    scripts_dir = cfg.paths.repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        import threading
        import lancedb

        from shared import EMBED_DIMS, make_embed_fn

        # Import the indexing functions
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_index", scripts_dir / "03_build_index.py"
        )
        index_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(index_module)

        embed_batch, _ = make_embed_fn("openrouter")
        db = lancedb.connect(str(cfg.paths.index_dir))

        # Open or create tables
        table_names = db.table_names()
        if "parents" in table_names:
            parents_table = db.open_table("parents")
        else:
            parents_table = db.create_table("parents", schema=index_module.PARENT_SCHEMA)

        if "children" in table_names:
            children_table = db.open_table("children")
        else:
            children_table = db.create_table("children", schema=index_module.CHILD_SCHEMA)

        # Chunk the book
        book_data = index_module.prepare_book(md_path)
        logger.info(
            "%s: %d parents, %d children to embed",
            book_id,
            len(book_data["parent_records"]),
            len(book_data["child_records"]),
        )

        # Embed and store
        db_lock = threading.Lock()
        stats = index_module.embed_and_store(
            embed_batch, parents_table, children_table, book_data, db_lock
        )

        # Rebuild FTS index
        try:
            parents_table.create_fts_index("text", replace=True)
        except Exception as e:
            logger.warning("FTS index rebuild warning: %s", e)

        logger.info(
            "%s: indexed %d parents, %d children",
            book_id,
            stats["parents"],
            stats["children"],
        )
        return True

    except Exception as e:
        logger.error("Indexing failed for %s: %s", book_id, e)
        return False


def process_rag_ingest(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Full RAG ingestion: copy → map → embed.

    Returns True if all steps succeed.
    """
    try:
        book_id, md_path = copy_to_library(staging_folder, cfg)
    except Exception as e:
        logger.error("Copy to library failed: %s", e)
        return False

    write_meta(staging_folder, book_id=book_id, md_path=str(md_path))

    map_ok = generate_book_map(book_id, md_path, cfg)
    if not map_ok:
        logger.warning("Map generation failed for %s — continuing to index anyway", book_id)

    index_ok = index_book(book_id, md_path, cfg)

    write_meta(
        staging_folder,
        rag_map_generated=map_ok,
        rag_indexed=index_ok,
    )

    return index_ok
