"""File-based queue with lockfiles for staging folder coordination.

Each book gets a UUID folder in staging. A `.lock` file prevents concurrent
processing. `meta.json` tracks state, scores, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_staging_folder(staging_root: Path, category: str) -> Path:
    """Create a new UUID-named staging folder under staging_root/category/.

    Returns the folder path.
    """
    folder_id = uuid.uuid4().hex[:12]
    folder = staging_root / category / folder_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_meta(folder: Path, **kwargs) -> Path:
    """Write or update meta.json in a staging folder.

    Merges kwargs into existing meta if present.
    """
    meta_path = folder / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}

    meta.update(kwargs)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta_path


def read_meta(folder: Path) -> dict:
    """Read meta.json from a staging folder. Returns empty dict if missing."""
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def acquire_lock(folder: Path) -> bool:
    """Try to acquire a .lock file on a staging folder.

    Returns True if lock acquired, False if already locked.
    Uses atomic file creation to prevent races.
    """
    lock_path = folder / ".lock"
    try:
        fd = lock_path.open("x")  # O_CREAT | O_EXCL — atomic
        fd.write(json.dumps({
            "pid": __import__("os").getpid(),
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }))
        fd.close()
        return True
    except FileExistsError:
        return False


def release_lock(folder: Path) -> None:
    """Release the .lock file on a staging folder."""
    lock_path = folder / ".lock"
    lock_path.unlink(missing_ok=True)


def is_locked(folder: Path) -> bool:
    """Check if a staging folder is locked."""
    return (folder / ".lock").exists()


def find_pending_folders(staging_root: Path, category: str) -> list[Path]:
    """Find staging folders that are not locked and not yet processed.

    A folder is pending if it has meta.json but no .lock and no clean.md.
    Returns folders sorted by creation time (oldest first).
    """
    category_dir = staging_root / category
    if not category_dir.exists():
        return []

    pending = []
    for folder in category_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.startswith("."):
            continue
        if is_locked(folder):
            continue
        meta = read_meta(folder)
        if not meta:
            continue
        # Already processed if clean.md exists
        if (folder / "clean.md").exists():
            continue
        pending.append((meta.get("created_at", ""), folder))

    pending.sort(key=lambda x: x[0])
    return [f for _, f in pending]


def find_ready_for_fanout(staging_root: Path, category: str) -> list[Path]:
    """Find staging folders that have clean.md but haven't been fanned out.

    A folder is ready for fan-out if clean.md exists but done marker doesn't.
    """
    category_dir = staging_root / category
    if not category_dir.exists():
        return []

    ready = []
    for folder in category_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if (folder / "clean.md").exists() and not (folder / ".done").exists():
            if not is_locked(folder):
                ready.append(folder)
    return ready


def mark_done(folder: Path) -> None:
    """Mark a staging folder as fully processed."""
    (folder / ".done").touch()
