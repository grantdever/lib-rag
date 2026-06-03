"""Pipeline configuration — loads from ~/.config/book-pipeline/config.toml with defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root for API keys
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

CONFIG_PATH = Path.home() / ".config" / "book-pipeline" / "config.toml"

# Default template written if config doesn't exist
CONFIG_TEMPLATE = """\
# lib-rag Pipeline Configuration

[paths]
inbox = "~/inbox/books"
done = "~/inbox/done"
quarantine = "~/inbox/quarantine"
staging = "~/staging"
obsidian_vault = "~/obsidian/vault"

# Repo paths (auto-detected, override if needed)
# repo_root = "~/lib-rag"

[workers]
local_workers = 3          # multiprocessing pool size for PyMuPDF/pandoc
cleanup_concurrency = 4    # max in-flight async cleanup tasks

[conversion]
# "local" = PyMuPDF4LLM (free, fast, ok for RAG, poor for reading)
# "cloud" = Gemini 2.5 Flash via OpenRouter (~$0.70/500 pages, reading-quality)
pdf_engine = "cloud"

[cleanup]
# Heuristic quality score below which DeepSeek fuzzy cleanup is invoked
fuzzy_threshold = 0.7
# Strip images and tables from converted markdown
strip_images = true
strip_tables = true

[kindle]
# kindle_email = "yourname@kindle.com"
# sender_email = "you@yourdomain.com"
# Requires a verified sending domain in Resend (resend.com/domains)
# resend_api_key is read from .env (RESEND_API_KEY)
enabled = false

[api_keys]
# These are read from .env by default. Override here if preferred.
# openrouter = "sk-or-..."
# gemini = "AIza..."
# mistral = "..."

[batch]
poll_interval_minutes = 60
"""


@dataclass
class PathsConfig:
    inbox: Path = field(default_factory=lambda: Path.home() / "inbox" / "books")
    done: Path = field(default_factory=lambda: Path.home() / "inbox" / "done")
    quarantine: Path = field(default_factory=lambda: Path.home() / "inbox" / "quarantine")
    staging: Path = field(default_factory=lambda: Path.home() / "staging")
    obsidian_vault: Path = field(default_factory=lambda: Path.home() / "obsidian" / "vault")
    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)

    @property
    def staging_local(self) -> Path:
        return self.staging / "local"

    @property
    def staging_epub(self) -> Path:
        return self.staging / "epub"

    @property
    def staging_cloud(self) -> Path:
        return self.staging / "cloud"

    @property
    def batches_dir(self) -> Path:
        return self.staging / "cloud" / ".batches"

    @property
    def markdown_dir(self) -> Path:
        return self.repo_root / "markdown"

    @property
    def maps_dir(self) -> Path:
        return self.repo_root / "maps"

    @property
    def index_dir(self) -> Path:
        return self.repo_root / "index"


@dataclass
class WorkersConfig:
    local_workers: int = 3
    cleanup_concurrency: int = 4


@dataclass
class ConversionConfig:
    pdf_engine: str = "cloud"  # "cloud" (Gemini 2.5 Flash via OpenRouter) or "local" (PyMuPDF4LLM)


@dataclass
class CleanupConfig:
    fuzzy_threshold: float = 0.7
    strip_images: bool = True
    strip_tables: bool = True


@dataclass
class KindleConfig:
    enabled: bool = False
    kindle_email: str = ""
    sender_email: str = ""
    resend_api_key: str = ""


@dataclass
class BatchConfig:
    poll_interval_minutes: int = 60


@dataclass
class ApiKeys:
    openrouter: str = ""
    gemini: str = ""
    mistral: str = ""


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    kindle: KindleConfig = field(default_factory=KindleConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    api_keys: ApiKeys = field(default_factory=ApiKeys)


def _expand_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from TOML file + environment variables.

    Priority: TOML values → .env values → defaults.
    """
    path = config_path or CONFIG_PATH
    raw = _load_toml(path)

    cfg = PipelineConfig()

    # Paths
    paths_raw = raw.get("paths", {})
    if "inbox" in paths_raw:
        cfg.paths.inbox = _expand_path(paths_raw["inbox"])
    if "done" in paths_raw:
        cfg.paths.done = _expand_path(paths_raw["done"])
    if "quarantine" in paths_raw:
        cfg.paths.quarantine = _expand_path(paths_raw["quarantine"])
    if "staging" in paths_raw:
        cfg.paths.staging = _expand_path(paths_raw["staging"])
    if "obsidian_vault" in paths_raw:
        cfg.paths.obsidian_vault = _expand_path(paths_raw["obsidian_vault"])
    if "repo_root" in paths_raw:
        cfg.paths.repo_root = _expand_path(paths_raw["repo_root"])

    # Workers
    workers_raw = raw.get("workers", {})
    if "local_workers" in workers_raw:
        cfg.workers.local_workers = int(workers_raw["local_workers"])
    if "cleanup_concurrency" in workers_raw:
        cfg.workers.cleanup_concurrency = int(workers_raw["cleanup_concurrency"])

    # Conversion
    conversion_raw = raw.get("conversion", {})
    if "pdf_engine" in conversion_raw:
        cfg.conversion.pdf_engine = str(conversion_raw["pdf_engine"])

    # Cleanup
    cleanup_raw = raw.get("cleanup", {})
    if "fuzzy_threshold" in cleanup_raw:
        cfg.cleanup.fuzzy_threshold = float(cleanup_raw["fuzzy_threshold"])
    if "strip_images" in cleanup_raw:
        cfg.cleanup.strip_images = bool(cleanup_raw["strip_images"])
    if "strip_tables" in cleanup_raw:
        cfg.cleanup.strip_tables = bool(cleanup_raw["strip_tables"])

    # Kindle
    kindle_raw = raw.get("kindle", {})
    for k in ("enabled", "kindle_email", "sender_email"):
        if k in kindle_raw:
            setattr(cfg.kindle, k, kindle_raw[k])
    cfg.kindle.resend_api_key = kindle_raw.get("resend_api_key") or os.getenv("RESEND_API_KEY", "")

    # Batch
    batch_raw = raw.get("batch", {})
    if "poll_interval_minutes" in batch_raw:
        cfg.batch.poll_interval_minutes = int(batch_raw["poll_interval_minutes"])

    # API keys: TOML overrides → .env fallback
    keys_raw = raw.get("api_keys", {})
    cfg.api_keys.openrouter = keys_raw.get("openrouter") or os.getenv("OPENROUTER_API_KEY", "")
    cfg.api_keys.gemini = keys_raw.get("gemini") or os.getenv("GEMINI_API_KEY", "")
    cfg.api_keys.mistral = keys_raw.get("mistral") or os.getenv("MISTRAL_API_KEY", "")

    return cfg


def ensure_config() -> PipelineConfig:
    """Load config, creating the template file if it doesn't exist."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return load_config()


def ensure_directories(cfg: PipelineConfig) -> None:
    """Create all required directories if they don't exist."""
    for d in (
        cfg.paths.inbox,
        cfg.paths.done,
        cfg.paths.quarantine,
        cfg.paths.staging_local,
        cfg.paths.staging_epub,
        cfg.paths.staging_cloud,
        cfg.paths.batches_dir,
        cfg.paths.markdown_dir,
        cfg.paths.maps_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
