# CLAUDE.md — lib-rag

Instructions for Claude Code working in this repo.

## What this repo is

A RAG pipeline for building a queryable library from book-length PDFs and EPUBs. Three layers: source files → JSON book maps → LanceDB vector index with hybrid search.

## Environment

- **Python 3.11 venv** at `.venv/` — always use `.venv/bin/python3`, not system Python
- **API keys** in `.env` (not committed): `OPENROUTER_API_KEY` required, others optional
- **Config** at `~/.config/book-pipeline/config.toml` (auto-created on first run)

## Repo structure

```
lib-rag/
├── scripts/
│   ├── shared.py                 # Shared utilities (API clients, embedding factories, retry logic)
│   ├── 01_convert_epubs.py       # EPUB → clean Markdown
│   ├── 02_generate_maps.py       # Markdown → JSON book maps (via DeepSeek)
│   ├── 03_build_index.py         # Markdown → LanceDB vector index (via Gemini embeddings)
│   └── 04_query.py               # Query the index (main tool)
├── pipeline/                     # File-drop ingestion pipeline
│   ├── config.py                 # TOML config loader
│   ├── triage.py                 # PDF/EPUB classification
│   ├── pdf_cloud.py              # PDF → Markdown via Gemini 2.5 Flash
│   ├── pdf_local.py              # PDF → Markdown via PyMuPDF4LLM
│   ├── cleanup.py                # Markdown cleanup (regex + pandoc + optional DeepSeek)
│   ├── kindle.py                 # Send to Kindle via Resend API
│   ├── fanout.py                 # Parallel fan-out (Kindle + RAG + archive)
│   └── watcher.py                # Watchdog daemon
├── maps/                         # JSON book maps
├── index/                        # LanceDB vector store (gitignored)
├── source/                       # Original files (gitignored)
└── markdown/                     # Cleaned Markdown (gitignored)
```

## Querying

```bash
.venv/bin/python3 scripts/04_query.py "<query>" --top-k 8 --format json --provider openrouter
```

| Flag | Values | Default |
|------|--------|---------|
| `--top-k` | 1–20 | 5 |
| `--mode` | `hybrid`, `semantic`, `keyword` | `hybrid` |
| `--format` | `json`, `pretty`, `obsidian` | `pretty` |
| `--provider` | `openrouter`, `gemini` | `openrouter` |
| `--author-filter` | partial name | none |
| `--book-filter` | exact book_id | none |

## Adding books

### File-drop (pipeline)
Drop PDF/EPUB into `~/inbox/books/`. Start watcher: `.venv/bin/python3 -m pipeline`

### Manual (scripts)
```bash
.venv/bin/python3 scripts/01_convert_epubs.py
.venv/bin/python3 scripts/02_generate_maps.py --book <book-id>
.venv/bin/python3 scripts/03_build_index.py --provider openrouter --book <book-id>
```

## Architecture

- **Book maps** (`maps/*.json`): LLM-generated summaries with themes and chapter breakdowns. Navigation layer.
- **Parent chunks** (~800 tokens): BM25-indexed for keyword search. Returned as context.
- **Child chunks** (~256 tokens): vector-indexed via Gemini embeddings (768 dims). Used for semantic retrieval.
- **Hybrid search**: RRF fusion of vector + BM25 results.

## Key conventions

- Book IDs: `author-short-title` (e.g., `burke-reflections-on-the-revolution`)
- Maps are navigation aids, not authoritative — source markdown is canonical
- When map summary and retrieved passage disagree, the passage wins
- Never fabricate quotes or chapter references
