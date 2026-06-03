# lib-rag

A RAG pipeline for building a queryable, indexed library from book-length PDFs and EPUBs. Drop a file in, get it converted, indexed, and optionally sent to your Kindle — with hybrid search across your entire collection.

## What it does

1. **Converts** PDFs and EPUBs to clean Markdown (via Gemini 2.5 Flash or PyMuPDF4LLM)
2. **Generates** structured JSON book maps with summaries, themes, and chapter breakdowns
3. **Indexes** into a LanceDB vector store with parent-child chunks and BM25 keyword search
4. **Searches** with hybrid retrieval (semantic + keyword via Reciprocal Rank Fusion)
5. **Optionally** sends converted EPUBs to your Kindle via the Resend email API

## Setup

```bash
git clone https://github.com/grantdever/lib-rag.git
cd lib-rag

# Create venv (Python 3.11+)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Install pandoc (required for EPUB conversion)
brew install pandoc  # macOS

# Add API keys
cp .env.example .env
# Edit .env with your OpenRouter API key (required)
```

**Required:** `OPENROUTER_API_KEY` — used for embeddings (Gemini), map generation (DeepSeek), and PDF conversion (Gemini 2.5 Flash)

**Optional:** `GEMINI_API_KEY` (direct Gemini access), `MISTRAL_API_KEY` (scanned PDF OCR), `RESEND_API_KEY` (Kindle delivery)

## Quick start

```bash
# Add your first book (EPUB)
mkdir -p source/
cp ~/path/to/book.epub source/author-short-title.epub

# Convert to markdown
.venv/bin/python3 scripts/01_convert_epubs.py

# Generate book map (~$0.003 via DeepSeek)
.venv/bin/python3 scripts/02_generate_maps.py --book author-short-title

# Build vector index (~$0.05 via Gemini embeddings)
.venv/bin/python3 scripts/03_build_index.py --provider openrouter --book author-short-title

# Query
.venv/bin/python3 scripts/04_query.py "your question here" --top-k 5
```

## Querying

Once you've indexed a book, you can search it. The repo includes two example book maps (Burke's *Reflections on the Revolution in France* and Nisbet's *Quest for Community*) to show the schema — index your own books to query them.

### Example

```bash
# Hybrid search (semantic + keyword) — best for most queries
$ .venv/bin/python3 scripts/04_query.py "why are inherited institutions important" --top-k 3

[0.0328] Edmund Burke — Reflections on the Revolution in France, Ch. 1: Reflections on the Revolution in France
  You will observe that from Magna Charta to the Declaration of Right it has
  been the uniform policy of our constitution to claim and assert our liberties
  as an entailed inheritance derived to us from our forefathers...
  → markdown/burke-reflections-on-the-revolution.md

[0.0272] Edmund Burke — Reflections on the Revolution in France, Ch. 1: Reflections on the Revolution in France
  By this means our constitution preserves a unity in so great a diversity of
  its parts. We have an inheritable crown, an inheritable peerage, and a House
  of Commons and a people inheriting privileges, franchises, and liberties...
  → markdown/burke-reflections-on-the-revolution.md

# Keyword search — exact phrase hunting
$ .venv/bin/python3 scripts/04_query.py "entailed inheritance" --mode keyword --top-k 3

# JSON output — for piping to other tools or LLMs
$ .venv/bin/python3 scripts/04_query.py "tradition vs abstract reason" --format json --top-k 5
```

### Options

```bash
.venv/bin/python3 scripts/04_query.py "your query" [options]
```

| Flag | Values | Default | Purpose |
|------|--------|---------|---------|
| `--top-k` | 1–20 | 5 | Number of results |
| `--mode` | `hybrid`, `semantic`, `keyword` | `hybrid` | Search strategy |
| `--format` | `json`, `pretty`, `obsidian` | `pretty` | Output format |
| `--provider` | `openrouter`, `gemini` | `openrouter` | Embedding provider |
| `--author-filter` | partial name | none | Filter by author |
| `--book-filter` | exact book_id | none | Filter by book |

**When to use which mode:**
- **`hybrid`** (default) — best for most queries, combines semantic + keyword via Reciprocal Rank Fusion
- **`keyword`** — exact phrase hunting ("entailed inheritance", "social contract")
- **`semantic`** — conceptual queries where the exact words won't appear in the text

## Pipeline architecture

For automated ingestion, drop a PDF or EPUB into `~/inbox/books/`:

```
~/inbox/books/  →  triage  →  convert  →  cleanup  →  fan-out
                     │          │           │           ├── Kindle (Resend API)
                     │          │           │           ├── RAG ingest (map + embed)
                     │          │           │           └── Archive (done/)
                     │          │           └── regex + pandoc + optional DeepSeek
                     │          ├── Native PDF → Gemini 2.5 Flash (default)
                     │          ├── Native PDF → PyMuPDF4LLM (local, free)
                     │          ├── Scanned PDF → Mistral OCR batch (future)
                     │          └── EPUB → pandoc
                     └── classify: native PDF / scanned PDF / EPUB
```

```bash
# Start the watcher daemon
.venv/bin/python3 -m pipeline
```

### How conversion works

**Cloud (default):** Gemini 2.5 Flash via OpenRouter. The PDF is split into ~100-page chunks, each uploaded natively to Gemini. Produces reading-quality markdown with proper paragraphs, dehyphenation, footnote preservation, and header/footer removal. Cost: ~$0.70–0.90 per 500 pages. Batches save incrementally, so interrupted conversions resume where they left off.

**Local:** PyMuPDF4LLM. Free and fast, but lower quality — line-break artifacts, spaced capitals, running headers. Adequate for search indexing but not for reading.

### How indexing works

```
Markdown → JSON Maps + LanceDB Index
              │              │
        Navigation      Retrieval
     (summaries,      (vector + BM25
      themes,          hybrid search)
      chapters)
```

- **Parent chunks** (~800 tokens): BM25-indexed, returned as full context
- **Child chunks** (~256 tokens): vector-indexed via `gemini-embedding-001` (768 dims)
- **Hybrid search**: Reciprocal Rank Fusion merging semantic + keyword results
- **Book maps**: JSON files with author, title, summary, key themes, and per-chapter breakdowns

## Configuration

Config lives at `~/.config/book-pipeline/config.toml` (auto-created on first run). See `config.toml.example` for all options.

| Section | Key settings |
|---------|-------------|
| `[paths]` | `inbox`, `done`, `quarantine`, `staging`, `obsidian_vault` |
| `[conversion]` | `pdf_engine` — `"cloud"` (default) or `"local"` |
| `[cleanup]` | `fuzzy_threshold`, `strip_images`, `strip_tables` |
| `[kindle]` | `enabled`, `kindle_email`, `sender_email` |
| `[workers]` | `local_workers`, `cleanup_concurrency` |

## Kindle delivery (optional)

Send converted EPUBs to your Kindle via [Resend](https://resend.com):

1. Create a Resend account and verify a sending domain
2. Add the sender email to your [Amazon approved senders](https://www.amazon.com/hz/mycd/myx#/home/settings/payment)
3. Set `RESEND_API_KEY` in `.env`
4. Enable in `config.toml`:
   ```toml
   [kindle]
   enabled = true
   kindle_email = "yourname@kindle.com"
   sender_email = "kindle@yourdomain.com"
   ```

EPUBs include metadata (title, author, publisher, date) for proper Kindle display.

## Daemon setup (macOS launchd)

```bash
# Edit plists to set your paths first
# Then install:
cp launchctl/com.user.bookpipeline.watcher.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.bookpipeline.watcher.plist
```

## Project structure

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Batch processing (convert, map, index, query) |
| `pipeline/` | File-drop ingestion (watcher, triage, convert, cleanup, fan-out) |
| `source/` | Original EPUB/PDF files (gitignored) |
| `markdown/` | Cleaned Markdown output (gitignored) |
| `maps/` | JSON book maps (2 examples included to show the schema) |
| `index/` | LanceDB vector store (gitignored) |
| `launchctl/` | macOS launchd plists |

## Future improvements

- **Scanned PDF support via Mistral OCR** — triage already classifies scanned PDFs. The batch API modules exist but haven't been tested end-to-end.
- **Local ML conversion** — tools like [Marker](https://github.com/VikParuchuri/marker) could replace cloud conversion on machines with enough RAM/GPU.
- **Configurable cleanup rules** — regex patterns for PDF artifact removal could be externalized to a rule file.
- **Web UI** — expose the query script via FastAPI or Gradio.

## License

MIT
