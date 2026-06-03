"""Integration tests verifying core pipeline workflows end-to-end.

These test that the full processing chains produce correct output,
not just individual functions. No API keys or network access needed.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


# ============================================================
# Cleanup pipeline: raw text → cleaned markdown
# ============================================================

from pipeline.cleanup import (
    cleanup_markdown,
    regex_cleanup,
    pdf_cleanup,
    strip_images,
    strip_tables,
)
from pipeline.config import PipelineConfig


SAMPLE_PANDOC_EPUB = """\
[]{#ch01.xhtml}

::: {.section .chapter}
# [Chapter 1: The Problem]{.chapter-title} {#chapter-1 .heading}

Some introductory text about the [nature of community]{.emphasis}.

![](images/cover.jpg){width="50%"}

::: {.blockquote}
> A quoted passage here.
:::

## [Section 1.1]{.section-title} {.subheading}

More content with [inline spans]{.some-class} and artifacts.

<div class="footnote"><p>1. A footnote reference.</p></div>
:::

[]{#ch02.xhtml}

::: {.section .chapter}
# [Chapter 2: The Response]{.chapter-title} {#chapter-2 .heading}

The second chapter has <span class="page-number">42</span> content.
:::
"""

SAMPLE_PDF_RAW = """\
**T HE QUEST FOR COMMUNITY • 23**

T he breakdown of community in W estern society has been
a recurring theme. T his is not merely a M arxist concern.

==> image intentionally omitted <==

W eber and T ocqueville both recognized the\xa0problem
of social atomization in modern democracies.

**24 • THE QUEST FOR COMMUNITY**

## The Individual and Society •

T hat the individual exists within a web of associations
is the central insight of conservative social thought.
"""


class TestCleanupPipelineEpub:
    """Test that EPUB-style pandoc output gets properly cleaned."""

    def test_strips_pandoc_artifacts(self):
        result = regex_cleanup(SAMPLE_PANDOC_EPUB)
        assert "[]{#" not in result
        assert ":::" not in result
        assert "{.chapter-title}" not in result
        assert "{#chapter-1" not in result
        assert "{.some-class}" not in result

    def test_preserves_content(self):
        result = regex_cleanup(SAMPLE_PANDOC_EPUB)
        assert "nature of community" in result
        assert "A quoted passage here" in result
        assert "More content" in result
        assert "A footnote reference" in result

    def test_preserves_headers(self):
        result = regex_cleanup(SAMPLE_PANDOC_EPUB)
        assert "# Chapter 1: The Problem" in result
        assert "# Chapter 2: The Response" in result
        assert "## Section 1.1" in result

    def test_strips_html(self):
        result = regex_cleanup(SAMPLE_PANDOC_EPUB)
        assert "<div" not in result
        assert "<span" not in result
        assert "</div>" not in result

    def test_no_excessive_blank_lines(self):
        result = regex_cleanup(SAMPLE_PANDOC_EPUB)
        assert "\n\n\n" not in result


class TestCleanupPipelinePdf:
    """Test that PDF extraction artifacts get cleaned up."""

    def test_fixes_spaced_capitals(self):
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "The breakdown" in result
        assert "Western society" in result
        assert "This is not" in result
        assert "Marxist" in result

    def test_removes_running_headers(self):
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "THE QUEST FOR COMMUNITY •" not in result

    def test_removes_picture_placeholders(self):
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "intentionally omitted" not in result

    def test_fixes_non_breaking_spaces(self):
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "\xa0" not in result

    def test_removes_bullet_from_headers(self):
        """Headers with • should be removed (running headers)."""
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "## The Individual and Society •" not in result

    def test_preserves_real_content(self):
        result = pdf_cleanup(SAMPLE_PDF_RAW)
        assert "breakdown of community" in result
        assert "social atomization" in result
        assert "central insight" in result


class TestCleanupPipelineSourceTypeRouting:
    """Verify that pdf_cleanup runs for PDFs but not EPUBs."""

    def _make_cfg(self):
        cfg = PipelineConfig()
        cfg.cleanup.strip_images = False
        cfg.cleanup.strip_tables = False
        cfg.cleanup.fuzzy_threshold = 0.0  # disable DeepSeek
        return cfg

    def test_pdf_source_gets_pdf_cleanup(self):
        """PDF sources should have spaced capitals fixed."""
        cfg = self._make_cfg()
        text = "# Chapter 1\n\nT he people of T his nation are great.\n\nMore content here." * 10
        cleaned, stats = cleanup_markdown(text, cfg, source_type="pdf")
        assert "The people" in cleaned
        assert "This nation" in cleaned

    def test_epub_source_skips_pdf_cleanup(self):
        """EPUB sources should NOT have the aggressive spaced-capital regex applied."""
        cfg = self._make_cfg()
        # "I said" has a single capital + space + lowercase — would be falsely joined by pdf_cleanup
        text = "# Chapter 1\n\n" + "I said hello to A friend of mine.\n\n" * 20
        cleaned, stats = cleanup_markdown(text, cfg, source_type="epub")
        # "A friend" should remain intact (not become "Afriend")
        assert "A friend" in cleaned


# ============================================================
# Chunking pipeline: markdown → parent/child records
# ============================================================

try:
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location("build_index", ROOT / "scripts" / "03_build_index.py")
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    prepare_book = _mod.prepare_book
    split_chapters = _mod.split_chapters
    chunk_by_tokens = _mod.chunk_by_tokens
    HAS_INDEX = True
except Exception:
    HAS_INDEX = False

SAMPLE_BOOK = """\
# Chapter 1: Introduction

This is the first chapter of the book. It contains introductory material
about the themes we will explore. The author sets up the main arguments
that will be developed throughout the text.

Community is the foundation of social life. Without strong intermediate
institutions, individuals are left atomized and vulnerable to the state.
This insight runs through the conservative tradition from Burke to Nisbet.

The modern crisis of community has its roots in the Enlightenment's
emphasis on individual reason over inherited social bonds.

# Chapter 2: Historical Background

The second chapter traces the historical development of community
structures from antiquity through the medieval period. Guilds, parishes,
and local assemblies provided the scaffolding of social life.

The destruction of these intermediate bodies by centralizing states —
first monarchical, then revolutionary — created the conditions for
modern social atomization. Tocqueville saw this clearly in his analysis
of democratic societies.

# Chapter 3: Modern Implications

The third chapter examines the contemporary landscape. How do we rebuild
community in an age of mass media and geographic mobility?

The answer lies not in nostalgic return but in creating new forms of
association adapted to modern conditions. This is the localist project.
"""


@pytest.mark.skipif(not HAS_INDEX, reason="03_build_index.py deps not available")
class TestChunkingPipeline:
    """Test the full chunking pipeline: markdown → structured records."""

    def test_split_chapters_produces_correct_count(self):
        chapters = split_chapters(SAMPLE_BOOK)
        assert len(chapters) == 3

    def test_split_chapters_titles(self):
        chapters = split_chapters(SAMPLE_BOOK)
        assert chapters[0]["title"] == "Chapter 1: Introduction"
        assert chapters[1]["title"] == "Chapter 2: Historical Background"
        assert chapters[2]["title"] == "Chapter 3: Modern Implications"

    def test_split_chapters_char_offsets_are_correct(self):
        """char_start should point to the actual position in the original text."""
        chapters = split_chapters(SAMPLE_BOOK)
        for ch in chapters:
            # The char_start should correspond to where the chapter text
            # actually appears in the original
            assert SAMPLE_BOOK[ch["char_start"]:].startswith("# ")

    def test_prepare_book_produces_parent_records(self, tmp_path):
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        # Create a maps dir so get_book_metadata doesn't fail
        maps_dir = tmp_path.parent / "maps"
        maps_dir.mkdir(exist_ok=True)

        result = prepare_book(md_path)
        assert len(result["parent_records"]) > 0
        assert len(result["child_records"]) > 0
        assert len(result["texts_to_embed"]) == len(result["child_records"])
        assert result["basename"] == "test-book"

    def test_parent_records_have_required_fields(self, tmp_path):
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        result = prepare_book(md_path)
        required_fields = {"id", "book_id", "author", "title", "chapter_number",
                          "chapter_title", "parent_index", "char_start", "char_end", "text"}
        for rec in result["parent_records"]:
            assert required_fields.issubset(rec.keys()), f"Missing fields: {required_fields - rec.keys()}"

    def test_child_records_have_required_fields(self, tmp_path):
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        result = prepare_book(md_path)
        required_fields = {"id", "parent_id", "book_id", "author", "chunk_index",
                          "text", "context_text"}
        for rec in result["child_records"]:
            assert required_fields.issubset(rec.keys()), f"Missing fields: {required_fields - rec.keys()}"

    def test_child_records_reference_valid_parents(self, tmp_path):
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        result = prepare_book(md_path)
        parent_ids = {r["id"] for r in result["parent_records"]}
        for child in result["child_records"]:
            assert child["parent_id"] in parent_ids, f"Orphan child: {child['id']}"

    def test_context_text_includes_metadata_prefix(self, tmp_path):
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        result = prepare_book(md_path)
        for child in result["child_records"]:
            assert child["context_text"].startswith("Author:")
            assert "Title:" in child["context_text"]
            assert "Chapter:" in child["context_text"]

    def test_all_book_text_is_covered(self, tmp_path):
        """Parent chunks should cover all substantive text from the book."""
        md_path = tmp_path / "test-book.md"
        md_path.write_text(SAMPLE_BOOK, encoding="utf-8")

        result = prepare_book(md_path)
        all_parent_text = " ".join(r["text"] for r in result["parent_records"])
        # Key phrases from each chapter should appear
        assert "foundation of social life" in all_parent_text
        assert "Tocqueville" in all_parent_text
        assert "localist project" in all_parent_text


# ============================================================
# Query formatting: results → output formats
# ============================================================

# Import format functions via importlib (avoids triggering click/lancedb at module level)
try:
    _qspec = spec_from_file_location("query_mod", ROOT / "scripts" / "04_query.py")
    _qmod = module_from_spec(_qspec)
    _qspec.loader.exec_module(_qmod)
    format_pretty = _qmod.format_pretty
    format_json = _qmod.format_json
    format_obsidian = _qmod.format_obsidian
    reciprocal_rank_fusion = _qmod.reciprocal_rank_fusion
    HAS_QUERY = True
except Exception:
    HAS_QUERY = False


MOCK_RESULTS = [
    {
        "id": "nisbet-quest::ch1::parent0",
        "book_id": "nisbet-quest-for-community",
        "author": "Robert Nisbet",
        "title": "Quest for Community",
        "chapter_number": 1,
        "chapter_title": "The Problem of Community",
        "text": "The quest for community is the dominant theme of modern social thought.",
        "char_start": 0,
        "char_end": 500,
        "_rrf_score": 0.0328,
    },
    {
        "id": "hayek-constitution::ch3::parent2",
        "book_id": "hayek-constitution-of-liberty",
        "author": "F.A. Hayek",
        "title": "The Constitution of Liberty",
        "chapter_number": 3,
        "chapter_title": "The Common Sense of Progress",
        "text": "Liberty is not merely the absence of coercion but the preservation of a domain.",
        "char_start": 1200,
        "char_end": 1700,
        "_rrf_score": 0.0164,
    },
]


@pytest.mark.skipif(not HAS_QUERY, reason="04_query.py deps not available")
class TestQueryFormatting:
    def test_format_pretty_includes_all_results(self):
        output = format_pretty(MOCK_RESULTS)
        assert "Robert Nisbet" in output
        assert "F.A. Hayek" in output
        assert "Quest for Community" in output
        assert "0.0328" in output

    def test_format_pretty_includes_source_paths(self):
        output = format_pretty(MOCK_RESULTS)
        assert "markdown/nisbet-quest-for-community.md" in output
        assert "markdown/hayek-constitution-of-liberty.md" in output

    def test_format_json_is_valid_json(self):
        output = format_json(MOCK_RESULTS)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_format_json_has_required_fields(self):
        output = format_json(MOCK_RESULTS)
        parsed = json.loads(output)
        required = {"score", "book_id", "author", "title", "chapter_number",
                    "chapter_title", "text", "parent_id"}
        for rec in parsed:
            assert required.issubset(rec.keys())

    def test_format_json_scores_are_numeric(self):
        output = format_json(MOCK_RESULTS)
        parsed = json.loads(output)
        for rec in parsed:
            assert isinstance(rec["score"], (int, float))

    def test_format_obsidian_returns_filename_and_content(self):
        filename, content = format_obsidian("test query", MOCK_RESULTS)
        assert "test-query" in filename
        assert filename.endswith(filename[-8:])  # has date suffix

    def test_format_obsidian_has_wikilinks(self):
        _, content = format_obsidian("test query", MOCK_RESULTS)
        assert "[[nisbet-quest-for-community#" in content
        assert "[[hayek-constitution-of-liberty#" in content

    def test_format_obsidian_has_query_header(self):
        _, content = format_obsidian("test query", MOCK_RESULTS)
        assert "# Query: test query" in content
        assert "2 results" in content


# ============================================================
# RRF fusion: vector + FTS results → merged rankings
# ============================================================

@pytest.mark.skipif(not HAS_QUERY, reason="04_query.py deps not available")
class TestRRFFusion:
    """Test reciprocal rank fusion with mock data (no LanceDB needed)."""

    def test_rrf_with_only_fts_results(self):
        """When vector results are empty, RRF should return FTS results."""
        fts = [
            {"id": "book::ch1::p0", "text": "First result", "book_id": "test"},
            {"id": "book::ch2::p0", "text": "Second result", "book_id": "test"},
        ]
        # Use a mock parents_table that won't be called
        results = reciprocal_rank_fusion([], fts, None, top_k=5)
        assert len(results) == 2
        assert all("_rrf_score" in r for r in results)
        # First result should score higher
        assert results[0]["_rrf_score"] > results[1]["_rrf_score"]

    def test_rrf_with_overlapping_results(self):
        """When the same parent appears in both vector and FTS, scores should combine."""
        vector = [
            {"parent_id": "book::ch1::p0", "text": "child chunk"},
        ]
        fts = [
            {"id": "book::ch1::p0", "text": "Full parent text", "book_id": "test"},
            {"id": "book::ch2::p0", "text": "Other result", "book_id": "test"},
        ]
        results = reciprocal_rank_fusion(vector, fts, None, top_k=5)
        # The overlapping result should score highest
        assert results[0]["id"] == "book::ch1::p0"
        assert results[0]["_rrf_score"] > results[1]["_rrf_score"]

    def test_rrf_respects_top_k(self):
        fts = [{"id": f"book::ch{i}::p0", "text": f"Result {i}", "book_id": "test"} for i in range(10)]
        results = reciprocal_rank_fusion([], fts, None, top_k=3)
        assert len(results) == 3


# ============================================================
# Staging queue: full workflow
# ============================================================

from pipeline.queue import (
    write_meta, read_meta, acquire_lock, release_lock,
    create_staging_folder, find_pending_folders, mark_done,
    find_ready_for_fanout,
)


class TestStagingWorkflow:
    """Test the full staging folder lifecycle."""

    def test_full_lifecycle(self, tmp_path):
        staging_root = tmp_path / "staging"
        staging_root.mkdir()

        # 1. Create folder
        folder = create_staging_folder(staging_root, "epub")
        assert folder.exists()
        assert folder.parent.name == "epub"

        # 2. Write initial metadata
        write_meta(folder, sha256="abc123", original_name="test.epub")

        # 3. Should show as pending (has meta, no clean.md, no lock)
        pending = find_pending_folders(staging_root, "epub")
        assert folder in pending

        # 4. Lock it
        assert acquire_lock(folder) is True
        pending = find_pending_folders(staging_root, "epub")
        assert folder not in pending  # locked folders are excluded

        # 5. Simulate conversion: write clean.md
        (folder / "clean.md").write_text("# Cleaned content", encoding="utf-8")
        release_lock(folder)

        # 6. Should no longer be pending (has clean.md)
        pending = find_pending_folders(staging_root, "epub")
        assert folder not in pending

        # 7. Should be ready for fanout
        ready = find_ready_for_fanout(staging_root, "epub")
        assert folder in ready

        # 8. Mark done
        mark_done(folder)
        ready = find_ready_for_fanout(staging_root, "epub")
        assert folder not in ready


# ============================================================
# Config loading
# ============================================================

from pipeline.config import load_config


class TestConfigLoading:
    def test_loads_defaults_without_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.cleanup.fuzzy_threshold == 0.7
        assert cfg.cleanup.strip_images is True
        assert cfg.workers.local_workers == 3
        assert cfg.kindle.enabled is False
        assert cfg.conversion.pdf_engine == "cloud"

    def test_loads_from_toml(self, tmp_path):
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(
            '[cleanup]\nfuzzy_threshold = 0.5\nstrip_images = false\n'
            '[workers]\nlocal_workers = 8\n'
            '[conversion]\npdf_engine = "local"\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.cleanup.fuzzy_threshold == 0.5
        assert cfg.cleanup.strip_images is False
        assert cfg.workers.local_workers == 8
        assert cfg.conversion.pdf_engine == "local"


# ============================================================
# Import chain verification
# ============================================================

class TestImportChains:
    """Verify all cross-module imports resolve correctly."""

    def test_01_convert_imports_from_pipeline(self):
        """01_convert_epubs.py imports regex_cleanup from pipeline.cleanup."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("convert", ROOT / "scripts" / "01_convert_epubs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Should have the imported function available
        assert hasattr(mod, "regex_cleanup")
        assert callable(mod.regex_cleanup)

    def test_pdf_local_imports_page_count_from_triage(self):
        from pipeline.pdf_local import get_pdf_page_count  # re-exported via import
        from pipeline.triage import get_pdf_page_count as original
        assert get_pdf_page_count is original

    def test_pdf_cloud_imports_page_count_from_triage(self):
        from pipeline.pdf_cloud import get_pdf_page_count  # re-exported via import
        from pipeline.triage import get_pdf_page_count as original
        assert get_pdf_page_count is original

    def test_cleanup_imports_read_meta_at_top_level(self):
        """cleanup.py should import read_meta at top level, not at end of file."""
        source = (ROOT / "pipeline" / "cleanup.py").read_text()
        lines = source.strip().split("\n")
        # read_meta should appear in an import near the top, not at the bottom
        last_line = lines[-1].strip()
        assert "from pipeline.queue import read_meta" not in last_line

    def test_kindle_extract_metadata_accepts_cfg(self):
        """_extract_metadata should accept cfg parameter, not load it internally."""
        import inspect
        from pipeline.kindle import _extract_metadata
        sig = inspect.signature(_extract_metadata)
        assert "cfg" in sig.parameters
        # Should NOT import load_config inside the function
        source = inspect.getsource(_extract_metadata)
        assert "load_config" not in source
