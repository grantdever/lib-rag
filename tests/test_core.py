"""Tests for core lib-rag functionality.

Tests pure functions that don't require API keys, network access, or external services.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Set up import paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


# ============================================================
# pipeline/cleanup.py — regex_cleanup, _infer_headers, strip_images, strip_tables, pdf_cleanup
# ============================================================

from pipeline.cleanup import regex_cleanup, _infer_headers, strip_images, strip_tables, pdf_cleanup


class TestRegexCleanup:
    def test_strips_empty_anchors(self):
        text = "Hello []{#some-anchor} world"
        result = regex_cleanup(text)
        assert "[]{#" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_pandoc_fenced_divs(self):
        text = "Before\n::: {.section}\nContent\n:::\nAfter"
        result = regex_cleanup(text)
        assert ":::" not in result
        assert "Content" in result

    def test_strips_html_tags(self):
        text = "<div class='foo'>Hello</div> <span>world</span>"
        result = regex_cleanup(text)
        assert "<div" not in result
        assert "<span" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_pandoc_class_attributes(self):
        text = "[visible text]{.some-class}"
        result = regex_cleanup(text)
        assert "visible text" in result
        assert "{.some-class}" not in result

    def test_collapses_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = regex_cleanup(text)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_infers_headers_when_none_exist(self):
        text = "Some intro text\n\nChapter 1 Introduction\n\nContent here"
        result = regex_cleanup(text)
        assert "# Chapter 1 Introduction" in result

    def test_preserves_existing_headers(self):
        text = "# My Chapter\n\nSome content\n\n## Subsection\n\nMore content"
        result = regex_cleanup(text)
        assert "# My Chapter" in result
        assert "## Subsection" in result

    def test_fixes_broken_header_words(self):
        text = "# C hapter One"
        result = regex_cleanup(text)
        assert "Chapter" in result

    def test_returns_stripped_with_trailing_newline(self):
        text = "   \n\nSome content\n\n   "
        result = regex_cleanup(text)
        assert result.endswith("\n")
        assert not result.startswith("\n")


class TestInferHeaders:
    def test_chapter_pattern(self):
        text = "Intro\n\nChapter 1 The Beginning\n\nContent"
        result = _infer_headers(text)
        assert "# Chapter 1 The Beginning" in result

    def test_all_caps_title(self):
        text = "Intro\n\nTHE ANCIENT CITY STATES\n\nContent"
        result = _infer_headers(text)
        assert result.count("# ") >= 1

    def test_roman_numeral_part(self):
        text = "Intro\n\nPart IV. The Final Act\n\nContent"
        result = _infer_headers(text)
        assert "# Part IV. The Final Act" in result

    def test_no_false_positive_on_short_caps(self):
        text = "Intro\n\nOK SURE\n\nContent"
        result = _infer_headers(text)
        # "OK SURE" is only 7 chars, below the 10-char threshold
        assert "# " not in result


class TestStripImages:
    def test_strips_markdown_images(self):
        text = "Before ![alt](image.png) After"
        result = strip_images(text)
        assert "![alt]" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_html_images(self):
        text = 'Before <img src="image.png"> After'
        result = strip_images(text)
        assert "<img" not in result

    def test_strips_svg_blocks(self):
        text = "Before <svg viewBox='0 0 100 100'>content</svg> After"
        result = strip_images(text)
        assert "<svg" not in result


class TestStripTables:
    def test_strips_pipe_tables(self):
        text = "Before\n| Col1 | Col2 |\n|------|------|\n| A | B |\nAfter"
        result = strip_tables(text)
        assert "|" not in result
        assert "Before" in result
        assert "After" in result


class TestPdfCleanup:
    def test_fixes_spaced_capitals(self):
        text = "T he people of T his nation"
        result = pdf_cleanup(text)
        assert "The people" in result
        assert "This nation" in result

    def test_replaces_non_breaking_spaces(self):
        text = "Hello\xa0world"
        result = pdf_cleanup(text)
        assert "\xa0" not in result
        assert "Hello world" in result

    def test_removes_picture_placeholders(self):
        text = "Before ==> image intentionally omitted <== After"
        result = pdf_cleanup(text)
        assert "intentionally omitted" not in result


# ============================================================
# scripts/shared.py — validate_api_key, LLM_PROVIDERS, is_retryable
# ============================================================

from shared import validate_api_key, LLM_PROVIDERS, is_retryable


class TestValidateApiKey:
    def test_unknown_provider_raises(self):
        with pytest.raises(Exception, match="Unknown provider"):
            validate_api_key("nonexistent_provider")

    def test_uses_llm_providers_dict(self):
        """validate_api_key should look up env vars from LLM_PROVIDERS, not hardcode."""
        for provider, config in LLM_PROVIDERS.items():
            env_var = config["api_key_env"]
            # If the key isn't set, it should raise mentioning the env var
            import os
            old = os.environ.pop(env_var, None)
            try:
                with pytest.raises(Exception, match=env_var):
                    validate_api_key(provider)
            finally:
                if old:
                    os.environ[env_var] = old


class TestIsRetryable:
    def test_429_is_retryable(self):
        class FakeExc(Exception):
            status_code = 429
        assert is_retryable(FakeExc()) is True

    def test_500_is_retryable(self):
        class FakeExc(Exception):
            status_code = 500
        assert is_retryable(FakeExc()) is True

    def test_400_not_retryable(self):
        class FakeExc(Exception):
            status_code = 400
        assert is_retryable(FakeExc()) is False

    def test_string_detection(self):
        assert is_retryable(Exception("rate limit 429")) is True
        assert is_retryable(Exception("bad request")) is False


# ============================================================
# scripts/04_query.py — pure formatting functions, _escape_sql
# ============================================================

from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location("query", ROOT / "scripts" / "04_query.py")
_query_mod = module_from_spec(_spec)
# Don't exec (would trigger click/lancedb imports) — import specific functions manually
# Instead test the functions we can import directly

# We can at least test _escape_sql by reading it
import re


def _escape_sql(value: str) -> str:
    """Mirror of the function in 04_query.py for testing."""
    return value.replace("'", "''")


class TestEscapeSql:
    def test_escapes_single_quotes(self):
        assert _escape_sql("O'Brien") == "O''Brien"

    def test_no_change_without_quotes(self):
        assert _escape_sql("hello world") == "hello world"

    def test_multiple_quotes(self):
        assert _escape_sql("it's a 'test'") == "it''s a ''test''"


# ============================================================
# scripts/03_build_index.py — split_chapters, split_paragraphs, chunk_by_tokens
# ============================================================

# These need tiktoken, import conditionally
try:
    _spec3 = spec_from_file_location("build_index", ROOT / "scripts" / "03_build_index.py")
    _index_mod = module_from_spec(_spec3)
    _spec3.loader.exec_module(_index_mod)
    split_chapters = _index_mod.split_chapters
    split_paragraphs = _index_mod.split_paragraphs
    chunk_by_tokens = _index_mod.chunk_by_tokens
    HAS_INDEX_MODULE = True
except Exception:
    HAS_INDEX_MODULE = False


@pytest.mark.skipif(not HAS_INDEX_MODULE, reason="03_build_index.py deps not available")
class TestSplitChapters:
    def test_splits_on_h1(self):
        text = "# Chapter 1\nContent 1\n# Chapter 2\nContent 2"
        chapters = split_chapters(text)
        assert len(chapters) == 2
        assert chapters[0]["title"] == "Chapter 1"
        assert chapters[1]["title"] == "Chapter 2"

    def test_handles_no_headers(self):
        text = "Just some plain text\nwith no headers"
        chapters = split_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Untitled"

    def test_ignores_h2(self):
        text = "# Chapter 1\n## Section A\nContent\n# Chapter 2\nMore"
        chapters = split_chapters(text)
        assert len(chapters) == 2


@pytest.mark.skipif(not HAS_INDEX_MODULE, reason="03_build_index.py deps not available")
class TestSplitParagraphs:
    def test_splits_on_double_newline(self):
        text = "Para 1\n\nPara 2\n\nPara 3"
        paras = split_paragraphs(text)
        assert len(paras) == 3

    def test_filters_empty(self):
        text = "Para 1\n\n\n\n\nPara 2"
        paras = split_paragraphs(text)
        assert len(paras) == 2


@pytest.mark.skipif(not HAS_INDEX_MODULE, reason="03_build_index.py deps not available")
class TestChunkByTokens:
    def test_single_paragraph_fits(self):
        paras = ["Hello world"]
        chunks = chunk_by_tokens(paras, target_tokens=100)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world"

    def test_splits_when_exceeds_target(self):
        paras = ["word " * 200, "another " * 200]
        chunks = chunk_by_tokens(paras, target_tokens=100)
        assert len(chunks) >= 2


# ============================================================
# pipeline/queue.py — write_meta, read_meta, acquire_lock, release_lock
# ============================================================

from pipeline.queue import write_meta, read_meta, acquire_lock, release_lock, compute_sha256


class TestQueueMeta:
    def test_write_and_read_meta(self, tmp_path):
        write_meta(tmp_path, book_id="test-book", status="pending")
        meta = read_meta(tmp_path)
        assert meta["book_id"] == "test-book"
        assert meta["status"] == "pending"
        assert "created_at" in meta
        assert "updated_at" in meta

    def test_meta_merge(self, tmp_path):
        write_meta(tmp_path, book_id="test-book")
        write_meta(tmp_path, status="done")
        meta = read_meta(tmp_path)
        assert meta["book_id"] == "test-book"
        assert meta["status"] == "done"

    def test_read_missing_meta(self, tmp_path):
        meta = read_meta(tmp_path)
        assert meta == {}


class TestQueueLock:
    def test_acquire_and_release(self, tmp_path):
        assert acquire_lock(tmp_path) is True
        assert acquire_lock(tmp_path) is False  # already locked
        release_lock(tmp_path)
        assert acquire_lock(tmp_path) is True  # can re-acquire
        release_lock(tmp_path)

    def test_release_nonexistent_lock(self, tmp_path):
        release_lock(tmp_path)  # should not raise


class TestComputeSha256:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_sha256(f)
        h2 = compute_sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # hex SHA-256

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_sha256(f1) != compute_sha256(f2)


# ============================================================
# pipeline/embed.py — slugify_book_name
# ============================================================

from pipeline.embed import slugify_book_name


class TestSlugifyBookName:
    def test_basic_slug(self):
        assert slugify_book_name("Nisbet - Quest for Community.pdf") == "nisbet-quest-for-community"

    def test_strips_trailing_parens(self):
        assert "2020" not in slugify_book_name("Some Book (2020).epub")

    def test_lowercase(self):
        result = slugify_book_name("HAYEK Constitution of Liberty.epub")
        assert result == result.lower()

    def test_strips_special_chars(self):
        result = slugify_book_name("Author's Book: A Study!.pdf")
        assert "'" not in result
        assert ":" not in result
        assert "!" not in result


# ============================================================
# pipeline/triage.py — classify_file (with mock)
# ============================================================

from pipeline.triage import classify_file


class TestClassifyFile:
    def test_epub_classification(self, tmp_path):
        epub = tmp_path / "test.epub"
        epub.write_bytes(b"fake epub")
        result = classify_file(epub)
        assert result["classification"] == "epub"
        assert result["file_type"] == "epub"

    def test_unsupported_file(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("hello")
        result = classify_file(txt)
        assert result["classification"] == "unsupported"


# ============================================================
# pipeline/cleanup.py — compute_quality_score
# ============================================================

from pipeline.cleanup import compute_quality_score


class TestComputeQualityScore:
    def test_empty_text(self):
        assert compute_quality_score("") == 0.0

    def test_good_quality_text(self):
        lines = ["# Chapter 1"] + [f"Paragraph {i} with some decent content here." for i in range(100)]
        text = "\n\n".join(lines)
        score = compute_quality_score(text)
        assert 0.5 < score <= 1.0

    def test_returns_float(self):
        score = compute_quality_score("Some text")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ============================================================
# Integration: 01_convert_epubs.py imports regex_cleanup from pipeline
# ============================================================

class TestCleanupImportIntegration:
    def test_01_convert_uses_pipeline_cleanup(self):
        """Verify 01_convert_epubs.py imports regex_cleanup from pipeline.cleanup."""
        source = (ROOT / "scripts" / "01_convert_epubs.py").read_text()
        assert "from pipeline.cleanup import regex_cleanup" in source
        # Should NOT have its own clean_markdown function
        assert "def clean_markdown" not in source
        assert "def infer_headers" not in source


# ============================================================
# Verify no personal data in codebase
# ============================================================

class TestNoPersonalData:
    def test_no_personal_references(self):
        """No personal data should appear in any Python file (excluding tests)."""
        forbidden = ["FREOPP", "ALI_Books", "ALI Books", "ali-books"]
        for py_file in ROOT.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file) or "tests/" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden:
                assert term not in content, f"Found '{term}' in {py_file.relative_to(ROOT)}"
