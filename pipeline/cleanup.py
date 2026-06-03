"""Markdown cleanup: shared path for all conversion outputs.

Flow:
1. Strip images and tables (configurable)
2. Regex cleanup (ported from scripts/01_convert_epubs.py clean_markdown)
3. Pandoc round-trip normalization
4. Optional DeepSeek fuzzy pass (if heuristic quality score < threshold)

Output: clean.md in the staging folder.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.queue import write_meta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image / table stripping
# ---------------------------------------------------------------------------

def strip_images(text: str) -> str:
    """Remove markdown images: ![alt](url) and HTML <img> tags."""
    # Markdown images
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", text)
    # HTML images
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    # SVG blocks
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    return text


def strip_tables(text: str) -> str:
    """Remove markdown tables (pipe-delimited rows).

    Removes contiguous blocks of lines that look like table rows.
    """
    lines = text.split("\n")
    result = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        # Table row: starts and ends with |, or is a separator like |---|---|
        is_table_line = (
            stripped.startswith("|") and stripped.endswith("|")
        ) or re.match(r"^\|[\s\-:|]+\|$", stripped)

        if is_table_line:
            in_table = True
            continue
        else:
            if in_table:
                in_table = False
            result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Regex cleanup — ported from scripts/01_convert_epubs.py clean_markdown()
# ---------------------------------------------------------------------------

def pdf_cleanup(text: str) -> str:
    """Fix PDF-specific extraction artifacts.

    Handles soft hyphens, spaced capitals, running headers, picture
    placeholders, and non-breaking spaces that PyMuPDF4LLM produces.
    """
    # 1. Rejoin soft-hyphenated words: "utiliza\xad tion" → "utilization"
    #    Also handles regular hyphens at line breaks: "com-\nmuity" → "community"
    text = re.sub(r"(\w)\xad\s+(\w)", r"\1\2", text)
    text = re.sub(r"(\w)­\s*\n\s*(\w)", r"\1\2", text)

    # 2. Fix spaced capitals from PDF kerning: "T he" → "The", "W eber" → "Weber"
    #    Only at word boundaries to avoid false positives
    text = re.sub(r"\bT he\b", "The", text)
    text = re.sub(r"\bW hat\b", "What", text)
    text = re.sub(r"\bW hen\b", "When", text)
    text = re.sub(r"\bW ith\b", "With", text)
    text = re.sub(r"\bW orld\b", "World", text)
    text = re.sub(r"\bW ar\b", "War", text)
    text = re.sub(r"\bT hat\b", "That", text)
    text = re.sub(r"\bT homas\b", "Thomas", text)
    text = re.sub(r"\bT here\b", "There", text)
    text = re.sub(r"\bT hose\b", "Those", text)
    text = re.sub(r"\bT hus\b", "Thus", text)
    text = re.sub(r"\bT his\b", "This", text)
    text = re.sub(r"\bW estern\b", "Western", text)
    text = re.sub(r"\bW eber\b", "Weber", text)
    text = re.sub(r"\bM arx\b", "Marx", text)
    text = re.sub(r"\bW hich\b", "Which", text)
    text = re.sub(r"\bW here\b", "Where", text)
    text = re.sub(r"\bW hile\b", "While", text)
    text = re.sub(r"\bW hy\b", "Why", text)
    text = re.sub(r"\bW hether\b", "Whether", text)
    text = re.sub(r"\bW illiam\b", "William", text)
    # General pattern: single capital + space + lowercase continuation (>1 char)
    # Catches remaining spaced-capital artifacts not listed above.
    # Excludes "A" and "I" which are real English words.
    text = re.sub(r"\b([B-HJ-Z]) ([a-z]{2,})", r"\1\2", text)

    # 3. Remove picture placeholders
    text = re.sub(r"==>.*?intentionally omitted.*?<==", "", text)
    text = re.sub(r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*", "", text, flags=re.DOTALL)

    # 4. Remove running headers/footers
    # Plain bold: **THE MILITARY COMMUNITY •** or **WAR AND THE GREEK POLIS • 33**
    text = re.sub(r"^\*\*[A-Z][A-Z &,'-]{8,}[•\*].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*?\*?\d{1,3}\s*[•·]\s*[A-Z][A-Z &,'-]{8,}\*?\*?\s*$", "", text, flags=re.MULTILINE)
    # H2/H3 running headers containing bullet (•) or square (■) — always page headers
    text = re.sub(r"^#{1,3}\s+.*[•■].*$", "", text, flags=re.MULTILINE)

    # 5. Fix letter-spaced words (e.g. "T h e A n c i e n t C i t y" → "The Ancient City")
    def fix_letterspaced(m):
        spaced = m.group(0)
        return re.sub(r"(?<=\w) (?=\w)", "", spaced)
    text = re.sub(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b", fix_letterspaced, text)

    # 6. Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")

    # 7. Remove stray decorative characters
    text = re.sub(r"^[■•·]\s*$", "", text, flags=re.MULTILINE)

    return text


def regex_cleanup(text: str) -> str:
    """Apply regex-based cleanup rules.

    These handle Pandoc artifacts, HTML remnants, header fixing, and
    broken word repair. Ported from the existing EPUB conversion pipeline.
    """
    # PDF-specific fixes (soft hyphens, spaced caps, running headers)
    text = pdf_cleanup(text)

    # Line-level cleaning
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = re.sub(r"\[\]\{#[^}]*\}", "", line)            # empty anchors
        line = re.sub(r"\[\]\{(\.[a-zA-Z0-9_-]+\s*)+\}", "", line)  # empty class spans
        line = re.sub(r"</?div[^>]*>", "", line)
        line = re.sub(r"</?p[^>]*>", "", line)
        line = re.sub(r"</?span[^>]*>", "", line)

        if re.match(r"^:::", line):  # Pandoc fenced divs
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Clean headers: unwrap [text]{.class} spans, strip trailing {#anchor .class}
    def clean_header(m):
        prefix = m.group(1)
        rest = m.group(2)
        rest = re.sub(r"\[([^\]]*)\]\{[^}]*\}", r"\1", rest)
        rest = re.sub(r"\s*\{[^}]*\}\s*$", "", rest)
        rest = re.sub(r"\s+", " ", rest).strip()
        if not rest:
            return ""
        return f"{prefix} {rest}"

    text = re.sub(r"^(#{1,6})\s+(.+)$", clean_header, text, flags=re.MULTILINE)
    text = re.sub(r"^\s*$\n", "\n", text, flags=re.MULTILINE)

    # Strip inline class attributes: [visible text]{.class} → visible text
    text = re.sub(r"\[([^\]]+)\]\{(\.[a-zA-Z0-9_-]+\s*)+\}", r"\1", text)

    # Strip all remaining Pandoc attribute blocks
    text = re.sub(r"\{[#.][^}]*\}", "", text)

    # Fix broken words in headers (Pandoc span artifacts)
    def fix_broken_header(m):
        prefix = m.group(1)
        rest = m.group(2)
        rest = re.sub(r"(?<![A-Za-z])([A-Z]) ([a-z])", r"\1\2", rest)
        rest = re.sub(r"([A-Z]{2,}) ([A-Z])(?=[\s.,;:]|$)", r"\1\2", rest)
        rest = re.sub(r"(?<![A-Za-z])([A-Z]) ([A-Z]{2,})", r"\1\2", rest)
        rest = re.sub(r" +([.,;:])", r"\1", rest)
        rest = re.sub(r"  +", " ", rest).strip()
        return f"{prefix} {rest}" if rest else ""
    text = re.sub(r"^(#{1,6})\s+(.+)$", fix_broken_header, text, flags=re.MULTILINE)

    # Collapse excessive blank lines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Infer headers for books where none exist
    has_h1 = any(line.startswith("# ") for line in text.splitlines())
    if not has_h1:
        text = _infer_headers(text)

    return text.strip() + "\n"


def _infer_headers(text: str) -> str:
    """For books with no H1 headers, try to detect chapter boundaries."""
    lines = text.split("\n")
    result = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            prev_blank = True
            result.append(line)
            continue

        ch_match = re.match(
            r"^(Chapter\s+\d+[\.:]?\s*.*|CHAPTER\s+[IVXLCDM\d]+[\.:]?\s*.*)$",
            stripped,
            re.IGNORECASE,
        )
        if ch_match and prev_blank:
            result.append(f"# {stripped}")
            prev_blank = False
            continue

        if (
            prev_blank
            and len(stripped) > 10
            and stripped == stripped.upper()
            and re.match(r'^[A-Z\s\-:,\'"]+$', stripped)
        ):
            if len(stripped.split()) >= 2:
                result.append(f"# {stripped.title()}")
                prev_blank = False
                continue

        part_match = re.match(
            r"^(Part\s+[IVXLCDM]+[\.:]\s*.+|[IVXLCDM]+\.\s+[A-Z].+)$", stripped
        )
        if part_match and prev_blank:
            result.append(f"# {stripped}")
            prev_blank = False
            continue

        prev_blank = False
        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Pandoc round-trip normalization
# ---------------------------------------------------------------------------

def pandoc_normalize(text: str) -> str:
    """Run markdown through pandoc to normalize formatting."""
    try:
        result = subprocess.run(
            ["pandoc", "-f", "markdown", "-t", "markdown", "--wrap=none"],
            input=text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception as e:
        logger.warning("Pandoc normalize failed: %s", e)

    return text  # fallback to input on failure


# ---------------------------------------------------------------------------
# Heuristic quality score
# ---------------------------------------------------------------------------

def compute_quality_score(text: str) -> float:
    """Compute a heuristic quality score (0-1) for cleaned markdown.

    Blends several signals:
    - Has H1 headers
    - Reasonable paragraph structure (>10 paragraphs)
    - Low ratio of short lines (broken formatting)
    - Low ratio of non-ASCII noise
    """
    lines = text.splitlines()
    if not lines:
        return 0.0

    scores = []

    # Has H1 headers
    h1_count = sum(1 for l in lines if l.startswith("# "))
    scores.append(min(h1_count / 5, 1.0))

    # Paragraph count (double newline separated)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    scores.append(min(len(paragraphs) / 50, 1.0))

    # Short line ratio (lines < 20 chars that aren't headers or blank)
    content_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    if content_lines:
        short = sum(1 for l in content_lines if len(l.strip()) < 20)
        scores.append(1.0 - min(short / len(content_lines), 1.0))
    else:
        scores.append(0.0)

    # Non-ASCII noise ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text) if text else 1
    scores.append(ascii_chars / total_chars)

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Optional DeepSeek fuzzy cleanup
# ---------------------------------------------------------------------------

def deepseek_fuzzy_cleanup(text: str, api_key: str) -> str:
    """Use DeepSeek V3 via OpenRouter to fix remaining quality issues.

    Only called when heuristic quality score is below threshold.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/ali-books",
            "X-Title": "lib-rag",
        },
    )

    # Process in chunks to stay within context limits
    # Take first ~20K chars for cleanup
    sample = text[:20_000]

    response = client.chat.completions.create(
        model="deepseek/deepseek-v3",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a text cleanup assistant. Fix OCR artifacts, broken words, "
                    "bad line breaks, and formatting issues in this markdown text. "
                    "Preserve all content and structure. Return ONLY the cleaned markdown, "
                    "no explanation."
                ),
            },
            {"role": "user", "content": sample},
        ],
        max_tokens=25_000,
        temperature=0.1,
    )

    cleaned_sample = response.choices[0].message.content or sample

    # If the sample was truncated, append the rest unchanged
    if len(text) > 20_000:
        return cleaned_sample + text[20_000:]
    return cleaned_sample


# ---------------------------------------------------------------------------
# Main cleanup pipeline
# ---------------------------------------------------------------------------

def cleanup_markdown(
    raw_text: str,
    cfg: PipelineConfig,
    source_type: str = "pdf",
) -> tuple[str, dict]:
    """Run the full cleanup pipeline on raw markdown.

    Returns (cleaned_text, stats_dict).
    """
    stats: dict = {}

    text = raw_text

    # Step 1: Strip images/tables
    if cfg.cleanup.strip_images:
        text = strip_images(text)
    if cfg.cleanup.strip_tables:
        text = strip_tables(text)

    # Step 2: Regex cleanup
    text = regex_cleanup(text)

    # Step 3: Pandoc round-trip
    text = pandoc_normalize(text)

    # Step 4: Quality check + optional fuzzy pass
    quality = compute_quality_score(text)
    stats["quality_score"] = round(quality, 3)
    stats["cleanup_method"] = "regex+pandoc"

    if quality < cfg.cleanup.fuzzy_threshold and cfg.api_keys.openrouter:
        logger.info("Quality score %.2f < %.2f — running DeepSeek cleanup", quality, cfg.cleanup.fuzzy_threshold)
        try:
            text = deepseek_fuzzy_cleanup(text, cfg.api_keys.openrouter)
            stats["cleanup_method"] = "regex+pandoc+deepseek"
            stats["quality_score_post_deepseek"] = round(compute_quality_score(text), 3)
        except Exception as e:
            logger.warning("DeepSeek cleanup failed: %s", e)

    return text, stats


def process_cleanup(staging_folder: Path, cfg: PipelineConfig) -> bool:
    """Run cleanup on raw.md in a staging folder, produce clean.md.

    Returns True on success.
    """
    raw_path = staging_folder / "raw.md"
    if not raw_path.exists():
        logger.error("No raw.md in %s", staging_folder)
        return False

    raw_text = raw_path.read_text(encoding="utf-8")
    meta = dict(read_meta(staging_folder))  # avoid import cycle by using dict()
    source_type = meta.get("file_type", "pdf")

    cleaned, stats = cleanup_markdown(raw_text, cfg, source_type)

    clean_path = staging_folder / "clean.md"
    clean_path.write_text(cleaned, encoding="utf-8")

    write_meta(staging_folder, clean_chars=len(cleaned), **stats)

    logger.info(
        "%s: cleanup done (quality=%.2f, method=%s, %d→%d chars)",
        staging_folder.name,
        stats.get("quality_score", 0),
        stats.get("cleanup_method", "?"),
        len(raw_text),
        len(cleaned),
    )
    return True


# Need to import read_meta at function level to avoid issues
from pipeline.queue import read_meta  # noqa: E402
