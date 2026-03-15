"""
PDF parser for academic papers.

Extracts structured content from PDFs using PyMuPDF for layout analysis
and pdfplumber for table / caption extraction. Returns a list of Section
objects that preserve the logical structure of the paper (Abstract,
Introduction, Methods, etc.) rather than a flat stream of pages.

Usage:
    parser = PDFParser()
    document = parser.parse("papers/attention_is_all_you_need.pdf")
    for section in document.sections:
        print(section.title, len(section.text))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A logical section of a paper (e.g. Abstract, Introduction, Methods)."""
    title: str
    text: str
    page_start: int       # 0-indexed
    page_end: int         # 0-indexed, inclusive
    section_index: int    # Position in the paper (0 = first section)


@dataclass
class ParsedDocument:
    """Fully parsed representation of a single academic PDF."""
    paper_id: str                              # Caller-supplied identifier
    file_path: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    total_pages: int = 0

    @property
    def body_sections(self) -> list[Section]:
        """Sections excluding abstract and references."""
        skip = {"abstract", "references", "bibliography", "acknowledgements",
                "acknowledgments", "appendix"}
        return [s for s in self.sections if s.title.lower() not in skip]


# ---------------------------------------------------------------------------
# Heading detection helpers
# ---------------------------------------------------------------------------

# Common academic paper section headings — used to detect section boundaries.
_KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work",
    "methodology", "methods", "method", "approach", "model",
    "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "discussion", "conclusion", "conclusions",
    "future work", "limitations", "references", "bibliography",
    "acknowledgements", "acknowledgments", "appendix",
}

# Numbered headings: "1 Introduction", "2.1 Dataset", "III. Methods"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:[IVX]+\.?\s+|[\d]+(?:\.\d+)*\.?\s+)([A-Z][^\n]{2,60})$"
)

# Heuristic: all-caps short line is probably a heading
_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\d:]{3,50}$")


def _is_heading(text: str, font_size: float, page_median_size: float) -> bool:
    """Return True if this text block looks like a section heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False

    # Font size meaningfully larger than body text
    if font_size >= page_median_size * 1.15:
        return True

    lower = stripped.lower().rstrip(".")
    if lower in _KNOWN_HEADINGS:
        return True

    if _NUMBERED_HEADING_RE.match(stripped):
        return True

    if _CAPS_HEADING_RE.match(stripped) and len(stripped.split()) <= 6:
        return True

    return False


def _clean_heading(text: str) -> str:
    """Normalise a raw heading string."""
    # Strip leading numbering: "2.1 Related Work" → "Related Work"
    text = re.sub(r"^(?:[IVX]+\.?\s+|[\d]+(?:\.\d+)*\.?\s+)", "", text.strip())
    return text.strip().rstrip(".")


def _extract_year(text: str) -> Optional[int]:
    """Best-effort year extraction from first two pages of a paper."""
    matches = re.findall(r"\b(19|20)\d{2}\b", text)
    if matches:
        years = [int(y) for y in matches]
        # Most likely the submission/publication year
        return max(y for y in years if y <= 2025)
    return None


def _extract_authors(first_page_text: str) -> list[str]:
    """
    Very rough author extraction from the first page.
    Returns a list of candidate author strings. Not perfect — good enough
    for metadata storage and display.
    """
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]
    # Authors usually appear in the first 15 lines, between the title and abstract
    candidate_lines = lines[1:15]
    authors: list[str] = []
    for line in candidate_lines:
        # Skip lines that look like affiliations, emails, or headings
        if any(tok in line.lower() for tok in ["university", "institute", "@", "http",
                                                "abstract", "department", "lab", "inc."]):
            continue
        # Lines with comma-separated capitalised names
        if re.search(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", line):
            # Split on commas or " and "
            parts = re.split(r",\s*|\s+and\s+", line)
            for part in parts:
                part = part.strip()
                if 3 < len(part) < 50 and re.search(r"[A-Z]", part):
                    authors.append(part)
            if authors:
                break
    return authors[:12]  # Cap at 12 — enough for any sane paper


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Parse an academic PDF into a structured ParsedDocument.

    Strategy:
    1. Use PyMuPDF to iterate blocks with font-size metadata for heading detection.
    2. Use pdfplumber for figure caption extraction (better table/caption parsing).
    3. Segment text into sections based on detected headings.
    4. Extract abstract, authors, year as first-class fields.
    """

    def parse(self, path: str | Path, paper_id: Optional[str] = None) -> ParsedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        paper_id = paper_id or path.stem
        log.info("Parsing %s (id=%s)", path.name, paper_id)

        doc = ParsedDocument(paper_id=paper_id, file_path=str(path))

        with fitz.open(str(path)) as pdf:
            doc.total_pages = len(pdf)
            raw_blocks = self._extract_blocks(pdf)
            doc.title = self._extract_title(pdf)

        doc.figure_captions = self._extract_captions(path)
        doc.sections = self._segment_sections(raw_blocks)

        # Populate metadata from parsed content
        first_page_text = " ".join(
            b["text"] for b in raw_blocks if b["page"] == 0
        )
        doc.authors = _extract_authors(first_page_text)
        doc.year = _extract_year(first_page_text)

        abstract_section = next(
            (s for s in doc.sections if s.title.lower() == "abstract"), None
        )
        if abstract_section:
            doc.abstract = abstract_section.text

        ref_section = next(
            (s for s in doc.sections
             if s.title.lower() in {"references", "bibliography"}), None
        )
        if ref_section:
            doc.references = self._parse_references(ref_section.text)

        log.info(
            "Parsed %s: %d sections, %d pages, %d figure captions",
            paper_id, len(doc.sections), doc.total_pages, len(doc.figure_captions),
        )
        return doc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_blocks(self, pdf: fitz.Document) -> list[dict]:
        """
        Extract text blocks with font-size metadata from all pages.
        Returns a list of dicts: {text, size, page, bbox}.
        """
        blocks: list[dict] = []
        for page_num, page in enumerate(pdf):
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            sizes_on_page: list[float] = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes_on_page.append(span.get("size", 12.0))

            median_size = (
                sorted(sizes_on_page)[len(sizes_on_page) // 2]
                if sizes_on_page else 12.0
            )

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                block_text_parts: list[str] = []
                block_max_size = 0.0
                for line in block.get("lines", []):
                    line_parts: list[str] = []
                    for span in line.get("spans", []):
                        line_parts.append(span.get("text", ""))
                        block_max_size = max(block_max_size, span.get("size", 0.0))
                    block_text_parts.append("".join(line_parts))

                block_text = "\n".join(block_text_parts).strip()
                if not block_text:
                    continue

                blocks.append({
                    "text": block_text,
                    "size": block_max_size,
                    "page": page_num,
                    "bbox": block.get("bbox", ()),
                    "median_page_size": median_size,
                })
        return blocks

    def _extract_title(self, pdf: fitz.Document) -> str:
        """
        Extract title from PDF metadata, falling back to the largest text
        block on the first page.
        """
        meta_title = pdf.metadata.get("title", "").strip()
        if meta_title and len(meta_title) > 4:
            return meta_title

        # Largest font on page 0 is almost always the title
        page = pdf[0]
        page_dict = page.get_text("dict")
        best_text, best_size = "", 0.0
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0.0)
                    text = span.get("text", "").strip()
                    if size > best_size and len(text) > 10:
                        best_size = size
                        best_text = text
        return best_text

    def _extract_captions(self, path: Path) -> list[str]:
        """Use pdfplumber to extract figure and table captions."""
        captions: list[str] = []
        caption_re = re.compile(
            r"^(?:Figure|Fig\.|Table)\s+\d+[\.:]\s*.+", re.IGNORECASE
        )
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for line in text.split("\n"):
                        line = line.strip()
                        if caption_re.match(line):
                            captions.append(line)
        except Exception as exc:  # pdfplumber can fail on malformed PDFs
            log.warning("pdfplumber caption extraction failed: %s", exc)
        return captions

    def _segment_sections(self, blocks: list[dict]) -> list[Section]:
        """
        Walk extracted blocks and split them into Section objects at
        each detected heading boundary.
        """
        sections: list[Section] = []
        current_title = "Preamble"
        current_page_start = 0
        current_page = 0
        current_text_parts: list[str] = []

        def _flush(end_page: int) -> None:
            text = "\n".join(current_text_parts).strip()
            if text:
                sections.append(Section(
                    title=current_title,
                    text=text,
                    page_start=current_page_start,
                    page_end=end_page,
                    section_index=len(sections),
                ))

        for block in blocks:
            text = block["text"]
            size = block["size"]
            page = block["page"]
            median = block["median_page_size"]

            if _is_heading(text, size, median):
                _flush(current_page)
                current_title = _clean_heading(text)
                current_page_start = page
                current_text_parts = []
            else:
                current_text_parts.append(text)

            current_page = page

        _flush(current_page)
        return sections

    def _parse_references(self, ref_text: str) -> list[str]:
        """
        Split reference section text into individual reference strings.
        Handles numbered ([1], 1.) and author-year styles.
        """
        # Split on common reference delimiters
        entries = re.split(r"\n(?=\[\d+\]|\d+\.|\[[\w]+)", ref_text)
        return [e.strip() for e in entries if len(e.strip()) > 20]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="Parse a PDF and print section summaries.")
    ap.add_argument("--input", required=True, help="Path to PDF file")
    ap.add_argument("--paper-id", default=None, help="Optional paper identifier")
    ap.add_argument("--json", action="store_true", help="Output full JSON")
    args = ap.parse_args()

    from logger import configure_logging
    configure_logging()

    parser = PDFParser()
    doc = parser.parse(args.input, paper_id=args.paper_id)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(doc), indent=2, default=str))
    else:
        print(f"Title   : {doc.title}")
        print(f"Authors : {', '.join(doc.authors)}")
        print(f"Year    : {doc.year}")
        print(f"Pages   : {doc.total_pages}")
        print(f"Sections: {len(doc.sections)}")
        for s in doc.sections:
            print(f"  [{s.section_index:02d}] {s.title:<30} "
                  f"p{s.page_start+1}-{s.page_end+1}  "
                  f"({len(s.text)} chars)")
