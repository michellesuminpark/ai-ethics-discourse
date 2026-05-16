"""
parse.py — PDF text extraction for two downstream tasks.

Reads data/raw/arxiv_pilot.csv and for each paper extracts:

  1. Affiliation text — first 2 pages + any acknowledgments section.
     → data/processed/affiliation_text.csv  (paper_id, year, affiliation_text)

  2. Ethics-relevant section text — introduction, discussion, conclusion,
     ethics, broader impact, limitations, societal impact.
     Falls back to first 3 + last 3 body paragraphs when no target headers
     are found.
     → data/processed/ethics_sections.csv
       (paper_id, year, ethics_text, extraction_method)
"""

import fitz
import pandas as pd
import re
import os

CSV_IN     = "data/raw/arxiv_pilot.csv"
AFFIL_OUT  = "data/processed/affiliation_text.csv"
ETHICS_OUT = "data/processed/ethics_sections.csv"
PDF_DIR    = "data/pdfs"
MIN_BLOCK  = 6   # low enough to capture short section headers (e.g. "Ethics")

os.makedirs("data/processed", exist_ok=True)

# ── Header regexes ─────────────────────────────────────────────────────────────

_NUM_PREFIX = r"(?:\d+(?:\.\d+)*\.?\s+)?"

# Looser: search for target word anywhere in a short block (handles trailing
# periods, "Conclusions and Future Work", non-breaking spaces, etc.)
_ETHICS_WORDS_RE = re.compile(
    r"\b(introduction|discussion|conclusions?|ethics|ethical"
    r"|broader\s+impacts?|limitations?|societal\s+impacts?)\b",
    re.IGNORECASE,
)

_ACKNOWLEDGMENT_RE = re.compile(
    r"\backnowledg(?:e)?ments?\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "",
                  text.strip().replace("\n", " "))


def _looks_like_header(text: str) -> bool:
    s = text.strip()
    if len(s) > 130 or len(s.split()) > 20:
        return False
    # Headers don't end mid-sentence
    if s.endswith((",", ";", "...", "—")):
        return False
    # Contains an internal sentence break → likely a paragraph
    # Require lowercase before the period so "1. Introduction" isn't rejected
    if re.search(r"[a-z]\. [A-Z]", s):
        return False
    return True


def _get_blocks(doc) -> list[dict]:
    """All text blocks in reading order, header/footer margins stripped."""
    blocks = []
    for page_no, page in enumerate(doc):
        h      = page.rect.height
        margin = h * 0.06
        for b in page.get_text("blocks"):
            if not isinstance(b[4], str):
                continue
            y0, y1 = b[1], b[3]
            if y0 < margin or y1 > h - margin:
                continue
            text = _clean(b[4])
            if len(text) < MIN_BLOCK:
                continue
            blocks.append({"page": page_no, "text": text})
    return blocks


# ── Task 1: affiliation text ──────────────────────────────────────────────────

def extract_affiliation_text(pdf_path: str) -> str:
    """First 2 pages + full acknowledgments section text."""
    if not os.path.exists(pdf_path):
        return ""
    try:
        doc    = fitz.open(pdf_path)
        blocks = _get_blocks(doc)
        doc.close()
    except Exception as e:
        print(f"  [fitz error] {pdf_path}: {e}", flush=True)
        return ""

    parts = [b["text"] for b in blocks if b["page"] < 2]

    # Acknowledgments section from anywhere in the doc (often has "work done
    # while at X" and "supported by Y" which reveal industry affiliations)
    in_ack = False
    ack    = []
    for b in blocks:
        if b["page"] < 2:
            continue
        if _looks_like_header(b["text"]) and _ACKNOWLEDGMENT_RE.search(b["text"]):
            in_ack = True
            continue
        if in_ack:
            if _looks_like_header(b["text"]):
                break
            ack.append(b["text"])

    if ack:
        parts.append("ACKNOWLEDGMENTS: " + " ".join(ack))

    return " ".join(parts)


# ── Task 2: ethics section text ───────────────────────────────────────────────

def extract_ethics_sections(pdf_path: str) -> tuple[str, str]:
    """
    Return (text, method).
    method: 'sections' | 'fallback_no_headers' | 'fallback_no_target_sections'
            | 'fallback_empty_sections' | 'missing' | 'error' | 'empty'
    """
    if not os.path.exists(pdf_path):
        return "", "missing"
    try:
        doc    = fitz.open(pdf_path)
        blocks = _get_blocks(doc)
        doc.close()
    except Exception as e:
        print(f"  [fitz error] {pdf_path}: {e}", flush=True)
        return "", "error"

    if not blocks:
        return "", "empty"

    # Annotate each block as 'target', 'header', or 'body'
    annotated: list[tuple[dict, str]] = []
    for b in blocks:
        t = b["text"].strip()
        if _looks_like_header(t) and _ETHICS_WORDS_RE.search(t):
            annotated.append((b, "target"))
        elif _looks_like_header(t):
            annotated.append((b, "header"))
        else:
            annotated.append((b, "body"))

    has_any_header = any(k in ("target", "header") for _, k in annotated)
    has_target     = any(k == "target" for _, k in annotated)

    if not has_any_header:
        return _fallback(annotated), "fallback_no_headers"
    if not has_target:
        return _fallback(annotated), "fallback_no_target_sections"

    # Collect body text under each target header, stopping at any header
    target_chunks: list[str] = []
    in_target      = False
    current_header = ""
    current_body: list[str] = []

    for b, kind in annotated:
        if kind == "target":
            if in_target and current_body:
                target_chunks.append(f"[{current_header}]\n" + " ".join(current_body))
            in_target      = True
            current_header = b["text"].strip()
            current_body   = []
        elif kind == "header" and in_target:
            if current_body:
                target_chunks.append(f"[{current_header}]\n" + " ".join(current_body))
            in_target      = False
            current_header = ""
            current_body   = []
        elif kind == "body" and in_target:
            current_body.append(b["text"])

    if in_target and current_body:
        target_chunks.append(f"[{current_header}]\n" + " ".join(current_body))

    if not target_chunks:
        return _fallback(annotated), "fallback_empty_sections"

    return "\n\n".join(target_chunks), "sections"


def _fallback(annotated: list) -> str:
    body = [b["text"] for b, k in annotated if k == "body"]
    if len(body) <= 6:
        return " ".join(body)
    return " ".join(body[:3] + body[-3:])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(CSV_IN, dtype={"paper_id": str})
    print(f"Loaded {len(df)} papers from {CSV_IN}")

    affil_rows:  list[dict] = []
    ethics_rows: list[dict] = []
    errors:      list[str]  = []
    methods:     dict[str, int] = {}

    for i, paper in df.iterrows():
        pid      = str(paper["paper_id"])
        year     = int(paper["year"])
        pdf_path = f"{PDF_DIR}/{pid}.pdf"

        affil               = extract_affiliation_text(pdf_path)
        ethics_text, method = extract_ethics_sections(pdf_path)

        methods[method] = methods.get(method, 0) + 1
        if method in ("missing", "error", "empty"):
            errors.append(pid)

        affil_rows.append({"paper_id": pid, "year": year, "affiliation_text": affil})
        ethics_rows.append({
            "paper_id":          pid,
            "year":              year,
            "ethics_text":       ethics_text,
            "extraction_method": method,
        })

        print(
            f"  [{i+1}/{len(df)}] {pid}: "
            f"affil={len(affil):,}c  ethics={len(ethics_text):,}c  [{method}]",
            flush=True,
        )

    pd.DataFrame(affil_rows).to_csv(AFFIL_OUT,  index=False)
    pd.DataFrame(ethics_rows).to_csv(ETHICS_OUT, index=False)

    print(f"\nSaved {len(affil_rows):,} rows  -> {AFFIL_OUT}")
    print(f"Saved {len(ethics_rows):,} rows  -> {ETHICS_OUT}")

    print("\nExtraction method breakdown:")
    for m, n in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {m:<45} {n:>4}")

    if errors:
        print(f"\nFailed/missing papers ({len(errors)}): {', '.join(errors)}")


if __name__ == "__main__":
    main()
