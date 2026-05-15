"""
parse.py — Paragraph extraction from PDFs.

Reads data/raw/arxiv_pilot.csv, extracts body paragraphs from each PDF,
and writes data/processed/paragraphs.csv with one row per paragraph.

Columns in output:
  paper_id, year, sector, para_idx, text
"""

import fitz
import pandas as pd
import os
import re
import sys

CSV_IN  = "data/raw/arxiv_pilot.csv"
CSV_OUT = "data/processed/paragraphs.csv"
PDF_DIR = "data/pdfs"
MIN_CHARS = 50

os.makedirs("data/processed", exist_ok=True)

PAGE_NO_RE = re.compile(r"^\s*\d{1,4}\s*$")


def extract_paragraphs(pdf_path: str, min_chars: int = MIN_CHARS) -> list[str]:
    if not isinstance(pdf_path, str) or not os.path.exists(pdf_path):
        return []
    try:
        doc   = fitz.open(pdf_path)
        paras = []
        for page in doc:
            h      = page.rect.height
            margin = h * 0.07
            for b in page.get_text("blocks"):
                if not isinstance(b[4], str):
                    continue
                y0, y1 = b[1], b[3]
                text = b[4].strip().replace("\n", " ")
                text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
                # skip header/footer zone and bare page numbers
                if y0 < margin or y1 > h - margin:
                    continue
                if PAGE_NO_RE.match(text):
                    continue
                if len(text) >= min_chars:
                    paras.append(text)
        doc.close()
        return paras
    except Exception as e:
        print(f"  [error] {pdf_path}: {e}", flush=True)
        return []


def main() -> None:
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} papers from {CSV_IN}")

    rows: list[dict] = []
    for i, paper in df.iterrows():
        pid      = str(paper["paper_id"])
        pdf_path = f"{PDF_DIR}/{pid}.pdf"
        sector   = str(paper.get("sector", "unknown"))
        year     = int(paper["year"])

        paras = extract_paragraphs(pdf_path)
        for idx, text in enumerate(paras):
            rows.append({
                "paper_id": pid,
                "year":     year,
                "sector":   sector,
                "para_idx": idx,
                "text":     text,
            })

        print(f"  [{i+1}/{len(df)}] {pid}: {len(paras)} paragraphs", flush=True)

    out = pd.DataFrame(rows, columns=["paper_id", "year", "sector", "para_idx", "text"])
    out.to_csv(CSV_OUT, index=False)
    print(f"\nSaved {len(out):,} paragraphs from {len(df)} papers -> {CSV_OUT}")
    print(f"  median paragraphs/paper: {out.groupby('paper_id').size().median():.0f}")


if __name__ == "__main__":
    main()
