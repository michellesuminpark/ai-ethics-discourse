"""
collect.py — arXiv cs.AI paper collection + PDF download.

Stratified random sample of SAMPLE_N papers per year (START_YEAR–END_YEAR),
primary category cs.AI only. Writes per-year checkpoints so the run is
resumable, then assembles data/raw/arxiv_pilot.csv.
"""

import arxiv
import fitz
import pandas as pd
import requests
import os
import time
import random
import re

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/checkpoints", exist_ok=True)
os.makedirs("data/pdfs", exist_ok=True)

CATEGORY    = "cs.AI"
START_YEAR  = 2018
END_YEAR    = 2025
SAMPLE_N    = 50
MAX_RESULTS = 10000
RANDOM_SEED = 42

API_DELAY   = 15
API_RETRIES = 5
RETRY_WAIT  = 60

EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
BRACE_RE   = re.compile(r"\{([^}]+)\}@([\w.\-]+\.[a-zA-Z]{2,})")
PAGE_NO_RE = re.compile(r"^\s*\d{1,4}\s*$")

_JUNK_LOCAL  = {
    "permissions", "rights", "copyright", "info", "contact", "support",
    "admin", "editor", "editors", "editorial", "noreply", "no-reply",
    "postmaster", "webmaster", "help", "submission", "submissions",
    "review", "correspondence", "firstname", "lastname", "name", "author",
}
_JUNK_DOMAIN = {"acm.org", "ieee.org", "springer.com", "elsevier.com", "wiley.com"}

COLS = [
    "paper_id", "title", "names", "emails", "texts",
    "abstract", "n_authors", "published_date", "year", "month",
    "primary_cat", "all_cats", "journal_ref", "url", "pdf_path", "extraction_status",
]


def _is_junk(email: str) -> bool:
    if "@" not in email:
        return False
    local, domain = email.lower().rsplit("@", 1)
    return local in _JUNK_LOCAL or domain in _JUNK_DOMAIN


def _find_emails(text: str) -> list[str]:
    found = []
    for m in BRACE_RE.finditer(text):
        domain = m.group(2)
        for name in m.group(1).split(","):
            name = name.strip()
            if name:
                found.append(f"{name}@{domain}")
    found.extend(EMAIL_RE.findall(text))
    seen: dict[str, bool] = {}
    for e in found:
        if e not in seen and not _is_junk(e):
            seen[e] = True
    return list(seen)


def collect_year_pool(client, year: int) -> list[dict]:
    date_from = f"{year}01010000"
    date_to   = f"{year}12312359"
    query     = f"cat:{CATEGORY} AND submittedDate:[{date_from} TO {date_to}]"

    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )

    for attempt in range(1, API_RETRIES + 1):
        try:
            pool = []
            for result in client.results(search):
                if result.primary_category != CATEGORY:
                    continue
                pid = result.entry_id.split("/")[-1].split("v")[0]
                pool.append({
                    "paper_id":       pid,
                    "title":          result.title.replace("\n", " ").strip(),
                    "abstract":       result.summary.replace("\n", " ").strip(),
                    "names":          "; ".join(a.name for a in result.authors),
                    "n_authors":      len(result.authors),
                    "published_date": result.published.strftime("%Y-%m-%d"),
                    "year":           result.published.year,
                    "month":          result.published.month,
                    "primary_cat":    result.primary_category,
                    "all_cats":       "; ".join(result.categories),
                    "journal_ref":    result.journal_ref or "",
                    "url":            result.entry_id,
                })
            return pool
        except arxiv.HTTPError as e:
            if e.status == 429:
                wait = RETRY_WAIT * attempt
                print(f"\n    [429] attempt {attempt}/{API_RETRIES} — sleeping {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            wait = RETRY_WAIT * attempt
            print(f"\n    [error: {e}] attempt {attempt}/{API_RETRIES} — sleeping {wait}s...", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Failed to collect pool for {year} after {API_RETRIES} attempts")


def sample_year_pool(pool: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def download_pdf(paper_id: str) -> str | None:
    pdf_path = f"data/pdfs/{paper_id}.pdf"
    if os.path.exists(pdf_path):
        return pdf_path
    url = f"https://arxiv.org/pdf/{paper_id}"
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "arxiv-ethics-study/1.0"})
            if resp.status_code == 429:
                wait = RETRY_WAIT * attempt
                print(f"\n    [PDF 429] sleeping {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            return pdf_path
        except Exception as e:
            print(f"\n    [PDF fail attempt {attempt}] {paper_id}: {e}", flush=True)
            time.sleep(10 * attempt)
    return None


def extract_emails(pdf_path: str) -> tuple[str, str]:
    if not isinstance(pdf_path, str) or not pdf_path or not os.path.exists(pdf_path):
        return "", "failed"
    try:
        doc   = fitz.open(pdf_path)
        pages = range(min(2, len(doc)))

        for method in ("", "text", "blocks"):
            if method == "blocks":
                text = ""
                for p in pages:
                    text += " ".join(b[4] for b in doc[p].get_text("blocks") if isinstance(b[4], str))
            else:
                text = "".join(doc[p].get_text(method) for p in pages)
            emails = _find_emails(text)
            if emails:
                doc.close()
                status = "success" if method == "" else "fallback_success"
                return "; ".join(emails), status

        doc.close()
        return "", "failed"
    except Exception as e:
        print(f"    [email error] {pdf_path}: {e}", flush=True)
        return "", "failed"


def extract_full_text(pdf_path: str) -> str:
    if not isinstance(pdf_path, str) or not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        doc  = fitz.open(pdf_path)
        body = []
        for page in doc:
            h      = page.rect.height
            margin = h * 0.07
            for b in page.get_text("blocks"):
                if not isinstance(b[4], str):
                    continue
                y0, y1 = b[1], b[3]
                text = b[4].strip().replace("\n", " ")
                if y0 < margin or y1 > h - margin:
                    continue
                if PAGE_NO_RE.match(text):
                    continue
                if len(text) < 40:
                    continue
                body.append(text)
        doc.close()
        raw = " ".join(body)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    except Exception as e:
        print(f"    [text error] {pdf_path}: {e}", flush=True)
        return ""


def checkpoint_path(year: int) -> str:
    return f"data/raw/checkpoints/{year}.csv"


def load_checkpoint(year: int) -> list[dict] | None:
    path = checkpoint_path(year)
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"paper_id": str})
        if "extraction_status" not in df.columns:
            df["extraction_status"] = "success"
        return df.to_dict("records")
    return None


def save_checkpoint(year: int, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=COLS).to_csv(checkpoint_path(year), index=False)


def collect() -> pd.DataFrame:
    client = arxiv.Client(
        page_size=100,
        delay_seconds=API_DELAY,
        num_retries=API_RETRIES,
    )

    years = list(range(START_YEAR, END_YEAR + 1))
    print(f"Category: {CATEGORY} (primary only)")
    print(f"Years: {START_YEAR}–{END_YEAR}  |  Sample: {SAMPLE_N}/year  |  Seed: {RANDOM_SEED}")
    print(f"Checkpoints: data/raw/checkpoints/\n")

    all_rows: list[dict] = []

    for i, year in enumerate(years, 1):
        cached = load_checkpoint(year)
        if cached is not None:
            print(f"[{i}/{len(years)}] {year}: checkpoint ({len(cached)} papers)", flush=True)
            all_rows.extend(cached)
            continue

        print(f"[{i}/{len(years)}] {year}: fetching pool...", flush=True)
        pool    = collect_year_pool(client, year)
        print(f"        pool={len(pool):,}", flush=True)
        sampled = sample_year_pool(pool, SAMPLE_N, seed=RANDOM_SEED + year)
        print(f"        sample={len(sampled)}", flush=True)

        year_rows: list[dict] = []
        for j, paper in enumerate(sampled, 1):
            pid = paper["paper_id"]
            print(f"        [{j}/{len(sampled)}] {pid} ...", end=" ", flush=True)

            pdf_path       = download_pdf(pid)
            time.sleep(3)
            emails, status = extract_emails(pdf_path)
            texts          = extract_full_text(pdf_path)
            print(f"[{status}] emails:{emails[:50] if emails else '(none)'}  chars:{len(texts)}", flush=True)

            paper["pdf_path"]          = pdf_path or ""
            paper["emails"]            = emails
            paper["extraction_status"] = status
            paper["texts"]             = texts
            year_rows.append(paper)

        save_checkpoint(year, year_rows)
        all_rows.extend(year_rows)
        print(f"        checkpoint saved\n", flush=True)
        time.sleep(5)

    df  = pd.DataFrame(all_rows, columns=COLS)
    out = "data/raw/arxiv_pilot.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} papers -> {out}")
    print(df["year"].value_counts().sort_index().to_string())
    return df


if __name__ == "__main__":
    collect()
