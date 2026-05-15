"""
classify.py — Affiliation classification + LOO-CV evaluation.

Step 1 (if sector column missing or --reclassify flag): extract affiliation
text from PDFs and classify each paper as academia / industry / mixed /
unknown using heuristic keyword + email matching. Updates arxiv_pilot.csv.

Step 2: LOO-CV on the binary subset (academia=0, industry=1, n≈208).
Trains TF-IDF + Logistic Regression on title-page text to predict sector.
Saves AUC, accuracy, F1 per class to results/classify_loocv.json.
"""

import fitz
import pandas as pd
import numpy as np
import os
import re
import json
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

os.makedirs("results", exist_ok=True)

CSV_PATH = "data/raw/arxiv_pilot.csv"
PDF_DIR  = "data/pdfs"


# ── Affiliation text extraction ───────────────────────────────────────────────

def _zone_blocks(page) -> list[str]:
    """Top 60% + bottom 15% of page (title, authors, affiliation footnotes)."""
    h      = page.rect.height
    parts  = []
    for b in page.get_text("blocks"):
        y0, y1 = b[1], b[3]
        text   = b[4] if isinstance(b[4], str) else ""
        text   = text.strip().replace("\n", " ")
        if not text:
            continue
        if y0 < h * 0.60 or y1 > h * 0.85:
            parts.append(text)
    return parts


def extract_affiliation_raw(pdf_path: str) -> str:
    if not isinstance(pdf_path, str) or not os.path.exists(pdf_path):
        return ""
    try:
        doc   = fitz.open(pdf_path)
        parts = []
        for pg_no in range(min(2, len(doc))):
            parts.extend(_zone_blocks(doc[pg_no]))
        doc.close()
        raw = " | ".join(parts)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    except Exception:
        return ""


# ── Keyword patterns ──────────────────────────────────────────────────────────

def _compile(patterns: list[str]):
    return re.compile("|".join(patterns), re.IGNORECASE)


ACADEMIC_RE = _compile([
    r"universit(?:y|é|ät|at|á|i|ad|à|ade)",
    r"institute of technology", r"polytechnic",
    r"\bcollege\b", r"\bschool of\b", r"\bacademy\b",
    r"\binria\b", r"\bcnrs\b", r"max planck", r"fraunhofer",
    r"faculty of", r"department of", r"\.edu\b", r"\bac\.[a-z]{2}\b",
    r"\bcnr\b", r"\briken\b", r"\bcsiro\b", r"\bkaist\b",
    r"\bntu\b", r"\bnus\b", r"\beth\s+zurich\b", r"\bepfl\b", r"\bdtu\b",
])
INDUSTRY_RE = _compile([
    r"\bgoogle\b", r"\bdeepmind\b", r"\balphabet\b",
    r"\bmicrosoft\b", r"\bmsra?\b",
    r"\bmeta(?!\s*-)\b", r"\bfacebook\b", r"\bfair\b",
    r"\bamazon\b", r"\baws\b", r"\bapple(?!t)\b",
    r"\bopenai\b", r"\bhugging\s*face\b", r"\blinkedin\b",
    r"\bibm\b", r"\bsamsung\b", r"\bbaidu\b", r"\balibaba\b",
    r"\btencent\b", r"\bbytedance\b", r"\btiktok\b",
    r"\bnvidia\b", r"\bintel\b", r"\badobe\b", r"\bsalesforce\b",
    r"\btwitter\b", r"\bx\s*corp\b", r"\bwaymo\b", r"\btesla\b",
    r"\buber\b", r"\bbosch\b", r"\bsiemens\b", r"\bsony\b",
    r"\bqualcomm\b", r"\banthropic\b", r"\bcohere\b", r"\bmistral\b",
    r"\bscale\s*ai\b", r"\bai21\b", r"\bstability\s*ai\b",
    r"\binc\.", r"\bcorp\.", r"\bcorporation\b",
    r"\bltd\.", r"\bllc\b", r"\bs\.a\.\b", r"\bgmbh\b",
])
GOVT_RE = _compile([
    r"\bnist\b", r"\bdarpa\b", r"\bnasa\b",
    r"national\s+laborator(?:y|ies)", r"national\s+lab\b",
    r"\bsandia\b", r"los\s+alamos", r"lawrence\s+livermore",
    r"oak\s+ridge", r"\bargonne\b", r"pacific\s+northwest\s+national",
])
NONPROFIT_RE = _compile([
    r"partnership\s+on\s+ai", r"ai\s+now\s+institute",
    r"future\s+of\s+life", r"center\s+for\s+ai\s+safety",
    r"allen\s+institute", r"vector\s+institute",
    r"\bmila\b", r"montreal\s+institute", r"alan\s+turing\s+institute",
])


# ── Email domain helpers ──────────────────────────────────────────────────────

_IND_EMAIL_DOMAINS = {
    "google.com", "deepmind.com", "microsoft.com", "meta.com", "facebook.com",
    "amazon.com", "apple.com", "openai.com", "linkedin.com", "ibm.com",
    "samsung.com", "baidu.com", "alibaba-inc.com", "tencent.com",
    "bytedance.com", "nvidia.com", "intel.com", "adobe.com", "salesforce.com",
    "twitter.com", "waymo.com", "tesla.com", "uber.com", "bosch.com",
    "siemens.com", "sony.com", "qualcomm.com", "anthropic.com", "cohere.com",
    "mistral.ai", "huggingface.co", "ea.com", "stability.ai", "scaleai.com",
}


def _domain(email: str) -> str:
    return email.lower().rsplit("@", 1)[-1] if "@" in email else ""


def _is_academic_email(email: str) -> bool:
    d = _domain(email)
    if not d:
        return False
    if re.search(r"\.edu(\.[a-z]{2})?$", d):
        return True
    if re.search(r"\.ac\.[a-z]{2}$", d):
        return True
    if re.search(r"(^|\.)univ?[-.]", d):
        return True
    if d.endswith(".cnr.it") or d.endswith(".inria.fr"):
        return True
    if re.search(r"\.(dtu\.dk|epfl\.ch|ethz\.ch|kaist\.ac\.kr|ntu\.edu\.sg|nus\.edu\.sg)$", d):
        return True
    parts = d.split(".")
    if parts[-1] == "it" and any(p.startswith("uni") for p in parts[:-1]):
        return True
    if parts[-1] == "br" and len(parts) >= 2 and parts[-2].startswith("u"):
        return True
    return False


def _is_industry_email(email: str) -> bool:
    d = _domain(email)
    if not d:
        return False
    if d in _IND_EMAIL_DOMAINS or any(d.endswith("." + x) for x in _IND_EMAIL_DOMAINS):
        return True
    if d.endswith(".ai"):
        return True
    if d.endswith(".com") and not _is_academic_email(email):
        return True
    return False


_BRACE_RE = re.compile(r"\{([^}]+)\}@")


def count_email_sectors(emails_str: str) -> tuple[int, int, int]:
    if not isinstance(emails_str, str) or not emails_str.strip():
        return 0, 0, 0
    emails = []
    for token in emails_str.split(";"):
        token = token.strip()
        if not token:
            continue
        m = _BRACE_RE.match(token)
        if m:
            domain = token.rsplit("@", 1)[-1]
            for name in m.group(1).split(","):
                name = name.strip()
                if name:
                    emails.append(f"{name}@{domain}")
        else:
            emails.append(token)
    n_acad = sum(1 for e in emails if _is_academic_email(e))
    n_ind  = sum(1 for e in emails if _is_industry_email(e))
    return n_acad, n_ind, len(emails) - n_acad - n_ind


def classify_sector(affiliation_raw: str, emails: str = "") -> tuple[str, str]:
    affil = affiliation_raw or ""
    em    = emails or ""

    acad_from_email = any(_is_academic_email(e.strip()) for e in em.split(";") if e.strip())
    ind_from_email  = any(_is_industry_email(e.strip())  for e in em.split(";") if e.strip())
    acad_from_affil = bool(ACADEMIC_RE.search(affil))
    ind_from_affil  = bool(INDUSTRY_RE.search(affil))
    govt_from_affil = bool(GOVT_RE.search(affil + " " + em))
    ngo_from_affil  = bool(NONPROFIT_RE.search(affil + " " + em))

    from_email = acad_from_email or ind_from_email
    from_affil = acad_from_affil or ind_from_affil or govt_from_affil or ngo_from_affil

    if from_email and from_affil:
        source = "both"
    elif from_email:
        source = "email"
    elif from_affil:
        source = "affiliation_raw"
    else:
        source = "neither"

    labels = []
    if acad_from_email or acad_from_affil: labels.append("academia")
    if ind_from_email  or ind_from_affil:  labels.append("industry")
    if govt_from_affil:                    labels.append("government")
    if ngo_from_affil:                     labels.append("nonprofit")

    if not labels:
        sector = "unknown"
    elif len(labels) == 1:
        sector = labels[0]
    else:
        sector = "mixed:" + "+".join(labels)

    return sector, source


# ── Heuristic classification (Step 1) ────────────────────────────────────────

def run_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    affil_list, sector_list, source_list = [], [], []
    n_acad_list, n_ind_list, n_unk_list  = [], [], []

    for i, row in df.iterrows():
        pid      = str(row["paper_id"])
        pdf_path = f"{PDF_DIR}/{pid}.pdf"
        emails   = str(row.get("emails") or "")

        affil          = extract_affiliation_raw(pdf_path)
        sector, source = classify_sector(affil, emails)
        n_a, n_i, n_u  = count_email_sectors(emails)

        affil_list.append(affil)
        sector_list.append(sector)
        source_list.append(source)
        n_acad_list.append(n_a)
        n_ind_list.append(n_i)
        n_unk_list.append(n_u)

        print(f"  [{i+1}/{len(df)}] {pid}: {sector}  (src:{source}, acad:{n_a}, ind:{n_i})", flush=True)

    df = df.copy()
    df["affiliation_raw"]       = affil_list
    df["sector"]                = sector_list
    df["classification_source"] = source_list
    df["n_academic_emails"]     = n_acad_list
    df["n_industry_emails"]     = n_ind_list
    df["n_unknown_emails"]      = n_unk_list
    df["n_emails_found"]        = df["n_academic_emails"] + df["n_industry_emails"] + df["n_unknown_emails"]
    return df


# ── LOO-CV evaluation (Step 2) ───────────────────────────────────────────────

def run_loocv(df: pd.DataFrame) -> dict:
    """
    Train TF-IDF + LogisticRegression on title + affiliation_raw text to
    predict binary sector (academia=0, industry=1). Excludes mixed/unknown.
    Returns dict with AUC, accuracy, and per-class F1.
    """
    binary = df[df["sector"].isin(["academia", "industry"])].copy()
    binary["label"] = (binary["sector"] == "industry").astype(int)

    affil_col = "affiliation_raw" if "affiliation_raw" in binary.columns else None

    def make_text(row):
        parts = [str(row.get("title", "")), str(row.get("abstract", ""))]
        if affil_col:
            parts.append(str(row.get(affil_col, "")))
        return " ".join(parts)

    binary["feature_text"] = binary.apply(make_text, axis=1)

    X = binary["feature_text"].values
    y = binary["label"].values
    n = len(binary)

    print(f"\nLOO-CV on {n} papers (academia={int((y==0).sum())}, industry={int((y==1).sum())})")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])

    loo        = LeaveOneOut()
    y_true     = []
    y_pred     = []
    y_prob     = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), 1):
        if fold % 20 == 0:
            print(f"  fold {fold}/{n}...", flush=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[0, 1]
        pred = int(prob >= 0.5)

        y_true.append(int(y_test[0]))
        y_pred.append(pred)
        y_prob.append(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    auc      = roc_auc_score(y_true, y_prob)
    acc      = accuracy_score(y_true, y_pred)
    f1_acad  = f1_score(y_true, y_pred, pos_label=0)
    f1_ind   = f1_score(y_true, y_pred, pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"LOO-CV Results")
    print(f"  n={n}  (academia={int((y_true==0).sum())}, industry={int((y_true==1).sum())})")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 macro : {f1_macro:.4f}")
    print(f"  F1 academia (0): {f1_acad:.4f}")
    print(f"  F1 industry (1): {f1_ind:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["academia", "industry"]))

    results = {
        "n_papers":      n,
        "n_academia":    int((y_true == 0).sum()),
        "n_industry":    int((y_true == 1).sum()),
        "auc":           round(float(auc), 4),
        "accuracy":      round(float(acc), 4),
        "f1_macro":      round(float(f1_macro), 4),
        "f1_academia":   round(float(f1_acad), 4),
        "f1_industry":   round(float(f1_ind), 4),
    }

    out = "results/classify_loocv.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"LOO-CV results saved -> {out}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclassify", action="store_true",
                        help="Re-run heuristic classification even if sector column exists")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, dtype={"paper_id": str})
    print(f"Loaded {len(df)} papers")

    needs_classification = args.reclassify or "sector" not in df.columns

    if needs_classification:
        print("\nRunning heuristic affiliation classification...")
        df = run_heuristic(df)
        df.to_csv(CSV_PATH, index=False, escapechar="\\")
        print(f"Saved -> {CSV_PATH}")
    else:
        print("Sector column present — skipping heuristic step (use --reclassify to rerun)")

    print("\nSector breakdown:")
    print(df["sector"].value_counts().to_string())

    print("\nRunning LOO-CV evaluation...")
    run_loocv(df)


if __name__ == "__main__":
    main()
