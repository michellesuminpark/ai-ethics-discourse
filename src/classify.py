"""
classify.py — LLM-based affiliation classification + LOO-CV validation.

Input:
  data/raw/arxiv_pilot.csv            — paper metadata (title, abstract, names)
  data/processed/affiliation_text.csv — first 2 pages + acknowledgments

Steps:
  1. LLM open-ended per-author classification
     → results/author_affiliations_raw.csv

  2. Aggregate to paper level (counts + proportions per institution type)
     → results/author_affiliations_paper_level.csv

  3. Post-hoc collapsing: institution types appearing in < MIN_PAPERS papers → "other"
     → results/author_affiliations_collapsed.csv
     The proportion columns from this file are the primary independent variable
     for all downstream analysis.

  4. LOO-CV validation (binary: top-2 non-other institution types)
     → results/loo_cv_results.json
     Checks coherence of LLM labels from text features alone.

Requires ANTHROPIC_API_KEY (loaded from .env).
"""

import anthropic
import pandas as pd
import numpy as np
import json
import os
import re
import time
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

load_dotenv()

CSV_IN    = "data/raw/arxiv_pilot.csv"
AFFIL_IN  = "data/processed/affiliation_text.csv"
RAW_OUT   = "results/author_affiliations_raw.csv"
PAPER_OUT = "results/author_affiliations_paper_level.csv"
COLL_OUT  = "results/author_affiliations_collapsed.csv"
LOOCV_OUT = "results/loo_cv_results.json"

MODEL           = "claude-sonnet-4-6"
REQUEST_DELAY   = 1.0
MAX_RETRIES     = 5
RETRY_BASE      = 10
MAX_AFFIL_CHARS = 4000
MIN_PAPERS      = 15

os.makedirs("results", exist_ok=True)


CLASSIFY_SYSTEM = (
    "You are a research assistant helping classify author affiliations for an "
    "academic study of AI ethics. Be precise and return only valid JSON."
)

CLASSIFY_USER = """\
Below is affiliation text from the first pages and acknowledgments of an AI research paper.
The paper's authors are: {names}

For each author, identify their institutional affiliation from the text and assign a short \
free-form institution type label. Do not use a fixed taxonomy — use whatever label best \
describes the institution.

Examples of possible labels (but don't limit yourself to these):
  "research university", "tech company", "government lab", "national lab",
  "nonprofit", "hospital", "independent researcher", "industry research lab", "think tank"

Return a JSON array with exactly one object per author:
[
  {{"author_name": "...", "affiliation_string": "...", "institution_type": "..."}},
  ...
]

If an author's affiliation cannot be determined from the text, set institution_type to "unknown".
Return ONLY the JSON array, no other text.

Affiliation text:
{affiliation_text}
"""


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _call_llm(client: anthropic.Anthropic, paper_id: str,
              names: str, affil_text: str) -> str:
    user_msg = CLASSIFY_USER.format(
        names=names,
        affiliation_text=affil_text[:MAX_AFFIL_CHARS],
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                temperature=1,
                system=CLASSIFY_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = RETRY_BASE * (2 ** (attempt - 1))
            print(f"    [rate limit] {paper_id} — sleeping {wait}s", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == MAX_RETRIES:
                print(f"    [API error] {paper_id}: {e}", flush=True)
                return ""
            time.sleep(RETRY_BASE * attempt)
    return ""


def _parse_json_response(response: str, fallback_names: list[str]) -> list[dict]:
    """Extract JSON array from LLM response, with graceful fallback."""
    for pattern in (r"\[\s*\{.+?\}\s*\]", r"\[.+?\]"):
        m = re.search(pattern, response, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, list) and data:
                    return data
            except json.JSONDecodeError:
                pass
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [
        {"author_name": n, "affiliation_string": "", "institution_type": "unknown"}
        for n in fallback_names
    ]


def _safe_col(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).strip("_")


# ── Step 1: LLM classification ────────────────────────────────────────────────

def run_llm_classification(meta_df: pd.DataFrame, affil_df: pd.DataFrame,
                            client: anthropic.Anthropic) -> pd.DataFrame:
    done_ids: set[str] = set()
    if os.path.exists(RAW_OUT):
        existing = pd.read_csv(RAW_OUT, dtype={"paper_id": str})
        done_ids = set(existing["paper_id"].astype(str))
        print(f"Resuming — {len(done_ids)} papers already classified")

    merged = meta_df.merge(affil_df.drop(columns=["year"], errors="ignore"),
                           on="paper_id", how="left")

    for i, paper in merged.iterrows():
        pid   = str(paper["paper_id"])
        year  = int(paper["year"])
        names = str(paper.get("names", ""))
        affil = str(paper.get("affiliation_text", ""))

        if pid in done_ids:
            continue

        print(f"  [{i+1}/{len(merged)}] {pid}...", end=" ", flush=True)

        raw        = _call_llm(client, pid, names, affil)
        name_list  = [n.strip() for n in names.split(";") if n.strip()]
        authors    = _parse_json_response(raw, name_list)

        rows = [
            {
                "paper_id":           pid,
                "year":               year,
                "author_name":        a.get("author_name", ""),
                "affiliation_string": a.get("affiliation_string", ""),
                "institution_type":   a.get("institution_type", "unknown"),
            }
            for a in authors
        ]

        n_types = len({r["institution_type"] for r in rows})
        print(f"{len(rows)} authors, {n_types} type(s)", flush=True)

        pd.DataFrame(rows).to_csv(
            RAW_OUT, mode="a", header=not os.path.exists(RAW_OUT), index=False,
        )
        done_ids.add(pid)
        time.sleep(REQUEST_DELAY)

    return pd.read_csv(RAW_OUT, dtype={"paper_id": str})


# ── Step 2: Aggregate to paper level ─────────────────────────────────────────

def aggregate_paper_level(raw_df: pd.DataFrame) -> pd.DataFrame:
    all_types = sorted(raw_df["institution_type"].dropna().unique())
    rows: list[dict] = []

    for pid, grp in raw_df.groupby("paper_id"):
        total = len(grp)
        vc    = grp["institution_type"].value_counts()
        top   = vc.index[0] if len(vc) > 0 else "unknown"
        dominant = top if (len(vc) > 0 and vc.iloc[0] / total >= 0.5) else "mixed"

        row: dict = {
            "paper_id":      pid,
            "year":          grp["year"].iloc[0],
            "total_authors": total,
            "dominant_type": dominant,
        }
        for t in all_types:
            n    = int(vc.get(t, 0))
            safe = _safe_col(t)
            row[f"n_{safe}"]    = n
            row[f"prop_{safe}"] = round(n / total, 3) if total > 0 else 0.0
        rows.append(row)

    paper_df = pd.DataFrame(rows)
    paper_df.to_csv(PAPER_OUT, index=False)
    print(f"Saved paper-level affiliations -> {PAPER_OUT}")
    print(f"  {len(paper_df)} papers, {len(all_types)} institution types")
    print("\nDominant type distribution (uncollapsed):")
    print(paper_df["dominant_type"].value_counts().to_string())
    return paper_df


# ── Step 3: Post-hoc collapsing ──────────────────────────────────────────────

def collapse_rare_types(raw_df: pd.DataFrame, min_papers: int = MIN_PAPERS) -> pd.DataFrame:
    # Count distinct papers each institution type appears in
    type_paper_counts = raw_df.groupby("institution_type")["paper_id"].nunique()
    rare = set(type_paper_counts[type_paper_counts < min_papers].index)

    print(f"\nCollapsing {len(rare)} rare types (< {min_papers} papers) → 'other':")
    for t in sorted(rare):
        print(f"  {t}: {type_paper_counts[t]} papers")

    raw_c = raw_df.copy()
    raw_c["institution_type"] = raw_c["institution_type"].map(
        lambda t: "other" if t in rare else t
    )

    all_types_c = sorted(raw_c["institution_type"].unique())
    rows: list[dict] = []

    for pid, grp in raw_c.groupby("paper_id"):
        total = len(grp)
        vc    = grp["institution_type"].value_counts()
        top   = vc.index[0] if len(vc) > 0 else "unknown"
        dominant = top if (len(vc) > 0 and vc.iloc[0] / total >= 0.5) else "mixed"

        row: dict = {
            "paper_id":      pid,
            "year":          grp["year"].iloc[0],
            "total_authors": total,
            "dominant_type": dominant,
        }
        for t in all_types_c:
            n    = int(vc.get(t, 0))
            safe = _safe_col(t)
            row[f"n_{safe}"]    = n
            row[f"prop_{safe}"] = round(n / total, 3) if total > 0 else 0.0
        rows.append(row)

    collapsed_df = pd.DataFrame(rows)
    collapsed_df.to_csv(COLL_OUT, index=False)
    print(f"Saved collapsed affiliations -> {COLL_OUT}")
    print("\nDominant type distribution (collapsed):")
    print(collapsed_df["dominant_type"].value_counts().to_string())
    return collapsed_df


# ── Step 4: LOO-CV validation ─────────────────────────────────────────────────

def run_loocv(collapsed_df: pd.DataFrame, meta_df: pd.DataFrame,
              affil_df: pd.DataFrame) -> dict:
    EXCLUDED = {"other", "mixed", "unknown"}
    type_dist = (
        collapsed_df["dominant_type"]
        .value_counts()
        .drop(labels=list(EXCLUDED & set(collapsed_df["dominant_type"].unique())),
              errors="ignore")
    )

    if len(type_dist) < 2:
        print("\nLOO-CV: fewer than 2 distinct types — skipping")
        return {}

    label_0, label_1 = type_dist.index[0], type_dist.index[1]
    print(f"\nLOO-CV: '{label_0}' (0) vs '{label_1}' (1)")

    binary = collapsed_df[collapsed_df["dominant_type"].isin([label_0, label_1])].copy()
    binary["label"] = (binary["dominant_type"] == label_1).astype(int)

    binary = (
        binary
        .merge(meta_df[["paper_id", "title", "abstract"]].astype({"paper_id": str}),
               on="paper_id", how="left")
        .merge(affil_df[["paper_id", "affiliation_text"]].astype({"paper_id": str}),
               on="paper_id", how="left")
    )
    binary["feature_text"] = (
        binary["title"].fillna("") + " "
        + binary["abstract"].fillna("") + " "
        + binary["affiliation_text"].fillna("")
    )

    X = binary["feature_text"].values
    y = binary["label"].values
    n = len(binary)

    if n < 10:
        print(f"  Only {n} papers for LOO-CV — skipping (too few)")
        return {}

    print(f"  n={n}  ({label_0}={int((y==0).sum())}, {label_1}={int((y==1).sum())})")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20_000,
                                   sublinear_tf=True, min_df=2)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])

    loo               = LeaveOneOut()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), 1):
        if fold % 20 == 0:
            print(f"  fold {fold}/{n}...", flush=True)
        pipeline.fit(X[train_idx], y[train_idx])
        prob = pipeline.predict_proba(X[test_idx])[0, 1]
        y_true.append(int(y[test_idx[0]]))
        y_pred.append(int(prob >= 0.5))
        y_prob.append(prob)

    ya = np.array(y_true)
    yp = np.array(y_pred)
    yb = np.array(y_prob)

    results = {
        "label_0":    label_0,
        "label_1":    label_1,
        "n_papers":   n,
        "n_label_0":  int((ya == 0).sum()),
        "n_label_1":  int((ya == 1).sum()),
        "auc":        round(float(roc_auc_score(ya, yb)), 4),
        "accuracy":   round(float(accuracy_score(ya, yp)), 4),
        "f1_macro":   round(float(f1_score(ya, yp, average="macro")), 4),
        "f1_label_0": round(float(f1_score(ya, yp, pos_label=0)), 4),
        "f1_label_1": round(float(f1_score(ya, yp, pos_label=1)), 4),
    }

    print(f"\n{'='*50}\nLOO-CV Results")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print()
    print(classification_report(ya, yp, target_names=[label_0, label_1]))

    with open(LOOCV_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved LOO-CV results -> {LOOCV_OUT}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    client = anthropic.Anthropic(api_key=api_key)

    meta_df  = pd.read_csv(CSV_IN,   dtype={"paper_id": str})
    affil_df = pd.read_csv(AFFIL_IN, dtype={"paper_id": str})
    print(f"Loaded {len(meta_df)} papers  from {CSV_IN}")
    print(f"Loaded {len(affil_df)} affil rows from {AFFIL_IN}")

    print("\nStep 1: LLM affiliation classification...")
    raw_df = run_llm_classification(meta_df, affil_df, client)
    print(f"  Total author records: {len(raw_df):,}")

    print("\nStep 2: Aggregating to paper level...")
    aggregate_paper_level(raw_df)

    print("\nStep 3: Collapsing rare institution types...")
    collapsed_df = collapse_rare_types(raw_df)

    print("\nStep 4: LOO-CV validation...")
    run_loocv(collapsed_df, meta_df, affil_df)


if __name__ == "__main__":
    main()
