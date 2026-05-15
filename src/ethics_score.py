"""
ethics_score.py — Paper-level ethics scoring from extracted paragraphs.

Reads data/processed/paragraphs.csv, applies clean ethics keyword matching
(excludes 'bias' and 'safety' which are contaminated by technical uses),
and writes data/processed/ethics_scores.csv.

Output columns:
  paper_id, year, sector,
  has_ethics_term         (1 if any paragraph matches, else 0),
  ethics_paragraph_count  (number of matching paragraphs),
  ethics_paragraphs       (pipe-separated matched paragraph texts, truncated)
"""

import pandas as pd
import re
import os

CSV_IN  = "data/processed/paragraphs.csv"
CSV_OUT = "data/processed/ethics_scores.csv"

# Intentionally excludes 'bias' (bias-variance tradeoff) and 'safety'
# (physical/vehicle safety) which fire too many false positives in cs.AI text.
CLEAN_ETHICS_TERMS = [
    "fairness", "ethical", "ethics", "harm", "transparency",
    "accountability", "justice", "equity",
]

TERM_PATTERNS = {t: re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
                 for t in CLEAN_ETHICS_TERMS}

ANY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in CLEAN_ETHICS_TERMS) + r")\b",
    re.IGNORECASE,
)

MAX_PARA_LEN = 500  # chars to keep per matched paragraph in the stored list


def score_paper_group(group: pd.DataFrame) -> dict:
    matched_texts = [
        row["text"][:MAX_PARA_LEN]
        for _, row in group.iterrows()
        if ANY_PATTERN.search(str(row["text"]))
    ]
    return {
        "paper_id":               group.iloc[0]["paper_id"],
        "year":                   group.iloc[0]["year"],
        "sector":                 group.iloc[0]["sector"],
        "has_ethics_term":        int(len(matched_texts) > 0),
        "ethics_paragraph_count": len(matched_texts),
        "ethics_paragraphs":      " ||| ".join(matched_texts),
    }


def main() -> None:
    paras = pd.read_csv(CSV_IN)
    print(f"Loaded {len(paras):,} paragraphs from {len(paras['paper_id'].unique())} papers")

    records = [
        score_paper_group(grp)
        for _, grp in paras.groupby("paper_id", sort=False)
    ]
    out = pd.DataFrame(records)

    out.to_csv(CSV_OUT, index=False)

    n_total  = len(out)
    n_hit    = out["has_ethics_term"].sum()
    n_para   = out["ethics_paragraph_count"].sum()
    print(f"\nEthics scoring complete:")
    print(f"  Papers with ethics hit : {n_hit}/{n_total} ({n_hit/n_total*100:.1f}%)")
    print(f"  Ethics paragraphs total: {n_para:,}")
    print(f"  Avg per hit paper      : {n_para/max(n_hit,1):.1f}")
    print(f"\nTerm used (clean set): {', '.join(CLEAN_ETHICS_TERMS)}")
    print(f"\nSaved -> {CSV_OUT}")

    # Per-term breakdown
    print("\nPer-term paragraph hit counts:")
    for term, pat in TERM_PATTERNS.items():
        count = paras["text"].apply(lambda t: bool(pat.search(str(t)))).sum()
        pct   = count / len(paras) * 100
        print(f"  {term:<18} {count:>7,}  ({pct:.2f}%)")


if __name__ == "__main__":
    main()
