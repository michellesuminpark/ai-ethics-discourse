"""
bertopic_run.py — BERTopic topic modeling in two modes.

Usage:
    python src/bertopic_run.py --mode ethics   # 44 substantive papers (default)
    python src/bertopic_run.py --mode corpus   # all 400 papers

ethics mode:
  Input : papers where ethics_substantive=1 (N=44)
  Goal  : validate A/B/C LLM theme structure; check emergent cluster alignment
  Output: results/bertopic_ethics_*.csv + results/bertopic_ethics_model/

corpus mode:
  Input : all 400 papers
  Goal  : map full CS.AI topic landscape; overlay ethics-substantive papers
  Output: results/bertopic_corpus_*.csv + results/bertopic_corpus_model/

Fixes applied to both modes:
  - Section header tokens ([Introduction], [1. Conclusion], etc.) stripped
  - Extended stopword list removes citation noise (arxiv, et al, years, venues)
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from umap import UMAP

ETHICS_CSV = "data/processed/ethics_sections.csv"
CODES_CSV  = "results/ethics_codes.csv"
AFFIL_CSV  = "results/author_affiliations_collapsed.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("results", exist_ok=True)

# ── Noise tokens to suppress in c-TF-IDF ─────────────────────────────────────

EXTRA_STOPWORDS = [
    # Citation / reference noise
    "et", "al", "arxiv", "doi", "http", "https", "www", "preprint",
    "fig", "figure", "table", "eq", "equation", "section", "appendix",
    # Publication venues
    "ieee", "acm", "neurips", "icml", "iclr", "aaai", "ijcai",
    "cvpr", "emnlp", "naacl", "proceedings", "conference", "workshop", "journal",
    # Years
    "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    # Generic filler
    "paper", "work", "approach", "method", "model", "models",
    "propose", "proposed", "show", "shown", "result", "results",
    "use", "used", "using", "also", "however", "thus", "therefore",
    "based", "given", "may", "one", "two", "three",
]

_HEADER_RE = re.compile(r"\[[^\]]{1,80}\]\s*", re.IGNORECASE)


def clean_doc(text: str) -> str:
    """Strip section headers and collapse whitespace."""
    text = _HEADER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── BERTopic runner ───────────────────────────────────────────────────────────

def run_bertopic(
    docs:      list[str],
    pids:      list[str],
    years:     list[int],
    mode:      str,
    codes_df:  pd.DataFrame,
) -> None:
    n_docs = len(docs)
    print(f"\nCorpus: {n_docs} documents  |  mode: {mode}")

    # Cleaned documents for c-TF-IDF (raw embeddings use original text)
    docs_clean = [clean_doc(d) for d in docs]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder   = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding documents...")
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=32)

    # UMAP — smaller n_neighbors for small corpora
    umap_model = UMAP(
        n_neighbors=min(15, n_docs - 1),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN — tighter for large corpus, looser for small
    if mode == "ethics":
        min_cluster_size = 3
        min_samples      = 1
    else:
        min_cluster_size = 10
        min_samples      = 3

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stops = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=all_stops,
        min_df=1,
    )

    print("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=15,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs_clean, embeddings)

    # ── Topic summary ──────────────────────────────────────────────────────────

    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info) - 1
    n_noise    = int((np.array(topics) == -1).sum())
    print(f"\nFound {n_topics} topics  |  {n_noise} noise docs (-1)")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    topic_rows = []
    for tid in topic_info["Topic"]:
        if tid == -1:
            continue
        terms    = topic_model.get_topic(tid)
        top10    = "; ".join(f"{w}({s:.3f})" for w, s in terms[:10])
        info_row = topic_info[topic_info["Topic"] == tid].iloc[0]
        topic_rows.append({
            "topic_id":  tid,
            "label":     info_row["Name"],
            "count":     info_row["Count"],
            "top_terms": top10,
        })

    topics_out = f"results/bertopic_{mode}_topics.csv"
    pd.DataFrame(topic_rows).to_csv(topics_out, index=False)
    print(f"Saved topic summary -> {topics_out}")

    # ── Per-paper output ───────────────────────────────────────────────────────

    out_df = pd.DataFrame({"paper_id": pids, "year": years, "topic": topics})

    theme_cols = ["paper_id", "ethics_theme", "ethics_substantive",
                  "ethics_justification"]
    out_df = out_df.merge(
        codes_df[theme_cols].astype({"paper_id": str}),
        on="paper_id", how="left",
    )

    if os.path.exists(AFFIL_CSV):
        affil_df  = pd.read_csv(AFFIL_CSV, dtype={"paper_id": str})
        prop_cols = ["paper_id", "dominant_type"] + [
            c for c in affil_df.columns if c.startswith("prop_")
        ]
        out_df = out_df.merge(
            affil_df[prop_cols].astype({"paper_id": str}),
            on="paper_id", how="left",
        )

    output_csv = f"results/bertopic_{mode}_output.csv"
    out_df.to_csv(output_csv, index=False)
    print(f"Saved per-paper assignments -> {output_csv}")

    # ── Distributions ─────────────────────────────────────────────────────────

    assigned = out_df[out_df["topic"] >= 0]

    print("\nTopic count by year:")
    if len(assigned) > 0:
        print(assigned.groupby(["year", "topic"]).size().unstack(fill_value=0).to_string())

    prop_cols_present = [c for c in out_df.columns if c.startswith("prop_")]
    if prop_cols_present and len(assigned) > 0:
        print("\nMean institution type proportions by topic:")
        print(assigned.groupby("topic")[prop_cols_present].mean().round(3).to_string())

    # ── A/B/C alignment (ethics mode only) ────────────────────────────────────

    if mode == "ethics":
        theme_sub = assigned[assigned["ethics_theme"].notna()].copy()
        if len(theme_sub) > 0:
            topic_to_theme = (
                theme_sub.groupby("topic")["ethics_theme"]
                .agg(lambda s: s.value_counts().index[0])
                .to_dict()
            )
            theme_sub["predicted_theme"] = theme_sub["topic"].map(topic_to_theme)
            agreement = (theme_sub["predicted_theme"] == theme_sub["ethics_theme"]).mean()
            ami       = adjusted_mutual_info_score(
                theme_sub["ethics_theme"], theme_sub["topic"]
            )
            print(f"\n{'='*60}")
            print("BERTopic ↔ LLM theme alignment:")
            print(f"  Papers evaluated : {len(theme_sub)}")
            print(f"  Topic agreement  : {agreement:.1%}")
            print(f"  Adj. Mutual Info : {ami:.4f}")
            print("\nTopic → dominant LLM theme:")
            for tid, theme in sorted(topic_to_theme.items()):
                n = len(theme_sub[theme_sub["topic"] == tid])
                print(f"  Topic {tid:>3d}: {theme}  ({n} papers)")
            print("\nCross-tabulation (topic × LLM theme):")
            print(pd.crosstab(theme_sub["topic"], theme_sub["ethics_theme"]).to_string())

    # ── Corpus mode: ethics overlap ───────────────────────────────────────────

    if mode == "corpus":
        print(f"\n{'='*60}")
        print("Ethics-substantive papers by topic:")
        sub_ids = set(
            codes_df[codes_df["ethics_substantive"] == 1]["paper_id"].astype(str)
        )
        out_df["is_ethics"] = out_df["paper_id"].isin(sub_ids).astype(int)
        eth_by_topic = (
            out_df[out_df["topic"] >= 0]
            .groupby("topic")
            .agg(n_papers=("paper_id", "count"),
                 n_ethics=("is_ethics", "sum"))
            .assign(pct_ethics=lambda d: (d["n_ethics"] / d["n_papers"] * 100).round(1))
        )
        print(eth_by_topic.to_string())
        # Save updated output with is_ethics flag
        out_df.to_csv(output_csv, index=False)

    # ── Save model ─────────────────────────────────────────────────────────────

    model_dir = f"results/bertopic_{mode}_model"
    topic_model.save(model_dir, serialization="safetensors", save_ctfidf=True)
    print(f"\nSaved BERTopic model -> {model_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ethics", "corpus", "both"],
                        default="both",
                        help="Which corpus to model (default: both)")
    args = parser.parse_args()

    print("Loading data...")
    ethics_df = pd.read_csv(ETHICS_CSV, dtype={"paper_id": str})
    codes_df  = pd.read_csv(CODES_CSV,  dtype={"paper_id": str})

    sub_ids = set(codes_df[codes_df["ethics_substantive"] == 1]["paper_id"].astype(str))
    print(f"Total papers: {len(ethics_df)}  |  Substantive: {len(sub_ids)}")

    modes = ["ethics", "corpus"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        if mode == "ethics":
            df = ethics_df[ethics_df["paper_id"].isin(sub_ids)].copy()
        else:
            df = ethics_df.copy()

        docs  = df["ethics_text"].fillna("").tolist()
        pids  = df["paper_id"].tolist()
        years = df["year"].tolist()

        run_bertopic(docs, pids, years, mode, codes_df)


if __name__ == "__main__":
    main()
