"""
bertopic_run.py — BERTopic topic modeling on ethics-adjacent paragraphs.

Uses sentence-transformers/all-MiniLM-L6-v2 embeddings + HDBSCAN + c-TF-IDF.
Runs only on paragraphs from papers with has_ethics_term=1.

Outputs:
  results/bertopic_topics.csv     — topic labels and top terms
  results/bertopic_doc_topics.csv — per-paragraph topic assignments
  results/bertopic_model/         — serialized BERTopic model

The topic distribution is compared against LLM theme codes (A/B/C) when
results/llm_codes.csv exists, to triangulate validity.
"""

import pandas as pd
import numpy as np
import os
import json

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

PARA_CSV    = "data/processed/paragraphs.csv"
ETHICS_CSV  = "data/processed/ethics_scores.csv"
LLM_CSV     = "results/llm_codes.csv"
TOPICS_OUT  = "results/bertopic_topics.csv"
DOCS_OUT    = "results/bertopic_doc_topics.csv"
MODEL_OUT   = "results/bertopic_model"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 15   # minimum paragraphs per topic cluster
MIN_SAMPLES      = 5    # HDBSCAN min_samples for noise tolerance

os.makedirs("results", exist_ok=True)


def main() -> None:
    print("Loading data...")
    paras  = pd.read_csv(PARA_CSV)
    scores = pd.read_csv(ETHICS_CSV)

    ethics_ids = set(scores[scores["has_ethics_term"] == 1]["paper_id"].astype(str))
    paras["paper_id"] = paras["paper_id"].astype(str)
    ethics_paras = paras[paras["paper_id"].isin(ethics_ids)].copy()

    print(f"Ethics-adjacent paragraphs: {len(ethics_paras):,} "
          f"from {len(ethics_ids)} papers")

    docs  = ethics_paras["text"].fillna("").tolist()
    pids  = ethics_paras["paper_id"].tolist()
    years = ethics_paras["year"].tolist()
    sects = ethics_paras["sector"].tolist()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding paragraphs...")
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
    )

    print("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedder,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=15,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # ── Topic summary ─────────────────────────────────────────────────────────

    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics + outlier cluster (-1)")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    topic_rows = []
    for tid in topic_info["Topic"]:
        if tid == -1:
            continue
        terms = topic_model.get_topic(tid)
        top_terms = "; ".join(f"{w}({s:.3f})" for w, s in terms[:10])
        row = topic_info[topic_info["Topic"] == tid].iloc[0]
        topic_rows.append({
            "topic_id":   tid,
            "label":      row["Name"],
            "count":      row["Count"],
            "top_terms":  top_terms,
        })
    pd.DataFrame(topic_rows).to_csv(TOPICS_OUT, index=False)
    print(f"\nSaved topics -> {TOPICS_OUT}")

    # ── Per-document topic assignments ────────────────────────────────────────

    doc_df = pd.DataFrame({
        "paper_id":  pids,
        "year":      years,
        "sector":    sects,
        "topic":     topics,
        "text":      [d[:200] for d in docs],
    })

    if os.path.exists(LLM_CSV):
        llm = pd.read_csv(LLM_CSV)[["paper_id", "theme", "substantive"]].copy()
        llm["paper_id"] = llm["paper_id"].astype(str)
        doc_df = doc_df.merge(llm, on="paper_id", how="left")
        print("Merged LLM theme codes into doc dataframe")

    doc_df.to_csv(DOCS_OUT, index=False)
    print(f"Saved doc-topic assignments -> {DOCS_OUT}")

    # ── Save model ────────────────────────────────────────────────────────────
    topic_model.save(MODEL_OUT, serialization="safetensors", save_ctfidf=True)
    print(f"Saved BERTopic model -> {MODEL_OUT}")

    # ── Alignment with LLM themes (if available) ──────────────────────────────

    if "theme" in doc_df.columns:
        print("\n" + "="*60)
        print("Topic distribution by LLM theme (substantive papers only):")
        sub = doc_df[doc_df["substantive"] == 1]
        if len(sub) > 0:
            pivot = (sub.groupby(["theme", "topic"])
                       .size()
                       .reset_index(name="count")
                       .sort_values(["theme", "count"], ascending=[True, False]))
            print(pivot.to_string(index=False))

    # ── Topic distribution by sector and year ────────────────────────────────

    print("\nTopic distribution by sector:")
    sect_pivot = (doc_df[doc_df["topic"] >= 0]
                  .groupby(["sector", "topic"])
                  .size()
                  .unstack(fill_value=0))
    print(sect_pivot.to_string())

    print("\nTopic distribution by year:")
    year_pivot = (doc_df[doc_df["topic"] >= 0]
                  .groupby(["year", "topic"])
                  .size()
                  .unstack(fill_value=0))
    print(year_pivot.to_string())


if __name__ == "__main__":
    main()
