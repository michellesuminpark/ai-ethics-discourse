"""
visualize.py — Revised publication-quality figures for AI ethics discourse study.

All figures saved to results/figures/ as .pdf + .png (300 DPI).
No annotation text inside any figure — all interpretation goes in LaTeX captions.

Usage:
    python src/visualize.py
"""

import json
import os
import re
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Color constants ────────────────────────────────────────────────────────────

INST_COLORS = {
    "research university":   "#C08080",
    "industry research lab": "#800000",
    "tech company":          "#A04040",
    "mixed":                 "#404040",
    "other":                 "#808080",
}

THEME_COLORS = {"A": "#1B3A6B", "B": "#4A7AB5", "C": "#A8C4E0"}
THEME_LABELS = {
    "A": "A — Technical Ethics",
    "B": "B — Governance/Compliance",
    "C": "C — Structural/Justice",
}

ETHICS_BLUE = "#1B3A6B"
GRAY_BASE   = "#D0D0D0"
GRAY_OTHER  = "#A0A0A0"

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          16,
    "axes.titlesize":     18,
    "axes.labelsize":     16,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "figure.dpi":         150,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "#CCCCCC",
    "legend.fontsize":    14,
})

_generated: list = []
_failed:    list = []


def _save(fig, slug: str, n, suffix: str = "") -> None:
    stem = f"fig{n}{suffix}_{slug}"
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/{stem}.{ext}",
                    dpi=300, bbox_inches="tight", format=ext)
    _generated.append((n, slug, stem))
    print(f"  [ok] {stem}")


def _hgrid(ax) -> None:
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _vgrid(ax) -> None:
    ax.xaxis.grid(True, color="#EBEBEB", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _right_legend(fig, ax, ncol: int = 1, **kw) -> None:
    """Legend box to the right of the axes, outside the plot area."""
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, ncol=ncol, **kw)
    fig.subplots_adjust(right=0.68)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple:
    collapsed         = pd.read_csv("results/author_affiliations_collapsed.csv",
                                    dtype={"paper_id": str})
    ethics            = pd.read_csv("results/ethics_codes.csv",
                                    dtype={"paper_id": str})
    pilot             = pd.read_csv("data/raw/arxiv_pilot.csv",
                                    dtype={"paper_id": str})
    bert_corpus_out   = pd.read_csv("results/bertopic_corpus_output.csv",
                                    dtype={"paper_id": str})
    bert_corpus_topics = pd.read_csv("results/bertopic_corpus_topics.csv")
    with open("results/loo_cv_results.json") as f:
        loocv = json.load(f)
    return collapsed, ethics, pilot, loocv, bert_corpus_out, bert_corpus_topics


# ── Figure 1: Corpus composition ──────────────────────────────────────────────

def fig1(collapsed: pd.DataFrame) -> None:
    counts = collapsed["dominant_type"].value_counts()
    order  = [t for t in counts.index if t != "other"]
    if "other" in counts.index:
        order.append("other")
    counts = counts.reindex(order)
    total  = len(collapsed)
    colors = [INST_COLORS.get(t, "#808080") for t in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(counts)), counts.values, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Number of Papers")
    ax.set_title(f"Pilot Corpus Composition by Institution Type (N=399)", pad=10)
    _vgrid(ax)
    for i, (val, idx) in enumerate(zip(counts.values, counts.index)):
        ax.text(val + 1.5, i, f"{val} ({val/total*100:.1f}%)",
                va="center", fontsize=10)
    ax.set_xlim(0, counts.max() * 1.3)
    fig.tight_layout()
    _save(fig, "corpus_composition", 1)
    plt.close(fig)


# ── Figure 2: Ethics engagement rate by institution type ──────────────────────

def fig2(collapsed: pd.DataFrame, ethics: pd.DataFrame) -> None:
    merged = collapsed[["paper_id", "dominant_type"]].merge(
        ethics[["paper_id", "ethics_substantive"]], on="paper_id", how="inner"
    )
    grp  = merged.groupby("dominant_type")
    rate = (grp["ethics_substantive"].mean() * 100).sort_values(ascending=False)
    n    = grp.size()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(rate)), rate.values, color="#800000", width=0.6, edgecolor="white")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(rate.index, rotation=20, ha="right")
    ax.set_ylabel("Ethics Engagement Rate (%)")
    ax.set_title("Substantive Ethics Engagement Rate by Institution Type", pad=10)
    _hgrid(ax)
    for i, (val, idx) in enumerate(zip(rate.values, rate.index)):
        ax.text(i, val + 0.4, f"{val:.1f}%\n(n={n[idx]})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, rate.max() * 1.5)
    fig.tight_layout()
    _save(fig, "ethics_by_insttype", 2)
    plt.close(fig)


# ── Figure 3: Theme distribution overall ─────────────────────────────────────

def fig3(ethics: pd.DataFrame) -> None:
    sub    = ethics[ethics["ethics_substantive"] == 1].copy()
    counts = sub["ethics_theme"].value_counts().reindex(["A", "B", "C"])
    total  = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8, 2.5))
    left = 0
    for theme in ["A", "B", "C"]:
        val = int(counts[theme])
        pct = val / total * 100
        ax.barh(0, val, left=left, color=THEME_COLORS[theme], height=0.5,
                label=f"{THEME_LABELS[theme]}  —  {val} ({pct:.0f}%)")
        left += val

    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Number of Papers")
    ax.set_title(f"Ethics Framing Theme Distribution (N={total} substantive papers)", pad=10)
    _right_legend(fig, ax)
    _save(fig, "theme_overall", 3)
    plt.close(fig)


# ── Figure 4A: Theme by institution type — grouped bar ────────────────────────

def fig4a(collapsed: pd.DataFrame, ethics: pd.DataFrame) -> None:
    sub    = ethics[ethics["ethics_substantive"] == 1][["paper_id", "ethics_theme"]]
    merged = collapsed[["paper_id", "dominant_type"]].merge(sub, on="paper_id", how="inner")
    keep   = merged.groupby("dominant_type").size()
    merged = merged[merged["dominant_type"].isin(keep[keep >= 3].index)]
    pivot  = (merged.groupby(["dominant_type", "ethics_theme"])
              .size().unstack(fill_value=0)
              .reindex(columns=["A", "B", "C"], fill_value=0))

    x, width = np.arange(len(pivot)), 0.25
    fig, ax  = plt.subplots(figsize=(9, 4))
    for i, theme in enumerate(["A", "B", "C"]):
        bars = ax.bar(x + (i - 1) * width, pivot[theme], width,
                      label=THEME_LABELS[theme], color=THEME_COLORS[theme],
                      edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08,
                        str(int(h)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Ethics Framing Theme by Institution Type", pad=10)
    _hgrid(ax)
    _right_legend(fig, ax)
    _save(fig, "theme_by_insttype_grouped", 4, suffix="a")
    plt.close(fig)


# ── Figure 4B: Theme by institution type — stacked proportional ───────────────

def fig4b(collapsed: pd.DataFrame, ethics: pd.DataFrame) -> None:
    sub    = ethics[ethics["ethics_substantive"] == 1][["paper_id", "ethics_theme"]]
    merged = collapsed[["paper_id", "dominant_type"]].merge(sub, on="paper_id", how="inner")
    keep   = merged.groupby("dominant_type").size()
    merged = merged[merged["dominant_type"].isin(keep[keep >= 3].index)]
    pivot  = (merged.groupby(["dominant_type", "ethics_theme"])
              .size().unstack(fill_value=0)
              .reindex(columns=["A", "B", "C"], fill_value=0))
    prop   = pivot.div(pivot.sum(axis=1), axis=0)

    x, width = np.arange(len(prop)), 0.5
    fig, ax  = plt.subplots(figsize=(9, 4))
    bottom   = np.zeros(len(prop))
    for theme in ["A", "B", "C"]:
        ax.bar(x, prop[theme].values, width, bottom=bottom,
               label=THEME_LABELS[theme], color=THEME_COLORS[theme], edgecolor="white")
        bottom += prop[theme].values

    ax.set_xticks(x)
    ax.set_xticklabels(prop.index, rotation=15, ha="right")
    ax.set_ylabel("Proportion of Substantive Papers")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Ethics Framing Theme Proportion by Institution Type", pad=10)
    ax.legend(loc="lower left", frameon=True, fontsize=12)
    _save(fig, "theme_by_insttype_stacked", 4, suffix="b")
    plt.close(fig)


# ── Figure 5: Ethics engagement rate over time ────────────────────────────────

def fig5(ethics: pd.DataFrame, pilot: pd.DataFrame) -> None:
    merged = pilot[["paper_id", "year"]].merge(
        ethics[["paper_id", "ethics_substantive"]], on="paper_id", how="left"
    )
    merged["ethics_substantive"] = (
        pd.to_numeric(merged["ethics_substantive"], errors="coerce").fillna(0)
    )
    by_year = merged.groupby("year")
    rate    = (by_year["ethics_substantive"].sum() / by_year.size() * 100).round(1)
    years   = sorted(rate.index)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, [rate[y] for y in years], color="#800000",
            marker="o", linewidth=2, markersize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Ethics Engagement Rate (%)")
    ax.set_title("Ethics Engagement Rate by Year (2018–2025)", pad=10)
    ax.set_xticks(years)
    ax.set_ylim(0, rate.max() * 1.3)
    _hgrid(ax)
    fig.tight_layout()
    _save(fig, "ethics_over_time", 5)
    plt.close(fig)


# ── Figure 6: Thematic trends over time ───────────────────────────────────────

def fig6(ethics: pd.DataFrame, pilot: pd.DataFrame) -> None:
    sub    = ethics[ethics["ethics_substantive"] == 1][["paper_id", "ethics_theme"]]
    merged = pilot[["paper_id", "year"]].merge(sub, on="paper_id", how="inner")
    by_year = (merged.groupby("year")["ethics_theme"]
               .value_counts().unstack(fill_value=0)
               .reindex(columns=["A", "B", "C"], fill_value=0))
    by_year  = by_year.loc[by_year.sum(axis=1) >= 2]
    years    = list(by_year.index)
    x, width = np.arange(len(years)), 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, theme in enumerate(["A", "B", "C"]):
        bars = ax.bar(x + (i - 1) * width, by_year[theme].values, width,
                      label=THEME_LABELS[theme], color=THEME_COLORS[theme],
                      edgecolor="white")
        for bar, v in zip(bars, by_year[theme].values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                        str(int(v)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Substantive Papers")
    ax.set_title("Ethics Framing Theme Distribution Over Time", pad=10)
    _hgrid(ax)
    _right_legend(fig, ax)
    _save(fig, "theme_over_time", 6)
    plt.close(fig)


# ── Figure 7: LOO-CV results ──────────────────────────────────────────────────

def fig7(loocv: dict) -> None:
    l0 = loocv["label_0"]
    l1 = loocv["label_1"]
    metrics = [
        ("AUC",                 loocv["auc"],        "#4A7AB5"),
        ("Accuracy",            loocv["accuracy"],   "#4A7AB5"),
        ("F1 macro",            loocv["f1_macro"],   "#4A7AB5"),
        (f"F1 ({l0[:14]})",     loocv["f1_label_0"], "#C08080"),
        (f"F1 ({l1[:14]})",     loocv["f1_label_1"], "#800000"),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(metrics)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.22)
    ax.set_title(f"LOO-CV Classifier Performance:\n{l0.title()} vs. {l1.title()}", pad=10)
    _vgrid(ax)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)
    fig.tight_layout()
    _save(fig, "loocv_results", 7)
    plt.close(fig)


# ── Figure 8: BERTopic corpus landscape ──────────────────────────────────────

def _topic_label(tid: int, topics_df: pd.DataFrame) -> str:
    row = topics_df[topics_df["topic_id"] == tid]
    if row.empty:
        return f"T{tid}"
    raw   = str(row.iloc[0]["label"])
    words = raw.split("_")[1:]          # strip leading number
    seen, unique = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return f"T{tid}: {', '.join(unique[:4])}"


def fig8(bert_corpus_out: pd.DataFrame, bert_corpus_topics: pd.DataFrame) -> None:
    assigned = bert_corpus_out[bert_corpus_out["topic"] >= 0].copy()

    # Derive is_ethics from ethics_substantive if is_ethics column missing
    if "is_ethics" not in assigned.columns:
        assigned["is_ethics"] = (
            pd.to_numeric(assigned.get("ethics_substantive", 0), errors="coerce")
            .fillna(0).astype(int)
        )

    stats = (assigned.groupby("topic")
             .agg(n_papers=("paper_id", "count"),
                  n_ethics=("is_ethics", "sum"))
             .reset_index())
    stats["pct_ethics"] = stats["n_ethics"] / stats["n_papers"] * 100
    stats["label"]      = stats["topic"].apply(
        lambda t: _topic_label(int(t), bert_corpus_topics)
    )
    stats = stats.sort_values("pct_ethics", ascending=True)   # ascending → top row = highest

    max_rate = stats["pct_ethics"].max()
    eth_cmap = mcolors.LinearSegmentedColormap.from_list("eth", [GRAY_BASE, ETHICS_BLUE])

    def bar_color(row):
        if row["topic"] == 5:
            return ETHICS_BLUE
        t = row["pct_ethics"] / max_rate if max_rate > 0 else 0
        return eth_cmap(t)

    colors = [bar_color(row) for _, row in stats.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(stats))
    ax.barh(y, stats["n_papers"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(stats["label"], fontsize=9.5)
    ax.set_xlabel("Number of Papers in Topic Cluster")
    ax.set_title("BERTopic Topic Landscape: Ethics Rate by Cluster (N=399)", pad=10)
    _vgrid(ax)

    for i, (_, row) in enumerate(stats.iterrows()):
        if row["pct_ethics"] > 0:
            ax.text(row["n_papers"] + 0.5, i,
                    f"{row['pct_ethics']:.0f}% ({int(row['n_ethics'])})",
                    va="center", fontsize=9)
    ax.set_xlim(0, stats["n_papers"].max() * 1.3)

    sm   = plt.cm.ScalarMappable(cmap=eth_cmap,
                                  norm=mcolors.Normalize(0, max_rate))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Ethics Engagement Rate (%)", fontsize=9)

    fig.tight_layout()
    _save(fig, "bertopic_landscape", 8)
    plt.close(fig)


# ── Figure 9: Clustered vs embedded ───────────────────────────────────────────

def fig9(bert_corpus_out: pd.DataFrame, ethics: pd.DataFrame) -> None:
    sub_ids = set(ethics[ethics["ethics_substantive"] == 1]["paper_id"].astype(str))
    sub_df  = bert_corpus_out[bert_corpus_out["paper_id"].isin(sub_ids)].copy()
    total   = len(sub_df)
    n_t5    = int((sub_df["topic"] == 5).sum())
    n_other = total - n_t5

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(0, n_t5, color=ETHICS_BLUE, height=0.7,
            label=f"Core ethics cluster (T5)  —  {n_t5} papers ({n_t5/total:.0%})")
    ax.barh(0, n_other, left=n_t5, color=THEME_COLORS["C"], height=0.7,
            label=f"Embedded in technical topics  —  {n_other} papers ({n_other/total:.0%})")
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Number of Substantive Papers")
    ax.set_title("Distribution of Ethics-Substantive Papers Across BERTopic Topics", pad=10)
    _right_legend(fig, ax)
    _save(fig, "clustered_vs_embedded", 9)
    plt.close(fig)


# ── Figure 10: UMAP of full corpus ────────────────────────────────────────────

def fig10(bert_corpus_out: pd.DataFrame) -> None:
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    ethics_df = pd.read_csv("data/processed/ethics_sections.csv",
                            dtype={"paper_id": str})
    merged    = bert_corpus_out[["paper_id", "topic"]].merge(
        ethics_df[["paper_id", "ethics_text"]], on="paper_id", how="left"
    )
    docs   = merged["ethics_text"].fillna("").tolist()
    topics = merged["topic"].tolist()

    print("  Encoding all papers...")
    embedder   = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

    print("  Running 2D UMAP...")
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1,
                   metric="cosine", random_state=42)
    coords  = umap_2d.fit_transform(embeddings)

    df = pd.DataFrame({
        "topic": topics,
        "ux":    coords[:, 0],
        "uy":    coords[:, 1],
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    # Non-T5 (background)
    mask_other = df["topic"] != 5
    ax.scatter(df.loc[mask_other, "ux"], df.loc[mask_other, "uy"],
               c=GRAY_OTHER, s=25, alpha=0.55, linewidths=0, label="Other topics")

    # T5 (foreground)
    mask_t5 = df["topic"] == 5
    ax.scatter(df.loc[mask_t5, "ux"], df.loc[mask_t5, "uy"],
               c=ETHICS_BLUE, s=80, alpha=0.95, linewidths=0,
               label="T5: Ethics cluster")

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_title(
        "Semantic Space of cs.AI Papers (2018–2025): Ethics Cluster Highlighted",
        pad=10,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=12, markerscale=1.5)
    _save(fig, "umap_corpus", 10)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    collapsed, ethics, pilot, loocv, bert_corpus_out, bert_corpus_topics = load_data()
    print(f"  {len(collapsed)} papers | "
          f"{int((ethics['ethics_substantive']==1).sum())} substantive")

    steps = [
        (1,  fig1,  (collapsed,),                    ""),
        (2,  fig2,  (collapsed, ethics),              ""),
        (3,  fig3,  (ethics,),                        ""),
        (4,  fig4a, (collapsed, ethics),              "a"),
        (4,  fig4b, (collapsed, ethics),              "b"),
        (5,  fig5,  (ethics, pilot),                  ""),
        (6,  fig6,  (ethics, pilot),                  ""),
        (7,  fig7,  (loocv,),                         ""),
        (8,  fig8,  (bert_corpus_out, bert_corpus_topics), ""),
        (9,  fig9,  (bert_corpus_out, ethics),        ""),
    ]

    print("\nGenerating figures 1–9...")
    for n, fn, args, _ in steps:
        try:
            fn(*args)
        except Exception as e:
            name = fn.__name__
            _failed.append((n, name, str(e)))
            print(f"  [fail] fig{n} ({name}): {e}")

    print("\nFigure 10: UMAP (~60s)...")
    try:
        fig10(bert_corpus_out)
    except Exception as e:
        _failed.append((10, "umap_corpus", str(e)))
        print(f"  [skip] fig10: {e}")

    print(f"\n{'='*55}")
    print(f"Generated {len(_generated)} file(s):")
    for n, slug, stem in sorted(_generated, key=lambda x: (str(x[0]), x[1])):
        print(f"  Fig {n}: results/figures/{stem}.[pdf|png]")
    if _failed:
        print(f"\nFailed ({len(_failed)}):")
        for n, slug, reason in _failed:
            print(f"  Fig {n}: {reason}")

    rows = [
        "# Figure Index\n",
        "| Fig | File stem | Data | Description |",
        "|-----|-----------|------|-------------|",
        "| 1 | fig1_corpus_composition | author_affiliations_collapsed.csv | Paper counts by dominant institution type (maroon shades). |",
        "| 2 | fig2_ethics_by_insttype | affiliations + ethics_codes.csv | Ethics engagement rate per institution type. |",
        "| 3 | fig3_theme_overall | ethics_codes.csv | A/B/C theme distribution across substantive papers (blue shades, legend outside). |",
        "| 4a | fig4a_theme_by_insttype_grouped | affiliations + ethics_codes.csv | Grouped bar of A/B/C counts per institution type (≥3 papers). |",
        "| 4b | fig4b_theme_by_insttype_stacked | affiliations + ethics_codes.csv | Proportional stacked bar of A/B/C per institution type. |",
        "| 5 | fig5_ethics_over_time | arxiv_pilot + ethics_codes.csv | Ethics engagement rate by year, line only. |",
        "| 6 | fig6_theme_over_time | arxiv_pilot + ethics_codes.csv | A/B/C theme counts per year (substantive papers only). |",
        "| 7 | fig7_loocv_results | loo_cv_results.json | LOO-CV metric bars (AUC, accuracy, F1). |",
        "| 8 | fig8_bertopic_landscape | bertopic_corpus_*.csv | Topic clusters sorted by ethics rate; gray-to-blue gradient. |",
        "| 9 | fig9_clustered_vs_embedded | bertopic_corpus_output + ethics_codes | 44 ethics papers: T5 cluster vs. embedded in technical topics. |",
        "| 10 | fig10_umap_corpus | ethics_sections + bertopic_corpus_output | 2D UMAP of all papers; T5 ethics cluster highlighted in blue. |",
    ]
    with open("results/figures/README.md", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Wrote results/figures/README.md")


if __name__ == "__main__":
    main()
