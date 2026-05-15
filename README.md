# AI Ethics Discourse Analysis

A computational text analysis project studying how ethics is discussed *within* AI development research — not what ethicists say about AI, but what AI researchers say about ethics, and whether that framing varies by institutional sector (academia vs. industry).

**Central question:** does where you work shape how you talk about AI ethics?

The hypothesis, drawing on Bourdieusian field theory, is that ethics framing in AI research is predominantly technical/managerial (metrics, benchmarks, compliance) rather than substantive (justice, power, harm to communities), and that this is more pronounced in industry-affiliated papers.

---

## Corpus

- **Source:** arXiv cs.AI (primary category), 2018–2025
- **Sample:** 50 papers/year × 8 years = 400 papers, stratified random sample (seed 42)
- **PDFs:** `data/pdfs/` (399 on disk; not tracked in git)
- **Metadata:** `data/raw/arxiv_pilot.csv`

---

## Repo structure

```
├── data/
│   ├── pdfs/                   # downloaded PDFs (~400)
│   ├── raw/
│   │   ├── arxiv_pilot.csv     # paper metadata + sector labels
│   │   └── checkpoints/        # per-year collection checkpoints
│   └── processed/
│       ├── paragraphs.csv      # one row per paragraph (output of parse.py)
│       └── ethics_scores.csv   # paper-level ethics scoring (output of ethics_score.py)
├── src/
│   ├── collect.py              # arXiv API collection + PDF download
│   ├── parse.py                # paragraph extraction from PDFs
│   ├── classify.py             # affiliation classification + LOO-CV
│   ├── ethics_score.py         # paper-level ethics scoring (clean keywords)
│   ├── llm_code.py             # LLM binary + thematic coding (Anthropic API)
│   └── bertopic_run.py         # BERTopic topic modeling
├── results/
│   ├── classify_loocv.json     # LOO-CV evaluation metrics
│   ├── llm_codes.csv           # LLM-assigned substantive/incidental + A/B/C theme
│   ├── bertopic_topics.csv     # BERTopic topic labels and top terms
│   └── bertopic_doc_topics.csv # per-paragraph topic assignments
├── notebooks/                  # exploratory analysis
├── outputs/figures/            # generated figures
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> BERTopic pulls in `torch` transitively via `sentence-transformers`. If you're on a GPU machine, install PyTorch separately first for CUDA support.

---

## Reproducing the analysis

Run steps in order. Each step reads from the previous step's output.

### Step 1 — Collect papers

```bash
python src/collect.py
```

Downloads metadata and PDFs for 50 cs.AI papers per year (2018–2025). Uses per-year checkpoints in `data/raw/checkpoints/` — safe to interrupt and resume. Writes `data/raw/arxiv_pilot.csv`.

*Skip if you already have the pilot data on disk.*

### Step 2 — Extract paragraphs

```bash
python src/parse.py
```

Reads each PDF with PyMuPDF, extracts body text blocks (≥50 chars), filters running headers/footers (top/bottom 7% of page height) and bare page numbers. Writes `data/processed/paragraphs.csv` (one row per paragraph).

### Step 3 — Affiliation classification + LOO-CV

```bash
python src/classify.py
```

Two stages:

1. **Heuristic classification** (skipped if `sector` column already exists — run `--reclassify` to force): extracts affiliation text from pages 1–2 of each PDF and classifies each paper as `academia / industry / mixed / unknown` using email domain matching + keyword patterns.

2. **LOO-CV evaluation**: trains TF-IDF + Logistic Regression on title + abstract + affiliation text to predict sector (academia=0, industry=1), using leave-one-out cross-validation on the ~208 unambiguously classified papers. Saves AUC, accuracy, F1 to `results/classify_loocv.json`.

```bash
python src/classify.py --reclassify   # force re-run heuristic step
```

### Step 4 — Paper-level ethics scoring

```bash
python src/ethics_score.py
```

For each paper, computes:
- `has_ethics_term` (1/0) — does any paragraph contain a clean ethics keyword?
- `ethics_paragraph_count` — number of matching paragraphs
- `ethics_paragraphs` — pipe-separated matched paragraph texts (truncated)

**Clean keyword set** (intentionally excludes `bias` and `safety`):
`fairness`, `ethical`, `ethics`, `harm`, `transparency`, `accountability`, `justice`, `equity`

`bias` and `safety` are excluded because they fire too many false positives in cs.AI text (bias-variance tradeoff; vehicle/robotic safety constraints).

Writes `data/processed/ethics_scores.csv`.

### Step 5 — LLM-based coding

```bash
export ANTHROPIC_API_KEY=your_key_here
python src/llm_code.py
```

For each paper with `has_ethics_term=1`, sends ethics-adjacent paragraphs to `claude-sonnet-4-20250514` in two stages:

**Stage 1 — Binary filter:** "Does this paper engage with AI ethics substantively (genuine normative concern) or only incidentally/technically (e.g. bias-variance tradeoff, physical safety)?" → score 0 or 1 + justification.

**Stage 2 (substantive papers only) — Thematic coding:**
- **A — Technical Ethics:** fairness metrics, robustness, differential privacy, adversarial testing
- **B — Governance/Compliance:** regulation, oversight, audit frameworks, policy, accountability mechanisms
- **C — Structural/Justice:** harm to communities, systemic discrimination, power asymmetries, structural inequality, equity

Output is appended incrementally to `results/llm_codes.csv` after each paper, so the run is resumable.

### Step 6 — BERTopic validation

```bash
python src/bertopic_run.py
```

Runs BERTopic on the ethics-adjacent paragraphs using `sentence-transformers/all-MiniLM-L6-v2` embeddings + HDBSCAN + c-TF-IDF. Outputs:
- `results/bertopic_topics.csv` — topic labels and top terms
- `results/bertopic_doc_topics.csv` — per-paragraph topic assignments (merged with LLM codes for triangulation)
- `results/bertopic_model/` — serialized model

The goal is to check whether BERTopic clusters align with LLM-assigned A/B/C themes — this triangulation strengthens validity.

---

## Key design decisions

**Why LOO-CV?** The heuristic labels are used as the primary affiliation signal. LOO-CV evaluates how much affiliation information is recoverable from title-page text alone, providing a proxy for how reliable the labels are. It is an evaluation of the heuristic, not an alternative classifier.

**Why exclude `bias` and `safety`?** In cs.AI papers, "bias" most often means the constant term in a model or the bias-variance tradeoff; "safety" in robotics and control systems means constraint satisfaction. Including these terms inflates the ethics keyword rate artificially. The LLM coding step handles polysemy correctly because it reads context, not just keywords.

**Why BERTopic + LLM triangulation?** The LLM coding is judgement-based and provides clean A/B/C categories. BERTopic is fully unsupervised and provides an independent grouping. Alignment between the two (measured as topic-by-theme crosstabs) gives a validity check that neither method provides alone.

---

## Preliminary findings (pilot, N=400)

- Overall ethics keyword rate (clean set): ~3.6% of paragraphs
- Industry papers: ~0.8% — roughly 1/4 the rate of academia (~3.0%)
- Mixed (academia + industry) collaborations: ~5.5% — highest rate
- 2022 anomaly: ~11.8% ethics keyword rate vs. 1.5–6.4% in other years
- `bias` was the most frequent term in the original set (0.99%) but is excluded in the clean set for the reasons above

---

## Data files (not tracked in git)

- `data/pdfs/` — ~399 PDFs, ~several GB
- `data/processed/` — generated by pipeline steps 2–4
- `results/` — generated by pipeline steps 3–6
