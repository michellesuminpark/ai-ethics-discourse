# AI Ethics Discourse in arXiv cs.AI Papers (2018–2025)

Longitudinal study of how AI ethics is framed in academic AI research. Stratified random sample of ~50 papers/year from arXiv cs.AI (primary category), 2018–2025.

**Central question:** does where you work shape how you talk about AI ethics?

The hypothesis (drawing on Bourdieusian field theory) is that ethics framing in AI research is predominantly technical/managerial (metrics, benchmarks, compliance) rather than substantive (justice, power, harm to communities), and that this is more pronounced in industry-affiliated papers.

---

## Corpus

- **Source:** arXiv cs.AI (primary category), 2018–2025
- **Sample:** 50 papers/year × 8 years = 400 papers, stratified random sample (seed 42)
- **PDFs:** `data/pdfs/` — 399 on disk (not tracked in git)
- **Missing:** `1911.01156` (AAAI FSS-19 proceedings — not downloadable from arXiv)
- **Metadata:** `data/raw/arxiv_pilot.csv`

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> BERTopic pulls in `torch` transitively via `sentence-transformers`. On a GPU machine, install PyTorch with CUDA support first.

### 2. Configure API key

Create `.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Scripts load this automatically via `python-dotenv`.

### 3. Link PDFs

```bash
# Symlink your existing PDF directory:
ln -s /path/to/existing/pdfs data/pdfs
```

---

## Pipeline

Run scripts in order from the repo root.

### Step 1 — Collect (skip if pilot data already present)

```bash
python src/collect.py
```

Fetches metadata from the arXiv API and downloads PDFs for 50 cs.AI papers per year. Uses per-year checkpoints in `data/raw/checkpoints/` — safe to interrupt and resume. Writes `data/raw/arxiv_pilot.csv`.

### Step 2 — Parse

```bash
python src/parse.py
```

Extracts two separate text sets from each PDF:

**Affiliation text** — first 2 pages + any acknowledgments section (captures "work done while at X" notes throughout):
→ `data/processed/affiliation_text.csv`

**Ethics section text** — searches for section headers (introduction, discussion, conclusion, ethics, broader impact, limitations, societal impact) and extracts their body text. Falls back to first 3 + last 3 body paragraphs when no target headers are found:
→ `data/processed/ethics_sections.csv`

### Step 3 — Classify affiliations

```bash
python src/classify.py
```

Requires `ANTHROPIC_API_KEY`. Checkpoints after each paper.

1. **LLM open-ended classification** — passes affiliation text to `claude-sonnet-4-20250514` and asks for free-form institution type labels per author (e.g. "research university", "tech company", "government lab")
   → `results/author_affiliations_raw.csv`

2. **Paper-level aggregation** — counts and proportions of each institution type per paper
   → `results/author_affiliations_paper_level.csv`

3. **Post-hoc collapsing** — institution types appearing in < 15 papers collapsed to "other"
   → `results/author_affiliations_collapsed.csv`
   *(proportion columns = primary independent variable for all downstream analysis)*

4. **LOO-CV validation** — TF-IDF + Logistic Regression trained on title + abstract + affiliation text, evaluated with leave-one-out CV on the top-2 institution types. Checks coherence of LLM labels from text features alone.
   → `results/loo_cv_results.json`

### Step 4 — Ethics scoring

```bash
python src/ethics_score.py
```

Requires `ANTHROPIC_API_KEY`. Checkpoints after each paper. Runs on all 399 papers.

**Stage 1 — Binary substantiveness** (all papers):
Does the paper engage with AI ethics substantively (genuine normative concern: fairness, harm, accountability, justice, governance) or only incidentally/technically (bias-variance, physical safety, performance metrics labeled "fair")?
→ `ethics_substantive`: 0 or 1 + one-sentence justification

**Stage 2 — Thematic coding** (substantive papers only):
- **A — Technical Ethics:** fairness metrics, robustness certification, differential privacy, adversarial testing, safety constraints
- **B — Governance/Compliance:** regulation, oversight, audit frameworks, policy, accountability mechanisms, institutional compliance
- **C — Structural/Justice:** harm to communities, systemic discrimination, power asymmetries, structural inequality, equity, justice

Token usage is logged per call and reported at completion.
→ `results/ethics_codes.csv`

### Step 5 — BERTopic validation

```bash
python src/bertopic_run.py
```

Runs BERTopic (`sentence-transformers/all-MiniLM-L6-v2` + HDBSCAN + c-TF-IDF) on the ethics section text of substantive papers. Reports:

- Topic labels and top 10 terms per topic
- Topic distribution by year
- Mean institution type proportions by topic
- Agreement rate and Adjusted Mutual Information vs. LLM A/B/C themes

Outputs:
- `results/bertopic_topics.csv` — topic summary
- `results/bertopic_output.csv` — per-paper topic + LLM theme + affiliation proportions
- `results/bertopic_model/` — serialized model (not tracked in git)

---

## Output files

| File | Description |
|------|-------------|
| `data/raw/arxiv_pilot.csv` | Paper metadata (400 rows) |
| `data/processed/affiliation_text.csv` | Per-paper affiliation text from PDFs |
| `data/processed/ethics_sections.csv` | Per-paper ethics section text |
| `results/author_affiliations_raw.csv` | Per-author LLM institution type labels |
| `results/author_affiliations_paper_level.csv` | Paper-level proportions (uncollapsed) |
| `results/author_affiliations_collapsed.csv` | Paper-level proportions (collapsed) |
| `results/loo_cv_results.json` | LOO-CV metrics for affiliation coherence |
| `results/ethics_codes.csv` | LLM ethics substantiveness + A/B/C theme per paper |
| `results/bertopic_topics.csv` | BERTopic topic labels and top terms |
| `results/bertopic_output.csv` | Per-paper topic + theme + affiliation data |

---

## Repo structure

```
ai-ethics-discourse/
├── data/
│   ├── pdfs/                         # downloaded PDFs (not tracked in git)
│   ├── raw/
│   │   ├── arxiv_pilot.csv           # paper metadata
│   │   └── checkpoints/              # per-year collection checkpoints
│   └── processed/                    # generated by parse.py (not tracked)
├── src/
│   ├── collect.py                    # arXiv API collection + PDF download
│   ├── parse.py                      # affiliation + ethics section extraction
│   ├── classify.py                   # LLM affiliation classification + LOO-CV
│   ├── ethics_score.py               # LLM binary + thematic ethics coding
│   └── bertopic_run.py               # BERTopic topic modeling + validation
├── notebooks/                        # exploratory analysis
├── results/                          # generated outputs
├── requirements.txt
└── README.md
```

---

## Design notes

**Why LLM for affiliation classification?** Regex heuristics struggle with the diversity of institution names across languages and naming conventions. Free-form LLM labels capture nuance (e.g. distinguishing industry research labs from product teams) without requiring a predefined taxonomy. LOO-CV then checks whether the labels are recoverable from text alone, validating their consistency.

**Why section-based extraction (not abstract only)?** Ethics framing appears in discussion and conclusion sections, not abstracts. Abstracts describe contributions; discussions situate them normatively.

**Why BERTopic + LLM triangulation?** The LLM coding is judgment-based and provides clean A/B/C categories. BERTopic is fully unsupervised. Alignment between the two (measured as topic-by-theme crosstab + AMI) gives a validity check neither method provides alone.
