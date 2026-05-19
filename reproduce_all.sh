#!/usr/bin/env bash
# reproduce_all.sh — Full pipeline for AI Ethics Discourse study.
#
# Usage:
#   bash reproduce_all.sh            # run everything, skipping steps with existing outputs
#   bash reproduce_all.sh --force    # rerun all steps (WARNING: re-calls LLM APIs)
#
# Requirements:
#   - conda environment activated (see environment.yml)
#   - .env file in repo root containing: ANTHROPIC_API_KEY=sk-ant-...
#   - PDFs in data/pdfs/ (download separately — see README Step 1)
#
# Estimated runtime:
#   Step 1 (collect)  : 2–4 h   (arXiv API rate limits)
#   Step 2 (parse)    : ~5 min
#   Step 3 (classify) : ~30 min  (~$2 Anthropic API)
#   Step 4 (ethics)   : ~20 min  (~$2 Anthropic API)
#   Step 5 (bertopic) : ~10 min
#   Step 6 (figures)  : ~2 min   (fig10 UMAP adds ~60 s)

set -euo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

echo "=================================================="
echo " AI Ethics Discourse — Reproduction Pipeline"
echo " Python: $(python --version 2>&1)"
echo " Date  : $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "=================================================="

# ── Preflight checks ──────────────────────────────────────────────────────────

if [ ! -f ".env" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "ERROR: ANTHROPIC_API_KEY not found."
    echo "  Create .env in the repo root:  echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env"
    exit 1
fi

for pkg in pandas numpy bertopic matplotlib anthropic; do
    python -c "import $pkg" 2>/dev/null || {
        echo "ERROR: Python package '$pkg' not found. Run: conda env create -f environment.yml"
        exit 1
    }
done

# ── Step 1: Corpus collection ─────────────────────────────────────────────────

if [ "$FORCE" = true ] || [ ! -f "data/raw/arxiv_pilot.csv" ]; then
    echo ""
    echo "[Step 1] Corpus collection (arXiv API + PDF download)..."
    echo "  This takes 2–4 hours due to arXiv rate limits."
    echo "  Safe to interrupt and resume — checkpoints in data/raw/checkpoints/"
    python src/collect.py
else
    echo "[Step 1] SKIP — data/raw/arxiv_pilot.csv exists"
fi

# ── Step 2: PDF parsing ───────────────────────────────────────────────────────

if [ "$FORCE" = true ] || \
   [ ! -f "data/processed/affiliation_text.csv" ] || \
   [ ! -f "data/processed/ethics_sections.csv" ]; then
    echo ""
    echo "[Step 2] PDF parsing..."
    python src/parse.py
else
    echo "[Step 2] SKIP — parsed CSVs exist"
fi

# ── Step 3: Affiliation classification (LLM) ─────────────────────────────────

if [ "$FORCE" = true ] || [ ! -f "results/author_affiliations_collapsed.csv" ]; then
    echo ""
    echo "[Step 3] Affiliation classification (~30 min, ~\$2 API cost)..."
    echo "  NOTE: LLM outputs are non-deterministic. Results will be"
    echo "  semantically equivalent but not bit-identical to the paper."
    python src/classify.py
else
    echo "[Step 3] SKIP — affiliation CSVs exist"
fi

# ── Step 4: Ethics scoring (LLM) ─────────────────────────────────────────────

if [ "$FORCE" = true ] || [ ! -f "results/ethics_codes.csv" ]; then
    echo ""
    echo "[Step 4] Ethics scoring (~20 min, ~\$2 API cost)..."
    echo "  NOTE: LLM outputs are non-deterministic."
    python src/ethics_score.py
else
    echo "[Step 4] SKIP — results/ethics_codes.csv exists"
fi

# ── Step 5: BERTopic ──────────────────────────────────────────────────────────

if [ "$FORCE" = true ] || \
   [ ! -f "results/bertopic_corpus_output.csv" ] || \
   [ ! -f "results/bertopic_ethics_output.csv" ]; then
    echo ""
    echo "[Step 5] BERTopic topic modeling (~10 min)..."
    python src/bertopic_run.py --mode both
else
    echo "[Step 5] SKIP — BERTopic output CSVs exist"
fi

# ── Step 6: Figure generation ─────────────────────────────────────────────────

echo ""
echo "[Step 6] Generating figures..."
python src/visualize.py

echo ""
echo "=================================================="
echo " Done. Figures saved to results/figures/"
echo "=================================================="
