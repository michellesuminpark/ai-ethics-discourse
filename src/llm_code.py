"""
llm_code.py — LLM-based binary and thematic coding of ethics-adjacent papers.

For each paper with has_ethics_term=1, sends its ethics paragraphs to
Claude via the Anthropic API in two stages:

  Stage 1 (all hit papers):
    Binary filter — substantive vs incidental ethics engagement.
    Returns 0 (incidental) or 1 (substantive) + one-sentence justification.

  Stage 2 (substantive papers only):
    Thematic coding into one of three categories:
      A — Technical Ethics
      B — Governance/Compliance
      C — Structural/Justice

Saves raw responses + parsed codes to results/llm_codes.csv.
Requires ANTHROPIC_API_KEY environment variable.
"""

import anthropic
import pandas as pd
import os
import time
import re
import json

ETHICS_CSV  = "data/processed/ethics_scores.csv"
PARA_CSV    = "data/processed/paragraphs.csv"
OUT_CSV     = "results/llm_codes.csv"
MODEL       = "claude-sonnet-4-20250514"

# Rate limiting: stay under Anthropic tier limits
REQUEST_DELAY = 1.0   # seconds between requests
MAX_RETRIES   = 5
RETRY_BASE    = 10    # seconds; exponential backoff

# Context budget: how many chars of ethics paragraphs to send per paper
MAX_CONTEXT_CHARS = 6000

os.makedirs("results", exist_ok=True)


STAGE1_SYSTEM = (
    "You are a research assistant helping classify AI research papers "
    "for a study of AI ethics discourse. Be precise and consistent."
)

STAGE1_USER = """\
Below are excerpts from an AI research paper. Determine whether the paper \
engages with AI ethics SUBSTANTIVELY (genuine normative concern about \
fairness, harm, accountability, justice, or governance) or only \
INCIDENTALLY/TECHNICALLY (e.g. bias-variance tradeoff, physical safety \
constraints, mentioning ethics only in passing).

Respond with ONLY this format on two lines:
SCORE: <0 or 1>
REASON: <one sentence>

Where 0 = incidental/technical, 1 = substantive ethics engagement.

Paper excerpts:
{excerpts}
"""

STAGE2_SYSTEM = STAGE1_SYSTEM

STAGE2_USER = """\
This paper engages substantively with AI ethics. Assign it to exactly ONE \
of the following three categories based on how it frames AI ethics:

  A — Technical Ethics: fairness metrics, robustness, differential privacy, \
adversarial testing, bias mitigation techniques
  B — Governance/Compliance: regulation, oversight, audit frameworks, policy, \
accountability mechanisms, standards
  C — Structural/Justice: harm to communities, systemic discrimination, power \
asymmetries, structural inequality, equity, voice, participation

Respond with ONLY this format on two lines:
CATEGORY: <A, B, or C>
REASON: <one sentence>

Paper excerpts:
{excerpts}
"""


def _call_api(client: anthropic.Anthropic, system: str, user: str,
              paper_id: str, stage: int) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = RETRY_BASE * (2 ** (attempt - 1))
            print(f"    [rate limit] {paper_id} stage{stage} — sleeping {wait}s", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == MAX_RETRIES:
                print(f"    [API error] {paper_id} stage{stage}: {e}", flush=True)
                return ""
            time.sleep(RETRY_BASE * attempt)
    return ""


def _parse_stage1(response: str) -> tuple[int | None, str]:
    score_m  = re.search(r"SCORE:\s*([01])", response, re.IGNORECASE)
    reason_m = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
    score  = int(score_m.group(1)) if score_m else None
    reason = reason_m.group(1).strip() if reason_m else ""
    return score, reason


def _parse_stage2(response: str) -> tuple[str | None, str]:
    cat_m    = re.search(r"CATEGORY:\s*([ABC])", response, re.IGNORECASE)
    reason_m = re.search(r"REASON:\s*(.+)", response, re.IGNORECASE)
    category = cat_m.group(1).upper() if cat_m else None
    reason   = reason_m.group(1).strip() if reason_m else ""
    return category, reason


def build_excerpts(ethics_paragraphs_str: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Select and format ethics paragraphs up to max_chars."""
    if not isinstance(ethics_paragraphs_str, str) or not ethics_paragraphs_str.strip():
        return ""
    paras = [p.strip() for p in ethics_paragraphs_str.split("|||") if p.strip()]
    lines = []
    total = 0
    for i, p in enumerate(paras, 1):
        chunk = f"[{i}] {p}"
        if total + len(chunk) > max_chars:
            break
        lines.append(chunk)
        total += len(chunk)
    return "\n\n".join(lines)


def load_checkpoint(out_csv: str) -> set[str]:
    if os.path.exists(out_csv):
        done = pd.read_csv(out_csv, dtype={"paper_id": str})
        return set(done["paper_id"].astype(str))
    return set()


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    scores = pd.read_csv(ETHICS_CSV, dtype={"paper_id": str})
    hit    = scores[scores["has_ethics_term"] == 1].copy()
    print(f"Papers with ethics hit: {len(hit)} / {len(scores)}")

    done_ids = load_checkpoint(OUT_CSV)
    if done_ids:
        print(f"Resuming — {len(done_ids)} papers already coded")

    rows: list[dict] = []

    for i, (_, paper) in enumerate(hit.iterrows(), 1):
        pid = str(paper["paper_id"])

        if pid in done_ids:
            continue

        excerpts = build_excerpts(str(paper.get("ethics_paragraphs", "")))
        if not excerpts:
            print(f"  [{i}/{len(hit)}] {pid}: no excerpts — skipping", flush=True)
            rows.append({
                "paper_id": pid, "year": paper["year"], "sector": paper["sector"],
                "ethics_paragraph_count": paper["ethics_paragraph_count"],
                "stage1_response": "", "substantive": None, "stage1_reason": "",
                "stage2_response": "", "theme": None, "stage2_reason": "",
            })
            continue

        print(f"  [{i}/{len(hit)}] {pid}: stage 1...", end=" ", flush=True)

        # Stage 1
        s1_prompt  = STAGE1_USER.format(excerpts=excerpts)
        s1_raw     = _call_api(client, STAGE1_SYSTEM, s1_prompt, pid, 1)
        substantive, s1_reason = _parse_stage1(s1_raw)
        time.sleep(REQUEST_DELAY)

        # Stage 2 (only if stage 1 = 1)
        s2_raw = ""
        theme  = None
        s2_reason = ""

        if substantive == 1:
            print("stage 2...", end=" ", flush=True)
            s2_prompt = STAGE2_USER.format(excerpts=excerpts)
            s2_raw    = _call_api(client, STAGE2_SYSTEM, s2_prompt, pid, 2)
            theme, s2_reason = _parse_stage2(s2_raw)
            time.sleep(REQUEST_DELAY)

        label = {1: "substantive", 0: "incidental", None: "parse_error"}.get(substantive, "parse_error")
        print(f"→ {label} {('| theme:' + str(theme)) if theme else ''}", flush=True)

        rows.append({
            "paper_id":               pid,
            "year":                   paper["year"],
            "sector":                 paper["sector"],
            "ethics_paragraph_count": paper["ethics_paragraph_count"],
            "stage1_response":        s1_raw,
            "substantive":            substantive,
            "stage1_reason":          s1_reason,
            "stage2_response":        s2_raw,
            "theme":                  theme,
            "stage2_reason":          s2_reason,
        })

        # Append to checkpoint after each paper
        pd.DataFrame([rows[-1]]).to_csv(
            OUT_CSV,
            mode="a",
            header=not os.path.exists(OUT_CSV),
            index=False,
        )
        done_ids.add(pid)

    final = pd.read_csv(OUT_CSV, dtype={"paper_id": str}) if os.path.exists(OUT_CSV) else pd.DataFrame(rows)
    n_sub   = (final["substantive"] == 1).sum()
    n_inc   = (final["substantive"] == 0).sum()
    n_err   = final["substantive"].isna().sum()
    print(f"\n{'='*50}")
    print(f"LLM coding complete: {len(final)} papers")
    print(f"  Substantive : {n_sub}")
    print(f"  Incidental  : {n_inc}")
    print(f"  Parse error : {n_err}")
    print(f"\nTheme distribution (substantive papers):")
    print(final[final["substantive"]==1]["theme"].value_counts().to_string())
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
