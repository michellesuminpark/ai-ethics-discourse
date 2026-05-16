"""
ethics_score.py — Two-stage LLM ethics coding for all papers.

Reads data/processed/ethics_sections.csv (from parse.py) and for each paper:

  Stage 1 (all papers): Binary substantiveness classification.
    1 = substantive ethical engagement (genuine normative concern: fairness,
        harm, accountability, justice, governance).
    0 = incidental/technical only (bias-variance, physical safety,
        performance metrics labeled "fair").

  Stage 2 (substantive papers only): Thematic categorization.
    A — Technical Ethics: fairness metrics, robustness, differential privacy,
         adversarial testing, safety constraints.
    B — Governance/Compliance: regulation, oversight, audit frameworks, policy,
         accountability mechanisms, institutional compliance.
    C — Structural/Justice: harm to communities, systemic discrimination,
         power asymmetries, structural inequality, equity, justice.

Output columns:
  paper_id, year, extraction_method,
  ethics_substantive (0/1/None), ethics_justification,
  input_tokens_s1, output_tokens_s1,
  ethics_theme (A/B/C/None), theme_justification,
  input_tokens_s2, output_tokens_s2

Results are checkpointed after each paper for resumability.
Token usage is logged per call and summed at the end.

Requires ANTHROPIC_API_KEY (loaded from .env).
"""

import anthropic
import pandas as pd
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

ETHICS_IN     = "data/processed/ethics_sections.csv"
OUT_CSV       = "results/ethics_codes.csv"
MODEL         = "claude-sonnet-4-6"
REQUEST_DELAY = 0.5
MAX_RETRIES   = 5
RETRY_BASE    = 10
MAX_CHARS     = 6000

os.makedirs("results", exist_ok=True)


SYSTEM_PROMPT = (
    "You are a research assistant helping classify AI research papers for a study "
    "of AI ethics discourse. Be precise and consistent."
)

STAGE1_USER = """\
Read the following sections from an AI research paper. Does this paper engage with AI ethics \
substantively — meaning genuine normative concern about fairness, harm, accountability, \
justice, or governance — or only incidentally or technically (e.g. bias-variance tradeoff, \
physical safety constraints, performance metrics labeled as "fair")?

Answer 1 for substantive ethical engagement or 0 for incidental/technical only.
Provide a one-sentence justification.

Respond with ONLY this format on two lines:
SCORE: <0 or 1>
REASON: <one sentence>

Paper sections:
{text}
"""

STAGE2_USER = """\
Assign this paper to exactly one of the following three categories based on how it primarily \
frames AI ethics:

  A — Technical Ethics: fairness metrics, robustness certification, differential privacy, \
adversarial testing, safety constraints.
  B — Governance/Compliance: regulation, oversight, audit frameworks, policy, accountability \
mechanisms, institutional compliance.
  C — Structural/Justice: harm to communities, systemic discrimination, power asymmetries, \
structural inequality, equity, justice.

Return the letter (A, B, or C) and a one-sentence justification.

Respond with ONLY this format on two lines:
CATEGORY: <A, B, or C>
REASON: <one sentence>

Paper sections:
{text}
"""


def _call_api(
    client: anthropic.Anthropic,
    user: str,
    paper_id: str,
    stage: int,
) -> tuple[str, int, int]:
    """Return (response_text, input_tokens, output_tokens)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text.strip()
            return text, msg.usage.input_tokens, msg.usage.output_tokens
        except anthropic.RateLimitError:
            wait = RETRY_BASE * (2 ** (attempt - 1))
            print(f"    [rate limit] {paper_id} s{stage} — sleeping {wait}s", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt == MAX_RETRIES:
                print(f"    [API error] {paper_id} s{stage}: {e}", flush=True)
                return "", 0, 0
            time.sleep(RETRY_BASE * attempt)
    return "", 0, 0


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


def _empty_record(pid: str, year, method: str) -> dict:
    return {
        "paper_id":            pid,
        "year":                year,
        "extraction_method":   method,
        "stage1_response":     "",
        "ethics_substantive":  None,
        "ethics_justification": "",
        "input_tokens_s1":     0,
        "output_tokens_s1":    0,
        "stage2_response":     "",
        "ethics_theme":        None,
        "theme_justification": "",
        "input_tokens_s2":     0,
        "output_tokens_s2":    0,
    }


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(ETHICS_IN, dtype={"paper_id": str})
    print(f"Loaded {len(df)} papers from {ETHICS_IN}")

    done_ids: set[str] = set()
    if os.path.exists(OUT_CSV):
        done_ids = set(pd.read_csv(OUT_CSV, dtype={"paper_id": str})["paper_id"].astype(str))
        print(f"Resuming — {len(done_ids)} papers already coded")

    total_in  = 0
    total_out = 0

    for i, row in df.iterrows():
        pid    = str(row["paper_id"])
        year   = row["year"]
        method = str(row.get("extraction_method", ""))
        text   = str(row.get("ethics_text", ""))[:MAX_CHARS]

        if pid in done_ids:
            continue

        if not text.strip():
            print(f"  [{i+1}/{len(df)}] {pid}: no text — skipping", flush=True)
            record = _empty_record(pid, year, method)
            pd.DataFrame([record]).to_csv(
                OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False,
            )
            done_ids.add(pid)
            continue

        print(f"  [{i+1}/{len(df)}] {pid}: s1...", end=" ", flush=True)

        s1_raw, s1_in, s1_out = _call_api(client, STAGE1_USER.format(text=text), pid, 1)
        substantive, justification = _parse_stage1(s1_raw)
        total_in  += s1_in
        total_out += s1_out
        time.sleep(REQUEST_DELAY)

        s2_raw = ""
        theme  = None
        theme_just = ""
        s2_in  = s2_out = 0

        if substantive == 1:
            print("s2...", end=" ", flush=True)
            s2_raw, s2_in, s2_out = _call_api(client, STAGE2_USER.format(text=text), pid, 2)
            theme, theme_just = _parse_stage2(s2_raw)
            total_in  += s2_in
            total_out += s2_out
            time.sleep(REQUEST_DELAY)

        label = {1: "substantive", 0: "incidental", None: "parse_error"}.get(
            substantive, "parse_error"
        )
        print(f"→ {label}" + (f" | theme:{theme}" if theme else ""), flush=True)

        record = {
            "paper_id":            pid,
            "year":                year,
            "extraction_method":   method,
            "stage1_response":     s1_raw,
            "ethics_substantive":  substantive,
            "ethics_justification": justification,
            "input_tokens_s1":     s1_in,
            "output_tokens_s1":    s1_out,
            "stage2_response":     s2_raw,
            "ethics_theme":        theme,
            "theme_justification": theme_just,
            "input_tokens_s2":     s2_in,
            "output_tokens_s2":    s2_out,
        }
        pd.DataFrame([record]).to_csv(
            OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False,
        )
        done_ids.add(pid)

    final = (
        pd.read_csv(OUT_CSV, dtype={"paper_id": str})
        if os.path.exists(OUT_CSV)
        else pd.DataFrame()
    )

    n_sub = int((final["ethics_substantive"] == 1).sum())
    n_inc = int((final["ethics_substantive"] == 0).sum())
    n_err = int(final["ethics_substantive"].isna().sum())

    print(f"\n{'='*55}")
    print(f"Ethics coding complete: {len(final)} papers")
    print(f"  Substantive  : {n_sub}")
    print(f"  Incidental   : {n_inc}")
    print(f"  Parse error  : {n_err}")
    if n_sub > 0:
        print(f"\nTheme distribution (substantive papers):")
        print(final[final["ethics_substantive"] == 1]["ethics_theme"].value_counts().to_string())
    print(f"\nToken usage this session:")
    print(f"  Input tokens : {total_in:,}")
    print(f"  Output tokens: {total_out:,}")
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
