"""
Step 6: Simplifying the Codebook (EECS paper's Appendix prompts)

Two passes over the initial codebook:
  1. Deduplication pass -- for each code, retrieve the top-k most similar
     codes already accepted into the "survivors" list, and use the paper's
     actual "Prompt for Editing the Codebook" to ask the model whether the
     existing codes already sufficiently cover this one. No arbitrary
     similarity threshold gates whether a code gets checked -- every code
     goes through this check, matching the paper's description.
  2. Quality pass -- for each surviving code, ask the model whether it's
     specific/clear enough to keep as-is, or whether it should be split /
     reworded (per the paper's "Prompt for Simplifying the Codebook").

The final output is meant to be reviewed by you -- per the paper, human
oversight of the generated codebook is an explicit, expected part of this
method, not an optional add-on.
"""

import json
import os
import re

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config

SIMPLIFY_CODE_PROMPT_TEMPLATE = """Act as if you are the world's best social science researcher. You specialize in qualitative data analysis and thematic coding. I need your help. I have a list of codes in a codebook to label themes in {data_type}s collected from {data_collection_context}. These codes need to be useful and informative for the research team, meaning they need to be specific and clear. Your task is to read a code from the codebook that I will give you in the <text> tag and tell me whether or not the code is a good code that I should keep in the codebook or a bad code that I should discard. Format your response as "My response:" followed by your step-by-step reasoning about whether it is a good code. Then, on a new line, say "My suggestion:" followed by your suggestion for whether or not to keep the code. If you think we should discard the code then you should add yet another line that says "Alternate suggestion:" followed by a better version of the code you reviewed.

Here is the code for you to review: <text>{label}: {definition}</text>"""

# This is the paper's actual "Prompt for Editing the Codebook" (same prompt
# reused from step3_codebook.py's RAG check), applied here as a second pass
# over the FULL generated codebook -- matching the paper's description that
# simplification "asked the model to check each code [for] a similar one
# already in the codebook."
EDITING_CODEBOOK_PROMPT_TEMPLATE = """Act as if you are the world's best qualitative data analyst with expertise in generating qualitative codebooks for thematic analysis. Your task is to do the following four steps. Step 1: examine an existing codebook given to you in the <existing_codebook> tag. Step 2: read a new code given to you in the <text> tag. Step 3: determine whether the existing codebook already has enough labels to describe the possible new code in the <text> tag. Step 4: Either accept the new code because it is sufficiently different from other codes or reject the new code because it is redundant. This is admittedly subjective, but you should be able to use your expert judgment to decide whether or not existing codes together provide sufficient coverage at an abstract level. Either way, be sure to provide your reasoning for whether or not you think you need to accept or reject the possible new code. Please note that there is a penalty for adding redundant codes, so you should only create a new code if you are certain the existing ones are insufficient. You should start your response with "My analysis:" followed by your step-by-step reasoning. You should conclude your response with "My final verdict:" followed by "Keep code <code>" or "Reject code <code>" where <code> is a placeholder for the actual label you are judging.

For your reference, here is the existing codebook:
<existing_codebook>
{existing_codes}
</existing_codebook>

Here is the new code to evaluate:
<text>{candidate_label}: {candidate_definition}</text>"""


def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": config.MODEL_TEMPERATURE},
    )
    return response["message"]["content"]


def parse_verdict(raw_output: str) -> bool:
    """Returns True if the model's verdict was to keep/accept the code."""
    match = re.search(r"final verdict:?\s*(.+)", raw_output, re.IGNORECASE)
    verdict_line = match.group(1).lower() if match else raw_output.lower()
    return "keep" in verdict_line and "reject" not in verdict_line.split("keep")[0][-20:]


def main():
    codebook_path = os.path.join(config.OUTPUT_DIR, "3_initial_codebook.json")
    if not os.path.exists(codebook_path):
        print(f"ERROR: {codebook_path} not found. Run step3_codebook.py first.")
        return

    with open(codebook_path) as f:
        codebook = json.load(f)

    print(f"Loaded {len(codebook)} codes. Loading embedding model...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [f"{c['label']}: {c['definition']}" for c in codebook]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    # --- Pass 1: Deduplication via top-k RAG check (matches paper's method) ---
    # NOTE: the paper doesn't specify a processing order for this pass. We
    # process largest-cluster (most-supported-by-data) codes first, so that
    # when two codes ARE found redundant, the one with more underlying data
    # is more likely to already be in the "survivors" list acting as the
    # existing codebook, and the newer/smaller one gets rejected as covered.
    # This is our own reasonable choice, not something the paper specifies.
    print("\n--- Deduplication pass (top-k RAG check, no code excluded from checking) ---")

    order = sorted(range(len(codebook)), key=lambda i: -codebook[i].get("cluster_size", 0))

    survivor_indices = []  # indices into `codebook`, in the order they were accepted
    dedup_log = []

    for idx in order:
        candidate = codebook[idx]

        if not survivor_indices:
            # First code processed: nothing to compare against yet, matches
            # step 3's own bootstrap logic for the very first entries.
            survivor_indices.append(idx)
            dedup_log.append({"code": candidate["label"], "decision": "accepted (first code, no comparison possible)"})
            print(f"  '{candidate['label']}': accepted (first code)")
            continue

        survivor_embeddings = embeddings[survivor_indices]
        candidate_embedding = embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(candidate_embedding, survivor_embeddings)[0]

        k = min(config.TOP_K_CODES, len(survivor_indices))
        top_k_local = np.argsort(-sims)[:k]
        top_k_survivor_indices = [survivor_indices[j] for j in top_k_local]
        top_k_codes = [codebook[j] for j in top_k_survivor_indices]

        existing_codes_str = "\n".join(f"- {c['label']}: {c['definition']}" for c in top_k_codes)
        prompt = EDITING_CODEBOOK_PROMPT_TEMPLATE.format(
            existing_codes=existing_codes_str,
            candidate_label=candidate["label"],
            candidate_definition=candidate["definition"],
        )
        raw = call_llm(prompt)
        keep = parse_verdict(raw)

        if keep:
            survivor_indices.append(idx)
            decision = "accepted (distinct from top-k existing codes)"
        else:
            decision = f"rejected (redundant with one of: {[c['label'] for c in top_k_codes]})"

        dedup_log.append({"code": candidate["label"], "decision": decision, "raw_response": raw})
        print(f"  '{candidate['label']}': {decision}")

    survivors = [codebook[i] for i in survivor_indices]
    print(f"\nAfter dedup: {len(survivors)} codes remain (dropped {len(codebook) - len(survivors)}).")

    # --- Pass 2: Quality/clarity check ---
    print("\n--- Quality check pass ---")
    final_codebook = []
    quality_log = []
    for c in survivors:
        prompt = SIMPLIFY_CODE_PROMPT_TEMPLATE.format(
            data_type=config.DATA_TYPE,
            data_collection_context=config.DATA_COLLECTION_CONTEXT,
            label=c["label"], definition=c["definition"],
        )
        raw = call_llm(prompt)

        suggestion_match = re.search(r"my suggestion:?\s*(.+)", raw, re.IGNORECASE)
        alt_match = re.search(r"alternate suggestion:?\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        suggestion = suggestion_match.group(1).strip() if suggestion_match else ""
        alternate = alt_match.group(1).strip() if alt_match else None

        keep = "keep" in suggestion.lower()
        c_out = dict(c)
        c_out["quality_suggestion"] = suggestion
        c_out["alternate_suggestion"] = alternate
        c_out.pop("embedding", None)

        if keep:
            final_codebook.append(c_out)
        else:
            # Still include it, flagged, so nothing silently disappears --
            # you review and decide (per the paper, human review is expected)
            c_out["flagged_for_review"] = True
            final_codebook.append(c_out)

        quality_log.append({"label": c["label"], "raw_response": raw})
        print(f"  '{c['label']}': {'KEEP' if keep else 'FLAGGED - review suggested'}")

    out_path = os.path.join(config.OUTPUT_DIR, "4_final_codebook.json")
    with open(out_path, "w") as f:
        json.dump(final_codebook, f, indent=2)

    with open(os.path.join(config.OUTPUT_DIR, "4_dedup_log.json"), "w") as f:
        json.dump(dedup_log, f, indent=2)
    with open(os.path.join(config.OUTPUT_DIR, "4_quality_log.json"), "w") as f:
        json.dump(quality_log, f, indent=2)

    n_flagged = sum(1 for c in final_codebook if c.get("flagged_for_review"))
    print(f"\nDone. Final codebook: {len(final_codebook)} codes ({n_flagged} flagged for your review).")
    print(f"Output: {out_path}")
    print("\nNext step: open this file and review any 'flagged_for_review' entries")
    print("plus their 'alternate_suggestion' -- this is the human-in-the-loop step")
    print("the paper explicitly recommends, not an optional extra.")


if __name__ == "__main__":
    main()
