"""
Step 5: Create Initial Codebook (EECS paper, Appendix prompts 2 & 3)

For each cluster (largest/most substantive first):
  1. Generate a candidate code (label + definition + example) from the
     cluster's ideas.
  2. For the first N clusters (config.BOOTSTRAP_CLUSTERS), just accept the
     candidate directly -- there aren't enough codes yet for RAG to be useful.
  3. For subsequent clusters, retrieve the top-k most similar existing codes
     and ask the model whether the candidate is redundant with them. Only add
     it if the model decides it's genuinely new.

This mirrors the paper's two-prompt approach: a "summarize the cluster" prompt
and a separate "check against existing codebook" prompt.
"""

import json
import os
import re

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config

SUMMARIZE_CLUSTER_PROMPT_TEMPLATE = """Act as if you are {persona}. You specialize in creating short labels to summarize the main idea being discussed in a collection of text. The labels you generate will be used to systematically label all the {data_type}s to identify key themes in an entire dataset of text. I need your help analyzing a cluster of {data_type}s from {data_collection_context}. In theory, the cluster of {data_type}s should be semantically similar and related to each other, but occasionally it may not be homogenous. Your task is to generate a small number of labels that capture the main concepts or themes discussed in the majority of the {data_type}s you review. To do this, please start by reviewing the {data_type}s provided to you in the <text> tag. Each of the {data_type}s is separated by a new line. After reading the {data_type}s in the cluster, please then generate your summary labels. For each label you generate, please provide:
- The label (a short phrase)
- A brief definition of what the label represents
- An example of a {data_type} from the actual text in the <text> tag

The goal is to capture the essence and meaning of the *majority* of comments in each cluster of {data_type}s through the labels applied. Your response should start "Label:" followed by your label and then on a new line "Definition:" followed by your definition for that label.

For example:
Label: <label 1>
Definition: <definition 1>
Example: <example 1>

As a reminder, here is the cluster of {data_type}s for you to analyze:
<text>
{cluster_text}
</text>"""

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


def parse_candidate_code(raw_output: str):
    label_match = re.search(r"label:\s*(.+)", raw_output, re.IGNORECASE)
    definition_match = re.search(r"definition:\s*(.+?)(?=\n\s*(example:|label:|$))", raw_output, re.IGNORECASE | re.DOTALL)
    example_match = re.search(r"example:\s*(.+)", raw_output, re.IGNORECASE)

    label = label_match.group(1).strip().strip('"') if label_match else None
    definition = definition_match.group(1).strip().strip('"') if definition_match else ""
    example = example_match.group(1).strip().strip('"') if example_match else ""
    return label, definition, example


def parse_verdict(raw_output: str) -> bool:
    """Returns True if the model's verdict was to keep/accept the new code."""
    match = re.search(r"final verdict:?\s*(.+)", raw_output, re.IGNORECASE)
    verdict_line = match.group(1).lower() if match else raw_output.lower()
    return "keep" in verdict_line and "reject" not in verdict_line.split("keep")[0][-20:]


def main():
    clusters_path = os.path.join(config.OUTPUT_DIR, "2_clusters.json")
    if not os.path.exists(clusters_path):
        print(f"ERROR: {clusters_path} not found. Run step2_embed_cluster.py first.")
        return

    with open(clusters_path) as f:
        clusters = json.load(f)

    print(f"Loading embedding model for RAG lookups...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    codebook = []  # list of dicts: {label, definition, example, embedding, source_cluster_id}
    log = []

    for i, cluster in enumerate(clusters):
        # Per the paper (Table 2's worked example shows a single-line input,
        # e.g. "The instructor used powerpoint presentations effectively"):
        # only the cluster's single representative idea is passed to the
        # model, not the full list of member ideas.
        cluster_text = cluster["representative_idea"]
        prompt = SUMMARIZE_CLUSTER_PROMPT_TEMPLATE.format(
            persona=config.PERSONA,
            data_type=config.DATA_TYPE,
            data_collection_context=config.DATA_COLLECTION_CONTEXT,
            cluster_text=cluster_text,
        )
        raw = call_llm(prompt)
        label, definition, example = parse_candidate_code(raw)

        if not label:
            log.append({"cluster_id": cluster["cluster_id"], "status": "parse_failed", "raw": raw})
            print(f"[{i+1}/{len(clusters)}] Cluster {cluster['cluster_id']}: FAILED to parse candidate label")
            continue

        if i < config.BOOTSTRAP_CLUSTERS or len(codebook) == 0:
            # Bootstrap phase: not enough codes yet for meaningful RAG comparison
            decision = "accepted (bootstrap)"
            accept = True
        else:
            candidate_embedding = embed_model.encode([f"{label}: {definition}"], normalize_embeddings=True)
            existing_embeddings = np.array([c["embedding"] for c in codebook])
            sims = cosine_similarity(candidate_embedding, existing_embeddings)[0]
            top_k_idx = np.argsort(-sims)[:config.TOP_K_CODES]
            top_k_codes = [codebook[j] for j in top_k_idx]

            existing_codes_str = "\n".join(f"- {c['label']}: {c['definition']}" for c in top_k_codes)
            edit_prompt = EDITING_CODEBOOK_PROMPT_TEMPLATE.format(
                existing_codes=existing_codes_str,
                candidate_label=label,
                candidate_definition=definition,
            )
            edit_raw = call_llm(edit_prompt)
            accept = parse_verdict(edit_raw)
            decision = "accepted (passed RAG check)" if accept else "rejected (redundant per RAG check)"

        if accept:
            emb = embed_model.encode([f"{label}: {definition}"], normalize_embeddings=True)[0]
            codebook.append({
                "label": label,
                "definition": definition,
                "example": example,
                "embedding": emb.tolist(),
                "source_cluster_id": cluster["cluster_id"],
                "cluster_size": cluster["size"],
            })

        log.append({"cluster_id": cluster["cluster_id"], "candidate_label": label, "decision": decision})
        print(f"[{i+1}/{len(clusters)}] Cluster {cluster['cluster_id']} (n={cluster['size']}): '{label}' -> {decision}")

    out_path = os.path.join(config.OUTPUT_DIR, "3_initial_codebook.json")
    # Don't persist raw embeddings in the human-readable output
    codebook_clean = [{k: v for k, v in c.items() if k != "embedding"} for c in codebook]
    with open(out_path, "w") as f:
        json.dump(codebook_clean, f, indent=2)

    log_path = os.path.join(config.OUTPUT_DIR, "3_codebook_generation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Generated {len(codebook)} codes from {len(clusters)} clusters.")
    print(f"Codebook: {out_path}")
    print(f"Full decision log: {log_path}")


if __name__ == "__main__":
    main()
