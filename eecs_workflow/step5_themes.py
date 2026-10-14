"""
Theme creation (optional final step, inspired by the GATOS paper's theme
creation step).

Groups your final, merged codes into a smaller number of broader themes.
Unlike the deduplication step, this does NOT assume codes are redundant with
each other -- it's fine (expected, even) for a theme to contain several
distinct-but-related codes. The goal is a two-level structure:

    Theme (broad category)
      -> Code (specific, granular idea)
        -> supported by N responses

Run this AFTER merge_codes.py. Reads outputs/5_merged_codebook.json, writes
outputs/6_themes.json.
"""

import json
import os

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

import config

THEME_PROMPT_TEMPLATE = """Act as if you are an expert qualitative researcher. You are given a group of related codes from a qualitative codebook about {data_collection_context}. Each code has a label and a definition. Your task is to generate ONE short, clear theme name that captures what unites this group of codes, plus a one-sentence definition of the theme.

Here are the codes in this group:
{codes_list}

Respond in exactly this format:
Theme: <theme name>
Definition: <one sentence definition of the theme>"""


def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": config.MODEL_TEMPERATURE},
    )
    return response["message"]["content"]


def parse_theme(raw_output: str):
    import re
    theme_match = re.search(r"theme:\s*(.+)", raw_output, re.IGNORECASE)
    def_match = re.search(r"definition:\s*(.+)", raw_output, re.IGNORECASE | re.DOTALL)
    theme = theme_match.group(1).strip().strip('"') if theme_match else None
    definition = def_match.group(1).strip().strip('"') if def_match else ""
    return theme, definition


def main():
    in_path = os.path.join(config.OUTPUT_DIR, "5_merged_codebook.json")
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run merge_codes.py first.")
        return

    with open(in_path) as f:
        codebook = json.load(f)

    print(f"Loaded {len(codebook)} codes. Grouping into {config.N_THEMES} themes...")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [f"{c['label']}: {c['definition']}" for c in codebook]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    n_themes = min(config.N_THEMES, len(codebook))  # can't have more themes than codes
    clusterer = AgglomerativeClustering(n_clusters=n_themes, metric="cosine", linkage="average")
    labels = clusterer.fit_predict(embeddings)

    themes_output = []
    for theme_id in range(n_themes):
        member_indices = [i for i, l in enumerate(labels) if l == theme_id]
        member_codes = [codebook[i] for i in member_indices]

        codes_list_str = "\n".join(
            f"- {c['label']}: {c['definition']}" for c in member_codes
        )
        prompt = THEME_PROMPT_TEMPLATE.format(
            data_collection_context=config.DATA_COLLECTION_CONTEXT,
            codes_list=codes_list_str,
        )
        raw = call_llm(prompt)
        theme_name, theme_definition = parse_theme(raw)

        if not theme_name:
            theme_name = f"[UNPARSED - see raw_response] Theme {theme_id}"

        total_responses = sum(c.get("cluster_size", 0) for c in member_codes)

        themes_output.append({
            "theme": theme_name,
            "theme_definition": theme_definition,
            "total_responses_represented": total_responses,
            "n_codes": len(member_codes),
            "codes": [
                {"label": c["label"], "definition": c["definition"], "cluster_size": c.get("cluster_size", 0)}
                for c in member_codes
            ],
        })
        print(f"Theme {theme_id + 1}/{n_themes}: '{theme_name}' ({len(member_codes)} codes, {total_responses} responses)")

    # Sort themes by how much data they represent, most common first
    themes_output.sort(key=lambda t: -t["total_responses_represented"])

    out_path = os.path.join(config.OUTPUT_DIR, "6_themes.json")
    with open(out_path, "w") as f:
        json.dump(themes_output, f, indent=2)

    print(f"\nDone. {n_themes} themes written to {out_path}")
    print("\nTip: if these groupings don't feel right, just change N_THEMES in")
    print("config.py (try a smaller number for broader themes, larger for more")
    print("specific ones) and re-run -- this step is fast, no need to redo earlier steps.")


if __name__ == "__main__":
    main()
