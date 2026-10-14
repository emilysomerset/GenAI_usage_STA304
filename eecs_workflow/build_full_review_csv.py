"""
Builds one comprehensive CSV with one row per extracted idea, including:
  - response_id
  - the ORIGINAL full raw response text
  - the extracted idea text
  - the final code label (from 5_merged_codebook.json), if this idea's
    cluster made it all the way through to a surviving code
  - that code's definition

Includes EVERY extracted idea, even ones whose cluster was labeled noise or
was rejected/dropped at some stage of codebook generation/simplification --
for those, code_label and code_definition are left blank, since no final
code was assigned. This is expected, not a bug: those ideas still need to be
manually assigned a code by you.
"""

import json
import os

import pandas as pd

import config


def build_cluster_to_final_code_map(pre_merge_codebook, merged_codebook):
    """Returns dict: cluster_id -> (final_label, final_definition)."""
    # Map label -> list of pre-merge entries with that exact label (there can
    # be more than one if multiple clusters independently produced identical labels)
    label_to_entries = {}
    for entry in pre_merge_codebook:
        label_to_entries.setdefault(entry["label"], []).append(entry)

    cluster_to_final = {}

    for code in merged_codebook:
        if "merged_from_labels" in code:
            claimed = set()
            for label in code["merged_from_labels"]:
                for entry in label_to_entries.get(label, []):
                    cid = entry.get("source_cluster_id")
                    if cid is not None and cid not in claimed:
                        claimed.add(cid)
                        cluster_to_final[cid] = (code["label"], code["definition"])
                        break
        else:
            cid = code.get("source_cluster_id")
            if cid is not None:
                cluster_to_final[cid] = (code["label"], code["definition"])

    return cluster_to_final


def main():
    ideas_with_clusters_path = os.path.join(config.OUTPUT_DIR, "2_ideas_with_clusters.csv")
    pre_merge_path = os.path.join(config.OUTPUT_DIR, "4_final_codebook.json")
    merged_path = os.path.join(config.OUTPUT_DIR, "5_merged_codebook.json")

    for p in (ideas_with_clusters_path, pre_merge_path, merged_path, config.INPUT_CSV):
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            return

    # --- Load original raw responses, keyed by response_id (same logic as step1_extract.py) ---
    raw_df = pd.read_csv(config.INPUT_CSV)
    if config.ID_COLUMN and config.ID_COLUMN in raw_df.columns:
        raw_ids = raw_df[config.ID_COLUMN].astype(str).tolist()
    else:
        raw_ids = [str(i) for i in range(len(raw_df))]
    raw_texts = raw_df[config.TEXT_COLUMN].astype(str).tolist()
    response_id_to_text = dict(zip(raw_ids, raw_texts))

    # --- Load all extracted ideas with their cluster assignment (includes noise, cluster_id=-1) ---
    ideas_df = pd.read_csv(ideas_with_clusters_path)
    ideas_df["response_id"] = ideas_df["response_id"].astype(str)

    # --- Build cluster_id -> final code mapping ---
    with open(pre_merge_path) as f:
        pre_merge_codebook = json.load(f)
    with open(merged_path) as f:
        merged_codebook = json.load(f)

    cluster_to_final = build_cluster_to_final_code_map(pre_merge_codebook, merged_codebook)

    # --- Assemble the output rows ---
    rows = []
    for _, row in ideas_df.iterrows():
        response_id = row["response_id"]
        idea_text = row["idea_text"]
        cluster_id = row["cluster_id"]

        label, definition = cluster_to_final.get(cluster_id, (None, None))

        rows.append({
            "response_id": response_id,
            "original_response": response_id_to_text.get(response_id, ""),
            "idea_extracted": idea_text,
            "code_label": label if label is not None else "",
            "code_definition": definition if definition is not None else "",
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(config.OUTPUT_DIR, "6_full_coding_review.csv")
    out_df.to_csv(out_path, index=False)

    n_total = len(out_df)
    n_coded = (out_df["code_label"] != "").sum()
    n_uncoded = n_total - n_coded

    print(f"Wrote {n_total} rows to {out_path}")
    print(f"  {n_coded} ideas matched to a final code")
    print(f"  {n_uncoded} ideas have NO code yet (noise, or rejected/dropped during")
    print(f"  codebook generation/simplification) -- these still need manual coding")


if __name__ == "__main__":
    main()
