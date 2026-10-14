"""
Step 2-4: Generate Text Embeddings, Cluster Text Embeddings, Find Representatives
(EECS paper, Steps 2, 3, 4)

Embeds each extracted idea, clusters semantically similar ideas together using
HDBSCAN (which can label points as noise rather than force bad-fit clusters),
then finds one representative idea per cluster to pass along to the codebook
generation step.

Runs entirely locally via sentence-transformers -- no data leaves your machine.
"""

import json
import os

import numpy as np
import pandas as pd
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


def main():
    in_path = os.path.join(config.OUTPUT_DIR, "1_extracted_ideas.csv")
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run step1_extract.py first.")
        return

    df = pd.read_csv(in_path)
    ideas = df["idea_text"].astype(str).tolist()
    print(f"Loaded {len(ideas)} extracted ideas.")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL}' (first run downloads it once)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Embedding ideas...")
    embeddings = model.encode(ideas, show_progress_bar=True, normalize_embeddings=True)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.MIN_CLUSTER_SIZE,
        min_samples=config.MIN_SAMPLES,
        metric="euclidean",  # embeddings are normalized, so euclidean ~ cosine ranking
    )
    labels = clusterer.fit_predict(embeddings)
    df["cluster_id"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"Found {n_clusters} clusters ({n_noise} ideas labeled as noise / unclustered).")

    # --- Find a representative idea for each cluster ---
    # Per the paper: concatenate all ideas in a cluster, embed that string, then
    # find which original idea in the cluster is most similar to it.
    clusters_output = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue  # skip noise points -- they don't form a cluster to summarize
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]
        member_ideas = [ideas[i] for i in member_indices]

        concatenated = " ".join(member_ideas)
        concat_embedding = model.encode([concatenated], normalize_embeddings=True)

        # Per the paper's Step 4 ("Finding Representatives for Each Cluster"):
        # compare the cluster's concatenated embedding against ALL extracted
        # ideas in the entire dataset (not just this cluster's own members).
        # The single closest match anywhere becomes the representative -- even
        # if, in principle, that idea happens to belong to a different cluster
        # or was labeled as noise. In practice this is usually one of the
        # cluster's own members anyway, but we don't restrict the search.
        sims = cosine_similarity(concat_embedding, embeddings)[0]
        best_global_idx = int(np.argmax(sims))
        representative_idea = ideas[best_global_idx]
        representative_is_own_member = best_global_idx in set(member_indices.tolist())

        clusters_output.append({
            "cluster_id": int(cluster_id),
            "size": len(member_ideas),
            "member_ideas": member_ideas,
            "representative_idea": representative_idea,
            "representative_is_own_member": representative_is_own_member,
        })

    n_external_reps = sum(1 for c in clusters_output if not c["representative_is_own_member"])
    if n_external_reps > 0:
        print(f"\nNote: {n_external_reps} of {len(clusters_output)} clusters had a representative")
        print("idea that came from OUTSIDE the cluster itself (i.e., a different cluster or a")
        print("noise point) -- this matches the paper's stated method (Step 4 searches the")
        print("entire dataset, not just the cluster), but is worth knowing about.")

    # Sort clusters largest-first -- reasonable default processing order for
    # step 3 (bootstrap on the most substantive clusters first)
    clusters_output.sort(key=lambda c: -c["size"])

    out_path = os.path.join(config.OUTPUT_DIR, "2_clusters.json")
    with open(out_path, "w") as f:
        json.dump(clusters_output, f, indent=2)

    # Also save the embeddings so step 3 doesn't have to re-embed anything
    np.save(os.path.join(config.OUTPUT_DIR, "2_idea_embeddings.npy"), embeddings)
    df.to_csv(os.path.join(config.OUTPUT_DIR, "2_ideas_with_clusters.csv"), index=False)

    print(f"\nDone. Wrote {len(clusters_output)} clusters to {out_path}")
    print("Representative ideas and full cluster membership are saved for inspection.")


if __name__ == "__main__":
    main()
