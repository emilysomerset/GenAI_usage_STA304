"""Runs all four EECS steps in sequence. Prefer running steps individually the
first time so you can sanity-check intermediate outputs before committing to
the (slower) LLM-heavy steps 3 and 4."""

import step1_extract
import step2_embed_cluster
import step3_codebook
import step4_simplify

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Extracting ideas")
    print("=" * 60)
    step1_extract.main()

    print("\n" + "=" * 60)
    print("STEP 2-4: Embedding, clustering, finding representatives")
    print("=" * 60)
    step2_embed_cluster.main()

    print("\n" + "=" * 60)
    print("STEP 5: Generating initial codebook")
    print("=" * 60)
    step3_codebook.main()

    print("\n" + "=" * 60)
    print("STEP 6: Simplifying codebook")
    print("=" * 60)
    step4_simplify.main()

    print("\nAll done. Final codebook is in outputs/4_final_codebook.json")
