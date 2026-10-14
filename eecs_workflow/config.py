"""
Central configuration for the EECS pipeline.
Edit the values below to match your dataset and setup. Nothing here is sent
anywhere except to your local Ollama server (127.0.0.1).
"""

# ---- Input data ----
INPUT_CSV = "free_text_genai_use.csv"     # path to your CSV file
TEXT_COLUMN = "genai_other_tasks"      # column containing the free-text responses
ID_COLUMN = None                   # optional: column with a unique response ID.
                                    # Leave as None to auto-generate row indices.

# ---- Prompt context (inserted into prompts, per the paper's Appendix) ----
# These describe your data source in plain language for the LLM.
DATA_TYPE = "survey response"                     # e.g. "student evaluation", "survey response"
DATA_COLLECTION_CONTEXT = "a course survey asking students whether they used generative AI tools for any other tasks in the course, beyond ones already asked about, and to describe what they used it for"
PERSONA = "an expert qualitative research assistant"        # used in the cluster-summarizing prompt

# ---- Local model settings ----
OLLAMA_MODEL = "dolphin-mistral"   # must match a model you've pulled via `ollama pull`
MODEL_TEMPERATURE = 0.0            # 0 = deterministic, matches paper's reliability approach

# ---- Embedding model ----
# A modern, strong open-source embedding model (successor lineage to the
# UAE-Angle model used in the paper), runs fine on CPU for this data size.
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# ---- Clustering settings (HDBSCAN) ----
# Paper used "conservative" settings to encourage MANY SMALL clusters.
#   - Too few, huge clusters -> decrease MIN_CLUSTER_SIZE
#   - Too many singleton "noise" points -> increase MIN_CLUSTER_SIZE slightly
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1

# ---- Codebook generation (RAG) settings ----
BOOTSTRAP_CLUSTERS = 15   # first N clusters generate codes directly (no RAG yet,
                          # since the codebook doesn't have enough entries to
                          # compare against — matches paper's approach)
TOP_K_CODES = 15          # how many similar existing codes to show the model
                          # when deciding whether a new code is needed

# ---- Output ----
OUTPUT_DIR = "outputs"

# ---- Theme creation settings (optional final step) ----
# Groups your final codes into a smaller number of broader themes, similar to
# the GATOS paper's theme creation step. Adjust N_THEMES up/down and re-run to
# get coarser or finer groupings -- there's no single "correct" number, it
# depends on how you want to report results.
N_THEMES = 12
