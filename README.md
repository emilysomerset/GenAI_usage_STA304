# Generative AI Use in Undergraduate Statistics Assignments

## Overview

This repository contains the R and Python code used to support the analysis of student-reported generative artificial intelligence (GenAI) use in STA304: Surveys, Sampling and Observational Data, an undergraduate statistics course at the University of Toronto.

The analysis draws on student survey responses collected across three course offerings: Fall 2023, Winter 2025, and Fall 2025.

The repository supports two components of the study:

1.  **Descriptive analysis:** Summarizing students' reported GenAI use across assignments, course offerings, and specific academic tasks.
2.  **Qualitative analysis:** Developing a codebook to characterize additional uses of GenAI described in students' open-ended survey responses.

## Data Availability

The underlying student survey data are stored in a restricted-access Zenodo repository:

<https://doi.org/10.5281/zenodo.23376068>

The data are not publicly available due to privacy and ethical restrictions. Access requests can be submitted through Zenodo.

The analysis scripts are provided to document the methods used in the study. Reproducing the analyses requires access to the restricted data.

## Descriptive Analysis

The descriptive analysis was conducted in R using packages including `dplyr`, `tidyr`, `ggplot2`, and `kableExtra` and R version 4.5.2 (2025-10-31) running under macOS Sequoia 15.7.4.

The scripts generate summaries of:

-   GenAI use on Assignment 1, Assignment 2, both assignments, or neither assignment, by course offering.
-   Reported use of GenAI for specific tasks, including generating and editing code and written text.
-   Tables and figures used to describe patterns of GenAI use across course offerings.

## Qualitative Analysis

Open-ended survey responses describing additional uses of GenAI were analyzed using a locally implemented computational workflow adapted from the **Extract, Embed, Cluster, Summarize (EECS)** framework described by Katz, Gerhardt, and Soledad (2024), *Using Generative Text Models to Create Qualitative Codebooks for Student Evaluations of Teaching*.

The computational workflow was used to support the development of an initial qualitative codebook. The workflow consists of the following stages:

1.  **Idea extraction (`step1_extract.py`):** A locally hosted language model identifies discrete ideas within individual open-ended responses.
2.  **Embedding and clustering (`step2_embed_cluster.py`):** Extracted ideas are converted into numerical embeddings using a sentence-transformer model and grouped according to semantic similarity.
3.  **Initial codebook generation (`step3_codebook.py`):** A locally hosted language model generates candidate code labels and definitions using representative ideas from the clusters.
4.  **Codebook simplification (`step4_simplify.py`):** The initial codebook is refined computationally to reduce overlapping or redundant categories.

Additional scripts support codebook review, organization, and consolidation.

### Human Review

The computationally generated codebook served as a starting point for the qualitative analysis rather than as a final, independently validated coding scheme.

The researcher reviewed and refined the coding results to develop the categories of GenAI use reported in the manuscript.

The purpose of this analysis was to identify and describe different ways students reported using GenAI, rather than to estimate the prevalence of individual qualitative categories.

## Local Processing and Data Privacy

The qualitative analysis was designed to run locally to protect the confidentiality of student responses.

-   **Ollama** was used to run language models locally.
-   **Sentence-transformers** was used to generate text embeddings locally.
-   **HDBSCAN** was used for clustering.
-   Intermediate outputs were stored locally.

The workflow did not require transmitting student responses to externally hosted language-model APIs.

## Repository Structure

| File | Description |
|------------------------------------|------------------------------------|
| `config.py` | Configuration settings for the qualitative analysis |
| `step1_extract.py` | Extracts discrete ideas from open-ended responses |
| `step2_embed_cluster.py` | Generates embeddings and clusters similar ideas |
| `step3_codebook.py` | Generates an initial qualitative codebook |
| `step4_simplify.py` | Simplifies and consolidates the initial codebook |
| `step5_themes.py` | Supports the organization of codes into themes |
| `build_full_review_csv.py` | Prepares coding results for human review |
| `merge_codes.py` | Merges coding results |
| `run_all.py` | Runs the automated workflow |
| `requirements.txt` | Python package dependencies |

The R analysis script is also provided for reproducing the descriptive summaries and figures.

The original student responses and intermediate outputs containing student text are excluded from the public repository.

## Software Requirements

The descriptive analysis requires R and the packages specified in the accompanying R script.

The qualitative workflow requires Python and the dependencies listed in `requirements.txt`, together with a local installation of Ollama.

To install the Python dependencies:

``` bash
pip install -r requirements.txt
```

The language model must also be downloaded locally using Ollama. Model selection and other analysis settings are specified in `config.py`.

## Running the Qualitative Workflow

After obtaining authorized access to the underlying data:

1.  Configure the input file path, response column, language model, and analysis settings in `config.py`.
2.  Run the extraction, embedding, clustering, and codebook generation scripts sequentially:

``` bash
python step1_extract.py
python step2_embed_cluster.py
python step3_codebook.py
python step4_simplify.py
```

Alternatively, the automated stages can be executed using:

``` bash
python run_all.py
```

Additional scripts support subsequent review and refinement of the generated coding results.

The computational workflow does not replace the human review and interpretation described in the manuscript.

## Reproducibility and Limitations

The code is provided to support methodological transparency and reproducibility, subject to the data access restrictions.

The resulting codebook may depend on the language model, embedding model, clustering parameters, and other configuration settings. Outputs may also vary across model versions and runs.

The qualitative analysis involved researcher judgment during codebook review and refinement. Consequently, executing the automated scripts alone may not reproduce the final qualitative categories presented in the manuscript.

## References

Katz, Gerhardt, and Soledad (2024). *Using Generative Text Models to Create Qualitative Codebooks for Student Evaluations of Teaching*.

Full bibliographic details are provided in the accompanying manuscript.
