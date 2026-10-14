"""
Manual code merging tool.

Edit the MERGE_GROUPS list below to specify which codes should be combined
into one. Each group becomes a single entry in the merged codebook, with:
  - the new_label / new_definition you specify
  - cluster_size = sum of the merged codes' cluster sizes (so you still know
    how much of the data this code represents)
  - all original labels + examples kept as a record, in case you want to
    trace back what got merged

Any code NOT mentioned in any group is carried through unchanged.

Run this AFTER step4_simplify.py. It reads outputs/4_final_codebook.json and
writes outputs/5_merged_codebook.json -- your true final deliverable.
"""

import json
import os

import config

# ---------------------------------------------------------------------------
# EDIT THIS SECTION to specify your merges.
# Labels must match exactly (copy-paste from your codebook file) what's in
# outputs/4_final_codebook.json.
# ---------------------------------------------------------------------------
MERGE_GROUPS = [
    {
        "labels_to_merge": [
            "Generative AI for Concept Clarification",
            "Generative AI for Concept Clarification",
            "Generative AI for Concept Clarification",
            "Generative AI for Explanation",
            "Generative AI for Explanation",
            "Generative AI for Explanation",
            "Generative AI for Explanation",
            "Generative AI for Explanation",
            "Generative AI for Explanation",
            "Learning Support Tool",
            "AI Assistance in Course Content Understanding",
            "Generative AI for Conceptual Understanding",
            "Generative AI for Concept Understanding",
            "Generative AI for Concept Explanation",
            "Generative AI for Lecture Understanding",
            "Generative AI for Formula Explanation",
            "Generative AI for Statistical Understanding",
            "Generative AI for Statistical Methods",
        ],
        "new_label": "Concept Understanding and Explanation",
        "new_definition": "Using generative AI tools to improve understanding of course material by explaining concepts, formulas, definitions, lecture content, statistical methods, or other difficult topics, often through simplified explanations or examples.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Studying",
            "Generative AI for Studying",
            "Generative AI for Exam Preparation",
            "Generative AI for Exam Preparation",
            "Generative AI for Study Guides",
            "Generative AI for Self-Study",
            "Generative AI for Self-Assessment",
            "Generative AI for Practice Questions",
            "Generative AI for Scheduling",
        ],
        "new_label": "Studying and Exam Preparation",
        "new_definition": "Using generative AI tools to support studying, exam preparation, self-assessment, or review activities, including generating practice questions, study guides, review schedules, or other materials to prepare for quizzes, tests, or exams.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Summarization",
            "Generative AI for Summarization",
            "Generative AI for Summarization",
            "Generative AI for Note Summarization",
            "Generative AI for Note Summarization and Organization",
            "Generative AI for Note-taking",
        ],
        "new_label": "Summarization and Note Support",
        "new_definition": "Using generative AI tools to summarize, organize, clarify, or generate notes from lectures, readings, slides, or other course materials to facilitate learning and review.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Debugging",
            "Generative AI for Debugging",
            "Generative AI for Troubleshooting",
            "Generative AI for coding assistance",
            "Generative AI for Code Editing and Debugging",
            "Generative AI for coding tasks",
            "Generative AI for Code Explanation",
            "Generative AI for Code Interpretation and Learning",
        ],
        "new_label": "Coding Assistance",
        "new_definition": "Using generative AI tools to write, edit, explain, interpret, troubleshoot, or debug computer code, or to support learning and understanding of programming concepts.",
    },
    {
        "labels_to_merge": [
            "Grammar and Word Correction",
            "Grammar and Clarity Checking",
            "Generative AI for Writing Improvement",
            "Generative AI for Sentence Structure Refinement",
            "Generative AI for Editing and Rephrasing Text",
            "Generative AI for Rephrasing",
            "Language Syntax Assistance",
            "Generative AI for Written Assignment Improvement",
        ],
        "new_label": "Writing Improvement",
        "new_definition": "Using generative AI tools to improve the quality of written work by editing grammar, spelling, clarity, organization, sentence structure, wording, readability, or overall writing style.",
    },
    {
        "labels_to_merge": [
            "Additional Tasks Not Specified",
            "Additional Tasks Not Reported",
            "Generative AI Non-Use",
        ],
        "new_label": "No Additional AI Use Reported",
        "new_definition": "Responses indicating that students did not use generative AI for any additional tasks beyond those already captured by the survey questions.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Research",
            "Generative AI for academic tasks",
        ],
        "new_label": "Research and Information Seeking",
        "new_definition": "Using generative AI tools to locate, retrieve, or synthesize information relevant to course content or assignments, including finding references, researching topics, or answering general academic questions.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Assignments",
            "Generative AI for Assignment Understanding",
            "Generative AI for Question Clarification",
        ],
        "new_label": "Assignment Support",
        "new_definition": "Using generative AI tools to better understand assignment requirements, clarify assignment questions, or assist with completing coursework beyond general studying or concept explanation.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Rubric Examination",
            "Generative AI for Rubric Editing",
            "Generative AI for Guidelines and Checklists",
        ],
        "new_label": "Rubric and Guideline Support",
        "new_definition": "Using generative AI tools to interpret, review, or generate assignment rubrics, grading criteria, checklists, templates, or other guidelines to ensure coursework met assignment expectations.",
    },
    {
        "labels_to_merge": [
            "Generative AI for Citation Purposes",
            "Generative AI for Citation Formatting",
        ],
        "new_label": "Citation Support",
        "new_definition": "Using generative AI tools to generate, verify, or format academic citations and references according to required citation styles.",
    },
]
# ---------------------------------------------------------------------------


def main():
    in_path = os.path.join(config.OUTPUT_DIR, "4_final_codebook.json")
    with open(in_path) as f:
        codebook = json.load(f)

    print(f"Loaded {len(codebook)} codes.")

    # Build lookup by label; note labels may not be unique (that's the problem
    # we're fixing), so we track by index instead and mark consumed ones.
    consumed = [False] * len(codebook)
    merged_output = []

    for group in MERGE_GROUPS:
        matched_indices = []
        remaining_labels_to_find = list(group["labels_to_merge"])

        for i, c in enumerate(codebook):
            if consumed[i]:
                continue
            if c["label"] in remaining_labels_to_find:
                matched_indices.append(i)
                remaining_labels_to_find.remove(c["label"])  # consume one instance

        if not matched_indices:
            print(f"WARNING: no codes matched for merge group '{group['new_label']}' -- check label spelling.")
            continue

        total_cluster_size = sum(codebook[i].get("cluster_size", 0) for i in matched_indices)
        all_examples = [codebook[i].get("example", "") for i in matched_indices if codebook[i].get("example")]
        original_labels = [codebook[i]["label"] for i in matched_indices]

        merged_output.append({
            "label": group["new_label"],
            "definition": group["new_definition"],
            "cluster_size": total_cluster_size,
            "example": all_examples[0] if all_examples else "",
            "all_examples_from_merge": all_examples,
            "merged_from_labels": original_labels,
            "n_codes_merged": len(matched_indices),
        })

        for i in matched_indices:
            consumed[i] = True

        print(f"Merged {len(matched_indices)} codes -> '{group['new_label']}' (total cluster_size={total_cluster_size})")

    # Carry through everything not touched by a merge
    n_untouched = 0
    for i, c in enumerate(codebook):
        if not consumed[i]:
            merged_output.append(c)
            n_untouched += 1

    out_path = os.path.join(config.OUTPUT_DIR, "5_merged_codebook.json")
    with open(out_path, "w") as f:
        json.dump(merged_output, f, indent=2)

    print(f"\n{n_untouched} codes carried through unchanged.")
    print(f"Final merged codebook: {len(merged_output)} total codes.")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
