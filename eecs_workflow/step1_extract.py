"""
Step 1: Extracting Information (EECS paper, Step 1 / Appendix prompt 1)

Reads raw free-text responses and asks a local LLM (via Ollama) to break each
one into a numbered list of discrete, standardized ideas. This disentangles
compound statements ("I liked the demos and the office hours") into separate
atomic ideas that can be compared/clustered later.

Everything here runs against your local Ollama server -- no data leaves your
machine.
"""

import csv
import json
import re
import sys
import os

import ollama
import pandas as pd

import config

CANONICAL_NO_USE_PHRASE = "No use of generative AI tools beyond those already asked about"

# Patterns that indicate an idea is really just a paraphrase of "no additional
# use to report," regardless of exact wording the model happened to produce.
# We stopped relying on the model to output an exact phrase verbatim (local
# quantized models don't reliably comply with that instruction even at
# temperature=0) and instead normalize its output ourselves, deterministically,
# after the fact. Add more patterns here as you spot new paraphrase variants.
NO_USE_PARAPHRASE_PATTERNS = [
    r"negative response",
    r"no additional information",
    r"did not use",
    r"didn'?t use",
    r"\bnot used\b",
    r"no use of",
    r"no other (task|use|tool)",
    r"no additional (task|use|tool)",
    r"non-?use of",
    r"reported no use",
    r"no further (task|use|information)",
    r"nothing (else|additional|further) to (add|report|mention)",
    r"has not used",
    r"have not used",
    r"never used (generative )?ai",
    r"student did not",
    r"respondent did not",
]
NO_USE_REGEX = re.compile("|".join(NO_USE_PARAPHRASE_PATTERNS), re.IGNORECASE)

EXTRACTION_PROMPT_TEMPLATE = """You are an expert text analyst reading {data_type}s collected in {data_collection_context}. I am going to send you one of these {data_type}s. I need you to use your expertise to analyze the provided text in the <text> tag below and summarize it in an enumerated list. You should do this analysis by providing several short descriptive phrases that summarize each idea discussed in the {data_type} that answered the prompt. When you suggest multiple items, separate each one in your response with a new line. You MUST remove anyone's names and *use gender neutral pronouns* for deidentification purposes. Start your response with "My summary:". Here is an example of input and desired output for this specific task, showing both how to split multiple distinct ideas AND how to handle a vague mention that doesn't specify a task.

Example input: "I used it to help debug my code for the final project, and I also just used it a couple other random times for other stuff."
Example output: "My summary:
1. Used a generative AI tool to help debug code for the final project
2. Used a generative AI tool without specifying the task"

Notice three things in this example: (1) the first idea captures a SPECIFIC task (debugging code) exactly as stated, without adding invented detail; (2) the second idea uses the EXACT required wording "Used a generative AI tool without specifying the task" because "other random times for other stuff" does not name any specific task; (3) no names or pronouns are included. You must NEVER make up information that is not in the input text you receive because there is a severe penalty for that. If the text you receive is very short and says "nothing," do not make up new things.

Here is a second example showing two more patterns to handle correctly:

Example input: "All the tasks described above are basically what I have done using Generative AI, it makes me more perfect and get high performance."
Example output: "My summary:
1. No additional generative AI tasks beyond those already asked about"

Notice two things in this second example: (1) "all the tasks described above" means the respondent is referring back to answers given elsewhere (e.g., to earlier survey questions) rather than naming any new task here, so it should be summarized as reporting no additional task, using wording like "No additional generative AI tasks beyond those already described"; (2) the vague self-praise ("makes me more perfect and get high performance") is a general opinion/sentiment, NOT a specific task, so it is correctly left OUT of the summary entirely -- do not extract sentiment, opinions, or vague evaluative statements as if they were tasks.

IMPORTANT: Any time part of a response mentions using a generative AI tool but does NOT specify what particular task, activity, or purpose it was used for, phrase that idea using EXACTLY this wording: "Used a generative AI tool without specifying the task." Do not invent a specific task that is not stated in the text. Using this exact consistent phrasing (rather than paraphrasing it differently each time) helps later steps group these mentions together correctly.

There are THREE different situations where the respondent is reporting they have NOTHING NEW to add, beyond tasks already covered elsewhere in the survey, and ALL THREE must use the EXACT SAME canonical phrase below, so they can be correctly grouped together later. Do NOT vary this wording between these situations:
  (a) A short negative response with no other content, such as "No," "None," "N/A," "Nothing," "Nope," or similar.
  (b) A response that refers back to previously-described tasks rather than naming anything new (e.g., "same as above," "all the tasks already mentioned").
  (c) Any other phrasing that amounts to "I did not use it for anything else [beyond what was already asked about]."

For ALL THREE of these situations, output exactly one item using this EXACT phrase and nothing else:
"No additional generative AI tasks beyond those already asked about"

IMPORTANT -- this is a DIFFERENT and SEPARATE situation, do not confuse it with the above:
  (d) A response that denies using generative AI AT ALL in the course, not just "nothing more to add" (e.g., "No, I haven't used generative AI in this course," "I never used AI for this course"). This is a broader claim than (a)-(c) above -- it says the respondent never used it for anything, not just that they have nothing additional to report.

For situation (d), output exactly one item using this EXACT phrase and nothing else:
"Reported no use of generative AI in the course at all"

Only extract statements that describe an actual task or reason for use -- do not extract pure opinions, sentiment, or vague self-evaluation (e.g., "it's great," "makes me more efficient") as if they were tasks.

Here is the text for you to summarize: <text>{text}</text>

Begin your analysis now."""


def parse_summary(raw_output: str):
    """Extract the numbered list items following 'My summary:'."""
    # Grab everything after "My summary:" (case-insensitive, tolerant of quotes)
    match = re.search(r"my summary:?(.*)", raw_output, re.IGNORECASE | re.DOTALL)
    body = match.group(1) if match else raw_output

    # Match lines like "1. idea text" or "1) idea text"
    items = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", body, re.MULTILINE)
    items = [i.strip().strip('"').strip() for i in items if i.strip()]

    # Normalize any "no use" paraphrase to the exact canonical phrase,
    # regardless of exactly how the model worded it.
    items = [CANONICAL_NO_USE_PHRASE if NO_USE_REGEX.search(i) else i for i in items]

    return items


def extract_ideas_for_response(text: str):
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        data_type=config.DATA_TYPE,
        data_collection_context=config.DATA_COLLECTION_CONTEXT,
        text=text,
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": config.MODEL_TEMPERATURE},
    )
    raw_output = response["message"]["content"]
    return parse_summary(raw_output), raw_output


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(config.INPUT_CSV):
        print(f"ERROR: could not find {config.INPUT_CSV}. Edit config.py or place your CSV here.")
        sys.exit(1)

    df = pd.read_csv(config.INPUT_CSV)
    if config.TEXT_COLUMN not in df.columns:
        print(f"ERROR: column '{config.TEXT_COLUMN}' not found. Columns available: {list(df.columns)}")
        sys.exit(1)

    if config.ID_COLUMN and config.ID_COLUMN in df.columns:
        ids = df[config.ID_COLUMN].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    texts = df[config.TEXT_COLUMN].astype(str).tolist()

    all_rows = []
    errors = []

    print(f"Extracting ideas from {len(texts)} responses using model '{config.OLLAMA_MODEL}'...")
    for i, (response_id, text) in enumerate(zip(ids, texts)):
        if not text.strip() or text.strip().lower() in {"nan", "none", "n/a", "na"}:
            continue
        try:
            ideas, raw_output = extract_ideas_for_response(text)
            if not ideas:
                errors.append({"response_id": response_id, "text": text, "raw_output": raw_output})
                continue
            for idea in ideas:
                all_rows.append({"response_id": response_id, "idea_text": idea})
        except Exception as e:
            errors.append({"response_id": response_id, "text": text, "error": str(e)})

        if (i + 1) % 25 == 0 or (i + 1) == len(texts):
            print(f"  ...{i + 1}/{len(texts)} responses processed, {len(all_rows)} ideas extracted so far")

    out_path = os.path.join(config.OUTPUT_DIR, "1_extracted_ideas.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nDone. Wrote {len(all_rows)} extracted ideas to {out_path}")

    if errors:
        err_path = os.path.join(config.OUTPUT_DIR, "1_extract_errors.log")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"WARNING: {len(errors)} responses failed to parse cleanly. See {err_path}")


if __name__ == "__main__":
    main()
