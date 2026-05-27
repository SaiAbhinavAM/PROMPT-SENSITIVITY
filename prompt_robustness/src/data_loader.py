import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def load_gensens_dataset(path: str) -> pd.DataFrame:
    """Load a GenSens JSONL file into the evaluator's expected schema.

    This is the bridge (rectification R1) that closes the loop between Phase 1
    (GenSens paraphrase generation) and Phase 2 (PRI evaluation). Instead of the
    evaluator inventing its own d1/d2/d3 templates, it now consumes the actual
    SBERT-filtered paraphrase variants produced by GenSens.

    Each GenSens record contributes one evaluation sample:
        input_text       ← metadata.input_text   (the grounding source)
        reference_output  ← metadata.reference_output
        prompt_variants   ← [variant.full_prompt for each variant]
        topic_label       ← task

    The four GenSens tasks are heterogeneous (summarization / creative /
    dialogue / qa), so the grounding + reference are read from the canonical
    metadata keys ``input_text`` / ``reference_output`` (which every loader now
    emits). The legacy summarization aliases ``article`` / ``gold_summary`` are
    accepted as a fallback so older datasets still load unchanged.
    """
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            variants = [v for v in (rec.get("variants", []) or []) if v.get("full_prompt")]
            prompt_variants = [v["full_prompt"] for v in variants]
            if not prompt_variants:
                continue
            records.append({
                "input_text": meta.get("input_text", meta.get("article", rec.get("base_text", ""))),
                "reference_output": meta.get("reference_output", meta.get("gold_summary", "")),
                "topic_label": rec.get("task", "summarization"),
                "prompt_variants": prompt_variants,
                "strategies": [v.get("strategy", "") for v in variants],
                "instance_id": rec.get("instance_id", ""),
            })
    return pd.DataFrame(records)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset for the PRI benchmark.

    Routes by extension:
      * ``.jsonl`` → GenSens paraphrase dataset (uses real prompt variants).
      * ``.json``  → legacy list-of-dicts with input_text/reference_output.

    Legacy JSON format:
    [
        {"input_text": "...", "reference_output": "...", "topic_label": "..."},
        ...
    ]
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if data_path.suffix == ".jsonl":
        return load_gensens_dataset(path)

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)
