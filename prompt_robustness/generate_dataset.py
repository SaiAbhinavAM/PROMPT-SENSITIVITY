"""
generate_dataset.py — Sample dataset generator for all 4 GenSens task families.

Reads source data directly from the local HuggingFace parquet cache (zero downloads).
Generates a self-contained dataset with pre-computed model responses, perplexity,
and branching factor so the full evaluation pipeline runs with NO GPU after this.

Schema per record:
  input_text       : source input for the task
  reference_output : gold/canonical output
  topic_label      : "summarization" | "code" | "creative_writing" | "dialogue"
  prompt_variants  : {d1, d2, d3} → prompt string (3 perturbation levels)
  model_responses  : {model_name: {d1, d2, d3} → response string}
  model_ppl        : {model_name: {d1, d2, d3} → perplexity float}
  model_bf         : {model_name: {d1, d2, d3} → branching_factor float}

Usage:
    cd prompt_robustness
    ./venv/bin/python3.9 generate_dataset.py

Output:
    data/dataset_summarization.json
    data/dataset_code.json
    data/dataset_creative_writing.json
    data/dataset_dialogue.json
    data/dataset_all_tasks.json   ← combined, used by evaluator
"""

import csv
import glob
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_PER_TASK = 5
MAX_NEW_TOKENS   = 80
DATA_DIR         = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# All seq2seq models already cached locally — no downloads
MODELS = [
    "google/flan-t5-base",
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
]

# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ── Prompt variant templates ───────────────────────────────────────────────────

TEMPLATES = {
    "summarization": {
        "d1": "Summarize the following text: {input}",
        "d2": "Condense the main ideas from this passage: {input}",
        "d3": "Produce a concise, neutral summary of the article below: {input}",
    },
    "code": {
        "d1": "Write a Python function for the following problem: {input}",
        "d2": "Implement a Python solution for this programming task: {input}",
        "d3": "Provide a working Python function that solves the problem described below: {input}",
    },
    "creative_writing": {
        "d1": "Write a short story based on this prompt: {input}",
        "d2": "Create a brief narrative inspired by: {input}",
        "d3": "Compose a short creative passage that explores the following idea: {input}",
    },
    "dialogue": {
        "d1": "Continue this conversation as a helpful assistant: {input}",
        "d2": "Respond naturally to the following dialogue: {input}",
        "d3": "Provide a helpful and coherent reply to the conversation below: {input}",
    },
}

def build_variants(input_text: str, task: str) -> Dict[str, str]:
    truncated = input_text[:600]
    return {lvl: tmpl.format(input=truncated) for lvl, tmpl in TEMPLATES[task].items()}

# ── Data loaders (read from local parquet cache) ──────────────────────────────

def _snap(dataset_id: str) -> Path:
    """Return the first snapshot directory for a cached HF dataset."""
    snaps = list((HF_CACHE / dataset_id / "snapshots").iterdir())
    return snaps[0]


def load_summarization(n: int) -> List[Dict]:
    print("[data] Loading CNN/DailyMail from local cache …")
    snap = _snap("datasets--cnn_dailymail")
    pq   = snap / "3.0.0" / "test-00000-of-00001.parquet"
    df   = pd.read_parquet(pq).head(n)
    return [
        {
            "input_text":      row["article"].strip(),
            "reference_output": row["highlights"].strip(),
            "topic_label":     "summarization",
        }
        for _, row in df.iterrows()
    ]


def load_code(n: int) -> List[Dict]:
    print("[data] Loading HumanEval from local cache …")
    snap = _snap("datasets--openai_humaneval")
    pq   = snap / "openai_humaneval" / "test-00000-of-00001.parquet"
    df   = pd.read_parquet(pq).head(n)
    return [
        {
            "input_text":      row["prompt"].strip(),
            "reference_output": row["canonical_solution"].strip(),
            "topic_label":     "code",
        }
        for _, row in df.iterrows()
    ]


def load_creative_writing(n: int) -> List[Dict]:
    print("[data] Loading WritingPrompts from local cache …")
    snap  = _snap("datasets--euclaise--writingprompts")
    files = sorted(glob.glob(str(snap / "data" / "train*.parquet")))
    df    = pd.read_parquet(files[0]).head(n * 3)   # oversample, filter empties
    samples = []
    for _, row in df.iterrows():
        p = str(row.get("prompt", "")).strip()
        s = str(row.get("story",  "")).strip()
        if not p or not s:
            continue
        samples.append({
            "input_text":      p[:600],
            "reference_output": s[:400],
            "topic_label":     "creative_writing",
        })
        if len(samples) >= n:
            break
    return samples


def load_dialogue(n: int) -> List[Dict]:
    """
    MultiWOZ is script-based (no parquet files in cache).
    Use realistic hand-crafted samples for the local smoke test.
    On H100, swap this for the full multi_woz_v22 dataset via HuggingFace datasets lib.
    """
    print("[data] Using hand-crafted dialogue samples (MultiWOZ script-based, no parquet) …")
    samples = [
        {
            "input_text": (
                "User: I need to find a cheap hotel in the north part of town.\n"
                "Assistant: I found a few options. Do you have a preference for the number of stars?\n"
                "User: I'd like at least 3 stars please."
            ),
            "reference_output": "I have a 3-star hotel called the Ashley Hotel in the north. Would you like me to book it?",
        },
        {
            "input_text": (
                "User: Can you help me book a train from Cambridge to London?\n"
                "Assistant: Sure! What day would you like to travel?\n"
                "User: I need to leave on Monday and arrive before 12:00."
            ),
            "reference_output": "There is a train departing at 09:01 and arriving at 10:51. Shall I book that for you?",
        },
        {
            "input_text": (
                "User: I'm looking for an Italian restaurant that is moderately priced.\n"
                "Assistant: I found several. Do you have a preference for area?\n"
                "User: The city centre please."
            ),
            "reference_output": "Prezzo is a moderately priced Italian restaurant in the city centre. Would you like their phone number?",
        },
        {
            "input_text": (
                "User: I need a taxi to get to the train station by 3pm.\n"
                "Assistant: Where will you be departing from?\n"
                "User: From the hotel I'm staying at, the Gonville Hotel."
            ),
            "reference_output": "I'll book a taxi from the Gonville Hotel to the train station arriving by 3pm. Can I get your contact number?",
        },
        {
            "input_text": (
                "User: Is there an attraction in Cambridge related to science?\n"
                "Assistant: Yes, the Whipple Museum of the History of Science is a great option. Is there anything specific you're interested in?\n"
                "User: I'd like the address and opening hours."
            ),
            "reference_output": "The Whipple Museum is located on Free School Lane. It is open Monday to Friday from 12:30 to 16:30 and is free to enter.",
        },
    ]
    return [
        {**s, "topic_label": "dialogue"}
        for s in samples[:n]
    ]

# ── Model wrapper ─────────────────────────────────────────────────────────────

class LocalModel:
    """Seq2seq model wrapper for generation, perplexity, and branching factor."""

    _TASK_PREFIX = {
        "summarization":   "summarize: ",
        "code":            "generate Python code: ",
        "creative_writing": "write: ",
        "dialogue":        "respond: ",
    }

    def __init__(self, model_name: str):
        self.name = model_name
        print(f"\n[model] Loading {model_name} on {DEVICE} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def _fmt(self, prompt: str, task: str) -> str:
        if "flan-t5" in self.name.lower():
            prefix = self._TASK_PREFIX.get(task, "")
            if not prompt.lower().startswith(prefix.strip().rstrip(":")):
                return prefix + prompt
        return prompt

    def generate(self, prompt: str, task: str) -> str:
        fmt    = self._fmt(prompt, task)
        inputs = self.tokenizer(
            fmt, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def perplexity(self, prompt: str, response: str) -> float:
        enc = self.tokenizer(prompt,   return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        dec = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        if enc["input_ids"].shape[1] == 0 or dec["input_ids"].shape[1] == 0:
            return 0.0
        with torch.no_grad():
            loss = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                labels=dec["input_ids"],
            ).loss.item()
        return round(math.exp(min(loss, 20.0)), 4)

    def branching_factor(self, response: str) -> float:
        inputs = self.tokenizer(
            response, return_tensors="pt", truncation=True, max_length=128
        ).to(DEVICE)
        if inputs["input_ids"].shape[1] == 0:
            return 0.0
        with torch.no_grad():
            logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs["input_ids"],
            ).logits.squeeze(0)
        probs   = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-12), dim=-1).mean().item()
        return round(max(0.0, min(1.0, entropy / 8.0)), 4)

    def unload(self):
        del self.model, self.tokenizer
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        print(f"[model] Unloaded {self.name}")

# ── Dataset builder ───────────────────────────────────────────────────────────

def process_model(m: LocalModel, all_records: List[Dict], task_data: Dict[str, List[Dict]]):
    """Fill in model_responses / model_ppl / model_bf for every record."""
    for task, raw_samples in task_data.items():
        print(f"\n{'─'*55}")
        print(f"  {m.name.split('/')[-1]}  ·  {task.upper()}")
        print(f"{'─'*55}")

        for idx, sample in enumerate(raw_samples):
            # Find the pre-created record for this sample
            record = next(
                r for r in all_records
                if r["input_text"] == sample["input_text"] and r["topic_label"] == task
            )
            variants = record["prompt_variants"]

            record["model_responses"][m.name] = {}
            record["model_ppl"][m.name]       = {}
            record["model_bf"][m.name]        = {}

            print(f"\n  Sample {idx+1}/{len(raw_samples)}")
            for level, prompt in variants.items():
                t0       = time.time()
                response = m.generate(prompt, task)
                ppl      = m.perplexity(prompt, response)
                bf       = m.branching_factor(response)
                elapsed  = time.time() - t0

                record["model_responses"][m.name][level] = response
                record["model_ppl"][m.name][level]       = ppl
                record["model_bf"][m.name][level]        = bf

                preview = response[:100].replace("\n", " ")
                print(f"    [{level}] {elapsed:.1f}s | ppl={ppl:.2f} bf={bf:.3f} → {preview}")


def save(records: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[save] {len(records)} records → {path}")


def save_csv(records: List[Dict], path: Path):
    """
    Flatten nested JSON into one row per (sample × model × variant).
    Columns:
      topic_label, sample_idx, model, variant,
      input_text, reference_output,
      prompt_variant_text, response, ppl, bf
    """
    rows = []
    for s_idx, rec in enumerate(records):
        for model_name, variants in rec["model_responses"].items():
            for level, response in variants.items():
                rows.append({
                    "topic_label":        rec["topic_label"],
                    "sample_idx":         s_idx,
                    "model":              model_name,
                    "variant":            level,
                    "input_text":         rec["input_text"][:300].replace("\n", " "),
                    "reference_output":   rec["reference_output"][:200].replace("\n", " "),
                    "prompt_variant_text": rec["prompt_variants"].get(level, "")[:200].replace("\n", " "),
                    "response":           response[:300].replace("\n", " "),
                    "ppl":                rec["model_ppl"].get(model_name, {}).get(level, ""),
                    "bf":                 rec["model_bf"].get(model_name,  {}).get(level, ""),
                })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] {len(rows)} rows → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  GenSens Sample Dataset Generator")
    print(f"  Device: {DEVICE}  |  Samples/task: {SAMPLES_PER_TASK}")
    print(f"  Models: {len(MODELS)}  |  Variants: d1/d2/d3")
    print("=" * 55)

    # ── Step 1: load source data ──────────────────────────────
    task_data = {
        "summarization":   load_summarization(SAMPLES_PER_TASK),
        "code":            load_code(SAMPLES_PER_TASK),
        "creative_writing": load_creative_writing(SAMPLES_PER_TASK),
        "dialogue":        load_dialogue(SAMPLES_PER_TASK),
    }

    # ── Step 2: create skeleton records ──────────────────────
    all_records: List[Dict] = []
    for task, raw_samples in task_data.items():
        for sample in raw_samples:
            all_records.append({
                "input_text":       sample["input_text"],
                "reference_output": sample["reference_output"],
                "topic_label":      task,
                "prompt_variants":  build_variants(sample["input_text"], task),
                "model_responses":  {},
                "model_ppl":        {},
                "model_bf":         {},
            })

    # ── Step 3: fill responses — one model at a time (low RAM) ─
    for model_name in MODELS:
        m = LocalModel(model_name)
        process_model(m, all_records, task_data)
        m.unload()

    # ── Step 4: save JSON + CSV ───────────────────────────────
    print("\n")
    for task in task_data:
        task_records = [r for r in all_records if r["topic_label"] == task]
        save(task_records,     DATA_DIR / f"dataset_{task}.json")
        save_csv(task_records, DATA_DIR / f"dataset_{task}.csv")

    save(all_records,     DATA_DIR / "dataset_all_tasks.json")
    save_csv(all_records, DATA_DIR / "dataset_all_tasks.csv")

    print("\n[done] Generation complete.")
    print(f"  Total records : {len(all_records)}")
    print(f"  Output dir    : {DATA_DIR}")
    print("\n  Files saved:")
    for f in sorted(DATA_DIR.glob("dataset_*")):
        print(f"    {f.name}")
    print("\n  Next step: run the evaluator with --data data/dataset_all_tasks.json")
    print("  (All responses + PPL pre-stored → no GPU needed for metric computation)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
