"""
GenSens Data Loaders
====================
Loads source instances for all 4 tasks:
  - Summarization  (CNN/DailyMail)
  - Code Generation (HumanEval + MBPP)
  - Creative Writing (WritingPrompts)
  - Dialogue       (MultiWOZ 2.2)

Each loader returns a list of dicts with the unified schema:
  {
    "instance_id": str,
    "task":        str,
    "base_text":   str,    # TEXT THAT GETS PARAPHRASED
    "base_prompt": str,    # full prompt = base_text + context
    "metadata":    dict
  }
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TASK 1 — Summarization (CNN/DailyMail)
# ─────────────────────────────────────────────────────────────

def load_summarization(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load CNN/DailyMail test split, stratify by article length,
    and return n instances.
    """
    logger.info("Loading CNN/DailyMail dataset...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    logger.info(f"  Loaded {len(ds)} test articles")

    # Compute word counts and bucket
    short, medium, long = [], [], []
    for item in ds:
        wc    = len(item["article"].split())
        entry = {**item, "word_count": wc}
        if wc < 400:
            short.append(entry)
        elif wc <= 800:
            medium.append(entry)
        else:
            long.append(entry)

    logger.info(
        f"  Buckets — short(<400): {len(short)}, "
        f"medium(400-800): {len(medium)}, long(>800): {len(long)}"
    )

    rng = random.Random(seed)

    # Stratified sampling: roughly equal from each bucket
    per_bucket = n // 3
    remainder  = n - per_bucket * 3

    rng.shuffle(short)
    rng.shuffle(medium)
    rng.shuffle(long)

    selected = (
        short[:per_bucket]
        + medium[:per_bucket]
        + long[:per_bucket + remainder]
    )

    # If any bucket too small, fill from the rest
    if len(selected) < n:
        all_items  = short + medium + long
        ids_seen   = {id(x) for x in selected}
        rng.shuffle(all_items)
        for item in all_items:
            if id(item) not in ids_seen:
                selected.append(item)
                ids_seen.add(id(item))
            if len(selected) >= n:
                break

    selected = selected[:n]
    rng.shuffle(selected)

    instruction = (
        "Summarize the following news article in 3-4 sentences, "
        "capturing the main events and key details."
    )

    records = []
    for i, item in enumerate(selected):
        base_text   = instruction
        base_prompt = (
            base_text
            + "\n\nArticle:\n"
            + item["article"]
            + "\n\nSummary:"
        )
        records.append({
            "instance_id": f"summ_{i:04d}",
            "task":        "summarization",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "article":      item["article"],
                "gold_summary": item["highlights"],
                "word_count":   item["word_count"],
            },
        })

    logger.info(f"  Loaded {len(records)} summarization instances")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 2 — Code Generation (HumanEval + MBPP)
# ─────────────────────────────────────────────────────────────

def _extract_docstring(prompt: str) -> str:
    """Extract the triple-quoted docstring from a HumanEval prompt."""
    match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if match:
        # Strip out example lines (>>> ...) to keep pure description
        lines = [
            l for l in match.group(1).strip().splitlines()
            if not l.strip().startswith(">>>")
        ]
        return "\n".join(lines).strip()
    match = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: extract lines after the def signature, before any >>>
    lines, in_body, desc = prompt.strip().split("\n"), False, []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            in_body = True
            continue
        if in_body and stripped and not stripped.startswith(">>>"):
            desc.append(stripped)
    return " ".join(desc) if desc else prompt.strip()


def load_code(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load HumanEval (164 problems) + MBPP (36 to reach 200).
    """
    logger.info("Loading HumanEval dataset...")
    he_ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    logger.info(f"  Loaded {len(he_ds)} HumanEval problems")

    logger.info("Loading MBPP dataset...")
    mbpp_ds = load_dataset("mbpp", split="test", trust_remote_code=True)
    logger.info(f"  Loaded {len(mbpp_ds)} MBPP problems")

    rng     = random.Random(seed)
    records = []

    # ── HumanEval ──────────────────────────────────────────
    for item in he_ds:
        docstring = _extract_docstring(item["prompt"])
        records.append({
            "instance_id": f"code_{len(records):04d}",
            "task":        "code",
            "base_text":   docstring,
            "base_prompt": item["prompt"],
            "metadata": {
                "task_id":            item["task_id"],
                "test_code":          item["test"],
                "entry_point":        item["entry_point"],
                "source":             "humaneval",
                "canonical_solution": item["canonical_solution"],
            },
        })

    # ── MBPP (fill remaining) ──────────────────────────────
    n_mbpp_needed = n - len(records)
    mbpp_list     = list(mbpp_ds)
    rng.shuffle(mbpp_list)

    for item in mbpp_list[:n_mbpp_needed]:
        description = item["text"]
        base_prompt = f"# {description}\n\ndef solution():\n"
        records.append({
            "instance_id": f"code_{len(records):04d}",
            "task":        "code",
            "base_text":   description,
            "base_prompt": base_prompt,
            "metadata": {
                "task_id":     f"mbpp_{item['task_id']}",
                "test_list":   item["test_list"],
                "entry_point": "solution",
                "source":      "mbpp",
            },
        })

    records = records[:n]
    logger.info(f"  Loaded {len(records)} code instances (HumanEval + MBPP)")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 3 — Creative Writing (WritingPrompts)
# ─────────────────────────────────────────────────────────────

# Matches [WP], [ CW ], [TT], [Writing Prompt], etc. (space-padded or not)
_REDDIT_TAG_RE = re.compile(
    r'\[\s*(?:WP|TT|EU|CW|RF|MK|PI|CC|OT|TH|PM|SP|MP|LP|HP|DP|FP|EP|IP|WS|NS|NF|'
    r'TI|DC|RT|FF|Prompt\s+Me|Writing\s+Prompt|Theme\s+Thursday|Serial|[A-Z]{1,3})\s*\]',
    re.IGNORECASE,
)
_URL_RE      = re.compile(r'https?://\S+')
_EXTRA_SPACE = re.compile(r'\s+')


def _clean_writing_prompt(text: str) -> str:
    """Remove Reddit tag artifacts, URLs, and normalise whitespace."""
    text = _REDDIT_TAG_RE.sub('', text)
    text = _URL_RE.sub('', text)
    text = _EXTRA_SPACE.sub(' ', text).strip()
    return text


def _is_likely_english(text: str) -> bool:
    """Simple ASCII-ratio heuristic for English detection."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.90


def load_creative(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load WritingPrompts train split, filter for English prompts of 10-60 words,
    clean Reddit artifacts, select n random prompts (seed=42).
    """
    logger.info("Loading WritingPrompts dataset...")
    ds = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=True)
    logger.info(f"  Loaded {len(ds)} writing prompts")

    candidates = []
    for item in ds:
        # The dataset uses "prompt" field (not "title")
        raw     = item.get("prompt") or item.get("title") or item.get("text") or ""
        cleaned = _clean_writing_prompt(raw)
        wc      = len(cleaned.split())
        if 10 <= wc <= 60 and _is_likely_english(cleaned) and len(cleaned) > 20:
            candidates.append({"text": cleaned, "word_count": wc})

    logger.info(f"  After filtering (10-60 words, English): {len(candidates)} prompts")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    if len(selected) < n:
        logger.warning(
            f"  Only {len(selected)} qualifying prompts found (wanted {n})."
        )

    records = []
    for i, item in enumerate(selected):
        base_text   = item["text"]
        base_prompt = (
            "Write a short story (150-200 words) that expands on the following prompt.\n\n"
            f"Prompt: {base_text}\n\nStory:"
        )
        records.append({
            "instance_id": f"crea_{i:04d}",
            "task":        "creative",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "original_prompt": item["text"],
                "word_count":      item["word_count"],
            },
        })

    logger.info(f"  Loaded {len(records)} creative writing instances")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 4 — Dialogue (MultiWOZ 2.2)
# ─────────────────────────────────────────────────────────────

def load_dialogue(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load MultiWOZ 2.2 test split, filter for dialogues with 4-8 total turns.
    MultiWOZ always ends with a system turn, so the last USER utterance is
    the penultimate speaker=0 turn — that becomes base_text.
    History = all turns before that last user turn (kept fixed across variants).
    """
    logger.info("Loading MultiWOZ 2.2 dataset...")
    ds = load_dataset("multi_woz_v22", split="test", trust_remote_code=True)
    logger.info(f"  Loaded {len(ds)} dialogues")

    candidates = []
    for item in ds:
        turns      = item.get("turns", {})
        utterances = turns.get("utterance", [])
        speakers   = turns.get("speaker",   [])

        if not utterances or not speakers:
            continue

        n_turns = len(utterances)
        if not (4 <= n_turns <= 8):
            continue

        # Find last USER turn (speaker == 0)
        last_user_idx = None
        for idx in range(len(speakers) - 1, -1, -1):
            if int(speakers[idx]) == 0:
                last_user_idx = idx
                break

        # Skip if no user turn found, or it's the very first turn (no history)
        if last_user_idx is None or last_user_idx == 0:
            continue

        last_question = utterances[last_user_idx]

        # Build history string (all turns before the last user turn)
        history_lines = []
        for j in range(last_user_idx):
            role = "[USER]" if int(speakers[j]) == 0 else "[ASSISTANT]"
            history_lines.append(f"{role}: {utterances[j]}")
        history = "\n".join(history_lines)

        candidates.append({
            "dial_id":       item.get("dialogue_id", f"dial_{len(candidates)}"),
            "history":       history,
            "last_question": last_question,
            "n_turns":       n_turns,
        })

    logger.info(f"  After filtering (4-8 turns, ends with user): {len(candidates)} dialogues")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    if len(selected) < n:
        logger.warning(
            f"  Only {len(selected)} qualifying dialogues found (wanted {n})."
        )

    records = []
    for i, item in enumerate(selected):
        base_text   = item["last_question"]
        system_msg  = "[SYSTEM]: You are a helpful assistant for travel, hotel, and restaurant bookings.\n\n"
        base_prompt = (
            system_msg
            + item["history"]
            + "\n"
            + f"[USER]: {base_text}\n[ASSISTANT]:"
        )
        records.append({
            "instance_id": f"dial_{i:04d}",
            "task":        "dialogue",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "dial_id":       item["dial_id"],
                "history":       item["history"],
                "last_question": item["last_question"],
            },
        })

    logger.info(f"  Loaded {len(records)} dialogue instances")
    return records


# ─────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────

TASK_LOADERS = {
    "summarization": load_summarization,
    "code":          load_code,
    "creative":      load_creative,
    "dialogue":      load_dialogue,
}


def load_all_tasks(
    n_per_task: int = 200,
    seed:       int = 42,
    tasks: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load data for all (or specified) tasks.
    Returns dict mapping task_name → list of records.
    Catches per-task errors so one failure doesn't abort the rest.
    """
    if tasks is None:
        tasks = list(TASK_LOADERS.keys())

    results: Dict[str, List[Dict[str, Any]]] = {}
    for task_name in tasks:
        if task_name not in TASK_LOADERS:
            logger.error(f"Unknown task: {task_name}")
            continue
        try:
            records = TASK_LOADERS[task_name](n=n_per_task, seed=seed)
            results[task_name] = records
        except Exception as e:
            logger.error(f"Failed to load task '{task_name}': {e}", exc_info=True)
            results[task_name] = []

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data = load_all_tasks(n_per_task=3)
    for task, recs in data.items():
        print(f"\n{'='*60}")
        print(f"Task: {task} — {len(recs)} records")
        if recs:
            print(f"  base_text[:100]:   {recs[0]['base_text'][:100]}")
            print(f"  base_prompt[:150]: {recs[0]['base_prompt'][:150]}")
