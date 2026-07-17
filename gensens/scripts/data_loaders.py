"""
GenSens Data Loaders
====================
Loads source instances for all 4 tasks:
  - Summarization  (CNN/DailyMail 3.0.0 — "sum_in_brief" framing)
  - Creative       (CNN/DailyMail 3.0.0 — "generate_story" framing)
  - Dialogue       (DREAM reading-comprehension)
  - QA             (sentence-transformers/eli5, "pair" config)

Each loader returns a list of dicts with the unified schema:
  {
    "instance_id": str,
    "task":        str,
    "base_text":   str,    # TEXT THAT GETS PARAPHRASED (instruction OR question)
    "base_prompt": str,    # full prompt = base_text embedded in the template
    "metadata":    dict    # MUST carry input_text (grounding) + reference_output (gold)
  }

Bridge contract (closes the reference gap — ALL FOUR tasks carry a reference):
  metadata["input_text"]        → grounding source (article / summary / dialogue / question)
  metadata["reference_output"]  → gold output     (highlights / article / answer / answer)
Both are non-empty for every emitted record; rows lacking a reference are skipped.
"""

import re
import random
import logging
from typing import List, Dict, Any, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _is_likely_english(text: str) -> bool:
    """Simple ASCII-ratio heuristic for English detection."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.90


def _select_cnndm_stratified(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load CNN/DailyMail test split and stratify by article length.

    Shared by both CNN/DM tasks (summarization + creative) so the same
    length-stratified sampler (short <400 / medium 400–800 / long >800 words,
    seed 42) is applied to each, per the task spec.

    Returns a list of {article, highlights, word_count} dicts.
    """
    logger.info("Loading CNN/DailyMail dataset (3.0.0, test split)...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    logger.info(f"  Loaded {len(ds)} test articles")

    short, medium, long = [], [], []
    for item in ds:
        article = item["article"]
        highlights = item["highlights"]
        # HARD REQUIREMENT: both grounding and reference must be non-empty.
        if not article or not article.strip() or not highlights or not highlights.strip():
            continue
        wc = len(article.split())
        entry = {"article": article, "highlights": highlights, "word_count": wc}
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
    per_bucket = n // 3
    remainder = n - per_bucket * 3

    rng.shuffle(short)
    rng.shuffle(medium)
    rng.shuffle(long)

    selected = (
        short[:per_bucket]
        + medium[:per_bucket]
        + long[:per_bucket + remainder]
    )

    # If any bucket too small, fill from the rest.
    if len(selected) < n:
        all_items = short + medium + long
        ids_seen = {id(x) for x in selected}
        rng.shuffle(all_items)
        for item in all_items:
            if id(item) not in ids_seen:
                selected.append(item)
                ids_seen.add(id(item))
            if len(selected) >= n:
                break

    selected = selected[:n]
    rng.shuffle(selected)
    return selected


# ─────────────────────────────────────────────────────────────
# TASK 1 — Summarization (CNN/DailyMail, "sum_in_brief")
# ─────────────────────────────────────────────────────────────

def load_summarization(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    CNN/DailyMail "sum_in_brief": paraphrase the summarization instruction.
    Fixed: the article. Reference: the gold highlights.
    """
    selected = _select_cnndm_stratified(n, seed)

    instruction = (
        "Summarize the following news article in 3-4 sentences, "
        "capturing the main events and key details."
    )

    records = []
    for i, item in enumerate(selected):
        base_text = instruction
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
                "input_text":       item["article"],      # grounding
                "reference_output": item["highlights"],   # gold
                "source":           "cnn_dailymail:3.0.0/sum_in_brief",
                "word_count":       item["word_count"],
                # Legacy aliases (kept so any older summarization reader still works):
                "article":      item["article"],
                "gold_summary": item["highlights"],
            },
        })

    logger.info(f"  Loaded {len(records)} summarization instances")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 2 — Creative (CNN/DailyMail, "generate_story")
# ─────────────────────────────────────────────────────────────

def load_creative(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    CNN/DailyMail "generate_story": paraphrase the highlights (summary premise).
    Fixed: the story-writing instruction. Reference: the full news story (article).

    The instruction is constant across all instances and variants; only the
    highlights are paraphrased. This tests whether lexically different phrasings
    of the same facts/premise lead to different generated stories.
    """
    selected = _select_cnndm_stratified(n, seed)

    instruction = (
        "Write a detailed news story that expands on the following summary points, "
        "elaborating them into a complete article."
    )

    records = []
    for i, item in enumerate(selected):
        base_text = item["highlights"]   # PARAPHRASE THIS
        base_prompt = (
            instruction
            + "\n\nSummary:\n"
            + base_text
            + "\n\nStory:"
        )
        records.append({
            "instance_id": f"crea_{i:04d}",
            "task":        "creative",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "input_text":       item["highlights"],   # grounding (premise)
                "reference_output": item["article"],       # gold story
                "source":           "cnn_dailymail:3.0.0/generate_story",
                "word_count":       item["word_count"],
                "instruction":      instruction,           # fixed; stored for readers
            },
        })

    logger.info(f"  Loaded {len(records)} creative instances")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 3 — Dialogue (DREAM)
# ─────────────────────────────────────────────────────────────

def _format_choices(choices: List[str]) -> str:
    return "\n".join(f"- {c}" for c in choices)


def _load_dream_from_github() -> List[Dict[str, Any]]:
    """Fallback: HF `datasets` ≥3.0 rejects script-based loaders, so download the
    canonical DREAM train.json straight from the upstream GitHub repo and
    reshape it to the same per-row schema HF used to emit."""
    import json as _json
    from urllib.request import urlopen

    url = "https://raw.githubusercontent.com/nlpdata/dream/master/data/train.json"
    logger.info(f"  HF load failed — falling back to upstream GitHub: {url}")
    with urlopen(url, timeout=60) as resp:
        raw = _json.loads(resp.read().decode("utf-8"))
    rows: List[Dict[str, Any]] = []
    # Schema (nlpdata/dream): [ [dialogue_turns], [{question, choice, answer}, ...], dialogue_id ]
    for entry in raw:
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        turns, qas, did = entry[0], entry[1], entry[2]
        for qa in qas:
            rows.append({
                "dialogue_id": did,
                "dialogue":    turns,
                "question":    qa.get("question", ""),
                "choice":      qa.get("choice", []),
                "answer":      qa.get("answer", ""),
            })
    return rows


def load_dialogue(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    DREAM dialogue reading-comprehension: paraphrase the question.
    Fixed: the dialogue turns (+ answer options). Reference: the correct answer choice.
    """
    logger.info("Loading DREAM dataset (train split)...")
    try:
        ds = load_dataset("dream", split="train", trust_remote_code=True)
        logger.info(f"  Loaded {len(ds)} DREAM instances via HF datasets")
    except Exception as e:
        # datasets >= 3.0 dropped script-based loaders. Fall back to upstream JSON.
        logger.warning(f"  HF load failed: {e}")
        ds = _load_dream_from_github()
        logger.info(f"  Loaded {len(ds)} DREAM instances from upstream JSON")

    candidates = []
    for item in ds:
        dialogue = item.get("dialogue") or []
        question = (item.get("question") or "").strip()
        choices = item.get("choice") or []
        answer = (item.get("answer") or "").strip()

        # HARD REQUIREMENT: non-empty reference (answer) + usable grounding/question.
        if not dialogue or not question or not answer or not choices:
            continue

        dialogue_text = "\n".join(t.strip() for t in dialogue if t and t.strip())
        if not dialogue_text:
            continue

        candidates.append({
            "dialogue_id":   item.get("dialogue_id", f"dream_{len(candidates)}"),
            "dialogue_text": dialogue_text,
            "question":      question,
            "choices":       list(choices),
            "answer":        answer,
        })

    logger.info(f"  After filtering (non-empty dialogue/question/answer): {len(candidates)}")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    if len(selected) < n:
        logger.warning(f"  Only {len(selected)} qualifying dialogues found (wanted {n}).")

    system_msg = (
        "[SYSTEM]: Read the dialogue and answer the question using the options provided.\n\n"
    )

    records = []
    for i, item in enumerate(selected):
        base_text = item["question"]
        base_prompt = (
            system_msg
            + "Dialogue:\n"
            + item["dialogue_text"]
            + "\n\nQuestion: " + base_text
            + "\nOptions:\n" + _format_choices(item["choices"])
            + "\nAnswer:"
        )
        records.append({
            "instance_id": f"dial_{i:04d}",
            "task":        "dialogue",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "input_text":       item["dialogue_text"],  # grounding (the turns)
                "reference_output": item["answer"],          # gold answer choice
                "source":           "dream",
                "dialogue_id":      item["dialogue_id"],
                "choices":          item["choices"],
            },
        })

    logger.info(f"  Loaded {len(records)} dialogue instances")
    return records


# ─────────────────────────────────────────────────────────────
# TASK 4 — QA (sentence-transformers/eli5)
# ─────────────────────────────────────────────────────────────

def load_qa(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    ELI5 long-form QA: paraphrase the question.
    Reference: the answer/explanation. Grounding (input_text): the question.
    """
    logger.info("Loading sentence-transformers/eli5 (pair config, train split)...")
    ds = load_dataset("sentence-transformers/eli5", "pair", split="train", trust_remote_code=True)
    logger.info(f"  Loaded {len(ds)} ELI5 question/answer pairs")

    candidates = []
    for item in ds:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()

        # HARD REQUIREMENT: non-empty reference (answer). Light quality filters:
        #   - question must be a real, paraphrasable question (5–60 words, English)
        #   - answer must be a usable reference (≥ 10 words)
        if not question or not answer:
            continue
        q_wc = len(question.split())
        a_wc = len(answer.split())
        if not (5 <= q_wc <= 60):
            continue
        if a_wc < 10:
            continue
        if not _is_likely_english(question) or not _is_likely_english(answer):
            continue

        candidates.append({"question": question, "answer": answer})

    logger.info(f"  After filtering (5-60 word EN question, ≥10 word answer): {len(candidates)}")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    if len(selected) < n:
        logger.warning(f"  Only {len(selected)} qualifying QA pairs found (wanted {n}).")

    records = []
    for i, item in enumerate(selected):
        base_text = item["question"]
        base_prompt = (
            "Answer the following question clearly and accurately.\n\n"
            + "Question: " + base_text
            + "\n\nAnswer:"
        )
        records.append({
            "instance_id": f"qa_{i:04d}",
            "task":        "qa",
            "base_text":   base_text,
            "base_prompt": base_prompt,
            "metadata": {
                "input_text":       item["question"],  # grounding (the question)
                "reference_output": item["answer"],     # gold answer/explanation
                "source":           "sentence-transformers/eli5:pair",
            },
        })

    logger.info(f"  Loaded {len(records)} QA instances")
    return records


# ─────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────

TASK_LOADERS = {
    "summarization": load_summarization,
    "creative":      load_creative,
    "dialogue":      load_dialogue,
    "qa":            load_qa,
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
            print(f"  base_text[:100]:        {recs[0]['base_text'][:100]}")
            print(f"  base_prompt[:150]:      {recs[0]['base_prompt'][:150]}")
            print(f"  input_text[:80]:        {recs[0]['metadata']['input_text'][:80]}")
            print(f"  reference_output[:80]:  {recs[0]['metadata']['reference_output'][:80]}")
