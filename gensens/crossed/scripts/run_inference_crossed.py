#!/usr/bin/env python3
"""
run_inference_crossed.py — Crossed-design inference for the GenSens
summarization prompt-sensitivity benchmark.

CROSSED DESIGN (instance-fixed, prompt-varied): every one of the 50 seed
prompt groups is applied to the SAME set of N_ARTICLES articles. For each
(article, seed) cell we run all 6 variants (1 base + 5 paraphrases) and
later measure how the 6 outputs differ (see compute_cell_metrics.py).
This mirrors POSIX (2410.02185), ProSA (2410.12405), PromptBench — never
pair one prompt with one article.

Decoding is hard-coded to greedy (temperature=0.0) for the entire main run:
sampling would conflate prompt sensitivity with decoding-randomness noise.

Run from anywhere; paths resolve relative to this file via common.py.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_inference_crossed")

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def check_hf_auth() -> None:
    """Best-effort `huggingface-cli whoami` — logs the result, never fatal
    (some environments authenticate via HF_TOKEN env var without a cached
    login, and gated-model access is validated for real at model load time
    regardless)."""
    try:
        out = subprocess.run(
            ["huggingface-cli", "whoami"], capture_output=True, text=True, timeout=30
        )
        if out.returncode == 0:
            log.info(f"huggingface-cli whoami: {out.stdout.strip()}")
        else:
            log.warning(f"huggingface-cli whoami failed (rc={out.returncode}): {out.stderr.strip()}")
    except Exception as e:
        log.warning(f"Could not run huggingface-cli whoami: {e}")


def build_task_list(articles, seeds):
    """Nested article -> seed -> variant, exactly per the crossed-design
    pseudocode. Order only affects checkpoint-chunk boundaries; the design
    itself (every seed x every article) is what matters."""
    tasks = []
    for article in articles:
        for seed in seeds:
            for variant in seed["variants"]:
                rendered = variant["full_prompt"].replace("{{article}}", article["text"])
                tasks.append({
                    "article_id": article["article_id"],
                    "seed_id": seed["seed_id"],
                    "pool": seed["pool"],
                    "dimension": seed["dimension"],
                    "variant_idx": variant["variant_idx"],
                    "full_prompt": rendered,
                    "gold_summary": article["gold_summary"],
                    "article_word_count": article["word_count"],
                })
    return tasks


def load_completed_keys(responses_path) -> set:
    done = set()
    for row in common.iter_jsonl(responses_path):
        done.add((row["article_id"], row["seed_id"], row["variant_idx"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_articles", type=int, default=100)
    ap.add_argument("--article_seed", type=int, default=42)
    ap.add_argument("--seed_file", default=None, help="Override path to the 50-seed/5-paraphrase jsonl")
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem_frac", type=float, default=0.88)
    ap.add_argument("--checkpoint_every", type=int, default=2000)
    ap.add_argument("--limit_seeds", type=int, default=None, help="Debug: only use the first K seeds")
    ap.add_argument("--limit_articles", type=int, default=None, help="Debug: only use the first K articles")
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Build the task list and validate prompts/resume state WITHOUT loading vLLM or "
             "generating anything. For pipeline/engineering validation only — never used for "
             "the real run (no GPU here would otherwise be silently faked).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else common.RESULTS_DIR_DEFAULT
    results_dir.mkdir(parents=True, exist_ok=True)
    responses_path = results_dir / "responses_crossed.jsonl"

    check_hf_auth()

    seeds = common.load_seed_prompts(args.seed_file)
    log.info(f"Loaded {len(seeds)} seeds (expected {common.N_SEEDS_EXPECTED}).")
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]
        log.warning(f"--limit_seeds set: using only {len(seeds)} seeds (DEBUG ONLY, not a valid full run).")

    articles = common.sample_articles(
        n_articles=args.n_articles, seed=args.article_seed,
        cache_path=results_dir / "articles_sample.jsonl",
    )
    log.info(f"Sampled/loaded {len(articles)} articles (stratified by word count, seed={args.article_seed}).")
    if args.limit_articles:
        articles = articles[: args.limit_articles]
        log.warning(f"--limit_articles set: using only {len(articles)} articles (DEBUG ONLY).")

    tasks = build_task_list(articles, seeds)
    # Per-seed variant counts can be < N_VARIANTS_PER_SEED for known-flagged seeds
    # (e.g. B12, see common.py) — expected total is the sum of each seed's actual
    # variant count x n_articles, not a flat n_seeds x 6 product.
    variants_per_seed_total = sum(len(s["variants"]) for s in seeds)
    expected_total = len(articles) * variants_per_seed_total
    log.info(f"Built {len(tasks)} total (article, seed, variant) tasks.")
    if len(tasks) != expected_total:
        raise AssertionError(f"Task count mismatch: {len(tasks)} != {expected_total}")

    done = load_completed_keys(responses_path)
    pending = [t for t in tasks if (t["article_id"], t["seed_id"], t["variant_idx"]) not in done]
    log.info(f"{len(done)} already completed (resume), {len(pending)} pending.")

    if args.dry_run:
        log.info("DRY RUN: skipping vLLM load and generation. Task list validated OK.")
        return

    if not pending:
        log.info("Nothing to do — all tasks already completed.")
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    log.info(f"Loading tokenizer + vLLM engine for {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_frac,
        max_model_len=args.max_model_len,
        dtype="auto",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    n_done_this_run = 0
    t0 = time.time()
    for chunk_start in range(0, len(pending), args.checkpoint_every):
        chunk = pending[chunk_start: chunk_start + args.checkpoint_every]
        chat_prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": t["full_prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in chunk
        ]
        outs = llm.generate(chat_prompts, sampling_params)
        rows = []
        for t, o in zip(chunk, outs):
            rows.append({
                "article_id": t["article_id"],
                "seed_id": t["seed_id"],
                "pool": t["pool"],
                "dimension": t["dimension"],
                "variant_idx": t["variant_idx"],
                "full_prompt": t["full_prompt"],
                "output": o.outputs[0].text.strip(),
                "gold_summary": t["gold_summary"],
                "article_word_count": t["article_word_count"],
            })
        common.append_jsonl(responses_path, rows)
        n_done_this_run += len(rows)
        elapsed = time.time() - t0
        rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
        remaining = len(pending) - n_done_this_run
        eta_min = (remaining / rate / 60.0) if rate > 0 else float("nan")
        log.info(
            f"Checkpoint: {n_done_this_run}/{len(pending)} this run "
            f"({len(done) + n_done_this_run}/{len(tasks)} total). "
            f"{rate:.2f} outputs/s, ETA {eta_min:.1f} min."
        )

    log.info(f"Done. Wrote {n_done_this_run} new outputs to {responses_path}.")
    total_now = len(load_completed_keys(responses_path))
    if total_now != len(tasks):
        log.warning(f"Total completed ({total_now}) != expected total ({len(tasks)}). Re-run to fill gaps.")
    else:
        log.info(f"All {total_now} outputs present — inference phase complete.")


if __name__ == "__main__":
    main()
