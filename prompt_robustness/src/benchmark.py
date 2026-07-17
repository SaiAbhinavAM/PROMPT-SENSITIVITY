"""
benchmark.py — Multi-model benchmarking orchestrator for the Prompt Robustness Framework.

Manages the end-to-end benchmark pipeline:
1. Load dataset and configure models
2. Run per-sample evaluation across all models
3. Aggregate per-model statistics
4. Compute correlation analysis (STEP 6)
5. Save comprehensive results (JSON, CSV)
"""

import gc
import time
import os
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from .config import Config
from .data_loader import load_dataset
from .model_interface import ModelInterface
from .cache_manager import CacheManager
from .evaluator import evaluate_sample
from .utils import generate_run_id, save_json
from . import csv_io

# STEP 6: Correlation analysis
from .analysis import (
    compute_full_correlation_analysis,
    save_correlation_results,
    compute_metric_summary_statistics,
)

# Table rendering
from .table_utils import render_stability_table, render_correlation_table

logger = logging.getLogger(__name__)

def evaluate_sample_wrapper(args):
    sample, config, model_interface, cache_manager = args
    return evaluate_sample(sample, config, model_interface, cache_manager)

def _save_task_split_csvs(resp_path: str, scored_path: str, results_dir: str) -> None:
    """Split responses.csv and scored_samples.csv into per-task files.

    Writes:
      results/responses_<task>.csv        — full prompt+response+judge cols per task
      results/scored_samples_<task>.csv   — all metric cols per task
    """
    for src_path, prefix in [(resp_path, "responses"), (scored_path, "scored_samples")]:
        if not os.path.exists(src_path):
            continue
        try:
            df = pd.read_csv(src_path, dtype=str)
        except Exception:
            continue
        label_col = "topic_label" if "topic_label" in df.columns else None
        if label_col is None:
            continue
        for task in df[label_col].dropna().unique():
            task_df = df[df[label_col] == task]
            out = os.path.join(results_dir, f"{prefix}_{task}.csv")
            task_df.to_csv(out, index=False)
            print(f"  📄 {prefix}_{task}.csv — {len(task_df)} rows → {out}")


def benchmark_models(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict]]:
    """
    Run the full benchmark pipeline.

    Returns:
        Tuple of (aggregated_df, sample_df, correlation_results).
        - aggregated_df: Per-model aggregated metrics.
        - sample_df: Per-sample metrics (for stability analysis).
        - correlation_results: Correlation analysis dict (or None).
    """
    df = load_dataset(config.data_path)
    limit = config.max_samples if getattr(config, "max_samples", 0) else len(df)
    samples = df.to_dict('records')[:limit]
    for i, s in enumerate(samples):
        s.setdefault("instance_id", f"idx_{i}")
    print(f"Running evaluation on {len(samples)} samples")
    print(f"Models to run: {config.models}")

    cache_manager = CacheManager(config.cache_dir, config.enable_cache)

    # Fault-tolerant incremental CSV persistence (resume-safe): every completed
    # sample is flushed+fsync'd immediately, so a crash never loses computed work
    # and a re-run skips what is already on disk.
    os.makedirs(config.results_dir, exist_ok=True)
    resp_path = os.path.join(config.results_dir, "responses.csv")
    scored_path = os.path.join(config.results_dir, "scored_samples.csv")
    done_keys = csv_io.existing_scored_keys(scored_path)
    if done_keys:
        print(f"↻ Resuming: {len(done_keys)} (model,instance) pairs already scored — skipping them")
    resp_writer = csv_io.IncrementalCSVWriter(resp_path, csv_io.RESPONSES_COLS, resume=True)
    scored_writer = csv_io.IncrementalCSVWriter(scored_path, csv_io.SCORED_COLS, resume=True)

    all_results = []
    last_sample_df = pd.DataFrame()
    
    run_id = generate_run_id()
    run_details = {
        "run_id": run_id,
        "config": {
            "models": config.models,
            "device": config.device,
            "enable_cache": config.enable_cache,
            "enable_parallel": config.enable_parallel,
            "enable_advanced_metrics": config.enable_advanced_metrics,
            "enable_dynamic_weighting": config.enable_dynamic_weighting,
            "enable_rouge": config.enable_rouge,
            "enable_bertscore": config.enable_bertscore,
            "enable_correlation_analysis": config.enable_correlation_analysis,
        },
        "dataset_size": len(samples),
        "results": []
    }

    for model_idx, model_name in enumerate(config.models, 1):
        print(f"\n{'='*50}")
        print(f"🚀 Running model [{model_idx}/{len(config.models)}]: {model_name}")
        print(f"{'='*50}")
        
        logger.info(f" Benchmarking model: {model_name}")
        start_time = time.time()
        
        # Load model interface (pass quantization spec for AWQ/GPTQ models)
        quant = getattr(config, "subject_quantizations", {}).get(model_name)
        try:
            model_interface = ModelInterface(model_name, config, quantization=quant)
        except Exception as e:
            logger.error(f"Cannot load model '{model_name}'. Skipping... Error: {e}")
            print(f"Model {model_name} failed. Continuing...")
            continue
            
        # Skip samples already scored for this model (resume).
        pending = [s for s in samples if (model_name, s.get("instance_id", "")) not in done_keys]
        if len(pending) < len(samples):
            print(f"   Skipping {len(samples) - len(pending)} already-scored samples for {model_name}")
        tasks = [(sample, config, model_interface, cache_manager) for sample in pending]

        def _record(r):
            """Persist one result immediately (crash-safe) and accumulate."""
            resp_writer.write_rows(csv_io.responses_rows_from_result(r))
            scored_writer.write_rows([csv_io.scored_row_from_result(r)])
            results.append(r)

        results = []
        if config.enable_parallel and model_interface.device.type == "cpu":
            # Parallel execution for API models or CPU models
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = [executor.submit(evaluate_sample_wrapper, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model_name}"):
                    try:
                        _record(future.result())
                    except Exception as e:
                        logger.error(f"Error evaluating sample: {e}")
        else:
            # Sequential execution for CUDA to avoid OOM or GIL contention with Torch
            for task in tqdm(tasks, desc=f"Evaluating {model_name}"):
                try:
                    _record(evaluate_sample_wrapper(task))
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                
        end_time = time.time()
        runtime = end_time - start_time
        
        for r in results:
            r["runtime"] = runtime / len(results) if len(results) > 0 else 0
            all_results.append(r)

        print(f"\n✅ Completed model: {model_name}")
        print(f"   Samples evaluated: {len(results)}, Runtime: {runtime:.1f}s")
        
        # Build per-sample metrics DataFrame
        # Build per-sample metrics DataFrame with model name
        sample_metrics = []
        for r in results:
            row = {
                "PRI": r.get("pri", 0),
                "ORI": r.get("ori_score", 0),
                "IFI": r.get("ifi_score", 0),
                "Diagnostic_PRI": r.get("diagnostic_pri", 0),
                "Diagnostic_ORI": r.get("diagnostic_ori", 0),
                "Diagnostic_IFI": r.get("diagnostic_ifi", 0),
                "Diagnosis": r.get("diagnosis", ""),
                "CS": r.get("cs", 0),
                "HS": r.get("hs_score", 0),
                "Faithfulness": r.get("faithfulness", 0),
                "Consistency": r.get("consistency", 0),
                "Human_Score": r.get("human_score", 0),
                "Final_Score": r.get("final_score", 0),
            }
            # Include advanced metrics in sample-level CSV
            if config.enable_advanced_metrics:
                row["Faithfulness"] = r.get("faithfulness", 0)
                row["TRD_Semantic"] = r.get("trd_semantic", 0)
                row["KPIG_Advanced"] = r.get("kpig_advanced", 0)
                row["USD"] = r.get("usd", 0)

            # Include ROUGE if enabled
            if config.enable_rouge:
                rouge = r.get("rouge", {})
                row["ROUGE_1"] = rouge.get("rouge1", 0)
                row["ROUGE_2"] = rouge.get("rouge2", 0)
                row["ROUGE_L"] = rouge.get("rougeL", 0)

            sample_metrics.append(row)

        sample_df = pd.DataFrame(sample_metrics)
        sample_df.insert(0, "Model", model_name)

        # Save per-model sample CSV (model name sanitized for filename)
        safe_name = model_name.replace("/", "_")
        sample_df.to_csv(f"results/sample_level_{safe_name}.csv", index=False)

        # Also save/append to combined sample-level CSV
        if last_sample_df.empty:
            last_sample_df = sample_df.copy()
        else:
            last_sample_df = pd.concat([last_sample_df, sample_df], ignore_index=True)
        last_sample_df.to_csv("results/sample_level_results.csv", index=False)

        # Render per-model stability table (TABLE 5)
        print()
        render_stability_table(sample_df)

        # Free model memory before loading the next one
        del model_interface
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # responses.csv (Layer B) and scored_samples.csv (Layer C) were written
    # incrementally above; close the writers (data already flushed+fsync'd).
    resp_writer.close()
    scored_writer.close()
    print(f"💾 CSV layers persisted: {resp_path}, {scored_path}")

    # Flaw §2.2 — per-model IFI normalization. ppl_var and bf live on
    # different scales for every model; without this pass IFI saturates at
    # 1.0 and is useless for cross-row diagnosis. Runs once at the end so
    # incremental rows stay crash-safe.
    csv_io.normalize_ifi_per_model(scored_path)
    print(f"💾 IFI normalized per-model in {scored_path}")

    # Split responses.csv and scored_samples.csv by task (topic_label) so each
    # task's outputs are available as a dedicated file for easy analysis/reuse.
    _save_task_split_csvs(resp_path, scored_path, config.results_dir)

    # =========================================================================
    # STEP 6: Correlation Analysis
    # =========================================================================
    correlation_results = None
    if config.enable_correlation_analysis and len(all_results) >= 3:
        print("\n📊 Running Correlation Analysis...")
        correlation_results = compute_full_correlation_analysis(all_results)

        # Add summary statistics
        summary_stats = compute_metric_summary_statistics(all_results)
        correlation_results["summary_statistics"] = summary_stats

        # Save to results/correlations.json
        corr_path = save_correlation_results(correlation_results, config.results_dir)
        print(f"✅ Correlation results saved to {corr_path}")

        # Render correlation table (TABLE 4) immediately after computation
        render_correlation_table(correlation_results)

    # Save experiment tracking data
    run_details["results"] = all_results
    os.makedirs(config.results_dir, exist_ok=True)
    save_json(run_details, os.path.join(config.results_dir, f"{run_id}.json"))
    
    # Save aggregated output per model specifically mapped
    benchmark_results = []
    for m in config.models:
        model_runs = [r for r in all_results if r["model"] == m]
        if not model_runs:
            continue
        
        agg = {
            "Model": m,
            "PRI": sum(r.get("pri", 0) for r in model_runs) / len(model_runs),
            "ORI": sum(r.get("ori_score", 0) for r in model_runs) / len(model_runs),
            "IFI": sum(r.get("ifi_score", 0) for r in model_runs) / len(model_runs),
            "Diagnostic_PRI": sum(r.get("diagnostic_pri", 0) for r in model_runs) / len(model_runs),
            "Diagnostic_ORI": sum(r.get("diagnostic_ori", 0) for r in model_runs) / len(model_runs),
            "Diagnostic_IFI": sum(r.get("diagnostic_ifi", 0) for r in model_runs) / len(model_runs),
            "CS": sum(r.get("cs", 0) for r in model_runs) / len(model_runs),
            "HS": sum(r.get("hs_score", 0) for r in model_runs) / len(model_runs),
            "Faithfulness": sum(r.get("faithfulness", 0) for r in model_runs) / len(model_runs),
            "Consistency": sum(r.get("consistency", 0) for r in model_runs) / len(model_runs),
            "Human_Score": sum(r.get("human_score", 0) for r in model_runs) / len(model_runs),
            "Final_Score": sum(r.get("final_score", 0) for r in model_runs) / len(model_runs),
            "Avg_Len": sum(r.get("avg_length", 0) for r in model_runs) / len(model_runs),
            "Avg_Cov": sum(r.get("avg_coverage", 0) for r in model_runs) / len(model_runs),
        }

        # Add advanced aggregate metrics
        if config.enable_advanced_metrics:
            agg["Faithfulness"] = sum(r.get("faithfulness", 0) for r in model_runs) / len(model_runs)
            agg["TRD_Semantic"] = sum(r.get("trd_semantic", 0) for r in model_runs) / len(model_runs)
            agg["KPIG_Advanced"] = sum(r.get("kpig_advanced", 0) for r in model_runs) / len(model_runs)
            agg["USD"] = sum(r.get("usd", 0) for r in model_runs) / len(model_runs)

        if config.enable_rouge:
            agg["ROUGE_1"] = sum(r.get("rouge", {}).get("rouge1", 0) for r in model_runs) / len(model_runs)
            agg["ROUGE_2"] = sum(r.get("rouge", {}).get("rouge2", 0) for r in model_runs) / len(model_runs)
            agg["ROUGE_L"] = sum(r.get("rouge", {}).get("rougeL", 0) for r in model_runs) / len(model_runs)

        benchmark_results.append(agg)
        
    df_results = pd.DataFrame(benchmark_results)
    df_results.to_csv(os.path.join(config.results_dir, "benchmark.csv"), index=False)

    return df_results, last_sample_df, correlation_results


def generate_responses_to_csv(config: Config) -> str:
    """GENERATE-ONLY: produce responses.csv with ONLY the subject model loaded.

    No scorer models (embedder/judge/NLI) are loaded, so on a single 80GB card
    the subject never has to co-reside with the 70B judge. Rows are flushed +
    fsync'd per sample (crash-safe) and already-generated (model,instance) pairs
    are skipped on re-run (resume). Returns the responses.csv path.
    """
    df = load_dataset(config.data_path)
    limit = config.max_samples if getattr(config, "max_samples", 0) else len(df)
    samples = df.to_dict("records")[:limit]
    for i, s in enumerate(samples):
        s.setdefault("instance_id", f"idx_{i}")

    os.makedirs(config.results_dir, exist_ok=True)
    resp_path = os.path.join(config.results_dir, "responses.csv")
    done = csv_io.existing_response_keys(resp_path)
    if done:
        print(f"↻ Resuming generation: {len(done)} (model,instance) pairs already on disk")
    writer = csv_io.IncrementalCSVWriter(resp_path, csv_io.RESPONSES_COLS, resume=True)

    for model_name in config.models:
        print(f"\n🚀 Generating responses: {model_name}")
        quant = getattr(config, "subject_quantizations", {}).get(model_name)
        try:
            mi = ModelInterface(model_name, config, quantization=quant)
        except Exception as e:
            logger.error(f"Cannot load model '{model_name}': {e}")
            continue
        for sample in tqdm(samples, desc=f"Generating {model_name}"):
            inst = sample.get("instance_id", "")
            if (model_name, inst) in done:
                continue
            try:
                prompts = sample.get("prompt_variants")
                if not prompts:
                    from .prompt_generator import generate_prompt_variants, flatten_prompt_variants
                    prompts = flatten_prompt_variants(generate_prompt_variants(sample["input_text"]))
                responses = mi.generate_responses(prompts)
                strategies = sample.get("strategies") or [None] * len(prompts)
                # FLAWS §2.2 fix: compute the intra-model diagnostics (PPL
                # variance, branching factor) WHILE the subject model is
                # resident. The downstream score-from-CSV phase has no
                # generation model, so without these IFI saturates at 1.0.
                # Compute once per (model, instance) and replicate across
                # the K variant rows so a partial resume still recovers the
                # value. Wrapped in try/except so a metric-internal error
                # never blocks a successful generation row from being saved.
                try:
                    ppl_var_inst = float(mi.compute_perplexity_variance(prompts, responses))
                except Exception as ifi_e:
                    logger.warning(f"PPL variance failed for {model_name}/{inst}: {ifi_e}")
                    ppl_var_inst = ""
                try:
                    bf_inst = float(mi.compute_branching_factor(responses))
                except Exception as ifi_e:
                    logger.warning(f"Branching factor failed for {model_name}/{inst}: {ifi_e}")
                    bf_inst = ""
                rows = [{
                    "model": model_name, "instance_id": inst,
                    "topic_label": sample.get("topic_label", sample.get("task", "")),
                    "variant_idx": i, "strategy": strategies[i] if i < len(strategies) else None,
                    "prompt": p, "response": r,
                    "input_text": sample.get("input_text", ""),
                    "reference_output": sample.get("reference_output", ""),
                    "ppl_var_inst": ppl_var_inst,
                    "bf_inst": bf_inst,
                } for i, (p, r) in enumerate(zip(prompts, responses))]
                writer.write_rows(rows)   # flush + fsync per sample
            except Exception as e:
                logger.error(f"Error generating {model_name}/{inst}: {e}")
        del mi
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    writer.close()
    print(f"💾 responses.csv written → {resp_path}")
    return resp_path


def score_from_responses_csv(csv_path: str, config: Config) -> pd.DataFrame:
    """Re-evaluate from a persisted responses.csv WITHOUT a generation model.

    Loads pre-generated responses (Layer B), runs only the scoring metrics
    (embedder / judge / NLI / ROUGE), and writes scored_samples.csv
    incrementally (flush+fsync per group, resume-safe). The subject model is
    never loaded — the cheap, crash-safe re-evaluation path.
    """
    grouped = csv_io.read_responses_grouped(csv_path)
    scored_path = os.path.join(config.results_dir, "scored_samples.csv")
    done = csv_io.existing_scored_keys(scored_path)
    pending = {k: v for k, v in grouped.items() if k not in done}
    print(f"Scoring {len(pending)}/{len(grouped)} groups from {csv_path} "
          f"({len(done)} already done) — no generation model")
    os.makedirs(config.results_dir, exist_ok=True)
    writer = csv_io.IncrementalCSVWriter(scored_path, csv_io.SCORED_COLS, resume=True)
    # Load embedder once for all samples — avoids reloading 7B model per sample (OOM).
    from .embeddings import EmbeddingHelper
    shared_embedder = EmbeddingHelper()
    rows = []
    for (model, inst), sample in pending.items():
        try:
            r = evaluate_sample(sample, config, model_interface=None, cache_manager=None, embedder=shared_embedder)
            row = csv_io.scored_row_from_result(r)
            writer.write_rows([row])   # flush + fsync per group
            rows.append(row)
        except Exception as e:
            logger.error(f"Error scoring {model}/{inst}: {e}")
    writer.close()
    print(f"💾 scored_samples.csv updated → {scored_path}")
    return pd.DataFrame(rows)
