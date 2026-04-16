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
    samples = df.to_dict('records')[:100]
    print(f"Running evaluation on {len(samples)} samples")
    print(f"Models to run: {config.models}")

    cache_manager = CacheManager(config.cache_dir, config.enable_cache)

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
        
        # Load model interface
        try:
            model_interface = ModelInterface(model_name, config)
        except Exception as e:
            logger.error(f"Cannot load model '{model_name}'. Skipping... Error: {e}")
            print(f"Model {model_name} failed. Continuing...")
            continue
            
        # Tasks for multiprocessing
        tasks = [(sample, config, model_interface, cache_manager) for sample in samples]
        
        results = []
        if config.enable_parallel and model_interface.device.type == "cpu":
            # Parallel execution for API models or CPU models
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = [executor.submit(evaluate_sample_wrapper, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model_name}"):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Error evaluating sample: {e}")
        else:
            # Sequential execution for CUDA to avoid OOM or GIL contention with Torch
            for task in tqdm(tasks, desc=f"Evaluating {model_name}"):
                try:
                    results.append(evaluate_sample_wrapper(task))
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
                "CS": r.get("cs", 0),
                "HS": r.get("hs_score", 0),
                "Consistency": r.get("consistency", 0),
                "Human_Score": r.get("human_score", 0),
                "Final_Score": r.get("final_score", 0),
            }
            # Include advanced metrics in sample-level CSV
            if config.enable_advanced_metrics:
                row["SMS_Wasserstein"] = r.get("sms_wasserstein", 0)
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
            "CS": sum(r.get("cs", 0) for r in model_runs) / len(model_runs),
            "HS": sum(r.get("hs_score", 0) for r in model_runs) / len(model_runs),
            "Consistency": sum(r.get("consistency", 0) for r in model_runs) / len(model_runs),
            "Human_Score": sum(r.get("human_score", 0) for r in model_runs) / len(model_runs),
            "Final_Score": sum(r.get("final_score", 0) for r in model_runs) / len(model_runs),
            "Avg_Len": sum(r.get("avg_length", 0) for r in model_runs) / len(model_runs),
            "Avg_Cov": sum(r.get("avg_coverage", 0) for r in model_runs) / len(model_runs),
        }

        # Add advanced aggregate metrics
        if config.enable_advanced_metrics:
            agg["SMS_Wasserstein"] = sum(r.get("sms_wasserstein", 0) for r in model_runs) / len(model_runs)
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

