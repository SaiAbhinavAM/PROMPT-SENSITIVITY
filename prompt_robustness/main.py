"""
main.py — Entry point for the Prompt Robustness Evaluation Platform.

Supports CLI arguments for model selection, dataset path, caching, and parallelism.
After benchmark completion, renders publication-quality tables and saves static plots.
"""

import argparse
import logging
from src.config import Config
from src.benchmark import benchmark_models
from src.table_utils import (
    render_performance_table,
    render_advanced_metrics_table,
    render_rouge_table,
    render_correlation_table,
    render_stability_table,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Prompt Robustness Evaluation Platform")
    parser.add_argument("--models", type=str, nargs='+', help="List of models to benchmark", default=None)
    parser.add_argument("--dataset", type=str, help="Path to json dataset", default=None)
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--dynamic-weighting", action="store_true", help="Enable dynamic Final_Score weighting based on USD")
    parser.add_argument("--no-advanced", action="store_true", help="Disable advanced metrics")
    parser.add_argument("--enable-bertscore", action="store_true", help="Enable BERTScore computation")
    parser.add_argument("--no-rouge", action="store_true", help="Disable ROUGE computation")
    parser.add_argument("--no-correlation", action="store_true", help="Disable correlation analysis")
    
    args = parser.parse_args()
    config = Config()
    
    if args.models:
        config.models = args.models
    if args.dataset:
        config.data_path = args.dataset
    if args.no_cache:
        config.enable_cache = False
    if args.no_parallel:
        config.enable_parallel = False
    if args.dynamic_weighting:
        config.enable_dynamic_weighting = True
    if args.no_advanced:
        config.enable_advanced_metrics = False
    if args.enable_bertscore:
        config.enable_bertscore = True
    if args.no_rouge:
        config.enable_rouge = False
    if args.no_correlation:
        config.enable_correlation_analysis = False
        
    print(f"🚀 Starting Benchmark Pipeline...")
    print(f"Models: {config.models}")
    print(f"Dataset: {config.data_path}")
    print(f"Cache Enabled: {config.enable_cache}")
    print(f"Advanced Metrics: {config.enable_advanced_metrics}")
    print(f"Dynamic Weighting: {config.enable_dynamic_weighting}")
    print(f"ROUGE: {config.enable_rouge}")
    print(f"BERTScore: {config.enable_bertscore}")
    print(f"Correlation Analysis: {config.enable_correlation_analysis}")
    
    # Run benchmark — returns (aggregated_df, sample_df, correlation_results)
    results_df, sample_df, correlation_results = benchmark_models(config)
    
    if not results_df.empty and "Model" in results_df.columns:
        # ====================================================================
        # Render structured tables (replaces raw log output)
        # ====================================================================

        # TABLE 1: Model Performance Summary
        render_performance_table(results_df)

        # TABLE 2: Advanced Metrics
        render_advanced_metrics_table(results_df)

        # TABLE 3: ROUGE Scores
        render_rouge_table(results_df)

        # Save full CSV
        results_df.to_csv("results/final_results.csv", index=False)
        
        # ========================
        # STATIC VISUALIZATIONS
        # ========================
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs("results/plots", exist_ok=True)
        
        # 1. Bar Plot - Final Score
        plt.figure()
        df_sorted = results_df.sort_values(by="Final_Score", ascending=False)
        plt.bar(df_sorted["Model"], df_sorted["Final_Score"], color="#4CAF50")
        plt.xticks(rotation=30, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Final Score")
        plt.title("Model Ranking (Final Score)")
        plt.tight_layout()
        plt.savefig("results/plots/final_score.png")
        plt.close()
        
        # 2. Bar Plot - PRI vs Human
        plt.figure()
        x = range(len(results_df))
        plt.bar(x, results_df["PRI"], width=0.4, label="PRI", color="#2196F3")
        plt.bar([i + 0.4 for i in x], results_df["Human_Score"], width=0.4, label="Human", color="#FF9800")
        plt.xticks([i + 0.2 for i in x], results_df["Model"], rotation=30, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("PRI vs Human Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plots/pri_vs_human.png")
        plt.close()
        
        # 3. Bar Plot - Metric Breakdown
        metrics = ["PRI", "CS", "HS", "Consistency"]
        for metric in metrics:
            plt.figure()
            plt.bar(results_df["Model"], results_df[metric], color="#9C27B0" if metric=="HS" else "#3F51B5")
            plt.xticks(rotation=30, ha="right")
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.title(f"{metric} Comparison")
            plt.tight_layout()
            plt.savefig(f"results/plots/{metric.lower()}.png")
            plt.close()

        # 4. Advanced metrics plots (if available)
        adv_plot_metrics = ["SMS_Wasserstein", "TRD_Semantic", "KPIG_Advanced", "USD"]
        for metric in adv_plot_metrics:
            if metric in results_df.columns:
                plt.figure()
                plt.bar(results_df["Model"], results_df[metric], color="#009688")
                plt.xticks(rotation=30, ha="right")
                plt.xlabel("Model")
                plt.ylabel(metric)
                plt.title(f"{metric} Comparison")
                plt.tight_layout()
                plt.savefig(f"results/plots/{metric.lower()}.png")
                plt.close()

        # 5. ROUGE plots (if available)
        rouge_cols = [c for c in ["ROUGE_1", "ROUGE_2", "ROUGE_L"] if c in results_df.columns]
        if rouge_cols:
            plt.figure(figsize=(10, 5))
            bar_width = 0.25
            x_pos = range(len(results_df))
            colors = ["#E91E63", "#FF5722", "#FF9800"]
            for j, col in enumerate(rouge_cols):
                plt.bar([i + j * bar_width for i in x_pos], results_df[col],
                        width=bar_width, label=col, color=colors[j % len(colors)])
            plt.xticks([i + bar_width for i in x_pos], results_df["Model"], rotation=30, ha="right")
            plt.xlabel("Model")
            plt.ylabel("ROUGE Score")
            plt.title("ROUGE Score Comparison")
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/plots/rouge_comparison.png")
            plt.close()
            
        print("\n📊 Plots saved in results/plots/")
    else:
        print("⚠️ No results generated. Check earlier errors.")
    print("\nTo view the dashboard, run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
