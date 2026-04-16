# Prompt Robustness Evaluation Platform - Methods Documentation

This document provides a detailed technical overview of the methods and metrics implemented in the Prompt Robustness Evaluation Platform.

## 1. Core Pipeline and Orchestration

### `main.py`
- **`main()`**: The entry point of the application. It parses command-line arguments (models, dataset, cache/parallel flags), initializes the `Config`, and triggers the `benchmark_models` function. It also handles post-evaluation tasks like generating static visualizations (bar plots) and saving results to CSV.

### `src/benchmark.py`
- **`benchmark_models(config)`**: Coordinates the evaluation of multiple models. 
    - Loads the dataset.
    - Iterates through the list of models.
    - Initializes a `ModelInterface` for each model.
    - Uses `ThreadPoolExecutor` for parallel execution (for API-based or CPU models) or sequential execution (for GPU models to avoid OOM).
    - Aggregates results per model and calculates mean/std for metric stability.
    - Saves both per-sample and aggregated results.

### `src/evaluator.py`
- **`evaluate_sample(sample, config, model_interface, cache_manager)`**: The core evaluation logic for a single data point.
    1. **Prompt Generation**: Calls `generate_prompt_variants` to create multiple versions of the input.
    2. **Response Generation**: Generates model responses, utilizing `CacheManager` to avoid redundant computations.
    3. **Metric Computation**: Calculates a suite of metrics (SMS, AUC-E, TRD, KPIG, PPL Variance, Branching Factor).
    4. **Score Normalization & Penalties**:
        - Adjusts Correctness Score (CS) based on KPIG and Hallucination Score (HS).
        - Computes the **Prompt Robustness Index (PRI)** as a weighted combination of consistency, correctness, and hallucination resistance.
    5. **LLM-as-a-Judge**: Uses a T5-based model (`llm_judge`) to provide a "human-like" quality score (1-5).
    6. **Final Score**: Combines PRI (60%) and Human Score (40%).
    7. **Attribution**: Uses `compute_attribution_matrix` to classify the robustness type (e.g., "True Robustness", "Stochastic Luck").

---

## 2. Metric Implementations

### `src/pri_calculator.py`
- **`compute_pri(metrics, weights)`**: Calculates the **Prompt Robustness Index (PRI)**. It uses a **weighted harmonic mean** of normalized metrics. This approach ensures that a failure in any single critical metric significantly penalizes the overall robustness score.

### `src/sms_metric.py`
- **`compute_sms_metric(embeddings, alpha=0.5)`**: **Semantic Manifold Stability (SMS)**. 
    - Measures the semantic similarity (cosine similarity) between multiple response embeddings.
    - Subtracts a **diversity penalty** (variance of embeddings) to penalize inconsistent or drifting responses.

### `src/auc_e_metric.py`
- **`compute_auc_e_metric(reference, responses)`**: **Performance Elasticity (AUC-E)**.
    - Simulates input perturbation by truncating the reference output at levels (0.1, 0.2, 0.3).
    - Measures how much the model's response deviates from these "imperfect" references.
    - Returns the Area Under the Curve (AUC) using the trapezoidal rule, indicating how robust the model is to slightly varying reference expectations.

### `src/trd_metric.py`
- **`compute_trd_metric(responses)`**: **Thematic Robustness Drift (TRD)**.
    - Calculates the variance of the response lengths relative to the mean length.
    - High variance indicates significant drift in output style or detail level across prompts.

### `src/kpig_metric.py`
- **`compute_kpig_metric(prompts, responses)`**: **Key Point Information Gain (KPIG)**.
    - Extracts "facts" (words > 4 chars) from all responses.
    - Measures what percentage of the "total unique facts" discovered across all runs are preserved in each individual response.

### `src/correctness_metric.py`
- **`compute_correctness(responses, reference, embeddings, embedder)`**: **Correctness Score (CS)**.
    - Combines **Semantic Similarity** (cosine similarity with reference) and **Keyword Coverage** (intersection of named entities).
    - Penalizes scores if entity coverage is below 20%.

### `src/hallucination_metric.py`
- **`compute_hallucination_score(responses, input_text)`**: **Hallucination Score (HS)**.
    - Identifies "facts" in the response that are not present in the original input text.
    - The score is the ratio of hallucinated facts to total facts in the response.

### `src/ppl_variance.py` & `src/branching_factor.py`
- **`compute_ppl_variance`**: Proxy for model confidence; measures the variance in Perplexity (PPL) across responses.
- **`compute_branching_factor`**: Estimates the model's "uncertainty" during generation based on token-level entropy.

---

## 3. Utility and Helper Methods

### `src/model_interface.py`
- **`ModelInterface`**: A wrapper for HuggingFace Transformers.
    - Handles model/tokenizer loading and device placement (CPU/MPS/CUDA).
    - **`generate_responses`**: Performs batch inference with configurable temperature and sampling.
    - **`compute_perplexity_variance`**: Calculates PPL by passing the response back through the model.
    - **`compute_branching_factor`**: Computes average token entropy and converts it to a branching factor.

### `src/prompt_generator.py`
- **`generate_prompt_variants`**: Creates structured prompts based on the **ORI (Observable Robustness Index)** framework.
    - **d1 (Literal)**: Low perturbation (e.g., "Summarize the text:").
    - **d2 (Stylistic)**: Medium perturbation (e.g., "Condense the ideas:").
    - **d3 (Creative)**: High perturbation (e.g., "Produce a neutral summary:").

### `src/llm_judge.py`
- **`llm_judge`**: Uses `google/flan-t5-base` as a zero-shot evaluator to score summaries from 1 to 5.
- **`parse_judge_output`**: Extracts the numeric score from the judge's text output and normalizes it to [0, 1].

### `src/cache_manager.py`
- **`CacheManager`**: Implements a disk-based JSON cache.
    - Keys are generated using deterministic MD5 hashing of prompts and model names.
    - Saves significant time and cost by avoiding redundant LLM calls.

### `src/scoring_aggregator.py`
- **`ScoringAggregator`**: Provides alternative mathematical methods (Arithmetic, Geometric, Harmonic) for aggregating individual metric scores.

### `src/attribution_matrix.py`
- **`compute_attribution_matrix`**: Classifies the evaluation result into one of four categories:
    - **True Robustness**: High scores across all metrics.
    - **Evaluation Artifact**: Very low scores in any metric.
    - **Stochastic Luck**: High PRI but inconsistent underlying metrics.
    - **Knowledge Boundary**: Moderate results.

### `src/data_loader.py`
- **`load_dataset`**: Loads and validates the input JSON dataset using Pandas.

### `src/embeddings.py`
- **`EmbeddingHelper`**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate vector representations of text for semantic similarity calculations.
