<#
.SYNOPSIS
  End-to-end crossed-design prompt-sensitivity benchmark on native Windows
  (RTX 4090, no vLLM) — HF-transformers backend.

.DESCRIPTION
  One-shot runner: generates the balanced 10-paraphrase summarization dataset
  (if missing), then runs all 6 crossed phases for each of 3 subject models
  (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B), then the cross-model comparison and
  robustness validation. Every step is resumable — re-run the script and it
  continues from the last checkpoint.

.NOTES
  Prereqs (see gensens/crossed/RUN_ON_RTX4090.md):
    - Python venv active, deps installed (WITHOUT vllm):
        pip install torch transformers accelerate sentence-transformers datasets `
                    rouge-score scikit-learn scipy nltk bert-score
        pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
    - huggingface-cli login   (gated Llama weights)

.EXAMPLE
  # defaults (30 articles, greedy, HF micro-batch 4):
  powershell -ExecutionPolicy Bypass -File gensens\crossed\run_crossed_windows.ps1

.EXAMPLE
  # more articles + skip the OOM-prone PPL reload:
  $env:N_ARTICLES=100; $env:SKIP_PPL_ENTROPY="1"; `
    powershell -ExecutionPolicy Bypass -File gensens\crossed\run_crossed_windows.ps1
#>

$ErrorActionPreference = "Stop"

# ── Paths ────────────────────────────────────────────────────────────────────
$Crossed = $PSScriptRoot
$Gensens = Split-Path -Parent $Crossed
$Scripts = Join-Path $Crossed "scripts"
$Dataset = Join-Path $Gensens "data\gensens_summ_50seed_10para.jsonl"
$Py      = if ($env:PYTHON) { $env:PYTHON } else { "python" }

# ── Config (override via environment) ────────────────────────────────────────
$NArticles    = if ($env:N_ARTICLES)      { $env:N_ARTICLES }      else { 30 }
$MaxTokens    = if ($env:MAX_TOKENS)      { $env:MAX_TOKENS }      else { 512 }
$HfBatch      = if ($env:HF_BATCH)        { $env:HF_BATCH }        else { 4 }
$CheckptEvery = if ($env:CHECKPOINT_EVERY){ $env:CHECKPOINT_EVERY} else { 200 }
$MaxModelLen  = if ($env:MAX_MODEL_LEN)   { $env:MAX_MODEL_LEN }   else { 8192 }
$SkipPpl      = $env:SKIP_PPL_ENTROPY     # "1" to skip the PPL/branching pass
$FaithWhole   = $env:FAITH_WHOLE_SUMMARY  # "1" for ~4x faster MiniCheck

$Models = @(
    @{ Id = "meta-llama/Meta-Llama-3.1-8B-Instruct"; Dir = "results_llama"   },
    @{ Id = "Qwen/Qwen2.5-7B-Instruct";              Dir = "results_qwen_new" },
    @{ Id = "mistralai/Mistral-7B-Instruct-v0.3";    Dir = "results_mistral" }
)

function Section($msg) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor Cyan
}

# ── Step 1 — paraphrase dataset (balanced 5-axis x 2 = 10/seed) ──────────────
if (-not (Test-Path $Dataset)) {
    Section "Step 1 — Generate balanced 10-paraphrase summarization dataset (Llama-8B, HF)"
    & $Py (Join-Path $Gensens "scripts\generate_summ_paraphrases.py") --backend llama
    if ($LASTEXITCODE -ne 0) { throw "Paraphrase generation failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "Dataset already present: $Dataset (skipping Step 1)" -ForegroundColor Green
}

# ── Step 2 — per-model crossed pipeline (6 phases) ───────────────────────────
foreach ($m in $Models) {
    $Id  = $m.Id
    $R   = Join-Path $Crossed $m.Dir
    $Logs = Join-Path $R "logs"
    New-Item -ItemType Directory -Force -Path $Logs | Out-Null

    Section "MODEL: $Id  ->  $($m.Dir)  (N_ARTICLES=$NArticles)"

    # Phase 1 — inference (HF backend, greedy, resumable)
    & $Py (Join-Path $Scripts "run_inference_crossed.py") `
        --model $Id --n_articles $NArticles --max_tokens $MaxTokens `
        --max_model_len $MaxModelLen --results_dir $R `
        --backend hf --hf_batch_size $HfBatch --checkpoint_every $CheckptEvery
    if ($LASTEXITCODE -ne 0) { throw "Phase 1 failed for $Id (exit $LASTEXITCODE)" }

    # Phase 2 — per-cell metrics (MiniCheck faithfulness + BERTScore)
    $p2 = @(
        "--model", $Id, "--results_dir", $R,
        "--faithfulness_backend", "minicheck",
        "--faithfulness_model", "deberta-v3-large",
        "--bertscore_model", "roberta-large"
    )
    if ($SkipPpl -eq "1")    { $p2 += "--skip_ppl_entropy" }
    if ($FaithWhole -eq "1") { $p2 += "--faith_whole_summary" }
    & $Py (Join-Path $Scripts "compute_cell_metrics.py") @p2
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 failed for $Id (exit $LASTEXITCODE)" }

    # Phase 3 — composite + hierarchical aggregation
    & $Py (Join-Path $Scripts "aggregate_scores.py") --results_dir $R
    if ($LASTEXITCODE -ne 0) { throw "Phase 3 failed for $Id (exit $LASTEXITCODE)" }

    # Phase 4 — diagnosis 2x2
    & $Py (Join-Path $Scripts "diagnosis_matrix.py") --results_dir $R
    if ($LASTEXITCODE -ne 0) { throw "Phase 4 failed for $Id (exit $LASTEXITCODE)" }

    # Phase 5 — significance (+ dimension analysis; non-fatal if statsmodels absent)
    & $Py (Join-Path $Scripts "significance.py") --results_dir $R
    if ($LASTEXITCODE -ne 0) { throw "Phase 5 (significance) failed for $Id (exit $LASTEXITCODE)" }
    & $Py (Join-Path $Scripts "dimension_analysis.py") --results_dir $R
    if ($LASTEXITCODE -ne 0) { Write-Host "  dimension_analysis skipped (pip install statsmodels for the mixed-effects model)" -ForegroundColor Yellow }

    # Phase 6 — summary report
    & $Py (Join-Path $Scripts "summary_report.py") --results_dir $R
    if ($LASTEXITCODE -ne 0) { throw "Phase 6 failed for $Id (exit $LASTEXITCODE)" }

    Write-Host "DONE $Id -> see $R\summary_report.md" -ForegroundColor Green
}

# ── Step 3 — cross-model comparison + robustness ─────────────────────────────
Section "Step 3 — Cross-model comparison + robustness validation"
$L = Join-Path $Crossed "results_llama"
$Q = Join-Path $Crossed "results_qwen_new"
$M = Join-Path $Crossed "results_mistral"

& $Py (Join-Path $Scripts "compare_models.py") --run_a $L --label_a Llama-3.1-8B --run_b $Q --label_b Qwen2.5-7B
& $Py (Join-Path $Scripts "compare_models.py") --run_a $L --label_a Llama-3.1-8B --run_b $M --label_b Mistral-7B
& $Py (Join-Path $Scripts "compare_models.py") --run_a $Q --label_a Qwen2.5-7B  --run_b $M --label_b Mistral-7B

foreach ($d in @($L, $Q, $M)) {
    & $Py (Join-Path $Scripts "validate_robustness.py") --results_dir $d
}

Section "ALL DONE — per-model reports in results_*/summary_report.md; comparisons printed above."
