#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# run_crossed_h100.sh — Crossed-design GenSens summarization
# prompt-sensitivity benchmark, end-to-end, for an H100 (Jarvis Labs
# or equivalent single-GPU box). Mirrors ../../run_on_gpu.sh conventions.
#
# QUICK START (on the GPU box, from the repo root):
#   git clone <repo> && cd "PROMPT SENSITIVITY"
#   pip install -r requirements_gpu.txt -r gensens/crossed/requirements_crossed.txt
#   huggingface-cli login   # needed for gated Llama-3.1 weights
#   bash gensens/crossed/run_crossed_h100.sh
#
# OVERRIDE DEFAULTS:
#   MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct N_ARTICLES=100 \
#     bash gensens/crossed/run_crossed_h100.sh
#
# RUN SPECIFIC PHASES ONLY:
#   bash gensens/crossed/run_crossed_h100.sh --phases 1        # inference only
#   bash gensens/crossed/run_crossed_h100.sh --phases 2,3,4,5  # metrics onward
#
# PHASES:
#   1 — run_inference_crossed.py   (vLLM, greedy, resumable)
#   2 — compute_cell_metrics.py    (SMS/CS/Faith/PPL/PC/ROUGE/BERTScore)
#   3 — aggregate_scores.py        (Sensitivity + PRI, 3 levels)
#   4 — diagnosis_matrix.py        (quality x sensitivity 2x2)
#   5 — significance.py            (Mann-Whitney, Wilcoxon)
#   6 — summary_report.py          (prints + saves summary_report.md)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
N_ARTICLES="${N_ARTICLES:-100}"
MAX_TOKENS="${MAX_TOKENS:-512}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.88}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
PHASES="${PHASES:-1,2,3,4,5,6}"
# PPL/branching-factor pass. DEFAULT = 0 (RUN it) for the production benchmark:
# it yields the full 6-metric composite and activates the de-confounded pc_stab_cv
# (method.md 2026-08-09). Costs ~30 min + reloads the 8B model (the most OOM-prone
# step — scorer models are now freed before it loads). Set to 1 to skip for a
# faster/cheaper 4-metric run (ppl_var/pc_stab_var correlate r=0.68 and did not
# change the 30-article conclusion — method.md 2026-07-14).
SKIP_PPL_ENTROPY="${SKIP_PPL_ENTROPY:-0}"
# Faithfulness + correctness scorers (method.md 2026-08-09). Defaults use the
# best deterministic scorers: MiniCheck (deberta-v3-large) for faithfulness and
# BERTScore for correctness-vs-gold. Fall back to the NLI backend by setting
# FAITHFULNESS_BACKEND=nli if the minicheck package can't be installed.
FAITHFULNESS_BACKEND="${FAITHFULNESS_BACKEND:-minicheck}"
FAITHFULNESS_MODEL="${FAITHFULNESS_MODEL:-deberta-v3-large}"
BERTSCORE_MODEL="${BERTSCORE_MODEL:-roberta-large}"
# MiniCheck per-sentence scoring is ~4x slower (each summary -> many doc/claim
# pairs). FAITH_WHOLE_SUMMARY=1 scores each summary as ONE claim: same SOTA
# MiniCheck model, ~4x faster, minimal granularity loss on short summaries.
FAITH_WHOLE_SUMMARY="${FAITH_WHOLE_SUMMARY:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases)     PHASES="$2";     shift 2 ;;
        --model-id)   MODEL_ID="$2";   shift 2 ;;
        --n-articles) N_ARTICLES="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,25p' "$0"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
mkdir -p "$RESULTS" "$LOGS"

log()     { echo "[$(date '+%H:%M:%S')] $*"; }
section() { echo; echo "══════════════════════════════════════════════════════"; \
            echo "  $*"; echo "══════════════════════════════════════════════════════"; }
die()     { echo "FATAL: $*" >&2; exit 1; }
has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

command -v python3 >/dev/null || die "python3 not found"

if [[ -f "$ROOT/../../.env" ]]; then
    set -a; source "$ROOT/../../.env"; set +a
    log "Sourced .env"
fi

section "Config: MODEL_ID=$MODEL_ID  N_ARTICLES=$N_ARTICLES  PHASES=$PHASES"

if has_phase 1; then
    section "Phase 1 — Crossed-design inference (vLLM, greedy)"
    python3 "$SCRIPTS/run_inference_crossed.py" \
        --model "$MODEL_ID" --n_articles "$N_ARTICLES" \
        --max_tokens "$MAX_TOKENS" --gpu_mem_frac "$GPU_MEM_FRAC" \
        --checkpoint_every "$CHECKPOINT_EVERY" --results_dir "$RESULTS" \
        2>&1 | tee -a "$LOGS/phase1_inference.log"
fi

if has_phase 2; then
    section "Phase 2 — Per-cell sensitivity/quality metrics"
    EXTRA_ARGS=()
    if [[ "$SKIP_PPL_ENTROPY" == "1" ]]; then
        EXTRA_ARGS+=(--skip_ppl_entropy)
    fi
    if [[ "$FAITH_WHOLE_SUMMARY" == "1" ]]; then
        EXTRA_ARGS+=(--faith_whole_summary)
    fi
    python3 "$SCRIPTS/compute_cell_metrics.py" --model "$MODEL_ID" --results_dir "$RESULTS" \
        --faithfulness_backend "$FAITHFULNESS_BACKEND" --faithfulness_model "$FAITHFULNESS_MODEL" \
        --bertscore_model "$BERTSCORE_MODEL" "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$LOGS/phase2_cell_metrics.log"
fi

if has_phase 3; then
    section "Phase 3 — Composite scores + hierarchical aggregation"
    python3 "$SCRIPTS/aggregate_scores.py" --results_dir "$RESULTS" 2>&1 | tee -a "$LOGS/phase3_aggregate.log"
fi

if has_phase 4; then
    section "Phase 4 — Diagnosis matrix"
    python3 "$SCRIPTS/diagnosis_matrix.py" --results_dir "$RESULTS" 2>&1 | tee -a "$LOGS/phase4_diagnosis.log"
fi

if has_phase 5; then
    section "Phase 5 — Significance tests"
    python3 "$SCRIPTS/significance.py" --results_dir "$RESULTS" 2>&1 | tee -a "$LOGS/phase5_significance.log"
    # Dimension effect: mixed-effects model + quality-residualized sensitivity.
    # Non-fatal if statsmodels is missing — the seed-aggregated Kruskal in
    # significance.json still covers the dimension test.
    python3 "$SCRIPTS/dimension_analysis.py" --results_dir "$RESULTS" 2>&1 | tee -a "$LOGS/phase5_significance.log" \
        || log "dimension_analysis.py skipped (install statsmodels for the mixed-effects model)"
fi

if has_phase 6; then
    section "Phase 6 — Summary report"
    python3 "$SCRIPTS/summary_report.py" --results_dir "$RESULTS" 2>&1 | tee -a "$LOGS/phase6_report.log"
fi

section "Done. See $RESULTS/summary_report.md"
