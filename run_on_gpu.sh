#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# run_on_gpu.sh — Full GenSens + PRI pipeline for Jarvis Labs H100/A100
#
# QUICK START (Jarvis Labs terminal):
#   git clone <repo> && cd "PROMPT SENSITIVITY"
#   pip install -r requirements_gpu.txt
#   bash run_on_gpu.sh
#
# RUN SPECIFIC PHASES ONLY:
#   bash run_on_gpu.sh --phases 0,1          # instruction pool + dataset only
#   bash run_on_gpu.sh --phases 2,3,4        # benchmark + PIRC + eval only
#   bash run_on_gpu.sh --phases 0            # just generate summarization instructions
#
# OVERRIDE DEFAULTS:
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct N_INSTANCES=100 bash run_on_gpu.sh
#
# PHASES:
#   0 — Generate summarization instruction pool (Qwen 8B, simple→complex)
#   1 — Generate GenSens dataset for all 4 tasks (paraphrasing via vLLM)
#   2 — Run PRI benchmark (model evaluation)
#   3 — Run LL-PIRC experiment (clamping intervention)
#   4 — Run statistical evaluation (Wilcoxon, bootstrap CI, effect sizes)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Defaults — override via environment variables or --flags
# ─────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
N_INSTANCES="${N_INSTANCES:-200}"
N_VARIANTS="${N_VARIANTS:-8}"
N_PER_LEVEL="${N_PER_LEVEL:-25}"         # 25 × 4 levels = 100 instructions
TASKS="${TASKS:-all}"
SEED="${SEED:-42}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.88}"     # adjusted automatically based on VRAM
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PHASES="${PHASES:-0,1,2,3,4}"           # comma-separated list of phases to run

# ─────────────────────────────────────────────────────────────
# Parse CLI args
# ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases)     PHASES="$2";       shift 2 ;;
        --model-id)   MODEL_ID="$2";     shift 2 ;;
        --n-instances) N_INSTANCES="$2"; shift 2 ;;
        --n-variants) N_VARIANTS="$2";   shift 2 ;;
        --tasks)      TASKS="$2";        shift 2 ;;
        --seed)       SEED="$2";         shift 2 ;;
        --help|-h)
            sed -n '3,25p' "$0"          # print the header block
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENSENS="$ROOT/gensens"
ROBUSTNESS="$ROOT/prompt_robustness"
POOL_FILE="$GENSENS/data/summ_instruction_pool.jsonl"

mkdir -p "$GENSENS/data" "$GENSENS/logs" "$ROBUSTNESS/results"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
section() { echo; echo "══════════════════════════════════════════════════════"; \
            echo "  $*"; echo "══════════════════════════════════════════════════════"; }
die()     { echo "FATAL: $*" >&2; exit 1; }

phase_enabled() {
    # Returns 0 (true) if phase $1 is in the PHASES list
    [[ ",$PHASES," == *",$1,"* ]]
}

# ─────────────────────────────────────────────────────────────
# GPU detection — sets GPU_MEM_FRAC and MAX_MODEL_LEN
# ─────────────────────────────────────────────────────────────
detect_gpu() {
    command -v nvidia-smi &>/dev/null || die "nvidia-smi not found — no GPU detected."
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log "GPU: $GPU_NAME  |  VRAM: ${GPU_MEM_MB} MiB"

    if [[ "$GPU_MEM_MB" -lt 45000 ]]; then
        # A100 40 GB or smaller — be conservative
        GPU_MEM_FRAC=0.82
        MAX_MODEL_LEN=2048
        log "  Profile: A100-40GB  →  gpu_mem_frac=$GPU_MEM_FRAC  max_model_len=$MAX_MODEL_LEN"
    elif [[ "$GPU_MEM_MB" -lt 75000 ]]; then
        # A100 80 GB
        GPU_MEM_FRAC=0.88
        MAX_MODEL_LEN=4096
        log "  Profile: A100-80GB  →  gpu_mem_frac=$GPU_MEM_FRAC  max_model_len=$MAX_MODEL_LEN"
    else
        # H100 80 GB (SXM or PCIe)
        GPU_MEM_FRAC=0.90
        MAX_MODEL_LEN=8192
        log "  Profile: H100-80GB  →  gpu_mem_frac=$GPU_MEM_FRAC  max_model_len=$MAX_MODEL_LEN"
    fi
}

# ─────────────────────────────────────────────────────────────
# Phase 0 — Summarization instruction pool (Qwen 8B, simple→complex)
# ─────────────────────────────────────────────────────────────
run_phase0() {
    section "Phase 0 │ Summarization Instruction Pool  [Qwen 8B → $POOL_FILE]"

    if [[ -f "$POOL_FILE" ]]; then
        N_LINES=$(wc -l < "$POOL_FILE")
        log "Pool file already exists ($N_LINES instructions) — skipping."
        log "  Delete $POOL_FILE to force regeneration."
        return 0
    fi

    log "Model: $MODEL_ID"
    log "Generating $((N_PER_LEVEL * 4)) instructions across 4 complexity levels ($N_PER_LEVEL each)..."

    python "$GENSENS/scripts/generate_summ_instructions.py" \
        --backend vllm \
        --model-id "$MODEL_ID" \
        --n-per-level "$N_PER_LEVEL" \
        --gpu-memory-utilization "$GPU_MEM_FRAC" \
        --max-model-len "$MAX_MODEL_LEN" \
        --output "$POOL_FILE"

    log "✓ Phase 0 done — $(wc -l < "$POOL_FILE") instructions saved to $POOL_FILE"
}

# ─────────────────────────────────────────────────────────────
# Phase 1 — GenSens dataset generation (all 4 tasks, paraphrasing)
# ─────────────────────────────────────────────────────────────
run_phase1() {
    section "Phase 1 │ GenSens Dataset Generation  [tasks=$TASKS, instances=$N_INSTANCES, variants=$N_VARIANTS]"

    # Check if all task outputs already exist
    declare -A TASK_FILES=(
        [summarization]="$GENSENS/data/gensens_summarization_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
        [creative]="$GENSENS/data/gensens_creative_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
        [dialogue]="$GENSENS/data/gensens_dialogue_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
        [qa]="$GENSENS/data/gensens_qa_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
    )

    if [[ "$TASKS" == "all" ]]; then
        ALL_DONE=true
        for f in "${TASK_FILES[@]}"; do
            [[ -f "$f" ]] || { ALL_DONE=false; break; }
        done
        if [[ "$ALL_DONE" == "true" ]]; then
            log "All task output files found — skipping dataset generation."
            log "  Delete files in $GENSENS/data/ to force regeneration."
            return 0
        fi
    fi

    log "Model: $MODEL_ID (vLLM backend)"
    log "Summarization will auto-load instruction pool from $POOL_FILE (if present)"

    python "$GENSENS/scripts/generate_dataset.py" \
        --task "$TASKS" \
        --n_instances "$N_INSTANCES" \
        --n_variants "$N_VARIANTS" \
        --seed "$SEED" \
        --model vllm \
        --model-id "$MODEL_ID" \
        --gpu-memory-utilization "$GPU_MEM_FRAC" \
        --max-model-len "$MAX_MODEL_LEN"

    log "✓ Phase 1 done — datasets in $GENSENS/data/"
}

# ─────────────────────────────────────────────────────────────
# Phase 2 — PRI benchmark
# Three subject models (8B/7B, fp16) + 70B AWQ run one at a time.
# Memory-safe two-step: generate responses first (subject model only),
# then score from CSV (judge/NLI only) so they never co-reside on GPU.
# ─────────────────────────────────────────────────────────────
SUBJECT_MODELS="${SUBJECT_MODELS:-meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"

run_phase2() {
    section "Phase 2 │ PRI Benchmark  [models: $SUBJECT_MODELS]"
    [[ -f "$ROBUSTNESS/main.py" ]] || die "main.py not found in $ROBUSTNESS"

    cd "$ROBUSTNESS"

    # Step 2a: generate responses for every subject model (one model at a time,
    # GPU freed between loads) — no judge/NLI loaded here.
    log "Step 2a: Generating responses for all subject models..."
    MODEL_NAME="$SUBJECT_MODELS" python main.py --generate-only

    # Step 2b: score from the persisted responses.csv — no subject model loaded.
    log "Step 2b: Scoring from responses.csv (judge + NLI, no subject model)..."
    MODEL_NAME="$SUBJECT_MODELS" python main.py --responses-csv results/responses.csv

    cd "$ROOT"
    log "✓ Phase 2 done — results in $ROBUSTNESS/results/"
}

# ─────────────────────────────────────────────────────────────
# Phase 3 — LL-PIRC experiment
# ─────────────────────────────────────────────────────────────
run_phase3() {
    section "Phase 3 │ LL-PIRC Experiment"
    [[ -f "$ROBUSTNESS/experiment_pirc.py" ]] || die "experiment_pirc.py not found"

    cd "$ROBUSTNESS"
    python experiment_pirc.py
    cd "$ROOT"
    log "✓ Phase 3 done"
}

# ─────────────────────────────────────────────────────────────
# Phase 4 — Statistical evaluation
# ─────────────────────────────────────────────────────────────
run_phase4() {
    section "Phase 4 │ Statistical Evaluation  [Wilcoxon, Bootstrap CI, Effect Sizes]"
    [[ -f "$ROBUSTNESS/evaluate.py" ]] || die "evaluate.py not found"

    cd "$ROBUSTNESS"
    python evaluate.py
    cd "$ROOT"
    log "✓ Phase 4 done — evaluation report in $ROBUSTNESS/results/"
}

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
main() {
    section "GenSens + PRI Pipeline  │  Jarvis Labs GPU Runner"
    log "Phases to run : $PHASES"
    log "Model         : $MODEL_ID"
    log "Tasks         : $TASKS"
    log "Instances/task: $N_INSTANCES"
    log "Variants/inst : $N_VARIANTS"
    log "Seed          : $SEED"

    detect_gpu

    T_START=$(date +%s)

    phase_enabled 0 && run_phase0
    phase_enabled 1 && run_phase1
    phase_enabled 2 && run_phase2
    phase_enabled 3 && run_phase3
    phase_enabled 4 && run_phase4

    T_END=$(date +%s)
    ELAPSED=$(( T_END - T_START ))
    section "ALL DONE  │  Total time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
    log "Results : $ROBUSTNESS/results/"
    log "Datasets: $GENSENS/data/"
}

main "$@"
