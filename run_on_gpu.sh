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
# Note: MODEL_ID / GENERATOR_QUANTIZATION are resolved AFTER sourcing .env
# so the values in .env (GENERATOR_MODEL=..., GENERATOR_QUANTIZATION=...)
# take precedence over the hardcoded fallback.
# ─────────────────────────────────────────────────────────────
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
CLI_MODEL_ID=""    # set if --model-id passed; wins over env / fallback
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases)     PHASES="$2";       shift 2 ;;
        --model-id)   CLI_MODEL_ID="$2"; shift 2 ;;
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

# ─────────────────────────────────────────────────────────────
# Load .env so HF_TOKEN, MODEL_NAME, JUDGE_MODEL, EMBEDDER_MODEL, etc.
# propagate to every Python subprocess (vLLM / HF / huggingface-cli use
# HF_TOKEN to download gated weights — without this, Phase 1 fails with
# "401 Unauthorized" on the Llama-3.1 download).
# Skip when CI=1 (CI provides its own env).
# ─────────────────────────────────────────────────────────────
ENV_FILE="$ROOT/.env"
if [[ "${CI:-}" != "1" && -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    log "Sourced $ENV_FILE (HF_TOKEN $( [[ -n "${HF_TOKEN:-}" ]] && echo set || echo MISSING ))"
elif [[ "${CI:-}" != "1" ]]; then
    log "WARNING: $ENV_FILE not found — gated model downloads (Llama-3.1) will fail. Create .env from .env.example."
fi

# Force vLLM workers to use `spawn` instead of `fork`. Even with the
# vLLM-before-SBERT ordering in gensens/scripts/paraphrase_generator.py,
# fork can still inherit CUDA state from any other library (e.g. transformers
# import path) and crash workers with "Cannot re-initialize CUDA in forked
# subprocess". `spawn` is the safe default; vLLM 0.8+ supports it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Mirror HF_TOKEN to HUGGING_FACE_HUB_TOKEN so older huggingface_hub releases
# (used transitively by some vllm builds) pick it up too.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ─────────────────────────────────────────────────────────────
# Phase 0 / Phase 1 generator selection (resolved after .env load)
#   precedence: --model-id CLI > $MODEL_ID env > $GENERATOR_MODEL (.env)
#               > Qwen 7B fallback
#   GENERATOR_QUANTIZATION (.env) drives --quantization for AWQ generator.
# ─────────────────────────────────────────────────────────────
if [[ -n "$CLI_MODEL_ID" ]]; then
    MODEL_ID="$CLI_MODEL_ID"
else
    MODEL_ID="${MODEL_ID:-${GENERATOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
fi
QUANT_ARG=""
if [[ -n "${GENERATOR_QUANTIZATION:-}" ]]; then
    QUANT_ARG="--quantization ${GENERATOR_QUANTIZATION}"
fi
log "Generator: $MODEL_ID  ${QUANT_ARG:+($GENERATOR_QUANTIZATION)}"

phase_enabled() {
    # Returns 0 (true) if phase $1 is in the PHASES list
    [[ ",$PHASES," == *",$1,"* ]]
}

# ─────────────────────────────────────────────────────────────
# Dependency check — reinstall if core packages missing (survives pause/resume)
# ─────────────────────────────────────────────────────────────
ensure_torch() {
    # Verify torch is installed AND can actually run a kernel on the GPU.
    # PyTorch 2.11.0+cu130 (the current pip default) needs driver >= 570.124;
    # older A100 instances ship 570.86 and crash with "NVIDIA driver too old".
    # If a real CUDA op fails, force-install 2.6.0+cu124 (broad driver support).
    if python -c "import torch; assert torch.cuda.is_available(); x = torch.zeros(1).cuda(); _ = (x + 1).cpu()" 2>/dev/null; then
        return 0
    fi
    log "torch missing or incompatible with installed NVIDIA driver — installing torch 2.6.0+cu124..."
    pip install --force-reinstall 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' 'torchaudio==2.6.0' \
        --index-url https://download.pytorch.org/whl/cu124 \
        >> /tmp/pip_ensure.log 2>&1 \
        || die "torch install failed — check /tmp/pip_ensure.log"
    # Sanity-check the new install before declaring success.
    python -c "import torch; assert torch.cuda.is_available(); x = torch.zeros(1).cuda(); _ = (x + 1).cpu()" 2>/dev/null \
        || die "torch installed but CUDA still unavailable — driver may be < 525 (cu124 minimum). Check nvidia-smi."
    log "  torch 2.6.0+cu124 verified on GPU."
}

ensure_deps() {
    ensure_torch
    if python -c "import sentence_transformers, vllm, datasets, awq" 2>/dev/null; then
        log "Dependencies OK."
        return 0
    fi
    # Pick a vLLM version that matches the *installed* torch.
    #   torch 2.6.x (cu124) → vllm 0.8.x  (vllm 0.21+ pins torch==2.11)
    #   torch 2.11.x (cu130) → vllm >=0.21 (default; matches torch pin)
    # Wrong combo triggers a transitive torch downgrade → broken install.
    local torch_major torch_minor vllm_spec
    torch_major=$(python -c "import torch,sys; v=torch.__version__.split('+')[0]; print(v.split('.')[0])" 2>/dev/null || echo 0)
    torch_minor=$(python -c "import torch,sys; v=torch.__version__.split('+')[0]; print(v.split('.')[1])" 2>/dev/null || echo 0)
    if [[ "$torch_major" == "2" && "$torch_minor" -lt 10 ]]; then
        vllm_spec="vllm>=0.8.0,<0.9.0"
    else
        vllm_spec="vllm>=0.21.0"
    fi
    log "Core packages missing — installing ($vllm_spec to match torch ${torch_major}.${torch_minor})..."
    pip install sentence-transformers "$vllm_spec" accelerate datasets autoawq \
        rouge-score scipy nltk scikit-learn python-dotenv tqdm pyyaml tabulate \
        >> /tmp/pip_ensure.log 2>&1 \
        && log "  Packages installed." \
        || die "pip install failed — check /tmp/pip_ensure.log"
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
        $QUANT_ARG \
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
        --max-model-len "$MAX_MODEL_LEN" \
        $QUANT_ARG

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

    # ── Merge all 4 task JONSLs into a single combined dataset ──────────────
    # Phase 1 writes one file per task; Phase 2 needs all tasks in one file so
    # every task is benchmarked in a single run. If any task file is missing we
    # fall back to whatever is available (then to sample_dataset.json).
    COMBINED="$GENSENS/data/gensens_all_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
    declare -a TASK_JSONLS=()
    for task in summarization creative dialogue qa; do
        f="$GENSENS/data/gensens_${task}_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
        [[ -f "$f" ]] && TASK_JSONLS+=("$f")
    done

    if [[ ${#TASK_JSONLS[@]} -gt 0 ]]; then
        cat "${TASK_JSONLS[@]}" > "$COMBINED"
        N_COMBINED=$(wc -l < "$COMBINED")
        log "Combined ${#TASK_JSONLS[@]} task datasets → $COMBINED  ($N_COMBINED instances)"
        DATASET_FLAG="--dataset $COMBINED"
    else
        # Fall back: any summarization JSONL present?
        FALLBACK=$(ls -t "$GENSENS/data"/gensens_summarization_*var.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$FALLBACK" && -f "$FALLBACK" ]]; then
            log "WARNING: Only summarization dataset found — using $FALLBACK"
            DATASET_FLAG="--dataset $FALLBACK"
        else
            log "WARNING: No GenSens dataset found — falling back to sample_dataset.json"
            DATASET_FLAG=""
        fi
    fi

    # Step 2a: generate responses for every subject model (one model at a time,
    # GPU freed between loads) — no judge/NLI loaded here.
    log "Step 2a: Generating responses for all subject models..."
    MODEL_NAME="$SUBJECT_MODELS" python main.py --generate-only $DATASET_FLAG

    # Step 2b: score from the persisted responses.csv — no subject model loaded.
    log "Step 2b: Scoring from responses.csv (judge + NLI, no subject model)..."
    MODEL_NAME="$SUBJECT_MODELS" python main.py --responses-csv results/responses.csv $DATASET_FLAG

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

    ensure_deps
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
