#!/usr/bin/env bash
# =============================================================================
# run_h100_eval.sh — Evaluation pipeline for a single H100 (Jarvis Labs raw box)
# =============================================================================
# Loads ONE big model per process (avoids 70B + 8B + 7B co-residency on 80GB):
#
#   Phase 1   generate paraphrase dataset      (70B-AWQ via vLLM)
#   Phase 2a  per subject: GENERATE responses  (8B only, no scorers)   → responses.csv
#   Phase 2b  SCORE from responses.csv         (7B embedder + 70B judge/NLI)
#   Phase 2c  reaggregate                       (no models)            → PRI/ORI/Final
#
# Every CSV write is incremental (flush+fsync) and resume-safe: re-running after
# a crash continues where it stopped — computed values are never lost.
#
# Prereqs on the box:
#   export HF_TOKEN=hf_...           # and accept the Llama licenses on HF
#   pip install vllm sentence-transformers transformers accelerate \
#               rouge-score scipy scikit-learn pandas pyyaml tqdm
#
# Usage:
#   ./run_h100_eval.sh                 # all phases
#   ./run_h100_eval.sh --skip-gensens  # reuse an existing dataset
#   SUBJECTS="meta-llama/Llama-3.1-8B-Instruct" ./run_h100_eval.sh   # override
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
GENSENS_DIR="${REPO_ROOT}/gensens"
PR_DIR="${REPO_ROOT}/prompt_robustness"
PY="${PYTHON:-python}"

# Load keys/config from the single .env (HF_TOKEN etc.) for gensens + vLLM + HF CLI.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; . "${REPO_ROOT}/.env"; set +a
  echo "Loaded ${REPO_ROOT}/.env"
fi

# --- Model registry (override via env) ---
GENERATOR="${GENERATOR:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"
GENERATOR_QUANT="${GENERATOR_QUANT:-awq_marlin}"
SUBJECTS="${SUBJECTS:-meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3}"
JUDGE="${JUDGE:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"
JUDGE_QUANT="${JUDGE_QUANT:-awq_marlin}"
EMBEDDER="${EMBEDDER:-Alibaba-NLP/gte-Qwen2-7B-instruct}"

# --- Dataset scale ---
TASK="${TASK:-summarization}"
N_INSTANCES="${N_INSTANCES:-200}"
N_VARIANTS="${N_VARIANTS:-8}"

SKIP_GENSENS=0
[[ "${1:-}" == "--skip-gensens" ]] && SKIP_GENSENS=1

section() { printf '\n============================================================\n  %s\n============================================================\n' "$1"; }

# ─── Phase 1: dataset generation (70B generator) ─────────────────────────────
section "PHASE 1 — paraphrase dataset (${GENERATOR})"
DATASET="${GENSENS_DIR}/data/gensens_${TASK}_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
if [[ $SKIP_GENSENS -eq 1 && -f "$DATASET" ]]; then
  echo "Skipping generation; using ${DATASET}"
else
  ( cd "${GENSENS_DIR}" && "${PY}" scripts/generate_dataset.py \
      --model vllm --model-id "${GENERATOR}" --quantization "${GENERATOR_QUANT}" \
      --task "${TASK}" --n_instances "${N_INSTANCES}" --n_variants "${N_VARIANTS}" \
      --checkpoint_every 10 )
fi

# ─── Phase 2a: generate subject responses (one 8B at a time, no scorers) ─────
section "PHASE 2a — subject responses (generate-only)"
for SUBJ in ${SUBJECTS}; do
  echo "→ ${SUBJ}"
  ( cd "${PR_DIR}" && MODEL_NAME="${SUBJ}" "${PY}" main.py \
      --dataset "${DATASET}" --generate-only )
done
echo "responses.csv now holds all subjects (resume-safe)."

# ─── Phase 2b: score from CSV (7B embedder + 70B judge/NLI; no subject) ──────
section "PHASE 2b — score from responses.csv (7B embedder + 70B judge/NLI)"
( cd "${PR_DIR}" && \
  EMBEDDER_MODEL="${EMBEDDER}" \
  JUDGE_MODEL="${JUDGE}" JUDGE_QUANTIZATION="${JUDGE_QUANT}" \
  FAITHFULNESS_BACKEND="llm" FAITHFULNESS_MODEL="${JUDGE}" \
  "${PY}" main.py --responses-csv results/responses.csv )

# ─── Phase 2c: reaggregate (no models) ───────────────────────────────────────
section "PHASE 2c — reaggregate (no GPU)"
( cd "${PR_DIR}" && "${PY}" reaggregate.py --scored results/scored_samples.csv )

section "DONE"
echo "Pull these back to your laptop (re-aggregate offline, free):"
echo "  ${GENSENS_DIR}/data/gensens_${TASK}_${N_INSTANCES}inst_${N_VARIANTS}var.{jsonl,csv}"
echo "  ${PR_DIR}/results/responses.csv"
echo "  ${PR_DIR}/results/scored_samples.csv"
echo "  ${PR_DIR}/results/benchmark_reaggregated.csv"
