#!/usr/bin/env bash
# =============================================================================
# start.sh — End-to-end runner for the Prompt Sensitivity project (<2B models)
# =============================================================================
# Runs the full research pipeline using ONLY models under 2B parameters, so it
# works on a laptop (Apple Silicon MPS / CPU) with no H100 and no gated weights.
#
#   Phase 1  GenSens dataset generation   → google/flan-t5-large   (~770M)
#   Phase 2  PRI robustness benchmark      → flan-t5-base/large,
#                                            distilbart-cnn, bart-large-cnn (all <2B)
#   Phase 3  LL-PIRC intervention test     → gpt2-medium            (~355M)
#
# The production Phase 3/4 path (experiment_pirc.py + evaluate.py) is driven by
# config.yaml's meta-llama/Meta-Llama-3-8B-Instruct (8B, gated, H100-only) and
# is therefore intentionally SKIPPED here. The gpt2-medium smoke test exercises
# the identical LL-PIRC code path (Logit Lens → ℓ* → anchors → clamping) at <2B.
#
# Usage:
#   chmod +x start.sh
#   ./start.sh                       # run all three phases
#   ./start.sh --skip-gensens        # skip Phase 1
#   ./start.sh --skip-pri            # skip Phase 2
#   ./start.sh --skip-pirc           # skip Phase 3
#   ./start.sh --soft-clamp          # use soft clamping (alpha=0.5) in Phase 3
#   ./start.sh --n-instances 10 --n-variants 4
#   ./start.sh --models "google/flan-t5-base,sshleifer/distilbart-cnn-12-6"
#
# Environment overrides (optional):
#   VENV_DIR   path to the virtualenv          (default: ./smoke_venv)
#   PRI_MODELS comma-separated <2B model list  (default: the 4 models below)
#   DATASET    PRI benchmark dataset json      (default: data/sample_dataset.json)
# =============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SKIP_GENSENS=0
SKIP_PRI=0
SKIP_PIRC=0
SOFT_CLAMP=0
N_INSTANCES=6
N_VARIANTS=4
GENSENS_TASK="summarization"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/smoke_venv}"
GENSENS_DIR="${REPO_ROOT}/gensens"
PR_DIR="${REPO_ROOT}/prompt_robustness"
SMOKE_SCRIPT="${REPO_ROOT}/local_pirc_smoke_test.py"

# All strictly < 2B parameters.
PRI_MODELS="${PRI_MODELS:-google/flan-t5-base,google/flan-t5-large,sshleifer/distilbart-cnn-12-6,facebook/bart-large-cnn}"
# Phase 2 evaluates the GenSens paraphrase dataset produced in Phase 1 (the loop
# is closed via the .jsonl bridge in data_loader.py). Override with DATASET=...
DATASET="${DATASET:-}"

# Load keys/config from the single .env if present (HF_TOKEN, model roles, etc.).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

# Apple Silicon: let unsupported MPS ops fall back to CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Avoid tokenizer fork-deadlock warnings under the thread pool.
export TOKENIZERS_PARALLELISM=false

# ─── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gensens) SKIP_GENSENS=1; shift ;;
    --skip-pri)     SKIP_PRI=1;     shift ;;
    --skip-pirc)    SKIP_PIRC=1;    shift ;;
    --soft-clamp)   SOFT_CLAMP=1;   shift ;;
    --n-instances)  N_INSTANCES="$2"; shift 2 ;;
    --n-variants)   N_VARIANTS="$2";  shift 2 ;;
    --task)         GENSENS_TASK="$2"; shift 2 ;;
    --models)       PRI_MODELS="$2"; shift 2 ;;
    --dataset)      DATASET="$2";    shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

section() { printf '\n============================================================\n  %s\n============================================================\n' "$1"; }

# ─── Phase 0: Virtual environment ────────────────────────────────────────────
section "PHASE 0 — Environment"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"
echo "Python: $(${VENV_PY} --version)"

# Install deps only if torch is missing (keeps re-runs fast).
if ! "${VENV_PY}" -c "import torch, transformers, sentence_transformers" 2>/dev/null; then
  echo "Installing dependencies (first run only)..."
  "${VENV_PIP}" install --upgrade pip
  "${VENV_PIP}" install \
    torch transformers sentence-transformers datasets accelerate \
    rouge-score numpy scipy scikit-learn pandas pyyaml tqdm nltk \
    python-dotenv rich tabulate matplotlib seaborn plotly
fi
echo "Dependencies OK."
echo "PRI models (<2B): ${PRI_MODELS}"

# ─── Phase 1: GenSens dataset generation ─────────────────────────────────────
section "PHASE 1 — GenSens dataset generation (flan-t5-large, <2B)"
if [[ $SKIP_GENSENS -eq 1 ]]; then
  echo "Skipped (--skip-gensens)"
else
  ( cd "${GENSENS_DIR}" && "${VENV_PY}" scripts/generate_dataset.py \
      --task "${GENSENS_TASK}" \
      --n_instances "${N_INSTANCES}" \
      --n_variants  "${N_VARIANTS}" \
      --model       local \
      --checkpoint_every 2 )

  ( cd "${GENSENS_DIR}" && "${VENV_PY}" scripts/validate_dataset.py \
      --expected-instances "${N_INSTANCES}" \
      --expected-variants  "${N_VARIANTS}" ) || \
      echo "(validation reported warnings — see report; continuing)"
fi

# ─── Phase 2: PRI robustness benchmark ───────────────────────────────────────
section "PHASE 2 — PRI benchmark (<2B models)"
if [[ $SKIP_PRI -eq 1 ]]; then
  echo "Skipped (--skip-pri)"
else
  # Default: evaluate the newest GenSens summarization dataset from Phase 1.
  if [[ -z "${DATASET}" ]]; then
    DATASET="$(ls -t "${GENSENS_DIR}"/data/gensens_${GENSENS_TASK}_*var.jsonl 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${DATASET}" || ! -f "${DATASET}" ]]; then
    echo "No GenSens dataset found; falling back to legacy sample_dataset.json"
    DATASET="${PR_DIR}/data/sample_dataset.json"
  fi
  echo "Dataset: ${DATASET}"
  # MODEL_NAME is read by prompt_robustness/src/config.py (comma-separated).
  ( cd "${PR_DIR}" && MODEL_NAME="${PRI_MODELS}" "${VENV_PY}" main.py --dataset "${DATASET}" )
  echo "PRI results written to ${PR_DIR}/results/ (benchmark.csv, final_results.csv, plots/)"
fi

# ─── Phase 3: LL-PIRC intervention smoke test ────────────────────────────────
section "PHASE 3 — LL-PIRC intervention (gpt2-medium, <2B)"
if [[ $SKIP_PIRC -eq 1 ]]; then
  echo "Skipped (--skip-pirc)"
else
  # Avoid empty-array expansion (unbound under `set -u` on macOS bash 3.2).
  if [[ $SOFT_CLAMP -eq 1 ]]; then
    "${VENV_PY}" "${SMOKE_SCRIPT}" --soft
  else
    "${VENV_PY}" "${SMOKE_SCRIPT}"
  fi
  echo "LL-PIRC results written to ${REPO_ROOT}/pirc_smoke_test_results.json"
fi

section "DONE"
echo "Phase 1 dataset : ${GENSENS_DIR}/data/"
echo "Phase 2 metrics : ${PR_DIR}/results/"
echo "Phase 3 report  : ${REPO_ROOT}/pirc_smoke_test_results.json"
echo ""
echo "NOTE: the production Phase 3/4 path (experiment_pirc.py + evaluate.py) uses"
echo "      meta-llama/Meta-Llama-3-8B-Instruct (8B) per config.yaml and is NOT"
echo "      run here because it exceeds the 2B-parameter limit."
