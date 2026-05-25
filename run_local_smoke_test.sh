#!/usr/bin/env zsh
# =============================================================================
# run_local_smoke_test.sh
# =============================================================================
# Runs all 4 phases of the Mac smoke test in order.
# Sets up a shared venv if not already present.
#
# Usage:
#   chmod +x run_local_smoke_test.sh
#   ./run_local_smoke_test.sh
#
# Optional flags:
#   --skip-gensens      Skip Steps 1 & 2 (dataset gen + validation)
#   --skip-pri          Skip Step 3 (PRI pilot)
#   --skip-pirc         Skip Step 4 (LL-PIRC)
#   --soft-clamp        Use soft clamping (alpha=0.5) in PIRC test
#   --n-instances N     Number of GenSens instances (default: 5)
#   --n-variants N      Number of paraphrase variants (default: 4)
# =============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SKIP_GENSENS=0
SKIP_PRI=0
SKIP_PIRC=0
SOFT_CLAMP=0
N_INSTANCES=5
N_VARIANTS=4
VENV_DIR="$(pwd)/smoke_venv"

PASS="✅"
FAIL="❌"
INFO="🔹"

# ─── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gensens)  SKIP_GENSENS=1; shift ;;
    --skip-pri)      SKIP_PRI=1;     shift ;;
    --skip-pirc)     SKIP_PIRC=1;    shift ;;
    --soft-clamp)    SOFT_CLAMP=1;   shift ;;
    --n-instances)   N_INSTANCES="$2"; shift 2 ;;
    --n-variants)    N_VARIANTS="$2";  shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ─── Helper functions ─────────────────────────────────────────────────────────
section() { echo "\n$(printf '=%.0s' {1..60})\n  $1\n$(printf '=%.0s' {1..60})"; }
pass()    { echo "${PASS} $1"; }
fail()    { echo "${FAIL} $1"; }
info()    { echo "${INFO} $1"; }

# Track results
STEP_RESULTS=()
record_result() { STEP_RESULTS+=("$1|$2"); }   # name|PASS or name|FAIL

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
GENSENS_DIR="${REPO_ROOT}/gensens"
PR_DIR="${REPO_ROOT}/prompt_robustness"
SMOKE_SCRIPT="${REPO_ROOT}/local_pirc_smoke_test.py"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PROMPT SENSITIVITY — MAC SMOKE TEST             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
info "Repo root    : ${REPO_ROOT}"
info "Venv dir     : ${VENV_DIR}"
info "GenSens      : ${N_INSTANCES} instances × ${N_VARIANTS} variants"
echo ""

# ─── Step 0: Virtual Environment Setup ───────────────────────────────────────
section "STEP 0 — Virtual Environment Setup"

if [[ ! -d "${VENV_DIR}" ]]; then
  info "Creating venv at ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
  pass "venv created"
else
  pass "venv already exists — skipping creation"
fi

# Use explicit binary paths instead of relying on source activate
VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"

info "Python: $(${VENV_PYTHON} --version)"

# Install / upgrade packages
info "Installing dependencies (may take 2–5 minutes on first run)..."
${VENV_PIP} install --upgrade pip

# Core packages for ALL steps
${VENV_PIP} install \
  torch \
  transformers \
  sentence-transformers \
  datasets \
  accelerate \
  rouge-score \
  numpy \
  scipy \
  scikit-learn \
  pandas \
  pyyaml \
  tqdm \
  nltk \
  python-dotenv \
  rich \
  tabulate \
  matplotlib \
  seaborn \
  plotly

pass "All dependencies installed"

# ─── Step 1: GenSens Dataset Generation ──────────────────────────────────────
section "STEP 1 — GenSens Dataset Generation (flan-t5-large)"

if [[ $SKIP_GENSENS -eq 1 ]]; then
  info "Skipped (--skip-gensens)"
  record_result "GenSens generation" "SKIP"
else
  info "Running generate_dataset.py  (this takes 3–8 minutes on MPS/CPU)..."
  cd "${GENSENS_DIR}"

  set +e
  ${VENV_PYTHON} scripts/generate_dataset.py \
    --task summarization \
    --n_instances "${N_INSTANCES}" \
    --n_variants  "${N_VARIANTS}" \
    --model       local \
    --checkpoint_every 2
  GEN_EXIT=$?
  set -e

  if [[ $GEN_EXIT -eq 0 ]]; then
    # Check output file exists and has at least N_INSTANCES lines
    OUT_FILE=$(ls data/gensens_summarization_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl 2>/dev/null || true)
    if [[ -n "$OUT_FILE" && -f "$OUT_FILE" ]]; then
      N_LINES=$(wc -l < "$OUT_FILE" | tr -d ' ')
      pass "Generated output: ${OUT_FILE}  (${N_LINES} lines)"
      record_result "GenSens generation" "PASS"
    else
      fail "Output file not found after generation"
      record_result "GenSens generation" "FAIL"
    fi
  else
    fail "generate_dataset.py exited with code ${GEN_EXIT}"
    record_result "GenSens generation" "FAIL"
  fi

  cd "${REPO_ROOT}"
fi

# ─── Step 2: Validation ───────────────────────────────────────────────────────
section "STEP 2 — Dataset Validation"

if [[ $SKIP_GENSENS -eq 1 ]]; then
  info "Skipped (--skip-gensens)"
  record_result "Dataset validation" "SKIP"
else
  cd "${GENSENS_DIR}"

  set +e
  ${VENV_PYTHON} scripts/validate_dataset.py \
    --expected-instances "${N_INSTANCES}" \
    --expected-variants  "${N_VARIANTS}"
  VAL_EXIT=$?
  set -e

  if [[ $VAL_EXIT -eq 0 ]]; then
    pass "validate_dataset.py completed"
    record_result "Dataset validation" "PASS"
  else
    fail "validate_dataset.py exited with code ${VAL_EXIT}"
    record_result "Dataset validation" "FAIL"
  fi

  cd "${REPO_ROOT}"
fi

# ─── Step 3: PRI Pilot (Flan-T5 models) ──────────────────────────────────────
section "STEP 3 — PRI Pilot Evaluation (flan-t5-base + flan-t5-large)"

if [[ $SKIP_PRI -eq 1 ]]; then
  info "Skipped (--skip-pri)"
  record_result "PRI pilot" "SKIP"
else
  cd "${PR_DIR}"

  info "Running PRI benchmark on Flan-T5 models..."
  set +e
  ${VENV_PYTHON} main.py 2>&1 | tee /tmp/pri_smoke_output.txt
  PRI_EXIT=$?
  set -e

  # Check for NaN in output
  if grep -q "nan" /tmp/pri_smoke_output.txt 2>/dev/null; then
    fail "NaN detected in PRI output — check metric computations"
    record_result "PRI pilot" "FAIL"
  elif [[ $PRI_EXIT -eq 0 ]]; then
    pass "PRI pipeline completed without NaN"
    record_result "PRI pilot" "PASS"
  else
    fail "main.py exited with code ${PRI_EXIT}"
    record_result "PRI pilot" "FAIL"
  fi

  cd "${REPO_ROOT}"
fi

# ─── Step 4: LL-PIRC Smoke Test (GPT-2-medium) ───────────────────────────────
section "STEP 4 — LL-PIRC Smoke Test (gpt2-medium)"

if [[ $SKIP_PIRC -eq 1 ]]; then
  info "Skipped (--skip-pirc)"
  record_result "LL-PIRC smoke test" "SKIP"
else
  PIRC_ARGS=""
  [[ $SOFT_CLAMP -eq 1 ]] && PIRC_ARGS="--soft"

  info "Running LL-PIRC pipeline test (gpt2-medium, ~5 min on CPU)..."
  set +e
  ${VENV_PYTHON} "${SMOKE_SCRIPT}" ${PIRC_ARGS}
  PIRC_EXIT=$?
  set -e

  if [[ $PIRC_EXIT -eq 0 ]]; then
    pass "LL-PIRC smoke test passed"
    record_result "LL-PIRC smoke test" "PASS"
  elif [[ $PIRC_EXIT -eq 1 ]]; then
    fail "LL-PIRC smoke test: some checks failed (see output above)"
    record_result "LL-PIRC smoke test" "FAIL"
  else
    fail "LL-PIRC smoke test crashed with exit code ${PIRC_EXIT}"
    record_result "LL-PIRC smoke test" "FAIL"
  fi
fi

# ─── Final Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  SMOKE TEST RESULTS                    ║"
echo "╠══════════════════════════════════════════════════════════╣"

ALL_PASS=1
for entry in "${STEP_RESULTS[@]}"; do
  STEP_NAME="${entry%%|*}"
  STEP_STATUS="${entry##*|}"
  if [[ "$STEP_STATUS" == "PASS" ]]; then
    printf "║  ✅  %-50s ║\n" "$STEP_NAME"
  elif [[ "$STEP_STATUS" == "SKIP" ]]; then
    printf "║  ⏭️   %-50s ║\n" "$STEP_NAME (skipped)"
  else
    printf "║  ❌  %-50s ║\n" "$STEP_NAME"
    ALL_PASS=0
  fi
done

echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [[ $ALL_PASS -eq 1 ]]; then
  echo "✅  ALL STEPS PASSED — pipeline is ready for H100."
  echo "   Run on Jarvis Labs with:  python scripts/generate_dataset.py --model llama"
  echo "   Then:                     python experiment_pirc.py"
else
  echo "❌  SOME STEPS FAILED — fix issues above before spending H100 budget."
fi

echo ""
