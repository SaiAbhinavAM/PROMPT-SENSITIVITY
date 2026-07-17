#!/usr/bin/env bash
# stage0a_smoke.sh — Stage 0a pipeline-validation smoke test for the
# NeurIPS D&B 2026 submission. Validates the new paraphrase methodology
# on a small scale (10 instances × 3 tasks) so we catch code-shape bugs
# before paying for the full 200 × 3 production run.
#
# Target: A30 × 2 (48GB pooled via TP=2) or A40 (48GB single).
# Generator: Qwen2.5-7B-Instruct (NOT the 70B AWQ — smoke test only).
# Expected wall-clock: ~20–30 minutes
# Expected cost on A40 spot ($0.50/hr): ~$0.20

set -euo pipefail
set -E
shopt -s inherit_errexit 2>/dev/null || true

# ─── Configuration ─────────────────────────────────────────────────────
ROOT="${ROOT:-$(pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/results/stage0a}"
DATA_DIR="${DATA_DIR:-${ROOT}/gensens/data}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
N_INSTANCES="${N_INSTANCES:-10}"
N_VARIANTS="${N_VARIANTS:-8}"
GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QUANTIZATION="${QUANTIZATION:-}"          # e.g. awq_marlin for 70B AWQ; blank for Qwen-7B bf16
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"   # set 2 on A30 × 2 for 70B AWQ
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ENABLE_NLI_ENSEMBLE="${ENABLE_NLI_ENSEMBLE:-1}"     # Tier-2 §7.1: 1=on, 0=off
BEST_OF_N="${BEST_OF_N:-3}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${DATA_DIR}"

START_T="$(date +%s)"
PASS=()
FAIL=()
SKIP=()

# ─── Helpers ───────────────────────────────────────────────────────────
log()  { printf '%s [%s] %s\n' "$(date +%H:%M:%S)" "$1" "$2" | tee -a "${LOG_DIR}/stage0a.log"; }
info() { log "INFO" "$1"; }
ok()   { log " OK " "$1"; PASS+=("$1"); }
fail() { log "FAIL" "$1"; FAIL+=("$1"); }
skip() { log "SKIP" "$1"; SKIP+=("$1"); }
section() {
    printf '\n%s\n' '════════════════════════════════════════════════════════════════════════' | tee -a "${LOG_DIR}/stage0a.log"
    printf ' %s\n' "$1" | tee -a "${LOG_DIR}/stage0a.log"
    printf '%s\n' '════════════════════════════════════════════════════════════════════════' | tee -a "${LOG_DIR}/stage0a.log"
}

# ─── 0. Environment sanity ─────────────────────────────────────────────
section "Stage 0a — Pipeline validation smoke test"
info "ROOT          = ${ROOT}"
info "RESULTS_DIR   = ${RESULTS_DIR}"
info "GENERATOR     = ${GENERATOR_MODEL}"
info "N_INSTANCES   = ${N_INSTANCES}  (per task)"
info "N_VARIANTS    = ${N_VARIANTS}   (K)"

# Sanity: HF_TOKEN, GPU presence
if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    info "warning: HF_TOKEN not set in environment — gated models may fail"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | tee -a "${LOG_DIR}/stage0a.log"
else
    info "warning: nvidia-smi not found"
fi

# ─── 1. Dependency check ───────────────────────────────────────────────
section "1. Dependency check"
python3 - <<'PY' 2>&1 | tee -a "${LOG_DIR}/stage0a.log"
import sys, importlib
required = ['torch', 'transformers', 'sentence_transformers', 'vllm', 'numpy',
            'pandas', 'yaml', 'rouge_score']
missing = []
for m in required:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', '?')
        print(f'  ✓ {m:24s} {v}')
    except ImportError:
        missing.append(m)
        print(f'  ✗ {m:24s} NOT INSTALLED')
sys.exit(1 if missing else 0)
PY
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    fail "missing required packages"
    echo "→ run: pip install torch transformers sentence-transformers vllm numpy pandas pyyaml rouge_score datasets"
    exit 1
fi
ok "all required packages present"

# ─── 2. Generate 10 × 3 dataset ────────────────────────────────────────
section "2. GenSens generation (10 instances × 3 tasks × K=${N_VARIANTS})"
GEN_LOG="${LOG_DIR}/gensens_generation.log"
GENSENS_OUT_BASE="${DATA_DIR}/gensens_all_${N_INSTANCES}inst_${N_VARIANTS}var"

if [[ -f "${DATA_DIR}/gensens_summarization_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl" ]] && \
   [[ -f "${DATA_DIR}/gensens_creative_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl" ]] && \
   [[ -f "${DATA_DIR}/gensens_dialogue_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl" ]]; then
    info "all three task JSONLs already exist on disk — skipping regeneration"
    info "  (delete them if you want a fresh run)"
    ok "dataset present (skipped regeneration)"
else
    cd "${ROOT}/gensens"
    info "launching generate_dataset.py …"

    # Compose CLI flags conditionally so the script works for both
    # smoke-test (Qwen-7B, single card) and production (70B AWQ, TP=2)
    # configurations without forking the script.
    GEN_ARGS=(
        --task all
        --n_instances "${N_INSTANCES}"
        --n_variants "${N_VARIANTS}"
        --model vllm
        --model-id "${GENERATOR_MODEL}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --best-of-n "${BEST_OF_N}"
    )
    if [[ -n "${QUANTIZATION}" ]]; then
        GEN_ARGS+=(--quantization "${QUANTIZATION}")
    fi
    if [[ "${ENABLE_NLI_ENSEMBLE}" == "1" ]]; then
        GEN_ARGS+=(--enable-nli-ensemble)
    fi

    info "CLI: python scripts/generate_dataset.py ${GEN_ARGS[*]}"
    if python3 scripts/generate_dataset.py "${GEN_ARGS[@]}" >> "${GEN_LOG}" 2>&1; then
        ok "GenSens generation finished"
    else
        fail "GenSens generation crashed — see ${GEN_LOG}"
        tail -40 "${GEN_LOG}"
        exit 1
    fi
    cd "${ROOT}"
fi

# Merge per-task JSONLs into one combined file for downstream stages.
COMBINED="${GENSENS_OUT_BASE}.jsonl"
> "${COMBINED}"
for task in summarization creative dialogue; do
    path="${DATA_DIR}/gensens_${task}_${N_INSTANCES}inst_${N_VARIANTS}var.jsonl"
    if [[ -f "${path}" ]]; then
        cat "${path}" >> "${COMBINED}"
    else
        info "warning: ${path} missing"
    fi
done
n_records="$(wc -l < "${COMBINED}" | tr -d ' ')"
info "combined JSONL has ${n_records} records → ${COMBINED}"

# ─── 3. Schema / yield audit on the new dataset ────────────────────────
section "3. Schema & yield audit on the generated dataset"
python3 - "${COMBINED}" "${N_VARIANTS}" 2>&1 | tee -a "${LOG_DIR}/stage0a.log" <<'PY'
import json, sys
from collections import defaultdict
path = sys.argv[1]; k_target = int(sys.argv[2])

records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"  records: {len(records)}")

# Per-task stats
by_task = defaultdict(list)
for r in records:
    by_task[r.get('task', 'unknown')].append(r)

# Per-variant audit fields the new Tier-1 code MUST populate
REQUIRED_FIELDS = ['paraphrased_text', 'strategy', 'sbert_similarity', 'variant_idx']
NEW_TIER1_FIELDS = ['strategy_family', 'length_ratio', 'token_overlap_to_base',
                    'max_cos_to_accepted', 'nli_entail_fwd', 'nli_entail_bwd',
                    'nli_passed', 'best_of_n_index']

print()
print(f"  {'Task':<15s} {'n_inst':>6s} {'avg_K':>6s} {'min_K':>5s} {'NLI%':>5s}")
for task, group in sorted(by_task.items()):
    if not group: continue
    ks = [len(r.get('variants', []) or []) for r in group]
    nli_passed_total = sum(
        1 for r in group for v in (r.get('variants') or [])
        if v.get('nli_passed') is True
    )
    n_variants_total = sum(ks)
    nli_pct = (100 * nli_passed_total / max(1, n_variants_total))
    print(f"  {task:<15s} {len(group):>6d} {sum(ks)/len(ks):>6.2f} "
          f"{min(ks):>5d} {nli_pct:>4.0f}%")

# Schema check: pick first variant we find with content
v_sample = None
for r in records:
    for v in r.get('variants') or []:
        v_sample = v
        break
    if v_sample: break

if v_sample:
    missing_req = [f for f in REQUIRED_FIELDS if f not in v_sample]
    missing_new = [f for f in NEW_TIER1_FIELDS if f not in v_sample]
    print()
    if missing_req:
        print(f"  ✗ REQUIRED fields missing: {missing_req}")
        sys.exit(2)
    if missing_new:
        print(f"  ⚠ NEW Tier-1 fields missing: {missing_new}")
        print(f"    → generation ran with legacy code path; new methodology NOT exercised")
        sys.exit(3)
    print(f"  ✓ all {len(REQUIRED_FIELDS)} required + {len(NEW_TIER1_FIELDS)} new fields populated")
else:
    print(f"  ✗ no variants found in any record")
    sys.exit(4)

# Yield gate — average variants per instance ≥ 5/k_target
overall_avg = sum(len(r.get('variants') or []) for r in records) / max(1, len(records))
print(f"\n  overall avg variants/instance: {overall_avg:.2f}  (target ≥ 5)")
if overall_avg < 4.0:
    print(f"  ⚠ yield below 4.0 — filter is too strict OR generator is producing low-quality candidates")
    sys.exit(5)
PY
rc=$?
case $rc in
  0) ok "schema + yield gates pass" ;;
  2) fail "required schema fields missing"; exit 1 ;;
  3) fail "new Tier-1 audit fields missing — methodology not exercised" ;;
  4) fail "no variants in any record"; exit 1 ;;
  5) fail "variant yield below 4.0/${N_VARIANTS} — investigate filters"; ;;
  *) fail "audit script exited rc=$rc" ;;
esac

# ─── 4. Back-translation augmentation ──────────────────────────────────
section "4. Back-translation family (Tier-2 §7.2)"
BT_LOG="${LOG_DIR}/back_translation.log"
BT_OUT="${GENSENS_OUT_BASE}_bt.jsonl"

if python3 "${ROOT}/gensens/scripts/back_translation_family.py" \
        --input "${COMBINED}" \
        --output "${BT_OUT}" \
        --pivot-langs de fr \
        --max-per-instance 2 \
        --enable-nli-ensemble \
        >> "${BT_LOG}" 2>&1; then
    # Count BT variants added
    bt_added="$(python3 -c "
import json, sys
records = [json.loads(l) for l in open('${BT_OUT}') if l.strip()]
n_bt = sum(1 for r in records for v in (r.get('variants') or [])
           if (v.get('strategy_family') == 'back_translation'))
n_records = len(records)
avg = n_bt / max(1, n_records)
print(f'{n_bt}|{n_records}|{avg:.2f}')
")"
    n_bt="${bt_added%%|*}"
    rest="${bt_added#*|}"
    n_rec="${rest%%|*}"
    avg="${rest##*|}"
    info "  back-translation variants added: ${n_bt} across ${n_rec} records (avg ${avg}/record)"
    if [[ "${n_bt}" -gt 0 ]]; then
        ok "back-translation produced ≥1 variants"
    else
        fail "back-translation added 0 variants — filter or NMT issue"
    fi
else
    fail "back_translation_family.py crashed — see ${BT_LOG}"
    tail -25 "${BT_LOG}"
fi

# ─── 5. TextFooler adversarial subset ──────────────────────────────────
section "5. TextFooler adversarial paraphrases (Tier-2 §7.3)"
TF_LOG="${LOG_DIR}/textfooler.log"
TF_OUT="${DATA_DIR}/adv_textfooler_${N_INSTANCES}inst.jsonl"

if python3 "${ROOT}/gensens/scripts/adversarial_textfooler.py" \
        --input "${COMBINED}" \
        --output "${TF_OUT}" \
        --max-substitutions 3 \
        --top-k-mlm 6 \
        --max-prompts 30 \
        >> "${TF_LOG}" 2>&1; then
    stats="$(python3 -c "
import json
rows = [json.loads(l) for l in open('${TF_OUT}') if l.strip()]
n = len(rows)
n_with_sub = sum(1 for r in rows if r.get('num_substitutions', 0) > 0)
n_sub_total = sum(r.get('num_substitutions', 0) for r in rows)
print(f'{n}|{n_with_sub}|{n_sub_total}')
")"
    n_total="${stats%%|*}"
    rest="${stats#*|}"
    n_with="${rest%%|*}"
    n_sub_total="${rest##*|}"
    info "  prompts processed: ${n_total} | with ≥1 substitution: ${n_with} | total substitutions: ${n_sub_total}"
    if [[ "${n_with}" -ge $(( n_total / 2 )) ]]; then
        ok "TextFooler produced substitutions on ≥50% of prompts"
    else
        fail "TextFooler too few substitutions (${n_with}/${n_total}) — filter or MLM issue"
    fi
else
    fail "adversarial_textfooler.py crashed — see ${TF_LOG}"
    tail -25 "${TF_LOG}"
fi

# ─── 6. PAWS negative-control audit ────────────────────────────────────
section "6. PAWS-style negative-control audit (§8.3)"
PAWS_LOG="${LOG_DIR}/paws_audit.log"

# Ensure the 200-pair PAWS dataset exists
if [[ ! -f "${DATA_DIR}/paws_negative_controls.jsonl" ]]; then
    python3 "${ROOT}/prompt_robustness/scripts/paws_negative_controls.py" \
        --mode generate \
        --output "${DATA_DIR}/paws_negative_controls.jsonl" \
        --n-per-type 40 \
        >> "${PAWS_LOG}" 2>&1
fi

if python3 "${ROOT}/prompt_robustness/scripts/paws_negative_controls.py" \
        --mode audit \
        --input "${DATA_DIR}/paws_negative_controls.jsonl" \
        --output-dir "${RESULTS_DIR}/paws_audit" \
        --enable-nli-ensemble \
        >> "${PAWS_LOG}" 2>&1; then
    # Extract rejection rate
    rej_rate="$(python3 -c "
import json
s = json.load(open('${RESULTS_DIR}/paws_audit/paws_audit.json'))
print(f'{100*s[\"overall_rejection_rate\"]:.1f}')
")"
    info "  rejection rate on intentional non-paraphrases: ${rej_rate}%"
    # PASS at ≥ 85% (target is 90%, but smoke-test threshold is more lenient)
    pass_flag="$(python3 -c "print('yes' if ${rej_rate} >= 85.0 else 'no')")"
    if [[ "${pass_flag}" == "yes" ]]; then
        ok "PAWS rejection rate ≥ 85% (filters are sensitive to semantic meaning)"
    else
        fail "PAWS rejection rate ${rej_rate}% < 85% — filters too lenient"
    fi
else
    fail "PAWS audit crashed — see ${PAWS_LOG}"
    tail -25 "${PAWS_LOG}"
fi

# ─── Summary ───────────────────────────────────────────────────────────
END_T="$(date +%s)"
ELAPSED=$(( END_T - START_T ))
section "Stage 0a summary"
printf "  elapsed:  %d:%02d (mm:ss)\n" $(( ELAPSED / 60 )) $(( ELAPSED % 60 )) | tee -a "${LOG_DIR}/stage0a.log"
printf "  PASSES:   %d\n" "${#PASS[@]}" | tee -a "${LOG_DIR}/stage0a.log"
for p in "${PASS[@]:-}"; do printf "    ✓ %s\n" "$p"; done | tee -a "${LOG_DIR}/stage0a.log"
printf "  FAILURES: %d\n" "${#FAIL[@]}" | tee -a "${LOG_DIR}/stage0a.log"
for f in "${FAIL[@]:-}"; do printf "    ✗ %s\n" "$f"; done | tee -a "${LOG_DIR}/stage0a.log"

if [[ "${#FAIL[@]}" -eq 0 ]]; then
    printf "\n✓ STAGE 0a PASSED — methodology code is sound, safe to scale up.\n" | tee -a "${LOG_DIR}/stage0a.log"
    exit 0
else
    printf "\n✗ STAGE 0a FAILED — %d gate(s) failed. Inspect logs in %s\n" "${#FAIL[@]}" "${LOG_DIR}" | tee -a "${LOG_DIR}/stage0a.log"
    exit 1
fi
