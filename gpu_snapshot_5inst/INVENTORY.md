# LL-PIRC 5-Instance Pipeline Run — Snapshot + Deep-Research Briefing

A self-contained briefing document covering the **Prompt Sensitivity / LL-PIRC** research project, the artifacts produced by a complete 4-phase pipeline run on an A100 GPU instance (2026-05-29), and a ready-to-use prompt for deep-research analysis of the outstanding PIRC mitigation bug.

Total snapshot size: **7.1 MB across 36 files** (mirrored to `gpu_snapshot_5inst/` on the Mac before instance teardown).

---

## 1. Project Context

### What this project does

**LL-PIRC (Logit-Lens Paraphrase-Invariant Residual Clamping)** is an inference-time intervention to reduce prompt sensitivity in LLMs — the phenomenon where semantically equivalent but lexically different prompts produce inconsistent outputs.

### Pipeline (4 phases)

1. **GenSens (Phase 1) — Dataset generation.** Benchmark datasets of semantically equivalent prompt paraphrases for 4 task domains (Summarization & Creative from CNN/DailyMail, Dialogue from DREAM, QA from ELI5), generated with Llama-3.1-70B-AWQ-INT4 via 16 LLM strategies, filtered with SBERT cosine ≥ 0.82. Each instance carries a gold reference output.

2. **PRI Benchmark (Phase 2) — Prompt-Robustness evaluation.** Quantifies how robust each subject model's outputs are to paraphrased prompts using:
   - 6 core metrics (SMS, AUC-E, TRD, KPIG, PPL variance, Branching Factor)
   - 4 advanced metrics (Semantic TRD, Wasserstein SMS, Advanced KPIG, USD)
   - Quality metrics (Correctness, Hallucination, LLM-as-a-Judge)
   - Subjects: Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3, and Llama-3.1-70B-AWQ (4 models × 108 prompts = 432 responses).

3. **LL-PIRC (Phase 3) — Inference-time intervention** on Llama-3.1-8B-Instruct (32 layers):
   1. Run the model on K paraphrase variants of the same prompt.
   2. Use Logit Lens (per-layer `ln_f` + `lm_head`) to find the **sensitive layer ℓ\*** where per-token cross-paraphrase PPL variance peaks (inflection of S(ℓ)).
   3. Identify **anchor token positions** — bottom 30 % by combined `rank(mean_ppl) + rank(var_ppl)` at ℓ\*. Anchors should be tokens the model is both confident about AND consistent about across paraphrases.
   4. At generation time, register a forward hook at layer ℓ\* that replaces the hidden state at anchor positions with the cross-paraphrase consensus:
      `h = α · mean_h + (1 − α) · h_original`

4. **Statistical Evaluation (Phase 4).** Wilcoxon signed-rank tests on ROUGE-L variance and quality, ℓ\* distribution analysis, 95 % bootstrap CIs (10 000 resamples, seed 42), paired Cohen's dz.

### Key formulas (single source of truth: `src/scores.py`)

| Formula | Definition |
|---------|------------|
| **PRI** | `0.40·Consistency(SMS) + 0.35·Quality(CS) + 0.25·Faithfulness(NLI)` |
| **Final Score** | `0.6·PRI + 0.4·Human_Score` |
| **SMS** | `mean(cosine_sim) − 0.5·Var(L2_normed_embeddings)` |
| **ORI** | `(SMS + AUC-E + (1−TRD) + KPIG) / 4` |
| **IFI** | `1 − (PPL_var + BF) / 2` |
| **S(ℓ)** | `Var_k[mean_token_PPL at layer ℓ]` |
| **ℓ\*** | `argmax_ℓ[S(ℓ) − S(ℓ−1)]` |
| **Anchor selection** | Bottom 30 % by `rank(mean_ppl) + rank(var_ppl)` |
| **PIRC clamping** | `h[anchors] = α·mean_h + (1−α)·h_original` |

---

## 2. Empirical Result (the 5-instance test)

PIRC at both **α = 1.0** (full clamp) and **α = 0.5** (soft clamp) produced **byte-identical degenerate outputs**:

```
"Here://://://://://://://://://://://://://://://://://...."
```

(the token `://` repeats hundreds of times for every variant of every article).

| Metric | Baseline (no PIRC) | α = 1.0 PIRC | α = 0.5 PIRC |
|--------|-------------------|---------------|---------------|
| ROUGE-L (mean) | 0.227 | **0.012** | **0.012** |
| ROUGE-L change | — | **−94.7 %** | **−94.7 %** |
| Variance "reduction" | — | 47.1 % | 47.1 % |
| Wilcoxon variance p | — | 0.094 | 0.094 |
| Wilcoxon ROUGE-L p | — | 0.063 | 0.063 |
| Variance Cohen's dz | — | 0.617 | 0.617 |
| ROUGE-L Cohen's dz | — | −4.05 | −4.05 |
| ℓ\* (5 articles) | — | [9, 9, 9, 9, 9] (CV = 0) | identical |

The 47 % variance "reduction" is **meaningless** because both alphas produce the same broken text. The mitigation step is destroying generation, not stabilising it. Baseline (no PIRC) produces normal summaries.

### Bugs already found and fixed

1. **Hook fired on every decode step (KV-cache bug).** During `model.generate()`, the prefill pass has `seq_len == prompt_len`, but each subsequent decode step has `seq_len == 1`. The hook was overwriting the new token's hidden state with `mean_h[0]` (the prompt's first-anchor consensus), collapsing autoregressive generation. **Fix:** added a `clamp_state["prefilled"]` latch — hook now fires only on the prefill pass.
2. **Logit Lens dtype mismatch.** Hidden state cast to fp32 (numerical stability) but `lm_head` weights are fp16 → `F.linear` dtype error. **Fix:** cast `h_normed` back to `lm_head.weight.dtype` before the matmul.

### Still degenerate after both fixes

Even after both fixes, PIRC outputs are still `"Here://://"` garbage. The prefill clamping alone is enough to derail generation.

**Suspect:** anchor selection. Logged anchors show `mean_ppl=inf` and `var_ppl=nan`, which is the **opposite** of "most stable token." Yet the algorithm sorts by combined PPL rank ascending and picks these as the **best** anchors. Likely cause: `torch.argsort` on a tensor containing `inf`/`nan` puts those positions at the **front** of the rank order instead of the back, so the "bottom 30 %" selection grabs exactly the worst tokens.

---

## 3. Snapshot Inventory

### `gpu_snapshot_5inst/results/` — 14 files, 3.4 MB

| File | Size | What it is |
|------|------|------------|
| `responses.csv` | 3.0 MB | 432 model responses, 4 models × 108 prompts (Phase 2 cross-model PRI benchmark) |
| `scored_samples.csv` | 69 KB | 76 per-(model, instance) PRI/ORI/IFI/CS/HS/Faithfulness/Judge rows |
| `baseline.json` | 33 KB | LL-PIRC baseline (Llama-8B, 5 articles × K paraphrases, ROUGE-L per variant) |
| `pirc.json` | 38 KB | **Most recent PIRC run** — α=0.5, prefill-fix applied (still degenerate) |
| `pirc_alpha1.0.json` | 34 KB | α=1.0 with original buggy hook (degenerate) |
| `pirc_alpha0.5_BUGGY.json` | 34 KB | α=0.5 with original buggy hook (degenerate) |
| `eval_summary.json` | 1.5 KB | **Latest** stats — Wilcoxon, bootstrap CI, Cohen's dz, ℓ\* stats |
| `eval_summary_alpha1.0.json` | 1.5 KB | Stats for α=1.0 buggy run |
| `eval_summary_alpha0.5_BUGGY.json` | 1.5 KB | Stats for α=0.5 buggy run |
| `ifi_metrics.json` | 3.5 KB | Internal Fragility Index per subject model |
| `variance_comparison.png` | 92 KB | Baseline vs PIRC variance bar plot (latest) |
| `variance_comparison_alpha1.0.png` | 93 KB | Same plot, α=1.0 buggy version |
| `layer_sensitivity_plot.png` | 51 KB | S(ℓ) sensitivity curves per article (ℓ\*=9) |
| `layer_sensitivity_alpha1.0.png` | 51 KB | Same plot, α=1.0 archive |

### `gpu_snapshot_5inst/gensens_data/` — 12 files, 2.8 MB

| File | Size | What it is |
|------|------|------------|
| `gensens_summarization_5inst_8var.jsonl` | 297 KB | 5 CNN/DailyMail articles × 8 paraphrased instructions |
| `gensens_summarization_5inst_8var.csv` | 413 KB | Same, CSV format |
| `gensens_creative_5inst_8var.jsonl` | 73 KB | 5 creative-writing prompts × 8 variants |
| `gensens_creative_5inst_8var.csv` | 243 KB | CSV |
| `gensens_qa_5inst_8var.jsonl` | 18 KB | 5 ELI5 QA × 8 variants |
| `gensens_qa_5inst_8var.csv` | 28 KB | CSV |
| `gensens_dialogue_5inst_8var.jsonl` | 29 KB | 5 DREAM dialogues × 8 variants (post-fix) |
| `gensens_dialogue_5inst_8var.csv` | 34 KB | CSV |
| `gensens_all_5inst_8var.jsonl` | 417 KB | **Merged 20-instance benchmark set** used by Phase 2 |
| `gensens_stats.json` | 571 B | Paraphrase diversity stats (SBERT cosine, length variance) |
| `summ_instruction_pool.jsonl` | 2.6 KB | Phase 0 generated instruction pool |
| `gensens_validation_report.json` | 1.8 KB | Old validation snapshot |

### `gpu_snapshot_5inst/configs/` — 2 files, 12 KB

| File | What it is |
|------|------------|
| `config.yaml.remote` | Remote-edited config (`num_articles=5`, `pirc.alpha=0.5`, `alpha_values=[0.5]`) |
| `.env.remote` | Remote env file with HF token (**don't commit**) |

### `gpu_snapshot_5inst/logs/` — 8 files, 896 KB

| File | What it is |
|------|------------|
| `phase1.log` | Phase 1 dataset generation (3 of 4 tasks) |
| `full_pipeline.log` | First Phase 2 attempt (failed: disk full, gptqmodel missing) |
| `pipeline_resume.log` | Phase 2 recovery run (4-model responses + scoring) |
| `phase34.log` | Phase 3 baseline + first PIRC (dtype crash) |
| `phase34_retry.log` | Phase 3 retry with dtype fix (α=1.0 buggy run) |
| `phase34_alpha05.log` | α=0.5 buggy run |
| `phase34_alpha05_fixed.log` | α=0.5 with prefill-fix (still degenerate) |
| `pip_ensure.log` | Bootstrap dependency install trace |

### Earlier archived snapshots (subsets of the above)

- `results_5inst_alpha1.0/` — 9 files, 3.4 MB (initial α=1.0 run with first dtype fix only).
- `results_5inst_alpha0.5/` — 12 files, 3.5 MB (initial α=0.5 attempt with both alphas archived).

You only need to upload `gpu_snapshot_5inst/` for deep research.

---

## 4. Source Files to Upload (from `prompt_robustness/src/`)

| File | Why it matters |
|------|----------------|
| `pirc.py` | PIRCGenerator: the forward-hook clamping during `model.generate()`. Contains the **fixed** hook with the prefill latch. |
| `anchor_tokens.py` | `AnchorTokenIdentifier.identify_anchors()`: percentile-based selection over rank-sum stability. **Suspected bug: returns anchors with `mean_ppl=inf` / `var_ppl=nan`.** |
| `sensitive_layer.py` | `S(ℓ) = Var_k[mean_token_PPL at layer ℓ]`; ℓ\* = argmax inflection. |
| `logit_lens.py` | Per-layer `ln_f` + `lm_head` projection, per-token PPL. Contains **fixed** dtype cast. |
| `experiment_baseline.py` | Generates K baseline outputs per article; writes `baseline.json`. |
| `experiment_pirc.py` | Loads `baseline.json`, runs PIRC for each article, writes `pirc.json`. |
| `evaluate.py` | Wilcoxon, bootstrap CIs, paired Cohen's dz on baseline-vs-PIRC variance and ROUGE-L. |
| `config.yaml` | `pirc.alpha`, `anchor_tokens.tau`, `sensitive_layer.scan_start_fraction`, etc. |

---

## 5. Deep-Research Prompt (paste into ChatGPT / Claude / Gemini Deep Research)

> I'm working on a prompt-sensitivity research pipeline. Below is the project context, full file inventory of artifacts I can upload, and the specific question I need help on. Treat this as a deep-research request — examine the methodology, code, and result files, and propose a fix.
>
> ## Project: LL-PIRC (Logit-Lens Paraphrase-Invariant Residual Clamping)
>
> ### Goal
> Mitigate LLM prompt-sensitivity: semantically equivalent but lexically different prompts produce inconsistent outputs. The intervention works at inference time:
>
> 1. Run the model on K paraphrase variants of the same prompt.
> 2. Use Logit Lens (per-layer `ln_f` + `lm_head`) to find the "sensitive layer" ℓ\* where per-token cross-paraphrase PPL variance peaks (inflection of S(ℓ)).
> 3. Identify "anchor token positions" — bottom 30 % by combined `rank(mean_ppl) + rank(var_ppl)` at ℓ\*. Anchors should be tokens the model is both confident about AND consistent about across paraphrases.
> 4. At generation time, register a forward hook at layer ℓ\* that replaces the hidden state at anchor positions with the cross-paraphrase consensus: `h = α · mean_h + (1 − α) · h_original`.
>
> Target model under intervention: `meta-llama/Llama-3.1-8B-Instruct` (32 layers).
>
> ### Empirical result (5-instance smoke test)
> PIRC at both α=1.0 (full clamp) and α=0.5 (soft clamp) produced byte-identical degenerate outputs:
> `"Here://://://://://://://://://://...."` (the token `://` repeats hundreds of times)
> ROUGE-L: baseline 0.227 → PIRC 0.012 (−94.7 %). Variance "reduction" of 47 % is meaningless because both alphas produce the same broken text.
>
> ℓ\* = 9 (out of 32) for every article in the test set — detected with CV=0, so detection itself is stable.
> Anchor selection picks ~76–122 tokens out of ~410 (≈18–30 % of the prompt).
> Logged anchors show `mean_ppl=inf` and `var_ppl=nan`, which is the opposite of "most stable token."
>
> ### Bugs already fixed
> 1. **Hook fired on every decode step (KV cache):** during `model.generate()`, the prefill pass has `seq_len=prompt_len`, but each subsequent decode step has `seq_len=1`. The hook was overwriting the new token's hidden state with `mean_h[0]` (the prompt's first-anchor consensus), collapsing autoregressive generation. Fixed: latch the hook to fire only on the prefill pass.
> 2. **Logit Lens dtype mismatch:** hidden state cast to fp32 (numerical stability) but `lm_head` weights fp16 → `F.linear` dtype error. Fixed: cast `h_normed` back to `lm_head.weight.dtype` before the matmul.
>
> ### After both fixes, PIRC outputs are STILL degenerate `"Here://://"` garbage.
> Baseline (no PIRC) produces normal summaries. The intervention itself is destroying generation even with α=0.5 (only 50 % of anchor positions modified).
>
> ## Files I will upload (artifacts from `gpu_snapshot_5inst/` and `prompt_robustness/src/`)
>
> ### Source code (`src/`)
> - `pirc.py` — PIRCGenerator: forward-hook clamping during `model.generate()`. Contains the FIXED hook with prefill latch.
> - `anchor_tokens.py` — `AnchorTokenIdentifier.identify_anchors()`: percentile-based selection over rank-sum stability. **Suspect: returns anchors with `mean_ppl=inf` / `var_ppl=nan`.**
> - `sensitive_layer.py` — `S(ℓ) = Var_k[mean_token_PPL at layer ℓ]`; ℓ\* = argmax inflection.
> - `logit_lens.py` — per-layer `ln_f` + `lm_head` projection, per-token PPL. Contains FIXED dtype cast.
> - `experiment_baseline.py` — generates K baseline outputs per article; writes `baseline.json`.
> - `experiment_pirc.py` — loads `baseline.json`, runs PIRC for each article, writes `pirc.json`.
> - `evaluate.py` — Wilcoxon, bootstrap CIs, paired Cohen's dz on baseline-vs-PIRC variance and ROUGE-L.
> - `config.yaml` — `pirc.alpha`, `anchor_tokens.tau`, `sensitive_layer.scan_start_fraction`, etc.
>
> ### Result files (`results/`)
> - `baseline.json` (33 KB) — K baseline outputs per article, ROUGE-L per variant, variance per article.
> - `pirc.json` (38 KB) — current α=0.5 PIRC run, post-fix (still degenerate).
> - `pirc_alpha1.0.json` (34 KB) — α=1.0 pre-fix run (degenerate).
> - `pirc_alpha0.5_BUGGY.json` (34 KB) — α=0.5 pre-fix run (degenerate, identical to α=1.0 because hook was clamping new tokens).
> - `eval_summary.json`, `eval_summary_alpha1.0.json`, `eval_summary_alpha0.5_BUGGY.json` — Wilcoxon p-values, dz, variance reduction %.
> - `layer_sensitivity_plot.png`, `variance_comparison.png` — final plots.
> - `ifi_metrics.json` — Internal Fragility Index per model (orthogonal Phase 2 metric).
> - `responses.csv` (3 MB) — 432 model responses (4 models × 108 prompts) from the cross-model PRI benchmark.
> - `scored_samples.csv` (69 KB) — per-(model,instance) PRI/ORI/IFI scores.
>
> ### Datasets (`gensens_data/`)
> - `gensens_summarization_5inst_8var.jsonl` — 5 CNN/DailyMail articles, each with 8 paraphrased instruction variants (generated by Llama-3.1-70B AWQ).
> - Same for `creative`, `qa` (ELI5), `dialogue` (DREAM).
> - `gensens_all_5inst_8var.jsonl` — merged 20-instance benchmark set.
> - `gensens_stats.json` — paraphrase diversity stats (SBERT cosine, length variance per task).
>
> ### Logs (`logs/`)
> - `phase34_alpha05_fixed.log` — most recent PIRC run with the prefill-only fix.
> - `phase34_retry.log` — α=1.0 run with anchor decode of pos / token / mean_ppl / var_ppl per article.
> - `phase34_alpha05.log`, `phase34_alpha05_BUGGY.log` — earlier α=0.5 attempts.
>
> ## Specific deep-research question
>
> The mitigation step is still destroying outputs after fixing the prefill/decode hook bug. Diagnose and propose fixes:
>
> 1. **Anchor selection sanity:** Why is `identify_anchors` returning positions with `mean_ppl=inf` / `var_ppl=nan` despite the algorithm sorting by combined PPL rank ascending? Is `torch.argsort` over a tensor containing `inf`/`nan` producing those positions at the FRONT of the rank order instead of the back? Show me the failure mode with a small numpy/torch reproducer.
>
> 2. **Anchor fraction:** ~18–30 % of prompt tokens being clamped to the cross-paraphrase mean is a lot. Is this fraction principled, or should it be much lower (e.g., 5 %)? What does the literature say (logit lens / activation steering / TruthfulQA-style interventions)?
>
> 3. **Layer choice (ℓ\*=9 in Llama-3.1-8B):** Is clamping at layer 9 of a 32-layer model the right place, or does this mid-network intervention corrupt representations that the upper layers depend on? Should ℓ\* be much deeper (e.g., 24–28)? Cite logit-lens / probing literature about where "semantic" vs "lexical" information lives in Llama.
>
> 4. **Alternative mitigations:** Is there a fundamentally better intervention than per-token hidden-state clamping for prompt-sensitivity reduction (e.g., activation steering vectors, attention head ablation, ICV)?
>
> 5. **What I should change in `pirc.py` and `anchor_tokens.py` to get a working intervention?** Concrete code patches preferred.
>
> Please cite recent papers (2023–2026) on logit lens, activation patching, anchor-token methods, and inference-time steering.

---

## 6. Upload Tips for Deep-Research Tools

- **Minimum upload set:** the 14 files in `gpu_snapshot_5inst/results/` plus the 5 source files (`pirc.py`, `anchor_tokens.py`, `sensitive_layer.py`, `logit_lens.py`, `config.yaml`) from `prompt_robustness/src/`.
- **Full upload (recommended for ChatGPT Deep Research / Claude Projects / Gemini Deep Research):** zip the entire `gpu_snapshot_5inst/` directory plus the `prompt_robustness/src/` folder.
- **For text-only chat:** paste the prompt from section 5 and offer "I'll paste source files in follow-up messages."
- **Sensitive:** do not upload `configs/.env.remote` — it contains the real HF token. Use the `.env.example` template instead.

---

## 7. Pipeline Timeline (this run)

| Time (UTC) | Phase | Outcome |
|------------|-------|---------|
| 13:33 | Phase 1 launch | 3/4 tasks OK; DREAM script loader broken |
| 14:02 | Dialogue re-run | All 4 tasks complete (20 instances total) |
| 14:08 | Phase 2a (responses) | 3/4 models complete; 70B AWQ crash (missing `gptqmodel`) |
| 14:19 | Phase 2a resume | After installing `gptqmodel` + disk cleanup → all 4 models |
| 14:34 | Phase 2b (scoring) | 76 scored rows, all 4 models |
| 14:44 | Phase 3 baseline | `baseline.json` written |
| 14:45 | Phase 3 PIRC | dtype crash → fixed |
| 14:50 | Phase 3 PIRC retry | Runs at α=1.0 — outputs degenerate (`"://://"`) |
| 14:53 | Phase 4 evaluate | All stats + plots written |
| 14:54–14:58 | α=0.5 run | Same degenerate outputs (hook bug revealed) |
| 15:04 | Prefill-fix retry | Still degenerate — anchor selector suspect |
| 15:11 | Snapshot mirrored to Mac | This document |

Total instance runtime: **~1h 30m**.
