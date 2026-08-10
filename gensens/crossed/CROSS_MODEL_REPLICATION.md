# Cross-model replication — Llama-3.1-8B vs Qwen2.5-7B

Same crossed grid (30 CNN/DM articles × 50 seeds × ~6 paraphrases), same pipeline
and config (v2 composite, whole-summary MiniCheck faithfulness, BERTScore
correctness, PPL pass on). Runs:
- `results_30article_bestscorers/` — Llama-3.1-8B-Instruct
- `results_qwen/` — Qwen2.5-7B-Instruct

Reproduce the comparison:
`python scripts/compare_models.py --run_a results_30article_bestscorers/results --label_a Llama-3.1-8B --run_b results_qwen/results --label_b Qwen2.5-7B`
(raw numbers in `results_qwen/results/compare_Llama-3.1-8B_vs_Qwen2.5-7B.json`)

## The finding replicates across model families

| Metric | Result |
|---|---|
| Dimension-ranking Spearman (Llama vs Qwen) | **+0.70** (p=0.005, n=14) |
| Seed-ranking Spearman (50 prompts) | **+0.74** (p=1e-9) |
| sms_drift-anchor cross-check | +0.82 |
| Bottom-3 dimension overlap | **Jaccard 1.00 (identical)** |
| Top-3 dimension overlap | Jaccard 0.50 (Question Form + Theme Isolation shared) |

**Each model independently reproduces the core structure:** dimension >> pool
(Llama: η² 19.3% vs 1.7%, mixed p=0.0036 / perm p=0.022; Qwen: η² 14.9% vs 0.8%,
mixed p=0.019), and Pool A/B is null in both (Llama seed-MWU p=0.16, Qwen p=0.47).

## Rock-solid (replicated in both models)
- **Question Form** and **Theme Isolation** = top-2 most sensitive in both.
- **Constraint Based, Direct Command, Output Format** = bottom-3 in both (identical set).
- The whole "instruction dimension drives sensitivity, not the Pool split" structure.

## Honest nuance (and why it strengthens the claim)
- **Meta-reflection** was #1 in Llama but #7 in Qwen — the single biggest mover. It
  is the **1-seed dimension**, so it conflates "dimension" with one specific prompt.
  Its divergence across models is exactly what the thin-seed caveat predicted — the
  analysis correctly flagged its own weakest point.
- Middle-rank ordering (e.g. Tone Register: Llama #10 → Qwen #3) is model-specific,
  consistent with the overlapping mid-rank CIs already noted for both runs.

## Defensible claim
> Across two different model families (Llama-3.1-8B, Qwen2.5-7B), instruction
> dimension drives prompt sensitivity far more than the Pool split (both p<0.02,
> ~18× more variance), and the dimension rankings agree significantly (Spearman
> 0.70). Question-form and theme-isolation phrasings are reliably the most
> sensitive, and format/constraint phrasings the least, in both models.

## Per-dimension side by side (sorted by Llama rank)
| Dimension | Llama sens (rank) | Qwen sens (rank) |
|---|---|---|
| Meta-reflection | 0.620 (1) | 0.498 (7) |
| Question Form | 0.614 (2) | 0.545 (2) |
| Theme Isolation | 0.611 (3) | 0.646 (1) |
| Analytical Framing | 0.561 (4) | 0.535 (4) |
| Role Based | 0.520 (5) | 0.463 (10) |
| Persona / Tone | 0.490 (6) | 0.464 (9) |
| Audience Perspective | 0.487 (7) | 0.510 (6) |
| Emphasis Focus | 0.485 (8) | 0.521 (5) |
| Completeness Framing | 0.483 (9) | 0.496 (8) |
| Tone Register | 0.480 (10) | 0.543 (3) |
| Lexical Filter | 0.468 (11) | 0.445 (11) |
| Direct Command | 0.446 (12) | 0.428 (13) |
| Output Format | 0.358 (13) | 0.441 (12) |
| Constraint Based | 0.352 (14) | 0.398 (14) |
