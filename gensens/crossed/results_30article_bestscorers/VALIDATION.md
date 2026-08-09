# Validation / cross-checks — 30-article run (SOTA scorers)

Post-hoc robustness checks on `results/cell_metrics_scored.jsonl` (1,500 cells).
Reproduce with: `python scripts/validate_robustness.py --results_dir results_30article_bestscorers/results`
Raw numbers in `results/validation.json`.

The headline finding — **instruction dimension drives prompt sensitivity far more than the Pool A/B split** — survives every check below.

| # | Check | Question it answers | Result | Verdict |
|---|---|---|---|---|
| 1 | **Length confound** | Is "sensitivity" just an output-length effect? | corr(Sensitivity, length) = **−0.07**; dimension ranking after regressing out length = **Spearman 0.97** | ✅ length is not the driver |
| 2 | **Split-half replication** | Does the ranking replicate within the study? | 300 random 15/15 article splits → **mean Spearman 0.87**, 100% of splits > 0.7; same top-3 (Meta-reflection, Question Form, Theme Isolation) and bottom-2 (Constraint Based, Output Format) in both halves | ✅ replicates within-study |
| 3 | **Permutation test** | Is the effect significant *without* a normality assumption? | seed-level η² = 0.46 vs chance 0.27 (95th pct 0.42), **permutation p = 0.022** | ✅ significant, assumption-free |
| 4 | **Paraphrase equivalence** | Do the reworded prompts truly mean the same thing? | bidirectional NLI entailment **0.98 / 0.97**, **100% nli_passed**, token overlap **0.36** | ✅ same meaning, different wording |

## Notes
- **The airtight p-value is 0.022** (permutation, no distributional assumption) — a bit weaker than the parametric mixed-model p=0.0036, and the more defensible number to report.
- **Length caveat (minor):** the two least-sensitive dimensions (Output Format, Constraint Based) also produced the shortest outputs, but the ranking survives length-adjustment (Spearman 0.97), so length is a weak secondary factor, not the cause.
- **Paraphrase equivalence is a dataset strength:** variants were filtered on *bidirectional NLI entailment* (not just SBERT similarity), so they mean the same thing (entail both ways ~0.98) while being lexically distinct (only 36% token overlap) — exactly the property a prompt-sensitivity benchmark needs.

## Still outstanding (NOT checkable on this data — require new evidence)
- **Cross-model replication** — everything here is one model (Llama-3.1-8B) on one task (CNN/DM summarization). The single highest-value next step is re-running the grid on a second model.
- Format/surface perturbations (only semantic paraphrases tested); human validation of the drift metric.
