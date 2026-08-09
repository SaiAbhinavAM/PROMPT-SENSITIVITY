# GenSens Crossed-Design Summarization — Summary Report

## 1. Run size
- Total outputs generated: **8970** (articles=30, seeds=50, target variants/cell=6; expected 8970)
- Complete cells scored: **1500** (expected 1500)

## 2. SENSITIVITY by pool (headline result)
*Sensitivity = rank-normalized composite (relative WITHIN this run — a value of 0.5 means 'median cell here', not an absolute level). `SBERT drift (abs)` is the raw 1−mean-pairwise-cosine anchor: run-independent and comparable across models/runs. `Resid` = Sensitivity with quality (cs/faith) regressed out (#8) — a mean-zero 'pure spread' with the quality confound removed.*
| Pool | n_cells | Sensitivity mean | 95% CI | SBERT drift (abs) | Resid (quality-adj) |
|---|---|---|---|---|---|
| A | 1050 | 0.4878 | [0.4772, 0.4986] | 0.0882 | -0.0100 |
| B | 450 | 0.5285 | [0.5127, 0.5443] | 0.1079 | +0.0233 |

> ⚠️ **Pool-B caveat (#9):** Pool B prompts add a *constraint* (e.g. 'no proper nouns', 'isolate one theme'). Some cross-paraphrase output variation there reflects the *legitimate degrees of freedom in satisfying the constraint*, not model fragility — constrained tasks simply admit more valid answers. Read Pool-B sensitivity as an upper bound on fragility, not pure fragility.

## 3. PRI (quality-gated robustness) by pool
*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. Section 2's Sensitivity composite is the headline sensitivity measure.*

| Pool | n_cells | PRI mean | 95% CI |
|---|---|---|---|
| A | 1050 | 0.7559 | [0.7526, 0.7591] |
| B | 450 | 0.7129 | [0.7069, 0.7191] |

## 4. Top 5 MOST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| B04 | B | Theme Isolation | 0.7785 | [0.7259, 0.8283] |
| A07 | A | Question Form | 0.7540 | [0.7039, 0.8004] |
| B12 | B | Persona / Tone | 0.6964 | [0.6534, 0.7383] |
| A19 | A | Analytical Framing | 0.6670 | [0.6085, 0.7233] |
| A05 | A | Question Form | 0.6559 | [0.6115, 0.7005] |

## 5. Top 5 LEAST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| A13 | A | Constraint Based | 0.3177 | [0.2461, 0.3885] |
| A22 | A | Output Format | 0.3250 | [0.2781, 0.3704] |
| A14 | A | Constraint Based | 0.3384 | [0.2999, 0.3776] |
| A27 | A | Emphasis Focus | 0.3493 | [0.2873, 0.4132] |
| A20 | A | Output Format | 0.3526 | [0.3002, 0.4033] |

## 6. DIMENSION EFFECT — the primary finding
- Instruction **dimension** explains **17.6%** of sensitivity variance vs **1.1%** for the Pool A/B split — i.e. dimension explains ~16x more sensitivity variance than pool.
- **Mixed-effects test (headline, valid unit):** `sensitivity ~ dimension + (1|seed) + (1|article)` → LRT χ²=27.4, df=13, **p=0.01093** (significant). Variance: seed=0.0030, article=0.0071, residual=0.0159.
  - _seed-aggregated Kruskal-Wallis (cruder, underpowered — 50 seeds / 14 dims):_ H=21.6, p=0.0625
  - _robust to quality:_ dimension ranking is Spearman 0.96 between raw and quality-residualized Sensitivity (not a quality artifact).
  - _robust to thin dimensions:_ dropping single-seed dimensions (Meta-reflection) leaves η²=0.172 (vs 0.176 with all), mixed-model p=0.01094 — dimension effect is ROBUST to dropping single-seed dimensions.

### Per-dimension sensitivity ranking
| Rank | Dimension | n_seeds | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|---|---|
| 1 | Question Form | 4 | 120 | 0.6213 | [0.5875, 0.6536] |
| 2 | Meta-reflection | 1 | 30 | 0.6096 | [0.5564, 0.6627] |
| 3 | Theme Isolation | 5 | 150 | 0.6014 | [0.5747, 0.6283] |
| 4 | Analytical Framing | 4 | 120 | 0.5508 | [0.5210, 0.5809] |
| 5 | Role Based | 4 | 120 | 0.5152 | [0.4874, 0.5431] |
| 6 | Audience Perspective | 3 | 90 | 0.5023 | [0.4703, 0.5344] |
| 7 | Completeness Framing | 3 | 90 | 0.4947 | [0.4605, 0.5289] |
| 8 | Emphasis Focus | 3 | 90 | 0.4849 | [0.4468, 0.5236] |
| 9 | Tone Register | 4 | 120 | 0.4827 | [0.4591, 0.5066] |
| 10 | Persona / Tone | 6 | 180 | 0.4794 | [0.4552, 0.5043] |
| 11 | Lexical Filter | 3 | 90 | 0.4779 | [0.4511, 0.5045] |
| 12 | Direct Command | 4 | 120 | 0.4583 | [0.4296, 0.4879] |
| 13 | Constraint Based | 3 | 90 | 0.3551 | [0.3230, 0.3863] |
| 14 | Output Format | 3 | 90 | 0.3495 | [0.3199, 0.3791] |

## 7. Diagnosis matrix (quality x sensitivity)
- True Robustness: 440
- Fragile: 310
- Consistently Poor: 310
- Unreliable: 440

## 8. Pool A vs B (secondary — pool explains little variance)
- Seed-level Mann-Whitney U: n_A=35 seeds, n_B=15 seeds, U=321.0, p=0.2195, rank-biserial r=-0.2229 (Pool B more sensitive). Pool is NOT the primary axis — see §6; it explains ~10x less variance than dimension.
  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ p=0.0001426
- **Most (B04) vs least (A13) sensitive seed (Wilcoxon):** W=5.0, p=1.863e-08, dz=1.7202, n_paired=30

## 9. Warnings
- Instruction-echo preamble stripped from 4469/8970 outputs before SMS/CS/ROUGE/BERTScore/faithfulness; PPL/BF use raw generation.
