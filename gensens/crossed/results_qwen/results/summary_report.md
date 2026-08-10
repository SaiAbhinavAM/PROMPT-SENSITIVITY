# GenSens Crossed-Design Summarization — Summary Report

## 1. Run size
- Total outputs generated: **8970** (articles=30, seeds=50, target variants/cell=6; expected 8970)
- Complete cells scored: **1500** (expected 1500)

## 2. SENSITIVITY by pool (headline result)
*Sensitivity = rank-normalized composite (relative WITHIN this run — a value of 0.5 means 'median cell here', not an absolute level). `SBERT drift (abs)` is the raw 1−mean-pairwise-cosine anchor: run-independent and comparable across models/runs. `Resid` = Sensitivity with quality (cs/faith) regressed out (#8) — a mean-zero 'pure spread' with the quality confound removed.*
| Pool | n_cells | Sensitivity mean | 95% CI | SBERT drift (abs) | Resid (quality-adj) |
|---|---|---|---|---|---|
| A | 1050 | 0.4900 | [0.4798, 0.5003] | 0.0886 | -0.0053 |
| B | 450 | 0.5233 | [0.5077, 0.5387] | 0.1108 | +0.0124 |

> ⚠️ **Pool-B caveat (#9):** Pool B prompts add a *constraint* (e.g. 'no proper nouns', 'isolate one theme'). Some cross-paraphrase output variation there reflects the *legitimate degrees of freedom in satisfying the constraint*, not model fragility — constrained tasks simply admit more valid answers. Read Pool-B sensitivity as an upper bound on fragility, not pure fragility.

## 3. PRI (quality-gated robustness) by pool
*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. Section 2's Sensitivity composite is the headline sensitivity measure.*

| Pool | n_cells | PRI mean | 95% CI |
|---|---|---|---|
| A | 1050 | 0.5799 | [0.5749, 0.5850] |
| B | 450 | 0.5363 | [0.5282, 0.5446] |

## 4. Top 5 MOST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| B04 | B | Theme Isolation | 0.8153 | [0.7747, 0.8528] |
| A07 | A | Question Form | 0.6708 | [0.6198, 0.7197] |
| A28 | A | Emphasis Focus | 0.6668 | [0.5982, 0.7305] |
| B06 | B | Theme Isolation | 0.6373 | [0.5979, 0.6778] |
| A33 | A | Completeness Framing | 0.6213 | [0.5780, 0.6639] |

## 5. Top 5 LEAST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| A01 | A | Direct Command | 0.3000 | [0.2437, 0.3561] |
| A14 | A | Constraint Based | 0.3290 | [0.2839, 0.3749] |
| A13 | A | Constraint Based | 0.3434 | [0.2831, 0.4057] |
| A10 | A | Role Based | 0.3611 | [0.3177, 0.4070] |
| B11 | B | Persona / Tone | 0.3649 | [0.3173, 0.4141] |

## 6. DIMENSION EFFECT — the primary finding
- Instruction **dimension** explains **14.9%** of sensitivity variance vs **0.8%** for the Pool A/B split — i.e. dimension explains ~18x more sensitivity variance than pool.
- **Mixed-effects test (headline, valid unit):** `sensitivity ~ dimension + (1|seed) + (1|article)` → LRT χ²=25.7, df=13, **p=0.01863** (significant). Variance: seed=0.0027, article=0.0058, residual=0.0159.
  - _seed-aggregated Kruskal-Wallis (cruder, underpowered — 50 seeds / 14 dims):_ H=18.0, p=0.156
  - _robust to quality:_ dimension ranking is Spearman 0.97 between raw and quality-residualized Sensitivity (not a quality artifact).
  - _robust to thin dimensions:_ dropping single-seed dimensions (Meta-reflection) leaves η²=0.151 (vs 0.149 with all), mixed-model p=0.01394 — dimension effect is ROBUST to dropping single-seed dimensions.

### Per-dimension sensitivity ranking
| Rank | Dimension | n_seeds | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|---|---|
| 1 | Theme Isolation | 5 | 150 | 0.6462 | [0.6227, 0.6697] |
| 2 | Question Form | 4 | 120 | 0.5449 | [0.5121, 0.5773] |
| 3 | Tone Register | 4 | 120 | 0.5427 | [0.5215, 0.5639] |
| 4 | Analytical Framing | 4 | 120 | 0.5354 | [0.5056, 0.5656] |
| 5 | Emphasis Focus | 3 | 90 | 0.5209 | [0.4835, 0.5592] |
| 6 | Audience Perspective | 3 | 90 | 0.5097 | [0.4757, 0.5434] |
| 7 | Meta-reflection | 1 | 30 | 0.4976 | [0.4552, 0.5429] |
| 8 | Completeness Framing | 3 | 90 | 0.4960 | [0.4620, 0.5295] |
| 9 | Persona / Tone | 6 | 180 | 0.4641 | [0.4435, 0.4848] |
| 10 | Role Based | 4 | 120 | 0.4626 | [0.4370, 0.4879] |
| 11 | Lexical Filter | 3 | 90 | 0.4454 | [0.4161, 0.4759] |
| 12 | Output Format | 3 | 90 | 0.4408 | [0.4078, 0.4731] |
| 13 | Direct Command | 4 | 120 | 0.4278 | [0.3986, 0.4566] |
| 14 | Constraint Based | 3 | 90 | 0.3983 | [0.3648, 0.4323] |

## 7. Diagnosis matrix (quality x sensitivity)
- True Robustness: 421
- Fragile: 329
- Consistently Poor: 329
- Unreliable: 421

## 8. Pool A vs B (secondary — pool explains little variance)
- Seed-level Mann-Whitney U: n_A=35 seeds, n_B=15 seeds, U=297.0, p=0.4717, rank-biserial r=-0.1314 (Pool B more sensitive). Pool is NOT the primary axis — see §6; it explains ~10x less variance than dimension.
  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ p=0.001907
- **Most (B04) vs least (A01) sensitive seed (Wilcoxon):** W=0.0, p=1.863e-09, dz=2.4518, n_paired=30

## 9. Warnings
- Instruction-echo preamble stripped from 2154/8970 outputs before SMS/CS/ROUGE/BERTScore/faithfulness; PPL/BF use raw generation.
- Faithfulness: MiniCheck[deberta-v3-large], whole-summary.
