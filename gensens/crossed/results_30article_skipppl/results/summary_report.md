# GenSens Crossed-Design Summarization — Summary Report

## 1. Run size
- Total outputs generated: **8970** (articles=30, seeds=50, target variants/cell=6; expected 8970)
- Complete cells scored: **1500** (expected 1500)

## 2. SENSITIVITY by pool (headline result)
*Sensitivity = rank-normalized composite (relative WITHIN this run — a value of 0.5 means 'median cell here', not an absolute level). `SBERT drift (abs)` is the raw 1−mean-pairwise-cosine anchor: run-independent and comparable across models/runs. `Resid` = Sensitivity with quality (cs/faith) regressed out (#8) — a mean-zero 'pure spread' with the quality confound removed.*
| Pool | n_cells | Sensitivity mean | 95% CI | SBERT drift (abs) | Resid (quality-adj) |
|---|---|---|---|---|---|
| A | 1050 | 0.4898 | [0.4782, 0.5012] | 0.0882 | -0.0039 |
| B | 450 | 0.5237 | [0.5080, 0.5393] | 0.1079 | +0.0091 |

> ⚠️ **Pool-B caveat (#9):** Pool B prompts add a *constraint* (e.g. 'no proper nouns', 'isolate one theme'). Some cross-paraphrase output variation there reflects the *legitimate degrees of freedom in satisfying the constraint*, not model fragility — constrained tasks simply admit more valid answers. Read Pool-B sensitivity as an upper bound on fragility, not pure fragility.

## 3. PRI (quality-gated robustness) by pool
*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. Section 2's Sensitivity composite is the headline sensitivity measure.*

| Pool | n_cells | PRI mean | 95% CI |
|---|---|---|---|
| A | 1050 | 0.7344 | [0.7309, 0.7378] |
| B | 450 | 0.7034 | [0.6968, 0.7101] |

## 4. Top 5 MOST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| B04 | B | Theme Isolation | 0.7643 | [0.7076, 0.8177] |
| A19 | A | Analytical Framing | 0.7094 | [0.6466, 0.7680] |
| A07 | A | Question Form | 0.6803 | [0.6278, 0.7312] |
| A02 | A | Direct Command | 0.6657 | [0.6134, 0.7153] |
| B06 | B | Theme Isolation | 0.6641 | [0.6273, 0.6989] |

## 5. Top 5 LEAST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| A13 | A | Constraint Based | 0.2531 | [0.1837, 0.3253] |
| A21 | A | Output Format | 0.3363 | [0.2842, 0.3889] |
| A22 | A | Output Format | 0.3384 | [0.2925, 0.3847] |
| A20 | A | Output Format | 0.3421 | [0.2844, 0.3970] |
| B11 | B | Persona / Tone | 0.3563 | [0.3108, 0.4019] |

## 6. DIMENSION EFFECT — the primary finding
- Instruction **dimension** explains **21.1%** of sensitivity variance vs **0.7%** for the Pool A/B split — i.e. dimension explains ~30x more sensitivity variance than pool.
- **Mixed-effects test (headline, valid unit):** `sensitivity ~ dimension + (1|seed) + (1|article)` → LRT χ²=38.1, df=13, **p=0.0002769** (significant). Variance: seed=0.0060, article=0.0057, residual=0.0152.
  - _seed-aggregated Kruskal-Wallis (cruder, underpowered — 50 seeds / 14 dims):_ H=26.4, p=0.01485
  - _robust to quality:_ dimension ranking is Spearman 0.95 between raw and quality-residualized Sensitivity (not a quality artifact).
  - _robust to thin dimensions:_ dropping single-seed dimensions (Meta-reflection) leaves η²=0.210 (vs 0.211 with all), mixed-model p=0.0002403 — dimension effect is ROBUST to dropping single-seed dimensions.

### Per-dimension sensitivity ranking
| Rank | Dimension | n_seeds | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|---|---|
| 1 | Theme Isolation | 5 | 150 | 0.6188 | [0.5932, 0.6454] |
| 2 | Question Form | 4 | 120 | 0.6122 | [0.5798, 0.6426] |
| 3 | Analytical Framing | 4 | 120 | 0.6079 | [0.5769, 0.6390] |
| 4 | Meta-reflection | 1 | 30 | 0.5853 | [0.5257, 0.6428] |
| 5 | Emphasis Focus | 3 | 90 | 0.5460 | [0.5032, 0.5883] |
| 6 | Audience Perspective | 3 | 90 | 0.4892 | [0.4539, 0.5250] |
| 7 | Completeness Framing | 3 | 90 | 0.4887 | [0.4525, 0.5243] |
| 8 | Role Based | 4 | 120 | 0.4878 | [0.4598, 0.5157] |
| 9 | Persona / Tone | 6 | 180 | 0.4711 | [0.4494, 0.4928] |
| 10 | Tone Register | 4 | 120 | 0.4633 | [0.4389, 0.4875] |
| 11 | Direct Command | 4 | 120 | 0.4591 | [0.4287, 0.4900] |
| 12 | Lexical Filter | 3 | 90 | 0.4499 | [0.4212, 0.4789] |
| 13 | Constraint Based | 3 | 90 | 0.3450 | [0.3100, 0.3788] |
| 14 | Output Format | 3 | 90 | 0.3390 | [0.3089, 0.3691] |

## 7. Diagnosis matrix (quality x sensitivity)
- True Robustness: 431
- Fragile: 319
- Consistently Poor: 319
- Unreliable: 431

## 8. Pool A vs B (secondary — pool explains little variance)
- Seed-level Mann-Whitney U: n_A=35 seeds, n_B=15 seeds, U=311.0, p=0.3095, rank-biserial r=-0.1848 (Pool B more sensitive). Pool is NOT the primary axis — see §6; it explains ~10x less variance than dimension.
  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ p=0.001368
- **Most (B04) vs least (A13) sensitive seed (Wilcoxon):** W=3.0, p=9.313e-09, dz=1.8983, n_paired=30

## 9. Warnings
- Instruction-echo preamble stripped from 4469/8970 outputs before SMS/CS/ROUGE/BERTScore/faithfulness; PPL/BF use raw generation.
- 416 NLI pairs had an output too long for the premise to fit any context; scored as neutral (0.5).
- PPL_var/PC_stab_var not computed (--skip_ppl_entropy).
