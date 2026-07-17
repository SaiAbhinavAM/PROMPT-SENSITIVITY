# GenSens Crossed-Design Summarization — Summary Report

## 1. Run size
- Total outputs generated: **8970** (articles=30, seeds=50, target variants/cell=6; expected 8970)
- Complete cells scored: **1500** (expected 1500)

## 2. SENSITIVITY by pool (headline result)
| Pool | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|
| A | 1050 | 0.4835 | [0.4726, 0.4946] |
| B | 450 | 0.5386 | [0.5220, 0.5552] |

## 3. PRI (quality-gated robustness) by pool
*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. Section 2's Sensitivity composite is the headline sensitivity measure.*

| Pool | n_cells | PRI mean | 95% CI |
|---|---|---|---|
| A | 1050 | 0.7559 | [0.7526, 0.7591] |
| B | 450 | 0.7129 | [0.7069, 0.7191] |

## 4. Top 5 MOST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| B04 | B | Theme Isolation | 0.7980 | [0.7432, 0.8497] |
| A07 | A | Question Form | 0.7393 | [0.6865, 0.7866] |
| B12 | B | Persona / Tone | 0.7096 | [0.6609, 0.7572] |
| A05 | A | Question Form | 0.6546 | [0.6078, 0.7002] |
| A02 | A | Direct Command | 0.6518 | [0.6034, 0.7022] |

## 5. Top 5 LEAST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| A22 | A | Output Format | 0.3047 | [0.2523, 0.3546] |
| A13 | A | Constraint Based | 0.3090 | [0.2348, 0.3823] |
| A14 | A | Constraint Based | 0.3283 | [0.2882, 0.3705] |
| A20 | A | Output Format | 0.3450 | [0.2877, 0.4015] |
| B11 | B | Persona / Tone | 0.3490 | [0.3019, 0.3962] |

## 6. DIMENSION EFFECT — the primary finding
- Instruction **dimension** explains **18.0%** of sensitivity variance vs **1.9%** for the Pool A/B split — i.e. dimension explains ~10x more sensitivity variance than pool.
- **Mixed-effects test (headline, valid unit):** `sensitivity ~ dimension + (1|seed) + (1|article)` → LRT χ²=29.3, df=13, **p=0.00594** (significant). Variance: seed=0.0029, article=0.0071, residual=0.0182.
  - _seed-aggregated Kruskal-Wallis (cruder, underpowered — 50 seeds / 14 dims):_ H=22.3, p=0.05056
  - _robust to quality:_ dimension ranking is Spearman 0.93 between raw and quality-residualized Sensitivity (not a quality artifact).

### Per-dimension sensitivity ranking
| Rank | Dimension | n_seeds | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|---|---|
| 1 | Question Form | 4 | 120 | 0.6217 | [0.5891, 0.6533] |
| 2 | Theme Isolation | 5 | 150 | 0.6207 | [0.5931, 0.6482] |
| 3 | Meta-reflection | 1 | 30 | 0.5907 | [0.5343, 0.6470] |
| 4 | Analytical Framing | 4 | 120 | 0.5490 | [0.5185, 0.5798] |
| 5 | Role Based | 4 | 120 | 0.5004 | [0.4719, 0.5290] |
| 6 | Persona / Tone | 6 | 180 | 0.5000 | [0.4737, 0.5269] |
| 7 | Audience Perspective | 3 | 90 | 0.4969 | [0.4634, 0.5306] |
| 8 | Emphasis Focus | 3 | 90 | 0.4900 | [0.4508, 0.5291] |
| 9 | Completeness Framing | 3 | 90 | 0.4893 | [0.4549, 0.5237] |
| 10 | Tone Register | 4 | 120 | 0.4810 | [0.4558, 0.5064] |
| 11 | Lexical Filter | 3 | 90 | 0.4614 | [0.4331, 0.4897] |
| 12 | Direct Command | 4 | 120 | 0.4528 | [0.4209, 0.4853] |
| 13 | Constraint Based | 3 | 90 | 0.3540 | [0.3191, 0.3888] |
| 14 | Output Format | 3 | 90 | 0.3371 | [0.3060, 0.3678] |

## 7. Diagnosis matrix (quality x sensitivity)
- True Robustness: 443
- Fragile: 307
- Consistently Poor: 307
- Unreliable: 443

## 8. Pool A vs B (secondary — pool explains little variance)
- Seed-level Mann-Whitney U: n_A=35 seeds, n_B=15 seeds, U=324.0, p=0.1966, rank-biserial r=-0.2343 (Pool B more sensitive). Pool is NOT the primary axis — see §6; it explains ~10x less variance than dimension.
  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ p=5.899e-07
- **Most (B04) vs least (A22) sensitive seed (Wilcoxon):** W=1.0, p=3.725e-09, dz=2.6017, n_paired=30

## 9. Warnings
- Instruction-echo preamble stripped from 4469/8970 outputs before SMS/CS/ROUGE/BERTScore/faithfulness; PPL/BF use raw generation.
