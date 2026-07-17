# GenSens Crossed-Design Summarization — Summary Report

## 1. Run size
- Total outputs generated: **1495** (articles=5, seeds=50, target variants/cell=6; expected 1495)
- Complete cells scored: **250** (expected 250)

## 2. SENSITIVITY by pool (headline result)
| Pool | n_cells | Sensitivity mean | 95% CI |
|---|---|---|---|
| A | 175 | 0.4806 | [0.4537, 0.5081] |
| B | 75 | 0.5452 | [0.5017, 0.5882] |

## 3. PRI (quality-gated robustness) by pool
*Note: PRI is quality-gated robustness (good AND stable), not pure sensitivity. Section 2's Sensitivity composite is the headline sensitivity measure.*

| Pool | n_cells | PRI mean | 95% CI |
|---|---|---|---|
| A | 175 | 0.6975 | [0.6905, 0.7049] |
| B | 75 | 0.6736 | [0.6589, 0.6884] |

## 4. Top 5 MOST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| B12 | B | Persona / Tone | 0.7063 | [0.5448, 0.8195] |
| A19 | A | Analytical Framing | 0.7013 | [0.6557, 0.7602] |
| B06 | B | Theme Isolation | 0.7007 | [0.5548, 0.8062] |
| A17 | A | Analytical Framing | 0.6869 | [0.5611, 0.8155] |
| B04 | B | Theme Isolation | 0.6678 | [0.5339, 0.8132] |

## 5. Top 5 LEAST sensitive seeds
| seed_id | pool | dimension | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| A14 | A | Constraint Based | 0.2586 | [0.1686, 0.3275] |
| A22 | A | Output Format | 0.2674 | [0.1822, 0.3478] |
| B11 | B | Persona / Tone | 0.2804 | [0.1888, 0.3829] |
| A20 | A | Output Format | 0.2994 | [0.1641, 0.4178] |
| A03 | A | Direct Command | 0.3390 | [0.2654, 0.4218] |

## 6. Per-dimension sensitivity ranking
| Rank | Dimension | n | Sensitivity mean | 95% CI |
|---|---|---|---|---|
| 1 | Theme Isolation | 25 | 0.5941 | [0.5170, 0.6702] |
| 2 | Meta-reflection | 5 | 0.5863 | [0.4611, 0.7116] |
| 3 | Emphasis Focus | 15 | 0.5729 | [0.4998, 0.6512] |
| 4 | Analytical Framing | 20 | 0.5708 | [0.4880, 0.6501] |
| 5 | Question Form | 20 | 0.5410 | [0.4554, 0.6317] |
| 6 | Persona / Tone | 30 | 0.5154 | [0.4455, 0.5865] |
| 7 | Lexical Filter | 15 | 0.5097 | [0.4355, 0.5783] |
| 8 | Tone Register | 20 | 0.4884 | [0.4261, 0.5518] |
| 9 | Role Based | 20 | 0.4858 | [0.4171, 0.5608] |
| 10 | Audience Perspective | 15 | 0.4814 | [0.4047, 0.5584] |
| 11 | Completeness Framing | 15 | 0.4671 | [0.3985, 0.5380] |
| 12 | Direct Command | 20 | 0.4448 | [0.3772, 0.5187] |
| 13 | Constraint Based | 15 | 0.3617 | [0.2775, 0.4449] |
| 14 | Output Format | 15 | 0.3496 | [0.2585, 0.4548] |

## 7. Diagnosis matrix (quality x sensitivity)
- True Robustness: 64
- Fragile: 61
- Consistently Poor: 61
- Unreliable: 64

## 8. Significance tests
- **Pool A vs Pool B — seed-level Mann-Whitney U (headline, valid unit):** n_A=35 seeds, n_B=15 seeds, U=354.0, p=0.05404, rank-biserial r=-0.3486 (Pool B more sensitive)
  - _per-cell (pseudoreplicated, reference only — anticonservative, not for inference):_ p=0.01303
- Most-vs-least sensitive seed: not run (insufficient paired articles).

## 9. Warnings
- N_ARTICLES=5 is below the floor of 30 — significance results may be underpowered.
- Most-vs-least seed test skipped: only 5 paired articles (need >= 20).
- 89 NLI pairs had an output too long for the premise to fit any context; scored as neutral (0.5).
