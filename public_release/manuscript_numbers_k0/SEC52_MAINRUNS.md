# Section 5.2 material rebuilt from the MAIN 38-instance runs (replaces the restored-density probe)

Source: merged_analysis_39/aggregate_merged.csv (every individual of every generation). Best-so-far uses the run's own default-seed scores (penalised individuals excluded).

## 1. Mean best-so-far ATT (s) by generation

**J_k9_6od_x1.0**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 531.4 | 458.5 | 418.7 | 393.9 | 384.1 | 363.6 |
| init | 538.4 | 485.0 | 413.2 | 398.7 | 387.8 | 356.4 |
| mutation | 531.4 | 517.8 | 475.4 | 429.2 | 397.2 | 371.7 |
| both | 538.4 | 500.4 | 437.5 | 432.2 | 365.1 | 361.3 |

**network A (mean over its instances)**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 278.5 | 268.1 | 262.8 | 256.9 | 251.3 | 247.4 |
| init | 271.9 | 264.7 | 259.4 | 255.9 | 249.5 | 246.4 |
| mutation | 278.5 | 263.1 | 257.8 | 251.9 | 247.4 | 245.3 |
| both | 271.9 | 261.6 | 258.0 | 252.8 | 247.2 | 244.4 |

**network B (mean over its instances)**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 621.0 | 616.4 | 612.7 | 609.5 | 604.5 | 601.9 |
| init | 615.4 | 611.0 | 608.9 | 606.0 | 603.2 | 601.6 |
| mutation | 621.0 | 612.8 | 609.5 | 605.6 | 602.3 | 600.6 |
| both | 615.4 | 610.5 | 608.0 | 604.6 | 601.1 | 599.5 |

**network C (mean over its instances)**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 625.1 | 620.8 | 618.3 | 614.6 | 611.5 | 609.7 |
| init | 622.7 | 618.2 | 616.8 | 614.0 | 610.0 | 608.9 |
| mutation | 625.1 | 619.3 | 616.5 | 613.1 | 610.2 | 608.7 |
| both | 622.7 | 617.2 | 615.5 | 612.9 | 609.9 | 608.2 |

**network D (mean over its instances)**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 621.9 | 618.4 | 615.7 | 613.6 | 610.9 | 609.7 |
| init | 620.0 | 616.8 | 615.1 | 613.6 | 611.1 | 609.7 |
| mutation | 621.9 | 617.0 | 615.4 | 613.0 | 610.2 | 608.9 |
| both | 620.0 | 615.9 | 614.5 | 612.6 | 610.2 | 608.8 |

**network J (mean over its instances)**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 501.3 | 466.0 | 449.3 | 416.3 | 393.4 | 378.8 |
| init | 488.8 | 457.2 | 425.1 | 411.1 | 400.4 | 379.7 |
| mutation | 501.3 | 471.1 | 432.0 | 409.8 | 388.9 | 375.2 |
| both | 488.8 | 461.1 | 431.6 | 411.6 | 380.3 | 374.7 |

**all 38 instances**

| mode | g1 | g2 | g3 | g5 | g10 | g15 |
|---|---|---|---|---|---|---|
| none | 540.6 | 532.7 | 528.2 | 522.2 | 516.6 | 513.3 |
| init | 535.9 | 529.1 | 524.2 | 520.6 | 516.2 | 513.0 |
| mutation | 540.6 | 530.5 | 524.5 | 519.2 | 514.5 | 511.9 |
| both | 535.9 | 528.1 | 523.7 | 519.2 | 513.4 | 511.3 |

## 2. Generation-1 lead of guided initialisation and where it vanishes (per instance, default seed)

lead_g = mean bsf(none) − mean bsf(init) at generation g (positive = init ahead); same for both. 'gone at' = first generation where the lead ≤ 0 (— if never).

| instance | lead init g1 | init gone at | init lead g15 | lead both g1 | both gone at | both lead g15 |
|---|---|---|---|---|---|---|
| A_k4_1od_x0.8 | +1.4 | 8 | -0.5 | +1.4 | 2 | +1.2 |
| A_k4_1od_x1.0 | -1.3 | 1 | +3.0 | -1.3 | 1 | +4.0 |
| A_k4_2od_x0.8 | +4.3 | 3 | -0.1 | +4.3 | 4 | +0.3 |
| A_k4_2od_x1.0 | +8.9 | — | +4.1 | +8.9 | — | +9.0 |
| A_k4_2od_x1.2 | +23.6 | 4 | +0.7 | +23.6 | — | +4.3 |
| A_k4_3od_x0.8 | +4.0 | 2 | +1.8 | +4.0 | — | +4.7 |
| A_k4_3od_x1.0 | +3.4 | 2 | -1.3 | +3.4 | 4 | +1.2 |
| A_k4_3od_x1.2 | +8.8 | 6 | -0.1 | +8.8 | 6 | -0.9 |
| B_k9_1od_x0.8 | +1.0 | 10 | -1.2 | +1.0 | — | +6.5 |
| B_k9_1od_x1.0 | +5.0 | 14 | -1.0 | +5.0 | — | +2.1 |
| B_k9_1od_x1.2 | +7.4 | — | +1.9 | +7.4 | — | +4.1 |
| B_k9_2od_x0.8 | +19.8 | 10 | -2.9 | +19.8 | 10 | +0.2 |
| B_k9_2od_x1.0 | -5.1 | 1 | +1.0 | -5.1 | 1 | -1.5 |
| B_k9_2od_x1.2 | +0.8 | 3 | -0.1 | +0.8 | 7 | -0.4 |
| B_k9_3od_x0.8 | +6.3 | — | +1.7 | +6.3 | 7 | +1.3 |
| B_k9_3od_x1.0 | +8.8 | — | +4.5 | +8.8 | — | +7.8 |
| B_k9_3od_x1.2 | +6.0 | 7 | -1.2 | +6.0 | — | +1.2 |
| C_k9_1od_x0.8 | +8.3 | 12 | +0.3 | +8.3 | — | +2.2 |
| C_k9_1od_x1.0 | -1.1 | 1 | +0.6 | -1.1 | 1 | +1.0 |
| C_k9_1od_x1.2 | +8.1 | — | +1.7 | +8.1 | — | +6.9 |
| C_k9_2od_x0.8 | +0.8 | 3 | -1.1 | +0.8 | 5 | -1.7 |
| C_k9_2od_x1.0 | +2.1 | — | +5.0 | +2.1 | — | +1.2 |
| C_k9_2od_x1.2 | -1.6 | 1 | +2.9 | -1.6 | 1 | -0.2 |
| C_k9_3od_x0.8 | -5.1 | 1 | +1.1 | -5.1 | 1 | -2.4 |
| C_k9_3od_x1.0 | +9.5 | 11 | -4.1 | +9.5 | — | +3.9 |
| C_k9_3od_x1.2 | +0.2 | 4 | +1.0 | +0.2 | 4 | +2.1 |
| D_k9_1od_x0.8 | -0.9 | 1 | -1.0 | -0.9 | 1 | -0.8 |
| D_k9_1od_x1.0 | +3.2 | 4 | -1.8 | +3.2 | 3 | -0.2 |
| D_k9_1od_x1.2 | +0.4 | 15 | -0.1 | +0.4 | — | +0.3 |
| D_k9_2od_x0.8 | +1.8 | — | +0.1 | +1.8 | — | +1.1 |
| D_k9_2od_x1.0 | +1.4 | 5 | +0.0 | +1.4 | 10 | +0.9 |
| D_k9_2od_x1.2 | +1.1 | 2 | +0.1 | +1.1 | 3 | +0.6 |
| D_k9_3od_x0.8 | +5.7 | 7 | +0.4 | +5.7 | 7 | +0.5 |
| D_k9_3od_x1.0 | +6.9 | 3 | +4.3 | +6.9 | 4 | +5.4 |
| D_k9_3od_x1.2 | -2.4 | 1 | -2.1 | -2.4 | 1 | +0.1 |
| J_k9_6od_x0.8 | +8.6 | 5 | -0.1 | +8.6 | 6 | -2.7 |
| J_k9_6od_x1.0 | -7.0 | 1 | +7.2 | -7.0 | 1 | +2.3 |
| J_k9_6od_x1.2 | +35.7 | 7 | -9.9 | +35.7 | — | +12.6 |

init: ahead of none at g1 in 30/38 instances (median lead +3.3 s); lead vanishes at some generation in 31/38; still ahead at g15 in 21/38 (median +0.1 s).

both: ahead of none at g1 in 30/38 instances (median lead +3.3 s); lead vanishes at some generation in 23/38; still ahead at g15 in 29/38 (median +1.2 s).

## 3. Population diversity (mean over seeds; population 24)

Generation-1 figures are exact for all 38 instances. Later generations: the merged file omits cache hits (mean 22.9 of 24 individuals listed from generation 2 on), so distinct-edge counts at g15 are lower bounds and genotype counts at g>=2 are only available for the three A_3od instances read from raw run logs.

**J_k9_6od_x1.0**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 24.0 | 199.8 | 192.3 | 9.00 | 9.00 |
| init | 22.3 | 80.6 | 195.1 | 4.46 | 9.00 |
| mutation | 24.0 | 199.8 | 173.5 | 9.00 | 9.00 |
| both | 22.3 | 80.6 | 175.8 | 4.46 | 9.00 |

**network A**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 21.8 | 27.6 | 32.8 | 2.00 | 4.00 |
| init | 18.9 | 25.9 | 33.4 | 1.98 | 4.00 |
| mutation | 21.8 | 27.6 | 30.7 | 2.00 | 4.00 |
| both | 18.9 | 25.9 | 30.6 | 1.98 | 4.00 |

**network B**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 24.0 | 64.4 | 75.7 | 4.64 | 9.00 |
| init | 22.1 | 51.8 | 75.7 | 4.46 | 9.00 |
| mutation | 24.0 | 64.4 | 65.2 | 4.64 | 9.00 |
| both | 22.1 | 51.8 | 65.2 | 4.46 | 9.00 |

**network C**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 24.0 | 67.7 | 79.4 | 4.82 | 9.00 |
| init | 22.3 | 59.4 | 78.6 | 4.46 | 9.00 |
| mutation | 24.0 | 67.7 | 73.0 | 4.82 | 9.00 |
| both | 22.3 | 59.4 | 73.0 | 4.46 | 9.00 |

**network D**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 24.0 | 67.3 | 78.5 | 4.78 | 9.00 |
| init | 22.3 | 56.3 | 78.6 | 4.46 | 9.00 |
| mutation | 24.0 | 67.3 | 69.3 | 4.78 | 9.00 |
| both | 22.3 | 56.3 | 68.9 | 4.46 | 9.00 |

**network J**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 24.0 | 199.8 | 193.0 | 9.00 | 9.00 |
| init | 22.3 | 89.5 | 193.6 | 4.46 | 9.00 |
| mutation | 24.0 | 199.8 | 178.4 | 9.00 | 9.00 |
| both | 22.3 | 89.5 | 179.3 | 4.46 | 9.00 |

**all 38 instances**

| mode | distinct genotypes g1 (of 24) | distinct edges g1 | distinct edges g15 (lower bound) | mean k g1 | mean k g15 |
|---|---|---|---|---|---|
| none | 23.5 | 68.8 | 77.5 | 4.50 | 7.95 |
| init | 21.5 | 52.2 | 77.5 | 3.94 | 7.95 |
| mutation | 23.5 | 68.8 | 69.7 | 4.50 | 7.95 |
| both | 21.5 | 52.2 | 69.6 | 3.94 | 7.95 |

Generation-1 diversity of the guided-initialisation modes relative to the unguided modes, per instance: genotypes median ratio 0.93 (min 0.77, max 0.93; below 0.75 in 0/38 instances); distinct edges median ratio 0.86 (min 0.40, max 1.05).

**Full-population diversity by generation, A_3od pm=0.1 instances (raw logs, all 24 individuals)**

| mode | genotypes g1 | g2 | g3 | g5 | g10 | g15 | edges g1 | edges g15 |
|---|---|---|---|---|---|---|---|---|
| none | 21.6 | 24.0 | 24.0 | 23.9 | 23.9 | 24.0 | 27.0 | 32.3 |
| init | 19.3 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 27.8 | 32.9 |
| mutation | 21.6 | 24.0 | 24.0 | 23.8 | 23.9 | 24.0 | 27.0 | 32.0 |
| both | 19.3 | 24.0 | 23.9 | 24.0 | 24.0 | 23.9 | 27.8 | 31.8 |