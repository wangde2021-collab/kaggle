# Phase 4 tables (from frozen round-5 JSON)

## Table 1 — Model comparison (10×5 repeated CV, seed 42)

| Model | Y1 hurdle AUC | Y1 stage-2 r (log snaps) | Y2 r (per-play quality) |
|---|---|---|---|
| Draft pick only (log pick) (M0a) | 0.8631 | 0.6611 | 0.1009 |
| Stopwatch only (40-yd, 10-yd, vertical, broad jump) (M0b_stopwatch) | 0.5968 | 0.4681 | -0.0016 |
| NGS scores only (athleticism + final) (M0c_ngs) | 0.8082 | 0.5986 | 0.1984 |
| Stopwatch + NGS (legacy combined) (M0b) | 0.8022 | 0.5862 | 0.1756 |
| Pick + stopwatch + NGS (M0all) | 0.8640 | 0.6559 | 0.1829 |
| Tracking, single-stage Ridge (legacy) (M1b) | 0.7873 | 0.5483 | 0.1587 |
| Tracking only (vs stopwatch) (M1_track_only) | 0.5688 | 0.4245 | 0.0902 |
| Tracking + shadow (stacked on pick) (M1a) | 0.8651 | 0.6605 | 0.1403 |
| Tracking + pick + shadow (stacked) (M1) | 0.8641 | 0.6607 | 0.1335 |
| Tracking + pick + shadow + static (stacked) (M1full) | 0.8643 | 0.6609 | 0.1279 |
| Tracking stacked on stopwatch (M1_over_stopwatch) | 0.5724 | 0.4669 | 0.0755 |
| Tracking stacked on pick + stopwatch + NGS (M1_over_all) | 0.8600 | 0.6559 | 0.1823 |

Key deltas (Y2, mean Δr with 95% paired bootstrap CI):
- delta_M1_over_stopwatch_vs_M0b_stopwatch: +0.0771 [+0.0122, +0.1545]
- delta_M1_track_only_vs_M0b_stopwatch: +0.0918 [-0.0076, +0.2217]
- delta_M1a_vs_M0a: +0.0394 [-0.0097, +0.0951]
- delta_M1_vs_M0a: +0.0326 [-0.0114, +0.0819]
- delta_M1_over_all_vs_M0all: -0.0006 [-0.0169, +0.0158]
- Permutation test (M1 vs M0a, Y2): obs Δr = +0.0415, p = 0.0030 (null -0.0051 ± 0.0090, n_perm = 1000).
- Y2 R²: M1 0.0175 vs M0a 0.0098. Y2 n = 299 (≥75 regular-season plays); Y1 n = 510.

## Table 2 — Per-position out-of-fold ranking (Y2, M1)

| Position | n | Spearman ρ | p |
|---|---|---|---|
| DL | 73 | +0.507 | 4.78e-06 |
| WR | 66 | +0.242 | 0.0505 |
| DB | 71 | -0.166 | 0.168 |
| OL | 67 | -0.036 | 0.773 |
| TE | 22 | -0.158 | 0.484 |
| EDGE (DL split) | 45 | +0.438 | 0.00263 |
| IDL (DL split) | 28 | +0.313 | 0.105 |

