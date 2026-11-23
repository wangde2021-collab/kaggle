# Phase 4 figure captions and tables (from notes/writeup_draft.md)

## Figures

**Figure 1 — Double dissociation.** Out-of-fold increments, tracking (M1) over draft pick (M0a); Y1 n = 510/236, Y2 n = 299. Bars: 95% paired bootstrap CIs.

**Figure 2 — Beyond the stopwatch.** (a) Absolute out-of-fold r, per-play quality (Y2, n = 299). (b) Incremental gains (95% paired bootstrap CIs); bottom row is the honest boundary.

**Figure 3 — Differential market pricing efficiency.** (a) Draft pricing: β on −log(pick), by position group and draft year (95% CIs; n = 405–510, 239–299; BH-FDR × 14). (b) Same features on per-play quality z, given log pick, UDFA, position, draft year. The market pays for straight-line speed; only skill-drill peak speed also predicts quality.

**Figure 4 — Robustness ladder for skill_s_peak.** Standardised β for skill_s_peak across five control sets (UDFA/position/year always controlled; 95% CIs). Attenuation: attrition −0.020 (32.5%), shared 40-yard variance −0.042 (67.5%).

**Figure 5 — Role-specific kinematic translation.** Within-subgroup Pearson r and Y2 Spearman ρ by defensive-line role: EDGE (KTR 43–56, Y2 45) vs IDL (KTR 34–42, Y2 28).

**Figure 6 — Combine-to-pads transfer.** (a) Within-position partial r, combine vs game, by position group (40-yard rows n = 319; skill rows n = 398–401; * p < 0.05, ** p < 0.01, *** p < 0.001). (b) Mean game top speed as % of 40-yard peak speed.

**Figure 7 — Cross-cohort and temporal robustness.** (a) LOCO Y2 r (n = 112/104/83). (b) 2025 forward, Y1 (n_test 173/125). (c) Y2 threshold sensitivity (n = 299/286/261; 2×5 CV). (d) Adversarial validation (n = 510; chance 0.50).

## Tables

**Table 1 — Model comparison (10×5 repeated CV, seed 42).**

| Model | Y1 hurdle AUC (n=510) | Y1 stage-2 r (n=236) | Y2 r (n=299) |
|---|---|---|---|
| M0a | 0.8631 | 0.6611 | 0.1009 |
| M0b_stopwatch | 0.5968 | 0.4681 | −0.0016 |
| M0c_ngs | 0.8082 | 0.5986 | 0.1984 |
| M0all | 0.8640 | 0.6559 | 0.1829 |
| M1_track_only | 0.5688 | 0.4245 | 0.0902 |
| M1a | 0.8651 | 0.6605 | 0.1403 |
| M1 | 0.8641 | 0.6607 | 0.1335 |
| M1_over_stopwatch | 0.5724 | 0.4669 | 0.0755 |
| M1_over_all | 0.8600 | 0.6559 | 0.1823 |

Deltas in text use unrounded means.

**Table 2 — Per-position out-of-fold ranking (Y2, M1).**

| Position | n | Spearman ρ | p |
|---|---|---|---|
| DL | 73 | +0.507 | 4.8×10⁻⁶ |
| WR | 66 | +0.242 | 0.051 |
| DB | 71 | −0.166 | 0.168 |
| OL | 67 | −0.036 | 0.773 |
| TE | 22 | −0.158 | 0.484 |
| EDGE (DL split) | 45 | +0.438 | 0.0026 |
| IDL (DL split) | 28 | +0.313 | 0.105 |
