# Phase 4 figure captions and tables (from notes/writeup_draft.md)

## Figures
**Figure 1 — Double dissociation.** Out-of-fold increments of tracking (M1) over draft pick (M0a). Bars: 95% paired bootstrap CIs.

**Figure 2 — Beyond the stopwatch.** (a) Absolute out-of-fold r, per-play quality (Y2, n = 299). (b) Incremental gains (95% paired bootstrap CIs); bottom row is the honest boundary.

**Figure 3 — Role-specific kinematic translation.** Within-position partial correlations and Y2 Spearman ρ by defensive-line role: EDGE (n = 66) vs IDL (n = 52).

**Figure 4 — Differential market pricing efficiency.** (a) Draft pricing: β on −log(pick), controlling for position group and draft year. (b) Same features on per-play quality z, controlling for log pick, undrafted status, position, and draft year. The market pays for straight-line speed; only skill-drill peak speed also predicts quality.

**Figure 5 — Cross-cohort and temporal robustness.** (a) Leave-one-cohort-out Y2 r. (b) 2025 forward extrapolation, Y1. (c) Y2 sensitivity to threshold (2×5 CV). (d) Adversarial validation (chance 0.50).

**Figure 6 — Robustness ladder for skill_s_peak.** Standardised β for skill_s_peak across five control sets. Attenuation: sample attrition −0.020 (32.5%), shared 40-yard variance −0.042 (67.5%).

**Figure 7 — Combine-to-pads transfer.** (a) Within-position partial correlations, combine vs game (* p < 0.05, ** p < 0.01, *** p < 0.001). (b) Game top speed as % of 40-yard peak speed.

## Tables
**Table 1 — Model comparison (10×5 repeated CV, seed 42).**

| Model | Y1 hurdle AUC | Y1 stage-2 r | Y2 r |
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
