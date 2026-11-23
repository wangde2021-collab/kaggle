# Beyond the Stopwatch: What 10 Hz Combine Tracking Adds — and Does Not Add — to Draft Evaluation


## Abstract

We ask whether 10 Hz Combine tracking predicts regular-season performance beyond what draft position prices in. From 510 players in the 2023–2025 classes we build targets: snap volume (Y1) and empirical-Bayes per-play quality (Y2, n = 299, ≥75 plays). A double dissociation emerges: pick almost fully explains snap volume — tracking (M1) changes hurdle AUC by +0.0009 and stage-2 r by −0.0005 — yet for per-play quality it beats the stopwatch by Δr = +0.0771 (95% CI [+0.0122, +0.1545]) and pick alone by +0.0326 (permutation p = 0.0030). The gain is bounded: tracking cannot beat pick, stopwatch, and the NGS composite combined (Δr = −0.0006). The market fully prices straight-line speed but underprices functional skill-drill speed.

## 1. Question and data

The Combine's headline numbers — 40-yard dash, vertical, broad jump — are static snapshots of straight-line ability. Since 2023 every drill is also recorded at 10 Hz; we test whether that detail predicts Sunday performance, net of draft position.

Our sample: 510 players from the 2023–2025 classes, with combine results and tracking (431,094 player frames; 31 stable drills, 28 skill) and game tracking. Y1 is exposure-adjusted snap rate (ASR = snaps / eligible seasons): a two-stage hurdle (≥ 150 snaps/season; log ASR among n = 236 passers). Y2 is per-play quality: position-specific regular-season efficiency, empirical-Bayes shrunk (James–Stein), standardised within position, for 299 players clearing 75 plays. All evaluation is out-of-fold: 10×5 repeated CV (stratified for hurdles), 333 three-repeat-averaged permutation draws, 2,000 position-stratified paired bootstraps (seed 42).

## 2. Methods

**Features.** A Furusawa–Hill sprint curve (free onset t0) decomposes the 40 dash into peak speed, time constant, initial acceleration. Skill drills give peak speed, acceleration, deceleration, lateral acceleration, cuts. A Skill-Drill Movement Index (Shadow Agility Proxy) reconstructs agility for players missing timed tests (within-position OOF, 3-cone r = 0.421/0.494, n = 180; shuttle r = 0.284/0.446, n = 201).

**Harmonisation.** Sensor calibrations drift by year; every drill and tracking feature is standardised within its (position × draft-year) cohort — Within-Cohort Unsupervised Harmonisation — using unlabelled distributions only, with three-level median fallback (cohort, position, global) and 5% floor on group SDs. Afterwards no feature block predicts draft year (adversarial macro AUC 0.4879–0.4924; chance 0.50).

**Models.** Baselines: pick alone (M0a: log pick, UDFA, draft year, position effects), the four static tests (M0b_stopwatch), NGS scores alone (M0c_ngs), all three combined (M0all). Tracking models are two-stage stacks: baseline plus a Ridge on tracking residuals, or a baseline logit offset. M1a stacks pure tracking plus the Shadow Proxy on pick; M1 adds four static ratios; further stacks sit on the stopwatch (M1_over_stopwatch) and on all baselines (M1_over_all). A LightGBM variant (M3) tests nonlinearity. Single-feature effects use HC3 robust standard errors (recovered from β and p for plotting), Benjamini–Hochberg FDR over 14 preregistered kinematic features.

## 3. Findings

### 3.1 A double dissociation (Figure 1, Table 1)

Draft capital buys playing time, not per-play efficiency. For snap volume, tracking is indistinguishable from pick: hurdle AUC 0.8641 vs 0.8631 (Δ +0.0009, 95% CI [−0.0079, +0.0065], permutation p = 0.0599) and stage-2 r 0.6607 vs 0.6611 (Δ −0.0005, 95% CI [−0.0011, +0.0003], p = 0.1168). On per-play quality (Y2), tracking lifts r from 0.1009 to 0.1335 (M1; R² 0.0098 → 0.0175) and 0.1403 (M1a; Table 1). The Y1 nulls are precise to the third decimal, while Y2 improves over the stopwatch and pick before plateauing at the NGS composite.

### 3.2 Beyond the stopwatch (Figure 2)

**Layer 1 — tracking beats the stopwatch.** The four static tests have no signal for per-play quality (r = −0.0016); tracking stacked on them reaches r = 0.0755 (Δ +0.0771, 95% CI [+0.0122, +0.1545]). **Layer 2 — tracking adds signal on top of pick.** M1a vs M0a gives Δ +0.0394 and M1 vs M0a Δ +0.0326 (95% CI [−0.0114, +0.0819]); the conditional signal is real (permutation p = 0.0030), but the net out-of-sample gain is modest and its 95% CI just crosses zero. **Layer 3 — the honest boundary.** Stacking tracking on pick, stopwatch, and the NGS composite adds nothing (r = 0.1823 vs 0.1829, Δ −0.0006, 95% CI [−0.0169, +0.0158]; NGS alone reaches r = 0.1984). Tracking's value is real but circumscribed: it replaces the stopwatch and sharpens the pick, not the league's composite.

### 3.3 One feature survives FDR (Figures 3 and 4)

Of 14 preregistered kinematic features, exactly one passes BH-FDR for per-play quality: skill-drill peak speed (skill_s_peak), β = +0.156, p = 0.0003, q = 0.004 (n = 299), still significant after the NGS final score (β = +0.138, p = 0.0027, n = 298). A five-step ladder decomposes the attenuation on 253 players with a recorded 40: β = +0.136 (pick only, p = 0.0030), +0.094 (adding 40 time, p = 0.077), +0.093 (full controls, p = 0.089, n = 252). Sample attrition explains one-third of the attenuation (−0.020, 32.5%; 46 of 299 Y2 players lack a 40); shared 40 variance explains two-thirds (−0.042, 67.5%). skill_s_peak carries signal neither pick nor the NGS total absorbs, covering the 46 players the stopwatch cannot see.

The market pays for straight-line speed and jumps — −40 time β = +0.271, vertical +0.232, 40 peak speed +0.225, broad jump +0.221 on −log(pick) — yet none predicts per-play quality given pick (β = +0.048, +0.083, +0.048, +0.035; all p > 0.10). Skill_s_peak is the exception in both columns: priced (β = +0.153, q = 0.001), predictive (β = +0.156, q = 0.004). Straight-line speed and jumps are exhausted; functional skill-drill speed is only partially priced.

### 3.4 Where tracking works — and where it does not (Figure 5, Table 2)

Defensive line is the standout (ρ = +0.507, p = 4.8×10⁻⁶, n = 73); wide receiver is marginal (ρ = +0.242, p = 0.051, n = 66); the rest show no signal (Table 2).

Edge rushers translate speed, acceleration, and braking into game movement (top speed r = +0.389, p = 0.010; deceleration r = +0.357, p = 0.007; Y2 ρ = +0.438, p = 0.0026); interior linemen show no speed or deceleration transfer (r = −0.066, −0.031), converting lateral acceleration instead (r = +0.396, p = 0.009; Y2 ρ = +0.313, p = 0.105).

### 3.5 Speed survives the pads, unevenly (Figure 6)

Combine speed transfers to game speed: within-position partial correlations are uniformly positive (40 peak speed vs game top speed r = +0.363, p = 2.2×10⁻¹¹, n = 319); game top speed as a share of 40 peak speed gradients by position — WR 81.5% to OL 51.7% (Figure 6). But transfer is not mediation: game deceleration mediates −3.3% of the Shadow Proxy → Y2 effect (indirect −0.0050, 95% CI [−0.0335, +0.0237], n = 299), a one-second burst +7.8% (+0.0117, 95% CI [−0.0158, +0.0441]); both cross zero. A shallow LightGBM on M1's features adds nothing (Y2 r = 0.1457 vs M1a 0.1403 / M1 0.1335; Y1 hurdle 0.8462 vs 0.8641).

## 4. Robustness (Figure 7)

LOCO: M1 beats pick in 2024 (r = 0.1315 vs 0.0938) and 2025 (0.0513 vs 0.0106), trails in 2023 (0.1831 vs 0.1971). The 2025 forward extrapolation confirms the Y1 null (hurdle 0.8670 vs 0.8756; log-snaps 0.6241 vs 0.6293). Tightening the play threshold keeps Y2 positive (≥100 plays: r = 0.1768, n = 286; ≥150 plays: r = 0.1116, n = 261; 2×5 CV).

## 5. Limitations

Three limits bound the claims. First, scale: Y2 rests on 299 players, the 2025 cohort on 83; cross-cohort correlations carry sampling errors near 0.11. Second, the quality target is noisy — for defensive backs, the EPA-based component's shrinkage hits its κ = 200 cap (alongside DL quick-pressure) — minimal detectable variance. Third, the sample is Combine invitees, drafted or signed — nothing here covers players the Combine never measures. All results are predictive, not causal.

## 6. Conclusion

Draft position decides who gets on the field; tracking judges what they do once there. Tracking beats the stopwatch decisively, adds a modest conditional signal over pick, and plateaus at the NGS composite. The actionable piece is small: skill-drill peak speed is the single measure both market-priced and FDR-significant for per-play quality after pick and the NGS composite (p = 0.089 with 40 time added), covering the 46 players the 40 misses. On Draft Day, use it as a tie-breaker on Day 2–3 defensive-line and receiver boards, and for the 17% missing a 40 and over 60% missing agility drills. Price functional speed alongside straight-line speed; the stopwatch should no longer stand alone.

## Figures

**Figure 1 — Double dissociation.** Out-of-fold increments, tracking (M1) over pick (M0a); Y1 n = 510/236, Y2 n = 299. Bars: 95% paired bootstrap CIs.

**Figure 2 — Beyond the stopwatch.** (a) Absolute out-of-fold r, per-play quality (Y2, n = 299). (b) Incremental gains (95% paired bootstrap CIs); bottom row: the honest boundary.

**Figure 3 — Differential market pricing efficiency.** (a) Draft pricing: β on −log(pick), by position group and year (95% CIs; n = 405–510, 239–299; BH-FDR × 14). (b) Same features on per-play quality z, given log pick, UDFA, position, draft year.

**Figure 4 — Robustness ladder for skill_s_peak.** Standardised β for skill_s_peak across five control sets (UDFA/position/year controlled; 95% CIs). Attenuation: attrition −0.020 (32.5%), shared 40 variance −0.042 (67.5%).

**Figure 5 — Role-specific kinematic translation.** Within-subgroup Pearson r and Y2 Spearman ρ by DL role: EDGE (KTR 43–56, Y2 45) vs IDL (KTR 34–42, Y2 28).

**Figure 6 — Combine-to-pads transfer.** (a) Within-position partial r, combine vs game, by position group (40 rows n = 319; skill rows n = 398–401; * p < 0.05, ** p < 0.01, *** p < 0.001). (b) Mean game top speed as % of 40 peak speed.

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

## Data and code availability

Code (8 Python modules, MIT licence) and frozen results: https://github.com/wangde2021-collab/kaggle/tree/main/03-nfl-big-data-bowl . The Kaggle Notebook ([Beyond the Stopwatch](https://www.kaggle.com/code/dewang2025/nfl-big-data-bowl-2027-beyond-the-stopwatch)) reproduces the pipeline (CC BY-NC 4.0; ~5.8 h on 2 vCPUs, seed 42).
