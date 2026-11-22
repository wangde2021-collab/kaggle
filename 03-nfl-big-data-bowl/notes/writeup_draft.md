# Beyond the Stopwatch: What 10 Hz Combine Tracking Adds — and What It Does Not — to Draft Evaluation

<!-- DRAFT STATUS: Phase 5 draft for review (DRAFT — NOT SUBMITTED). Word count target 1,850–1,950 including captions and tables; counted by notes/count_writeup_words.py. Kaggle Notebook link to be filled after Step 3. All numbers from frozen notes/phase3_results_round5_server.json; audited by notes/audit_writeup_numbers.py. -->

## Abstract

We ask whether 10 Hz tracking from the NFL Scouting Combine predicts regular-season performance beyond what draft position already prices in. Using 510 players from the 2023–2025 draft classes, we build two targets: snap volume (Y1, n = 510) and empirical-Bayes per-play quality (Y2, n = 299, ≥75 regular-season plays). A double dissociation emerges. Draft pick almost fully explains snap volume — adding tracking changes hurdle AUC by +0.0009 and stage-2 r by −0.0005 — yet for per-play quality it beats the stopwatch by Δr = +0.0771 (95% CI [+0.0122, +0.1545]) and draft pick alone by +0.0394. The gain is bounded: tracking cannot beat draft pick plus the official NGS composite (Δr = −0.0006). The market prices straight-line speed efficiently and ignores the rest.

## 1. Question and data

The Combine is the league's most-watched job interview, yet its headline numbers — the 40-yard dash, vertical jump, broad jump — are static snapshots of straight-line ability. Since 2023 every drill has also been recorded at 10 Hz, capturing dynamics no stopwatch sees. We test whether that detail predicts who plays well on Sundays, once draft position is accounted for.

Our sample is 510 players from the 2023–2025 draft classes, with combine results and tracking (463,189 frames, 28 stable drills) and regular-season game tracking. Y1 is snap volume: a two-stage hurdle (clearing 150 snaps, then log snaps). Y2 is per-play quality: position-specific regular-season efficiency, shrunk with empirical-Bayes (James–Stein) weights and standardised within position, for the 299 players clearing 75 plays. Evaluation is out-of-fold throughout: 10×5 repeated stratified CV, 1,000 permutations, 2,000 paired bootstraps (seed 42).

## 2. Methods

**Features.** The 40-yard dash is decomposed with a Furusawa–Hill sprint curve (free onset t0), yielding peak speed, time constant, and initial acceleration. Skill drills give peak speed, acceleration, deceleration, lateral acceleration, jerk, braking power, and cut dynamics. Static tests and the official NGS scores enter as baselines. A Skill-Drill Movement Index (Shadow Agility Proxy) reconstructs agility for players missing timed tests: it predicts 3-cone time at r = 0.421 (Pure) and r = 0.494 (Full); short shuttle r = 0.284 and r = 0.446.

**Harmonisation.** Sensor calibrations drift by year, so every drill and tracking feature is standardised within its own (position × draft-year) cohort — Within-Cohort Unsupervised Harmonisation — using only unlabelled distributions, never the target. Unseen cohorts fall back through three levels (cohort, position, global median), with a 5% floor on within-group standard deviations. Afterwards no feature block can predict draft year (adversarial macro AUC 0.4879–0.4924, chance 0.50).

**Models.** Baselines are draft pick alone (M0a, log pick), the four static tests (M0b_stopwatch), NGS scores alone (M0c_ngs), and all three combined (M0all). Tracking models are two-stage stacks: a baseline plus a Ridge on tracking residuals, or a baseline logit as a fixed offset. Headline models stack tracking on pick (M1a, M1), on the stopwatch (M1_over_stopwatch), and on all baselines (M1_over_all). A LightGBM variant (M3) tests for non-linear signal. Single-feature effects use HC3 robust standard errors (recovered from β and p for plotting) with Benjamini–Hochberg FDR control over 14 pre-registered kinematic features.

## 3. Findings

### 3.1 A double dissociation (Figure 1, Table 1)

Draft capital buys playing time, not per-play efficiency. For snap volume, tracking is indistinguishable from draft pick alone: hurdle AUC 0.8641 vs 0.8631 (Δ +0.0009, 95% CI [−0.0079, +0.0065], permutation p = 0.0599) and stage-2 r 0.6607 vs 0.6611 (Δ −0.0005, 95% CI [−0.0011, +0.0003], p = 0.1168). Tracking lifts per-play quality from r = 0.1009 to r = 0.1335 (Δ +0.0326, 95% CI [−0.0114, +0.0819]; M1a Δ +0.0394, 95% CI [−0.0097, +0.0951]; R² 0.0098 → 0.0175). The Y1 nulls are precise to the third decimal while the Y2 gain holds across every contrast.

### 3.2 Beyond the stopwatch (Figure 2)

Three layers. **Layer 1 — tracking beats the stopwatch head-to-head.** The four static tests have no signal for per-play quality at all (r = −0.0016), while tracking stacked on them reaches r = 0.0755 (Δ +0.0771, 95% CI [+0.0122, +0.1545]); tracking alone reaches r = 0.0902 (Δ +0.0918, 95% CI [−0.0076, +0.2217]). **Layer 2 — tracking adds signal on top of draft pick.** M1a vs M0a gives Δ +0.0394 and M1 vs M0a Δ +0.0326; the conditional signal is real (permutation p = 0.0030), but the out-of-sample net gain is modest and its 95% CI just crosses zero. **Layer 3 — the honest boundary.** Stacking tracking on pick, stopwatch, and the official NGS composite adds nothing (r = 0.1823 vs 0.1829, Δ −0.0006, 95% CI [−0.0169, +0.0158]; NGS alone already reaches r = 0.1984). Tracking's value is real but circumscribed: it replaces the stopwatch and sharpens the pick, but does not beat the league's own composite.

### 3.3 One feature survives everything (Figures 4 and 6)

Of 14 pre-registered kinematic features, exactly one passes BH-FDR for per-play quality: skill-drill peak speed (skill_s_peak), β = +0.156, p = 0.0003, q = 0.004 (n = 299), still significant after adding the NGS final score (β = +0.138, p = 0.0027, n = 298). A five-step ladder decomposes the attenuation: on the 253 players with a recorded 40, controlling only for pick still gives β = +0.136 (p = 0.0030); adding the 40-yard time drops it to β = +0.094 (p = 0.077); full controls give β = +0.093 (p = 0.089). About one-third of the attenuation (−0.020, 32.5%) is sample attrition (46 players missing a 40-yard time); about two-thirds (−0.042, 67.5%) is variance shared with straight-line speed. So skill_s_peak carries signal neither pick nor the NGS total absorbs, and covers the 46 players the stopwatch cannot see.

The market pays heavily for static tests — −40-yard time β = +0.271, vertical +0.232, 40-yard peak speed +0.225, broad jump +0.221 on −log(pick) — yet none predicts per-play quality given pick (β = +0.048, +0.083, +0.048, +0.035; all p > 0.10). Skill_s_peak is the exception in both columns: priced (β = +0.153, q = 0.001) and predictive (β = +0.156, q = 0.004). The market prices straight-line speed efficiently and ignores the rest — where tracking's value lives.

### 3.4 Where tracking works — and where it does not (Figure 3, Table 2)

Defensive line is the standout (Spearman ρ = +0.507, p = 4.8×10⁻⁶, n = 73); wide receiver is marginally significant (ρ = +0.242, p = 0.051, n = 66); offensive line (ρ = −0.036), defensive back (ρ = −0.166), and tight end (ρ = −0.158, n = 22) show no signal. Initiator positions convert athleticism directly; reactive and collaborative roles depend on scheme, teammates, and quarterback decisions.

Edge rushers translate speed, acceleration, and braking into game movement (top speed r = +0.389, p = 0.010; deceleration r = +0.357, p = 0.007; Y2 ρ = +0.438, p = 0.0026), while interior linemen show no speed or deceleration transfer at all (r = −0.066 and −0.031) and instead convert lateral acceleration (r = +0.396, p = 0.009; Y2 ρ = +0.313, p = 0.105).

### 3.5 Speed survives the pads, unevenly (Figure 7)

Combine speed transfers to game speed: within-position partial correlations are uniformly positive (40-yard peak speed vs game top speed r = +0.363, p = 2.2×10⁻¹¹, n = 319), and game top speed as a share of 40-yard peak speed gradients by position — WR 81.5% (9.03/11.08 yd/s), TE 75.5%, DB 73.3%, DL 62.4%, OL 51.7%. But transfer is not mediation: game deceleration mediates −3.3% (indirect −0.0050, 95% CI [−0.0335, +0.0237]) and a one-second burst +7.8% (+0.0117, 95% CI [−0.0158, +0.0441]); both intervals cross zero. A shallow LightGBM adds nothing (Y2 r = 0.1457 vs 0.1403; Y1 hurdle AUC 0.8462 vs 0.8641).

## 4. Robustness (Figure 5)

Leave-one-cohort-out: M1 beats pick in 2024 (r = 0.1315 vs 0.0938) and 2025 (0.0513 vs 0.0106), ties in 2023 (0.1831 vs 0.1971). The 2025 forward extrapolation confirms the dissociation (hurdle AUC 0.8670 vs 0.8756; log-snaps r 0.6241 vs 0.6293). Tightening the play-count threshold keeps Y2 positive (≥100 plays: r = 0.1768, n = 286; ≥150 plays: r = 0.1116, n = 261; 2×5 CV, vs main r = 0.1335 at ≥75 plays, n = 299, 10×5 CV).

## 5. Limitations

Three limits bound the claims. First, scale: Y2 rests on 299 players and the 2025 cohort on 83, so cross-cohort correlations carry sampling errors near 0.11 and the 2025 LOCO estimate is imprecise. Second, the quality target is itself noisy — for defensive backs, the EPA-based component's empirical-Bayes shrinkage approaches its cap, signalling almost no between-player signal in one season. Third, the sample is Combine invitees who were drafted or signed; tracking says nothing about players the Combine never measures. All results are predictive, not causal.

## 6. Conclusion

Draft position decides who gets on the field; tracking judges what they do once there. Tracking decisively beats the stopwatch (Δr = +0.0771, CI excludes zero), adds a real but modest conditional signal on top of draft pick (permutation p = 0.0030), and stops at the league's own composite (Δr = −0.0006). The actionable piece is small: skill-drill peak speed is the single measure both priced by the market and predictive of per-play quality after every control; it survives the NGS total and covers the 46 players the 40-yard dash misses. Price functional speed alongside straight-line speed; the stopwatch's monopoly is over.

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

## Data and code availability

Code (8 Python modules, MIT licence) and the frozen result file are public at https://github.com/wangde2021-collab/kaggle/tree/main/03-nfl-big-data-bowl . The companion Kaggle Notebook ([Kaggle Notebook link to be added]) reproduces the pipeline from the competition data (CC BY-NC 4.0); raw data are not redistributed. Full pipeline: ~5.8 hours on a 2-vCPU machine (seed 42); quick smoke mode included.
