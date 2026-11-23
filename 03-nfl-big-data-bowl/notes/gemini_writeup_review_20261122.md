# Gemini 复核回复全文 — NFL Big Data Bowl 2027 Phase 5 成稿
<!-- 存档注记：2026-11-23 00:00 CST 左右回传（浏览器任务 browser-task:3007ef06，10 分钟）。
同一对话 https://gemini.google.com/app/8adf0ca845f706e2，账号 wangde2021@gmail.com，
Google AI Pro，Neon 深度思考、思考等级“高”。附件：notes/writeup_draft.md + notes/figure_captions_phase4.md。
未上传原始 CSV。以下为回复全文逐字转存，未改动。 -->

The draft is tight, well-disciplined, and faithful to the frozen Round-5 numbers. However, it is not ready to submit as-is: there are 2 direct internal contradictions (one in §3.1, one in Abstract/§3.3), 1 major target-definition misstatement (§1 drops exposure adjustment from Y1), out-of-order figure numbering (1, 2, 4, 6, 3, 7, 5), a sample-size clash between Figure 3 and Table 2, and a dangerously thin word-count margin (~1,941–1,949 words before fixes, which will likely exceed 2,000 under stricter tokenizers).

1. Numbers Audit (Cross-Checked Against Frozen Round-5 Results)

All 27 cells in Table 1 and all 21 cells in Table 2 match the frozen Round-5 server output exactly. Below are every numerical discrepancy, misstatement, or precision issue found in the prose and captions:

Y1 definition omits exposure adjustment (§1, para. 2)

Quote: "Y1 is snap volume: a two-stage hurdle (clearing 150 snaps, then log snaps)."

Issue: In targets.py, Y1 is annualised regular-season snaps (ASR = reg_snaps / E, where E = 2026 - draft_year ∈ {1, 2, 3}), the hurdle is ≥ 150 snaps/season (n = 236 passers), and Stage 2 is log(ASR). Writing "clearing 150 snaps, then log snaps" makes it look like you compared 3-season 2023 rookies against 1-season 2025 rookies on raw totals.

Expected wording: "Y1 is exposure-adjusted regular-season snap rate (ASR = snaps / eligible seasons): a two-stage hurdle (≥ 150 snaps/season, n = 510; then log ASR among n = 236 passers)."

Combine frame count includes BALL frames (§1, para. 2)

Quote: "with combine results and tracking (463,189 frames, 28 stable drills)"

Issue: 463,189 is the raw combine_tracking.csv row count including BALL frames; after filtering to entity_type == 'PLAYER', there are 431,094 player frames. Also, there are 31 three-year-stable drills overall and 28 stable skill drills normalized.

Expected: Cite 431,094 player frames, 28 stable skill drills (or 463,189 raw / 431,094 player frames).

Permutation count vs. actual null draws (§1, para. 2)

Quote: "10×5 repeated stratified CV, 1,000 permutations, 2,000 paired bootstraps (seed 42)."

Issue (two technical inaccuracies):

In model_phase3.py, 10×5 CV is stratified (StratifiedKFold) only for the binary hurdle targets, and unstratified KFold for continuous regression targets (Y1 stage-2 and Y2).

In permutation_p, n_perm = 1,000 is divided across n_repeats_perm = 3, producing 333 three-repeat-averaged null draws (999 permuted 5-fold CV fits), which is why the minimum achievable p-value is 1/334 = 0.0030 (p_perm = 0.0030 in §3.2). A sharp judge will immediately spot that 0.0030 cannot come from 1,000 independent draws (1/1001 = 0.0010).

Expected: "10×5 repeated CV (stratified for hurdles), 333 three-repeat-averaged permutation draws (~1,000 permuted fits), and 2,000 position-stratified paired bootstraps (seed 42)."

Abstract mixes M1 and M1a without distinction (Abstract)

Quote: "adding tracking changes hurdle AUC by +0.0009 and stage-2 r by −0.0005 — yet for per-play quality it beats the stopwatch by Δr = +0.0771 (95% CI [+0.0122, +0.1545]) and draft pick alone by +0.0394."

Issue: +0.0009 and −0.0005 are M1 vs M0a (M1a is +0.0020 and −0.0007), whereas +0.0394 is M1a vs M0a (M1 vs M0a is +0.0326). Switching models silently in the same sentence without mentioning that the M1a CI ([−0.0097, +0.0951]) crosses zero looks like silent cherry-picking.

Expected: Either use M1 consistently (+0.0326, CI [−0.0114, +0.0819], permutation p = 0.0030) or explicitly label +0.0394 as pure-tracking M1a (CI [−0.0097, +0.0951]).

Figure 3 sample sizes clash with Table 2 and JSON (figure_captions_phase4.md / §Figures)

Quote: "Within-position partial correlations and Y2 Spearman ρ by defensive-line role: EDGE (n = 66) vs IDL (n = 52)."

Issue: 66 and 52 are the full roster counts in players.csv/combine_results.csv, not the analysis sample sizes plotted in Figure 3. In the frozen JSON and Table 2, Y2 Spearman ρ uses n = 45 (EDGE) and n = 28 (IDL), while the KTR correlations use n = 43–56 (EDGE) and n = 34–42 (IDL).

Expected: Report the actual plotted n ranges in Figure 3's caption so it reconciles with Table 2.

Rounding note for Table 1 vs. reported deltas (§3.1 & Table 1)

Quote: "hurdle AUC 0.8641 vs 0.8631 (Δ +0.0009) ... stage-2 r 0.6607 vs 0.6611 (Δ −0.0005)"

Issue: All numbers match the frozen JSON, but 0.8641 − 0.8631 = +0.0010 and 0.6607 − 0.6611 = −0.0004 at 4 decimal places. Add a one-line footnote to Table 1 ("Deltas are computed from unrounded means.") so reviewers do not flag an arithmetic typo.

Minor number clarifications to verify/tighten:

§2 Shadow Proxy: Specify that r = 0.421 / 0.494 (3-cone, n = 180) and r = 0.284 / 0.446 (short shuttle, n = 201) are within-position out-of-fold correlations.

§3.3 46 players missing a 40-yard time: Clarify "46 of the 299 Y2 players" (across all 510 players, 86 are missing forty).

§3.3 full controls give β = +0.093 (p = 0.089): Add (n = 252).

§3.3 β = +0.048, +0.083, +0.048, +0.035: forty_s_peak (+0.048) matches fdr_Y2; double-check neg_forty (+0.048), vertical (+0.083), and broad_jump (+0.035) directly in fdr_Y2_full of your frozen JSON (consistent with lower_body_burst = +0.060 in earlier runs, and note forty_s_peak is a tracking metric, not a "static test").

§3.5 LightGBM: Explicitly name the linear comparators (Y2 r = 0.1457 vs linear M1a 0.1403 / M1 0.1335; Y1 hurdle AUC 0.8462 vs M1 0.8641) since M3 uses M1's exact feature set.

2. Wording Audit (Overclaims, Contradictions & Ambiguities)

Direct self-contradiction: "ignores the rest" (Abstract, §3.3, §6)

Quote (§3.3): "Skill_s_peak is the exception in both columns: priced (β = +0.153, q = 0.001) and predictive (β = +0.156, q = 0.004). The market prices straight-line speed efficiently and ignores the rest — where tracking's value lives."

Why it fails: You literally just proved the market does not ignore skill drills—skill_s_peak (β = +0.153, q = 0.001), skill_accel_p95 (β = +0.140, q = 0.002), and skill_lat_accel_p95 (β = +0.094, q = 0.019) are all significantly priced into draft capital. The actual finding is asymmetric / differential pricing: straight-line tests are heavily priced (+0.221 to +0.271) and fully exhausted on Y2 (p > 0.10), whereas functional skill-drill speed is priced more lightly (+0.153) and still retains quality Alpha (+0.156).

Fix: Replace "ignores the rest" with "underprices functional skill-drill speed" (Abstract: "The market fully prices straight-line speed but underprices functional skill-drill speed."; §3.3: "The market exhausts straight-line speed and jumps while leaving functional skill-drill speed only partially priced.").

Direct self-contradiction in §3.1: "holds across every contrast"

Quote: "The Y1 nulls are precise to the third decimal while the Y2 gain holds across every contrast."

Why it fails: It does not hold in M1_over_all vs M0all (Δ = −0.0006), M1b vs M0b (Δ = −0.0169), or 2023 LOCO (0.1831 vs 0.1971).

Fix: "The Y1 nulls are precise to the third decimal, while Y2 improves over both the stopwatch and draft pick alone before plateauing at the official NGS composite."

Overclaim in §3.3 heading and §6 conclusion: "survives everything" / "after every control"

Quotes: Heading "3.3 One feature survives everything" and §6 "predictive of per-play quality after every control".

Why it fails: After controlling for 40-yard time (p = 0.077) and full controls (p = 0.089), skill_s_peak is marginal, not significant at p < 0.05.

Fix: Rename §3.3 to 3.3 One feature survives FDR (or 3.3 Skill-drill peak speed: the lone FDR survivor), and revise §6 to "FDR-significant for per-play quality after draft pick and the NGS composite (attenuating to p = 0.089 once 40-yard time is also added)".

Review of the three-layer headline (§3.2) & Layer 3 ("the honest boundary")

Verdict: Strongest paragraph in the entire paper. Layer 1 (+0.0771, CI excludes zero), Layer 2 (+0.0394 / +0.0326, p_perm = 0.0030, CI crosses zero), and Layer 3 (−0.0006, NGS alone 0.1984) are crystal-clear, honest, and well-calibrated.

Two tiny polish fixes:

In the Abstract, change "draft pick plus the official NGS composite" to "draft pick, stopwatch, and official NGS composite" so it matches M0all in §3.2.

In Layer 2, clarify that permutation p = 0.0030 is specifically for M1 vs M0a.

Other minor wording fixes:

§2 Models: Distinguish M1a (pure tracking + Shadow Proxy on M0a) from M1 (M1a + 4 static derived ratios/composites), and note M0a includes UDFA flag, draft year, and position fixed effects, not just log pick.

§3.3: "The market pays heavily for static tests — ... 40-yard peak speed +0.225" → forty_s_peak is a 10 Hz tracking metric, not a static test. Change "static tests" to "straight-line speed and jump metrics".

§3.5 Mediation: Name the exposure being mediated: "game deceleration mediates −3.3% of the Shadow Proxy → Y2 effect (n = 299)".

§4 Robustness: Change "ties in 2023 (0.1831 vs 0.1971)" to "trails slightly in 2023 (0.1831 vs 0.1971)", and "confirms the dissociation" to "confirms the Y1 null".

§5 Limitations: Change "approaches its cap, signalling almost no between-player signal in one season" to "hits its κ = 200 cap (alongside DL quick-pressure), signalling minimal detectable between-player variance over 1–3 seasons".

§6 Closer: "the stopwatch's monopoly is over" is punchy, though slightly aggressive next to Layer 3 (Δ = −0.0006). Consider "the stopwatch should no longer stand alone" for maximum credibility with NFL data scientists.

3. Structure & Rubric Alignment

Overall flow vs. scoring weights (Research 40 / Innovation 30 / Presentation 20 / Visualisation 10):

Research quality (40) is exceptionally strong—double dissociation, nested CV, within-cohort harmonisation, adversarial validation, permutation + bootstrap + FDR, and the honest Layer 3 boundary directly hit NFL judge preferences.

Innovation (30) is well-served by Furusawa–Hill t0 decomposition, the Shadow Agility Proxy, Within-Cohort Unsupervised Harmonisation, and the EDGE vs. IDL transfer split—but the NFL football application ("so what should a GM do on Draft Day?") is currently compressed into one sentence at the end of §6. Adding 2 concrete sentences in §6 (e.g., use skill_s_peak as a tie-breaker on Day 2–3 DL/WR boards and to evaluate the 17% of players missing a 40 and >60% missing agility drills) will boost both Innovation and Presentation.

Presentation (20) & Visualisation (10) suffer from one major fixable flaw:

Figures are cited completely out of order: Fig 1 (§3.1) → Fig 2 (§3.2) → Figs 4 & 6 (§3.3) → Fig 3 (§3.4) → Fig 7 (§3.5) → Fig 5 (§4).

Fix: Renumber the figures strictly by first appearance (keep your narrative section order intact, because the prose flow is logical):

Fig 1 — Double dissociation (§3.1)

Fig 2 — Beyond the stopwatch (§3.2)

Fig 3 — Differential market pricing efficiency (§3.3, was Fig 4)

Fig 4 — Robustness ladder for skill_s_peak (§3.3, was Fig 6)

Fig 5 — Role-specific kinematic translation: EDGE vs. IDL (§3.4, was Fig 3)

Fig 6 — Combine-to-pads transfer (§3.5, was Fig 7)

Fig 7 — Cross-cohort and temporal robustness (§4, was Fig 5)

4. Figures & Tables Review

Count check: 7 figures + 2 tables = 9 — strictly satisfies < 10. Every figure and table is referenced in the text.

Caption-by-caption additions needed (using current draft numbering; renumber per §3 above):

Figure 1: Add sample sizes (Y1 hurdle n = 510; stage-2 n = 236; Y2 n = 299) and note 10×5 CV.

Figure 2: Add (10×5 CV) and clarify which models appear in (b) (M1_over_stopwatch, M1_track_only, M1a, M1, M1_over_all).

Figure 3 (EDGE vs. IDL): Must fix two errors — (1) change "Within-position partial correlations" to "Within-subgroup Pearson r" (code passes empty controls inside each DL subgroup); (2) fix sample sizes from (n = 66 / 52) to EDGE (n = 43–56 KTR; n = 45 Y2) vs IDL (n = 34–42 KTR; n = 28 Y2) so it matches Table 2.

Figure 4 (Pricing vs. Quality): Add sample sizes (n = 405–510 pricing; n = 239–299 quality), error-bar definition (95% HC3 CIs), and note BH-FDR across 14 pre-registered features.

Figure 5 (Robustness): Add sample sizes for (a) n = 112 / 104 / 83 (2023/2024/2025), (b) n_test = 173 hurdle / 125 log-snaps, (c) n = 299 / 286 / 261, and (d) n = 510 (5-fold CV).

Figure 6 (Ladder): Add n = 299, 253, 298, 252, 253 (or n = 252–299), state that UDFA status, position, and draft year are always controlled, and specify 95% HC3 CIs.

Figure 7 (Combine-to-Pads): State that panel (a) controls for position group (n = 319 for 40-yard rows; n = 398–401 for skill-drill rows) and panel (b) plots mean game_s_p99 / forty_s_peak (n = 28–77 per position).

Table 1: Add sample sizes to column headers: Y1 hurdle AUC (n = 510), Y1 stage-2 r (n = 236), Y2 r (n = 299), plus footnote "Deltas in text use unrounded means.".

Table 2: Clean and complete; matches frozen JSON.

5. Word Count Sanity Check & Exact Trim Plan

Sanity check: By manual section-by-section count (excluding the <!-- DRAFT STATUS ... --> HTML comment, including title, headings, 7 captions, 2 tables, and Data/Code availability), the draft sits at ~1,941–1,949 whitespace words, confirming your counter's 1,949.

CRITICAL RISK: 1,949 is too close to the 2,000-word hard cap.

Different platforms count hyphenated words (regular-season, per-play, straight-line, empirical-Bayes, out-of-fold, leave-one-cohort-out, two-stage) and standalone symbols (=, —, ±) differently—a strict regex counter (\w+) will already push this draft above 2,010–2,040 words.

The caption/method clarifications above will add ~60–80 words.

Make sure to delete the HTML <!-- DRAFT STATUS ... --> comment (~55 words) before submission so no parser counts it.

Target: Cut ~150–170 words of pure redundancy so the final version (after adding the required precision fixes) lands at 1,820–1,860 whitespace words (safely < 1,960 even under aggressive tokenizers).

Exact Cuts (High-Value Trims That Lose Zero Content)

§3.1 — Remove duplicate numbers already in Table 1 and §3.2 (~45 words saved)

Cut: "Tracking lifts per-play quality from r = 0.1009 to r = 0.1335 (Δ +0.0326, 95% CI [−0.0114, +0.0819]; M1a Δ +0.0394, 95% CI [−0.0097, +0.0951]; R² 0.0098 → 0.0175). The Y1 nulls are precise to the third decimal while the Y2 gain holds across every contrast."

Replace with (~22 words): "On per-play quality (Y2), tracking lifts r from 0.1009 to 0.1335 (M1; R² 0.0098 → 0.0175) and 0.1403 (M1a; Table 1)." (Leave the CIs to §3.2 directly below so you don't repeat them twice in 10 lines.)

§6 Conclusion — Stop re-quoting the same 4 numbers a third time (~45 words saved)

Cut: "Tracking decisively beats the stopwatch (Δr = +0.0771, CI excludes zero), adds a real but modest conditional signal on top of draft pick (permutation p = 0.0030), and stops at the league's own composite (Δr = −0.0006)."

Replace with (~18 words): "Tracking beats the stopwatch decisively, adds a modest conditional signal over draft pick, and plateaus at the NGS composite." (This frees ~25 words to add one concrete Draft-Day recommendation for GMs!)

§3.4 — Do not duplicate every cell of Table 2 in prose (~30 words saved)

Cut: "offensive line (ρ = −0.036), defensive back (ρ = −0.166), and tight end (ρ = −0.158, n = 22) show no signal."

Replace with: "offensive line, defensive back, and tight end show no signal (ρ = −0.036 to −0.166; Table 2)."

§2 Harmonisation — Tighten redundant phrasing (~20 words saved)

Cut: "using only unlabelled distributions, never the target. Unseen cohorts fall back through three levels (cohort, position, global median), with a 5% floor on within-group standard deviations."

Replace with: "using unlabelled X only (three-tier median fallback—cohort, position, global—and a 5% global-SD floor against degenerate groups)."

§2 Models — Drop plotting implementation note (~10 words saved)

Cut: "(recovered from β and p for plotting)" — unnecessary in a 2,000-word writeup; belongs in the GitHub README.

Data and code availability — Trim filler (~15 words saved)

Replace with: "Code (8 Python modules, MIT) and frozen results: [GitHub URL]. Companion Kaggle Notebook ([link]) reproduces the pipeline from competition data (CC BY-NC 4.0; ~5.8 h on 2 vCPUs, seed 42; smoke mode included)."

中文简评

整体成稿非常扎实、克制且忠实于冻结的 Round-5 全量结果，Table 1/2 全部 48 个数值与冻结 JSON 逐位吻合。但提交前必须修完 5 项硬伤：

定义与技术口径补全：§1 把 Y1 写成了未做暴露校正的“clearing 150 snaps, then log snaps”，必须补回年均常规赛档数（ASR = snaps / eligible seasons）；463,189 帧含球帧（纯球员帧为 431,094）；置换检验实际是 333 个三重复均值零样本（~1,000 次拟合，所以最小 p = 1/334 = 0.0030）；Abstract 应统一用 M1 或明确标注 M1a 并交代 CI 跨零。

消除两处自相矛盾与过强措辞：删除 "ignores the rest"（前一句刚证明 skill_s_peak 被定价 β=+0.153, q=0.001），改为 "underprices functional skill-drill speed"；删除 §3.1 的 "holds across every contrast"（与 Layer 3 −0.0006 矛盾）；把 §3.3 标题 "survives everything" 和 §6 "after every control" 降调为 "survives FDR" 并如实写出加 40 码后呈边际显著（p = 0.089）。

重排全部 7 张图的编号：当前正文中图号顺序是 1 → 2 → 4, 6 → 3 → 7 → 5，必须严格按首次出现顺序重排为 1–7。

修正 Figure 3 图注冲突：图注写 EDGE (n=66) vs IDL (n=52) 与 Table 2（n=45 / 28）直接打架，且亚组内算的是普通 Pearson r 而非偏相关，需按实际分析样本量（KTR n=43–56 / 34–42；Y2 n=45 / 28）修正。

压字数留足安全垫：当前 1,949 词离 2,000 红线太近（换分词器或加上本轮图注补齐极易爆表）；按上述 6 处删重方案压掉 ~150 词并删除文首 HTML 注释，把终稿稳控在 1,820–1,860 词后即可放心提交。
