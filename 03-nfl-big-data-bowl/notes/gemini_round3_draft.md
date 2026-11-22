# Gemini 第 3 轮提纲（草稿：服务器第 3 轮全量结果待回传后补数字）

对话：继续第 1/2 轮同一对话 https://gemini.google.com/app/8adf0ca845f706e2 ，同一账号 wangde2021@gmail.com（Google AI Pro），同一模型 Neon、思考等级“高”。
附件：本轮改后的 src/ 5 个文件（kinematics.py、features_combine.py、targets.py、game_kinematics.py、model_phase3.py）+ 本轮服务器结果（logs/phase3_results.md 全文 + combine_features_summary.json + targets_summary.json）+ 本文件。

## 第 2 轮“待核实项”核实结论（已按代码核对，Gemini 引用与代码不符处以代码为准）
- ① permutation_p 用 M1 的 alpha_median 拟合 M0a、只跑 repeat=0 —— 属实。model_phase3.py:572–577 传 `_fp = out_d["M1"].get("alpha_median")`；:384–403 fixed_factory 同一 fixed_param 用于 oof_m1 与 oof_m0；permutation_p 内 `_materialize_folds(..., 0, ...)` 只用 repeat=0。（Gemini 写的“model_phase3_2.py 第 307–311/438–444 行”文件名与行号均与实际代码不符。）
- ② calibrate_position_z 折前全局中位数填补、lower_body_burst_fc 折前 z-score —— 属实。features_combine.py:324 `X.fillna(X.median(...))` 在 KFold 循环之前（Gemini 写的“第 253 行”行号不符）；:399–401 lower_body_burst_fc 由 zscore_within 在全样本 df 上算出。另附：static_features 的 lower_body_burst（features_combine.py:91）同样是全样本位置内 z（文件第 34 行注释自认），本次未动——因 model_phase3 的 FeaturePrep 会对其再做折内位置 z，残余泄漏极小，请裁决是否可接受。
- ③ WR 的 v_max_best 均值 2024–2025 年 12.20–12.27 yd/s —— 属实（本地 forty_attempts.parquet，全量 762 次尝试；WR：2023 年 11.52 n=30、2024 年 12.20 n=26、2025 年 12.27 n=30）。关键对照：同批实测 s_peak_best 均值 11.03/11.07/11.08，三年稳定——漂移来自无 t0 拟合虚推 v_max，不是真实极速变化。
- ④ M0b 列定义混入 ngs_final_score —— 属实。model_phase3.py:94–95：M0B_COLS = [ngs_athleticism_score, ngs_final_score, forty, ten_yd_split, vertical, broad_jump]。

## 本轮改动摘要（第 2 轮最小闭环清单，实际约 +450 行，见运行日志）
- fit_sprint_curve：加起跑偏移 t0 ∈ [−0.3, 1.5] 与物理上界 v_max ≤ s_peak×1.12，输出实测 forty_s_peak（进 TRACK_PURE）。
- calibrate_position_z：填补移入折内（训练折分位置中位数→训练折全局中位数）；Shadow Full 去掉折前 lower_body_burst_fc，改用原始 vertical/broad_jump 列（折内位置 z）。
- targets.py：每位置×指标 EB kappa 记入 targets_summary.json 的 y2.eb_kappa（分 gated/raw 两组）。
- model_phase3.py：M0b 拆为 M0b_stopwatch 与 M0c_ngs（另有 M0all）；M1 系列改两阶段残差/偏置堆叠（回归=基线自身最优惩罚 + 残差 RidgeCV，alpha 至 1e5；分类=基线 logit 作固定偏置列、近乎免惩罚）；permutation_p 各自用正确正则化、聚合 3 次重复、置换堆叠模型的 X_track；对抗验证分块诊断（FORTY/SKILL/STATIC）+ forty_s_peak 分年均值；--within-year 同届内调和开关（默认关）；FDR 主族 = 9 技能 + 5 个 forty 核心参数（完整 19 项表另存 fdr_Y2_full）；新增选秀定价回归（−log_pick，Fig 4 数据）与 skill_s_peak 稳健性回归（控制 log_pick + forty + ngs_final_score）；KTR 完整 5×4 偏相关矩阵（附 n）+ 分位置对角值 + ktr_speed_n；M2 新增 game_snap_burst_1s 复测；时序外推 Y1_year1_logsnaps 补 year1_reg_snaps > 0 过滤；时序/LOCO/Y2 敏感性的 M1 改堆叠。

## 本轮要 Gemini 裁决的事（按优先级，服务器数字待补）

### 1. t0 修复是否消除了 40 码跨年漂移（首要）
- 服务器全量实测（2026-11-21/22，种子 42）：对抗验证分块 AUC：ALL 0.6627 ± 0.0297、FORTY_ONLY 0.6646 ± 0.0247、SKILL_ONLY 0.4773 ± 0.0298、STATIC_ONLY 0.5742 ± 0.0534——forty 特征仍能以 0.66 的 AUC 预测届别，远高于 0.55 门限；SKILL_ONLY 已降到 0.48（低于随机），说明技能 drill 本身没有跨年漂移，漂移集中在 forty/静态特征。forty_s_peak 分年均值（全位置）：2023 年 10.419（n=134）、2024 年 10.368（n=145）、2025 年 10.393（n=126）；WR：11.033/11.070/11.079，三年稳定。修 t0 后的 WR v_max_best 分年均值（本地冒烟小样本）：11.38/11.60/11.63（旧 11.52/12.20/12.27），残余梯度 +0.25。Shadow 标定（服务器）：pure vs three_cone oof_r2_within 0.1765、full 0.2436；pure vs shuttle 0.0794、full 0.1981。
- 问：若 FORTY_ONLY 仍 >0.56，是否启用 --within-year 同届内调和（DrillFoldClean 按 drill×届、FeaturePrep 按位置×届）？若 SKILL_ONLY 也高，是否说明技能 drill 本身也有传感漂移、须同样调和？可接受门限是否仍为 ≤0.55（均值 ≤0.56 且 95% 区间覆盖 0.50）？

### 2. M0b 拆分后的 Y2 故事
- 服务器全量实测 Y2_quality（r）：M0a 0.1009 ± 0.0275、M0b_stopwatch −0.0198 ± 0.0404、M0c_ngs 0.1926 ± 0.0204、M0b（合并）0.1959 ± 0.0280、M0all 0.1747 ± 0.0305、M1 0.1000 ± 0.0263、M1b 0.2020 ± 0.0192、M1_over_stopwatch 0.0843 ± 0.0355、M1_over_all 0.1745 ± 0.0309。增量：M1_over_stopwatch vs M0b_stopwatch Δ=+0.1040，95% CI [+0.0051, +0.2416]（下限刚过 0）；M1_over_all vs M0all Δ=−0.0001，CI [−0.0007, +0.0003]；M1 vs M0a Δ=−0.0009，CI [−0.0158, +0.0130]。Y2 分位置 Spearman：DL rho=+0.354（p=0.0021，n=73）是唯一显著位置；WR +0.211（p=0.090）、DB −0.127、OL −0.017、TE −0.204 均不显著。
- 问：若证实 M0b_stopwatch 本身很弱（与 fdr_Y2 一致），“超越秒表”的表述是否成立？M1_over_all 的净追踪增量应如何引用（均值 Δ + CI 为主，还是置换 p 为辅）？

### 3. 堆叠后的 Y1/Y2 增量是否干净
- 服务器全量实测（M1 vs 各自基线，均值 Δ + 配对自助 95% CI；新 permutation_p，1000 次置换）：Y1_hurdle_stage1 Δ=−0.0002，CI [−0.0096, +0.0053]，obs +0.0014，p_perm=0.2020；Y1_stage2_logASR Δ=−0.0000，CI [−0.0001, +0.0000]，obs −0.0000，p_perm=0.5010；Y2_quality Δ=−0.0009，CI [−0.0158, +0.0130]，obs +0.0042，p_perm=0.0350；Y1_year1_hurdle Δ=+0.0015，CI [−0.0065, +0.0106]，obs +0.0009，p_perm=0.1290；Y1_year1_logsnaps Δ=−0.0018，CI [−0.0065, +0.0047]，obs +0.0041，p_perm=0.0020。注意：Y2 与 Y1_year1_logsnaps 的置换 p 显著（0.035/0.002），但对应 Δ 的自助 CI 都覆盖 0——两种检验口径不一致，请裁决报告时以哪个为准。
- 问：堆叠实现是否正确——回归用残差 RidgeCV、分类用基线 logit 偏置列（偏置列用训练折内样本拟合，轻微近似，已披露）是否可接受？有无更干净的分类堆叠做法（交叉拟合偏置）值得在第 4 轮做？置换检验现在测的是堆叠模型的残差增量，报告应如何表述其原假设？

### 4. 选秀定价森林图（Fig 4）与稳健性回归
- 服务器全量实测 draft_pricing（−log_pick，只控制位置 + 届别）：forty_s_peak β=+0.225，p≈0（3.48e-09），q=0.000，n=405；v_max_best β=+0.189，p≈0（3.05e-06），q=0.000，n=405；skill_s_peak_z_mean β=+0.153，p=0.000254，q=0.001，n=510；skill_accel_p95_z_mean β=+0.140，p=0.0006，q=0.002，n=510；a0_mean β=+0.136，p=0.0106，q=0.025，n=405；其余 q 均 >0.02。fdr_Y2 主族：skill_s_peak_z_mean β=+0.156，p=0.000316，q=0.004，n=299（唯一 q<0.05）；其余 13 项 q 均 ≥0.21。稳健性回归 robustness_skill_s_peak（控制 log_pick + forty + ngs_final_score）：β=+0.093，p=0.0886，n=252——三重控制后不再显著。
- 问：定价斜率 β_draft 与质量斜率 β_Y2|draft 的对照是否构成“选秀错误定价”的硬证据？若 skill_s_peak 在三重控制后仍显著，是否可作为 Fig 4 的核心一对？forty / speed_score / lower_body_burst 的 β_draft 大而 β_Y2|draft ≈ 0 时，报告措辞应如何把握（避免“市场完全错误”的过度宣称）？

### 5. KTR 矩阵与 burst 中介复测
- 服务器全量实测：KTR 5×4 偏相关矩阵（位置内，完整表见 phase3_results.md）：forty_s_peak ~ game_accel_p95 r=+0.369（p=2.56e-11，n=307）、~ game_lat_accel_p95 r=+0.316（p=1.42e-08）；v_max_best ~ game_accel_p95 r=+0.317（p=1.37e-08）；skill_lat_accel ~ game_accel_p95 r=+0.254（p=4.67e-07）；其余多数 r 在 0.04–0.16。ktr_speed_mean（修 t0 后）= 0.6524——低于第 2 轮时预期的 ~0.82–0.90，请解释。分位置对角：DL 的 skill_accel~game_accel r=+0.474（p=9.20e-07，n=97）、skill_decel~game_decel r=+0.369（p=0.000236）、skill_lat~game_lat r=+0.437（p=7.63e-06）；WR 的 v_max~game_s_p99 r=+0.297（p=0.019，n=62）。M2 中介（n=299）：decel 中介 indirect=−0.0050（95% CI 待查 JSON），直接效应 0.1554，pct_mediated=−3.3%；burst_1s 复测 indirect=+0.0117，直接效应 0.1387，pct_mediated=+7.8%。
- 问：若 burst 中介仍为零，是否可定稿为“体测运动能力能跨越护具迁移到实战物理上限，但无条件聚合的实战运动极值不是单档质量的线性中介”？KTR 矩阵中哪些格子值得进 Fig 5，哪些只是描述性附录？

### 6. 冻结检查：是否可进 Phase 4 出图与 Phase 5 写作
- 服务器全量实测补充：时序外推 Y1_year1_hurdle M0a 0.8756 / M1 0.8799（n_train 337，n_test 173）；Y1_year1_logsnaps M0a 0.6293 / M1 0.6312；LOCO_Y2：2023 年 M0a r=0.197 / M1 0.185（n=112）、2024 年 0.094/0.094（n=104）、2025 年 0.011/0.015（n=83）——2025 年 LOCO 近乎为零。Y2 敏感性：reg_snaps_50 oof r=0.080（n=299）、reg_snaps_150 oof r=0.037（n=283）。M3 LightGBM：Y2 r=0.175，Y1 hurdle AUC=0.850。experiments.csv 本轮新增 3 行：Y1_hurdle_stage1 M1 AUC 0.8629（perm p=0.2020）、Y1_stage2_logASR M1 r=0.6611（p=0.5010）、Y2_quality M1 r=0.100（p=0.0350）。
- 问：按第 2 轮承诺，本轮改完重跑后数字应做到零泄漏、零口径矛盾、零过拟合虚高——请逐项核查是否还有泄漏/口径/过拟合问题（尤其：静态 lower_body_burst 的全样本位置 z 是否可接受；Shadow 指数本身是 Phase-2 预拟合表示；FDR 用预计算技能列作描述性筛选）。若仍有问题，请列出第 4 轮最小修改清单；若通过，请确认可冻结结果进入出图与写作。

## 附：本轮服务器结果（已回传，见附件）
- notes/phase3_results_round3_server.md：phase3_results.md 全文（556 行）+ Shadow 标定全文 + targets_summary.json 全文 + game_kinematics/oof/experiments 确认 + 服务器状态确认。
