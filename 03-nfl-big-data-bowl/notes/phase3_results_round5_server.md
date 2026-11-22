# Phase 3 全量结果 — 第 5 轮（服务器，2026-11-22 完成）

> 服务器第 5 轮 model_phase3 --within-year 已正常完成：2026-11-22 15:12:33 启动，约 20:58:55 退出，全量实际耗时 5 小时 46 分 22 秒（第 4 轮 7 小时 14 分；30–45 分钟提速预估未兑现，见末尾）。日志共 6910 行，全日志 grep -iE 'error|traceback' 0 条；尾部为 wrote phase3_results.json、wrote phase3_results.md、appended 3 rows to experiments.csv；ps aux 无 model_phase3 残留。结果文件时间戳均更新为 2026-11-22 20:58:55：logs/phase3_results.json（43,252 字节）、logs/phase3_results.md（28,691 字节）、data/derived/oof_predictions.parquet（255,738 字节）、logs/experiments.csv（2,738 字节）。
>
> 数字来源：浏览器任务回传的服务器文件全文（md 与 json 按 50–55 行分段通读，长浮点数均用 python repr 逐一核对，以 repr 值为准）。本文件为逐字转存，非人工抄录。服务器：tokyo-node-work，e2-standard-2（2 核 8G），IP 35.213.17.8，目录 `/home/wangde2021/03-nfl-big-data-bowl`。备份 prefix_snapshot_r4/。仅覆盖 src/model_phase3.py（md5 6d1c3f7dc117ce7574dfaa09a3955ebe）。
>
> 前三步（与第 4 轮同一天 14:37–15:10 完成，退出码 0）：features_combine stable drills 31；targets y2 covered 299；game_kinematics 401 人 / 10,424,763 帧。

---

# NFL Big Data Bowl 2027 — Phase 3 主建模结果

种子 42；10×5 折重复分层 CV；置换 1000 次；自助 2000 次。

特征分组：TRACK_PURE（四十曲线 + 9 项无质量技能指标，折内干净 drill 归一化）；STATIC（静态秒表派生）；SHADOW（纯技能影子指数）。
Round-2 改动：M0b 已拆为 M0b_stopwatch（纯秒表/双跳）与 M0c_ngs（NGS 评分）；M1 系列为两阶段残差/偏置堆叠（基线用自身最优惩罚）；置换检验基线与全模型各自用正确正则化、聚合 3 次重复。
Round-3 改动（第 4 轮修复，Gemini 第 3 轮核对）：DrillFoldClean 静默清零修复（单列 groupby 不用列表、transform 补 [m]、加非空断言）；M1_over_stopwatch基线去掉顺位、新增 M1_track_only 纯追踪对照；M0all/M1_over_all 的秒表与 NGS 列改走位置内标准化；--within-year 同届内调和开启（未见届用测试届自身无标签分布）；permutation_p 每个 null 样本也跨 3 次重复取均值（尺度对齐）；分类偏置列改折内 5 折交叉拟合；lower_body_burst 在 FeaturePrep 内由原始 vertical/broad_jump 折内重算。

Drill 归一化：fold-clean drill normalization from skill_events.parquet (28 stable drills, 4238 player-drill rows); WITHIN-YEAR harmonization ON

## Y1_hurdle_stage1
- M0a: 0.8631 ± 0.0033（内层 CV 选定 alpha/C 中位数 1.93）
- M0b_stopwatch: 0.5968 ± 0.0084（内层 CV 选定 alpha/C 中位数 0.0283）
- M0c_ngs: 0.8082 ± 0.0034（内层 CV 选定 alpha/C 中位数 1）
- M0b: 0.8022 ± 0.0045（内层 CV 选定 alpha/C 中位数 0.268）
- M0all: 0.8640 ± 0.0047（内层 CV 选定 alpha/C 中位数 0.139）
- M1a: 0.8651 ± 0.0028（内层 CV 选定 alpha/C 中位数 0.01）
- M1: 0.8641 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.01）
- M1full: 0.8643 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.01）
- M1b: 0.7873 ± 0.0061（内层 CV 选定 alpha/C 中位数 0.139）
- M1_track_only: 0.5688 ± 0.0081（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_stopwatch: 0.5724 ± 0.0088（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all: 0.8600 ± 0.0039（内层 CV 选定 alpha/C 中位数 0.01）
- delta_M1_vs_M0a: +0.0009, 95% CI [-0.0079, +0.0065]
- delta_M1b_vs_M0b: -0.0150, 95% CI [-0.0281, +0.0023]
- delta_M1a_vs_M0a: +0.0020, 95% CI [-0.0078, +0.0063]
- delta_M1_over_stopwatch_vs_M0b_stopwatch: -0.0244, 95% CI [-0.0643, +0.0129]
- delta_M1_track_only_vs_M0b_stopwatch: -0.0279, 95% CI [-0.0763, +0.0150]
- delta_M1_over_all_vs_M0all: -0.0041, 95% CI [-0.0106, +0.0032]
- 置换检验（折内 × 置换）：观测 Δ=+0.0015, p_perm=0.0599（零分布均值 -0.0009 ± 0.0016）

## Y1_stage2_logASR
- M0a: 0.6611 ± 0.0078（内层 CV 选定 alpha/C 中位数 14.7）
- M0b_stopwatch: 0.4681 ± 0.0126（内层 CV 选定 alpha/C 中位数 31.6）
- M0c_ngs: 0.5986 ± 0.0186（内层 CV 选定 alpha/C 中位数 10）
- M0b: 0.5862 ± 0.0175（内层 CV 选定 alpha/C 中位数 21.5）
- M0all: 0.6559 ± 0.0101（内层 CV 选定 alpha/C 中位数 31.6）
- M1a: 0.6605 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1: 0.6607 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1full: 0.6609 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1b: 0.5483 ± 0.0261（内层 CV 选定 alpha/C 中位数 31.6）
- M1_track_only: 0.4245 ± 0.0374（内层 CV 选定 alpha/C 中位数 46.4）
- M1_over_stopwatch: 0.4669 ± 0.0119（内层 CV 选定 alpha/C 中位数 1.47e+04）
- M1_over_all: 0.6559 ± 0.0101（内层 CV 选定 alpha/C 中位数 1e+05）
- delta_M1_vs_M0a: -0.0005, 95% CI [-0.0011, +0.0003]
- delta_M1b_vs_M0b: -0.0378, 95% CI [-0.0576, +0.0052]
- delta_M1a_vs_M0a: -0.0007, 95% CI [-0.0017, +0.0005]
- delta_M1_over_stopwatch_vs_M0b_stopwatch: -0.0013, 95% CI [-0.0056, +0.0042]
- delta_M1_track_only_vs_M0b_stopwatch: -0.0437, 95% CI [-0.1052, +0.0375]
- delta_M1_over_all_vs_M0all: -0.0000, 95% CI [-0.0003, +0.0003]
- 置换检验（折内 × 置换）：观测 Δ=+0.0000, p_perm=0.1168（零分布均值 -0.0000 ± 0.0000）

## Y2_quality
- M0a: 0.1009 ± 0.0275（内层 CV 选定 alpha/C 中位数 147）
- M0b_stopwatch: -0.0016 ± 0.0356（内层 CV 选定 alpha/C 中位数 1e+03）
- M0c_ngs: 0.1984 ± 0.0256（内层 CV 选定 alpha/C 中位数 46.4）
- M0b: 0.1756 ± 0.0279（内层 CV 选定 alpha/C 中位数 100）
- M0all: 0.1829 ± 0.0286（内层 CV 选定 alpha/C 中位数 215）
- M1a: 0.1403 ± 0.0271（内层 CV 选定 alpha/C 中位数 1.33e+03）
- M1: 0.1335 ± 0.0290（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1full: 0.1279 ± 0.0312（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1b: 0.1587 ± 0.0210（内层 CV 选定 alpha/C 中位数 464）
- M1_track_only: 0.0902 ± 0.0295（内层 CV 选定 alpha/C 中位数 681）
- M1_over_stopwatch: 0.0755 ± 0.0358（内层 CV 选定 alpha/C 中位数 1.33e+03）
- M1_over_all: 0.1823 ± 0.0317（内层 CV 选定 alpha/C 中位数 4.55e+03）
- delta_M1_vs_M0a: +0.0326, 95% CI [-0.0114, +0.0819]
- delta_M1b_vs_M0b: -0.0169, 95% CI [-0.0910, +0.0646]
- delta_M1a_vs_M0a: +0.0394, 95% CI [-0.0097, +0.0951]
- delta_M1_over_stopwatch_vs_M0b_stopwatch: +0.0771, 95% CI [+0.0122, +0.1545]
- delta_M1_track_only_vs_M0b_stopwatch: +0.0918, 95% CI [-0.0076, +0.2217]
- delta_M1_over_all_vs_M0all: -0.0006, 95% CI [-0.0169, +0.0158]
- 置换检验（折内 × 置换）：观测 Δ=+0.0415, p_perm=0.0030（零分布均值 -0.0051 ± 0.0090）

## Y1_year1_hurdle
- M0a: 0.8438 ± 0.0050（内层 CV 选定 alpha/C 中位数 0.518）
- M0b_stopwatch: 0.5760 ± 0.0117（内层 CV 选定 alpha/C 中位数 0.01）
- M0c_ngs: 0.7824 ± 0.0048（内层 CV 选定 alpha/C 中位数 0.393）
- M0b: 0.7759 ± 0.0060（内层 CV 选定 alpha/C 中位数 393）
- M0all: 0.8461 ± 0.0045（内层 CV 选定 alpha/C 中位数 0.139）
- M1a: 0.8422 ± 0.0043（内层 CV 选定 alpha/C 中位数 0.01）
- M1: 0.8427 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.01）
- M1full: 0.8426 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.01）
- M1b: 0.7510 ± 0.0072（内层 CV 选定 alpha/C 中位数 0.072）
- M1_track_only: 0.5327 ± 0.0121（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_stopwatch: 0.5312 ± 0.0122（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all: 0.8387 ± 0.0030（内层 CV 选定 alpha/C 中位数 0.01）
- delta_M1_vs_M0a: -0.0011, 95% CI [-0.0087, +0.0062]
- delta_M1b_vs_M0b: -0.0249, 95% CI [-0.0368, -0.0097]
- delta_M1a_vs_M0a: -0.0016, 95% CI [-0.0087, +0.0042]
- delta_M1_over_stopwatch_vs_M0b_stopwatch: -0.0448, 95% CI [-0.0807, -0.0075]
- delta_M1_track_only_vs_M0b_stopwatch: -0.0434, 95% CI [-0.0917, +0.0037]
- delta_M1_over_all_vs_M0all: -0.0073, 95% CI [-0.0131, -0.0012]
- 置换检验（折内 × 置换）：观测 Δ=-0.0057, p_perm=0.6856（零分布均值 -0.0050 ± 0.0018）

## Y1_year1_logsnaps
- M0a: 0.5794 ± 0.0042（内层 CV 选定 alpha/C 中位数 18.1）
- M0b_stopwatch: 0.2230 ± 0.0169（内层 CV 选定 alpha/C 中位数 100）
- M0c_ngs: 0.5176 ± 0.0062（内层 CV 选定 alpha/C 中位数 6.81）
- M0b: 0.5161 ± 0.0049（内层 CV 选定 alpha/C 中位数 18.1）
- M0all: 0.5910 ± 0.0048（内层 CV 选定 alpha/C 中位数 46.4）
- M1a: 0.5807 ± 0.0060（内层 CV 选定 alpha/C 中位数 1.74e+03）
- M1: 0.5807 ± 0.0057（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1full: 0.5806 ± 0.0058（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1b: 0.4904 ± 0.0103（内层 CV 选定 alpha/C 中位数 46.4）
- M1_track_only: 0.2077 ± 0.0177（内层 CV 选定 alpha/C 中位数 147）
- M1_over_stopwatch: 0.2216 ± 0.0150（内层 CV 选定 alpha/C 中位数 3.48e+03）
- M1_over_all: 0.5902 ± 0.0052（内层 CV 选定 alpha/C 中位数 7.35e+03）
- delta_M1_vs_M0a: +0.0013, 95% CI [-0.0047, +0.0095]
- delta_M1b_vs_M0b: -0.0258, 95% CI [-0.0481, +0.0102]
- delta_M1a_vs_M0a: +0.0014, 95% CI [-0.0048, +0.0098]
- delta_M1_over_stopwatch_vs_M0b_stopwatch: -0.0015, 95% CI [-0.0150, +0.0159]
- delta_M1_track_only_vs_M0b_stopwatch: -0.0153, 95% CI [-0.0730, +0.0644]
- delta_M1_over_all_vs_M0all: -0.0008, 95% CI [-0.0031, +0.0022]
- 置换检验（折内 × 置换）：观测 Δ=+0.0058, p_perm=0.0030（零分布均值 -0.0016 ± 0.0015）

## 时序外推与 LOCO
{
"Y1_year1_hurdle_M0a": {
"n_train": 337,
"n_test": 173,
"metric": 0.8756019261637239
},
"Y1_year1_hurdle_M1": {
"n_train": 337,
"n_test": 173,
"metric": 0.8670411985018726
},
"Y1_year1_logsnaps_M0a": {
"n_train": 253,
"n_test": 125,
"metric": 0.629337524994706
},
"Y1_year1_logsnaps_M1": {
"n_train": 253,
"n_test": 125,
"metric": 0.6241354815808838
},
"LOCO_Y2": {
"Y2_test_2023_M0a": {
"n_test": 112,
"r": 0.19713610636162895
},
"Y2_test_2023_M1": {
"n_test": 112,
"r": 0.18306095853637594
},
"Y2_test_2024_M0a": {
"n_test": 104,
"r": 0.09381314854689464
},
"Y2_test_2024_M1": {
"n_test": 104,
"r": 0.13149972876264582
},
"Y2_test_2025_M0a": {
"n_test": 83,
"r": 0.010633263245448521
},
"Y2_test_2025_M1": {
"n_test": 83,
"r": 0.05129599119990471
}
}
}

## 对抗验证（特征预测 draft_year，应 ≤0.55）
{
"draft_year_macro_auc_ALL": {
"mean": 0.49223052863347283,
"std": 0.022957338851309805
},
"draft_year_macro_auc_FORTY_ONLY": {
"mean": 0.492419102502197,
"std": 0.021861513647072423
},
"draft_year_macro_auc_SKILL_ONLY": {
"mean": 0.489179670081067,
"std": 0.027778736566068045
},
"draft_year_macro_auc_STATIC_ONLY": {
"mean": 0.48793166462376636,
"std": 0.03309186281154244
},
"draft_year_macro_auc": {
"mean": 0.49223052863347283,
"std": 0.022957338851309805
},
"forty_s_peak_by_year": {
"2023": {
"mean_all": 10.41943710015138,
"n_all": 134,
"mean_WR": 11.033392063619042,
"n_WR": 30
},
"2024": {
"mean_all": 10.368011165939894,
"n_all": 145,
"mean_WR": 11.070109889527465,
"n_WR": 26
},
"2025": {
"mean_all": 10.393349142858876,
"n_all": 126,
"mean_WR": 11.07884902225396,
"n_WR": 30
}
}
}

## Y2 分位置 Spearman（M1 折外预测，第 5 轮新增进 md）
- DB：n=71，Spearman ρ=-0.166，p=0.168
- DL：n=73，Spearman ρ=+0.507，p=4.78e-06
- OL：n=67，Spearman ρ=-0.036，p=0.773
- TE：n=22，Spearman ρ=-0.158，p=0.484
- WR：n=66，Spearman ρ=+0.242，p=0.0505

## BH-FDR 单特征检验（Y2，顺位校正后，HC3）
- skill_s_peak_z_mean：β=+0.156，p=0.000316，q=0.004，n=299
- skill_accel_p95_z_mean：β=+0.098，p=0.0359，q=0.237，n=299
- skill_cut_reaccel_mean_z_mean：β=+0.077，p=0.0509，q=0.237，n=294
- skill_decel_p95_z_mean：β=+0.078，p=0.101，q=0.316，n=299
- skill_lat_accel_p95_z_mean：β=+0.072，p=0.13，q=0.316，n=299
- skill_brake_power_p95_z_mean：β=+0.067，p=0.175，q=0.316，n=299
- skill_braking_impulse_max_z_mean：β=+0.058，p=0.178，q=0.316，n=299
- skill_cut_retention_mean_z_mean：β=-0.064，p=0.181，q=0.316，n=294
- v_max_best：β=+0.060，p=0.235，q=0.365，n=239
- forty_s_peak：β=+0.048，p=0.329，q=0.461，n=239
- a0_mean：β=+0.044，p=0.367，q=0.467，n=239
- tau_best：β=+0.035，p=0.521，q=0.608，n=239
- speed_maintenance_mean：β=-0.022，p=0.7，q=0.753，n=239
- skill_jerk_p95_z_mean：β=-0.000，p=1，q=1.000，n=299

## 选秀定价回归（-log_pick，Fig 4 森林图数据；只控制位置 + 届别）
- neg_forty：β=+0.271，p=2.92e-12，q=0.000，n=424
- vertical：β=+0.232，p=2.63e-09，q=0.000，n=438
- forty_s_peak：β=+0.225，p=3.48e-09，q=0.000，n=405
- broad_jump：β=+0.221，p=3.37e-07，q=0.000，n=423
- v_max_best：β=+0.189，p=3.05e-06，q=0.000，n=405
- skill_s_peak_z_mean：β=+0.153，p=0.000254，q=0.001，n=510
- skill_accel_p95_z_mean：β=+0.140，p=0.000632，q=0.002，n=510
- skill_lat_accel_p95_z_mean：β=+0.094，p=0.00881，q=0.019，n=510
- a0_mean：β=+0.136，p=0.0106，q=0.020，n=405
- skill_cut_reaccel_mean_z_mean：β=+0.074，p=0.0605，q=0.100，n=501
- skill_brake_power_p95_z_mean：β=+0.080，p=0.0644，q=0.100，n=510
- skill_decel_p95_z_mean：β=+0.063，p=0.131，q=0.185，n=510
- speed_maintenance_mean：β=+0.051，p=0.256，q=0.334，n=405
- tau_best：β=-0.048，p=0.309，q=0.357，n=405
- skill_braking_impulse_max_z_mean：β=+0.041，p=0.315，q=0.357，n=510
- skill_cut_retention_mean_z_mean：β=+0.026，p=0.508，q=0.540，n=501
- skill_jerk_p95_z_mean：β=+0.022，p=0.599，q=0.599，n=510

## 稳健性：skill_s_peak_z_mean 控制 log_pick + forty + ngs_final_score
{
"feature": "skill_s_peak_z_mean",
"controls": "log_pick+forty+ngs_final_score",
"subset": null,
"n": 252,
"beta": 0.09262906226480878,
"p": 0.0886269512288863
}

## 稳健性阶梯：skill_s_peak_z_mean（五步控制，第 5 轮新增第 5 步）
- log_pick：β=+0.156，p=0.000316，n=299
- log_pick+forty：β=+0.094，p=0.0774，n=253
- log_pick+ngs_final_score：β=+0.138，p=0.00268，n=298
- log_pick+forty+ngs_final_score：β=+0.093，p=0.0886，n=252
- log_pick（子集 forty.notna()）：β=+0.136，p=0.00302，n=253

## KTR 5x4 偏相关矩阵（位置内，附 n）
- v_max_best ~ game_s_p99：r=+0.306，p=2.47e-08，n=319
- v_max_best ~ game_accel_p95：r=+0.369，p=1.04e-11，n=319
- v_max_best ~ game_decel_p95：r=+0.311，p=1.57e-08，n=317
- v_max_best ~ game_lat_accel_p95：r=+0.348，p=1.59e-10，n=319
- forty_s_peak ~ game_s_p99：r=+0.363，p=2.17e-11，n=319
- forty_s_peak ~ game_accel_p95：r=+0.414，p=1.22e-14，n=319
- forty_s_peak ~ game_decel_p95：r=+0.338，p=6.57e-10，n=317
- forty_s_peak ~ game_lat_accel_p95：r=+0.368，p=1.16e-11，n=319
- skill_accel_p95_z_mean ~ game_s_p99：r=+0.185，p=0.000194，n=401
- skill_accel_p95_z_mean ~ game_accel_p95：r=+0.237，p=1.64e-06，n=400
- skill_accel_p95_z_mean ~ game_decel_p95：r=+0.212，p=1.93e-05，n=398
- skill_accel_p95_z_mean ~ game_lat_accel_p95：r=+0.187，p=0.000163，n=401
- skill_decel_p95_z_mean ~ game_s_p99：r=+0.126，p=0.0119，n=401
- skill_decel_p95_z_mean ~ game_accel_p95：r=+0.148，p=0.00308，n=400
- skill_decel_p95_z_mean ~ game_decel_p95：r=+0.164，p=0.00106，n=398
- skill_decel_p95_z_mean ~ game_lat_accel_p95：r=+0.087，p=0.0807，n=401
- skill_lat_accel_p95_z_mean ~ game_s_p99：r=+0.186，p=0.000179，n=401
- skill_lat_accel_p95_z_mean ~ game_accel_p95：r=+0.230，p=3.37e-06，n=400
- skill_lat_accel_p95_z_mean ~ game_decel_p95：r=+0.288，p=4.73e-09，n=398
- skill_lat_accel_p95_z_mean ~ game_lat_accel_p95：r=+0.201，p=5.09e-05，n=401

## M3 LightGBM / M2 中介 / Y2 敏感性
{
"M3_lightgbm": {
"Y2_r_mean": 0.14572077116356097,
"Y1_hurdle_auc_mean": 0.8462049981442533
},
"M2_mediation": {
"n": 299,
"a_path": 0.11526418508933539,
"b_path": -0.04338748770428838,
"indirect": -0.005001023413308359,
"indirect_ci95": [
-0.03347490197897308,
0.02374375842024337
],
"direct_c": 0.15544714340244187,
"pct_mediated": -0.03324129205638254,
"burst_1s": {
"n": 299,
"a_path": 0.08296041092246682,
"b_path": 0.1412405435594157,
"indirect": 0.011717373532601702,
"indirect_ci95": [
-0.015757457670865792,
0.04413513839774978
],
"direct_c": 0.13872874645653183,
"pct_mediated": 0.07788418560377647
},
"ktr_speed_corr_raw": 0.8302610214864193,
"ktr_speed_partial_r": 0.30584870889845667,
"ktr_speed_partial_p": 2.4702473873095804e-08,
"ktr_speed_n": 319,
"ktr_speed_mean": 0.6759132625444266,
"ktr_matrix": [
{
"combine": "v_max_best",
"game": "game_s_p99",
"partial_r": 0.30584870889845667,
"p": 2.4702473873095804e-08,
"n": 319
},
{
"combine": "v_max_best",
"game": "game_accel_p95",
"partial_r": 0.3687551359304724,
"p": 1.0352948763638276e-11,
"n": 319
},
{
"combine": "v_max_best",
"game": "game_decel_p95",
"partial_r": 0.3108924979982271,
"p": 1.5662596788455318e-08,
"n": 317
},
{
"combine": "v_max_best",
"game": "game_lat_accel_p95",
"partial_r": 0.3482468151460165,
"p": 1.5865384596324834e-10,
"n": 319
},
{
"combine": "forty_s_peak",
"game": "game_s_p99",
"partial_r": 0.36332457008180696,
"p": 2.173897113465811e-11,
"n": 319
},
{
"combine": "forty_s_peak",
"game": "game_accel_p95",
"partial_r": 0.41397860164762684,
"p": 1.2224015239603115e-14,
"n": 319
},
{
"combine": "forty_s_peak",
"game": "game_decel_p95",
"partial_r": 0.3379575938837356,
"p": 6.572290032607263e-10,
"n": 317
},
{
"combine": "forty_s_peak",
"game": "game_lat_accel_p95",
"partial_r": 0.36792912816375184,
"p": 1.160002746226227e-11,
"n": 319
},
{
"combine": "skill_accel_p95_z_mean",
"game": "game_s_p99",
"partial_r": 0.18505309286960975,
"p": 0.00019427218136585276,
"n": 401
},
{
"combine": "skill_accel_p95_z_mean",
"game": "game_accel_p95",
"partial_r": 0.23696670396584146,
"p": 1.6435662707292986e-06,
"n": 400
},
{
"combine": "skill_accel_p95_z_mean",
"game": "game_decel_p95",
"partial_r": 0.21239832296473232,
"p": 1.929848382556122e-05,
"n": 398
},
{
"combine": "skill_accel_p95_z_mean",
"game": "game_lat_accel_p95",
"partial_r": 0.1871992565568519,
"p": 0.00016302355054765633,
"n": 401
},
{
"combine": "skill_decel_p95_z_mean",
"game": "game_s_p99",
"partial_r": 0.12550468953996965,
"p": 0.011891867255327735,
"n": 401
},
{
"combine": "skill_decel_p95_z_mean",
"game": "game_accel_p95",
"partial_r": 0.14763454104889825,
"p": 0.003079539488936608,
"n": 400
},
{
"combine": "skill_decel_p95_z_mean",
"game": "game_decel_p95",
"partial_r": 0.16353817943101023,
"p": 0.0010589190631854022,
"n": 398
},
{
"combine": "skill_decel_p95_z_mean",
"game": "game_lat_accel_p95",
"partial_r": 0.0873258441675602,
"p": 0.08071062290867408,
"n": 401
},
{
"combine": "skill_lat_accel_p95_z_mean",
"game": "game_s_p99",
"partial_r": 0.18605211314235864,
"p": 0.00017908687170144696,
"n": 401
},
{
"combine": "skill_lat_accel_p95_z_mean",
"game": "game_accel_p95",
"partial_r": 0.22994575480600699,
"p": 3.3689883896874273e-06,
"n": 400
},
{
"combine": "skill_lat_accel_p95_z_mean",
"game": "game_decel_p95",
"partial_r": 0.28820351032574176,
"p": 4.733605048221379e-09,
"n": 398
},
{
"combine": "skill_lat_accel_p95_z_mean",
"game": "game_lat_accel_p95",
"partial_r": 0.20088033112573955,
"p": 5.0903309599263846e-05,
"n": 401
}
],
"ktr_by_position": {
"v_max_best~game_s_p99": {
"DB": {
"partial_r": 0.25607759913225514,
"p": 0.025561742830790945,
"n": 76
},
"DL": {
"partial_r": 0.5936637890859454,
"p": 1.2667276684723876e-08,
"n": 77
},
"OL": {
"partial_r": -0.028136335193679956,
"p": 0.8132041469972429,
"n": 73
},
"TE": {
"partial_r": 0.15805786309706765,
"p": 0.4218011622278819,
"n": 28
},
"WR": {
"partial_r": 0.20072963596879922,
"p": 0.10886878281128061,
"n": 65
}
},
"skill_accel_p95_z_mean~game_accel_p95": {
"DB": {
"partial_r": -0.041742341672021205,
"p": 0.6879466879904005,
"n": 95
},
"DL": {
"partial_r": 0.5682819181950063,
"p": 1.0380798116491278e-09,
"n": 98
},
"OL": {
"partial_r": 0.16225787393129068,
"p": 0.12437991231554929,
"n": 91
},
"TE": {
"partial_r": 0.18336273584807922,
"p": 0.3070570679651459,
"n": 33
},
"WR": {
"partial_r": -0.10229206982581392,
"p": 0.3574637035015873,
"n": 83
}
},
"skill_decel_p95_z_mean~game_decel_p95": {
"DB": {
"partial_r": 0.008960381954635754,
"p": 0.9316942003440274,
"n": 94
},
"DL": {
"partial_r": 0.43242189079508125,
"p": 8.727276112533414e-06,
"n": 98
},
"OL": {
"partial_r": 0.07488625409394818,
"p": 0.48050815870774294,
"n": 91
},
"TE": {
"partial_r": -0.08662160161546915,
"p": 0.6373564064500419,
"n": 32
},
"WR": {
"partial_r": 0.1494453598186939,
"p": 0.17751369069121364,
"n": 83
}
},
"skill_lat_accel_p95_z_mean~game_lat_accel_p95": {
"DB": {
"partial_r": 0.028166856840493866,
"p": 0.7853017233470178,
"n": 96
},
"DL": {
"partial_r": 0.5415255810686956,
"p": 8.525334169346017e-09,
"n": 98
},
"OL": {
"partial_r": -0.012927525825772196,
"p": 0.9031991573998387,
"n": 91
},
"TE": {
"partial_r": -0.07841220694403697,
"p": 0.6644740734218565,
"n": 33
},
"WR": {
"partial_r": 0.15645969425656078,
"p": 0.15779868349478232,
"n": 83
}
}
},
"ktr_speed_by_position": {
"DB": {
"n": 76,
"game_s_p99_mean": 8.072674849866921,
"game_s_p99_median": 8.087334306921269,
"forty_s_peak_mean": 11.016331611813904,
"forty_s_peak_median": 11.018238098571421,
"ktr_speed_mean": 0.7328610899351813,
"ktr_speed_median": 0.7414488536129277,
"ktr_speed_vmax_mean": 0.7025942764901021,
"ktr_speed_vmax_median": 0.7098063469105778
},
"DL": {
"n": 77,
"game_s_p99_mean": 6.290885704008676,
"game_s_p99_median": 6.250731421606875,
"forty_s_peak_mean": 10.081208186774761,
"forty_s_peak_median": 10.123532812574122,
"ktr_speed_mean": 0.6242664210414742,
"ktr_speed_median": 0.6273950349936569,
"ktr_speed_vmax_mean": 0.6059033524783656,
"ktr_speed_vmax_median": 0.6054477291876363
},
"OL": {
"n": 73,
"game_s_p99_mean": 4.904428540737426,
"game_s_p99_median": 4.9444113240923135,
"forty_s_peak_mean": 9.49608088866066,
"forty_s_peak_median": 9.475238095457136,
"ktr_speed_mean": 0.5169653348070008,
"ktr_speed_median": 0.5113179447384733,
"ktr_speed_vmax_mean": 0.5049506012199357,
"ktr_speed_vmax_median": 0.5042584060489878
},
"TE": {
"n": 28,
"game_s_p99_mean": 7.926847043090933,
"game_s_p99_median": 7.936759965487893,
"forty_s_peak_mean": 10.507090048983667,
"forty_s_peak_median": 10.487190478285708,
"ktr_speed_mean": 0.7547170080110531,
"ktr_speed_median": 0.7613903789590142,
"ktr_speed_vmax_mean": 0.729493975119058,
"ktr_speed_vmax_median": 0.7322444503307798
},
"WR": {
"n": 65,
"game_s_p99_mean": 9.029361579869864,
"game_s_p99_median": 8.982114244188574,
"forty_s_peak_mean": 11.079425740530398,
"forty_s_peak_median": 11.078857143999993,
"ktr_speed_mean": 0.8150742743259116,
"ktr_speed_median": 0.8215239258171904,
"ktr_speed_vmax_mean": 0.7812296240583602,
"ktr_speed_vmax_median": 0.7854944423239224
}
},
"dl_edge_idl_split": {
"EDGE": {
"n": 66,
"v_max_best~game_s_p99": {
"partial_r": 0.38879711781514187,
"p": 0.009976423301886238,
"n": 43
},
"skill_accel_p95_z_mean~game_accel_p95": {
"partial_r": 0.28858476252370857,
"p": 0.031006872713072092,
"n": 56
},
"skill_decel_p95_z_mean~game_decel_p95": {
"partial_r": 0.356791540768395,
"p": 0.006949999532595841,
"n": 56
},
"skill_lat_accel_p95_z_mean~game_lat_accel_p95": {
"partial_r": 0.24034377252947578,
"p": 0.07438186620144274,
"n": 56
},
"Y2_M1_spearman": {
"rho": 0.4378129117259552,
"p": 0.0026321231626446197,
"n": 45
}
},
"IDL": {
"n": 52,
"v_max_best~game_s_p99": {
"partial_r": -0.06578696266634802,
"p": 0.7116409230471191,
"n": 34
},
"skill_accel_p95_z_mean~game_accel_p95": {
"partial_r": 0.2349652481408933,
"p": 0.1341712460265999,
"n": 42
},
"skill_decel_p95_z_mean~game_decel_p95": {
"partial_r": -0.030729923716319156,
"p": 0.8468117156850605,
"n": 42
},
"skill_lat_accel_p95_z_mean~game_lat_accel_p95": {
"partial_r": 0.3957036680784356,
"p": 0.009489631932436758,
"n": 42
},
"Y2_M1_spearman": {
"rho": 0.31253420908593316,
"p": 0.10539680055946204,
"n": 28
}
}
},
"Y2_sensitivity": {
"n_plays_100": {
"n": 286,
"oof_r_mean": 0.17675125230751543
},
"n_plays_150": {
"n": 261,
"oof_r_mean": 0.1116163926404854
}
}
}

## 分位置实战速度保持率 game_s_p99 / forty_s_peak（第 4 轮新增）
- DB：n=76，game_s_p99 均值 8.07 / 中位数 8.09，forty_s_peak 均值 11.02 / 中位数 11.02，保持率均值 0.733 / 中位数 0.741
- DL：n=77，game_s_p99 均值 6.29 / 中位数 6.25，forty_s_peak 均值 10.08 / 中位数 10.12，保持率均值 0.624 / 中位数 0.627
- OL：n=73，game_s_p99 均值 4.90 / 中位数 4.94，forty_s_peak 均值 9.50 / 中位数 9.48，保持率均值 0.517 / 中位数 0.511
- TE：n=28，game_s_p99 均值 7.93 / 中位数 7.94，forty_s_peak 均值 10.51 / 中位数 10.49，保持率均值 0.755 / 中位数 0.761
- WR：n=65，game_s_p99 均值 9.03 / 中位数 8.98，forty_s_peak 均值 11.08 / 中位数 11.08，保持率均值 0.815 / 中位数 0.822

## DL 亚位置拆分 EDGE vs IDL（第 4 轮新增）
{
"EDGE": {
"n": 66,
"v_max_best~game_s_p99": {
"partial_r": 0.38879711781514187,
"p": 0.009976423301886238,
"n": 43
},
"skill_accel_p95_z_mean~game_accel_p95": {
"partial_r": 0.28858476252370857,
"p": 0.031006872713072092,
"n": 56
},
"skill_decel_p95_z_mean~game_decel_p95": {
"partial_r": 0.356791540768395,
"p": 0.006949999532595841,
"n": 56
},
"skill_lat_accel_p95_z_mean~game_lat_accel_p95": {
"partial_r": 0.24034377252947578,
"p": 0.07438186620144274,
"n": 56
},
"Y2_M1_spearman": {
"rho": 0.4378129117259552,
"p": 0.0026321231626446197,
"n": 45
}
},
"IDL": {
"n": 52,
"v_max_best~game_s_p99": {
"partial_r": -0.06578696266634802,
"p": 0.7116409230471191,
"n": 34
},
"skill_accel_p95_z_mean~game_accel_p95": {
"partial_r": 0.2349652481408933,
"p": 0.1341712460265999,
"n": 42
},
"skill_decel_p95_z_mean~game_decel_p95": {
"partial_r": -0.030729923716319156,
"p": 0.8468117156850605,
"n": 42
},
"skill_lat_accel_p95_z_mean~game_lat_accel_p95": {
"partial_r": 0.3957036680784356,
"p": 0.009489631932436758,
"n": 42
},
"Y2_M1_spearman": {
"rho": 0.31253420908593316,
"p": 0.10539680055946204,
"n": 28
}
}
}

（md 全文完。注：Y1_year1_hurdle 一节中 M0b 的 alpha/C 中位数在 md 文件里原文即印为“393”，与 json 文件的 0.3931085237255467 不同，属 md 生成时的印值问题，已如实保留原文；其余各节数值与 json 一致。）

---

## combine_features_summary.json — shadow_calibration 全文

{
"shadow_pure_vs_three_cone": {
"label": "shadow_pure_vs_three_cone",
"n": 180,
"oof_r_within": 0.42068022525442056,
"oof_r2_within": 0.17652730996847277,
"oof_r_pooled_raw": 0.2642410837417748,
"fold_r": [
0.504,
0.517,
0.48,
0.273,
0.395
],
"alpha": 146.7799267622069
},
"shadow_full_vs_three_cone": {
"label": "shadow_full_vs_three_cone",
"n": 180,
"oof_r_within": 0.4936569840534179,
"oof_r2_within": 0.24359559685812104,
"oof_r_pooled_raw": 0.2911076658407567,
"fold_r": [
0.557,
0.53,
0.636,
0.425,
0.462
],
"alpha": 146.7799267622069
},
"shadow_pure_vs_short_shuttle": {
"label": "shadow_pure_vs_short_shuttle",
"n": 201,
"oof_r_within": 0.28359515507775596,
"oof_r2_within": 0.07941512297855102,
"oof_r_pooled_raw": 0.2085928048079341,
"fold_r": [
0.412,
0.335,
0.172,
0.225,
0.247
],
"alpha": 316.22776601683796
},
"shadow_full_vs_short_shuttle": {
"label": "shadow_full_vs_short_shuttle",
"n": 201,
"oof_r_within": 0.4460276880310409,
"oof_r2_within": 0.19805768302552773,
"oof_r_pooled_raw": 0.3064685560012302,
"fold_r": [
0.529,
0.4,
0.505,
0.376,
0.388
],
"alpha": 215.44346900318823
}
}

## targets_summary.json — y2 一节全文

{
"covered": 299,
"covered_reg_50": 335,
"covered_reg_150": 285,
"by_position": {
"DB": 71,
"DL": 73,
"OL": 67,
"TE": 22,
"WR": 66
},
"z_qual_mean": 1.176316592349485e-15,
"z_qual_std": 0.7648342134280289,
"z_qual_raw_mean": -0.0032839922414159757,
"z_qual_raw_std": 0.7636412720371446,
"eb_kappa": {
"gated:DB::m_epa": 200.0,
"gated:DL::m_getoff": 8.568,
"gated:OL::m_ppa": 39.766,
"gated:OL::m_press_allowed_inv": 128.11,
"gated:DL::m_press_rate": 142.316,
"gated:OL::m_quick_collapse_inv": 161.912,
"gated:DL::m_quick_press": 200.0,
"gated:TE::m_sep": 180.574,
"gated:WR::m_sep": 62.027,
"gated:DB::m_tackle_rate": 159.359,
"gated:TE::m_tprr": 91.452,
"gated:WR::m_tprr": 66.728,
"gated:TE::m_yacoe": 12.285,
"gated:WR::m_yacoe": 29.311,
"gated:TE::m_yprr": 166.675,
"gated:WR::m_yprr": 111.539,
"raw:DB::m_epa": 65.081,
"raw:DL::m_getoff": 6.885,
"raw:OL::m_ppa": 5.693,
"raw:OL::m_press_allowed_inv": 7.248,
"raw:DL::m_press_rate": 59.046,
"raw:OL::m_quick_collapse_inv": 31.681,
"raw:DL::m_quick_press": 130.829,
"raw:TE::m_sep": 53.11,
"raw:WR::m_sep": 47.922,
"raw:DB::m_tackle_rate": 23.532,
"raw:TE::m_tprr": 30.82,
"raw:WR::m_tprr": 39.269,
"raw:TE::m_yacoe": 11.262,
"raw:WR::m_yacoe": 17.37,
"raw:TE::m_yprr": 68.696,
"raw:WR::m_yprr": 87.188
}
}

- y2 covered=299；covered_reg_50=335；covered_reg_150=285；分位置人数 DB 71、DL 73、OL 67、TE 22、WR 66（合计 299）。
- eb_kappa 贴 200 上限的条目共 2 条：gated:DB::m_epa = 200.0、gated:DL::m_quick_press = 200.0。其余 gated 值最高为 gated:TE::m_sep 180.574。
- targets_summary.json 共 67 行，顶层键：players、reg_plays_used、asr_vs_career_snaps_corr、first_season_gap_counts、y1、y2、y3_start_rate_mean。

## game_kinematics

- data/derived/game_kinematics.parquet：401 行 × 8 列；列名：nfl_id、game_frames、game_s_p99、game_s_p50、game_accel_p95、game_decel_p95、game_lat_accel_p95、game_snap_burst_1s。
- 球员数 401（nfl_id 去重 401），总帧数 game_frames 求和 = 10,424,763。

## oof_predictions.parquet 与 experiments.csv

- data/derived/oof_predictions.parquet：510 行 × 61 列；列为 nfl_id + 60 列折外预测（5 个目标 × 12 个模型：Y1_hurdle_stage1、Y1_stage2_logASR、Y2_quality、Y1_year1_hurdle、Y1_year1_logsnaps 各含 M0a、M0b_stopwatch、M0c_ngs、M0b、M0all、M1a、M1、M1full、M1b、M1_track_only、M1_over_stopwatch、M1_over_all）。
- logs/experiments.csv：共 16 行（表头 + 15 行数据），本轮新增 3 行（日期 2026-11-22），内容如下：
  - phase3_Y1_hurdle_stage1,2026-11-22,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,AUC,0.8641,0.0035,0.0011,M0a,0.0009,,perm p=0.0599
  - phase3_Y1_stage2_logASR,2026-11-22,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.6607,0.0077,0.0024,M0a,-0.0005,,perm p=0.1168
  - phase3_Y2_quality,2026-11-22,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.1335,0.029,0.0092,M0a,0.0326,,perm p=0.0030
  - 注：新增 3 行数值与文件中 2026-11-21 的同名 3 行完全相同，属确定性种子 42 下的复现；此前 12 行分别为 2026-11-21 的三组实验：M0a+CORE_TRACK(16) 三行、M0a+TRACK_PURE+STATIC+shadow_pure 两组各三行。

## 重点核对数（与第 4 轮对照）

- Y2_quality 各模型：M0a 0.1009、M0b_stopwatch −0.0016、M0c_ngs 0.1984、M0b 0.1756、**M0all 0.1829（第 4 轮 0.1629，Bug B 修复后 +0.0200）**、M1a 0.1403、M1 0.1335、M1full 0.1279、M1b 0.1587、M1_track_only 0.0902、M1_over_stopwatch 0.0755、M1_over_all 0.1823。
- Y2 增量：Δ M1 vs M0a +0.0326 CI [−0.0114, +0.0819]；Δ M1a vs M0a +0.0394 CI [−0.0097, +0.0951]；Δ M1_over_stopwatch +0.0771 CI [+0.0122, +0.1545]；Δ M1_track_only +0.0918 CI [−0.0076, +0.2217]；**Δ M1_over_all vs M0all −0.0006 CI [−0.0169, +0.0158]（第 4 轮 +0.0194 确为口径伪影，现已归零）**；置换 obs +0.0415，p_perm=0.0030。
- 对抗验证四分块 AUC：ALL 0.4922、FORTY_ONLY 0.4924、SKILL_ONLY 0.4892、STATIC_ONLY 0.4879——全部随机水平（第 4 轮 0.4922/0.4924/0.4892/0.4879，与第 4 轮逐数一致）。
- 选秀定价回归（-log_pick）：neg_forty +0.271、vertical +0.232、forty_s_peak +0.225、broad_jump +0.221、v_max_best +0.189、skill_s_peak_z_mean +0.153（p=0.000254，q=0.001）。
- 稳健性阶梯 5 步：① log_pick β=+0.156，p=0.000316，n=299；② +forty β=+0.094，p=0.0774，n=253；③ +ngs β=+0.138，p=0.00268，n=298；④ 三重 β=+0.093，p=0.0886，n=252；**⑤ forty.notna() 子集只控 log_pick β=+0.136，p=0.00302，n=253（全精度 β=0.1361201412957797；冒烟版为 β=0.1238）**。分解：样本缩水代价 0.156→0.136（−0.020）vs forty 共线性代价 0.136→0.094（−0.042），约 1:2。
- Y1_year1_hurdle：M0a 0.8438、M1 0.8427，Δ −0.0011；时序外推 M0a 0.8756、M1 0.8670（第 4 轮 0.8701，−0.003）。
- Y1_year1_logsnaps：M0a 0.5794、M1 0.5807，Δ +0.0013；时序外推 M0a 0.6293、M1 0.6241（第 4 轮 0.6200，+0.004）。
- **LOCO_Y2（Bug A 修复后，全量验证）**：2023 M0a 0.1971 / M1 0.1831（第 4 轮 M1 0.1537，+0.029；与 M0a 差距由 −0.043 收窄至 −0.014）；2024 M0a 0.0938 / M1 0.1315（第 4 轮 0.1303，+0.001；高于 M0a）；2025 M0a 0.0106 / M1 0.0513（第 4 轮 0.0496，+0.002；高于 M0a）。三届 M1 均回升或持平，无回归。
- ktr_speed_by_position WR 保持率：n=65，ktr_speed_mean=0.8151（9.03/11.08=0.815，逐数吻合）、中位数 0.8215；原 v_max 版 0.7812。
- Y2_sensitivity：n_plays_100 n=286 r=0.1768；n_plays_150 n=261 r=0.1116。
- Y2 分位置 Spearman（M1 折外）：DB −0.166、DL +0.507（p=4.78e-06）、OL −0.036、TE −0.158、WR +0.242（p=0.0505）。
- M3 LightGBM：Y2 r=0.1457（第 4 轮 0.1457，一致）、Y1 hurdle AUC 0.8462。M2 中介：decel indirect=−0.0050（−3.3%）；burst_1s +0.0117（+7.8%）。
- ktr_speed_mean=0.6759（第 4 轮 0.6524——分母改 forty_s_peak 后的新口径；原 v_max 版另存）。
- forty_s_peak 分年：10.419 / 10.368 / 10.393（WR 11.033 / 11.070 / 11.079），三年稳定。

## model_phase3 全量实际耗时与提速核对

- 15:12:33 启动，约 20:58:55 退出，全量实际耗时 5 小时 46 分 22 秒（第 4 轮 7 小时 14 分，提速约 20%）。
- **30–45 分钟预估仍未兑现**：首个结果章节约 88 分钟才打印；置换缓存只省 Stage 1 重拟合，主 CV 仍占主导。提速幅度有限（7h14m → 5h46m），不是量级变化；下次报时仍按小时级打底。

## 服务器收尾确认（2026-11-22 21:58，浏览器任务核对）

- tokyo-node-work：e2-standard-2（2 vCPU，8 GB），状态“正在运行”，asia-northeast1-a，外部 IP 35.213.17.8，内部 IP 10.146.0.2。
- x-ui：curl http://127.0.0.1:2053/287a58ff96/ 返回 HTTP 200，面板正常。
