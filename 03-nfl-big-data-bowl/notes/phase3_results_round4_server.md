# 服务器第 4 轮全量结果（2026-11-22）

- 浏览器任务：browser-task:45fd7042-d0c6-55a5-a951-ce681ef24964，2026-11-22 03:41–12:44 CST（运行时实测 9h03m）。
- 服务器：tokyo-node-work，e2-standard-2（2 核 8G），IP 35.213.17.8，未升降配、未停机改配置；备份到 prefix_snapshot_r3/（9 文件、1.4M，md5 核对）；上传 model_phase3.py（md5 8c8bf463b8f1dbeaa81724d9bf95c7a4，84,705 字节，ast.parse 通过）。
- 四步：features_combine 03:55–03:56（exit 0）→ targets 04:04（exit 0）→ game_kinematics 04:06–04:39 → model_phase3 --within-year 04:45–11:59，**历时约 7 小时 14 分**（第 3 轮为 4 小时 36 分；--within-year 版更慢，原因待查：可能是 (pos, year)/(drill, year) 分组统计量增多或服务器负载波动）。
- 完成证据：12:04 检查时 ps 已无 model_phase3 进程；logs/phase3_results.md（27,789 字节）、logs/phase3_results.json（42,396 字节）、data/derived/oof_predictions.parquet（255,723 字节）均于 11:59 重写；日志 logs/phase3c_model_r4.log（1,758,210 字节）尾部正常写出结果并 appended 3 rows to experiments.csv；grep error/traceback 计数 0（仅 sklearn FutureWarning）。
- 注意：浏览器本次未下载结果文件到本地（browser_downloads 无新文件），以下全文以浏览器任务回传文本为准，已逐字转存；phase3_results.json 本地无副本，数字提取改用解析本文件的脚本（notes/extract_round4_numbers_from_md.py）。
- experiments.csv 新增行日期栏为 2026-11-21（脚本内部日期），实际运行为 2026-11-22，属外观问题。
- x-ui：本轮确认面板端口可达——ss 显示 *:2053 监听（x-ui pid 596），外部实测 http://35.213.17.8:2053/287a58ff96/ 返回 HTTP 200（第 3 轮未能确认的问题本轮已确认）；54321 仍未监听、外部 curl 超时。x-ui.service Active (running)，自 2026-11-21 11:21 CST 运行至今；xray 进程 1142 在运行。

## （1）logs/phase3_results.md 全文（749 行，27,789 字节，2026-11-22 11:59 版）

```
# NFL Big Data Bowl 2027 — Phase 3 主建模结果
种子 42；10×5 折重复分层 CV；置换 1000 次；自助 2000 次。
特征分组：TRACK_PURE（四十曲线 + 9 项无质量技能指标，折内干净 drill 归一化）；STATIC（静态秒表派生）；SHADOW（纯技能影子指数）。
Round-2 改动：M0b 已拆为 M0b_stopwatch（纯秒表/双跳）与 M0c_ngs（NGS 评分）；M1 系列为两阶段残差/偏置堆叠（基线用自身最优惩罚）；置换检验基线与全模型各自用正确正则，聚合 3 次重复。
Round-3 改动（第 4 轮修复，Gemini 第 3 轮核对）：DrillFoldClean 静默清零修复（单列 groupby 不用列表、transform 补 [m]、加非空断言）；M1_over_stopwatch/M1_over_all 的秒表与 NGS 列改走位置内标准化；--within-year 同届内调和开启（未见届用测试届自身无标签分布）；permutation_p 每个 null 样本也跨 3 次重复取均值（尺度对齐）；分类偏置列改内 5 折拟合；lower_body_burst 在 FeaturePrep 内由原始 vertical/broad_jump 折内重算。
Drill 归一化：fold-clean drill normalization from skill_events.parquet (28 stable drills, 4238 player-drill rows); WITHIN-YEAR harmonization ON

## Y1_hurdle_stage1
- M0a：0.8631 ± 0.0033（内层 CV 选定 alpha/C 中位数 1.93）
- M0b_stopwatch：0.5968 ± 0.0084（内层 CV 选定 alpha/C 中位数 0.0283）
- M0c_ngs：0.8082 ± 0.0034（内层 CV 选定 alpha/C 中位数 1）
- M0b：0.8022 ± 0.0045（内层 CV 选定 alpha/C 中位数 0.268）
- M0all：0.8645 ± 0.0036（内层 CV 选定 alpha/C 中位数 0.0373）
- M1a：0.8651 ± 0.0028（内层 CV 选定 alpha/C 中位数 0.01）
- M1：0.8641 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.01）
- M1full：0.8643 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.01）
- M1b：0.7873 ± 0.0061（内层 CV 选定 alpha/C 中位数 0.139）
- M1_track_only：0.5688 ± 0.0081（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_stopwatch：0.5688 ± 0.0094（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all：0.8597 ± 0.0039（内层 CV 选定 alpha/C 中位数 0.01）
- delta_M1_vs_M0a：+0.0009，95% CI [-0.0079, +0.0065]
- delta_M1b_vs_M0b：-0.0150，95% CI [-0.0281, +0.0023]
- delta_M1a_vs_M0a：+0.0020，95% CI [-0.0078, +0.0063]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：-0.0280，95% CI [-0.0711, +0.0123]
- delta_M1_track_only_vs_M0b_stopwatch：-0.0279，95% CI [-0.0763, +0.0150]
- delta_M1_over_all_vs_M0all：-0.0048，95% CI [-0.0113, +0.0050]
- 置换检验（折内 X 置换）：观测 Δ=+0.0015，p_perm=0.0599（零分布均值 -0.0009 ± 0.0016）

## Y1_stage2_logASR
- M0a：0.6611 ± 0.0078（内层 CV 选定 alpha/C 中位数 14.7）
- M0b_stopwatch：0.4681 ± 0.0125（内层 CV 选定 alpha/C 中位数 31.6）
- M0c_ngs：0.5986 ± 0.0186（内层 CV 选定 alpha/C 中位数 10）
- M0b：0.5861 ± 0.0175（内层 CV 选定 alpha/C 中位数 21.5）
- M0all：0.6588 ± 0.0072（内层 CV 选定 alpha/C 中位数 1e+03）
- M1a：0.6605 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1：0.6607 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1full：0.6609 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1b：0.5483 ± 0.0261（内层 CV 选定 alpha/C 中位数 31.6）
- M1_track_only：0.4245 ± 0.0374（内层 CV 选定 alpha/C 中位数 46.4）
- M1_over_stopwatch：0.4668 ± 0.0118（内层 CV 选定 alpha/C 中位数 1.47e+04）
- M1_over_all：0.6559 ± 0.0101（内层 CV 选定 alpha/C 中位数 1e+05）
- delta_M1_vs_M0a：-0.0005，95% CI [-0.0011, +0.0003]
- delta_M1b_vs_M0b：-0.0378，95% CI [-0.0576, +0.0052]
- delta_M1a_vs_M0a：-0.0007，95% CI [-0.0017, +0.0005]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：-0.0013，95% CI [-0.0056, +0.0042]
- delta_M1_track_only_vs_M0b_stopwatch：-0.0436，95% CI [-0.1052, +0.0375]
- delta_M1_over_all_vs_M0all：-0.0029，95% CI [-0.0180, +0.0143]
- 置换检验（折内 X 置换）：观测 Δ=+0.0000，p_perm=0.1168（零分布均值 -0.0000 ± 0.0000）

## Y2_quality
- M0a：0.1009 ± 0.0275（内层 CV 选定 alpha/C 中位数 147）
- M0b_stopwatch：-0.0016 ± 0.0356（内层 CV 选定 alpha/C 中位数 1e+03）
- M0c_ngs：0.1984 ± 0.0256（内层 CV 选定 alpha/C 中位数 46.4）
- M0b：0.1756 ± 0.0279（内层 CV 选定 alpha/C 中位数 100）
- M0all：0.1629 ± 0.0315（内层 CV 选定 alpha/C 中位数 266）
- M1a：0.1403 ± 0.0271（内层 CV 选定 alpha/C 中位数 1.33e+03）
- M1：0.1335 ± 0.0290（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1full：0.1279 ± 0.0312（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1b：0.1587 ± 0.0210（内层 CV 选定 alpha/C 中位数 464）
- M1_track_only：0.0902 ± 0.0295（内层 CV 选定 alpha/C 中位数 681）
- M1_over_stopwatch：0.0755 ± 0.0358（内层 CV 选定 alpha/C 中位数 1.33e+03）
- M1_over_all：0.1823 ± 0.0317（内层 CV 选定 alpha/C 中位数 4.55e+03）
- delta_M1_vs_M0a：+0.0326，95% CI [-0.0114, +0.0819]
- delta_M1b_vs_M0b：-0.0169，95% CI [-0.0910, +0.0646]
- delta_M1a_vs_M0a：+0.0394，95% CI [-0.0097, +0.0951]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.0771，95% CI [+0.0122, +0.1545]
- delta_M1_track_only_vs_M0b_stopwatch：+0.0918，95% CI [-0.0076, +0.2217]
- delta_M1_over_all_vs_M0all：+0.0194，95% CI [-0.0237, +0.0568]
- 置换检验（折内 X 置换）：观测 Δ=+0.0415，p_perm=0.0030（零分布均值 -0.0051 ± 0.0090）

## Y1_year1_hurdle
- M0a：0.8438 ± 0.0050（内层 CV 选定 alpha/C 中位数 0.518）
- M0b_stopwatch：0.5760 ± 0.0117（内层 CV 选定 alpha/C 中位数 0.01）
- M0c_ngs：0.7824 ± 0.0048（内层 CV 选定 alpha/C 中位数 0.393）
- M0b：0.7759 ± 0.0060（内层 CV 选定 alpha/C 中位数 0.393）
- M0all：0.8459 ± 0.0045（内层 CV 选定 alpha/C 中位数 0.072）
- M1a：0.8422 ± 0.0043（内层 CV 选定 alpha/C 中位数 0.01）
- M1：0.8427 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.01）
- M1full：0.8426 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.01）
- M1b：0.7510 ± 0.0072（内层 CV 选定 alpha/C 中位数 0.072）
- M1_track_only：0.5327 ± 0.0121（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_stopwatch：0.5247 ± 0.0126（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all：0.8385 ± 0.0030（内层 CV 选定 alpha/C 中位数 0.01）
- delta_M1_vs_M0a：-0.0011，95% CI [-0.0087, +0.0062]
- delta_M1b_vs_M0b：-0.0249，95% CI [-0.0368, -0.0097]
- delta_M1a_vs_M0a：-0.0016，95% CI [-0.0087, +0.0042]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：-0.0513，95% CI [-0.0932, -0.0110]
- delta_M1_track_only_vs_M0b_stopwatch：-0.0434，95% CI [-0.0917, +0.0037]
- delta_M1_over_all_vs_M0all：-0.0074，95% CI [-0.0140, +0.0007]
- 置换检验（折内 X 置换）：观测 Δ=-0.0057，p_perm=0.6856（零分布均值 -0.0050 ± 0.0018）

## Y1_year1_logsnaps
- M0a：0.5794 ± 0.0042（内层 CV 选定 alpha/C 中位数 18.1）
- M0b_stopwatch：0.2230 ± 0.0169（内层 CV 选定 alpha/C 中位数 100）
- M0c_ngs：0.5176 ± 0.0062（内层 CV 选定 alpha/C 中位数 6.81）
- M0b：0.5161 ± 0.0049（内层 CV 选定 alpha/C 中位数 18.1）
- M0all：0.5868 ± 0.0067（内层 CV 选定 alpha/C 中位数 68.1）
- M1a：0.5807 ± 0.0060（内层 CV 选定 alpha/C 中位数 1.74e+03）
- M1：0.5807 ± 0.0057（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1full：0.5806 ± 0.0058（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1b：0.4904 ± 0.0103（内层 CV 选定 alpha/C 中位数 46.4）
- M1_track_only：0.2077 ± 0.0177（内层 CV 选定 alpha/C 中位数 147）
- M1_over_stopwatch：0.2216 ± 0.0150（内层 CV 选定 alpha/C 中位数 3.48e+03）
- M1_over_all：0.5902 ± 0.0052（内层 CV 选定 alpha/C 中位数 7.35e+03）
- delta_M1_vs_M0a：+0.0013，95% CI [-0.0047, +0.0095]
- delta_M1b_vs_M0b：-0.0258，95% CI [-0.0481, +0.0102]
- delta_M1a_vs_M0a：+0.0014，95% CI [-0.0048, +0.0098]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：-0.0015，95% CI [-0.0150, +0.0159]
- delta_M1_track_only_vs_M0b_stopwatch：-0.0153，95% CI [-0.0730, +0.0644]
- delta_M1_over_all_vs_M0all：+0.0034，95% CI [-0.0044, +0.0117]
- 置换检验（折内 X 置换）：观测 Δ=+0.0058，p_perm=0.0030（零分布均值 -0.0016 ± 0.0015）

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
"metric": 0.8701177100053504
},
"Y1_year1_logsnaps_M0a": {
"n_train": 253,
"n_test": 125,
"metric": 0.629337524994706
},
"Y1_year1_logsnaps_M1": {
"n_train": 253,
"n_test": 125,
"metric": 0.6200033504516129
},
"LOCO_Y2": {
"Y2_test_2023_M0a": {
"n_test": 112,
"r": 0.19713610636162895
},
"Y2_test_2023_M1": {
"n_test": 112,
"r": 0.1536861499431084
},
"Y2_test_2024_M0a": {
"n_test": 104,
"r": 0.09381314854689464
},
"Y2_test_2024_M1": {
"n_test": 104,
"r": 0.1303410956074565
},
"Y2_test_2025_M0a": {
"n_test": 83,
"r": 0.010633263245448521
},
"Y2_test_2025_M1": {
"n_test": 83,
"r": 0.04955113579713564
}
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
"n": 252,
"beta": 0.09262906226480878,
"p": 0.0886269512288863
}

## 稳健性阶梯：skill_s_peak_z_mean（四步控制，第 4 轮新增）
- log_pick：β=+0.156，p=0.000316，n=299
- log_pick+forty：β=+0.094，p=0.0774，n=253
- log_pick+ngs_final_score：β=+0.138，p=0.00268，n=298
- log_pick+forty+ngs_final_score：β=+0.093，p=0.0886，n=252

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
}
},
"ktr_speed_corr_raw": 0.8302610214864193,
"ktr_speed_partial_r": 0.30584870889845667,
"ktr_speed_partial_p": 2.4702473873095804e-08,
"ktr_speed_n": 319,
"ktr_speed_mean": 0.6524102630415808,
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
"partial_r": 0.2369667039584146,
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
"ktr_speed_mean": 0.7025942764901021,
"ktr_speed_median": 0.7098063469105778
},
"DL": {
"n": 77,
"game_s_p99_mean": 6.290885704008676,
"game_s_p99_median": 6.250731421606875,
"forty_s_peak_mean": 10.081208186774761,
"forty_s_peak_median": 10.123523812571422,
"ktr_speed_mean": 0.6059033524783656,
"ktr_speed_median": 0.6054477291876363
},
"OL": {
"n": 73,
"game_s_p99_mean": 4.904428540737426,
"game_s_p99_median": 4.9444113240923135,
"forty_s_peak_mean": 9.49608088866066,
"forty_s_peak_median": 9.475238095457136,
"ktr_speed_mean": 0.5049506012199357,
"ktr_speed_median": 0.5042584060489878
},
"TE": {
"n": 28,
"game_s_p99_mean": 7.926847043090933,
"game_s_p99_median": 7.936759965487893,
"forty_s_peak_mean": 10.507090048983667,
"forty_s_peak_median": 10.487190478285708,
"ktr_speed_mean": 0.729493975119058,
"ktr_speed_median": 0.7322444503307798
},
"WR": {
"n": 65,
"game_s_p99_mean": 9.029361579869864,
"game_s_p99_median": 8.98211424188574,
"forty_s_peak_mean": 11.079425740530398,
"forty_s_peak_median": 11.078857143999993,
"ktr_speed_mean": 0.7812296240583602,
"ktr_speed_median": 0.7854944423239224
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
"reg_snaps_50": {
"n": 299,
"oof_r_mean": 0.12165653717787678
},
"reg_snaps_150": {
"n": 283,
"oof_r_mean": 0.10112286698064447
}
}
}

## 分位置实战速度保持率 game_s_p99 / forty_s_peak（第 4 轮新增）
- DB：n=76，game_s_p99 均值 8.07 / 中位数 8.09，forty_s_peak 均值 11.02 / 中位数 11.02，保持率均值 0.703 / 中位数 0.710
- DL：n=77，game_s_p99 均值 6.29 / 中位数 6.25，forty_s_peak 均值 10.08 / 中位数 10.12，保持率均值 0.606 / 中位数 0.605
- OL：n=73，game_s_p99 均值 4.90 / 中位数 4.94，forty_s_peak 均值 9.50 / 中位数 9.48，保持率均值 0.505 / 中位数 0.504
- TE：n=28，game_s_p99 均值 7.93 / 中位数 7.94，forty_s_peak 均值 10.51 / 中位数 10.49，保持率均值 0.729 / 中位数 0.732
- WR：n=65，game_s_p99 均值 9.03 / 中位数 8.98，forty_s_peak 均值 11.08 / 中位数 11.08，保持率均值 0.781 / 中位数 0.785

## DL 亚位置拆分 EDGE vs IDL（第 4 轮新增）
（与前面 dl_edge_idl_split 的 JSON 数值完全一致，原文重复呈现，此处略，见上文。）
```
（全文结束。文件共 749 行，27,789 字节。）

## （2）Shadow 标定数（data/derived/combine_features_summary.json 的 shadow_calibration 全文）

文件头：players 510、forty_attempts 762、forty_r2_mean 0.9939181633291023、forty_r2_gt95_share 0.994750656167979、skill_events 5091、drills_normalized 28、stable_drills 31。

```
"shadow_calibration": {
"shadow_pure_vs_three_cone": {
"label": "shadow_pure_vs_three_cone",
"n": 180,
"oof_r_within": 0.42068022525442056,
"oof_r2_within": 0.17652730996847277,
"oof_r_pooled_raw": 0.2642410837417748,
"fold_r": [0.504, 0.517, 0.48, 0.273, 0.395],
"alpha": 146.7799267622069
},
"shadow_full_vs_three_cone": {
"label": "shadow_full_vs_three_cone",
"n": 180,
"oof_r_within": 0.4936569840534179,
"oof_r2_within": 0.24359559685812104,
"oof_r_pooled_raw": 0.2911076658407567,
"fold_r": [0.557, 0.53, 0.636, 0.425, 0.462],
"alpha": 146.7799267622069
},
"shadow_pure_vs_short_shuttle": {
"label": "shadow_pure_vs_short_shuttle",
"n": 201,
"oof_r_within": 0.28359515507775596,
"oof_r2_within": 0.07941512297855102,
"oof_r_pooled_raw": 0.2085928048079341,
"fold_r": [0.412, 0.335, 0.172, 0.225, 0.247],
"alpha": 316.22776601683796
},
"shadow_full_vs_short_shuttle": {
"label": "shadow_full_vs_short_shuttle",
"n": 201,
"oof_r_within": 0.4460276880310409,
"oof_r2_within": 0.19805768302552773,
"oof_r_pooled_raw": 0.3064685560012302,
"fold_r": [0.529, 0.4, 0.505, 0.376, 0.388],
"alpha": 215.44346900318823
}
},
"dropped_events": {
"bad_frames": 31,
"too_short": 0
}
```

## （3）targets_summary.json 关键数（67 行）

players 510；reg_plays_used 240412；asr_vs_career_snaps_corr 0.9655；first_season_gap_counts {"0": 378, "1": 20, "2": 3}；y1：zero_reg_snaps 109、hurdle_150_n 236、year1_hurdle_100_n 254、asr_median 108.75、asr_mean 235.7895424836601；y2：covered 299、covered_reg_50 335、covered_reg_150 285，分位置 DB 71、DL 73、OL 67、TE 22、WR 66，z_qual_mean 1.1763165692349485e-15、z_qual_std 0.7648342134280289、z_qual_raw_mean -0.0032839922414159757、z_qual_raw_std 0.7636412720371446；y3_start_rate_mean 0.226797385620915。

y2.eb_kappa 共 32 条（gated 16 条 + raw 16 条），其中 2 条贴 200 上限（均为 gated）：
- gated：DB::m_epa 200.0（贴上限）、DL::m_getoff 8.568、OL::m_ppa 39.766、OL::m_press_allowed_inv 128.11、DL::m_press_rate 142.316、OL::m_quick_collapse_inv 161.912、DL::m_quick_press 200.0（贴上限）、TE::m_sep 180.574、WR::m_sep 62.027、DB::m_tackle_rate 159.359、TE::m_tprr 91.452、WR::m_tprr 66.728、TE::m_yacoe 12.285、WR::m_yacoe 29.311、TE::m_yprr 166.675、WR::m_yprr 111.539
- raw：DB::m_epa 65.081、DL::m_getoff 6.885、OL::m_ppa 5.693、OL::m_press_allowed_inv 7.248、DL::m_press_rate 59.046、OL::m_quick_collapse_inv 31.681、DL::m_quick_press 130.829、TE::m_sep 53.11、WR::m_sep 47.922、DB::m_tackle_rate 23.532、TE::m_tprr 30.82、WR::m_tprr 39.269、TE::m_yacoe 11.262、WR::m_yacoe 17.37、TE::m_yprr 68.696、WR::m_yprr 87.188

## （4）game_kinematics（2026-11-22 04:44 观测）

game_kinematics.parquet：401 行、8 列（nfl_id、game_frames、game_s_p99、game_s_p50、game_accel_p95、game_decel_p95、game_lat_accel_p95、game_snap_burst_1s）；nfl_id 去重 401 人；game_frames 合计 10,424,763（均值 25,996.9，中位数 15,014，最小 20，最大 136,657）；分季累计人数 2023 年 122 人、2024 年累计 260 人、2025 年累计 401 人。

## （5）oof_predictions.parquet 与 experiments.csv

- oof_predictions.parquet（255,723 字节，11:59 版）：510 行 × 61 列，即 nfl_id + 5 个目标（Y1_hurdle_stage1、Y1_stage2_logASR、Y2_quality、Y1_year1_hurdle、Y1_year1_logsnaps）× 12 个模型（M0a、M0b_stopwatch、M0c_ngs、M0b、M0all、M1b、M1_track_only、M1a、M1、M1full、M1_over_stopwatch、M1_over_all）。
- experiments.csv：13 行（表头 + 12 条数据行），较跑前 10 行新增 3 行。新增的 3 行（特征集 M0a+TRACK_PURE+STATIC+shadow_pure，模型 M1）：
  - phase3_Y1_hurdle_stage1，2026-11-21，AUC 0.8641 ± 0.0035（se 0.0011），vs M0a +0.0009，perm p=0.0599
  - phase3_Y1_stage2_logASR，2026-11-21，r 0.6607 ± 0.0077（se 0.0024），vs M0a -0.0005，perm p=0.1168
  - phase3_Y2_quality，2026-11-21，r 0.1335 ± 0.029（se 0.0092），vs M0a +0.0326，perm p=0.0030
  - 注：新增行日期栏为 2026-11-21（脚本内部日期），实际运行为 2026-11-22；wins 列为空。其余 9 条旧行不变（3 条 M0a+CORE_TRACK(16) 与 6 条 TRACK_PURE+STATIC+shadow_pure）。
