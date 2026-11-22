# NFL Big Data Bowl 2027 — Phase 3 主建模结果
种子 42；10×5 折重复分层 CV；置换 1000 次；自助 2000 次。
特征分组：TRACK_PURE（四十曲线 + 9 项无质量技能指标，折内干净 drill 归一化）；STATIC（静态秒表派生）；SHADOW（纯技能影子指数）。
Drill 归一化：fold-clean drill normalization from skill_events.parquet (28 stable drills, 4238 player-drill rows)
## Y1_hurdle_stage1
- M0a：0.8631 ± 0.0033（内层 CV 选定 alpha/C 中位数 1.93）
- M0b：0.7976 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.268）
- M1a：0.8588 ± 0.0034（内层 CV 选定 alpha/C 中位数 0.072）
- M1：0.8556 ± 0.0032（内层 CV 选定 alpha/C 中位数 0.072）
- M1full：0.8552 ± 0.0031（内层 CV 选定 alpha/C 中位数 0.072）
- M1b：0.7889 ± 0.0064（内层 CV 选定 alpha/C 中位数 0.139）
- delta_M1_vs_M0a：-0.0076，95% CI [-0.0179, +0.0050]
- delta_M1b_vs_M0b：-0.0087，95% CI [-0.0225, +0.0097]
- delta_M1a_vs_M0a：-0.0043，95% CI [-0.0159, +0.0071]
- 置换检验（折内 X 置换）：观测 Δ=-0.0092，p_perm=0.5355（零分布均值 -0.0087 ± 0.0055）
## Y1_stage2_logASR
- M0a：0.6611 ± 0.0078（内层 CV 选定 alpha/C 中位数 14.7）
- M0b：0.5978 ± 0.0126（内层 CV 选定 alpha/C 中位数 14.7）
- M1a：0.6416 ± 0.0060（内层 CV 选定 alpha/C 中位数 31.6）
- M1：0.6315 ± 0.0070（内层 CV 选定 alpha/C 中位数 46.4）
- M1full：0.6310 ± 0.0070（内层 CV 选定 alpha/C 中位数 46.4）
- M1b：0.5742 ± 0.0094（内层 CV 选定 alpha/C 中位数 31.6）
- delta_M1_vs_M0a：-0.0296，95% CI [-0.0461, -0.0024]
- delta_M1b_vs_M0b：-0.0236，95% CI [-0.0423, +0.0040]
- delta_M1a_vs_M0a：-0.0195，95% CI [-0.0368, +0.0047]
- 置换检验（折内 X 置换）：观测 Δ=-0.0297，p_perm=0.4895（零分布均值 -0.0307 ± 0.0142）
## Y2_quality
- M0a：0.1009 ± 0.0275（内层 CV 选定 alpha/C 中位数 147）
- M0b：0.1959 ± 0.0280（内层 CV 选定 alpha/C 中位数 46.4）
- M1a：0.1402 ± 0.0218（内层 CV 选定 alpha/C 中位数 464）
- M1：0.1210 ± 0.0268（内层 CV 选定 alpha/C 中位数 681）
- M1full：0.1085 ± 0.0374（内层 CV 选定 alpha/C 中位数 464）
- M1b：0.1647 ± 0.0320（内层 CV 选定 alpha/C 中位数 215）
- delta_M1_vs_M0a：+0.0201，95% CI [-0.0817, +0.1097]
- delta_M1b_vs_M0b：-0.0312，95% CI [-0.1048, +0.0504]
- delta_M1a_vs_M0a：+0.0393，95% CI [-0.0553, +0.1363]
- 置换检验（折内 X 置换）：观测 Δ=+0.1068，p_perm=0.0010（零分布均值 -0.0224 ± 0.0432）
## Y1_year1_hurdle
- M0a：0.8438 ± 0.0050（内层 CV 选定 alpha/C 中位数 0.518）
- M0b：0.7737 ± 0.0025（内层 CV 选定 alpha/C 中位数 0.268）
- M1a：0.8334 ± 0.0061（内层 CV 选定 alpha/C 中位数 0.0373）
- M1：0.8321 ± 0.0052（内层 CV 选定 alpha/C 中位数 0.0373）
- M1full：0.8330 ± 0.0047（内层 CV 选定 alpha/C 中位数 0.0373）
- M1b：0.7540 ± 0.0052（内层 CV 选定 alpha/C 中位数 0.072）
- delta_M1_vs_M0a：-0.0116，95% CI [-0.0234, +0.0031]
- delta_M1b_vs_M0b：-0.0197，95% CI [-0.0346, -0.0019]
- delta_M1a_vs_M0a：-0.0104，95% CI [-0.0211, +0.0006]
- 置换检验（折内 X 置换）：观测 Δ=-0.0048，p_perm=0.2178（零分布均值 -0.0097 ± 0.0059）
## Y1_year1_logsnaps
- M0a：0.5794 ± 0.0042（内层 CV 选定 alpha/C 中位数 18.1）
- M0b：0.4987 ± 0.0046（内层 CV 选定 alpha/C 中位数 21.5）
- M1a：0.5733 ± 0.0098（内层 CV 选定 alpha/C 中位数 46.4）
- M1：0.5714 ± 0.0086（内层 CV 选定 alpha/C 中位数 46.4）
- M1full：0.5715 ± 0.0087（内层 CV 选定 alpha/C 中位数 46.4）
- M1b：0.4783 ± 0.0064（内层 CV 选定 alpha/C 中位数 46.4）
- delta_M1_vs_M0a：-0.0080，95% CI [-0.0292, +0.0216]
- delta_M1b_vs_M0b：-0.0204，95% CI [-0.0466, +0.0144]
- delta_M1a_vs_M0a：-0.0061，95% CI [-0.0287, +0.0230]
- 置换检验（折内 X 置换）：观测 Δ=+0.0063，p_perm=0.0160（零分布均值 -0.0240 ± 0.0142）
## 时序外推与 LOCO
{
"Y1_year1_hurdle_M0a": {"n_train": 337, "n_test": 173, "metric": 0.8756019261637239},
"Y1_year1_hurdle_M1": {"n_train": 337, "n_test": 173, "metric": 0.8424291064740503},
"Y1_year1_logsnaps_M0a": {"n_train": 337, "n_test": 173, "metric": 0.6935894135668179},
"Y1_year1_logsnaps_M1": {"n_train": 337, "n_test": 173, "metric": 0.6484356546055681},
"LOCO_Y2": {
"Y2_test_2023_M0a": {"n_test": 112, "r": 0.19713610636162895},
"Y2_test_2023_M1": {"n_test": 112, "r": 0.19772823760192193},
"Y2_test_2024_M0a": {"n_test": 104, "r": 0.09381314854689464},
"Y2_test_2024_M1": {"n_test": 104, "r": 0.12261727913949654},
"Y2_test_2025_M0a": {"n_test": 83, "r": 0.010633263245448521},
"Y2_test_2025_M1": {"n_test": 83, "r": 0.10448681137048718}
}
}
## 对抗验证（特征预测 draft_year，应 ≤0.55）
{
"draft_year_macro_auc": {"mean": 0.6975797446428074, "std": 0.031664830770500724}
}
## BH-FDR 单特征检验（Y2，顺位校正后，HC3）
- skill_s_peak_z_mean：β=+0.156，p=0.000316，q=0.006，n=299
- shadow_agility_pure：β=+0.111，p=0.0109，q=0.103，n=299
- skill_accel_p95_z_mean：β=+0.098，p=0.0359，q=0.227，n=299
- skill_cut_reaccel_mean_z_mean：β=+0.077，p=0.0509，q=0.242，n=294
- skill_decel_p95_z_mean：β=+0.078，p=0.101，q=0.343，n=299
- shadow_agility_full：β=+0.076，p=0.115，q=0.343，n=299
- skill_lat_accel_p95_z_mean：β=+0.072，p=0.13，q=0.343，n=299
- skill_brake_power_p95_z_mean：β=+0.067，p=0.175，q=0.343，n=299
- skill_braking_impulse_max_z_mean：β=+0.058，p=0.178，q=0.343，n=299
- skill_cut_retention_mean_z_mean：β=-0.064，p=0.181，q=0.343，n=294
- v_max_best：β=+0.063，p=0.22，q=0.380，n=239
- lower_body_burst：β=+0.060，p=0.27，q=0.427，n=265
- flying_30：β=-0.045，p=0.345，q=0.504，n=253
- tau_best：β=+0.048，p=0.381，q=0.516，n=239
- speed_maintenance_mean：β=-0.035，p=0.579，q=0.734，n=239
- a0_mean：β=+0.026，p=0.62，q=0.736，n=239
- speed_score：β=+0.006，p=0.909，q=1.000，n=253
- start_finish_ratio：β=+0.003，p=0.957，q=1.000，n=253
- skill_jerk_p95_z_mean：β=-0.000，p=1，q=1.000，n=299
## M3 LightGBM / M2 中介 / Y2 敏感性
{
"M3_lightgbm": {"Y2_r_mean": 0.14299010112315572, "Y1_hurdle_auc_mean": 0.8440415068662626},
"M2_mediation": {"n": 299, "a_path": 0.11526418508933539, "b_path": -0.04338748770428838, "indirect": -0.005001023413308359, "indirect_ci95": [-0.030765546200449006, 0.02073897709960343], "direct_c": 0.15544714340244187, "pct_mediated": -0.03324129205638254, "ktr_speed_corr_raw": 0.8063681738136682, "ktr_speed_partial_r": 0.2569004679872448, "ktr_speed_partial_p": 3.3414960637797814e-06, "ktr_speed_mean": 0.6383744094080624},
"Y2_sensitivity": {"reg_snaps_50": {"n": 299, "oof_r_mean": 0.13011510549660155}, "reg_snaps_150": {"n": 283, "oof_r_mean": 0.08780732823570991}}
}

---
来源：东京服务器 /home/wangde2021/03-nfl-big-data-bowl/logs/phase3_results.md（2026-11-21 16:38 写出，修复版全量），由浏览器任务回传全文后转录到本地。FDR Y1 stage2 补充（phase3_results.json，.md 未列出）：按 p 排序前 6 项 a0_mean β=0.0661、p=0.0412、q=0.699、n=181；speed_maintenance_mean β=0.0642、p=0.0991、q=0.699、n=181；skill_braking_impulse_max_z_mean β=0.0395、p=0.1104、q=0.699、n=236；skill_cut_retention_mean_z_mean β=-0.0293、p=0.2677、q=0.864、n=234；skill_accel_p95_z_mean β=0.0296、p=0.2929、q=0.864、n=236；skill_jerk_p95_z_mean β=0.0230、p=0.4033、q=0.864、n=236——Y1 stage2 的 FDR 检验无显著项（最小 q=0.699）。
