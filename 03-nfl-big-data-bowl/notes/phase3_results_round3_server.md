# NFL Big Data Bowl 2027 — 第 3 轮服务器全量结果（2026-11-21/22）

来源：服务器 tokyo-node-work（e2-standard-2，IP 35.213.17.8）/home/wangde2021/03-nfl-big-data-bowl/logs/，经 Cloud Shell 回传。
四步：features_combine（18:57–19:00，PID 22170）→ targets（19:02–19:03，PID 22598）→ game_kinematics（19:06–19:39，PID 22624，32 分钟）→ model_phase3（19:42–11-22 00:18，PID 24410，约 4 小时 36 分）。种子 42，10×5 CV，1000 置换，2000 自助。日志无 Traceback。
备份：服务器现有结果已备份到 prefix_snapshot_r2/（9 个文件、1.2M，未覆盖原 prefix_snapshot/）。4 个改后文件已按 md5 覆盖进 src/（kinematics.py 8948、features_combine.py 30765、targets.py 18060、model_phase3.py 65110）。

## phase3_results.md 全文（556 行，20,321 字节，逐字回传）

# NFL Big Data Bowl 2027 — Phase 3 主建模结果

种子 42；10×5 折重复分层 CV；置换 1000 次；自助 2000 次。

特征分组：TRACK_PURE（四十曲线 + 9 项无质量技能指标，折内干净 drill 归一化）；STATIC（静态秒表派生）；SHADOW（纯技能影子指数）。
Round-2 改动：M0b 已拆为 M0b_stopwatch（纯秒表/双跳）与 M0c_ngs（NGS 评分）；M1 系列为两阶段残差/偏置堆叠（基线用自身最优惩罚）；置换检验基线与全模型各自用正确正则化、聚合 3 次重复。
Drill 归一化：fold-clean drill normalization from skill_events.parquet (28 stable drills, 4238 player-drill rows)

## Y1_hurdle_stage1
- M0a：0.8631 ± 0.0033（内层 CV 选定 alpha/C 中位数 1.93）
- M0b_stopwatch：0.5948 ± 0.0064（内层 CV 选定 alpha/C 中位数 0.0193）
- M0c_ngs：0.8041 ± 0.0016（内层 CV 选定 alpha/C 中位数 0.518）
- M0b：0.7976 ± 0.0035（内层 CV 选定 alpha/C 中位数 0.268）
- M0all：0.8626 ± 0.0046（内层 CV 选定 alpha/C 中位数 0.268）
- M1a：0.8631 ± 0.0054（内层 CV 选定 alpha/C 中位数 0.0193）
- M1：0.8629 ± 0.0049（内层 CV 选定 alpha/C 中位数 0.01）
- M1full：0.8628 ± 0.0057（内层 CV 选定 alpha/C 中位数 0.01）
- M1b：0.7931 ± 0.0060（内层 CV 选定 alpha/C 中位数 0.268）
- M1_over_stopwatch：0.8601 ± 0.0049（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all：0.8630 ± 0.0048（内层 CV 选定 alpha/C 中位数 0.01）
- delta_M1_vs_M0a：-0.0002，95% CI [-0.0096, +0.0053]
- delta_M1b_vs_M0b：-0.0045，95% CI [-0.0155, +0.0072]
- delta_M1a_vs_M0a：-0.0000，95% CI [-0.0090, +0.0064]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.2653，95% CI [+0.2121, +0.3141]
- delta_M1_over_all_vs_M0all：+0.0005，95% CI [-0.0041, +0.0037]
- 置换检验（折内 X 置换）：观测 Δ=+0.0014，p_perm=0.2020（零分布均值 -0.0002 ± 0.0019）

## Y1_stage2_logASR
- M0a：0.6611 ± 0.0078（内层 CV 选定 alpha/C 中位数 14.7）
- M0b_stopwatch：0.4932 ± 0.0115（内层 CV 选定 alpha/C 中位数 21.5）
- M0c_ngs：0.5977 ± 0.0132（内层 CV 选定 alpha/C 中位数 10）
- M0b：0.5978 ± 0.0126（内层 CV 选定 alpha/C 中位数 14.7）
- M0all：0.6602 ± 0.0084（内层 CV 选定 alpha/C 中位数 31.6）
- M1a：0.6608 ± 0.0075（内层 CV 选定 alpha/C 中位数 1e+05）
- M1：0.6611 ± 0.0077（内层 CV 选定 alpha/C 中位数 1e+05）
- M1full：0.6611 ± 0.0078（内层 CV 选定 alpha/C 中位数 1e+05）
- M1b：0.5935 ± 0.0139（内层 CV 选定 alpha/C 中位数 21.5）
- M1_over_stopwatch：0.6592 ± 0.0053（内层 CV 选定 alpha/C 中位数 1e+05）
- M1_over_all：0.6602 ± 0.0084（内层 CV 选定 alpha/C 中位数 1e+05）
- delta_M1_vs_M0a：-0.0000，95% CI [-0.0001, +0.0000]
- delta_M1b_vs_M0b：-0.0044，95% CI [-0.0240, +0.0182]
- delta_M1a_vs_M0a：-0.0003，95% CI [-0.0008, +0.0002]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.1660，95% CI [+0.0946, +0.2347]
- delta_M1_over_all_vs_M0all：+0.0000，95% CI [-0.0001, +0.0001]
- 置换检验（折内 X 置换）：观测 Δ=-0.0000，p_perm=0.5010（零分布均值 -0.0000 ± 0.0000）

## Y2_quality
- M0a：0.1009 ± 0.0275（内层 CV 选定 alpha/C 中位数 147）
- M0b_stopwatch：-0.0198 ± 0.0404（内层 CV 选定 alpha/C 中位数 1e+03）
- M0c_ngs：0.1926 ± 0.0204（内层 CV 选定 alpha/C 中位数 31.6）
- M0b：0.1959 ± 0.0280（内层 CV 选定 alpha/C 中位数 46.4）
- M0all：0.1747 ± 0.0305（内层 CV 选定 alpha/C 中位数 147）
- M1a：0.0953 ± 0.0253（内层 CV 选定 alpha/C 中位数 3.48e+03）
- M1：0.1000 ± 0.0263（内层 CV 选定 alpha/C 中位数 1.19e+04）
- M1full：0.0977 ± 0.0280（内层 CV 选定 alpha/C 中位数 1e+05）
- M1b：0.2020 ± 0.0192（内层 CV 选定 alpha/C 中位数 57.3）
- M1_over_stopwatch：0.0843 ± 0.0355（内层 CV 选定 alpha/C 中位数 1e+05）
- M1_over_all：0.1745 ± 0.0309（内层 CV 选定 alpha/C 中位数 1e+05）
- delta_M1_vs_M0a：-0.0009，95% CI [-0.0158, +0.0130]
- delta_M1b_vs_M0b：+0.0061，95% CI [-0.0326, +0.0543]
- delta_M1a_vs_M0a：-0.0056，95% CI [-0.0242, +0.0307]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.1040，95% CI [+0.0051, +0.2416]
- delta_M1_over_all_vs_M0all：-0.0001，95% CI [-0.0007, +0.0003]
- 置换检验（折内 X 置换）：观测 Δ=+0.0042，p_perm=0.0350（零分布均值 -0.0007 ± 0.0027）

## Y1_year1_hurdle
- M0a：0.8438 ± 0.0050（内层 CV 选定 alpha/C 中位数 0.518）
- M0b_stopwatch：0.5740 ± 0.0101（内层 CV 选定 alpha/C 中位数 0.01）
- M0c_ngs：0.7796 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.518）
- M0b：0.7737 ± 0.0025（内层 CV 选定 alpha/C 中位数 0.268）
- M0all：0.8476 ± 0.0046（内层 CV 选定 alpha/C 中位数 1）
- M1a：0.8455 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.0193）
- M1：0.8453 ± 0.0043（内层 CV 选定 alpha/C 中位数 0.0193）
- M1full：0.8458 ± 0.0040（内层 CV 选定 alpha/C 中位数 0.0283）
- M1b：0.7662 ± 0.0034（内层 CV 选定 alpha/C 中位数 0.139）
- M1_over_stopwatch：0.8444 ± 0.0026（内层 CV 选定 alpha/C 中位数 0.01）
- M1_over_all：0.8465 ± 0.0042（内层 CV 选定 alpha/C 中位数 0.0193）
- delta_M1_vs_M0a：+0.0015，95% CI [-0.0065, +0.0106]
- delta_M1b_vs_M0b：-0.0076，95% CI [-0.0202, +0.0074]
- delta_M1a_vs_M0a：+0.0017，95% CI [-0.0055, +0.0077]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.2704，95% CI [+0.2184, +0.3219]
- delta_M1_over_all_vs_M0all：-0.0011，95% CI [-0.0059, +0.0035]
- 置换检验（折内 X 置换）：观测 Δ=+0.0009，p_perm=0.1290（零分布均值 -0.0020 ± 0.0027）

## Y1_year1_logsnaps
- M0a：0.5794 ± 0.0042（内层 CV 选定 alpha/C 中位数 18.1）
- M0b_stopwatch：0.2150 ± 0.0173（内层 CV 选定 alpha/C 中位数 147）
- M0c_ngs：0.5073 ± 0.0061（内层 CV 选定 alpha/C 中位数 10）
- M0b：0.4987 ± 0.0046（内层 CV 选定 alpha/C 中位数 21.5）
- M0all：0.5818 ± 0.0028（内层 CV 选定 alpha/C 中位数 46.4）
- M1a：0.5779 ± 0.0086（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1：0.5776 ± 0.0055（内层 CV 选定 alpha/C 中位数 2.82e+03）
- M1full：0.5778 ± 0.0056（内层 CV 选定 alpha/C 中位数 2.15e+03）
- M1b：0.4925 ± 0.0078（内层 CV 选定 alpha/C 中位数 31.6）
- M1_over_stopwatch：0.5735 ± 0.0066（内层 CV 选定 alpha/C 中位数 4.55e+03）
- M1_over_all：0.5804 ± 0.0045（内层 CV 选定 alpha/C 中位数 4.55e+03）
- delta_M1_vs_M0a：-0.0018，95% CI [-0.0065, +0.0047]
- delta_M1b_vs_M0b：-0.0063，95% CI [-0.0262, +0.0194]
- delta_M1a_vs_M0a：-0.0015，95% CI [-0.0069, +0.0061]
- delta_M1_over_stopwatch_vs_M0b_stopwatch：+0.3585，95% CI [+0.2724, +0.4421]
- delta_M1_over_all_vs_M0all：-0.0014，95% CI [-0.0047, +0.0030]
- 置换检验（折内 X 置换）：观测 Δ=+0.0041，p_perm=0.0020（零分布均值 -0.0002 ± 0.0014）

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
    "metric": 0.8798822899946497
  },
  "Y1_year1_logsnaps_M0a": {
    "n_train": 253,
    "n_test": 125,
    "metric": 0.629337524994706
  },
  "Y1_year1_logsnaps_M1": {
    "n_train": 253,
    "n_test": 125,
    "metric": 0.6312471971196336
  },
  "LOCO_Y2": {
    "Y2_test_2023_M0a": {
      "n_test": 112,
      "r": 0.19713610636162895
    },
    "Y2_test_2023_M1": {
      "n_test": 112,
      "r": 0.18487654897939867
    },
    "Y2_test_2024_M0a": {
      "n_test": 104,
      "r": 0.09381314854689464
    },
    "Y2_test_2024_M1": {
      "n_test": 104,
      "r": 0.09400451360278887
    },
    "Y2_test_2025_M0a": {
      "n_test": 83,
      "r": 0.010633263245448521
    },
    "Y2_test_2025_M1": {
      "n_test": 83,
      "r": 0.014921770175360418
    }
  }
}

## 对抗验证（特征预测 draft_year，应 ≤0.55）
{
  "draft_year_macro_auc_ALL": {
    "mean": 0.6626781910112124,
    "std": 0.029678657834316276
  },
  "draft_year_macro_auc_FORTY_ONLY": {
    "mean": 0.6646179755461364,
    "std": 0.02473211503492279
  },
  "draft_year_macro_auc_SKILL_ONLY": {
    "mean": 0.47729938612654144,
    "std": 0.029753462790870842
  },
  "draft_year_macro_auc_STATIC_ONLY": {
    "mean": 0.5742010957651182,
    "std": 0.05337299538088789
  },
  "draft_year_macro_auc": {
    "mean": 0.6626781910112124,
    "std": 0.029678657834316276
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
- forty_s_peak：β=+0.225，p=3.48e-09，q=0.000，n=405
- v_max_best：β=+0.189，p=3.05e-06，q=0.000，n=405
- skill_s_peak_z_mean：β=+0.153，p=0.000254，q=0.001，n=510
- skill_accel_p95_z_mean：β=+0.140，p=0.000632，q=0.002，n=510
- skill_lat_accel_p95_z_mean：β=+0.094，p=0.00881，q=0.025，n=510
- a0_mean：β=+0.136，p=0.0106，q=0.025，n=405
- skill_cut_reaccel_mean_z_mean：β=+0.074，p=0.0605，q=0.113，n=501
- skill_brake_power_p95_z_mean：β=+0.080，p=0.0644，q=0.113，n=510
- skill_decel_p95_z_mean：β=+0.063，p=0.131，q=0.203，n=510
- speed_maintenance_mean：β=+0.051，p=0.256，q=0.358，n=405
- tau_best：β=-0.048，p=0.309，q=0.368，n=405
- skill_braking_impulse_max_z_mean：β=+0.041，p=0.315，q=0.368，n=510
- skill_cut_retention_mean_z_mean：β=+0.026，p=0.508，q=0.547，n=501
- skill_jerk_p95_z_mean：β=+0.022，p=0.599，q=0.599，n=510

## 稳健性：skill_s_peak_z_mean 控制 log_pick + forty + ngs_final_score
{
  "feature": "skill_s_peak_z_mean",
  "n": 252,
  "beta": 0.09262906226480887,
  "p": 0.08862695122888513
}

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
    "Y2_r_mean": 0.17542247287375926,
    "Y1_hurdle_auc_mean": 0.8498422615365582
  },
  "M2_mediation": {
    "n": 299,
    "a_path": 0.11526418508933539,
    "b_path": -0.04338748770428838,
    "indirect": -0.005001023413308359,
    "indirect_ci95": [
      -0.030498751423455133,
      0.02389012638096978
    ],
    "direct_c": 0.15544714340244187,
    "pct_mediated": -0.03324129205638254,
    "burst_1s": {
      "n": 299,
      "a_path": 0.08296041092246682,
      "b_path": 0.1412405435594157,
      "indirect": 0.011717373532601702,
      "indirect_ci95": [
        -0.01664058353350336,
        0.04740848893260768
      ],
      "direct_c": 0.13872874645653183,
      "pct_mediated": 0.07788418560377647
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
    }
  },
  "Y2_sensitivity": {
    "reg_snaps_50": {
      "n": 299,
      "oof_r_mean": 0.0801422589835076
    },
    "reg_snaps_150": {
      "n": 283,
      "oof_r_mean": 0.03673594000298598
    }
  }
}

## Shadow 标定（combine_features_summary.json 的 shadow_calibration 全文）

```json
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
```

## targets_summary.json 关键数

- players 510；reg_plays_used 240412；asr_mean 235.7895；y3_start_rate_mean 0.226797385620915
- y2 covered 299（reg_50 335，reg_150 285）；分位置 {'DB': 71, 'DL': 73, 'OL': 67, 'TE': 22, 'WR': 66}
- y2.eb_kappa 共 32 条，贴 200 上限 2 条：gated::DB::m_epa, gated::DL::m_quick_press
```json
{
  "players": 510,
  "reg_plays_used": 240412,
  "asr_vs_career_snaps_corr": 0.9655,
  "first_season_gap_counts": {
    "0": 378,
    "1": 20,
    "2": 3
  },
  "y1": {
    "zero_reg_snaps": 109,
    "hurdle_150_n": 236,
    "year1_hurdle_100_n": 254,
    "asr_median": 108.75,
    "asr_mean": 235.7895424836601
  },
  "y2": {
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
    "z_qual_mean": 1.1763165692349485e-15,
    "z_qual_std": 0.7648342134280289,
    "z_qual_raw_mean": -0.0032839922414159757,
    "z_qual_raw_std": 0.7636412720371446,
    "eb_kappa": {
      "gated::DB::m_epa": 200.0,
      "gated::DL::m_getoff": 8.568,
      "gated::OL::m_ppa": 39.766,
      "gated::OL::m_press_allowed_inv": 128.11,
      "gated::DL::m_press_rate": 142.316,
      "gated::OL::m_quick_collapse_inv": 161.912,
      "gated::DL::m_quick_press": 200.0,
      "gated::TE::m_sep": 180.574,
      "gated::WR::m_sep": 62.027,
      "gated::DB::m_tackle_rate": 159.359,
      "gated::TE::m_tprr": 91.452,
      "gated::WR::m_tprr": 66.728,
      "gated::TE::m_yacoe": 12.285,
      "gated::WR::m_yacoe": 29.311,
      "gated::TE::m_yprr": 166.675,
      "gated::WR::m_yprr": 111.539,
      "raw::DB::m_epa": 65.081,
      "raw::DL::m_getoff": 6.885,
      "raw::OL::m_ppa": 5.693,
      "raw::OL::m_press_allowed_inv": 7.248,
      "raw::DL::m_press_rate": 59.046,
      "raw::OL::m_quick_collapse_inv": 31.681,
      "raw::DL::m_quick_press": 130.829,
      "raw::TE::m_sep": 53.11,
      "raw::WR::m_sep": 47.922,
      "raw::DB::m_tackle_rate": 23.532,
      "raw::TE::m_tprr": 30.82,
      "raw::WR::m_tprr": 39.269,
      "raw::TE::m_yacoe": 11.262,
      "raw::WR::m_yacoe": 17.37,
      "raw::TE::m_yprr": 68.696,
      "raw::WR::m_yprr": 87.188
    }
  },
  "y3_start_rate_mean": 0.226797385620915
}
```

## game_kinematics / oof / experiments 确认

- game_kinematics：401 人、帧总数 10,424,763；2023 累计 122 人、2024 累计 260 人、2025 累计 401 人；剔除 no_snap 599、too_short 159。game_kinematics.parquet 401 行 × 8 列。
- oof_predictions.parquet：510 行 × 56 列，230K，11-22 00:18 更新。
- experiments.csv：由 7 行增至 10 行（表头 1 行＋数据 9 行），本轮新增 3 行（见下）。
```csv
exp_id,date,stage,feature_set,model_or_method,params_file,n_seeds,cv_scheme,metric,metric_mean,metric_std,metric_se,comparison,diff_mean,wins,notes
phase3_Y1_hurdle_stage1,2026-11-21,phase3,M0a+CORE_TRACK(16),M1,src/model_phase3.py,10,10x5 repeated stratified,AUC,0.8506,0.0035,0.0011,M0a,-0.0119,,perm p=0.6074
phase3_Y1_stage2_logASR,2026-11-21,phase3,M0a+CORE_TRACK(16),M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.3668,0.0162,0.0051,M0a,-0.0536,,perm p=0.7782
phase3_Y2_quality,2026-11-21,phase3,M0a+CORE_TRACK(16),M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.2684,0.0226,0.0071,M0a,-0.0076,,perm p=0.7303
phase3_Y1_hurdle_stage1,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,AUC,0.8556,0.0032,0.001,M0a,-0.0076,,perm p=0.5355
phase3_Y1_stage2_logASR,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.6315,0.007,0.0022,M0a,-0.0296,,perm p=0.4895
phase3_Y2_quality,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.121,0.0268,0.0085,M0a,0.0201,,perm p=0.0010
phase3_Y1_hurdle_stage1,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,AUC,0.8629,0.0049,0.0015,M0a,-0.0002,,perm p=0.2020
phase3_Y1_stage2_logASR,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.6611,0.0077,0.0024,M0a,-0.0,,perm p=0.5010
phase3_Y2_quality,2026-11-21,phase3,M0a+TRACK_PURE+STATIC+shadow_pure,M1,src/model_phase3.py,10,10x5 repeated stratified,r,0.1,0.0263,0.0083,M0a,-0.0009,,perm p=0.0350
```

## 服务器状态确认（浏览器任务结束时）

- tokyo-node-work：运行中，e2-standard-2（2 vCPU，8 GB），外部 IP 35.213.17.8，asia-northeast1-a，Ubuntu 24.04，100 GB 磁盘。全程未升配、未降配、未停机改配置、未重启。未碰 02-house-price，未提交 Kaggle，未外传原始 CSV。
- x-ui：x-ui.service Active (running)，自 2026-11-21 11:21 CST 运行至今；xray 内核也在运行。但网页面板端口本次未能确认可达（54321 未监听，8080/2053/2096/20201/20202 探测无响应，仅 443 有响应 400；日志无端口记录，sqlite3 未装，无法从数据库读端口）。服务在运行，面板可达性待核实。

## 需决策的发现（浏览器任务未擅自处理）

- 对抗验证 ALL AUC 0.6627，超过 Round-2 备用阈值 0.56（model_phase3.py 帮助：“若对抗验证 AUC > 0.56，使用 --within-year”）。本轮按任务要求以默认参数运行，未加 --within-year；是否另跑一轮 --within-year 变体（约 4.5 小时），留待 Gemini 第 3 轮裁决。
