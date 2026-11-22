# Phase 3 全量结果 — 第 5 轮（服务器，进行中，未完成）

> **⚠️ 本文件为进行中快照，不是最终结果。** 服务器第 5 轮 model_phase3 仍在跑（PID 79903，2026-11-22 15:12:33 CST 启动）；截至 17:16 CST 只打印了 `[drill]` 与 `[Y1_hurdle_stage1]` 两个章节，Y2 数、对抗验证、定价、阶梯、LOCO、ktr 均未输出，结果文件尚未写入。收数后由后续任务用完整版替换本文件。数字来源：浏览器任务回传的日志尾部原文（已核对），非人工抄录。

## 运行状态（2026-11-22 17:16 CST 核查）

- 服务器：tokyo-node-work，e2-standard-2（2 核 8G），IP 35.213.17.8；目录 `/home/wangde2021/03-nfl-big-data-bowl`。
- 备份：`prefix_snapshot_r4/`（9 文件，逐文件 cmp 全部一致）。
- 覆盖文件：仅 `src/model_phase3.py`（md5 6d1c3f7dc117ce7574dfaa09a3955ebe，94,930 字节，1,819 行；服务器端再次核对一致）。
- 第 1 步 features_combine.py（--min-drill-players 20）：14:37:23 启动，约 1 分钟，退出码 0；stable drills 31；shadow_full_vs_short_shuttle oof_r_within 0.446、oof_r2_within 0.198。
- 第 2 步 targets.py：14:39:46 启动，约 1 分钟，退出码 0；y2 covered 299，covered_reg_50 335，covered_reg_150 285；分位置 DB 71、DL 73、OL 67、TE 22、WR 66。
- 第 3 步 game_kinematics.py：14:40:44 启动，约 15:10 完成（约 29 分钟），退出码 0；401 名球员，总帧数 10,424,763。
- 第 4 步 model_phase3.py --within-year：15:12:33 启动，`nohup .venv/bin/python -u src/model_phase3.py --within-year > logs/phase3c_model_r5.log 2>&1 &`，PID 79903；种子 42、10×5 CV、1000 置换、2000 自助均为默认参数；日志首行确认 `[drill] fold-clean drill normalization from skill_events.parquet (28 stable drills, 4238 player-drill rows); WITHIN-YEAR harmonization ON`。
- 17:14 核查（运行时长 2h01m）：进程存活，CPU 约 108%，累计 CPU 时间 132:22，内存 RSS 约 991 MB；日志 3,403 行，自约 16:41 后无新增输出（该阶段不打印进度，属正常）；无 traceback/error。17:16 再次核查：PID 79903 存活，累计 CPU 时间 134:01（继续推进），日志仍 3,403 行。
- **提速预估未兑现**：30–45 分钟预估在 2 核机器上未实现，首个结果章节约在 88 分钟时才打印；第 4 轮 model_phase3 实测 7h14m。

## 已打印的唯一结果章节：[Y1_hurdle_stage1]（约 16:41 打印）

> 注意：这是 Y1 hurdle 阶段 1 的数，不是 Y2 数；不得当作 Y2 结论引用。

- M0a metric_mean 0.8631263144871953（std 0.0033438416720065044，alpha_median 1.9306977288832496）
- M0b_stopwatch 0.5967563095385376（std 0.00840025832849946，alpha_median 0.02829145724599095）
- M0c_ngs 0.8081915749103056（std 0.00338868385795279725）
- M0all 0.8640124335024124（std 0.004669989709632406，alpha_median 0.13894954943731375）
- M1b 0.7872680316714091（std 0.006085212882010442）
- M1_track_only 0.5688373747371026（std 0.008051243270406945）
- M1a 0.8650949523691699；M1 0.8640696523567982；M1_full 0.8642954348632934；M1_over_stopwatch 0.5723833972534949；M1_over_all 0.8599545342075962
- delta_M1_vs_M0a 均值 0.000943378696028782，CI95 [-0.007913813880342447, 0.006458196169409511]
- delta_M1b_vs_M0b -0.014977421749350572，CI95 [-0.028053736863054245, -0.002340207004236925]
- delta_M1_over_all_vs_M0all -0.0040578992948162584，CI95 [-0.01057352579482993, 0.003215411735077962]
- delta_M1_over_stopwatch_vs_M0b_stopwatch -0.02437291228504279，CI95 [-0.064289700179331888, 0.012850984502013572]
- delta_M1_track_only_vs_M0b_stopwatch -0.027918934801435036，CI95 [-0.0763267858886784, 0.01498902643308904]
- permutation obs_delta 0.0015103715617139264，p_perm 0.059880239520958084，null_mean -0.0008983683671298931，null_sd 0.0016024676571825326

对照第 4 轮 Y1_hurdle_stage1（M0a 0.8631 / M1 0.8641）：本轮 stage1 与第 4 轮一致（Y1 零增量，双重分离的一半未变），符合预期；Y2 才是本轮裁决项，尚未输出。

## 尚未输出的章节与文件

- 未打印：`[temporal]`、`[adversarial]`、`[fdr_Y2]`、`[draft_pricing]`、`[robustness-ladder]`、`[robustness]`、`[M3]`、`[M2]`。
- 未写入（仍为第 4 轮 11:59 的时间戳）：`logs/phase3_results.json`、`logs/phase3_results.md`、`data/derived/oof_predictions.parquet`、`logs/experiments.csv`。

## 收数清单（后续任务）

待进程结束并确认日志退出码为 0 后回传：phase3_results.md 全文、combine_features_summary.json 的 shadow_calibration 全文、targets_summary.json 关键数、game_kinematics 帧数与人数、oof_predictions.parquet 行列数与 experiments.csv 新增行数、对抗验证分块 AUC、M0b/M0c/M0all/M1_over_* 的 Y2 数与增量 CI、选秀定价回归与稳健性阶梯（第 5 步 forty.notna() 子集）、Y1_year1_hurdle/logsnaps 与 LOCO_Y2（2023/2024/2025）、ktr_speed_by_position 的 WR 保持率与 Y2_sensitivity 的 n_plays_100/150，以及 model_phase3 全量实际耗时。
