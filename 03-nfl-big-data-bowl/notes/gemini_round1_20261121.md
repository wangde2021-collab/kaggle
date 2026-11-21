# Gemini 第 1 轮核对 — 回复全文逐字转录

- 对话：https://gemini.google.com/app/8adf0ca845f706e2
- 模型：Neon（最高档）、思考等级“高”；账号 wangde2021@gmail.com（Google AI Pro）
- 时间：2026-11-21 09:27–09:44
- 输入：7 个 src 全文 + data_profile.md + eda_audit.md + gemini_scheme.md（10 个附件全部上传成功）+ 评审请求原文（全量 Phase 1–2 结果 + Shadow 失败 + 已知局限 + Phase 3 代码预审）。
- 转录说明：以下为浏览器回传的逐字转录。仅有一处页面渲染缺陷无法恢复原文：P2 行“具体改动位置与代码改法”单元格在“例如窗口内累计转向角（”之后只显示了一个孤立的公式片段“Δθ”，句子后半部分与该行“预期 OOF r 提升”单元格在页面上均未渲染出来，下表 P2 行该两处按页面实际显示如实留空/截断，未做推测补写。另有几处公式符号在页面提取中被拆散（如 r_pooled 公式、κ_p ≈ 10^5），已按页面显示的符号原样转录。

---

通读全部代码、数据画像与审计报告后，给出首席评审逐项技术审查意见。当前流水线在工程骨架上很清晰，但存在多处会直接扭曲结果甚至导致指标失效的实质性代码 Bug 与统计学错误——好消息是，Shadow Agility 失败（r=0.105/0.179）大概率是目标量纲错配 + 欠正则导致的“假失败”，修复后有很大概率跨过 r ≥ 0.40 门限；坏消息是 Phase 3 的 targets.py、game_kinematics.py、model_phase3.py 中也有几处致命 Bug，全量跑完后必须重跑，切勿直接引用当前这一版数字。

## (1) Shadow Agility 为什么这么弱？逐项诊断与修复方案

### 逐项诊断

**目标定义与特征量纲的“致命错配”（第一大主因，贡献了绝大部分失败）**

在 features_combine.py:calibrate_shadow_agility 中，目标变量是跨 5 个位置混合的原始负秒数 sub["y"] = -sub[col]（three_cone / short_shuttle）。跨位置原始秒数中，约65%–75% 的总方差来自位置与体型差异（WR/DB 约 6.8–7.1s，OL 约 7.6–8.2s）。然而，输入的全部 10 个特征 skill_*_z_mean 都在位置专属的 drill_name（SKILL_DRILLS_WR/DB/DL/OL/TE）内部做了 Z-score。这意味着每个位置的 X 均值都已被强行归零，位置间信息被 100% 抹除。

数学后果：记 Y = μ_pos + ε_within。由于 E[X | pos] = 0，特征与 μ_pos 的协方差恒为 0。即便 X 在位置内达到 r_within = 0.50，在混合原始 Y 上的理论相关性也被直接打折到 r_pooled ≈ r_within · Var(ε)/Var(Y) ≈ 0.25；同时巨大的位置间残差 μ_pos 沦为回归误差项，直接淹没了 178 个样本上的梯度信号。

**跨 drill 聚合口径严重稀释有效信号，且混入反号特征**

drill_normalize 将全部48 个 drill_name 等权混在一起取均值（包括大量直线接球、OL 推挡、静态站姿起手等几乎不含变向的 drill），把真正含高强度变向/折返的 10–15 个 drill 稀释殆尽。聚合在 event_id（单次 attempt）层面直接 groupby("nfl_id").mean()，未先按 (nfl_id, drill_name) 聚合，导致重复次数多的非敏捷 drill 权重成倍放大。calibrate_shadow_agility 只用了 _mean 没用 _top2，且把 peak_mech_power 和 high_speed_brake_power 也塞了进去——这两项都乘了体重 m，在位置内部（如 DL 中 250 lb EDGE vs 330 lb NT）体重越大越慢、3-cone 越差，但体重加权功率反而越高，与目标方向相反。

**cut_events 定义与未平滑 dir 造成“抓噪漏真”**

kinematics.py:compute_kinematics 只对 s 做了 Savitzky-Golay 平滑，dir 完全未平滑，直接做一阶有限差分 wrap_deg(np.diff(direction)) / np.diff(t)。在 10 Hz（Δt=0.1s）下，光学传感仅仅抖动9°就会产生 90°/s 的角速度假峰，污染 lat_accel_p95 并触发大量伪 cut_events。cut_events 的速度门限 s > 2.5 yd/s（OL/DL 1.8）恰好过滤掉了最像 3-cone / short shuttle 的急停折返顶点：真实 90°/180° 变向在转折最低点时瞬时速度常跌至 0.5–2.0 yd/s，结果门限只抓到了中高速缓弯和噪声毛刺。retention = s[apex] / s[before]（取顶点前 0.3s）未限定减速段，若球员加速入弯会导致 retention > 1.0 甚至出现极大异常值。

**Ridge(alpha=10) 在高噪声低样本下欠正则**

5 折 CV 每折训练样本仅约 142–159 人，10 个高度共线且噪声大的特征配固定 alpha=10.0（特征矩阵特征值量级约 140，惩罚仅占 ~6%）明显欠正则，导致在极弱信号下方差大于偏差，出现 OOF R² = -0.030 < 0。

**样本量局限（n=178/199）**

均摊到 5 个位置每组仅 25–45 人。在小样本下必须尽量压缩无关特征维度、用位置内标准化的统一量纲借力（跨位置池化），绝不能让 10 个噪声特征在 140 行数据上自由拟合。

### 按性价比排序的 5 条可执行修复方案

| 优先级 | 修复方案 | 具体改动位置与代码改法 | 预期 OOF r 提升 |
|---|---|---|---|
| P0（先做诊断，5 分钟） | 黄金上限检验：直接在 THREE_CONE_DRILL（15,951 帧）和 SHORT_SHUTTLE（11,690 帧）自身轨迹上跑 skill_drill_features 预测位置内秒数 | 在 features_combine.py 临时加一段对 drill_type.isin(["THREE_CONE_DRILL", "SHORT_SHUTTLE"]) 提取同套运动学特征，看能否预测真实位置内秒数 | 立刻定位瓶颈：若自身轨迹 r < 0.55 说明运动学算子坏了；若 r > 0.65 说明瓶颈全在跨 drill 聚合 |
| P1（最高性价比） | 对齐目标量纲 + 折内嵌套正则 + 去掉体重反号特征 | features_combine.py:calibrate_shadow_agility：① 将目标改为位置内 Z-score 化的 -three_cone_pos_z 与 -short_shuttle_pos_z（在 CV 折内拟合位置均值/标准差！）；② 从 feat_cols 移除乘体重的 peak_mech_power / high_speed_brake_power，同时加入 _top2；③ 将 Ridge(alpha=10.0) 改为折内 RidgeCV(alphas=np.logspace(0, 4, 25))（并同步测试 1–2 分量 PLSRegression） | 仅修正位置量纲错配与欠正则，预计位置内 OOF r 从 0.10/0.18 跃升至 0.25–0.35，OOF R² 转正 |
| P2（高性价比） | 修复 dir 平滑与 cut_events 物理定义 | ① kinematics.py:compute_kinematics：对 np.unwrap(np.deg2rad(direction)) 施加 savgol_filter(window=5, poly=2) 后再求导得 omega，彻底消除 10 Hz 角度抖动；② cut_events：改用变向弧段定义——先识别连续制动-转向-再加速窗口（例如窗口内累计转向角（Δθ……[见文末转录说明] | [页面渲染中该单元格未显示，见文末转录说明] |
| P3（中高性价比） | 筛选高变向密度（COD-Heavy）Drill + 两级聚合 | features_combine.py:drill_normalize：① 仅保留三年均 ≥ 20 人的 31 个稳定 drill；② 按无监督物理准则筛选高变向 drill（例如 drill 中位 lat_accel_p95 或中位转向角位于该位置前 50% 的 drill，剔除直线接球/静止推挡）；③ 先按 (nfl_id, drill_name) 取均值/最佳，再跨 drill 取均值和 top2，消除 attempt 次数不均偏差 | 预计位置内 OOF r 再提升 +0.04 ~ +0.08 |
| P4（按叙事选择） | 构建“多模态 Shadow Agility”（Skill Drills + 40 码 + 纵跳/立定跳远） | three_cone 与 short_shuttle 的物理构成约 40% 是直线加速爆发 + 60% 是制动变向，而 40 码和双跳缺失率仅 14%–17%。在 calibrate_shadow_agility 中分两版汇报：①Pure Skill-Drill Shadow（仅专项追踪）；②Full Pre-Opt-Out Shadow（专项追踪 + 40 码 tau_best / v_max_best + lower_body_burst，均位置内 z 化） | Full 版位置内 OOF r 有望达到 0.45–0.60（远超 0.40 门限），并可通过增量 ΔR² 清晰拆分“专项变向追踪”在 40 码/双跳之上的净贡献 |

## (2) Phase 1 / Phase 2 代码核对（泄漏、划分、特征、Bug）

### A. 已知局限的定量影响评估：Phase 2 全样本 drill_normalize 泄漏

性质：属于无监督协变量分布泄漏（Transductive Leakage），不接触目标变量 Y，不是直接标签泄漏。

对 10×5 随机重复 CV 的影响：极小（对 r/AUC 影响通常 < 0.002），因为每个 drill 样本量 ≥ 20，单个验证折仅改变组均值约 0.2/n。

对 2025 时序外推与 LOCO 的影响：中等偏高，且存在实质隐患。目前 drill_normalize 按全样本 ≥ 20 人保留了48 个 drill，而 G4 审计明确显示“三年均 ≥ 20 人”只有31 个、“三年全覆盖”只有39 个。这意味着混入了9 个缺失部分年份的孤立 drill和17 个跨年人数不稳的 drill！跨年合并归一化既把 2025 年分布泄漏进了 2023+2024 训练集，又会让单年 drill 成为 draft_year 的隐蔽代理特征。

对 15 页报告评审可信度的影响：高。方案 §5 与 §6 第 7 条已白纸黑字承诺“drill 内 Z-score 100% 封装在 CV Pipeline 内拟合”，且你们已经落盘了 skill_events.parquet。只需约 30 行代码把它封装进自定义 DrillZScoreTransformer，顺手砍掉那 17 个非三年稳定 drill，即可彻底堵死漏洞，强烈建议修掉，不要留作“已知局限”。

### B. 逐文件问题清单

**audit_phase1.py:y1_preview（与 targets.py:build_y1 相同 Bug）**

错在哪里：y1 = y1[(y1["game_id"] // 1_000_000) == y1["draft_year"]] 用 game_id 前 4 位日历年份匹配选秀年。但 NFL 常规赛 Week 18（有时含 Week 17）在次年 1 月进行——data_profile.md 已给出铁证：game_tracking_2023.csv 第 0 行就是 game_id = 2024010707（2023 季 Week 18）！这导致三届新秀的 1 月份常规赛档数全部被错误丢弃，新秀季档数系统性少算约 6%–12%。

怎么改：从 games.parquet 读取 ["game_id", "season"]，用 season == draft_year 匹配。

**kinematics.py:compute_kinematics**

错在哪里：① dir 未解缠平滑直接差分（见上文）；② _clean_event 剔除坏帧后轨迹存在 Δt = 0.2s 甚至更大的时间空洞，直接套用假设等间隔采样的 savgol_filter 会引起滤波畸变；③ 若轨迹存在重复时间戳（Δt = 0），np.gradient(s, t) 和 steps / np.diff(t) 会除零产出 inf / NaN 并污染 jerk。

怎么改：先按 time 去重并检查 np.diff(t) > 0；若缺帧（dt > 0.15）先对连续分段线性插值到统一 0.1s 网格（或对孤立坏帧用线性插值替代直接删除），再对 s 和 np.unwrap(np.deg2rad(dir)) 做 SavGol 平滑。

**features_combine.py:forty_features**

错在哪里：①潜在数组越界崩溃 Bug：dist = np.cumsum(sub["dis"].to_numpy()) 在截断 sub = sub.iloc[: end + 1] 之前计算，随后却用未截断的 dist 去索引截断后的 s_end：tail = s_end[dist >= 30.0]！只要存在 end + 1 < len(dist) 就会直接抛 IndexError（目前没崩大概率因为多数尝试在 40 码附近截断或累计 dis 略小于 40.0）；② split_time 用 np.searchsorted 不做线性插值，0.1s 帧间隔给 5 码分段（~1.0s）引入高达 ±10% 的阶梯量化误差；③ 参数 weight_kg 传入后未使用。

怎么改：截断后同步切片 dist = dist[:len(sub)]；split_time(d0, d1) 改用 np.interp([d0, d1], dist, t)（先确保 dist[-1] >= d1）；若 dist[-1] < 38.0 标记为不完整尝试。

**features_combine.py:skill_drill_features**

错在哪里：① high_speed_brake_power 算的是 m * |min(a_t[hi], 0)|，单位是力（N）而非功率（W），少乘了速度 s[hi]，且按上文分析，用于敏捷表征时建议同时保留不含体重的纯运动学指标；② braking_impulse_max 硬编码帧间隔 0.1，应使用实际 dt；③ jerk_p95 对整段轨迹取 p95，与方案 §3b 定义的“启动前 1.0s 急动度”不符。

怎么改：区分纯运动学制动力/功率与体重加权版；用 dt 积分 braking_impulse_max；启动急动度限定在 t_sec <= 1.0。

**features_combine.py:static_features & drill_normalize & calibrate_shadow_agility**

错在哪里：① static_features 中的 lower_body_burst 在全 510 人上做了位置内 z-score（轻微无监督泄漏）；② drill_normalize 用全样本 min_players=20（48 个 drill）而非“三年均 ≥ 20 人”（31 个 drill），违反防年份泄漏原则；③ calibrate_shadow_agility 拟合出的 shadow_agility 是 skill_*_z_mean 的线性组合，写入 combine_features.parquet 后又与原特征一起放进 CORE_TRACK，造成完全共线。

怎么改：lower_body_burst 与 drill_normalize 统一移入 Phase 3 CV Pipeline 折内拟合；仅保留 31 个三年稳定 drill；shadow_agility 与明细特征分开做对照实验，不并列塞进同一线性回归。

**etl_parquet.py:etl_game_tracking**

错在哪里（隐患）：① ETL 阶段直接丢弃了 POST 季后赛，若 Phase 3 敏感性分析需跑方案承诺的 REG+POST 对比，将缺少 POST 追踪数据；② 分块写 Parquet 时未显式指定 event 列为 string 类型，若某百万行块全空会引发 schema 冲突。

怎么改：显式传 dtype={"event": "string"}；REG+POST 敏感性若仅针对目标 Y1/Y2（player_play.parquet 已保留全部行），在文档中注明即可。

## (3) Phase 3 代码预审（硬 Bug、统计错误与出数复核清单）

当前服务器上正在跑的 Phase 3 有6 处高危 Bug / 统计错误，这批结果出来后只能当诊断草稿，必须修完重跑。

### 逐模块预审

**Y1 / Y2 / Y3 目标定义（targets.py）**

Bug 1（Y1 新秀季截断）：build_y1 第 106 行 (game_id // 1_000_000) == draft_year 丢掉了每年 1 月的常规赛（同上，必须改用 games["season"]）。

Bug 2（DB 覆盖档漏滤 "NO_ASSIGNMENT"）：build_y2 第 169 行只过滤 db["coverage_assignment"].notna()，但 data_profile.md 第 2 样例行明晃晃写着 coverage_assignment = NO_ASSIGNMENT！大量非覆盖档混入了 DB 分母与 EPA 计算。必须加 & (db["coverage_assignment"] != "NO_ASSIGNMENT")。

Bug 3（_residualize 缺失回填污染）：第 59 行 out.loc[sub.index[~ok]] = sub.loc[~ok, y_col] 在协变量（如 cushion）缺失时，把原始正值（例如分离度 3.1 码）直接塞进了零均值残差列！这会给缺 cushion 的档位凭空制造 +3 码的虚假超级分离度。必须改为保留 NaN 或减去位置均值 y - mean(y)。

Bug 4（Y2 门槛与标准化顺序错误）：① 覆盖门槛 y2_covered_75 用的是总档数 reg_snaps >= 75，而非位置专属有效档 n_plays >= 75（一个打 100 档但只跑 12 条路线的阻挡 TE 会混进来）；② y2 在门控之前对全体球员（包含只打 1–5 档的极值离群点）做 m_* 位置内 z-score，极端小样本会严重拉大标准差、压扁真实球员的方差。

Bug 5（object 布尔列隐患 & 缺失方案指标）：player_play.csv 中 pressure_allowed、quick_pressure、unblocked_pressure 为 object 类型（混含 True/False/NaN），直接 == 1 或 .mean() 脆弱，必须先显式映射为 float（0/1/NaN）。另外 OL 漏了 pass_rushers_encountered >= 1；DB 用全队 expected_points_added 噪声极大且未按方案补入 game_tracking 的 Closing Burst。

次要核对：Y3 start_rate 计算正确（建议加 assert (t["start_rate"] <= 1.05).all()）；5 名跨组边缘球员（RB/ILB/MLB 归入 combine_position=DB）仍在 Y1 建模中，需与论文口径保持一致。

**EB 收缩（targets.py:_eb_shrink）—— 致命数学错误**

错在哪里：z_qual_raw 已经在每个 pos_group 内做过 Z-score，所以 5 个位置组的均值全约等于0。_eb_shrink 却拿“5 个位置均值的方差”当 between（≈ 0），拿“位置内球员方差”当 within（≈ 0.6），算出 κ_p = within / between ≈ 10^5，随后被 np.clip(kappa, 5.0, 200.0) 把全部 5 个位置强行锁死在 kappa = 200.0！

危害：这既不是正确的经验贝叶斯公式（正确 κ_p = σ²单档组内噪声 / τ²同位置球员间真实方差），又会导致严重后果——kappa=200 过大，把 2025 届仅打 100 档的新秀压成 0.33 * z，而 2023 届打 1500 档的老兵保留 0.88 * z，把“单档质量 Y2”严重污染成了“出场档数/选秀年份代理”！

怎么改：在单档明细表上，对每个位置、每项质量指标直接用一维随机效应方差分解（Method of Moments / ANOVA）估计真实 σ²_w,p 与 τ²_b,p，算出每个指标独立的 κ_p,m = σ²_w,p / τ²_b,p 先收缩再合成，并在 targets_summary.json 打印各位置实际 κ 值。

**常规赛实战运动学（game_kinematics.py）—— 致命业务逻辑 Bug**

错在哪里：① END_EVENTS 包含了 "handoff" 和 "run"！NFL 几乎所有跑球档在 ball_snap 后 0.6–0.8 秒就会触发 handoff（仅 2025 季就有 46,564 次），导致所有跑球档刚开球就被砍断且因 <10 帧被丢弃！必须立刻从 END_EVENTS 删除 "handoff" 和 "run"；② 缺失 ball_snap 时 _trim_window 返回未裁剪的整段赛前站桩轨迹，污染 snap_burst_1s；③ 未关联 player_play 过滤 play_nullified_by_penalty == 'N'；④ 沿用了未平滑 dir 的 compute_kinematics，污染 game_lat_accel_p95。

**位置内 Z + 中位数插补 + 主模型（model_phase3.py）—— 5 处硬伤**

Bug 1（跨位置中位数插补灾难）：PositionZScore.fit 先用全样本全局中位数填补缺失值，再按位置算 Z-score！一名跳过 40 码的 OL（真实均值 ~5.15s）会被填上全局中位数4.62s，在 OL 组内瞬间变成-3.5 SD 的历史级怪兽；一名跳过 40 码的顶级 WR 反而被填慢！必须改成按 pos_group 分别计算训练折中位数插补。

Bug 2（丢弃 pos_group 固定效应，错位对比 M1 vs M0b）：① PositionZScore 丢弃了 pos_group 且把 log_pick/is_udfa/draft_year 也做了位置内标准化，导致 M0a 在 Y1 上丢失了位置间天然的出场档数差异（OL 首发打满 100% 档 vs 轮换 DL/TE），人为拉低了 M0a；② delta_M1_vs_M0b 用含选秀顺位的 M1 减不含顺位的 M0b，测的是“选秀顺位比秒表强”，而真正算了的 M1b 却没做对比！应补充 M1b vs M0b（纯选前追踪增量）和 (M0a+M0b+TRACK) vs (M0a+M0b)（同时控制顺位和 NGS+秒表后的净追踪增量）；③ CORE_TRACK 混入了 4 个静态秒表/跳远派生列（speed_score、flying_30、start_finish_ratio、lower_body_burst），应将其剥离到 M0b 或单独组，确保 TRACK 纯净。

Bug 3（19 特征固定 Ridge(alpha=10) 欠正则 + 回归未分层）：对 N=236–320、19 个共线特征，alpha=10 过小，极易造成 M1 折外反而低于 M0a；必须改为折内 RidgeCV / LogisticRegressionCV。同时回归 CV 应按 pos_group 分层，且删掉无用的 rs 变量。

Bug 4（bootstrap_ci 与点估计脱节）：out_d[label]["oof_mean"] 实际存的是 oofs[0]（仅第 1 次重复），却用来配 10 次重复的 metric_mean，可能导致点估计落在 95% CI 之外。应统一对 10 次重复均值 OOF 做配对分层自助。

Bug 5（oof_predictions.parquet 严重错位 Bug）：第 455–458 行把子集长度（如 236 行、320 行）的 oofs[0] 按默认下标 0..319 拼到了 510 行的 df["nfl_id"] 上，所有非全样本目标的 OOF 预测值全部张冠李戴到了错误球员身上！必须按 sub["nfl_id"] reindex 回 df["nfl_id"]。

**置换检验（model_phase3.py:permutation_p）—— 统计原假设错误**

错在哪里：全局打乱 y 同时破坏了 M0a -> y 和 TRACK -> y，此时零分布里的基线 AUC ≈ 0.50（而非真实的 ~0.80），测的不是“在控制 M0a 条件下，TRACK 增量等于零”的零假设。至于此前层内打乱出现 +0.38 异常，正是因为 PositionZScore 丢弃了 pos_group 截距、且全局中位数插补让 TRACK 反而把 pos_group 差异偷偷泄了出来。

怎么改：保持 (M0a, pos_group, y) 完整不动，只置换 CORE_TRACK 特征矩阵的行（回归用 Freedman-Lane 残差置换，或在 pos_group × draft_round_bin 层内整行置换 X_track）。这样每轮置换中 M0a 的真实预测力 100% 保留，严格检验 X_track 的条件增量。

**时序外推与 LOCO（model_phase3.py）**

错在哪里：① M0A_COLS 包含数值型 draft_year，在 2023+2024 -> 2025 外推时线性外推年份斜率毫无物理意义（对已对齐暴露时长的 Year-1 目标应移除 draft_year）；② Y1_year1_logsnaps 在 CV 中过滤了 year1_reg_snaps > 0，在 2025 外推中却未过滤，口径不一致；③ LOCO_Y2 只跑了 M1 没跑 M0a，无法判断增量方向是否跨三年一致。

**中介分析（M2_mediation）与 BH-FDR（single_feature_tests）**

M2 错在哪里：① a_path 和 b_path 回归未控制 pos_group，而 game_decel_p95 是跨位置原始值，严重受 WR vs OL 混杂；② 中介起点用的是已失效的 shadow_agility；③ ktr_speed_corr 是跨位置混合相关（天然虚高至 0.7+），必须补充控制 pos_group 后的位置内偏相关。

BH-FDR 错在哪里：single_feature_tests 未对 v_max_best / tau_best 等原始量纲特征做位置内 Z-score，导致各特征 beta 单位完全不同，无法直接画森林图；draft_year 当连续变量处理；建议改用 C(draft_year) + cov_type="HC3" 稳健标准误。

**对抗验证（adversarial_validation）**

设计基本正确，但一旦跑出 AUC > 0.55，第一怀疑对象就是 Phase 2 混入的那 17 个非三年稳定 drill。

### 当前全量结果出来后，必须第一时间复核的 8 个红灯数字

1. targets_summary.json 中的 y2.z_qual_std 与 by_position_75：若 z_qual_std 异常偏小（如 <0.45），就是 kappa=200 过度收缩的直接症状；同时检查 OL m_press_allowed_inv 和 DL m_quick_press 是否有全 0 或大面积 NaN（验证 object 布尔列解析是否翻车）。
2. game_kinematics.parquet 的 game_frames 总量：按方案预估 REG 有效帧应在 950–1,100 万帧；由于 handoff 误入 END_EVENTS，本次跑出的帧数大概率异常偏低（可能掉到 500–700 万帧）。
3. delta_M1_vs_M0a 是否为负：若为负，不要急着判定“追踪特征无用”，先排查是否由 Ridge(alpha=10) 欠正则 + OL 40 码全局中位数插补错位导致。
4. delta_M1_vs_M0b 虚高假象：切勿采信当前 M1 - M0b，重点改看 M1b - M0b。
5. adversarial_validation.draft_year_macro_auc：若 > 0.55，立即核查 48 个 drill 中单年/两年 drill 的贡献并裁减到 31 个。
6. permutation.null_mean：若绝对值偏离 0 较大（如 <-0.02 或 >+0.02），说明当前全局置换与固定 alpha=10 存在严重偏差，结果不可用于论文。
7. fdr_Y2 / fdr_Y1_stage2 中排名靠前的特征：重点核对显著的是纯 10 Hz 追踪特征（如 tau_best、skill_decel_p95）还是混进去的静态秒表列（speed_score、lower_body_burst），以及位置内单特征方向是否跨 3 届一致。
8. M2_mediation.pct_mediated：若出现负数或 >100%，说明存在符号相反的压抑效应或 shadow_agility 噪声崩塌，绝不能直接引用。

## (4) Shadow 标定失败后，研究叙事应如何调整？

明确建议：不要仓促放弃原方向，也绝不要把 Shadow 改成“直接监督拟合 NFL 表现”的弱监督黑箱；先用 2–3 小时跑完 P0–P3 修复，再按实测阈值走“两条清晰路径之一”。

### 为什么绝不能改成“直接监督拟合 NFL 表现的 Shadow Index”？

总样本仅 N=510（Y2 仅 ~320 人）。如果用技能训练特征监督拟合 NFL 表现（Y1/Y2）合成一个“Shadow Index”，再用它去讲“预测 NFL 表现”或做中介分析，在 NFL 数据科学评审眼里属于典型的目标泄漏 / 循环论证（Double Dipping），且失去了“用无标签/选前体测指标解释敏捷度”的独立构念效度（Construct Validity），在研究质量（40 分）上会直接扣重分。

### 两步走决策树（可执行、零赌博）

第一步（必做，半天内完成）：跑完 P0（3-cone/shuttle 自身轨迹上限检验）+ P1（位置内 Z 化目标 + 折内 RidgeCV）+ P2（dir 平滑 + 改 cut_events）+ P3（筛选高变向稳定 drill）。原因：目前 r = 0.105/0.179 几乎肯定是被“去位置均值 X 预测跨位置原始秒数 Y”的 Bug 误杀的，真实位置内相关性尚未测出。

第二步：按修复后的实测位置内 OOF r 拍板叙事

路径 A（若 Pure Skill-Drill 位置内 r ≥ 0.35，或加入 40 码/双跳的 Full 版 r ≥ 0.50 且 Skill-Drill 增量显著）：保留 Shadow Agility 主线，但必须同步修正 Opt-Out 叙事：Phase 1 审计已经证伪了“顶级新秀扎堆跳测 MNAR”（Mann-Whitney p = 0.401，1–7 轮及 UDFA 跳测率全部 >50%）。不要再硬写“顶级新秀策略性躲测试”，而是把叙事升级为更真实、更打动球探的事实——“全联盟结构性敏捷数据缺口（>60% 跨轮次普遍缺失），并非只影响首轮，而是系统性盲区；通过全覆盖的 10 Hz 专项训练重建 Shadow Agility，可为全部 510 名新秀补齐统一标尺。”（gemini_scheme.md Fig 1 的结论文字需同步改为展示“普遍高缺失 + 专项追踪 100% 覆盖”）。

路径 B（若修完所有 Bug 后，Pure Skill-Drill 对 3-cone 的位置内 r 仍 < 0.30）：果断升级为更高级的“构念分离（Construct Divergence）”叙事——这反而更契合“Uncover Non-Obvious Linkages”！

核心论点改为：

a. 为什么不用预测 3-cone：脚本化绕桩测试（3-cone / shuttle，封闭环境固定步点）与场上专项训练（开放式变向、接球/冲传/覆盖中的动态减速与出弯再加速）测的是两种不同的运动能力——用修复后的低相关 r 作为实证证据，而不是当作失败掩盖。

b. 替代指标怎么建（零泄漏、强物理可解释）：放弃用 three_cone 去回归加权，改用纯无监督/领域先验的 3 个正交生物力学维度直接入模：①冲刺加速动力学（40 码 Furusawa-Hill 的 τ, v_max，目前拟合 R² = 0.990 极其硬核）；②动态制动能力（专项训练无体重加权 decel_p95 + braking_impulse）；③开放式变向效率（修复后的 lat_accel_p95 + 急停出弯再加速）。

c. 完整落脚点：主线彻底聚焦到"From Stopwatch to Game Kinematics: Deceleration Kinetics, Pads Translation (KTR), and Draft-Capital Mispricing"——如果修复 Phase 3 后，这些无监督专项制动/变向指标不怎么相关于 3-cone，却能显著迁移到常规赛 game_kinematics（位置内 KTR）或在控制选秀顺位后贡献 Alpha，评审会认为这是一个比"拟合 3-cone"深刻得多的橄榄球生物力学发现；即便最终仅 40 码动力学（τ vs v_max）与穿垫迁移率（KTR）显著、专项训练为空结果，也能坦诚立住"哪些体测信号真能穿垫迁移、哪些不能"的高质量研究边界结论。

建议你先让服务器上的当前任务跑完落盘作为“修复前对照快照”，随后立即按P0 → 修复 targets.py / game_kinematics.py / kinematics.py / features_combine.py → 修复 model_phase3.py 的顺序重跑一版干净全量实验，拿到真实数字后再最终锁定路径 A 还是路径 B。需要的话，我可以直接按这个清单先给你出关键修复函数的完整补丁代码。

—— 全文转录结束 ——
