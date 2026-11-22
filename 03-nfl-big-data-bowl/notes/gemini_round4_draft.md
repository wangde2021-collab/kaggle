# Gemini 第 4 轮提纲

对话：继续第 1/2/3 轮同一对话 https://gemini.google.com/app/8adf0ca845f706e2 ，同一账号 wangde2021@gmail.com（Google AI Pro），同一模型 Neon、思考等级“高”。
附件：本轮改后的 src/model_phase3.py 全文（第 4 轮唯一改动文件；其余 4 个 src 文件自第 3 轮后未改，随附 kinematics.py、features_combine.py、targets.py、game_kinematics.py 供完整核对）+ 数据画像 notes/data_profile.md + 本轮服务器结果 notes/phase3_results_round4_server.md（phase3_results.md 全文 749 行 + Shadow 标定全文 + targets 关键数 + oof/experiments 确认）+ 本文件。
数字来源：以下全部数字均由脚本从本轮冻结的服务器结果文件解析生成（notes/extract_round4_numbers_from_md.py），未人工抄录；括号内为第 3 轮服务器数对照。

## 第 3 轮裁决的落实情况
- Bug 1（DrillFoldClean 静默清零 9 个技能特征）：已修复（单列 groupby 改裸字符串、transform 补 [m] 取值、未见届 fallback、transform 末尾硬断言 skill_s_peak_z_mean 非空 >0.90）。验证：本地直接测试 9 列 NaN 0.0；服务器第 4 轮 Y2 M1a 由第 3 轮 0.0953 回升到 0.1403，LOCO 2025 M1 由 0.0149 回升到 0.0496。
- Bug 2（M1_over_stopwatch 基线偷混选秀顺位）:已修复（基线仅 M0B_STOPWATCH_COLS；M0all 秒表+NGS 改 track 列；新增 M1_track_only 仅 TRACK_PURE）。第 3 轮“超越秒表 +0.1040”已撤回；本轮干净值见下。
- KTR 抄错：第 3 轮提纲第 5 节误抄冒烟版数字的问题已纠正——本提纲全部数字来自服务器第 4 轮冻结文件：forty_s_peak~game_accel_p95 r=+0.414（p=1.22e-14，n=319）、DL 对角 skill_accel~game_accel +0.568 / skill_decel~game_decel +0.432 / skill_lat~game_lat +0.542（n=98）、WR v_max~game_s_p99 +0.201（p=0.109，n=65）。

## 第 3 轮之后新发现并修复的问题（Gemini 第 3 轮未见过）
- --within-year 探针崩塌（本地冒烟，2026-11-22）：`model_phase3.py --quick --within-year` 时 Y1_stage2_logASR 崩塌（M0b_stopwatch 0.2673，默认 0.4910）。根因已用 6 步诊断锁定（/tmp/nfl_wy_diag*.py）：**旧代码固有的退化标准差 bug**，与 within-year 方法无关——训练折 TE_2025 组 ten_yd_split 原始值 [1.56, NaN, NaN]（n=3），组中位数填补后三值全为 1.56，filled std = 2.72e-16（浮点残差，非精确 0.0），通过旧 `sd > 0` 守卫，验证集 z = (1.69−1.56)/2.72e-16 ≈ 1e14，汇合 Pearson r 被天文数字拖垮（逐折 r 均 ~0.4）。同类第二处：TE_2024 ngs_athleticism_score 组内 sd 精确为 0 时旧代码回退 sd=1.0，致验证集 z 高达 13.6（分数单位当 z 用）。
- 修复（仅 model_phase3.py，新增 `_safe_sd(sd, ref_sd)`）：组内 sd < 全局 sd 的 5% 即视为退化，改用全局 sd 标准化（组均值仍用于中心化）；全局 sd 本身 <1e-9 时置 1.0。应用 6 处：DrillFoldClean fit + transform 未见届 fallback（按指标全局 sd）、FeaturePrep fit 通用路径 + lbb_stats_、transform 与 _lbb_z 的未见组 fallback。5% 阈值依据：实测真实组内 sd / 全局 sd 比值 0.19–0.7，退化值 2.7e-16 远在阈值之下。验证：修复后诊断脚本 by_year=True M0b_stopwatch stage2 per-repeat r = [0.477, 0.4749]（修复前 [0.4753, 0.0594]）；默认路径冒烟 Y2 与第 3 轮冒烟逐位一致（M1a 0.1203、M1 0.1105），修复不改变默认路径结果。

## 本轮改动摘要（第 4 轮，相对第 3 轮）
- 仅改 src/model_phase3.py：上述 DrillFoldClean 修复、M1_over_stopwatch/M1_over_all 基线修正、M1_track_only 新增、permutation_p 尺度对齐（每个 null = 同组 3 次重复置换 Δ 的均值）、分类偏置列改折内 5 折交叉拟合、lower_body_burst 折内重算、skill_s_peak 四步阶梯稳健性表、draft_pricing 与 fdr_Y2_full 补 neg_forty/vertical/broad_jump、分位置 game_s_p99/forty_s_peak 均值/中位数、DL 内 EDGE（DE/OLB）vs IDL（DT/NT）拆分、_safe_sd 退化标准差地板。
- 服务器全量：--within-year 开启（未见届用测试届自身无标签分布），种子 42，10×5 CV，1000 置换，2000 自助；model 步骤历时约 7 小时 14 分（第 3 轮 4 小时 36 分，变慢原因疑似 (pos, year)/(drill, year) 分组统计量增多 + 服务器负载波动，见问题 7）。

## 本轮服务器全量实测（2026-11-22，--within-year ON）

### 1. 对抗验证（首要）
- draft_year_macro_auc：ALL 0.4922 ± 0.0230、FORTY_ONLY 0.4924 ± 0.0219、SKILL_ONLY 0.4892 ± 0.0278、STATIC_ONLY 0.4879 ± 0.0331——**四分块全部回到随机水平 0.50**（第 3 轮：0.6627/0.6646/0.4773/0.5742），与本地探针（0.4909/0.4924/0.4917/0.4879）逐数吻合。forty_s_peak 分年均值（全位置）：2023 年 10.419（n=134）、2024 年 10.368（n=145）、2025 年 10.393（n=126）；WR：11.033/11.070/11.079，三年稳定。

### 2. Y2_quality（r）
- M0a 0.1009 ± 0.0275、M0b_stopwatch −0.0016 ± 0.0356、M0c_ngs 0.1984 ± 0.0256、M0b（合并）0.1756 ± 0.0279、M0all 0.1629 ± 0.0315、M1 0.1335 ± 0.0290、M1a 0.1403 ± 0.0271、M1b 0.1587 ± 0.0210、M1_track_only 0.0902 ± 0.0295、M1_over_stopwatch 0.0755 ± 0.0358、M1_over_all 0.1823 ± 0.0317、M1full 0.1279 ± 0.0312。
- 增量：M1_over_stopwatch vs M0b_stopwatch Δ=+0.0771，95% CI [+0.0122, +0.1545]（不含 0）；M1_track_only vs M0b_stopwatch Δ=+0.0918，95% CI [−0.0076, +0.2217]（覆盖 0）；M1_over_all vs M0all Δ=+0.0194，CI [−0.0237, +0.0568]；M1 vs M0a Δ=+0.0326，CI [−0.0114, +0.0819]（覆盖 0）；M1a vs M0a Δ=+0.0394，CI [−0.0097, +0.0951]；M1b vs M0b Δ=−0.0169，CI [−0.0910, +0.0646]。置换检验（折内 X 置换）：观测 Δ=+0.0415，p_perm=0.0030（零分布均值 −0.0051 ± 0.0090）。
- Y2 敏感性：reg_snaps_50 oof r=0.1217（n=299）、reg_snaps_150 oof r=0.1011（n=283）。M3 LightGBM：Y2 r=0.1457，Y1 hurdle AUC=0.8462（第 3 轮：0.175 / 0.850）。

### 3. Y1（均无增量）
- hurdle_stage1：M0a 0.8631 ± 0.0033、M1 0.8641 ± 0.0035；Δ M1 vs M0a +0.0009，CI [−0.0079, +0.0065]；Δ M1_over_stopwatch −0.0280，CI [−0.0711, +0.0123]；置换 obs +0.0015，p_perm=0.0599。
- stage2_logASR：M0a 0.6611 ± 0.0078、M1 0.6607 ± 0.0077；Δ M1 vs M0a −0.0005，CI [−0.0011, +0.0003]；Δ M1_over_stopwatch −0.0013，CI [−0.0056, +0.0042]；置换 obs +0.0000，p_perm=0.1168。
- Y1_year1_hurdle（CV）：M0a 0.8438 ± 0.0050、M1 0.8427 ± 0.0040；Δ −0.0011，CI [−0.0087, +0.0062]，p_perm=0.6856。Y1_year1_logsnaps（CV）：M0a 0.5794 ± 0.0042、M1 0.5807 ± 0.0057；Δ +0.0013，CI [−0.0047, +0.0095]，p_perm=0.0030。
- 时序外推（单次划分，train 2023–2024 / test 2025）：Y1_year1_hurdle M0a 0.8756 / M1 0.8701（n_train 337，n_test 173）；Y1_year1_logsnaps M0a 0.6293 / M1 0.6200（n_train 253，n_test 125）。
- LOCO_Y2（M1 的 r）：2023 年 0.1537（n=112；M0a 0.1971）、2024 年 0.1303（n=104；M0a 0.0938）、2025 年 0.0496（n=83；M0a 0.0106）。第 3 轮对照：2023 0.1849、2024 0.0940、2025 0.0149。

### 4. 选秀定价回归（−log_pick，只控制位置 + 届别，Fig 4 数据）与 FDR
- draft_pricing：neg_forty β=+0.271，p=2.92e-12，q=0.000，n=424；vertical β=+0.232，p=2.63e-09，q=0.000，n=438；forty_s_peak β=+0.225，p=3.48e-09，q=0.000，n=405；broad_jump β=+0.221，p=3.37e-07，q=0.000，n=423；v_max_best β=+0.189，p=3.05e-06，q=0.000，n=405；skill_s_peak_z_mean β=+0.153，p=0.000254，q=0.001，n=510；skill_accel_p95_z_mean β=+0.140，p=0.000632，q=0.002，n=510；skill_lat_accel_p95_z_mean β=+0.094，p=0.00881，q=0.019，n=510；a0_mean β=+0.136，p=0.0106，q=0.020，n=405；其余 8 项 q 均 ≥0.100。
- fdr_Y2 主族（14 项）：仅 skill_s_peak_z_mean q<0.05（β=+0.156，p=0.000316，q=0.004，n=299）；其余 q 均 ≥0.237。
- 稳健性阶梯（skill_s_peak_z_mean）：log_pick β=+0.156，p=0.000316，n=299 → log_pick+forty β=+0.094，p=0.0774，n=253 → log_pick+ngs_final_score β=+0.138，p=0.00268，n=298 → log_pick+forty+ngs_final_score β=+0.093，p=0.0886，n=252。

### 5. KTR 矩阵、M2 中介、DL 拆分
- KTR 5×4 偏相关（位置内）：forty_s_peak ~ game_accel_p95 r=+0.414（p=1.22e-14，n=319）、~ game_s_p99 r=+0.363（p=2.17e-11）；v_max_best ~ game_accel_p95 r=+0.369（p=1.04e-11）、~ game_s_p99 r=+0.306（p=2.47e-08）；skill_lat_accel ~ game_decel_p95 r=+0.288（p=4.73e-09）。ktr_speed_mean=0.6524（第 3 轮同值）。分位置对角：DL 的 skill_accel~game_accel r=+0.568（p=1.04e-09，n=98）、skill_decel~game_decel r=+0.432（p=8.73e-06）、skill_lat~game_lat r=+0.542（p=8.53e-09）；WR 的 v_max~game_s_p99 r=+0.201（p=0.109，n=65）。
- M2 中介（n=299）：decel 中介 indirect=−0.0050，95% CI [−0.0335, +0.0237]，直接效应 0.1554，pct_mediated=−3.3%；burst_1s 复测 indirect=+0.0117，CI [−0.0158, +0.0441]，直接效应 0.1387，pct_mediated=+7.8%。
- DL 亚位置拆分（第 4 轮新增）：EDGE（DE+OLB，n=66）：skill_decel~game_decel r=+0.357（p=0.0069，n=56），Y2 M1 Spearman rho=+0.438（p=0.0026，n=45）；IDL（DT+NT，n=52）：skill_decel~game_decel r=−0.031（p=0.847，n=42），skill_lat~game_lat r=+0.396（p=0.0095，n=42），Y2 M1 Spearman rho=+0.313（p=0.105，n=28）。

### 6. 其他
- Shadow 标定（服务器）：pure vs three_cone oof_r2_within 0.1765、full 0.2436；pure vs shuttle 0.0794、full 0.1981（与第 3 轮同值，features_combine 未改）。
- targets：510 人，y2 covered 299；eb_kappa 32 条，2 条贴 200 上限（gated::DB::m_epa、gated::DL::m_quick_press）。game_kinematics：401 人、10,424,763 帧。oof 510×61。experiments.csv 新增 3 行（共 12 条数据行）。

## 本轮要 Gemini 裁决的事（按优先级）

### 1. --within-year 是否可冻结为最终调和方法
- 对抗验证四分块全部回到 0.49（均值与 std 均在 0.50 附近），forty_s_peak 分年均值稳定（10.42/10.37/10.39）。问：届别泄漏问题是否已解决？--within-year（DrillFoldClean 按 drill×届、FeaturePrep 按位置×届，未见届用测试届自身无标签分布）是否可作为报告的最终方法冻结？未见届 fallback（用测试届自身分布标准化）是否有任何方法论上的保留——尤其 LOCO/2025 时序外推时，它是否应被视作“用未来信息”？业务时序是 3 月整届体测结束、4 月选秀，测试届的无标签 X 在预测时点已存在，请确认这一论证是否成立。

### 2. _safe_sd 退化标准差地板是否合理
- 修复细节见上。问：组内 sd < 全局 sd 的 5% 即改用全局 sd 标准化（组均值仍用于中心化），这一规则是否合理？5% 阈值是否需要敏感性分析（例如 1% / 10% 换阈值看 Y2 是否稳定），还是可直接作为方法细节写入报告？另：TE_2024 ngs_athleticism_score 组内 sd 精确为 0 时旧代码回退 sd=1.0 致 z=13.6 的第二处伪影，是否说明所有位置内标准化的旧结果（第 2/3 轮）都可能含同类小样本伪影、须在报告中披露为局限？

### 3. Y2 的 headline 增量应引用哪一个
- 干净的三个候选：M1_over_stopwatch vs M0b_stopwatch Δ=+0.0771，CI [+0.0122, +0.1545]（不含 0）；M1_track_only vs M0b_stopwatch Δ=+0.0918，CI [−0.0076, +0.2217]（覆盖 0）；M1 vs M0a Δ=+0.0326，CI [−0.0114, +0.0819]（覆盖 0），但置换 p_perm=0.0030。问：报告 headline 应引用哪一个增量？“超越秒表”的表述现在是否成立（以 M1_over_stopwatch 为准）？自助 CI 与置换 p 口径不一致时（M1 vs M0a：CI 覆盖 0 但 p=0.003），报告应以哪个为准、如何表述？M0b_stopwatch 本身 −0.0016（≈0）是否说明纯秒表对 Y2 质量几乎无预测力，从而 M1 的增量主要来自追踪特征而非基线弱？

### 4. Y1 零增量 + 时序/LOCO 的最终故事
- Y1 两阶段、year1 两目标的 M1 增量全部 ≈0（CI 均覆盖 0）；时序外推 M1 略低于 M0a（hurdle 0.8701 vs 0.8756；logsnaps 0.6200 vs 0.6293）；LOCO 2025 M1 由 0.0149 回升到 0.0496 但仍近乎为零，2023 年 M1 由 0.1849 降到 0.1537。问：“追踪特征提升新秀质量（Y2）预测，但不提升出场时间（Y1）预测”这一分野是否可作为最终结论？LOCO/时序的弱结果是否说明模型不具备跨届泛化、须在报告中明确写为局限，还是说 2025 年 n=83 的 LOCO 样本太小、不宜过度解读？

### 5. DL 的 EDGE vs IDL 拆分如何报告
- EDGE 的 decel 对角线 + Y2 Spearman 均显著，IDL 的 decel 对角线 ≈0 但 lat_accel 对角线显著、Y2 Spearman 仍为正（p=0.105）。问：这是否说明 KTR 对角线信号是 EDGE 驱动的、而 Y2 相关在两亚组都存在？报告中应如何表述——DL 整体对角线（第 3 轮裁决要求查的混杂）是否已解释清楚，还是须进一步按 EDGE/IDL 分别建模？

### 6. 稳健性阶梯与 Fig 4 措辞
- skill_s_peak 在 +pick+ngs 后仍显著（β=+0.138，p=0.0027），加入 forty 后衰减到 β=+0.094，p=0.077；三重控制 β=+0.093，p=0.0886。定价端 forty_s_peak β=+0.225（q=0.000）而质量端 FDR 仅 skill_s_peak 存活。问：按第 3 轮裁决，措辞用“差异化市场定价效率”、绝不写“市场完全错误定价”——本轮数字是否支持这一措辞？skill_s_peak 的衰减主要来自 forty（丢样本 + 控制），是否可定稿为“技能峰值速度是唯一经多重控制后仍具方向性的质量信号，但证据为边界水平”？

### 7. 计算效率与第 5 轮
- model 步骤由第 3 轮 4 小时 36 分延长到本轮 7 小时 14 分（同机型 e2-standard-2、同样本量、同置换/自助次数）。问：代码中有无明显的重复计算值得在最终轮修复（例如 --within-year 下每折的 (drill, year)/(pos, year) 分组统计、或置换路径的重复物化）？还是这只是分组数增多（28→84 个 drill 单元、5→15 个位置组）+ 服务器负载波动的预期代价、无需处理？
- 冻结检查：若以上 6 项通过，请确认 Phase 3c 可冻结、进入 Phase 4 出图与 Phase 5 写作；若仍有问题，请列出第 5 轮最小修改清单（只改代码 → 本地冒烟 → 服务器重跑 → 第 5 轮核对）。

## 附：本轮服务器结果
- notes/phase3_results_round4_server.md：phase3_results.md 全文（749 行）+ Shadow 标定全文 + targets_summary.json 关键数 + game_kinematics/oof/experiments 确认 + 服务器状态确认（e2-standard-2、IP 35.213.17.8；x-ui 面板端口 :2053/287a58ff96/ 本轮确认 HTTP 200 可达）。
