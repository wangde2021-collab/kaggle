# NFL Big Data Bowl 2027 — Gemini 五轮核对记录

约定：每轮把代码全文 + 数据画像 + 本轮验证分数与日志关键段落发给 Gemini 最高模型（Neon 高）核对：查数据泄漏、查验证划分是否可信、查特征与目标定义、查代码 bug，给可执行的修改意见。按意见改代码 → 服务器重跑 → 下一轮。共 5 轮。

**状态（2026-11-21 03:00 左右）：第 0 轮（方案 intake）已完成；第 1–5 轮待服务器计算结果。**
阻塞原因：东京服务器房价项目 stage2 调参仍在全力跑（02:40 查：负载 7.78，stage2_tune.py 占 631% CPU，SERVER_RUN_DONE 不存在），按规矩 NFL 计算须等它跑完再启动。五轮核对需等 Phase 1–3 在服务器上跑出结果后才能开始。

## 第 0 轮：方案 intake（2026-11-21 02:14–02:29）

- 输入：problem_statement.md + data_profile.md + rules.md（附件上传）+ 深度分析请求原文。
- 输出：Gemini 获奖方案全文，存 notes/gemini_scheme.md。对话 https://gemini.google.com/app/f42509acc9d07f4b 。
- 要点：主研究问题“超越秒表与跳测黑洞”；双核目标 Y1（暴露校正年均 REG 档，Hurdle）+ Y2（EB 收缩位置单档质量分）；Shadow Agility；选秀错误定价森林图；10×5 折重复分层嵌套 CV + Year-1 Only 时序外推 + LOCO + 1,000 次置换检验；8 项防泄漏；5 阶段里程碑。
- 本轮发现的方案假设偏差（本地实测纠正，见 notes/eda_audit.md）：
  - 跳测 MNAR 不成立：跳测组选秀顺位中位数 121 vs 非跳测组 124（p=0.401），ngs_final_score p=0.504；分轮跳测率均 >50%（第 1 轮 68.4% 仅略高）。
  - Y2 覆盖：生涯 REG 档 ≥75 档为 320 人（方案预估 310–350 一致）；零 REG 档球员 109 人（非方案写的 105 人，口径不同）。

## 第 1 轮：Phase 1–2 全量结果 + Shadow Agility 失败诊断（2026-11-21 09:27–09:44 完成）

- 输入：服务器全量 Phase 1（四道门控全 PASS）+ Phase 2（40 码拟合 R² 均值 0.990、99.1% >0.95；技能事件 5,091 个）+ 代码全文（src/ 7 个文件）+ data_profile.md + eda_audit.md + gemini_scheme.md。模型 Neon 最高档、思考等级“高”。对话 https://gemini.google.com/app/8adf0ca845f706e2 。回复全文存 notes/gemini_round1_20261121.md。
- 核心结论：Shadow 失败大概率是“假失败”——目标量纲错配（位置内 z 化的 X 去预测跨位置原始秒数 Y，位置间方差占 65–75%）+ Ridge(alpha=10) 欠正则。修复后有望跨过 r≥0.40 门限。
- Gemini 指出的硬 Bug 清单（须逐项修完重跑；当前服务器 Phase 3 全量结果只能当修复前对照快照，不可引用）：
  1. Year-1 截断：game_id//1e6 丢掉次年 1 月常规赛（2024010707 就是 2023 季 Week 18），audit 与 targets 同病，改用 games.season。
  2. DB 覆盖档漏滤 NO_ASSIGNMENT；_residualize 缺失协变量时回填原始正值污染残差列；Y2 门槛应用位置专属有效档 n_plays≥75 且先门控后标准化；object 布尔列须显式映射；OL 漏 pass_rushers_encountered≥1。
  3. EB 收缩数学错误：z_qual_raw 已位置内 z 化，组均值≈0 → κ 被 clip 锁死 200，Y2 被污染成档数/年份代理；须按指标做 ANOVA 方差分解估 κ。
  4. game_kinematics END_EVENTS 误含 handoff/run，跑球档全被砍断丢弃；缺 ball_snap 时返回未裁剪轨迹；未滤犯规取消档；dir 未平滑。
  5. PositionZScore 全局中位数插补（OL 缺 40 码会被填成 -3.5 SD 怪兽）；丢弃 pos_group 固定效应且把 log_pick 等也位置内标准化；M1 vs M0b 对比错位（应补 M1b vs M0b 与全对照）；CORE_TRACK 混入 4 个静态秒表列；固定 Ridge(alpha=10) 欠正则；bootstrap 用第 1 次重复的 OOF 配 10 次均值；oof_predictions.parquet 错位拼接（子集 OOF 按下标拼到 510 行上）。
  6. 置换检验原假设错误：应保持 (M0a, pos_group, y) 不动、只置换 X_track 行（Freedman-Lane / 层内置换），而非全局打乱 y。
  7. 时序外推：M0a 含数值 draft_year 应移除；Year-1 口径 CV 与外推不一致；LOCO 只跑 M1 没跑 M0a。
  8. M2 中介未控 pos_group；KTR 须位置内偏相关；FDR 特征须先位置内 z、draft_year 当分类、HC3 稳健误。
  9. drill 归一化泄漏：全样本 48 drill vs 三年稳定 31 drill，混入 17 个跨年不稳 drill；须封进折内 Transformer + 只留 31 个。
  10. forty_features dist 截断后索引越界隐患；split_time 改插值；high_speed_brake_power 单位错（力非功率）；braking_impulse 硬编码 0.1；jerk_p95 应限启动前 1s；cut_events 须平滑 dir、放宽 apex 速度门限；lower_body_burst 全样本位置 z（轻微泄漏）；shadow 与明细特征共线，不可并列入同一回归。
- 叙事决策：不改成“监督拟合 NFL 表现的 Shadow”（双重沾取）；先跑 P0–P3 修复，再按实测位置内 OOF r 定路径 A（r≥0.35 保留 Shadow 主线，同步修正 Opt-Out 叙事为“全联盟结构性缺口”）或路径 B（r<0.30 转“构念分离”叙事：3-cone 与专项变向测的是不同能力，主线改为制动动力学 + KTR + 选秀错误定价）。
- 本轮状态：意见已收，正在按清单修代码；服务器 Phase 3（修复前版）仍在跑，作对照快照。

### 第 1 轮修复实施与本地冒烟（2026-11-21 约 10:00–10:30）
- 改动文件：kinematics.py（dir 解缠绕 + SG 平滑、cut 事件重定义为平滑方向角速度弧）、features_combine.py（forty dist 切片 + np.interp 分段计时、制动功率改为真功率 + 质量无关功率、制动冲量用真实 dt、jerk 限首 1 秒、drill 归一化仅用三年稳定 drill + 两级聚合 + COD-heavy 变体、Shadow 标定改位置内 z 目标 + RidgeCV + Pure/Full 双版 + P0 黄金标准诊断）、targets.py（Year-1 用 games.season、DB 排除 NO_ASSIGNMENT、_residualize 不再回填、EB 收缩改为逐指标 ANOVA 方差分解 + 门控后 z、bool 显式映射、OL 要求 pass_rushers_encountered≥1、Y3 首赛季断言放宽到 gap≤2）、game_kinematics.py（END_EVENTS 去掉 handoff/run/fumble、无 ball_snap 轨迹丢弃、player_play 过滤被罚 nullified 的 play）、model_phase3.py（按位置中位数填补、pos 固定效应、特征分组 TRACK_PURE/STATIC/SHADOW、RidgeCV/LogisticRegressionCV、平均 OOF 自助、OOF 按 nfl_id 对齐、折内 X 置换、时序去掉 draft_year、LOCO 加 M0a、中介控制 pos、KTR 偏相关、FDR 用 HC3 + C(draft_year)、折内干净 drill 归一化、M3 改为真 LightGBM 工厂）。
- 冒烟中修的 bug：skill_drill_features 里 high_speed_brake_power 的 s 未加 hi 掩码导致广播错误；targets 首赛季断言过严——2023 届 Daniel Scott / Jovaughn Gwyn / Dalton Wagner 均 2025 年才首次打 REG（13/11/2 档），属真实晚首秀，断言放宽到 gap≤2（全量分布 {0:378, 1:20, 2:3}）；Y2 merge 后 y2_covered 带 NaN（QB/RB 等无 Y2 行），fillna(False)。
- Shadow 标定（smoke parquet 的 combine 数据即全量：762 次 forty、5,091 技能事件）：
  - Pure：three_cone 位置内 OOF r=0.419（R²=0.175，n=180）、short_shuttle r=0.309（n=201）；Full：0.459 / 0.459。旧版（服务器）：0.105 / 0.179——位置内 z 目标修复是主因。
  - P0 黄金标准（分 drill 类型，各自轨迹预测各自成绩）：three_cone r=0.321（n=149，202 事件）、short_shuttle r=0.300（n=159，221 事件），均 <0.55——运动学算子本身弱，不是跨 drill 聚合的瓶颈。
  - 机制证据：裁剪后轨迹时长 vs 官方成绩，位置内相关仅 3-cone 0.46、shuttle 0.13（shuttle 轨迹 4.99s vs 官方 4.46s）——10Hz 事件窗口未对准计时尝试，轨迹时长本身都还原不了官方成绩。
  - 结论（Path B，构念分化）：Shadow 指数不是变向能力度量，而是技能 drill 里的一般运动能力代理（Pure r≈0.4 仍有预测价值，保留为特征、修正解释）；主线改为减速动力学 + KTR + 选秀定价偏差。待第 2 轮裁决是否追一个“持续速度”算子变体。
- targets 冒烟：零 REG 档 109 人、Hurdle 236 人、Year-1 Hurdle 254 人、ASR 中位数 108.75、ASR 与生涯档数相关 0.9655；Y2 位置专属门控覆盖 299 人（DB 71 / DL 73 / OL 67 / TE 22 / WR 66，旧版生涯档≥75 为 320 人）；z_qual 均值 ~0、标准差 0.765。
- game_kinematics 冒烟：387 人、1,358,825 帧（旧版 1,085,631 帧——END_EVENTS 修复后跑球档不再被 handoff 截断）；丢弃 no_snap 87 + too_short 20 条轨迹；nullified play 5,196 个已滤。

## 第 2 轮：（2026-11-21 17:04 回传）

- 输入：修复后 5 个代码文件 + gemini_round2_draft.md + phase3_results_fixed_server.md。模型 Neon、思考等级“高”。对话 https://gemini.google.com/app/8adf0ca845f706e2 。回复全文存 notes/gemini_round2_20261121.md。
- 核心裁决（6 项，详见原文）：
  1. tau_best 跨年漂移根源 = fit_sprint_curve 缺 t0 项 + 起跑检出漂移；WR v_max_best 12.20–12.27 yd/s 是伪渐近值。修法：fit_sprint_curve 加 t0 ∈ [−0.3, 1.5] 与物理上界 v_max ≤ s_peak×1.12，输出实测 forty_s_peak；先做对抗验证分块诊断（FORTY/SKILL/STATIC），仍 >0.56 再做同届内调和，目标 draft_year macro AUC ≤0.55。
  2. Path B 确认：Shadow 改称 Skill-Drill Movement Index / Shadow Athleticism Proxy；不再追“持续速度”算子。
  3. Y2 增量：permutation_p 有两个 Bug（用 M1 的 alpha 拟合 M0a；只跑 repeat=0）+ 单一正则化误杀基线列（Y1 Hurdle M1 的 C=0.072 把 log_pick 惩罚加重 27 倍）。修法：残差/偏置堆叠、拆 M0b 为 M0b_stopwatch 与 M0c_ngs、增设 M1_over_stopwatch / M1_over_all、修 permutation_p（各自正确正则化、多次重复聚合）。
  4. KTR：定位为跨场景生物力学迁移证据；补 5×4 偏相关矩阵（记 n）+ game_snap_burst_1s 中介复测；ktr_speed_mean 0.6384 偏低因 v_max_best 虚高，修 t0 后应在 ~0.82–0.90。
  5. 主线切换：放弃“减速动力学优先”，改为“专项动作功能速度 vs 已被定价的直线秒表速度”；补选秀定价回归（−log_pick，Fig 4 森林图）与 skill_s_peak_z_mean 控制 log_pick+forty+ngs 后的稳健性回归；FDR 主族拆清（9 个纯专项追踪特征 + 修正后 40 码核心参数，去掉 2 个冗余 Shadow 与 4 个静态列）。
  6. Y1 vs Y2 双重分离升级为一号王牌发现：“Draft Capital Buys Snaps, Not Efficiency”。
- 最小闭环修改清单（原文见 gemini_round2_20261121.md 末尾，总量 <180 行）。

### 第 2 轮“待核实项”核实结论（2026-11-21 17:31–17:40，动手改代码前以代码为准）
- ① permutation_p 用 M1 的 alpha_median 拟合 M0a、只跑 repeat=0 —— 属实。model_phase3.py:572–577 传 `_fp = out_d["M1"].get("alpha_median")`；:384–403 fixed_factory 同一 fixed_param 用于 oof_m1 与 oof_m0；permutation_p 内 `_materialize_folds(..., 0, ...)` 只用 repeat=0。
- ② calibrate_position_z 折前全局中位数填补、lower_body_burst_fc 折前 z-score —— 属实。features_combine.py:324 `X.fillna(X.median(...))` 在 KFold 循环之前；:399–401 lower_body_burst_fc 由 zscore_within 在全样本 df 上算出。注意：Gemini 写的“第 253 行”与实际行号不符，以代码为准。另附：static_features 的 lower_body_burst（features_combine.py:91）同样是全样本位置内 z（文件第 34 行注释自认），进 model_phase3 的 STATIC 列，本次一并记录。
- ③ WR 的 v_max_best 均值 2024–2025 年 12.20–12.27 yd/s —— 属实（本地 forty_attempts.parquet，全量 762 次尝试；WR：2023 年 11.52 n=30、2024 年 12.20 n=26、2025 年 12.27 n=30）。关键对照：同批实测 s_peak_best 均值 11.03/11.07/11.08，三年稳定——漂移来自无 t0 拟合虚推 v_max，不是真实极速变化。
- ④ M0b 列定义混入 ngs_final_score —— 属实。model_phase3.py:94–95：M0B_COLS = [ngs_athleticism_score, ngs_final_score, forty, ten_yd_split, vertical, broad_jump]。
- 另：Gemini 引用的“model_phase3_2.py 第 307–311 行和第 438–444 行”实际文件名为 model_phase3.py，行号也不对应（实际为 :369–403 与 :559–577），第 3 轮提纲须注明以代码为准。

## 第 3 轮：（待定）

## 第 4 轮：（待定）

## 第 5 轮：（进行中——服务器全量未跑完，Gemini 第 5 轮未派；详见文末“第 5 轮”一节）

### 第 1 轮修复版服务器全量基线（2026-11-21 13:02–16:38，Task A 回传，browser-task:7614ea66…）
- Phase 2：players 510、forty 762 次、R² 均值 0.99013、99.1% >0.95；skill_events 5091；drills_normalized 28、stable_drills 31。Shadow 位置内 OOF r：Pure three_cone 0.4219（n=180，过 0.40 门）、Full 0.4572；Pure short_shuttle 0.2826（n=201，未过门）、Full 0.4299（过门）。对比修复前（0.1047 / 0.1794）——位置内 z 目标修复是主因。
- Phase 3a：players 510、reg_plays_used 240,412、ASR-生涯档相关 0.9655、首赛季 gap {0:378, 1:20, 2:3}；y1 零档 109、hurdle_150 236、year1_hurdle_100 254；ASR 中位数 108.75；y2 covered 299（DB 71 / DL 73 / OL 67 / TE 22 / WR 66）；z_qual_std 0.7648（修复前 0.3267——EB 污染已消除）；y3_start_rate_mean 0.2268。
- Phase 3b：401 人、10,424,763 帧（修复前 8,318,924）；REG 816 场、nullified 5196；丢弃 no_snap 599 + too_short 159。
- Phase 3c（2h44m，10×5 CV、1000 置换、2000 自助）：oof_predictions.parquet 已确认（510 行、31 列）。
  - Y1_hurdle_stage1：M0a 0.8631、M1 0.8556、Δ-0.0076、p_perm=0.5355。
  - Y1_stage2_logASR：M0a 0.6611、M1 0.6315、Δ-0.0296（CI 不含 0）、p_perm=0.4895。
  - Y2_quality：M0a 0.1009、M0b 0.1959、M1a 0.1402、M1 0.1210、M1full 0.1085、M1b 0.1647；Δ M1 vs M0a +0.0201（CI 含 0），置换检验观测 Δ=+0.1068、p_perm=0.0010（均值 Δ 与置换观测 Δ 口径不同，待第 2 轮解释）。
  - Y1_year1_hurdle：M0a 0.8438、M1 0.8321、Δ-0.0116、p_perm=0.2178。Y1_year1_logsnaps：M0a 0.5794、M1 0.5714、Δ-0.0080、p_perm=0.0160。
  - 时序外推（2025）：Hurdle M0a 0.8756 vs M1 0.8424；logsnaps M0a 0.6936 vs M1 0.6484。LOCO Y2：2023 0.1971/0.1977、2024 0.0938/0.1226、2025 0.0106/0.1045。
  - ★ 对抗验证：draft_year macro AUC 0.6976±0.0317，仍远超 0.55 红线——届别信号未消除。
  - FDR（Y2）：仅 skill_s_peak_z_mean 存活（β+0.156、p=0.000316、q=0.006）；shadow_agility_pure q=0.103；FDR Y1 stage2 无显著项（最小 q=0.699）。
  - M3 LightGBM：Y2 r 0.1430、Y1 hurdle AUC 0.8440。M2 中介：间接 -0.0050、CI 含 0、中介占比 -3.3%——无中介。
  - ★ KTR 全量与冒烟相反：混合 r 0.8064，位置内偏相关 r=0.2569（p=3.34e-06）——冒烟 0.049（p=0.386）不可信（冒烟 game 帧数仅 1.36M，全量 10.4M）。第 2 轮 Q4 须按全量数重写。
  - Y2 敏感性：reg_50 r 0.1301、reg_150 r 0.0878。
- 修复版结论：Y1 仍是选秀顺位主导（M1 全未胜 M0a，stage2 上追踪特征甚至显著拖累）；Y2 上 M1 相对 M0a 有置换显著的增量（p=0.0010），但绝对值仍低于秒表基线 M0b（0.121 vs 0.196）；对抗验证仍红灯；KTR 全量显著。备份 prefix_snapshot/ 含修复前 oof_predictions.parquet（84,972 字节）——证实修复前文件其实存在，只是 Task A 当时未核对。

### 修复前 Phase 3 全量基线快照（服务器，2026-11-21 09:27–11:24，Task A 回传）
- targets：y2 covered_75=320、ASR 与生涯档数相关 0.9655、year1_hurdle_100=244、z_qual_std=0.3267（EB 污染的铁证）。
- game_kinematics：401 人、8,318,924 帧（handoff 截断版）。
- 主建模（10×5 CV，1000 置换，2000 自助）：M1 在 5 个目标上全未胜过 M0a——Y1 Hurdle AUC M0a 0.8625 vs M1 0.8506（Δ-0.0119，p_perm=0.6074）；Y1 stage2 log-ASR r M0a 0.4204 vs M1 0.3668（Δ-0.0536，p=0.7782）；Y2 r M0a 0.2761 vs M1 0.2684（Δ-0.0076，p=0.7303）；Year-1 Hurdle M0a 0.8519 vs M1 0.8357（Δ-0.0161，CI 不含 0）；Year-1 logsnaps M0a 0.5125 vs M1 0.4986。五项置换 p 全 >0.6。
- 时序外推（2025）：M0a 均胜 M1（Hurdle 0.8910 vs 0.8522；logsnaps 0.7082 vs 0.6619）。LOCO Y2 r：0.133 / 0.278 / 0.220。
- 对抗验证：draft_year macro AUC 0.7147±0.0172——远超 0.55 红线，特征带很强的届别信号。
- FDR（Y2）：shadow_agility β+0.075、p=8.6e-06、q=0.000；skill_jerk/accel/lat_accel/decel 均显著；flying_30 β=-0.576。
- M3 LightGBM：Y2 r 0.2874、Hurdle AUC 0.8422。M2 中介：间接效应 0.00094、CI 含 0、中介占比 1.1%——无中介；KTR 混合 r 0.8139。
- Y2 敏感性：coverage_50 r 0.2701、coverage_150 r 0.2398。
- 异常：虚拟机在 11:17:32 被另一会话执行 gcloud stop 停掉（非 Task A 所为）；oof_predictions.parquet 与 experiments.csv 内容未能最终核对，重启后复查。

### 第 1 轮修复后本地冒烟（model_phase3 --quick，2026-11-21 约 12:40 通过）
- 全流程跑完，输出文件齐全。冒烟数仅供参考：对抗验证 draft_year macro AUC 0.7046±0.0334（折内干净 drill 后仍 >0.55）；FDR（Y2）仅 skill_s_peak_z_mean 存活（q=0.0057），shadow_agility_pure q=0.205；M3 Y2 r 0.1348；M2 无中介（1.5%）；KTR 混合 r 0.68、位置内偏相关 r=0.049（p=0.386）。
- ★ 对抗验证根源定位：tau_best 按年系统漂移（WR 1.296→1.459→1.525、DB 1.262→1.339→1.478、DL 1.225→1.229→1.406，均 p<0.02）；追踪推算的 0–10 码分段比官方秒表 ten_yd_split 系统性长 0.86 秒（2023 +0.815s、2024 +0.854s、2025 +0.927s），相关 0.638——40 码曲线参数跨年不可比，疑似 2025 追踪管线变化。第 2 轮首要议题：是否对 tau 做年份调和 / 用秒表分段约束曲线拟合 / 弃用 tau_best。

## 第 3 轮：服务器第 3 轮全量结果核对（2026-11-22 01:13 回传）

- 输入：本轮改后 5 个 src 文件 + data_profile.md + phase3_results_round3_server.md（服务器全量全文）+ gemini_round3_draft.md（6 个议题，数字已填服务器实测值）。模型 Neon、思考等级“高”。同一对话 https://gemini.google.com/app/8adf0ca845f706e2 。回复全文存 notes/gemini_round3_20261122.md。
- 总裁决：Phase 1 / Phase 2 / Phase 3a / Phase 3b 全部通过、可立即冻结；Phase 3c（model_phase3.py）暂不可冻结——查出 2 处静默高危 Bug，本轮 CV 与增量对比关键数字失真，须仅改 model_phase3.py 做第 4 轮最小修复并重跑一次。
- Bug 1（致命，已用直接执行测试证实）：DrillFoldClean 把全部 9 个 skill_*_z_mean 特征静默清零。fit() 用 sub.groupby(["drill_name"])——pandas 2.1.4 单列列表分组返回 1-元组键 (('DRILL',), None)，transform() 查 ('DRILL', None)，100% 查不到，全退回 (NaN, 1.0)；transform() 又写成 .get(k, (nan,1.0))[0]，漏了按指标名 [m] 取值（字典按 [0] 取会 KeyError）。实测：44 个键全是元组键、key match False、9 列 NaN 占比 1.0。反证：fdr_Y2（读 Phase-2 预计算列）skill_s_peak 仍 q=0.004，但本轮 CV M1a Y2 从第 2 轮 0.1402 跌回 0.0953、LOCO 2025 从 0.1045 跌回 0.0149、SKILL_ONLY 对抗 0.4773（只测了 shadow_pure）。第 2 轮对照值已对 resume-runlog 核实属实。
- Bug 2（致命，已对代码核实）：M1_over_stopwatch 基线含 M0A_COLS（顺位），却去减不含顺位的 M0b_stopwatch——“超越秒表 +0.1040”100% 来自偷混的选秀顺位（残差层 alpha 顶到 1e+05，追踪贡献≈0），必须撤回。同时 M0all 把秒表/NGS 列放进 raw_cols（全局中位数直通、不做位置内标准化），致 M0all 0.1747 反而不如 M0c_ngs 0.1926。
- 议题裁决要点：① t0 修复正确，forty_s_peak 三年平稳；FORTY_ONLY 仍 0.6646，裁决启用 --within-year（先修 Bug 1 再跑分块对抗验证定范围，统一开启更简洁）；LOCO/2025 未见年份须用测试届自身无标签 X 分布做届内标准化（零泄漏）。② M0b 拆分证实：秒表 M0b_stopwatch Y2 −0.0198（预测力为零），NGS M0c_ngs 0.1926——原 M0b 信号 100% 来自 NGS。③ 堆叠架构验证通过（Y1 增量归零）；置换 p 显著但自助 CI 覆盖 0 的裁决：置换存在尺度 Bug（obs 为 3 次重复均值、null 为单次重复池），须改为每个 null 样本也跨 3 次重复取均值；报告以“10×5 均值 Δ + 自助 CI”为准，p_perm 只作补充；分类偏置列改折内 5 折交叉拟合。④ Fig 4 核心故事立得住：定价端 forty 主导（β 0.225/0.189），质量端仅 skill_s_peak 存活（β 0.156，q=0.004）；三重控制 β 0.093、p=0.089 是合理边际衰减（丢 47 名跳测球员 + ngs 打包）；第 4 轮补 4 步阶梯稳健性表 + 定价/FDR 补 -forty、vertical、broad_jump；措辞用“差异化市场定价效率”，绝不写“市场完全错误定价”。⑤ KTR：提纲第 5 节多处数字误抄冒烟版（已核实 Gemini 指出的服务器数属实，见 REFLECTIONS.md）；ktr_speed_mean 0.6524 正常（锋线拉低）；DL 一枝独秀须查 EDGE vs IDL 亚位置混杂（DE 36 + OLB 30 vs DT 47 + NT 5）；burst_1s 中介 +0.0117（CI 覆盖 0，+7.8%）可定稿为边界结论；Fig 5 用 4×4 核心热力图 + 分位置条形图。⑥ 冻结：etl/kinematics/features_combine/targets/game_kinematics 立即冻结；lower_body_burst 全样本 z 改 FeaturePrep 内折内重算；shadow 预拟合表示可接受；FDR 预计算列筛选可接受。
- 第 4 轮最小修改清单（仅 model_phase3.py，约 90 行）：修 DrillFoldClean（单列 groupby 不用列表、补 [m]、加非空断言）+ 未见年份 fallback + M1_track_only 新增与 M1_over_stopwatch/M0all 列角色拆清 + permutation_p 尺度对齐 + 分类偏置交叉拟合 + 4 项轻量统计表（阶梯稳健性、定价/FDR 补 3 列、分位置速度保持率、DL EDGE/IDL 拆分）。改完先 --quick 确认断言过、对抗 ≤0.55、M1_over_stopwatch 干净、M0all ≥ M0c_ngs，再放服务器最后一次全量（--within-year）。

## 第 4 轮：服务器第 4 轮全量结果核对（2026-11-22 13:04 回传）

- 输入：本轮改后 model_phase3.py（唯一改动文件；其余 4 个 src 文件自第 3 轮后未改，随附供完整核对）+ data_profile.md + phase3_results_round4_server.md（服务器全量全文 749 行）+ gemini_round4_draft.md（7 个议题，数字已填服务器实测值）。模型 Neon、思考等级“高”。同一对话 https://gemini.google.com/app/8adf0ca845f706e2 。回复全文存 notes/gemini_round4_20261122.md。
- 总裁决：上游 4 文件持续冻结；主 CV 核心正向结论全部跑通（Y2 M1_over_stopwatch 显著超越纯秒表、M1a/M1 相对 M0a 稳定回升、对抗验证全块归零到 0.49、DL 亚组拆分立住）。但 Phase 3c 暂不可冻结——又查出 2 处隐蔽 Bug + 1 处口径不符，并定位 7h14m 耗时的根因。须仅改 model_phase3.py 做第 5 轮收口（约 60 行），跑完即终极冻结。
- Bug A（高危，待代码核实）：FeaturePrep.transform 在 --within-year 未见届场景下，缺失值填补退回跨位置全局中位数（med = gkey.map(track_med_).fillna(__global__)），LOCO/2025 时序外推中测试届 gkey 100% 未见，约 17%–20% 跳测球员被填全局中位数（如 315 磅 OL 填 4.62 秒 40 码，再按 OL_2025 组内标准化成 +3 SD 怪物）——解释了主 CV 回升但时序/LOCO 反跌。修法：fit 增存 pos_group 级训练中位数；transform 三级填补：训练折 (pos, year) 中位数 → 测试届自身 (pos, year) 无标签中位数 → 训练折 pos_group 中位数 → __global__（_lbb_z 同步）。
- Bug B（高危，待代码核实）：M0all 经 repeated_cv_oof 用 make_model_pipe(protect=True 默认)，M1_over_all 的 Stage 1 用 protect=False——M0all 里 BaselineProtect 把 M0A_COLS 放大 10 倍，致 M0C_NGS_COLS 被强惩罚压扁（alpha=266，M0all 仅 0.1629）；M1_over_all Stage 1 不压扁，单靠基线已 ~0.180，delta_M1_over_all_vs_M0all=+0.0194 几乎全是口径差。M0a/M0b_stopwatch/M1_over_stopwatch/M1_track_only 不受影响（单侧列不触发 protect）。修法：make_model_pipe 默认 protect=False，仅分类堆叠 Stage 2 的 rpipe 显式传 protect=True。
- Bug C（中低危）：ktr_speed_by_position 标题写 game_s_p99 / forty_s_peak，代码却是 game_s_p99 / v_max_best（WR 输出 0.781，实为 9.03/11.08=0.815）；Y2_sensitivity 的 reg_snaps_50 与主集完全重合（z_qual 已在 targets.py 以 n_plays>=75 门控，n=299 相同）。修法：ktr_speed 改 forty_s_peak 分母（原 v_max 版另存 ktr_speed_vmax_mean）；敏感性改为 n_plays>=100 与 >=150；experiments.csv 日期改 datetime.date.today().isoformat()；phase3_results.md 输出 per_position_spearman。
- 议题裁决要点：① --within-year 完全通过可冻结（修 Bug A 后）；未见届 fallback 的业务时序论证 100% 成立，建议 Method 节定义为 "Within-Cohort Unsupervised Harmonization"。② _safe_sd 地板合理且必要，可直接写入方法节，无需 1%/10% 敏感性重跑；旧轮次 bug 不必在正文披露为局限。③ Y2 headline 并列两个干净主增量：M1_over_stopwatch Δ=+0.0771 CI [+0.0122, +0.1545]（“超越秒表”成立且硬核）+ M1a/M1 vs M0a（Δ +0.0394/+0.0326，置换 p=0.0030 为条件信号补充）；报告口径：显著性以自助 95% CI 为准，不越界宣称“显著提升全样本外 r”。④ Y1 零增量 vs Y2 增量的双重分离（Double Dissociation）可作一号主结论；LOCO/2025 时序不要基于被 Bug A 污染的数字下悲观结论，等第 5 轮重看。⑤ DL EDGE/IDL 拆分是教科书级亚组拆解，直接写进正文（Role-Specific Kinematic Translation：EDGE 靠速度/加速/制动迁移，IDL 靠横向变向载荷迁移）。⑥ Fig 4 措辞“差异化市场定价效率”完全支持；第 5 轮补 1 行稳健性对照（n=253 子集上只控 log_pick），拆清 forty 衰减中样本缩水 vs 共线性占比。⑦ 耗时根因：permutation_p 每次置换重跑 Stage 1 基线拟合（约 80% 冗余）+ FeaturePrep.transform 的 Python for pos 循环；修法：置换循环外缓存 Stage 1（回归 base_va/resid、分类 cross-fitted off_tr/off_va）+ 向量化 .map()，全量预计 7h14m → ~25 分钟。
- 第 5 轮核对 3 项指标（本地 --quick --within-year 约 1 分钟）：① 2025 时序外推与 LOCO_Y2 恢复干净；② M0all 逼近或超过 M0c_ngs、delta_M1_over_all_vs_M0all 干净；③ ktr_speed_by_position 的 WR 保持率与 9.03/11.08=0.815 吻合。3 项全绿即冻结，不再开大审。

## 第 5 轮：服务器第 5 轮全量（已完成）+ Gemini 第 5 轮（已回传，Phase 1–3 正式冻结；详见文末“第 5 轮核对完成”一节）

- 第 5 轮修复实施（2026-11-22 约 13:10–13:30，仅改 src/model_phase3.py，py_compile 通过，md5 6d1c3f7dc117ce7574dfaa09a3955ebe，94,930 字节，1,819 行）：逐项对代码核实——Bug A 属实（transform 与 _lbb_z 均退回全局中位数；DrillFoldClean 无此问题，直接用测试折自身分布）；Bug B 属实（仅 M0all 受影响；分类 rpipe 依赖默认 True 保护 offset）；Bug C 属实（ktr_speed=game_s_p99/v_max_best vs md 标题 forty_s_peak；reg_snaps_50 冗余；日期硬编码；per_position_spearman 已算未写入 md）。提速方案核实可行：置换只对 M1 调用（base_track_cols=[]、raw_cols=M0A_COLS，与被置换 track 列无交集），Stage 1 确实 100% 置换不变；且置换里用固定正则（单次拟合非 CV），缓存后分类每调用省 6/7 次拟合。改动：① fit 增存 track_pos_med_ 与 lbb pos_median；transform 与 _lbb_z 改三级填补；② make_model_pipe 默认 protect=False，分类 Stage 2 rpipe 显式 protect=True；③ ktr_speed 改 forty_s_peak 分母（原 v_max 版另存 ktr_speed_vmax_mean/median）；Y2_sensitivity 改 n_plays>=100/150；experiments.csv 日期改 datetime.date.today().isoformat()；md 加 per_position_spearman 一节；④ _robustness_ladder 加第 5 步（forty.notna() 子集上只控 log_pick），res["robustness_skill_s_peak"] 改取 ladder[3]（仍为第 4 步全控制）；⑤ _stacked_fit_predict 拆为 _stacked_stage1 + _stacked_stage2，permutation_p 循环外缓存 Stage 1（加 base 列与 track 列无重叠断言），FeaturePrep.transform 组内 z 改 gkey.map() 向量化；⑥ 文件头加 Round-5 说明段。直接执行测试 /tmp/nfl_r5_unit_test.py（合成数据，6 项全过）：T1 未见届跳测 OL 的 forty z 由旧规则 −1.86 降为 −0.119（组中心）；T2 向量化 transform 与旧循环逐位一致（max diff 0.00e+00）；T3 stage1+stage2 与原 fit_predict 逐位一致（回归+分类）；T4 置换前后 Stage 1（base_va/resid/off_tr/off_va）逐位一致；T5 M0all（默认）与 M1_over_all Stage-1 基线逐位一致，旧 protect=True 版预测差 0.0674；T6 阶梯 5 步，第 5 步 controls=log_pick、subset=forty.notna()。
- 第 5 轮本地冒烟 r6（2026-11-22 约 13:28–14:21，总耗时 52 分 50 秒，CHAIN_EXIT=0，SMOKE-CHAIN-DONE）：链 hidden_files/nfl-smoke-r6-chain.sh（只跑 model_phase3 --quick，前三步沿用 r5 输出），前台跑。默认路径与 r5 逐位一致的项——Y2 M1a 0.1203、M1 0.1105、M1_track_only 0.0981、Δ M1_over_stopwatch +0.1019 CI [+0.036, +0.181]、Δ M1_track_only +0.1269、置换 p=0.0294；对抗 ALL 0.7211 / FORTY 0.6646 / SKILL 0.6473 / STATIC 0.5760；M3 Y2 r=0.1483。Bug B 验证：Y2 M0all 0.1960 ≥ M0c_ngs 0.1948，alpha_med 由 266 降至 146.8；delta_M1_over_all_vs_M0all = +0.0009 CI [−0.0101, +0.0109]。Bug C 验证：ktr_speed WR 0.8075；Y2_sensitivity n_plays_100 n=286 r=0.1669、n_plays_150 n=261 r=0.0907。阶梯第 5 步：forty.notna() 子集（n=253）上只控 log_pick，β=0.1238、p=0.0041（仍显著）；第 1 步（n=299）β=0.149 → 第 5 步 β=0.124（样本缩水代价 −0.025）→ 第 2 步（同子集加 forty）β=0.078、p=0.117（共线性代价 −0.046）：衰减主因是 forty 共线性（约 2/3），样本缩水约占 1/3。--within-year 探针（约 29 分钟）：Y2 M1 0.1205、M1a 0.1282、Δ M1_over_stopwatch +0.0935 CI [+0.0253, +0.1669]、Δ M1 vs M0a +0.0465 CI [+0.0013, +0.0917]、置换 p=0.0294；stage2 M0b_stopwatch 0.4761、M1 0.6613；对抗 0.4909/0.4924/0.4917/0.4879；M3 Y2 r=0.1146。Bug A 验证：LOCO_Y2 M1 2023 0.1835（r5-wy 0.1545，+0.029）、2024 0.1317、2025 0.0522；2025 时序 hurdle M1 0.8669、logsnaps 0.6234（±0.004 噪声级）。Bug B（wy）：Y2 M0all 0.1816（r5-wy 0.1522，+0.029）；delta_M1_over_all_vs_M0all = +0.0052 CI [−0.0093, +0.0198]。Gemini 3 项核对冒烟版三项均绿灯——冒烟通过。
- 服务器第 5 轮全量（2026-11-22 14:25 派发，浏览器任务 browser-task:80aa9d73-7588-51db-b554-af980ecc86bc，2h56m 后按决定结束；收数任务 browser-task:5626145c-a9d2-592c-8ad5-3c17b11f477b，实测 4h26m）：备份 prefix_snapshot_r4/（9 文件 cmp 全过）；model_phase3.py 上传 md5 核对一致；前三步全部完成（退出码 0）：features_combine stable drills 31、targets y2 covered 299、game_kinematics 401 人 / 10,424,763 帧。第 4 步 model_phase3 --within-year 15:12:33 启动（PID 79903），约 20:58:55 正常退出，**耗时 5h46m22s（第 4 轮 7h14m，提速约 20%）；30–45 分钟预估未兑现**（首个结果章节约 88 分钟才打印）。日志 6,910 行，error/traceback 0 条；结果文件时间戳均更新为 20:58:55。完整结果见 notes/phase3_results_round5_server.md（946 行）与 notes/phase3_results_round5_server.json（22 顶层键）；数字由脚本 notes/extract_round5_numbers.py 从 JSON 生成（输出 notes/extract_round5_numbers_output.txt，193 行）。
- Gemini 第 5 轮（已派发并回传，见文末“第 5 轮核对完成”一节）：提纲 notes/gemini_round5_draft.md 已填服务器实数；浏览器任务 browser-task:74bf503f-31da-5a04-b22d-87be33986ddb，实测 19 分钟；回复全文存 notes/gemini_round5_20261122.md。

## 第 5 轮核对完成：Phase 1–3 正式冻结（2026-11-22 22:36 回传）

- 输入：本轮改后 5 个 src 文件（model_phase3.py 为唯一改动文件）+ data_profile.md + phase3_results_round5_server.md（服务器全量全文，已填）+ gemini_round5_draft.md（6 个议题，数字已由脚本从服务器 phase3_results.json 生成）。模型 Neon、思考等级“高”。同一对话 https://gemini.google.com/app/8adf0ca845f706e2 。回复全文存 notes/gemini_round5_20261122.md。
- **总裁决：Phase 1–3 全链路正式冻结，无需第 6 轮重跑，即刻转入 Phase 4 出图与 Phase 5 写作。** 代码级复核：Bug A/B/C 与 Stage-1 缓存改造均已准确落地且零副作用；3 项验收全部通过；全流水线零目标泄漏、零跨年传感漂移（对抗验证四分块 0.488–0.492）、零口径矛盾。
- 代码复核要点：① Bug A：三级回退链正确；标准 CV 不受影响的模型与第 4 轮逐位一致；LOCO 2023 M1 0.1537→0.1831。细微注记（不阻塞）：极小训练组全体缺失时理论上可能触达验证折中位数，实测未触发，无需修改。② Bug B：6 处调用点仅分类 Stage 2 rpipe 显式 protect=True；M0all 0.1629→0.1829，delta −0.0006 干净归零。③ Bug C：ktr_speed 已统一 forty_s_peak 口径（WR 0.8151 与 9.0294/11.0794=0.8150 吻合）；Y2_sensitivity 新门槛；experiments.csv 日期 2026-11-22；per_position_spearman 已入 md；阶梯第 5 步已加且 res 锁定 ladder[3]。④ 提速缓存：置换列与基线列互斥硬断言已加；向量化与原逻辑算术等价；缓存为无损记忆化，零统计口径风险。
- Gemini 指出的两处回传转写瑕疵（不影响 JSON 与冻结）：① phase3_results_round5_server.md 中 Y1_year1_hurdle 的 M0b alpha 印作 393（JSON 为 0.3931）；② targets 摘录中 gated:DB::m_getoff 实为 DL::m_getoff——**已核实属实**：src/targets.py 第 245–249 行 m_getoff 仅定义于 DL；本地结果文件已更正为 gated:DL::m_getoff（值 8.568 不变，与第 4 轮一致）。铁律：Phase 4 出图与制表一律直接读冻结 JSON，严禁从 Markdown 转录取数。
- 6 个议题裁决要点：① 3 项验收全绿——LOCO 三届 M1 0.1831/0.1315/0.0513（2 届跑赢 M0a）；M0all 0.1829 未超 M0c_ngs 0.1984 属正常（L2 稀释 ~0.015，不阻塞）；WR 保持率 0.8151，五位置梯度 WR 0.815 > TE 0.755 > DB 0.733 > DL 0.624 > OL 0.517。② Y2 headline 三层递进：Headline 1 超越秒表 Δ=+0.0771 CI 不含 0；Headline 2 顺位校正 M1a/M1 Δ=+0.0394/+0.0326（置换 p=0.003 为条件信号补充，CI 略跨 0）；Headline 3 诚实边界 M1_over_all vs M0all Δ=−0.0006（必须引用）。③ Double Dissociation 稳固：Y1 顺位包揽方差、追踪零增量；Y2 顺位断崖衰减、追踪条件信号显著；Limitations 写法已给（2025 届 n=83、抽样标准误 ≈0.11）。④ 阶梯第 5 步分解可写入正文：样本缩水 −0.0201（32.5%）vs forty 共线性 −0.0417（67.5%），定稿措辞已给微调版。⑤ 提速未达预期：瓶颈在主 CV 重跑与置换的 pandas 开销；绝不为此破冻重跑；缓存只在附录 Reproducibility 一句带过。⑥ 冻结后 6 条数字口径铁律：单一真相源为冻结 JSON；Fig 4 双列森林图口径；Fig 5 4×4 热力图 + 保持率 + DL 亚组；per_position_spearman 异质性写法；Y2_sensitivity 门槛标注 2×5 CV；Shadow 统一命名 Skill-Drill Movement Index、M3 克制表述。
- 冻结范围：Phase 1 ETL、Phase 2 Combine 特征（含 combine_features.parquet 与 Shadow 标定）、Phase 3a 目标表、Phase 3b 实战运动学、Phase 3c 建模结果（phase3_results_round5_server.json、oof_predictions.parquet）全部永久冻结；后续写作与作图严禁再改模型超参或特征定义。
