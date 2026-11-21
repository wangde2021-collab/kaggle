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

## 第 2 轮：（待定）

## 第 3 轮：（待定）

## 第 4 轮：（待定）

## 第 5 轮：（待定）

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
