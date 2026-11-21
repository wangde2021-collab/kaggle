# NFL Big Data Bowl 2027 — Gemini 最高模型获奖方案（全文存档）

- 来源：Gemini 网页版，模型“更深入的分析 4 Neon”、思考等级“高”，账号 wangde2021@gmail.com（Google AI Pro）
- 对话：https://gemini.google.com/app/f42509acc9d07f4b
- 时间：2026-11-21 02:14–02:29（北京时间）生成完毕，未提示额度耗尽
- 输入：附件 problem_statement.md + data_profile.md + rules.md（均成功上传），请求原文照发
- 注意：浏览器回传时“原文中的公式符号按原意用文字/符号转写，内容、数字、口径未改写、未摘要；界面上的附件引用标记已略去”

---

完整核对三份附件后，先亮出决定本题胜负的 4 个底层数据硬发现（大多数参赛者会踩坑，而这四点恰恰是冲刺 University Track Finalist 与 Grand Prize 的突破口）：

1. 85.1% 的体测追踪帧在“位置专项技术训练（Skill Drills）”中，且覆盖全部 510 人：尽管 three_cone 缺失 64.7%、short_shuttle 缺失 60.6%、bench_reps 缺失 62.4%，但 combine_tracking.csv 拥有全部 510 名球员的 10Hz 传感器数据（6,310 次 drill、463,189 帧），其中 6 类 SKILL_DRILLS_* 合计 394,048 帧，占比高达 85.1%。这意味着你完全可以绕开“跳测黑洞”，从场上专项动作中重建隐性敏捷度。

2. 数据集高度集中在 5 个非持球核心位置，几乎没有 QB/RB/传统中卫：combine_position 仅有 5 类；按 nfl_position 统计，WR（106）、DB（CB 60 + SS 28 + FS 25 + DB 5 = 118）、OL（G 51 + T 50 + C 20 = 121）、DL/EDGE（DT 47 + DE 36 + OLB 30 + NT 5 = 118）和 TE（42）合计占 505/510（99.0%），而 RB/FB/ILB/MLB 合计仅 5 人、QB 为 0 人。方案必须围绕接球手（WR/TE）、锋线对抗（OL vs. DL/EDGE）、二线防守（DB）三大战区设计。

3. 追踪数据存在三个隐蔽口径陷阱，直接聚合必然失真：
- combine_tracking.csv 的 entity_type 同时包含 PLAYER 和 BALL，表中最大速度 s = 25.79 yd/s、最大加速度 a = 44.61 yd/s²、单帧位移 dis = 45.29 yd 极可能混入了传球飞行帧与坐标跳变。
- games.csv 包含 147 场季前赛（PRE）和 39 场季后赛（POST），常规赛（REG）为 816 场；game_tracking 2023–2025 均从 7 月底/8 月初季前赛起记，且 player_play 含 497 名球员，远多于常规/季后赛有攻防档的 405 人（510 − 105）。题目明确聚焦常规赛表现（regular-season NFL game performance），必须按 games.csv 严格隔离 PRE。
- game_tracking_2023 与 2024 的加速度 a 最小值为 0.0000（标量模长），而 game_tracking_2025 的 a 最小值为 -14.3960（出现有符号值或异常负值），跨赛季直接使用原始 a 列会制造严重的年份口径偏移。
- game_tracking 仅追踪 510 名新秀队列自身，不含场上全部 22 人：说明文档明确其追踪对象为 510 名新秀队列。因此 game_tracking 的正确用法是提取自我中心（egocentric）的 10Hz 生物力学与运动学特征，而对抗交互指标（分离距离、施压概率、冲传启动时间等）必须来自 player_play.csv。

## 1. 研究问题定义（直击“Uncover non-obvious linkages”）

NFL 评审团（清一色 NFL Data Science 负责人与资深数据科学家）最反感两件事：一是重复“跑得快=顺位高”的常识，二是只丢一堆 ML 黑箱特征不讲橄榄球机理。以下问题设计直击当前 NFL 球探体系最大的痛点——顶级新秀大面积跳过敏捷测试，且球队在选秀中过度为直线极速溢价。

主研究问题（Main RQ）：
Beyond the Stopwatch and Opt-Outs: Do 10Hz Kinematic Signatures from Combine Skill Drills—Specifically Deceleration, Curvilinear Speed Retention, and Acceleration Kinetics—Predict Exposure-Adjusted NFL Performance and Reveal Systematic Draft-Capital Mispricing?
（超越秒表与跳测黑洞：来自 Combine 专项训练的 10Hz 生物力学特征——尤其是减速制动能力、曲线变向保速效率与加速动力学——能否在控制选秀顺位和 NFL NGS 评分后，增量预测常规赛场上产出，并揭示 NFL 选秀市场系统性的“错误定价”？）

子问题 1（Shadow Agility — 破解 60%+ 跳测难题）：对跳过 3-cone（64.7% 缺失）和 short shuttle（60.6% 缺失）的新秀，能否利用全员覆盖的 SKILL_DRILLS_* 10Hz 追踪轨迹构建“影子敏捷指数（Shadow Agility Index）”？该指数在非跳测群体上对真实 3-cone/shuttle 的解释度有多高，并能否泛化预测跳测球员的 NFL 常规赛表现？

子问题 2（Draft-Capital Arbitrage — 直线极速已被定价，减速与变向效率存在 Alpha）：在对比“预测选秀顺位（市场愿意付什么钱）”与“预测控制顺位后的常规赛产出（球场上什么真正管用）”时，峰值直线速度（40 码极速）是否已被选秀顺位完全消化甚至过度定价，而高初速制动减速力、变向出弯再加速效率、40 码加速时间常数 τ 是否携带显著的超额选秀价值（Draft-Capital-Adjusted Alpha）？

子问题 3（Combine-to-Pads Kinematic Translation — 脱盔体测到穿垫实战的运动学迁移率）：将同一套 10Hz 运动学算子同时施加于 combine_tracking 与常规赛 game_tracking，不同位置（WR/TE、OL、DL/EDGE、DB）中哪些体测运动学能力能真实“迁移”到穿护具对抗的常规赛中（Kinematic Translation Ratio, KTR），并经由实战运动学中介转化为位置专属的 NGS 攻防质量（如接球分离度、YACOE、快速施压率、允许施压概率）？

## 2. 目标变量定义（破解零膨胀与 2023–2025 暴露时长不均）

核心约束：2023 届球员最多经历了 3 个赛季（51 场常规赛），2024 届最多 2 个赛季（34 场），2025 届仅 1 个赛季（17 场）。直接使用 career_offensive_snaps 或 career_games_started 会导致严重的暴露时长混杂（右删失偏差）；且 pro_bowl_original_ballot > 0 仅 11 人、ap_all_pro_1st_team > 0 仅 6 人，完全不具备统计建模功效，只能用于案例佐证。

候选目标变量对比：

Y1（推荐主目标）：两阶段暴露校正的常规赛攻防档产出（Two-Part Exposure-Adjusted REG Scrimmage Snap Rate）。先用 player_play 关联 games.csv 过滤 season_type == 'REG' 与 play_nullified_by_penalty == 'N'，统计每名球员的常规赛非特勤组总档数 REG_Snaps_i；计算可参赛赛季数 E_i = 2026 − draft_year_i ∈ {1,2,3}，定义年均常规赛攻防档 ASR_i = REG_Snaps_i / E_i（并做仅限新秀第 1 季 Year1_REG_Snaps 平行口径）。同时与 player_career_successes 的 (off+def snaps)/E_i 交叉核验。优点：1. 精准对齐“常规赛表现”，剔除 147 场季前赛污染；2. 对轮换型 DL/SLOT WR/Nickel DB 公平（他们不挂“首发”但打 45% 档数）；3. 消除 2023–2025 届赛季数差异。缺点：1. 受高顺位“沉没成本”影响（首轮水货早期也能拿到档数）；2. 仍有约 21% 的零值或极低值（生涯攻防档为 0 者 105 人）。零膨胀/删失处理：两阶段栅栏模型（Two-Part Hurdle）：Stage 1（Logistic）：跨过“NFL 轮换门槛”（ASR_i ≥ 150 档/季，或新秀季 ≥ 100 档）的概率；Stage 2（Gamma/Log-Linear 或 Tweedie 复合泊松）：在达标球员中建模 log(ASR_i)；并做控制选秀顺位残差（Draft-Residual ASR）。

Y2（推荐联合主目标 / 质量维）：经验贝叶斯收缩的位置专属单档质量综合分（EB-Shrunk Play-Quality Z-Score, Z_qual）。在 player_play（REG）中按位置提取高质量单档效率指标（见第 3 节 c 组），在位置内标准化后按权重合成，并对低档数球员做 Empirical Bayes（James-Stein）向位置均值收缩：θ̂_i^EB = [n_i/(n_i+κ_p)]·ȳ_i + [κ_p/(n_i+κ_p)]·μ_p（κ_p 为位置级组内/组间方差比）。优点：1. 衡量“打得好不好”而非“有没有上场”，直接剥离高顺位强行首发造成的假阳性；2. 与 Combine 运动学能力存在最直接的生物力学因果链路。缺点：1. 存在选择偏差（仅有足够常规赛档数者才可观测，设主门槛 ≥ 75 位置相关档，约覆盖 310–350 人）；2. 跨位置综合需依赖标准化。处理：1. 用 IPW（逆概率加权，基于 Stage 1 出场倾向得分）修正上场选择偏差；2. 用 EB 收缩消除 20 档小样本极值噪声；3. 敏感性测试覆盖 50 / 75 / 150 档门槛。

Y3（辅助稳健性目标）：常规赛首发率（REG Start Rate）。直接取 player_career_successes.career_games_started 除以可打常规赛总场数 17 × E_i，取值 [0,1]。优点：直接来自官方汇总表、无任何聚合误差、对球探和高管最直观。缺点：40.8% 为严格 0（208/510）；离散跳跃大；严重低估轮换冲传手与槽接手。处理：Zero-Inflated Beta 回归或 Hurdle Logistic + Fractional Logit，仅作稳健性检验，不作为唯一主目标。

主目标最终推荐：采用“量（Y1）+ 质（Y2）”双核目标架构。用 Y1（暴露校正的两阶段常规赛攻防档 + 新秀第 1 季档数）回答“谁能在 NFL 立足并赢得上场时间”，用 Y2（EB 收缩的位置单档质量分）回答“谁真正在场上打出高效率”。二者在“选秀顺位控制后”的一致发现，会让研究质量（40 分）极具说服力。

## 3. 特征工程（位置分组、物理清洗与三组特征明细）

前置：统一位置分组逻辑。按 data_profile.md 的分布，构建 5 大主战区（覆盖 505/510 人，占 99.0%），并在分析中明确标注其余 5 名边缘位置球员（ILB: 2、MLB: 1、RB: 1、FB: 1）仅纳入全局描述统计、不进入位置分层的推断模型（同时验证其 combine_position 归属）：
- WR（N=106）
- TE（N=42）——在接球路线分析中可与 WR 合并为 Pass Catchers（N=148）并加入位置交互项
- OL（N=121）：细分外侧截锋 OT（T=50）与内侧锋线 IOL（G=51, C=20）
- DL/EDGE（N=118）：细分外侧冲传 EDGE（DE=36, OLB=30；需在 EDA 核验 OLB 的 combine_position 归属）与内侧防守锋线 IDL（DT=47, NT=5）
- DB（N=118）：细分角卫 CB（60）与安全卫 SAF（SS=28, FS=25；外加 DB=5 按 lined_up_position 主位置归类）

### (a) combine_results 静态体测特征与跳测（Opt-Out）处理

1. 核心派生特征（全部按 combine_position 计算组内 Z-score，消除跨位置体型共线性）：
- 体型与杠杆比：bmi = 703 × weight / height²；arm_to_height = arm_length / height；ape_index = wing_span − height（OL/DL/DB 臂展对抗核心指标）；hand_to_height。
- 体重校正爆发力与速度：Speed Score：SpeedScore = combine_weight × 200 / forty⁴（经典 NFL 体重校正冲刺效率）；Flying-30 Split：forty − ten_yd_split（分离后程极速与前 10 码启动）；Start-to-Finish Ratio：ten_yd_split / forty（启动占比，越低代表起跑越炸裂）；Lower-Body Burst Composite：位置内 vertical Z-score 与 broad_jump Z-score 的等权均值（衡量垂直/水平下肢瞬时功率）。

2. Opt-Out（跳测缺失）处理策略（严谨区分 MNAR 机制）：three_cone（64.7%）、short_shuttle（60.6%）、bench_reps（62.4%）和 forty（16.9%）的缺失并非随机，而是顶级前景的经纪人策略性保身（MNAR）。
- 绝不直接把“跳测指示变量”混入“纯运动能力模型”：如果把 opted_out_agility 当作特征去预测 NFL 首发，模型只是学到了“高顺位新秀爱跳测 → 高顺位新秀容易首发”的伪关联（信息泄漏）。
- 正确三步法：① 构建专门的“跳测选择偏差诊断模块”：用 opt_out_count、opted_out_forty、opted_out_agility 对 draft_overall_pick 和 ngs_final_score 做相关性与逻辑回归，量化证明 MNAR 机制（写进论文第 2 节，加分项）。② 敏捷测试缺失用“追踪替代”而非纯插补：主模型放弃直接依赖缺失 60%+ 的 three_cone/short_shuttle，改用 100% 覆盖的 combine_tracking 提取的 Shadow Agility；仅在静态基线对照模型中，于每个 CV 训练集折内部使用位置分层 IterativeImputer（贝叶斯岭插补）并附缺失敏感性报告。③ 把 opted_out 作为分层调节变量（Moderator）：检验“高顺位跳测群体内部，谁的 Skill-Drill 追踪指标能识别出真正的水货 vs. 巨星”。

### (b) combine_tracking 10Hz 追踪聚合特征（核心创新引擎）

Step 0：强制物理清洗与统一运动学求导
- 严格过滤 entity_type == 'PLAYER'，彻底剔除 BALL 帧（消除球速 25+ yd/s 对球员速度分布的污染）。
- 按 (event_id, time) 排序，计算帧间时间差 Δt；剔除或线性插值孤立 1–2 帧传感器跳变（判定阈值：单帧位移 dis > 1.30 yd 即瞬时速度 > 13 yd/s，或 s > 12.5 yd/s）；若单个 event_id 异常帧占比 > 5% 或有效帧 < 10 帧，整次 attempt 剔除并记录剔除日志。
- 对连续轨迹 (x, y, s) 施加 Savitzky-Golay 滤波（window_length=5, polyorder=2），在避免相位偏移的前提下抑制 10Hz 光学抖动。
- 重算统一有符号运动学原语（不依赖原始无符号 a 列）：有符号切向加速度 a_t(t) = ds/dt（a_t > 0 为加速爆发，a_t < 0 为减速制动）；角速度 ω(t) = wrap(Δdir)/Δt（弧度/秒，处理 0°/360° 跳变）；法向/向心加速度（横向变向载荷）a_n(t) = s(t)·|ω(t)|；急动度（Jerk，爆发启动速率）j(t) = da_t/dt；体重加权机械功率与制动力代理：P_acc(t) = m·s(t)·max(a_t(t),0)，F_brake(t) = m·|min(a_t(t),0)|（m 为公斤体重）。

Step 1：40 码冲刺（FORTY_YARD_DASH）速度曲线动力学分解。对每次 40 码有效尝试，截取从起跑时刻（s 首次连续 3 帧 > 0.5 yd/s 且累计位移达 40 码或速度达峰后回落前）的 10Hz 速度曲线，拟合生物力学经典 Furusawa-Hill / Samozino 指数冲刺模型：v(t) = v_max·(1 − e^(−t/τ))。提取特征：v_max：渐近峰值速度（yd/s）；τ：加速时间常数（秒；τ 越小，达到 63% 极速越快，直接区分“前 15 码爆发型”与“后程起速型”）；a_0 = v_max/τ：理论最大水平起跑加速度；split_0_5yd、split_5_10yd、split_10_20yd（官方 combine_results 未提供的超细分段）；末段速度保持率（Speed Maintenance）：30–40 码平均速度 / v_max（识别耐速能力）；曲线拟合优度 R²_sprint（作为数据质量门控，过滤异常轨迹）。

Step 2：位置专项训练（SKILL_DRILLS_*，56 种 drill_name，85.1% 帧，510 人全覆盖）。先裁去每次 event_id 首尾的静止等待帧（保留有效运动窗口 s ≥ 0.5 yd/s），按以下 5 维提取每次 attempt 的特征：
- 减速制动能力（Deceleration Profile）：decel_p95：切向负加速度幅值 |a_t| 的 p95（a_t < 0 帧的 95 分位）；high_speed_brake_power：当 s > 5.0 yd/s 时的最大制动力 m·|min(a_t,0)|（OL/DL 降档至 s > 3.5 yd/s）；braking_impulse_max：单次连续减速段的最大速度降幅 Δs_brake = ∫(−a_t)dt。
- 变向切入与曲线保速（Curvilinear / Cut Efficiency — Shadow Agility 核心）：定义变向事件（Cut Event）：连续帧中 |ω(t)| > 90°/s 且当时 s(t) > 2.5 yd/s 的局部角速度峰值（OL/DL 放宽至 s > 1.8 yd/s）。cut_speed_retention：变向最低点速度 s_apex 与入弯前 0.3s 速度 s_entry 之比；lat_accel_p95：法向向心加速度 a_n(t) 的 95 分位（衡量高速过弯抗离心力能力）；reaccel_slope：变向最低点后 0.5s 内的平均正切向加速度（“出弯二次爆发力”）。
- 爆发启动与急动度（Burst Kinetics）：accel_p95（正 a_t 的 95 分位）、jerk_p95（启动前 1.0s 的最大急动度）、peak_mech_power（体重加权最大加速功率）。
- 路径效率与稳定性（Movement Economy & Consistency）：重复稳定性（Attempt Consistency）：同一 drill_name 多次 attempt 间峰值速度/减速的变异系数（CV，衡量技术动作可重复性）；Drill-Normalized Aggregation（核心聚合口径）：56 种 drill_name 的动作结构各异，必须先在同一 drill_name 内部对每个指标计算 Z-score（基于样本数 ≥ 20 人的标准 drill，严格在 CV 训练集内拟合均值/标准差），再按球员汇总为跨 drill 均值与最佳表现（如 top2_drill_mean），确保参加不同子集专项训练的球员完全可比。
- 地面真值校准（THREE_CONE_DRILL / SHORT_SHUTTLE）：在未跳测的 ~180–200 名球员中，同样提取变向保速与制动指标，用来标定和验证 SKILL_DRILLS 合成的 Shadow Agility Index。

### (c) game_tracking + player_play 常规赛表现聚合特征（仅作目标变量 Y2 或中介变量 M，严禁直接作 X 预测生涯成功）

严格前置过滤：关联 games.csv 仅保留 season_type == 'REG'（剔除 147 场 PRE，POST 留作敏感性分析）；剔除 play_nullified_by_penalty == 'Y'；在 game_tracking 中裁剪每档有效窗口（ball_snap 至 tackle / pass_outcome_caught / pass_outcome_incomplete / out_of_bounds / touchdown；缺失终结事件则截断至 ball_snap 后 6.0s），同样用物理阈值清洗（dis <= 1.3 yd）并从平滑 (x, y, s, dir) 重算统一的 a_t, a_n, ω，彻底规避 2025 赛季 a 负值口径断层。

1. 通用常规赛实战运动学特征（来自 game_tracking，用于构建 KTR 迁移矩阵）：game_s_p99（实战 99 分位极速，规避单帧毛刺）、game_accel_p95、game_decel_p95、game_lat_accel_p95；snap_burst_1s：ball_snap 后 1.0 秒内的最大速度增量与位移；orient_dir_gap：运动窗口内身体朝向与位移方向夹角均值 |wrap(o−dir)|（DB 倒退转髋与 OL 镜面滑步的核心生物力学特征，注意 o 仅在 game_tracking 存在）；运动学迁移率（KTR）：例如 KTR_speed = game_s_p99 / v_max,combine，KTR_decel = game_decel_p95 / combine_decel_p95。

2. 位置专属常规赛单档质量特征（来自 player_play + game_tracking 互补）：
- WR（106）& TE（42）：分母过滤口径（REG & 非犯规取消）为 route_ran.notna()（跑动路线档；目标/接球指标分别按 target==True 或 rec_yards.notna()）。核心单档表现指标（均做位置内标准化 + EB 收缩）：① 情境校正分离度：separation_at_pass_forward 残差均值；② 锐角路线分离度：在 OUT / SLANT / HITCH / CROSS 变向路线上的分离度（直接对应制动与 Shadow Agility！）；③ YACOE：yards_after_catch − expected_yards_after_catch 均值；④ 单路线被瞄准率（TPRR）与单路线接球码数（YPRR）。情境控制项（残差化基准）：控制 route_ran、team_coverage_man_zone、cushion（防区软防给的虚高分离度必须扣掉）。
- DL / EDGE（118）：分母为传球防守冲传档（player_get_off.notna() 或防守端 pass_result.notna() 且非 coverage_assignment 覆盖档）。核心指标：① 中位起跑响应（Get-Off）：−median(player_get_off)（越短越好，直接对应 40 码 τ 与启动急动度）；② 净快速施压率：quick_pressure == 1 且 unblocked_pressure != 1 的比率（剔除漏防捡漏！）；③ 总施压率：time_to_pressure.notna() 占比与 −median(time_to_pressure)；④ 单档防跑破坏率：(tackle_for_loss + sack) / def_snaps。控制：区分 EDGE（DE/OLB）与 IDL（DT/NT），控制是否 blitzing。
- OL（121：OT 50 / IOL 71）：分母为传球保护档（dropback_duration.notna() 且 pass_rushers_encountered >= 1）。核心指标：① 负向峰值允许施压概率：−mean(peak_pressure_probability_allowed)（极佳连续指标，每档传球保护都有值，方差远小于稀疏的 sack_allowed！）；② 允许施压率：1 − mean(pressure_allowed)；③ 2.5 秒内快速溃败率：time_to_pressure_allowed < 2.5 占比；④ 传保滑步镜面稳定性（game_tracking 传球保护前 2.0s 内横向加速度与 |o − dir| 控制）。控制：控制 dropback_duration（四分卫持球越久 OL 越容易丢压）与 pass_rushers_encountered。
- DB（118：CB 60 / SAF 53）：分母为防守盯人/区域覆盖档（coverage_assignment.notna() 且非 NO_ASSIGNMENT）。核心指标：① 出球后闭合爆发力（Closing Burst，来自 game_tracking）：pass_forward 到 pass_arrived/tackle 窗口内的峰值加速度与最高速度（精准弥补 player_play 中 DB 覆盖靶向指标较少的短板）；② 覆盖档防守 EPA 贡献：−mean(expected_points_added)（按人盯人/区域校正）；③ 单档擒抱参与率与单人擒抱比：(tackle + 0.5*assist) / def_snaps 与 tackle_for_loss（偏安全卫）。控制：控制 team_coverage_man_zone、coverage_assignment（区分 MAN 与 DEEP_THIRD/FLAT）。

## 4. 模型路线（统计推断为主、非线性对照为辅）

在 N=510、评审团为资深数据科学家的赛道里，“黑箱 GBDT 刷一个虚高 R²”拿不到高分；真正能拿 40 分研究质量的是层层递进的基准对照 + 严格的效应量与不确定性量化。

基准对照体系（必须显式报告增量价值）：
- M0a（市场基线 — Draft Capital）：仅含 log(draft_overall_pick)、is_udfa、combine_position、draft_year。衡量 NFL 球探与选秀市场已经知道的信息。
- M0b（官方体测评分基线 — NGS Scores）：加入 ngs_athleticism_score、ngs_final_score（或仅用 combine_results 静态秒表成绩）。衡量“没有 10Hz 传感器追踪时”官方现有工具的上限。

三大核心模型路线：

模型 1（主力推断模型）：两阶段分层广义线性模型 / 惩罚回归（Two-Part Hurdle Hierarchical GLM + Elastic-Net）。回答问题：在控制选秀顺位（M0a）和传统静态体测（M0b）之后，哪些 10Hz 运动学特征对常规赛出场（Y1）与单档质量（Y2）具有独立、稳健的增量效应？哪些特征被选秀市场高估/低估？实现细节：Y1（量）：Stage 1 用 L2 正则化/Firth 偏倚校正 Logistic 回归预测达标门槛（防止小样本完全分离）；Stage 2 在达标子样本上用带位置随机截距与位置交互项的混合效应线性模型（LMM，对 log(ASR)）或 Tweedie GLM（直接对零膨胀非负连续值建联合模型作交叉验证）。Y2（质）：用 IPW 加权岭回归 / 分层贝叶斯线性回归（PyMC 弱信息先验 N(0, 0.5²) 对位置级斜率做部分池化 Partial Pooling）——这是解决“单位置样本仅 42–121 人、单独拟合方差太大、完全池化又掩盖位置差异”的教科书级方案！效应量与不确定性报告：所有连续特征均为位置内标准化 Z-score，效应量统一报告为每 +1 SD 对应的结果变化（如年均常规赛档数变化百分比、单档质量 Z 分变化、分离度增加码数、施压率提升百分点），同时报告增量指标 ΔR²_OOF、ΔAUC_OOF、似然比检验（LRT）与 AIC/BIC 改善。置信区间：采用分层自助法（Stratified BCa Bootstrap，B=2,000，按 combine_position × draft_year 分层重抽样）给出 95% 置信区间，不依赖正态大样本近似。多重检验校正：将预注册检验的运动学核心特征族统一施加 Benjamini-Hochberg FDR 校正，同时报告原始 p 值与校正后 q 值（主结论仅声明 q < 0.05 或 q < 0.10 的发现，绝不挑拣未校正 p = 0.04 的假阳性）。

模型 2（隐性因子与中介路径模型）：正则化偏最小二乘 / 岭回归校准（Shadow Agility Bridge）+ 因果中介分解（Mediation Analysis）。回答问题：(1) 如何将多维高共线性的 10Hz 轨迹特征压缩为 4 个可解释的生物力学正交因子（Linear Speed、Explosive Burst、Braking Deceleration、Curvilinear Agility）并用非跳测样本校准 Shadow Agility？(2) Combine 追踪指标对常规赛单档质量的效应中，有多大比例是通过实战 game_tracking 运动学（中介变量 M）传导的（Indirect Effect / Proportion Mediated）？不确定性报告：Bootstrap 估计间接效应（ACME）与直接效应（ADE）的 95% CI，直接量化“体测动作 → 实战跑动 → 场上产出”的传导机制。

模型 3（非线性与交互探测对照）：可解释提升机（EBM, interpretML）或强约束浅层 LightGBM（max_depth=2, min_child_samples=25）+ SHAP。回答问题：体测指标是否存在非线性“淘汰阈值（Cliff Effects）”（例如：OL 横向减速低于某分位即大幅崩盘，但高于中位数后收益递减）或二阶交互（如 weight × decel_p95）？使用纪律：必须在与模型 1 完全一致的嵌套 CV 折上比较；只有当模型 3 的折外指标显著优于模型 1 且形状符合生物力学直觉时，才在报告中展示非线性阈值曲线，否则诚实报告“线性/分层效应已充分捕获信号，复杂树模型在 N=510 下无额外收益”——这种克制反而极大加分。

## 5. 验证方案（N=510 小样本下的可信度护城河）

小样本研究最大的死穴是评审质疑“你在 510 人身上试了 200 个特征凑出了一个相关性”。验证方案必须从数据切分、时间外推、统计证伪、敏感性稳健四层封死质疑：

主交叉验证：重复分层嵌套 CV（10 × 5-Fold Repeated Stratified Nested CV）。分层键：combine_position × 目标分位（零值层 + 正值二分位），确保每折 5 大位置与零值比例完全一致。严格嵌套 Pipeline：所有依赖统计量的步骤——包括 drill_name 内 Z-score 参数、MICE 缺失插补、Shadow Agility 映射权重、弹性网 λ、部分池化超参——100% 封装在 sklearn.pipeline.Pipeline 内，仅用当前训练折拟合。汇总 10 次重复的均值与标准差，消除单次随机切分的运气波动。

时序外推验证（Temporal Extrapolation — 按 draft_year）：统一时间窗口口径（关键！）：为了在 2023、2024、2025 三届之间做公平的跨年外推，构建“仅限新秀第 1 赛季（Year-1 Only）”目标变量（2023 届取 2023 常规赛、2024 届取 2024 常规赛、2025 届取 2025 常规赛），三届暴露长度均为 17 场常规赛！前向滚动验证（Forward Temporal Split）：用 2023 + 2024 届（N=337）训练，在完全未来的 2025 届（N=173）上测试零样本时间外推表现；留一届交叉验证（Leave-One-Cohort-Out, LOCO）：轮流留出 2023 / 2024 / 2025 一整届作测试集，检验核心效应方向和系数符号是否在连续 3 届新秀中稳定一致（如果 3 届符号全同，几乎不可能来自偶然噪声）。

位置分层验证：除全局指标外，强制按 5 大位置（WR、TE、OL、DL/EDGE、DB）分别输出折外相关性 / 增量 R² / 排序指标（Spearman ρ），避免出现“总样本显著但其实是某一个位置拖着走”的假象。

反过拟合与反数据挖掘铁证（Anti-Data-Mining Proofs）：预注册精简特征集（Pre-Specified Hypothesis Lock）：全模型核心检验严格限定在 12–16 个有明确物理定义的运动学指标（严禁跑几百个 tsfresh 特征再挑显著的），并在附录公开全部预注册检验结果（包括不显著的负结果！）。标签置换检验（Permutation Test，K=1,000）：在位置与选秀年份层内随机打乱目标变量，重跑完整拟合流程，生成零假设下的 ΔR²_OOF 分布；要求观测增量满足 p_perm < 0.01。多维敏感性分析矩阵（放附录表）：纳入 vs. 剔除 127 名落选秀（UDFAs）；仅常规赛（REG）vs. 常规赛+季后赛（REG+POST）；新秀第 1 季口径 vs. 全生涯暴露校正口径；单档质量样本门槛取 50 / 75 / 150 档；Cook's Distance 强影响点剔除测试：剔除残差/杠杆值最高的前 2% 球员（约 10 人），证明核心结论并非由个别超级新秀（如 Will Anderson 等）单点撑起。

## 6. 防泄漏检查清单（逐项靶向排查）

1. 生涯累计暴露时长泄漏（2023 vs 2025 届）：发生机理与危害：2023 届有 3 季累计档数，2025 届仅 1 季。若 combine_tracking 中某些 drill_name 仅在特定年份出现，模型会靠“识别年份”偷猜高档数。具体检查方法与防线：① 跑对抗验证（Adversarial Validation）：用体测特征预测 draft_year，检查 AUC 是否显著 > 0.55；② 检查 56 个 drill_name 在 3 年中的覆盖表，剔除仅单年出现的孤立 drill；③ 强制并行汇报 Year-1-Only 目标结果。
2. ngs_* 评分信息污染：ngs_college_production_score 和 ngs_final_score 含大学表现与球探评估，ngs_athleticism_score 已打包体测成绩。混入特征会导致虚假高分并掩盖追踪净价值。检查：在特征配置文件中用白名单硬隔离：X_track 严禁包含任何 ngs_*、draft_*、college_* 字段；ngs_* 仅作为对照基准 M0b 单独运行。
3. 选秀顺位（Post-Combine）混杂与沉没成本：选秀发生在 Combine 之后，球队已看到 40 码成绩并给高顺位新秀“保底首发”机会；若不区分因果时序，会高估已定价体测指标。检查：明确分离两套口径：① 纯选前评估模型（Pre-Draft）（不含顺位，评估球探替代价值）；② 顺位校正模型（Draft-Adjusted Alpha）（将 draft_overall_pick 纳入基线，专测市场未定价的追踪增量）。
4. 跳测缺失（Opt-Out）伪装成体测信号：顶级新秀扎堆跳过 three_cone（64.7%）/short_shuttle（60.6%），缺失指示灯等价于“高顺位标签”。检查：单元测试断言：纯运动学特征矩阵 X_track 中不存在任何 *_isnan 指示列；同时单独输出 opt_out 与 draft_overall_pick 的 Mann-Whitney U 检验表向评审透明展示。
5. 季前赛（PRE）与犯规取消档污染：games.csv 含 147 场 PRE，player_play 含 play_nullified_by_penalty == 'Y'；季前赛替补刷出的高分离度/施压率会严重扭曲常规赛目标。检查：ETL 入口强制硬断言：assert (df['season_type'] == 'REG').all() 与 assert (df['play_nullified_by_penalty'] == 'N').all()，输出过滤前后行数与球员数对账单。
6. 实战追踪反推生涯成功（时空倒置泄漏）：用 game_tracking 算出的比赛速度去预测 career_games_started 属于“用未来结果预测同期结果”，毫无选前预测意义。检查：代码级隔离：combine_* 为唯一合法预测输入 X；game_tracking 与 player_play 仅允许进入目标层 Y 或因果中介层 M（Combine → Game Kinematics → Play Quality）。
7. 跨折标准化 / 缺失插补 / 因子降维泄漏：在划分 CV 前对全体 510 人做 drill_name 内 Z-score、MICE 插补或 PLS 投影，会把验证集分布泄漏给训练集。检查：所有统计变换封装为自定义 BaseEstimator, TransformerMixin，在嵌套 CV 内部 fit(X_train) → transform(X_val)；用单元测试验证打乱验证集数值不会改变训练集转换输出。
8. 跨赛季加速度 a 口径断层泄漏：game_tracking_2025 的 a 最小值为 -14.396，而 2023/2024 最小值为 0.0，直接使用原始 a 列会制造 2025 届特征畸变。检查：彻底弃用原始 a 列，三季追踪统一由平滑后的 (x, y, s, dir) 重算切向加速度 a_t 与法向加速度 a_n；跑三季分布 KS 检验与箱线图叠合核验一致性。

## 7. 算力与内存预算（8 核 32GB 东京服务器）

总数据 2.31 GB CSV，瓶颈全在 2,265 万行的 game_tracking_2023/2024/2025.csv（合计约 1.98 GB）。关键原则：绝对不要做时间降采样（如 10Hz 降为 5Hz）——0.1s 采样间隔已是捕捉 0.7s 冲传启动和急停减速的最低物理分辨率；提速与省内存靠 Parquet 列式存储 + 常规赛/有效帧无损预裁剪 + Polars/DuckDB 分块流式计算。

特别注意 player_play.csv 解析：因 play_description 含 12,823 个内嵌换行符，初次转 Parquet 时必须使用支持 RFC-4180 多行引号的标准解析器（Python csv / pandas.read_csv(..., engine='c') / DuckDB read_csv(..., strict_mode=false, ignore_errors=false) 并校验行数严格等于 314,197），随后在特征表中丢弃 play_description 文本列，体积立即从 140 MB 降至 ~18 MB。

分阶段资源与耗时预算表：
- Stage 1：CSV → Parquet ETL & 数据体检。核心策略：逐年流式扫描 game_tracking_*.csv（每块 100 万行），先 semi-join games.csv(season_type=='REG') 剔除 147 场季前赛，并将 float64 坐标/速度安全转为 float32，按 season 写 Zstd Parquet。峰值内存 2.5–4.0 GB，预计耗时 1.5–2.5 min，磁盘产出 ~380 MB Parquet。
- Stage 2：combine_tracking 物理清洗 + 曲线拟合 + 特征提取。仅 46.3 万行（80 MB），过滤 PLAYER 后全量驻留内存；用 joblib 8 核并行按 6,310 个 event_id 跑 Savitzky-Golay 平滑、Furusawa-Hill 非线性最小二乘拟合（scipy.optimize.curve_fit）及变向事件检测。峰值内存 1.0–1.5 GB，预计耗时 15–30 sec，磁盘产出 ~2 MB 特征表。
- Stage 3：game_tracking 2,265 万帧运动学聚合（最大瓶颈）。两级无损减量：① 仅保留 REG 常规赛；② 按 (game_id, play_id, nfl_id) 截取 ball_snap 至终结事件的有效帧（无损砍掉 ~45–50% 站桩等待帧，剩余约 950–1,100 万帧）。用 Polars scan_parquet 按赛季 + game_id 分批（每批 30 场比赛，8 核并行）向量化计算 a_t, a_n, ω、起跑 1s 爆发及分位数聚合。峰值内存 5.0–7.5 GB，预计耗时 2.5–4.5 min，磁盘产出 ~15 MB 球员-赛季特征表。
- Stage 4：player_play 位置专属单档质量与 EB 收缩。读取 18 MB Parquet，按位置过滤常规赛有效档，用 statsmodels 做情境残差化（路线/防守阵型/持球时长）并计算 EB 收缩得分。峰值内存 < 1.0 GB，预计耗时 10–20 sec，磁盘产出 ~1 MB 目标表。
- Stage 5：全套建模、嵌套 CV、2,000 次 Bootstrap 与 1,000 次置换检验。表格规模仅 510 × ~40！主要算力消耗在 PyMC 分层贝叶斯采样（4 chains × 2,000 draws）+ 1,000 次置换检验（8 核并行）。峰值内存 2.0–3.5 GB，预计耗时 3.0–6.0 min，磁盘产出 ~50 MB 模型产物与图表。
- 端到端全量复现（One-Command Rebuild）：从原始 CSV 到所有统计表与 300 DPI 图表一键重跑，完全满足获奖代码复现审查要求。峰值 < 8 GB（安全余量 24 GB），总计 8–14 min，< 500 MB。

## 8. 分阶段里程碑（倒计时 47 天：2026-11-21 至 2027-01-07）

当前距截稿（2027-01-07 07:59 GMT+8）共 47 天（约 6.5 周），且 Hackathon 全队仅 1 次提交机会，必须预留完整一周用于写作打磨、页数卡控与冷环境复现测试。

- Phase 1：数据体检与无泄漏 ETL 底座。第 1–6 天（11/21–11/26）。核心交付物：1. 清洗后的 Parquet 数据仓与自动化校验脚本；2. 完整 EDA 审计报告（56 个 drill_name 年份分布、5 大位置与 OLB 归属核验、opt_out MNAR 检验表、2025 a 口径核对）。预期结果形式：完整数据质量表、opt_out vs draft_pick Mann-Whitney p 值、各位置常规赛档数分布图。硬性通过/不通过标准（Go / No-Go Gate）：PASS：① player_play 精确 314,197 行；② combine_tracking 剔除 BALL 后 s > 12.5 占比 < 0.1%；③ 三季重算 a_t 分布一致；④ 至少 15+ 个高覆盖 drill_name 跨 3 年稳定存在。不通过则先修清洗规则，严禁进入建模。
- Phase 2：生物力学特征工程与 Shadow Agility 标定。第 7–16 天（11/27–12/06）。交付物：1. 40 码 Furusawa-Hill 曲线拟合模块；2. 56 种 Skill-Drill 减速/变向/爆发特征库；3. Shadow Agility 校准报告；4. 常规赛实战运动学与 Y1/Y2 目标变量表。预期结果形式：40 码曲线拟合 R² 分布；Shadow Agility 对真实 3-cone/shuttle 的折外 r 与 R²；Y1/Y2 与 draft_pick 基线相关性。PASS：① >95% 的 40 码有效跑次曲线拟合 R² > 0.95；② 位置内 Shadow Agility 预测真实 3-cone/shuttle 达到折外 r ≥ 0.40（p < 0.001）；③ game_tracking 与 combine_tracking 极速呈正相关（r > 0.35，证明跨表对齐无误）。若 Shadow Agility 过弱，立即细化 cut 事件角速度与入弯阈值定义。
- Phase 3：主建模、选秀错误定价检验与全维度验证。第 17–28 天（12/07–12/18）。交付物：1. M0a/M0b 基线 vs. M1/M2/M3 完整对比表；2. BCa Bootstrap 95% CI 与 BH-FDR 校正表；3. 2025 时序外推 + LOCO + 1,000 次置换检验报告；4. 5 项敏感性分析附录表。预期结果形式：标准化 β（95% CI）、FDR q 值、折外 ΔR² / ΔAUC、置换 p_perm、中介效应比例（% Mediated）。PASS：① 至少 2 个位置或全局核心追踪指标在控制选秀顺位后满足 FDR q < 0.10（理想 q < 0.05）；② 相对 M0a/M0b 实现折外 ΔR² ≥ 0.02–0.05 或 ΔAUC ≥ 0.03 且 p_perm < 0.01；③ 核心方向在 2025 届 Year-1 外推中保持同号。若某子假设为空结果（如某位置无效），果断将其转化为“运动学迁移边界条件”发现，绝不强行 p-hack。
- Phase 4：出版级可视化与 NFL 实战案例叙事。第 29–37 天（12/19–12/27）。交付物：1. 6 张高信息密度、色盲友好、300 DPI 主图（统一配色与字体规范）；2. 4–6 名真实球员深度案例剖面（同 40 码不同减速/变向轨迹的“低顺位淘金 vs. 高顺位失手”对照）。预期结果形式：矢量 PDF/高分 PNG 图集 + 真实球员姓名与具体数据卡片（均严格来自赛题 510 人数据）。PASS：脱离正文仅看图表标题、副标题与图内标注即可读懂核心结论（10 分可视化满分标准）；每张图数值与 Phase 3 统计表 100% 自动脚本联动一致。
- Phase 5：15 页英文 Writeup、开源复现打包与终审提交。第 38–47 天（12/28–01/05，预留 48h 缓冲）。交付物：1. 严格 ≤ 15 页（目标 13–14 页）英文报告（LaTeX/Quarto 编译，含摘要、正文、方法、结果、NFL 球探落地建议、精简附录）；2. OSI 开源许可（MIT/Apache-2.0）干净代码库 + README + requirements.txt 一键复现脚本。预期结果形式：最终英文 Writeup + 完整可复现代码仓库（不上传任何原始比赛数据）。PASS：① PDF 总页数严格 ≤ 15 页（超页直接 DQ）；② 全新虚拟环境一键跑通且结果小数点后 3 位完全一致；③ 无任何比赛原始数据外传；④ 人工终审签字后于 1 月 5 日前完成唯一一次正式提交。

## 9. 创新点（冲刺 30 分）与核心可视化清单（冲刺 10 分）

3 个硬核创新角度：
- 用 100% 覆盖的 Skill Drills 解决 Combine“跳测危机”（Shadow Agility Framework）：过去 5 年 NFL 球探最头疼的就是精英新秀集体跳过 3-cone 和 shuttle（本数据缺失超 60%）。你不去硬补那两个缺失列，而是直接把镜头转向占体测 85.1% 帧数的场上专项训练，从变向保速、向心加速度和制动冲量中重建无须单测的“隐性敏捷度”——这是 NFL 现场评审（2027 Combine 现场展示）拍案叫绝的实战创新。
- 从“秒表终点值”升级为“连续动力学曲线”（Furusawa-Hill 冲刺分解 + 制动物理学）：用 2 个物理参数（τ 起跑时间常数与 v_max 极速）替代粗糙的 40 码总时间，并把焦点从“加速有多快”反转到橄榄球实战更稀缺的“高速下急停与出弯再加速能力（Deceleration-First Athleticism）”，完美呼应“non-obvious linkages”。
- 跨场景“体测—实战运动学迁移矩阵”与“选秀套利地图（Scouting Arbitrage Map）”：大多数报告只做 Combine → 生涯结果的黑箱映射；你用完全一致的 10Hz 生物力学算子同时解析体测与常规赛，第一次直观量化“穿垫损耗（Pads Tax）”与“选秀市场高估 vs. 低估的能力象限”，给 NFL 总经理一张可以直接带进选秀作战室的决策图。

6 张关键图表清单（每张都直接服务核心论点）：
- Fig 1：The Combine Opt-Out Illusion & Sensor Coverage Map（双联图）。画什么：左图：按选秀轮次（R1–R7 & UDFA）绘制 three_cone/short_shuttle/forty 缺失率阶梯折线图；右图：按 5 大位置堆叠展示 combine_tracking 帧数构成（突出 85.1% 的 SKILL_DRILLS 覆盖 510/510 人）。支撑的核心结论：证明静态敏捷测试存在严重 MNAR 选择偏差，同时证明专项训练 10Hz 传感数据是全覆盖的“隐性金矿”。
- Fig 2：Same 40-Time, Opposite Physics: Anatomy of a Sprint & Cut（双球员 10Hz 轨迹与波形对照图）。画什么：挑选 2 名真实同位置新秀（40 码成绩几乎相同、但常规赛表现一优一劣）：上图展示其 40 码 Furusawa-Hill 速度-时间拟合曲线（标注 τ 与 v_max 差异）；下图展示其在同一专项 Skill Drill 中的 2D 场地轨迹（颜色映射速度 s）及切向加速度 a_t 制动尖峰波形。支撑的核心结论：一图看懂“为什么秒表骗了球探”——极具画面感与橄榄球直觉，奠定全文物理叙事基调。
- Fig 3：Validating Shadow Agility Against Observed Agility Drills（联合回归散点 + 分布叠合图）。画什么：主图：非跳测子样本上，横轴为位置内 Skill-Drill Shadow Agility，纵轴为真实 3-cone/short-shuttle（取负号使越高越好），分位置着色并附 95% 置信带与折外 r；边缘密度图叠合展示“跳测群体”的 Shadow Agility 分布。支撑的核心结论：证明从专项训练提取的影子敏捷度真实捕获了变向能力，且能无偏外推至 60%+ 的跳测精英。
- Fig 4：The NFL Draft Mispricing Forest Plot（核心主图）。画什么：双列森林图（Forest Plot）：纵轴列举核心体测/追踪特征（40 码极速、τ、制动减速力、变向保速、Shadow Agility 等）；每行并排画两条带 95% BCa 置信区间的 standardized β——灰色点表示“预测选秀顺位（市场定价）”，彩色实心点表示“控制顺位后预测 NFL 产出（真实 Alpha）”，标星号指示 FDR q < 0.05。支撑的核心结论：全文最核心结论的视觉锚点：一眼看出哪些指标“灰点大、彩点近零”（已被充分/过度定价），哪些指标“灰点小、彩点显著为正”（选秀市场遗漏的非显性连接）。
- Fig 5：Combine-to-Pads Kinematic Translation Matrix（热力图 + 路径桑基/中介标注）。画什么：5 大位置（行）× 6 项运动学能力（列）的迁移相关系数矩阵（Combine 10Hz → 常规赛 game_tracking 10Hz），格子内注明显著性与穿垫保持率（KTR）；右侧附简洁的中介路径图展示“体测制动/变向 → 实战运动学 → 位置专属单档质量（如 WR 锐角分离度、EDGE 快速施压率）”。支撑的核心结论：揭示“体测动作如何真正转化为球场表现”的机理，证明连接并非数据巧合而是生物力学传导。
- Fig 6：Draft Room Arbitrage Quadrant & 2025 Out-of-Time Lift（决策四象限 + 校准增益图）。画什么：左图：横轴为选秀顺位（高到低），纵轴为模型预测的“运动学 Alpha 分”，按实际 NFL 常规赛产出（Y1/Y2）着气泡大小与颜色，直接点名标注 4–5 位“低顺位高 Alpha 淘金案例”与“高顺位低 Alpha 预警案例”；右图：2025 届纯未来样本上按预测五分位展示实际新秀季档数/质量单调递增柱状图（对比 M0 基线）。支撑的核心结论：同时交付可落地的球探工具与过硬的跨年外推验证，直接服务于 2027 Combine 现场展示的说服力。

## 10. 风险清单与具体应对办法

1. 小样本统计功效不足与多重检验假阳性。严重度：高。具体诱因：总样本仅 510 人，拆到 TE 仅 42 人、WR 106 人；直接做分位置复杂回归极易方差爆炸或 p 值不显著。可执行应对办法：① 主推断采用分层部分池化（Hierarchical Partial Pooling）跨位置借力，避免 TE 等小位置孤立估计；② 严格预注册 12–16 个物理特征、统一做 BH-FDR 校正与 1,000 次置换检验；③ 准备好“双向叙事”：即使某指标在常规赛中不显著，只要它对选秀顺位高度显著，同样构成高价值的“市场过度定价”发现。
2. 目标高度零膨胀与 2025 届右删失偏差。严重度：高。诱因：40.8% 从未首发、20.6% 零攻防档，且 2025 届仅 1 季数据，伤病（如 ACL 赛季报销）也会制造假零值。应对：① 采用暴露校正两阶段模型 + Year-1-Only 平行目标彻底对齐三届样本；② 引入连续单档质量指标 Y2（如 OL 的 peak_pressure_probability_allowed）大幅提升信噪比；③ 核查零档球员的 career_games_active，在敏感性分析中识别并讨论早期伤病/替补池的影响。
3. 传感器噪声、口径断层与数据陷阱。严重度：高。诱因：combine_tracking 含 BALL 帧与 45 码位移毛刺，game_tracking_2025 a 为负值，147 场 PRE 混入，内嵌换行符导致解析错行。应对：在 Phase 1 建立自动化数据体检单元测试套件（强制过滤 PLAYER、REG、非犯规取消、速度/位移物理界限、全三季重算有符号 a_t 与 a_n），把这部分数据治理深度写进报告 Method 章节——NFL 评审对“真正读懂追踪数据脏点”的作品极为青睐。
4. 评审主观性与 15 页硬红线淘汰。严重度：中高。诱因：无客观 LB 分数，评审按 40/30/20/10 主观打分；超 15 页（含标题、正文、附录）直接取消资格；过于偏学术公式而缺乏橄榄球洞察会丢分。应对：① 写作坚持“1 页执行摘要（写给 NFL GM）+ 清晰方法与结果 + 真实球员案例 + 球探落地指南”的混血结构；② 用 LaTeX/Quarto 严控总页数在 13–14 页（留 1 页安全缓冲），导出的最终 PDF 人工逐页核对页码；③ 把冗长敏感性表压缩为紧凑矩阵放附录，确保正文流畅易读。
5. 单次提交不可撤回与合规/复现红线。严重度：高。诱因：Hackathon 每队仅 1 次提交机会；获奖须提供 OSI 开源可复现代码；数据严禁外传与商用。应对：① 绝不在截止当天压哨提交，锁定 2027-01-05 为内部提交日，留 48 小时应对网络/平台异常；② 提交前执行“三项签字核查”：页数 ≤ 15、英文拼写与图表数字全核对、冷环境一键复现通过；③ 全程仅用 OSI 许可开源库（Polars/DuckDB/scikit-learn/statsmodels/PyMC/matplotlib），原始 CSV 严格保留在个人加密环境内不上传任何第三方公开托管服务。

（方案末尾 Gemini 还问：是否要先帮设计一份可在东京 8 核 32GB 服务器上直接跑的 Phase 1 体检指标清单与校验脚本架构——Phase 1 代码已按本方案自行编写，见 src/；该问题留待后续轮次再决定是否追问。）
