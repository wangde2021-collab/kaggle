# Gemini 第 3 轮核对回复全文（2026-11-22 01:13，同一对话，Neon、思考等级“高”，账号 wangde2021@gmail.com）

附件：本轮改后 5 个 src 文件 + data_profile.md + phase3_results_round3_server.md + gemini_round3_draft.md。以下为浏览器任务回传的回复全文原样转录（公式排版略有归一化，数字与结论未改动）。

总裁决：Phase 1 / Phase 2 / Phase 3a / Phase 3b 全部通过并可立即冻结，但 Phase 3c（model_phase3.py）暂不可冻结——逐行核对代码查出 2 处静默高危 Bug，导致本轮 CV 与增量对比中的关键数字失真，必须仅针对 model_phase3.py 做第 4 轮最小修复并重跑一次。

在展开 6 个议题前，先把查出的两个静默 Bug 亮在最前面（它们直接改变了议题 1、2、3、6 的结论）：

静默 Bug 1（致命）：DrillFoldClean 在本轮所有折内建模、LOCO、时序外推与对抗验证中，把全部 9 个 skill_*_z_mean 特征静默清零（全 NaN → 插补为 0.0）。根因在 model_phase3_3.py 的 DrillFoldClean.fit/transform：
1. fit() 用列表键 sub.groupby(["drill_name"])——在 pandas 2.1.4 中，单列列表分组返回的 name 是 1-元组 ('DRILL_A',) 而非字符串，导致存入字典的键是 (('DRILL_A',), None)，而 transform() 查询的是 ('DRILL_A', None)，100% 查不到键，全部退回默认 (np.nan, 1.0)；
2. transform() 写成 self.drill_stats_.get(k, (np.nan, 1.0))[0]，漏了按指标名 [m] 取值（self.drill_stats_[key] 是以指标名为键的字典）——如果开了 --within-year 键恰好匹配，会直接抛 KeyError: 0 崩溃。
结果反证：这解释了为什么 fdr_Y2（读取 Phase 2 预计算列）中 skill_s_peak_z_mean 依然高显著（β=+0.156, q=0.004），而本轮 CV 中 M1a 在 Y2 上却从第 2 轮的 0.1402 跌回 0.0953、LOCO 2025 从 0.1045 跌回 0.0149、SKILL_ONLY 对抗验证跌到 0.4773（实际只测了 1 个标量 shadow_agility_pure）。

静默 Bug 2（致命）：M1_over_stopwatch 混入了选秀顺位 M0A_COLS，导致“超越秒表 +0.1040”是虚假增量。STACKED_MODELS 中 M1_over_stopwatch 的基线列是 M0A_COLS + M0B_STOPWATCH_COLS，却去减不含顺位的 M0b_stopwatch。看 Y2 实测：M1_over_stopwatch 残差层 alpha 顶到上限 1e+05（追踪贡献 ≈0），全部 +0.1040（以及 Y1 的 +0.2653）100% 来自偷混进去的选秀顺位，而非追踪特征。同时，M0all 把 M0B_STOPWATCH_COLS 和 M0C_NGS_COLS 放进了不做位置内标准化、按全局中位数插补的 raw_cols，导致 M0all（0.1747）反而不如 M0c_ngs（0.1926）。

(1) t0 修复与 40 码跨年漂移：是否启用 --within-year？
诊断结论：fit_sprint_curve 加入 t0 和物理上界 s_peak * 1.12 是完全正确的修复——实测极速 forty_s_peak 三年完全平稳（全位置 10.419 / 10.368 / 10.393，WR 11.033 / 11.070 / 11.079），WR v_max_best 虚高也被压回合理区间。但 FORTY_ONLY 对抗验证 AUC 仍为 0.6646 ± 0.0247，说明 40 码追踪轨迹的起跑滤波/加速段斜率（tau_best、a0_mean、speed_maintenance_mean）仍携带 2023–2025 传感管线的年份尺度漂移。
SKILL_ONLY = 0.4773 目前不可采信：因上述 Bug 1，9 个技能特征全为 0，这个数字只测了 shadow_agility_pure。必须先修 Bug 1 再看真实的 SKILL_ONLY 对抗验证 AUC。
裁决：启用同届内调和（--within-year），但必须先修两处代码陷阱：
- 修掉 DrillFoldClean 的 KeyError: 0 与单列分组元组 Bug；
- 修复 LOCO / 2025 时序外推中的“未见年份”断层：目前 DrillFoldClean 和 FeaturePrep 在 by_year=True 时，用 2023+2024 拟合后去转换 2025 测试集，(drill_name, 2025) 与 pos_2025 均不在训练字典中，会导致 2025 测试集技能特征全 NaN、40 码退回跨年全局均值。正确改法：当测试集中出现训练折未见过的整届 draft_year（仅发生于 LOCO 与 2025 时序外推）时，直接用该届自身无标签的 Combine 特征 X 分布做届内标准化——这完全符合真实世界“3 月整届体测结束即可得当届均值/标准差，4 月再选秀”的业务时序，零目标/未来泄漏。
实施策略：第 4 轮先跑一次 30 秒分块对抗验证（修完 Bug 1 后）：如果真实 SKILL_ONLY <= 0.55 而 FORTY_ONLY > 0.56，只需对 FORTY_COLS 开启届内调和（或统一开启 --within-year 更简洁一致）。
可接受门限不变：ALL 及各追踪分块 macro AUC 均值 ≤0.56（目标 ≤0.55）且 95% 区间覆盖 0.50。

(2) M0b 拆分后的 Y2 故事：“超越秒表”是否成立？
重大实证收获：拆分 M0b 彻底证实了第 2 轮判断——在 Y2 单档质量上，纯秒表+双跳 M0b_stopwatch 的折外相关性为 -0.0198 ± 0.0404（预测力为零），而 M0c_ngs 高达 0.1926 ± 0.0204。这证明原 M0b = 0.1959 的信号 100% 来自 NFL 结合大学产出与球探录像的 ngs_final_score，传统体测秒表对“上场后打得好不好”毫无预测力。
必须撤回的错误数字：绝不可引用本轮的 delta_M1_over_stopwatch_vs_M0b_stopwatch = +0.1040，因为它是 Bug 2（混入 M0a 顺位）的产物。
“超越秒表”能否成立？方向大概率成立，但需要第 4 轮补出真正干净的对照后再用数字说话：
- 新增纯选前直面对比：M1_track_only（仅 TRACK_PURE，无顺位、无 NGS）直接对比 M0b_stopwatch（仅 4 项秒表/双跳，无顺位、无 NGS）；
- 修正 M1_over_stopwatch：基线仅放 M0B_STOPWATCH_COLS（删掉 M0A_COLS，且让 M0B_STOPWATCH_COLS 在基线层走位置内标准化而非 raw_cols）；
- 修正 M0all 与 M1_over_all：拆清 FeaturePrep 的列角色——只有 M0A_COLS（顺位/UDFA/届别）不走位置内标准化，M0B_STOPWATCH_COLS 和 M0C_NGS_COLS 必须走按位置中位数插补 + 位置内标准化。
增量引用口径：M1_over_all vs M0all 等全部模型对比，一律以 10×5 重复 CV 均值 Δ + 配对分层自助 95% CI 为主估计，置换 p 仅作补充检验。

(3) 堆叠后的 Y1/Y2 增量、口径冲突与分类偏置审查
堆叠架构验证：在 Y1 上，两阶段堆叠已经把第 2 轮“重惩罚勒死 log_pick”的问题彻底根除——Y1_hurdle_stage1 的 M1 vs M0a 从 -0.0076 归零到 -0.0002，Y1_stage2_logASR 从 -0.0296 归零到 -0.0000，2025 时序外推 M1 也不再掉分（Hurdle 0.8799 vs 0.8756，logsnaps 0.6312 vs 0.6293）。
为什么本轮 Y2 的 M1 / M1a 反而降到 0.1000 / 0.0953？首要原因是 Bug 1（9 个技能特征全为 0，最强的 skill_s_peak_z_mean 根本没进模型）；修完 Bug 1 后再看 Y2 残差堆叠的真实增量（注意观察 alpha_median 是否从 1.19e+04 正常回落）。
置换 p 显著（0.035 / 0.002）但自助 CI 覆盖 0 的裁决：
先修 permutation_p 里的统计口径 Bug：第 436–446 行把 obs 算成 3 次重复的均值，却把每一次单独重复的 null 直接塞进同一个 nulls 池子里比较（3 次均值方差是单次方差的 1/3，尺度不匹配）；同时 permutation_p 冻结了主 CV 选出的 full_param，消除了内层 RidgeCV 选参方差，导致 obs（+0.0042）系统性高于主 CV 均值 Δ（-0.0009）。
改法：将每次置换改为对同一组 n_repeats_perm（如 3 次）重复分别置换后取 3 次均值作为 1 个 null 样本（例如 333 次置换 × 3 次重复 = 999 次拟合，耗时与当前完全一致，精度与尺度严格对齐）；若时间允许，直接传 res_fn=None 或缩小候选 alphas 网格以包含选参方差。
报告口径裁决（若修完 Bug 1 与置换尺度后仍出现“p_perm<0.05 但自助 CI 含 0”）：
- 预测增益以“10×5 均值 Δ + 自助 95% CI”为准，绝不能拿 p_perm<0.05 去宣称“模型显著提升了预测精度”；
- 两者不矛盾，按原假设精确区分表述：自助 CI 检验的是样本外净预测增益 Δ>0（是否足以抵消增加高维特征的估计方差）；堆叠置换检验的是条件独立原假设 H0: X_track ⊥ Y | (X_base, pos, year)（在固定基线模型不变的前提下，真实追踪特征对残差的拟合是否优于同维度、同位置×届别层内打乱的纯随机噪声）。若出现 p_perm<0.05 而 CI 跨 0，标准学术表述为：“追踪特征携带统计可检出的条件信息，但在 n=299 的小样本下，其整体样本外净预测提升幅度尚不足以稳健跨过估计方差门槛。”
分类堆叠偏置列（训练折内样本拟合）裁决：
M0a 只有 3 个基础列加位置哑变量，训练折内样本拟合 logit 带来的只是轻微保守偏差（略微高估基线、压低追踪增量），并非乐观泄漏。
但既然第 4 轮必须改 model_phase3.py，建议花 3 行代码改成折内 5 折交叉拟合（cross_val_predict(..., cv=5, method="predict_proba")[:, 1]），把这一条披露直接升级为零近似的教科书级实现。

(4) 选秀定价森林图（Fig 4）与稳健性回归
Fig 4 核心故事完全立得住，而且“不对称定价效率”对比非常犀利：
选秀定价端（-log_pick）：40 码直线速度占据绝对主导——forty_s_peak（β=+0.225, q=0.000）、v_max_best（β=+0.189, q=0.000）远高于专项功能速度 skill_s_peak_z_mean（β=+0.153, q=0.001）。
顺位校正后质量端（Y2 | draft）：40 码直线速度彻底归零（forty_s_peak β=+0.048, p=0.329；v_max_best β=+0.060, p=0.235），全部 14 项特征中仅有 skill_s_peak_z_mean 存活且强显著（β=+0.156, p=0.000316, q=0.004, n=299）。
如何看待三重控制稳健性（β=+0.093, p=0.0886, n=252）？
非但没推翻 Fig 4，反而是极合理的边际衰减：加入 forty 后因 40 码跳测直接丢掉了 47 名球员（样本量缩水 16%，从 299 降到 252）；同时 ngs_final_score 已经打包了大学产出、录像评级与部分运动能力，在同时控制选秀顺位、40 码秒表和官方综合评分后，单个 10 Hz 特征仍保留了 60% 的效应量（β=+0.093）和边际显著性（p=0.089）。
第 4 轮补跑 4 步阶梯稳健性表（10 行代码），把衰减来源拆干净：
- + log_pick（n=299，基准 β=+0.156, p=0.0003）；
- + log_pick + forty（n=252，验证是否能打穿纯秒表）；
- + log_pick + ngs_final_score（n=298，不损失 47 名 40 码跳测球员！验证是否能打穿官方综合评分）；
- + log_pick + forty + ngs_final_score（n=252, β=+0.093, p=0.089）。
同时在 draft_pricing 与 fdr_Y2_full 中把传统 -forty、vertical、broad_jump 一起算出来放进 Fig 4 对照行，视觉冲击力会再上一个台阶。
报告措辞红线：绝不写“NFL 选秀市场完全错误定价”。精准表述为“差异化市场定价效率（Differential Pricing Efficiency）”：选秀资本对可见的直线秒表极速定价充分（已耗尽其单档质量 Alpha），而当前标准体测成绩单尚未量化的专项训练功能速度（skill_s_peak）在控制顺位后仍具稳健质量信号（且 100% 覆盖 40 码与敏捷跳测球员），在叠加官方综合评分与 40 码后仍呈正向边际趋势（p=0.089）。

(5) KTR 矩阵、ktr_speed_mean = 0.6524、DL 混杂预警与 burst_1s 中介
紧急数据一致性警报（防抄错数进论文！）：gemini_round3_draft.md 第 5 节引用的多处 KTR 数字与服务器全量 phase3_results_round3_server.md 不符（疑似误抄了旧版或本地冒烟数）：
- forty_s_peak ~ game_accel_p95：服务器实测是 r = +0.414（p=1.22×10^-14, n=319），提纲错写成了 +0.369, n=307；
- DL 分位置对角：服务器实测是 skill_accel +0.568（n=98）、skill_decel +0.432（n=98）、skill_lat +0.542（n=98），提纲写成了 0.474 / 0.369 / 0.437（n=97）；
- WR v_max_best ~ game_s_p99：服务器实测是 +0.201（p=0.109, n=65，不显著），提纲错写成了 +0.297, p=0.019, n=62（显著的是 DB +0.256, p=0.026 与 DL +0.594, p=1.27e-08）。
铁律：Phase 4 所有图表和正文表格必须用脚本直接从最终冻结的 phase3_results.json 生成，严禁人工抄录。
为什么 ktr_speed_mean = 0.6524（低于预期 ~0.82–0.90）？
正常且完全符合橄榄球物理：① df["ktr_speed"] 对全部 5 个位置取了混合均值，而 OL（121 人）+ DL（118 人）占样本近一半（47%），锋线球员在 2–5 码狭小战壕内肉搏，实战 game_s_p99 仅约 4.5–6.0 yd/s（对比 40 码直线 v_max 约 8.5–9.5 yd/s，比值仅 ~0.50–0.60），把 WR/DB 的 ~0.80+ 拉低到了全样本 0.6524；② game_s_p99 是全帧 99 分位（低于生涯单帧最高速），且分母 v_max_best 是渐近极速（比实测 forty_s_peak 高 ~4%–8%）。
改进：第 4 轮按 5 大位置分别输出 game_s_p99 / forty_s_peak 的均值与中位数，报告绝不只报一个混杂的全局 0.65。
高危领域核查预警（必做！）：DL 一枝独秀是否来自 EDGE vs IDL 亚位置混杂？
看服务器 ktr_by_position 和 Y2 分位置 Spearman：DL 在 4 条 KTR 对角线上高达 0.43–0.59（其余 4 个位置在 3 条专项对角线上全部不显著），且 DL 是 Y2 上唯一显著的位置（ρ=+0.354, p=0.0021）。
核对 data_profile_2.md：combine_position == 'DL' 同时打包了 250 lb 外侧冲传 EDGE（DE 36 + OLB 30）与 310+ lb 内侧防守锋线 IDL（DT 47 + NT 5）！EDGE 在体测和实战里天然比 NT/DT 跑得快、变向猛，混在一起必然制造极高的组间假相关！
第 4 轮必须加 15 行核查：在 DL 内部分拆 EDGE (DE/OLB) vs IDL (DT/NT)（或加入 nfl_position 哑变量/体重控制）重算 KTR 对角线与 Y2 相关。若拆开后依然为正，这是顶级加分项；若主要是 EDGE/IDL 体型差驱动，提前揭示这一边界条件能直接避免被 NFL 评审团当场击穿。
burst_1s 中介定稿与 Fig 5 选格：
burst_1s 间接效应 +0.0117，95% CI [-0.0166, +0.0474] 覆盖 0（占比 +7.8%）。可以定稿为你的精准边界结论：“体测运动能力能够跨越护具迁移到实战物理上限（尤其 40 码速度与专项加速度/横向载荷），但无条件聚合的常规赛运动极值与起跑爆发受战术角色与追防情境混杂，并非单档效率的线性中介。”
Fig 5 选格：主图用 4×4 核心迁移热力图（行取 forty_s_peak、skill_accel、skill_decel、skill_lat_accel，舍弃冗余且略弱的 v_max_best；列取 4 个 game_* 指标）+ 右侧配分位置（含 EDGE/IDL 拆分）穿垫速度保持率与对角相关条形图；完整 5×4 表放附录。

(6) 冻结检查与第 4 轮最小修改清单
分模块冻结裁决
立即冻结（零修改、无需重跑）：etl_parquet.py、kinematics.py、features_combine.py、targets.py（eb_kappa 分布极健康，仅 DB::m_epa 与 DL::m_quick_press 两个高噪声指标触及 200 上限，符合橄榄球规律）、game_kinematics.py（401 人、1042 万帧）。
对 3 项轻微方法学残留的裁决：
- static_features.lower_body_burst 全样本位置 z：既然第 4 轮要改 model_phase3.py，直接在 FeaturePrep 内从原始 vertical / broad_jump 折内计算并替换该列（5 行代码），做到零残留泄漏。
- shadow_agility_pure 为 Phase 2 预拟合表示：完全可接受——其权重仅在 Combine 内部用 three_cone/short_shuttle 标定，未接触任何 NFL 比赛结果 Y1/Y2，不存在目标泄漏；正文如实说明，并确认 M1a 在去掉 shadow_agility_pure 后结论一致即可。
- fdr_Y2 使用预计算技能列作单特征描述筛选：完全可接受（无监督协变量、不涉及交叉验证预测声明），正文保持与 CV 预测指标的清晰区分。
第 4 轮最小修改清单（仅改 model_phase3.py 一个文件，约 90 行代码，重跑 1 次即终极冻结）
- 修复 DrillFoldClean 静默清零 Bug（最高优先）：fit() 改为 if self.by_year: for (d, yr), grp in sub.groupby(["drill_name", "draft_year"]): / else: for d, grp in sub.groupby("drill_name"):（单列不用列表！），统一用 (d, yr) 或 d 存键；transform() 改为 self.drill_stats_.get(k, {}).get(m, (np.nan, 1.0))[0]（补上 [m]！）；加上硬断言防静默失效：在 transform() 末尾加 assert out["skill_s_peak_z_mean"].notna().mean() > 0.90，确保永远不可能再次静默产出全 NaN。
- 修复 --within-year 的 LOCO/2025 未见年份处理，并决定开启范围：在 DrillFoldClean.transform 与 FeaturePrep.transform 中，当 by_year=True 且某测试 draft_year 未在 fit() 中出现时，用该测试届自身无标签 X 均值/标准差做届内标准化；先跑 30 秒分块对抗验证确认修完 Bug 1 后的真实 SKILL_ONLY AUC，再开启 --within-year（至少覆盖 FORTY_COLS，或统一覆盖 FORTY + SKILL），使全块对抗验证 AUC ≤0.55。
- 修复 M1_over_stopwatch 与 M0all 的列归属 Bug：新增纯追踪对比 M1_track_only（仅 TRACK_PURE）对比 M0b_stopwatch；修正 M1_over_stopwatch 基线仅含 M0B_STOPWATCH_COLS（删 M0A_COLS）；修正 FeaturePrep / M0all / M1_over_all，使 M0B_STOPWATCH_COLS 和 M0C_NGS_COLS 走按位置中位数插补 + 位置内标准化，只有 M0A_COLS 走 raw_cols；同时把 lower_body_burst 改为在 FeaturePrep 内由原始 vertical/broad_jump 折内合成。
- 修复 permutation_p 尺度对齐 + 升级分类偏置交叉拟合：permutation_p 中每个 null 样本也跨同一组 n_repeats_perm=3 取均值（共 333–500 个独立置换均值样本，与 obs 方差严格同尺度）；_stacked_fit_predict 的分类偏置列 off_tr 改为折内 cross_val_predict(bpipe, Xtr, ytr, cv=5, method="predict_proba")[:, 1]。
- 补齐报告图表所需的 4 项轻量统计表（总计 ~40 行）：skill_s_peak_z_mean 四步阶梯稳健性表（+pick / +pick+forty / +pick+ngs / +pick+forty+ngs）；draft_pricing 与 fdr_Y2_full 补入 -forty、vertical、broad_jump（支撑 Fig 4）；分 5 大位置输出 game_s_p99 / forty_s_peak 的均值/中位数；在 DL 内部分拆 EDGE (DE/OLB) vs IDL (DT/NT) 核查 KTR 对角线与 Y2 相关。
改完后先跑 --quick（约 10 分钟）确认 skill_s_peak_z_mean 非空断言通过、对抗验证 ≤0.55、M1_over_stopwatch 干净、M0all >= M0c_ngs，再放服务器跑最后一次全量，即可正式冻结所有数字进入 Phase 4 出图与 Phase 5 写稿。需要的话，我可以直接给出 model_phase3.py 这 5 处改动的精确代码补丁。

（回传说明：以上为 Gemini 回复全文的忠实转录；页面中数学公式以 LaTeX/符号形式渲染，转录时按页面显示的符号与数值还原，个别公式排版略有归一化，数字与结论未改动。）
