# NFL Big Data Bowl 2027 — 最终报告（本次定时任务，2026-11-21 02:00–03:00）

> **本报告是本次定时任务的阶段报告，不是项目终稿。** 任务原定的“五轮核对 + 最终报告”因服务器被房价项目占用而未能在本轮跑完；以下如实记录已完成部分、阻塞点与下一步。**没有提交任何东西**（本比赛是 writeup 评审制，Hackathon 每队仅 1 次提交，提交前必须经二大爷另行点头）。

## 1. 比赛定性（关键发现）

- 本届是 **writeup（研究报告）评审制，不是预测排行榜比赛**：无 leaderboard、无提交文件格式；评分 100 分——研究质量 40 / 创新 30 / 呈现 20 / 可视化 10；writeup 限 15 页英文；Hackathon 每队只能提交 1 次。
- 截止：**2027-01-07 07:59（GMT+8）**提交 writeup；约 2027-02-22 那周公布获奖者。
- 奖金 $100,000 按规则页是双赛道：Open Track $45,000 与 University Track $45,000（各 3 名 Finalists 各 $10,000、3 名 Runners-Up 各 $5,000），另 $10,000 Grand Prize 给 Combine 现场展示优胜者；Finalists 均受邀到 2027 NFL Scouting Combine 展示。**二大爷是在校大学生，应走 University Track。**
- 账号 dewang2025 已接受规则并加入比赛（2026-11-21 02:06）。
- 规则要点：获奖代码须按 OSI 开源许可公开、可复现；数据 CC BY-NC 4.0 仅限非商业用途、不得外传；可用公开免费外部数据；队伍最多 5 人。

## 2. Gemini 方案要点（第三步，Neon 最高档，02:14–02:29）

方案全文：`notes/gemini_scheme.md`（对话 https://gemini.google.com/app/f42509acc9d07f4b ）。

- 主研究问题：超越秒表与跳测黑洞——Combine 专项训练的 10Hz 运动学特征（减速制动、曲线变向保速、加速动力学）能否在控制选秀顺位后增量预测常规赛表现，并揭示选秀市场系统性错误定价。
- 双核目标：Y1 暴露校正年均 REG 档（两阶段 Hurdle：≥150 档/季门槛 + Gamma/Tweedie）+ Y2 EB 收缩位置单档质量分（位置内标准化 + James-Stein 收缩 + IPW 选择偏差修正）；Y3 首发率仅作稳健性。
- 特征：静态体测派生（BMI、臂展指数、Speed Score、Flying-30 等）+ combine 追踪运动学（40 码 Furusawa-Hill 曲线 v_max/τ/a0、减速 p95、变向保速、急动度、体重加权功率）+ 位置专属比赛质量指标（WR 分离度/YACOE、EDGE get-off/快速施压率、OL 允许施压概率、DB 闭合爆发力）。
- 创新点：Shadow Agility（用 85.1% 帧数的技能训练重建隐性敏捷度，绕开 60%+ 跳测黑洞）、选秀错误定价森林图、体测→实战运动学迁移矩阵（KTR）。
- 验证：10×5 折重复分层嵌套 CV + Year-1 Only 时序外推（2023+2024 训练→2025 测试）+ LOCO + 1,000 次置换检验 + BH-FDR 校正 + 2,000 次分层 BCa Bootstrap。
- 里程碑（47 天）：Phase 1 11/21–11/26 ETL 与审计；Phase 2 11/27–12/06 特征与 Shadow Agility 标定；Phase 3 12/07–12/18 主建模与验证；Phase 4 12/19–12/27 可视化与案例；Phase 5 12/28–01/05 写报告与提交（内部提交日 2027-01-05）。

## 3. 已完成的工作（附验证结果）

### 数据与画像（第二步）
- 数据已下载解压：9 个 CSV，2.31 GB。行数：players 510、combine_results 510、combine_tracking 463,189、game_tracking 3,665,443 / 7,827,944 / 11,160,903、games 1,002、player_career_successes 510、player_play 314,197（实测；页面标注 316,338 相差 2,141，原因不明，以实测为准；物理行数 327,020 是 play_description 内嵌换行所致）。
- 画像存 `notes/data_profile.md`；题目原文存 `notes/problem_statement.md`；规则存 `notes/rules.md`。

### Phase 1：ETL + 审计（第四步，本地冒烟通过）
- 代码：`src/etl_parquet.py`（CSV→Parquet：REG 过滤、float32、player_play 断言 314,197 行并丢 play_description 文本列，140MB→14.68MB）+ `src/audit_phase1.py`（四道门控 + 跳测 MNAR 诊断 + 位置映射 + Y1 预览）。
- 四道 Go/No-Go 门控**全部 PASS**（`notes/eda_audit.md`）：G1 player_play 314,197 行；G2 PLAYER 帧 s>12.5 共 0 帧；G3 三季重算 a_t KS D=0.0078/0.0097/0.0149（均 <0.05）；G4 三年均 ≥20 人的 drill 31 个。
- 两处方案假设被实测纠正：①跳测 MNAR 不成立（跳测组选秀顺位中位数 121 vs 124，p=0.401；ngs p=0.504；分轮跳测率均 >50%，第 1 轮 68.4% 仅略高）；②生涯 REG 档 ≥75 档为 320 人（方案预估 310–350 一致），零 REG 档 109 人。
- 环境：本地 venv `kaggle-data/.venv`（--system-site-packages），numpy 1.26.4 / pandas 2.1.4 / scipy 1.11.4 / pyarrow 15.0.0 / statsmodels 0.15.0（`requirements.txt` 已记；注意 pyarrow 26 要求 numpy 2，已降回 15.0.0）。

### Phase 2 代码（已写 + 本地冒烟通过，全量待服务器跑）
- 代码：`src/kinematics.py`（Savitzky-Golay 平滑 + 重算有符号 a_t/ω/a_n/jerk + 变向事件检测 + Furusawa-Hill 拟合）+ `src/features_combine.py`（静态派生特征 + 40 码曲线分解 + 技能训练聚合 + drill 内 z-score + Shadow Agility 岭回归校准）。
- 冒烟（60 个 40 码事件 + 60 个技能事件）：40 码拟合 R² 均值 0.991、100% > 0.95（方案 Phase 2 门限：>95% 的跑次 R²>0.95）；输出 510 行 × 45 列 `combine_features.parquet`。Shadow Agility 校准在冒烟小样本下无意义（n=6/9），全量跑完后以 OOF r ≥ 0.40（p<0.001）为准。

### GitHub 同步（未完成，受阻）
- 12 个文件（代码 + 笔记 + requirements，不含数据）已备好清单，远端 head 已核对一致（a5aeed72…）、远端目录确认不存在；但 `push-files` 两次均被云端服务重启打断，查远端确认均未生效，未推进同步记录。**下次运行时重试推送。**

## 4. 当前最佳方案与分数

- 当前最佳方案 = Gemini 方案（尚未跑出数字）：主目标 Y1+Y2、M1 两阶段分层 GLM 为主力，M2 Shadow Agility 中介、M3 可解释树为对照。
- **尚无验证分数**：服务器计算未启动（见下）。本地冒烟仅验证了代码 mechanics，未产生可汇报的模型分数。

## 5. 离能提交还差什么

1. **服务器计算**：等房价项目跑完 → 传代码 + 数据上东京服务器 → 全量跑 Phase 1（ETL + 审计）→ Phase 2（特征 + Shadow Agility 标定，须过 OOF r ≥ 0.40 门）→ Phase 3（主建模、时序外推、置换检验）→ Phase 4（6 张图 + 球员案例）→ Phase 5（15 页英文 writeup + 开源复现包）。
2. **五轮 Gemini 核对**（`notes/gemini_review_rounds.md` 已建骨架，第 0 轮完成，第 1–5 轮待服务器结果）。
3. **数据上服务器**：2.3GB 原始数据尚未传服务器（GitHub 不能传数据；须走 Cloud Shell + gcloud scp 或服务器直下 Kaggle，待房价跑完后办）。
4. **GitHub 同步重试**：12 个文件待推（见上）。
5. **提交本身**：writeup 写完后须二大爷点头才能交（Hackathon 每队仅 1 次提交，不可撤回；内部提交日 2027-01-05）。

## 6. 风险与下一步建议

- **阻塞（当前）**：东京服务器房价项目 stage2 调参仍在全力跑（02:40 查：负载 7.78/8 核，stage2_tune.py 占 631% CPU，SERVER_RUN_DONE 不存在）。按规矩 NFL 计算须等它跑完，**本次运行未启动任何服务器计算**。
- **服务器未降配置（有意为之）**：任务原要求“结束时降回 2 核 8G”，但降配置要停机，会打断正在跑的房价项目（房价今晚刚升到 8 核重跑）。机器现在不闲，降配置是错的。**等房价跑完、NFL 计算告一段落后，再由后续运行执行降配置并验证 IP 与 x-ui。**
- **云端 VM 不稳定**：本次运行中 VM 至少被替换一次（pip 装的 pyarrow/statsmodels 丢失、/tmp 被清、GitHub 推送两次被服务重启打断）。教训已记进反思库：依赖一律放 ~/workspace 下的 venv，临时文件不放 /tmp，写操作中断后先查远端再重试。
- **统计风险**：样本仅 510 人，TE 仅 42 人；目标零膨胀（109 人零 REG 档）；跳测 MNAR 假设已被证伪，报告里不能再把 MNAR 当既定事实写；Shadow Agility 能否达到 OOF r ≥ 0.40 是 Phase 2 最大不确定点，不过就改切入角度。
- **下一步**：① 房价跑完后立即传数据+代码上服务器跑 Phase 1–2；② 重试 GitHub 推送；③ 出 Phase 2 结果后启动五轮 Gemini 核对；④ 按 47 天里程碑推进，内部提交日 2027-01-05。

## 7. 代码和数据位置

- 本地：`~/workspace/kaggle-data/03-nfl-big-data-bowl/`（data/ 2.31GB 原始 CSV + parquet_smoke/ + derived_smoke/；notes/ 7 个 md；src/ 5 个 py；requirements.txt；README.md）
- 本地 venv：`~/workspace/kaggle-data/.venv`（numpy 1.26.4 / pandas 2.1.4 / scipy 1.11.4 / pyarrow 15.0.0 / statsmodels 0.15.0）
- 服务器：`/home/wangde2021/03-nfl-big-data-bowl` 尚不存在（待建）；服务器 Python 3.12.3
- GitHub：`wangde2021-collab/kaggle` 仓库 `03-nfl-big-data-bowl/` 待推送（12 个文件）

---

# 终稿（2026-11-22，定时任务 nfl-big-data-bowl-2027-continue-2 收尾）

> **本节为本次定时任务（2026-11-21 18:10 – 2026-11-22 18:10 CST）的终稿，追加在阶段报告之后，阶段报告原文未动。** 五轮 Gemini 核对中第 3、4 轮已完成，第 5 轮服务器全量仍在跑、Gemini 第 5 轮未做——见“未完成事项”。**没有提交任何东西**（writeup 评审制，每队仅 1 次提交，提交前必须经二大爷另行点头）。数字来源：各轮服务器结果冻结文件（`notes/phase3_results_roundN_server.md`）与本地冒烟日志；冒烟数字只验代码、不算数。

## 1. 五轮总览

| 轮次 | Gemini 回传 | 总裁决 | 关键改动 | 服务器全量 |
|---|---|---|---|---|
| 第 1 轮 | 2026-11-21 09:44 | 方案假设多处被纠正；Shadow 假失败 | 10 项硬 Bug 修复（Year-1 截断、EB 收缩、END_EVENTS、drill 折内归一化等） | 修复前快照 + 修复后基线（Phase 3c 2h44m） |
| 第 2 轮 | 2026-11-21 17:04 | Path B 确认；主线切为“专项功能速度 vs 秒表速度” | t0 修复、M0b 拆分、permutation_p 修复、定价回归、FDR 拆族 | 第 3 轮服务器（6h10m） |
| 第 3 轮 | 2026-11-22 01:13 | Phase 1/2/3a/3b 冻结；Phase 3c 暂不可冻结 | DrillFoldClean 清零 Bug、M1_over_stopwatch 偷混顺位（“+0.1040”撤回） | 第 3 轮：Y2 M1 0.1000；对抗 ALL 0.6627 |
| 第 4 轮 | 2026-11-22 13:04 | 主 CV 结论跑通；Phase 3c 又查出 2 Bug + 1 口径问题 | 未见届填补三级、protect 默认改 False、ktr 分母、敏感性门槛、置换缓存提速 | 第 4 轮：model 7h14m；Y2 M1 0.1335；对抗全块 ~0.49 |
| 第 5 轮 | 2026-11-22 22:36 | **Phase 1–3 正式冻结，无需第 6 轮** | 同上（仅改 model_phase3.py，约 60 行） | 第 5 轮：model 5h46m22s；Y2 M1 0.1335；对抗全块 ~0.49；3 项验收全绿 |

## 2. 每轮验证分数（Y2 为主目标，10×5 CV 均值 r；种子 42）

### 第 3 轮服务器（2026-11-21 18:44 – 11-22 00:54，6h10m）
- Y2：M0a 0.1009、M0b_stopwatch −0.0198、M0c_ngs 0.1926、M0b 0.1959、M0all 0.1747、M1 0.1000、M1b 0.2020、M1_over_stopwatch 0.0843。Δ M1_over_stopwatch +0.1040（**后证实为混入顺位的假数，已撤回**）；Δ M1 vs M0a −0.0009；置换 p=0.0350。
- 对抗验证 draft_year macro AUC：ALL 0.6627、FORTY_ONLY 0.6646、SKILL_ONLY 0.4773、STATIC_ONLY 0.5742（届别信号未消除）。
- LOCO_Y2 M1：2023 0.197/0.185、2024 0.094/0.094、2025 0.011/0.015。targets 510 人、y2 covered 299；game_kinematics 401 人、10,424,763 帧。

### 第 4 轮服务器（2026-11-22 03:41–12:44，9h03m；model_phase3 --within-year 7h14m）
- Y2：M0a 0.1009、M0b_stopwatch −0.0016、M0c_ngs 0.1984、M0b 0.1756、M0all 0.1629、M1 0.1335、M1a 0.1403、M1b 0.1587、M1_track_only 0.0902、M1_over_stopwatch 0.0755、M1_over_all 0.1823。
- 干净主增量：Δ M1_over_stopwatch **+0.0771 CI [+0.0122, +0.1545]**；Δ M1_track_only +0.0918 CI [−0.0076, +0.2217]；Δ M1 vs M0a +0.0326 CI [−0.0114, +0.0819]；Δ M1a vs M0a +0.0394；置换 p=0.0030。
- Y1 双重分离：hurdle stage1 M0a 0.8631 / M1 0.8641；stage2_logASR M0a 0.6611 / M1 0.6607——Y1 零增量、Y2 有增量。
- 对抗验证：ALL 0.4922、FORTY_ONLY 0.4924、SKILL_ONLY 0.4892、STATIC_ONLY 0.4879——**四分块全部回到随机水平 0.50**。
- LOCO_Y2 M1：2023 0.1537、2024 0.1303、2025 0.0496（Gemini 指出该数被 Bug A 污染，不可据此下悲观结论）。时序 Y1_year1_hurdle M1 0.8701、logsnaps 0.6200。
- 定价：forty_s_peak β=+0.225、v_max_best +0.189、skill_s_peak +0.153（q=0.001）。稳健性阶梯：+pick β=0.156 p=0.0003 → +pick+forty β=0.094 p=0.077 → +pick+ngs β=0.138 p=0.0027 → 三重 β=0.093 p=0.0886（n=252）。
- DL 拆分：EDGE skill_decel~game_decel r=+0.357（p=0.0069，n=56），Y2 Spearman +0.438（p=0.0026，n=45）；IDL 各对角线 ≈0，但 lat_accel r=+0.396（p=0.0095），Y2 Spearman +0.313（p=0.105，n=28）。
- M3 LightGBM Y2 r=0.1457。M2 中介：decel −0.0050（−3.3%）、burst_1s +0.0117（+7.8%，边界结论）。

### 第 5 轮服务器（2026-11-22 15:12:33–20:58:55，model_phase3 --within-year 5h46m22s；**冻结版**）

- Y2：M0a 0.1009、M0b_stopwatch −0.0016、M0c_ngs 0.1984、M0b 0.1756、**M0all 0.1829（第 4 轮 0.1629，Bug B 修复后 +0.0200）**、M1 0.1335、M1a 0.1403、M1b 0.1587、M1_track_only 0.0902、M1_over_stopwatch 0.0755、M1_over_all 0.1823。
- 干净主增量：Δ M1_over_stopwatch **+0.0771 CI [+0.0122, +0.1545]**；Δ M1_track_only +0.0918 CI [−0.0076, +0.2217]；Δ M1 vs M0a +0.0326 CI [−0.0114, +0.0819]；Δ M1a vs M0a +0.0394；**Δ M1_over_all vs M0all −0.0006 CI [−0.0169, +0.0158]（第 4 轮 +0.0194 确为口径伪影，现已归零）**；置换 p=0.0030。
- Y1 双重分离：hurdle stage1 M0a 0.8631 / M1 0.8641；stage2_logASR M0a 0.6611 / M1 0.6607——Y1 零增量、Y2 有增量。
- 对抗验证：ALL 0.4922、FORTY_ONLY 0.4924、SKILL_ONLY 0.4892、STATIC_ONLY 0.4879——**四分块全部随机水平**（与第 4 轮逐数一致）。
- LOCO_Y2 M1（Bug A 修复后，全量验证）：2023 **0.1831**（第 4 轮 0.1537，+0.029；与 M0a 差距由 −0.043 收窄至 −0.014）、2024 0.1315（高于 M0a 0.0938）、2025 0.0513（高于 M0a 0.0106）。时序 Y1_year1_hurdle M1 0.8670、logsnaps 0.6241（仍略低于 M0a 0.8756/0.6293，噪声级）。
- 定价：forty_s_peak β=+0.225、v_max_best +0.189、skill_s_peak +0.153（q=0.001）。稳健性阶梯 5 步：+pick β=0.1562 p=0.000316（n=299）→ +pick+forty β=0.0944 p=0.0774（n=253）→ +pick+ngs β=0.1378 p=0.00268（n=298）→ 三重 β=0.0926 p=0.0886（n=252）→ **第 5 步 forty.notna() 子集只控 log_pick β=0.1361 p=0.00302（n=253，仍显著）**；分解：样本缩水 −0.020（32.5%）vs forty 共线性 −0.042（67.5%）。
- ktr WR 保持率 0.8151（9.03/11.08=0.815 逐数吻合）；五位置梯度 WR 0.815 > TE 0.755 > DB 0.733 > DL 0.624 > OL 0.517。DL 拆分：EDGE skill_decel~game_decel r=+0.357（p=0.0069，n=56），Y2 Spearman +0.438（p=0.0026，n=45）；IDL lat_accel r=+0.396（p=0.0095），Y2 Spearman +0.313（p=0.105，n=28）。
- M3 LightGBM Y2 r=0.1457。M2 中介：decel −0.0050（−3.3%）、burst_1s +0.0117（+7.8%，边界结论）。
- **提速复核：30–45 分钟预估未兑现**（5h46m vs 第 4 轮 7h14m，仅快约 20%）；缓存为无损记忆化，零统计口径风险；绝不为此破冻重跑。

## 3. 当前最佳方案与分数

- 当前最佳 = **第 5 轮服务器版（冻结版，--within-year，M1/M1a）**：Y2 M1 0.1335、M1a 0.1403；Y2 headline 三层递进引用（Gemini 第 5 轮裁决）：① 超越秒表 Δ M1_over_stopwatch +0.0771 CI [+0.0122, +0.1545]（强阳性）；② 顺位校正 M1a/M1 vs M0a Δ=+0.0394/+0.0326（条件信号显著，置换 p=0.0030；全样本外净增益 CI 略跨 0）；③ 诚实边界 M1_over_all vs M0all Δ=−0.0006（必须引用：追踪整体不能超越“顺位 + 官方 NGS 总分”组合）。
- 一号主结论（Gemini 第 5 轮裁决，Double Dissociation 稳固）：Y1 零增量 vs Y2 增量——选秀顺位买得到出场档数、买不到单档效率；追踪特征只在质量端（Y2）增量预测；LOCO 2024/2025 两届 M1 跑赢顺位基线。
- 二号结论：DL 亚组拆解（EDGE 靠速度/加速/制动迁移，IDL 靠横向变向载荷迁移）直接写正文。
- Fig 4 措辞：**“差异化市场定价效率”**，绝不写“市场完全错误定价”。
- 方法节术语：--within-year 定义为 "Within-Cohort Unsupervised Harmonization"；_safe_sd 地板（组内 sd < 全局 sd 5% 视为退化、改用全局 sd）直接写入方法节。
- 冻结铁律：Phase 4 出图与制表一律直接读冻结 JSON（`notes/phase3_results_round5_server.json` 等），严禁从 Markdown 转录取数（已发现两处转写印误：M0b alpha 印作 393、gated:DB::m_getoff 实为 DL——后者已核实并更正）。

## 4. 代码和数据位置

- 本地：`~/workspace/kaggle-data/03-nfl-big-data-bowl/`（src/ 8 个 py；notes/ 本报告与各轮记录；data/ 2.31GB 原始 CSV + parquet_smoke/ + derived_smoke/，**数据不上传、不外传**）。
- 本轮冻结版 model_phase3.py：md5 6d1c3f7dc117ce7574dfaa09a3955ebe，94,930 字节，1,819 行（2026-11-22 13:22 左右本地改完，py_compile 通过，6 项直接执行测试全过）。
- 服务器：`/home/wangde2021/03-nfl-big-data-bowl`，机型 e2-standard-2（2 核 8G），IP 35.213.17.8；第 5 轮 model_phase3 已于 2026-11-22 20:58:55 正常跑完（5h46m22s），日志 `logs/phase3c_model_r5.log`（6,910 行，error/traceback 0 条）。
- 服务器备份：`prefix_snapshot_r4/`（第 5 轮前备份，9 文件，逐文件 cmp 全部一致）。
- 冻结文件（单一真相源）：本地 `notes/phase3_results_round5_server.json`（服务器 phase3_results.json 逐字转存，22 顶层键）+ `notes/phase3_results_round5_server.md`（946 行）+ 服务器 `data/derived/oof_predictions.parquet`（510×61）。
- GitHub：`wangde2021-collab/kaggle` 仓库 `03-nfl-big-data-bowl/`——见 §6 同步状态。
- 运行日志：`~/workspace/goals/kaggle-learning-practice/hidden_files/nfl-20261121-continue-runlog.md`；轮次记录：`notes/gemini_review_rounds.md`；各轮 Gemini 回复全文：`notes/gemini_roundN_20261122.md`。

## 5. 未完成事项（停在哪一步、还剩什么）

1. **服务器第 5 轮全量与 Gemini 第 5 轮均已完成（2026-11-22 22:36）。** model_phase3 15:12:33–20:58:55 正常退出，耗时 5h46m22s；Gemini 第 5 轮裁决 **Phase 1–3 正式冻结，无需第 6 轮**（回复全文 `notes/gemini_round5_20261122.md`）。3 项验收全绿：LOCO 三届 M1 0.1831/0.1315/0.0513；M0all 0.1829、delta_M1_over_all −0.0006 干净归零；ktr WR 0.8151。详见本报告 §9。
2. **终稿报告的提交规格冲突仍待核准**（只写进报告，不替二大爷办）：比赛页现写文章 ≤2,000 词、图表 <10 张、附公开 Kaggle Notebook，与此前记的 15 页英文 writeup 不符；写报告前先核准以哪个为准。Kaggle 身份验证与手机验证已于 2026-11-22 经账号设置页核实完成（身份验证走 Persona），提交前无验证障碍。

## 6. GitHub 同步状态

- **已同步（2026-11-22 约 22:55 CST，本趟内完成并验证）。** 经 GitHub 连接器 `push-files` 推送 8 个文件（第 5 轮冻结版：`notes/final_report.md`、`notes/gemini_review_rounds.md`、`notes/gemini_round5_draft.md`、`notes/gemini_round5_20261122.md`、`notes/phase3_results_round5_server.md`、`notes/phase3_results_round5_server.json`、`notes/extract_round5_numbers.py`、`notes/extract_round5_numbers_output.txt`，约 184KB；**不含 data/**），新提交 `8f6199b9e4bea9318258000358f08f74464c6f76`，远端 main 分支 head 已核对一致。同步记录 `~/workspace/kaggle-data.muse-github-snapshot-sync.json` 已推进（第 6 条记录）。
- 同步前核对：远端 head 仍为 `965cbbc`（与记录一致，无其他人动过）；新增文件已确认远端不存在（`phase3_results_round5_server.json` 查无此文件），改动文件与同步基线（387c037/965cbbc）无分歧。
- **已同步（2026-11-22 约 17:35 CST，本趟内完成并验证）。** 经 GitHub 连接器 `push-files` 推送 31 个文件（`03-nfl-big-data-bowl/src/*.py` 8 个 + `notes/*.md` 21 个 + `README.md` + `requirements.txt`，约 570KB；**不含 data/**），新提交 `387c0373950ffa66b5267c3f946b91020bcb43af`，远端 main 分支 head 已核对一致。同步记录 `~/workspace/kaggle-data.muse-github-snapshot-sync.json` 已推进（第 4 条记录）。
- 同步前核对：远端已有 `03-nfl-big-data-bowl/`（2026-11-21 04:46 提交 a9a6c8e7，第 1 轮修复版，18 个文件）；本次为更新其中 18 个 + 新增 13 个（第 2–5 轮笔记与结果文件），无删除；自 a9a6c8e7 后无其他人动过该目录（远端 head 其间只多了一个无关的 Telco 提交）。
- 直接 `git` 直推此路不通：沙箱代理重置了到 github.com:22 的 SSH 连接（`Connection reset by peer`，经 198.19.0.1:3128），`gh` CLI 未登录；以后同步继续走连接器 `push-files`。

## 7. 离能提交还差什么

1. ~~收服务器第 5 轮全量结果 → 派 Gemini 第 5 轮 → 若宣布冻结，代码即最终冻结版。~~ **已完成并冻结（2026-11-22 22:36，Gemini 第 5 轮裁决）。**
2. 核准提交规格（≤2,000 词 + <10 图 + 公开 Notebook vs 15 页 writeup），以哪个为准。
3. 按核准后的规格写报告：Phase 4 可视化（6 张图：Fig 4 定价森林图、Fig 5 KTR 热力图等）+ 球员案例 + 15 页英文 writeup（或 ≤2,000 词文章）+ 开源复现包（代码按 OSI 许可公开）。
4. 提交：**必须二大爷另行点头**，每队仅 1 次。截止 2027-01-07 07:59（GMT+8），内部提交日 2027-01-05。

## 8. 风险与下一步建议

- **提速预估失真**：置换缓存只省了 Stage 1 重拟合，全量瓶颈仍在主 CV（第 5 轮 5h46m vs 第 4 轮 7h14m，仅快约 20%）；绝不为此破冻重跑。
- **服务器已空闲**：第 5 轮跑完后无重活在跑；按已拍板的状态保持 e2-standard-2（不要重复降级，降级步骤已提前做完）。
- **x-ui 已澄清**：面板在自定义路径 `/287a58ff96/`，服务器本地实测 HTTP 200（2026-11-22 21:58 再确认）；之前对 `/`、`/login` 等路径的 404 不是故障。服务 active（running），xray 在运行，2053 端口 LISTEN。
- **统计风险未变**：样本仅 510 人、Y2 仅 299 人；LOCO 2025（n=83）与时序外推仍是薄弱点，Limitations 按 Gemini 第 5 轮给的写法（抽样标准误 ≈0.11）交代；旧轮次 bug 不必在正文披露为局限（Gemini 第 4 轮裁决）。
- **下一步**：① 核准提交规格（≤2,000 词 + <10 图 + 公开 Notebook vs 15 页 writeup）；② Phase 4 可视化（6 张图，按 Gemini 第 5 轮锁定的 Fig 4/Fig 5 口径，数字一律读冻结 JSON）；③ Phase 5 英文成稿（按三层 headline + 阶梯分解措辞）；④ GitHub 同步（走 push-files，见 §6）。

## 9. 第 5 轮冻结结论（2026-11-22，定时任务 nfl-big-data-bowl-2027-continue-3 收尾）

> **Gemini 第 5 轮总裁决（2026-11-22 22:36 回传，回复全文 `notes/gemini_round5_20261122.md`）：Phase 1–3 全链路正式冻结，无需第 6 轮重跑，即刻转入 Phase 4 出图与 Phase 5 写作。** 本节追加在 continue-2 终稿之后，此前章节原文未动（§5 未完成事项已同步更新为已完成）。**没有提交任何东西**（writeup 评审制，每队仅 1 次提交，提交前必须经二大爷另行点头）。

### 冻结范围

Phase 1 ETL、Phase 2 Combine 特征（含 combine_features.parquet 与 Shadow 标定）、Phase 3a 目标表、Phase 3b 实战运动学、Phase 3c 建模结果（phase3_results_round5_server.json、oof_predictions.parquet）全部永久冻结。后续写作与作图严禁再改模型超参或特征定义。

### 服务器第 5 轮全量（冻结版数字，种子 42；10×5 CV；置换 1000 次；自助 2000 次）

- Y2_quality：M0a 0.1009、M0b_stopwatch −0.0016、M0c_ngs 0.1984、M0b 0.1756、M0all 0.1829、M1 0.1335、M1a 0.1403、M1b 0.1587、M1_track_only 0.0902、M1_over_stopwatch 0.0755、M1_over_all 0.1823；Δ M1_over_stopwatch +0.0771 CI [+0.0122, +0.1545]；Δ M1_track_only +0.0918 CI [−0.0076, +0.2217]；Δ M1 vs M0a +0.0326 CI [−0.0114, +0.0819]；Δ M1a vs M0a +0.0394 CI [−0.0097, +0.0951]；Δ M1_over_all vs M0all −0.0006 CI [−0.0169, +0.0158]；置换 obs +0.0415，p_perm=0.0030。
- Y1：hurdle stage1 M0a 0.8631 / M1 0.8641（Δ +0.0009，p=0.0599）；stage2_logASR M0a 0.6611 / M1 0.6607（Δ −0.0005，p=0.1168）。时序外推：hurdle M0a 0.8756 / M1 0.8670；logsnaps M0a 0.6293 / M1 0.6241。LOCO_Y2：2023 M0a 0.1971 / M1 0.1831；2024 M0a 0.0938 / M1 0.1315；2025 M0a 0.0106 / M1 0.0513。
- 对抗验证：ALL 0.4922、FORTY_ONLY 0.4924、SKILL_ONLY 0.4892、STATIC_ONLY 0.4879。forty_s_peak 分年 10.419 / 10.368 / 10.393。
- 定价（-log_pick）：neg_forty +0.271、vertical +0.232、forty_s_peak +0.225、broad_jump +0.221、v_max_best +0.189、skill_s_peak +0.153（q=0.001）。fdr_Y2 主族仅 skill_s_peak q=0.004。稳健性阶梯 5 步：0.1562 → 0.0944 → 0.1378 → 0.0926 → 0.1361（第 5 步 p=0.00302）。
- KTR：ktr_speed_mean=0.6759（forty_s_peak 口径）；WR 保持率 0.8151；M2 decel 中介 −0.0050（−3.3%）、burst_1s +0.0117（+7.8%）；M3 Y2 r=0.1457；Y2_sensitivity n_plays_100 r=0.1768、n_plays_150 r=0.1116。DL 拆分：EDGE decel r=+0.357、Y2 Spearman +0.438；IDL lat_accel r=+0.396、Y2 Spearman +0.313。
- 完整结果：`notes/phase3_results_round5_server.md`（946 行）与 `notes/phase3_results_round5_server.json`（单一真相源）。

### Gemini 第 5 轮 6 个议题裁决摘要

1. 3 项验收全绿，正式冻结（M0all 未超 M0c_ngs 属正常 L2 稀释，不阻塞）。
2. Y2 headline 三层递进：① 超越秒表 +0.0771（强阳性）；② 顺位校正 +0.0394/+0.0326（条件信号显著，CI 略跨 0）；③ 诚实边界 −0.0006（必须引用）。
3. Double Dissociation 稳固；Limitations 按 2025 届 n=83、抽样标准误 ≈0.11 的写法交代。
4. 阶梯第 5 步分解（样本缩水 32.5% vs forty 共线性 67.5%）可写入正文；定稿措辞微调版已给。
5. 提速未达预期（5h46m vs 7h14m，仅快约 20%），缓存零统计口径风险，绝不为此破冻重跑；只在附录 Reproducibility 一句带过。
6. 冻结后 6 条数字口径铁律：单一真相源为冻结 JSON；Fig 4 双列森林图口径；Fig 5 4×4 热力图 + 保持率 + DL 亚组；per_position_spearman 异质性写法；Y2_sensitivity 门槛标注 2×5 CV；Shadow 统一命名 Skill-Drill Movement Index、M3 克制表述。

### 还剩什么（Phase 4/5 与提交）

1. 核准提交规格（≤2,000 词 + <10 图 + 公开 Notebook vs 15 页 writeup），以哪个为准——须二大爷另行拍板。
2. Phase 4 可视化（6 张图，按 Gemini 第 5 轮锁定的 Fig 4/Fig 5 口径，数字一律读冻结 JSON）+ 球员案例。
3. Phase 5 英文成稿（按三层 headline + 阶梯分解措辞）+ 开源复现包（代码按 OSI 许可公开）。
4. 提交：**必须二大爷另行点头**，每队仅 1 次。截止 2027-01-07 07:59（GMT+8），内部提交日 2027-01-05。
