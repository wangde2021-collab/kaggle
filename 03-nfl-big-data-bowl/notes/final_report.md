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
