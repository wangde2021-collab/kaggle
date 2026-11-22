# Gemini 第 5 轮核对提纲（草稿，服务器数字待填）

- 对话：同一对话 https://gemini.google.com/app/8adf0ca845f706e2 ，账号 wangde2021@gmail.com（Google AI Pro），模型 Neon、思考等级“高”。
- 附件（8 个）：5 个 src 文件（model_phase3.py 为本轮唯一改动文件，其余 4 个自第 3 轮后未改）+ notes/data_profile.md + notes/phase3_results_round5_server.md（服务器全量全文，待填）+ 本提纲。
- 本轮定位：最终冻结确认轮（非新一轮 bug 狩猎）。第 4 轮裁决已定：修完 Bug A/B/C 并通过 3 项核对即终极冻结，直接进入 Phase 4 出图与 Phase 5 写作。请按以下顺序核对：① 先复核本轮 3 处修复是否正确落地（代码级）；② 对照服务器全量数字验证 3 项验收指标；③ 仅当发现影响结论的实质问题时才要求再改一轮，否则请明确宣布 Phase 3 冻结。

## 第 4 轮裁决落实情况（已做，请逐项复核代码）

### Bug A（未见届中位数填补）——已修
- `FeaturePrep.fit` 增存 `track_pos_med_[c]`（按 pos_group、不含年份的训练中位数）；`_lbb_z` 的 `lbb_stats_[c]` 增存 `pos_median`。
- `transform` 与 `_lbb_z` 改三级填补：训练折 (pos, year) 中位数 → 测试届自身 (pos, year) 无标签中位数 → 训练折 pos_group 中位数 → `__global__`。
- 直接执行测试（合成数据）证实：未见届跳测 OL 的 forty z 由旧规则 −1.86 降为 −0.12（组中心）；`_lbb_z` 同步改造。
- 冒烟验证（--within-year 探针，r6-wy）：LOCO_Y2 M1：2023 0.1835（r5-wy 0.1545，+0.029）、2024 0.1317（0.1306）、2025 0.0522（0.0505）；2025 时序 hurdle M1 0.8669（0.8707，−0.004）、logsnaps 0.6234（0.6193，+0.004）——冒烟样本小，±0.004 为噪声级，以服务器全量数为准。

### Bug B（M0all 与 M1_over_all 的 BaselineProtect 开关）——已修
- `make_model_pipe` 默认 `protect=True` → `protect=False`；仅分类 `_stacked_fit_predict` 的 Stage 2 rpipe 显式传 `protect=True`（offset 列必须近乎免惩罚）。
- 直接执行测试证实：M0all（默认）与 M1_over_all Stage-1 基线逐位一致（max diff 0.00e+00）；旧 protect=True 版预测差 0.0674。
- 冒烟验证：默认路径 Y2 M0all 0.1960 ≥ M0c_ngs 0.1948，alpha_med 由 266 降至 146.8；delta_M1_over_all_vs_M0all = +0.0009 CI [−0.0101, +0.0109]（第 4 轮 +0.0194 确为口径伪影，现已归零）；--within-year 探针 Y2 M0all 0.1816（r5-wy 0.1522，+0.029），delta = +0.0052 CI [−0.0093, +0.0198]。

### Bug C（口径收口）——已修
- `ktr_speed` 改为 `game_s_p99 / forty_s_peak`（原 v_max 版另存 `ktr_speed_vmax_mean/median`）。冒烟 WR 0.8075（原 v_max 版 0.7744；冒烟 game_s_p99 均值 8.94 vs 全量 9.03，样本差异在预期内）。
- `Y2_sensitivity` 改为 `n_plays >= 100 / >= 150`：冒烟 n_plays_100 n=286 r=0.1669、n_plays_150 n=261 r=0.0907。
- `experiments.csv` 日期改 `datetime.date.today().isoformat()`；`phase3_results.md` 加 `per_position_spearman` 一节。
- 稳健性阶梯加第 5 步（forty.notna() 子集 n=253 上只控 log_pick）：冒烟 β=0.1238、p=0.0041（仍显著）。对照：第 1 步（n=299 只控 log_pick）β=0.149 → 第 5 步 β=0.124（样本缩水代价 −0.025）→ 第 2 步（同子集加 forty）β=0.078、p=0.117（共线性代价 −0.046）：衰减主因是 forty 共线性（约 2/3），样本缩水约占 1/3。请裁决该分解是否可写入正文。

### 提速（permutation_p 缓存 Stage 1 + 向量化）
- `_stacked_fit_predict` 拆为 `_stacked_stage1` + `_stacked_stage2`；`permutation_p` 在循环外按 (repeat, fold) 缓存 Stage 1（回归 base_va/resid、分类 cross-fitted off_tr/off_va），循环内只跑 Stage 2；加断言：基线列（base_track_cols + raw_cols）与被置换 track 列无重叠（唯一调用点 M1 vs M0a：base_track_cols=[]、raw_cols=M0A_COLS，满足）。
- 直接执行测试证实：stage1+stage2 与原 fit_predict 逐位一致（回归+分类）；置换前后 Stage 1 逐位一致；向量化 transform 与旧循环逐位一致。
- 实测耗时：冒烟默认路径仍 24 分钟（主 CV 占主导），--within-year 探针 29 分钟；服务器全量 model_phase3 实际耗时见本轮回传（重点验证是否由 7h14m 降至 ~30–45 分钟）。
- 注意：`res["robustness_skill_s_peak"]` 仍取阶梯第 4 步（ladder[3]，全控制），第 5 步仅作分解对照，不取代 headline 行。

## 服务器第 5 轮全量关键数（待填：从 notes/phase3_results_round5_server.md 逐项抄录，严禁人工改写）

- Y2_quality：M0a / M0b_stopwatch / M0c_ngs / M0b / M0all / M1 / M1a / M1b / M1_track_only / M1_over_stopwatch / M1_over_all，Δ M1_over_stopwatch / Δ M1_track_only / Δ M1 vs M0a / Δ M1a vs M0a / Δ M1_over_all vs M0all（含 CI），置换 obs/p/null。
- Y1_hurdle_stage1 / Y1_stage2_logASR：同上。
- 对抗验证：ALL / FORTY_ONLY / SKILL_ONLY / STATIC_ONLY。
- 时序：Y1_year1_hurdle / Y1_year1_logsnaps 的 M0a 与 M1；LOCO_Y2（2023/2024/2025）的 M0a 与 M1。
- 定价：forty_s_peak / v_max_best / skill_s_peak 的 β；fdr_Y2 主族；稳健性阶梯 5 步（含第 5 步）。
- KTR：ktr_speed_mean；M2 decel 中介 indirect、burst_1s；M3 LightGBM Y2 r；Y2_sensitivity n_plays_100 / n_plays_150。
- DL 拆分：EDGE / IDL 的 skill_decel~game_decel 与 Y2 Spearman。
- 第 4 轮对照数（已核实，见 notes/phase3_results_round4_server.md）：Y2 M0a 0.1009、M0b_stopwatch −0.0016、M0c_ngs 0.1984、M0b 0.1756、M0all 0.1629、M1 0.1335、M1a 0.1403、M1b 0.1587、M1_track_only 0.0902、M1_over_stopwatch 0.0755、M1_over_all 0.1823；Δ M1_over_stopwatch +0.0771 CI [+0.0122, +0.1545]；Δ M1_track_only +0.0918 CI [−0.0076, +0.2217]；Δ M1 vs M0a +0.0326 CI [−0.0114, +0.0819]；置换 p=0.0030。对抗 0.4922/0.4924/0.4892/0.4879。时序 hurdle M1 0.8701、logsnaps 0.6200；LOCO 2023/2024/2025 M1 0.1537/0.1303/0.0496。定价 forty_s_peak β=+0.225、v_max_best +0.189、skill_s_peak +0.153（q=0.001）；阶梯三重 β=0.093、p=0.0886（n=252）。

## 请裁决的问题（第 5 轮，最终冻结确认）

1. **3 项验收是否全绿？** ① LOCO_Y2 与 2025 时序外推是否恢复干净（与第 4 轮被 Bug A 污染的数字对比）；② M0all 是否逼近或超过 M0c_ngs、delta_M1_over_all_vs_M0all 是否干净；③ ktr_speed_by_position 的 WR 保持率是否与 9.03/11.08=0.815 吻合。如全绿，请明确宣布 Phase 1–3 冻结。
2. **Y2 headline 增量最终引用**：M1_over_stopwatch（第 4 轮 Δ=+0.0771 CI 不含 0）与 M1a/M1 vs M0a 在第 5 轮数出来后如何并列呈现；M1_over_all vs M0all 修完 Bug B 后是否可引用（第 5 轮 delta 与 CI 为准）。
3. **LOCO/2025 时序的最终故事**：修完 Bug A 后 LOCO 与 2025 时序的 M1 数字如何写进 Limitations（2025 届 n=83、单季噪声大）与主结论（Double Dissociation 是否仍立得住）。
4. **稳健性阶梯第 5 步的分解**：样本缩水（β 0.149→0.124）vs forty 共线性（0.124→0.078）约 1:2 的分解，是否可写入正文；定稿措辞是否需要调整（第 4 轮已给的定稿措辞见 gemini_round4_20261122.md 第 6 节）。
5. **提速是否兑现**：全量 model_phase3 实际耗时 vs 7h14m；缓存 Stage 1 的做法是否可作为“可复现性”亮点写进 Method/附录；有无因缓存引入的统计口径风险（我们已用直接测试证实 Stage 1 逐位一致，请复核论证）。
6. **冻结后遗留**：Phase 3 冻结后，Phase 4 出图（Fig 4 森林图、Fig 5 KTR 热力图等）与 Phase 5 写作有无须在本轮先定的数字口径（如 per_position_spearman、Y2_sensitivity 新门槛的引用方式）；提交规格冲突（≤2,000 词 vs 15 页）与 Kaggle 身份验证两项我们只写进报告、不请你裁决（须二大爷另行拍板）。
