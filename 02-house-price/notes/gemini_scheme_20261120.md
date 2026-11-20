# 房价预测 · Gemini Neon 方案评估（2026-11-20）

来源：Gemini 网页版 Neon 档（思考等级最高），对话 https://gemini.google.com/app/e96da97654553410
数据：train 1460×81，test 1459×80，目标 SalePrice，指标 RMSLE。SalePrice 偏度 1.88。

## 1. 数据处理
- 结构性缺失（无此设施）：PoolQC、MiscFeature、Alley、Fence、FireplaceQu、MasVnrType、车库组、地下室组 → 类别填 "None"，对应面积/数量填 0。GarageYrBlt 线性分支用 YearBuilt 填 + HasGarage 标记。
- 脏缺失交叉校验：面积 >0 但质量列缺失的，按众数填，不填 "None"。
- 真缺失：LotFrontage 折内按 Neighborhood 分组中位数填；Electrical 众数 SBrkr。
- 测试集特有缺失：MSZoning、Utilities、Functional、Exterior1st/2nd、KitchenQual、SaleType → Functional 填 "Typ"，MSZoning 按 MSSubClass 分组众数，其余训练集众数；Utilities 近常量，删除。
- 编码：MSSubClass 转类别；MoSold 转类别或周期编码。质量量表有序映射（None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5）；BsmtExposure、BsmtFinType、Functional、GarageFinish、Fence、LotShape、LandSlope、PavedDrive、CentralAir 均有序映射。无序类别：线性/SVR 分支 One-Hot + 稀有类合并 "Rare" + Neighborhood 折内 Target Encoding；树分支 CatBoost 原生类别，LGBM/XGB 原生类别或折内平滑 TE + 频次编码。
- 目标 log1p 变换（与 RMSLE 对齐）；偏态特征变换只对线性/SVR 分支（Yeo-Johnson + RobustScaler），树分支不做。
- 异常值：删 2 个已知脏样本（GrLivArea>4000 且 SalePrice<30 万）；不做机械大面积删点，其余靠 log1p / RobustScaler / Huber 消化。

## 2. 模型与集成
- 不只押 GBDT：对数房价有强加性线性结构，树模型不能外推尾部。
- 主力：正则化线性（ElasticNet、Ridge、Huber）+ SVR（RBF）+ GBDT（CatBoost 首选，XGB/LGBM 强正则：浅树 3–4 层、低采样）。
- 融合：OOF 非负加权平均或强正则 Ridge 元学习器；线性/SVR 家族与 GBDT 家族权重大致各半。不用复杂 Stacking。

## 3. 验证
- 10 折分层（按 SalePrice 分位桶分层），可 Repeated 3×10；监控 log 尺度 RMSE（=RMSLE），另看折间标准差、train/val 差距、低价段残差。
- 所有统计量（分组中位数、λ、Scaler、TE）必须折内 fit，禁止 train+test 合并预处理。不迷信 Public LB（只占测试集约一半）。

## 4. 特征工程（7 个重点）
1. TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF；TotalFinishedSF（扣 BsmtUnfSF）
2. TotalBath = FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath
3. HouseAge = YrSold − YearBuilt；RemodAge；IsRemodeled
4. Qual_x_TotalSF = OverallQual × TotalSF；OverallQual × OverallCond
5. Neighborhood 相对特征（面积/品质对街区中位数的比值与差）+ 折内平滑 TE
6. TotalPorchSF（五种门廊面积合并）+ HasPool/Has2ndFloor/HasGarage/HasBsmt/HasFireplace
7. TotalQualScore（各质量评分有序映射后求和）

## 5. 预期与风险
- Baseline 0.135–0.145；进阶单模 0.118–0.125；完整方案 CV 0.110–0.116（真实测试约 0.113–0.119）。榜单前排有泄漏作弊的 0.000，不作基准；无泄漏做到 0.115 内已是扎实竞赛级。
- 风险：高维小样本过拟合（编码后 250–300 维 vs 1460 行）；RMSLE 对低价房相对误差惩罚重，要盯低价段残差；测试集类别/缺失漂移，流水线要全列防御性填充与未知类别兜底。
