# 04 - Telco Customer Churn Prediction（IBM 电信客户流失预测）

基于 IBM 电信客户示例数据集（`WA_Fn-UseC_-Telco-Customer-Churn.csv`，7,043 名客户 × 21 列）构建的端到端二分类流失预测与精准挽留决策项目。

- **交互式完整分析 Notebook**：[Telco_Customer_Churn_Analysis.ipynb](./Telco_Customer_Churn_Analysis.ipynb)（已完整执行并内嵌全部图表与结果表）

## 核心评估体系与方法亮点

1. **防数据泄露验证**：采用 `Churn × Contract` 组合分层 80/20 切分；全流程特征消融、不平衡策略对比、Optuna 调参、5×3 重复 CV 稳定性验证、概率校准、加权融合及业务阈值搜索均严格在开发集内层 CV / OOF 上完成，锁仓测试集（n = 1,409）仅作一次性盲测。
2. **四维评估**：
   - **排序能力**：ROC-AUC + PR-AUC（含 1,000 次 Bootstrap 95% 置信区间与配对显著性检验）；
   - **概率校准**：LogLoss / Brier Score / ECE + 可靠性曲线（Platt & Isotonic 校准对比）；
   - **业务收益**：Top-K% Lift 与累计增益、个体期望价值规则（`EV > 0`）、2D 成本敏感性热力图、风险 × 价值四象限分层；
   - **稳健与可解释**：子群切片公平性/校准审计、Logistic Regression Odds Ratios 与 TreeSHAP 全局/单客归因。
3. **工程流水线实测参考指标**（`Ensemble_Weighted_Blend`）：
   - Hold-out Test ROC-AUC：`0.8340`（95% CI: `[0.8123, 0.8560]`）| PR-AUC：`0.6276` | ECE：`0.0224`
   - 个体期望价值规则（`EV > 0`）：期望挽留净收益 **\$32,844.18**（触达 49.6% 客户，捕获 85.8% 流失客户，ROI **167.7%**）
