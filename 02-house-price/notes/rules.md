# 比赛规则要点（House Prices - Advanced Regression Techniques）

- 任务：回归，预测 test 中每套房的 SalePrice。
- 评估指标：RMSLE = sqrt(mean((log(1+y) - log(1+yhat))^2))，越低越好。本项目在 log1p 尺度上训练，折分数 = log 尺度 RMSE（预测截断到 >=0 后等价）。
- 提交格式：Id,SalePrice，共 1459 行（与 test.csv 的 Id 完全一致）。
- 训练集 1460 行 × 81 列，测试集 1459 行 × 80 列。
- 验证方案（本项目冻结）：按 SalePrice 十分位分桶分层，10 折 × 3 次重复 = 30 折，seed 42，存 data/folds/folds.csv，所有实验只读此文件。
- 预处理全部折内 fit（Pipeline/自定义 Transformer），禁止 train+test 合并统计。
- 已知脏样本 2 个（Id 524、1299：GrLivArea>4000 且 SalePrice<300000），仅从训练折删除，验证折保留。
- 本轮只产出提交文件，不提交 Kaggle（等用户点头）。
