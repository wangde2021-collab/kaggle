# Titanic 比赛规则（2026-11-19 核查）

来源：https://www.kaggle.com/competitions/titanic/rules（2026-11-19 抓取）、
https://www.kaggle.com/competitions/titanic/overview（抓取于 2026-11-16，文本较简）

## 已确认

- 评估指标：Accuracy（比赛说明；Overview 本次抓取文本未直接渲染指标行，以比赛 Evaluation 页为准）。
- 每日提交上限：每天最多 10 次。
- 最终提交：最多可选 5 个最终提交用于评判。
- 比赛无结束日期，滚动排行榜；Public LB 基于公开测试集，Private LB 基于私有测试集（具体划分不公开）。
- 团队上限 10 人；一人只能用一个账号提交。
- 外部数据：允许使用，但必须是公开、所有参赛者都能免费获取的数据。
- 禁止事项：禁止手工标注（hand-labeling）验证集/测试集记录；禁止使用泄漏的真实乘客生还名单；不要违背比赛精神作弊。
- 奖项：知识类入门赛，不授积分或奖牌，无奖金。
- 参赛人数（Overview 显示）：1,481,573 报名，11,445 参赛者，42,165 次提交。

## 提交格式

- PassengerId + Survived 两列，418 行 + 表头，Survived 为整数 0/1。

## 本轮约定

- 本轮只做到生成版本化提交文件，不提交到 Kaggle（需二大爷另外点头）。
- 不使用任何外部真实乘客名单；特征只用 train/test 自带列派生。
