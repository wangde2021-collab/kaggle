# NFL Big Data Bowl 2027

- 比赛：https://www.kaggle.com/competitions/nfl-big-data-bowl-2027
- 形式：**writeup 评审制**（研究质量 40 / 创新 30 / 呈现 20 / 可视化 10，总分 100），15 页英文报告，Hackathon 每队限提交 1 次
- 赛道：University Track（在校大学生）
- 截止：2027-01-07 07:59（GMT+8）
- 奖金：$100,000（Open Track $45,000 + University Track $45,000 + Grand Prize $10,000）
- 账号：dewang2025（规则已接受，2026-11-21）
- **绝不提交**：任何提交（writeup）都须二大爷另行点头

## 目录

- `data/`：9 个 CSV（2.31 GB），只读
- `notes/problem_statement.md`：题目原文（Overview / Evaluation / Data 全文）
- `notes/rules.md`：规则记录（双赛道、单次提交、开源许可、数据使用限制）
- `notes/data_profile.md`：数据画像（行数、列、缺失、样例、关键分布、跨表链接）
- `notes/gemini_scheme.md`：Gemini 最高模型给出的获奖方案（第三步）
- `notes/gemini_review_rounds.md`：五轮 Gemini 核对记录（第五步）
- `notes/final_report.md`：最终报告（第六步）
- `src/`：分析代码
- `logs/`：运行日志与 `experiments.csv` 实验记录

## 环境

- 本地云端 VM：Python 3.12.3、pandas 2.1.4、numpy 1.26.4、scipy 1.11.4、scikit-learn 1.9.1、lightgbm 4.7.0
- 东京服务器 tokyo-node-work：跑计算时 e2-standard-8（8 核 32G），闲时 e2-standard-2；venv 版本与本地对齐（numpy 1.26.4 / pandas 2.1.4 / scipy 1.11.4）
- 服务器目录：/home/wangde2021/03-nfl-big-data-bowl

## 状态

- 2026-11-21 02:00 开工：Gemini 额度已重置；数据已下载；画像已完成；等 Gemini 方案。
- 房价项目在服务器上还在跑（SERVER_RUN_DONE 不存在），NFL 计算须等它跑完再启动。
