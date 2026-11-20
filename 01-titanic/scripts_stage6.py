"""第 6 阶段：多样性分析 + 集成 E1-E6（全部基于已保存的 OOF 概率）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from config import BASE
from cv import fold_lists_from_df, load_folds, log_experiment
from features import add_nonlabel_features  # noqa: F401 (kept for symmetry)

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
y = train["Survived"].values
folds = load_folds()
fold_list = fold_lists_from_df(folds)
today = str(date.today())

MODELS = ["lr", "svc", "knn", "rf", "et", "xgb", "lgbm", "cat"]
OOF = {m: np.load(os.path.join(BASE, "oof", f"tuned_{m}.npy")) for m in MODELS}
TESTP = {m: np.load(os.path.join(BASE, "test_preds", f"tuned_{m}.npy")) for m in MODELS}
# fold_metrics -> 25 折 acc（与 B6 一致的顺序）
FM = {m: pd.read_csv(os.path.join(BASE, "logs", f"fold_metrics_tuned_{m}.csv"))
      .sort_values(["rep", "fold"])["acc"].values for m in MODELS}
ACC = {m: float(FM[m].mean()) for m in MODELS}
print("单模型 25 折 acc:", {m: round(v, 4) for m, v in
      sorted(ACC.items(), key=lambda kv: -kv[1])})

# ---- 多样性分析：rep 平均 OOF 概率的 Pearson 相关 + 标签不一致率 ----
repmean = {m: OOF[m].mean(axis=0) for m in MODELS}
M = pd.DataFrame({m: repmean[m] for m in MODELS})
corr = M.corr()
print("OOF 概率 Pearson 相关矩阵：")
print(corr.round(3).to_string())
pred_lab = {m: (repmean[m] >= 0.5).astype(int) for m in MODELS}
dis = pd.DataFrame({m: pred_lab[m] for m in MODELS})
print("标签不一致率（两两）：")
pairs = {}
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        pairs[f"{a}-{b}"] = round(float((dis[a] != dis[b]).mean()), 3)
print(pairs)

# 相关 >0.95 只留分高的：按 acc 从高到低贪心保留，
# 新模型与已保留模型的最大相关 >0.95 时跳过（避免逐对剔除把全部模型连锁删光）。
SEL = []
for m in sorted(MODELS, key=lambda m: -ACC[m]):
    if all(corr.loc[m, s] <= 0.95 for s in SEL):
        SEL.append(m)
print("多样性筛选后入选:", SEL)


def fold_metrics_from_oof(oof):
    rows = []
    for i, (tr_idx, va_idx) in enumerate(fold_list):
        rep, fold = divmod(i, 5)
        p = oof[rep, va_idx]
        rows.append({"rep": rep, "fold": fold,
                     "acc": accuracy_score(y[va_idx], (p >= 0.5).astype(int)),
                     "logloss": log_loss(y[va_idx], p, labels=[0, 1]),
                     "auc": roc_auc_score(y[va_idx], p)})
    return pd.DataFrame(rows)


def register(exp_id, oof, testp, method, vs="B6_softvote_F0", note=""):
    fm = fold_metrics_from_oof(oof)
    fm.to_csv(os.path.join(BASE, "logs", f"fold_metrics_{exp_id}.csv"), index=False)
    np.save(os.path.join(BASE, "oof", f"{exp_id}.npy"), oof)
    np.save(os.path.join(BASE, "test_preds", f"{exp_id}.npy"), testp)
    s = {"acc_mean": float(fm["acc"].mean()), "acc_std": float(fm["acc"].std(ddof=1)),
         "acc_se": float(fm["acc"].std(ddof=1) / 5),
         "logloss_mean": float(fm["logloss"].mean()),
         "auc_mean": float(fm["auc"].mean())}
    print(f"{exp_id}: acc={s['acc_mean']:.4f}±{s['acc_std']:.4f} "
          f"logloss={s['logloss_mean']:.4f} auc={s['auc_mean']:.4f}", flush=True)
    log_experiment({"exp_id": exp_id, "date": today, "feature_set": "F_best",
                    "model": method, "params_file": "", "n_seeds": 5,
                    "cv_scheme": "RSKF 5x5 seed42 (OOF集成)",
                    "acc_mean": round(s["acc_mean"], 4),
                    "acc_std": round(s["acc_std"], 4),
                    "acc_se": round(s["acc_se"], 4),
                    "logloss_mean": round(s["logloss_mean"], 4),
                    "auc_mean": round(s["auc_mean"], 4), "group_cv_acc": "",
                    "vs": vs, "d_mean": "", "wins": "", "submitted": "no",
                    "public_lb": "", "note": note})
    return s


def mean_oof(members):
    return np.mean([OOF[m] for m in members], axis=0)


def mean_test(members):
    return np.mean([TESTP[m] for m in members], axis=0)


# ---- E1 等权软投票 ----
register("E1_softvote", mean_oof(SEL), mean_test(SEL),
         "soft-vote-equal", note=f"成员: {SEL}")
# 诊断（不登记为候选，避免扩大候选池带来选择偏差）：8 模型全量等权
_diag = fold_metrics_from_oof(mean_oof(MODELS))
print("诊断 8模型全量等权软投票: acc=%.4f±%.4f logloss=%.4f auc=%.4f"
      % (_diag["acc"].mean(), _diag["acc"].std(ddof=1),
         _diag["logloss"].mean(), _diag["auc"].mean()))

# ---- E2 硬投票（奇数个：取 acc 前 5 名；OOF 存投票比例，acc 与多数表决一致）----
top5 = sorted(ACC, key=lambda m: -ACC[m])[:5]
oof_hard = np.zeros((5, len(y)))
for rep in range(5):
    votes = np.mean([(OOF[m][rep] >= 0.5).astype(int) for m in top5], axis=0)
    oof_hard[rep] = votes
test_hard = np.mean([(TESTP[m] >= 0.5).astype(int) for m in top5], axis=0)
register("E2_hardvote5", oof_hard, test_hard,
         "hard-vote-5", note=f"成员(acc前5): {top5}；OOF存投票比例")

# ---- E3 加权软投票（权重折内求：每折用其余 24 折 OOF 优化 LogLoss）----
def _opt_weights(fold_idx_list):
    # fold_idx_list: 折序号列表（fold_list 中的下标）；拼成样本级矩阵
    P, yy = [], []
    for i in fold_idx_list:
        rep, fold = divmod(i, 5)
        _, va_idx = fold_list[i]
        P.append(np.column_stack([OOF[m][rep, va_idx] for m in SEL]))
        yy.append(y[va_idx])
    P = np.vstack(P)
    yy = np.concatenate(yy)
    P = np.clip(P, 1e-6, 1 - 1e-6)
    k = len(SEL)
    def nll(w):
        p = np.clip(P @ w, 1e-6, 1 - 1e-6)
        return float(log_loss(yy, p, labels=[0, 1]))
    res = minimize(nll, np.ones(k) / k, method="SLSQP",
                   bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x if res.success else np.ones(k) / k

oof_e3 = np.zeros((5, len(y)))
w_all = []
for i, (tr_idx, va_idx) in enumerate(fold_list):
    rep, fold = divmod(i, 5)
    others = [j for j in range(25) if j != i]
    w = _opt_weights(others)
    w_all.append(w)
    oof_e3[rep, va_idx] = np.column_stack([OOF[m][rep, va_idx] for m in SEL]) @ w
w_global = _opt_weights(list(range(25)))
test_e3 = np.column_stack([TESTP[m] for m in SEL]) @ w_global
print("E3 全局权重:", dict(zip(SEL, np.round(w_global, 3))),
      "逐折权重均值:", dict(zip(SEL, np.round(np.mean(w_all, axis=0), 3))))
register("E3_weighted", oof_e3, test_e3, "soft-vote-weighted",
         note=f"权重折内求；全局权重 {dict(zip(SEL, np.round(w_global, 3)))}")

# ---- E4 Rank 平均 ----
from scipy.stats import rankdata
oof_e4 = np.zeros((5, len(y)))
for rep in range(5):
    ranks = np.mean([rankdata(OOF[m][rep]) / len(y) for m in SEL], axis=0)
    oof_e4[rep] = ranks
test_e4 = np.mean([rankdata(TESTP[m]) / len(TESTP[m]) for m in SEL], axis=0)
register("E4_rankavg", oof_e4, test_e4, "rank-average",
         note=f"成员: {SEL}")

# ---- E5 Stacking（元模型强正则 LR，C=0.5；外层 25 折评估元模型）----
X_meta = np.column_stack([repmean[m] for m in SEL])  # (891, k)
oof_e5 = np.zeros((5, len(y)))
for i, (tr_idx, va_idx) in enumerate(fold_list):
    rep, fold = divmod(i, 5)
    meta = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta.fit(X_meta[tr_idx], y[tr_idx])
    oof_e5[rep, va_idx] = meta.predict_proba(X_meta[va_idx])[:, 1]
meta_full = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
meta_full.fit(X_meta, y)
test_e5 = meta_full.predict_proba(
    np.column_stack([TESTP[m] for m in SEL]))[:, 1]
register("E5_stacking", oof_e5, test_e5, "stacking-LR-C0.5",
         note="元模型外层25折评估；基模型OOF本身已是折外预测，"
              "元模型训练折的基模型见过验证折样本，分数略乐观，见复盘")

# ---- E6 贪心前向选择（可重复加入；外折验证：按 rep 留一验证贪心过程）----
def _acc_of_members(members, fold_idx_list):
    """给定成员 multiset，在指定折上的平均 acc（OOF 概率平均后阈值 0.5）。"""
    accs = []
    for i in fold_idx_list:
        rep, fold = divmod(i, 5)
        _, va_idx = fold_list[i]
        p = np.mean([OOF[m][rep, va_idx] for m in members], axis=0)
        accs.append(accuracy_score(y[va_idx], (p >= 0.5).astype(int)))
    return float(np.mean(accs))

def greedy(fold_idx_list):
    best_single = max(MODELS, key=lambda m: _acc_of_members([m], fold_idx_list))
    members = [best_single]
    best = _acc_of_members(members, fold_idx_list)
    while True:
        cand_best, cand_acc = None, best
        for m in MODELS:
            a = _acc_of_members(members + [m], fold_idx_list)
            if a > cand_acc + 1e-9:
                cand_best, cand_acc = m, a
        if cand_best is None:
            break
        members.append(cand_best)
        best = cand_acc
    return members, best

# 外折验证：每次留一个 rep 做验证
outer_accs = []
for r in range(5):
    val_folds = [i for i in range(25) if divmod(i, 5)[0] == r]
    tr_folds = [i for i in range(25) if divmod(i, 5)[0] != r]
    members, _ = greedy(tr_folds)
    a = _acc_of_members(members, val_folds)
    outer_accs.append(a)
    print(f"E6 外折验证 rep{r}: 选{sorted(set(members))} (x{len(members)}) "
          f"验证acc={a:.4f}", flush=True)
print("E6 贪心外折验证平均 acc=%.4f" % float(np.mean(outer_accs)))
members_all, best_all = greedy(list(range(25)))
print("E6 全量贪心:", members_all, "OOF acc=%.4f" % best_all)
oof_e6 = np.zeros((5, len(y)))
for rep in range(5):
    oof_e6[rep] = np.mean([OOF[m][rep] for m in members_all], axis=0)
test_e6 = np.mean([TESTP[m] for m in members_all], axis=0)
register("E6_greedy", oof_e6, test_e6, "greedy-forward",
         note=f"成员 {members_all}；外折验证平均acc={float(np.mean(outer_accs)):.4f}（选择乐观偏差参考）")

print("STAGE6 DONE")
