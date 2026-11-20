"""特征工程：G1-G9（不含标签的统计可在 train+test 合并后算；含标签的 G8 必须折内算）。

add_nonlabel_features：生成全部非标签派生列（含 G6 的 AgeMissing/IsChild 标志，
但 Age 的分组中位数填补不在这里做，交给 TitleAgeImputer 在折内 fit）。
TitleAgeImputer / GroupSurvivalEncoder：sklearn 风格 Transformer，只在训练折上 fit。
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Capt": "Rare", "Sir": "Rare", "Lady": "Rare", "Don": "Rare",
    "Dona": "Rare", "Jonkheer": "Rare", "Countess": "Rare",
}

RARE_TITLES = set(TITLE_MAP.values())


def _title(name):
    try:
        t = name.split(",")[1].split(".")[0].strip()
    except Exception:
        return "Rare"
    return TITLE_MAP.get(t, t)


def _ticket_prefix(ticket):
    s = str(ticket).replace(".", "").replace("/", "").strip()
    letters = "".join(ch for ch in s.split(" ")[0] if ch.isalpha())
    return letters.upper() if letters else "NUM"


def add_nonlabel_features(train, test):
    """输入原始 train/test，返回带派生列的 (train, test)。
    TicketFreq / TicketPrefix 低频合并 / Fare 缺失按 Pclass 中位数填：
    都是不含标签的统计，在 train+test 合并后计算（比赛允许的常见做法）。"""
    tr = train.copy()
    te = test.copy()
    for df in (tr, te):
        df["Surname"] = df["Name"].str.split(",").str[0].str.strip()
        df["Title"] = df["Name"].map(_title)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FamilyBucket"] = pd.cut(
            df["FamilySize"], bins=[0, 1, 4, 99], labels=["1", "2-4", "5+"]
        ).astype(str)
        df["TicketPrefix"] = df["Ticket"].map(_ticket_prefix)
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].fillna("U").str[0].replace({"T": "U"})
        df["AgeMissing"] = df["Age"].isna().astype(int)
        df["IsChild"] = ((df["Age"] < 14) | (df["Title"] == "Master")).astype(int)
        df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
        df["NameLength"] = df["Name"].str.len()
    # TicketFreq：train+test 合并计数
    combined_ticket = pd.concat([tr["Ticket"], te["Ticket"]])
    freq = combined_ticket.value_counts()
    tr["TicketFreq"] = tr["Ticket"].map(freq)
    te["TicketFreq"] = te["Ticket"].map(freq)
    # TicketPrefix 低频（合并后 <5 次）并为 OTHER
    prefix_counts = pd.concat([tr["TicketPrefix"], te["TicketPrefix"]]).value_counts()
    rare_prefix = set(prefix_counts[prefix_counts < 5].index)
    tr["TicketPrefix"] = tr["TicketPrefix"].replace(list(rare_prefix), "OTHER")
    te["TicketPrefix"] = te["TicketPrefix"].replace(list(rare_prefix), "OTHER")
    # Fare：test 缺 1 个，用同 Pclass 中位数（合并数据算）填，再算派生列
    combined_fare = pd.concat([tr[["Pclass", "Fare"]], te[["Pclass", "Fare"]]])
    fare_med = combined_fare.groupby("Pclass")["Fare"].median()
    for df in (tr, te):
        df["Fare"] = df["Fare"].fillna(df["Pclass"].map(fare_med))
        df["FarePerPerson"] = df["Fare"] / df["TicketFreq"]
        df["LogFare"] = np.log1p(df["Fare"])
    return tr, te


class TitleAgeImputer(BaseEstimator, TransformerMixin):
    """G6：按 (Title, Pclass) 分组中位数填 Age，退到 Title 中位数，再退到全局中位数。
    只在训练折上 fit，杜绝泄漏。"""

    def fit(self, X, y=None):
        df = X[["Title", "Pclass", "Age"]]
        self.global_med_ = df["Age"].median()
        self.title_med_ = df.groupby("Title")["Age"].median().to_dict()
        self.tp_med_ = df.groupby(["Title", "Pclass"])["Age"].median().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        miss = X["Age"].isna()
        if miss.any():
            filled = X.loc[miss].apply(
                lambda r: self.tp_med_.get(
                    (r["Title"], r["Pclass"]),
                    self.title_med_.get(r["Title"], self.global_med_),
                ),
                axis=1,
            )
            X.loc[miss, "Age"] = filled
        return X


class GroupSurvivalEncoder(BaseEstimator, TransformerMixin):
    """G8：组生还率，按 Surname+Ticket 分组，组内其他人在训练折中的生还率。
    平滑：(sum + m*prior) / (n + m)，prior=0.5，m=5；无信息组填 0.5。
    只在训练折上 fit（fit 时用训练折标签），transform 时只查表。"""

    def __init__(self, prior=0.5, m=5):
        self.prior = prior
        self.m = m

    @staticmethod
    def _key(df):
        return df["Surname"].astype(str) + "|" + df["Ticket"].astype(str)

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        keys = self._key(X)
        tmp = pd.DataFrame({"k": keys.values, "y": y.values})
        g = tmp.groupby("k")["y"].agg(["sum", "count"])
        self.map_ = (
            (g["sum"] + self.m * self.prior) / (g["count"] + self.m)
        ).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X["GroupSurv"] = self._key(X).map(self.map_).fillna(self.prior)
        return X
