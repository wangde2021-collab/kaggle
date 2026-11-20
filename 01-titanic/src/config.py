"""Titanic 新流程：全局配置与特征集定义。"""
import os

SEED = 42
BASE = os.path.expanduser("~/workspace/kaggle-data/titanic")
DATA_RAW = os.path.join(BASE, "data", "raw")
FOLDS_PATH = os.path.join(BASE, "data", "folds", "folds.csv")

# 上轮 7 特征 F0（vote_ensemble.py / baseline.py）
F0_NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
F0_CAT = ["Sex", "Embarked"]

# 特征组（技能第 4 阶段）：(新增数值列, 新增类别列)
GROUPS = {
    "G1": ([], ["Title"]),
    "G2": (["FamilySize", "IsAlone"], ["FamilyBucket"]),
    "G3": (["TicketFreq"], ["TicketPrefix"]),
    "G4": (["FarePerPerson", "LogFare"], []),
    "G5": (["HasCabin"], ["Deck"]),
    "G6": (["AgeMissing", "IsChild"], []),  # 另附 Title/Pclass 分组中位数填 Age（折内 Transformer）
    "G7": ([], ["Sex_Pclass"]),
    "G8": (["GroupSurv"], []),  # 折内目标编码，高风险，单独决策
    "G9": (["NameLength"], []),
}

GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9"]  # G8 不在主线顺序内

# 最终特征集 F_best：消融结束后填入
F_BEST_NUM = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
F_BEST_CAT = ['Sex', 'Embarked', 'Title', 'Sex_Pclass']
F_BEST_USE_G8 = False
F_BEST_USE_GROUP_AGE = False
F_BEST_PCLASS_AS_CAT = False
