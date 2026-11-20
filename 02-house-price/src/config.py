"""House Prices competition config. Regression, target SalePrice, metric RMSLE."""
import os

SEED = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
TARGET = "SalePrice"
IDCOL = "Id"
N_REPS = 3
N_SPLITS = 10

DROP_COLS = ["Utilities"]

# structural missing -> "None" (categorical)
STRUCT_CAT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]
# structural missing -> 0 (numeric areas/counts)
STRUCT_NUM = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "GarageArea", "GarageCars",
]
# dirty-missing cross-check: cat col NaN but indicator > 0 -> mode fill instead of "None"
CROSS_CHECK = {
    "BsmtQual": "TotalBsmtSF", "BsmtCond": "TotalBsmtSF",
    "BsmtExposure": "TotalBsmtSF", "BsmtFinType1": "TotalBsmtSF",
    "BsmtFinType2": "TotalBsmtSF",
    "MasVnrType": "MasVnrArea",
    "FireplaceQu": "Fireplaces",
    "GarageType": "GarageArea", "GarageFinish": "GarageArea",
    "GarageQual": "GarageArea", "GarageCond": "GarageArea",
}

QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    "ExterQual": QUAL_MAP, "ExterCond": QUAL_MAP, "BsmtQual": QUAL_MAP,
    "BsmtCond": QUAL_MAP, "HeatingQC": QUAL_MAP, "KitchenQual": QUAL_MAP,
    "FireplaceQu": QUAL_MAP, "GarageQual": QUAL_MAP, "GarageCond": QUAL_MAP,
    "PoolQC": QUAL_MAP,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}
ORDINAL_COLS = list(ORDINAL_MAPS.keys())

# columns treated as nominal categories (MSSubClass / MoSold forced to str)
NOMINAL_COLS = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "Electrical", "GarageType", "MiscFeature",
    "SaleType", "SaleCondition", "MoSold",
]

# engineered feature groups (ablation units)
FEATURE_GROUPS = {
    "G1_area": ["TotalSF", "TotalFinishedSF"],
    "G2_bath": ["TotalBath"],
    "G3_age": ["HouseAge", "RemodAge", "IsRemodeled"],
    "G4_inter": ["Qual_x_TotalSF", "Qual_x_Cond"],
    "G5_neigh": ["NbhdSF_ratio", "NbhdSF_diff", "QualNbhdDiff", "NeighborhoodTE"],
    "G6_porch": ["TotalPorchSF", "HasPool", "Has2ndFloor", "HasGarage", "HasBsmt", "HasFireplace"],
    "G7_qual": ["TotalQualScore"],
}
ALL_ENG = [c for cols in FEATURE_GROUPS.values() for c in cols]

DIRTY = dict(grlivarea=4000, price=300000)  # drop from training folds only
