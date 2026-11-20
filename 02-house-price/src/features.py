"""In-fold preprocessing for House Prices.

HousePreprocessor is a single sklearn transformer: every statistic (modes,
group medians, neighborhood aggregates, target encoding, category codes) is
fit ONLY on the training fold. Target encoding and any y-dependent stat use
an internal out-of-fold computation inside fit_transform so the training
rows themselves are encoded without self-leakage.

cat_encoding:
  'onehot'  -> nominal cats left as strings (downstream OneHotEncoder)
  'ordinal' -> nominal cats mapped to in-fold integer codes (unseen -> -1)
  'native'  -> nominal cats left as strings (CatBoost cat_features)
garage_yr: 'yearbuilt' (linear branch) or 'zero' (tree branches)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

import config as C


class HousePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, cat_encoding="ordinal", garage_yr="zero", groups=None,
                 te_smoothing=10.0):
        self.cat_encoding = cat_encoding
        self.garage_yr = garage_yr
        # groups: list of engineered-group names to KEEP; None = keep all
        self.groups = groups
        self.te_smoothing = te_smoothing

    # ---------- helpers ----------
    def _basic_clean(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in C.DROP_COLS if c in X.columns])
        if C.IDCOL in X.columns:
            X = X.drop(columns=[C.IDCOL])
        X["MSSubClass"] = X["MSSubClass"].astype(str)
        X["MoSold"] = X["MoSold"].astype(str)
        return X

    def _structural_fill(self, X, modes):
        X = X.copy()
        # cross-check dirty missing first (uses raw indicator values)
        for cat_col, ind_col in C.CROSS_CHECK.items():
            if cat_col in X.columns:
                bad = X[cat_col].isna() & (X[ind_col].fillna(0) > 0)
                if bad.any():
                    X.loc[bad, cat_col] = modes.get(cat_col, "TA")
        for col in C.STRUCT_CAT:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in C.STRUCT_NUM:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        # GarageYrBlt
        if "GarageYrBlt" in X.columns:
            if self.garage_yr == "yearbuilt":
                X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
            else:
                X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)
        return X

    def _engineer(self, X, nbhd_stats=None, te_map=None, global_te=0.0):
        """Add engineered features. nbhd_stats/te_map None -> computed columns
        left for caller (fit path computes them)."""
        X = X.copy()
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalFinishedSF"] = X["TotalSF"] - X["BsmtUnfSF"]
        X["TotalBath"] = (X["FullBath"] + 0.5 * X["HalfBath"]
                          + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"])
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
        X["Qual_x_TotalSF"] = X["OverallQual"] * X["TotalSF"]
        X["Qual_x_Cond"] = X["OverallQual"] * X["OverallCond"]
        X["TotalPorchSF"] = (X["WoodDeckSF"] + X["OpenPorchSF"] + X["EnclosedPorch"]
                             + X["3SsnPorch"] + X["ScreenPorch"])
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        # TotalQualScore from mapped ordinal quality columns (already mapped
        # by the time _engineer is called in the pipeline order below)
        qcols = ["ExterQual", "KitchenQual", "BsmtQual", "GarageQual",
                 "HeatingQC", "FireplaceQu"]
        X["TotalQualScore"] = X[qcols].sum(axis=1)
        if nbhd_stats is not None:
            med_sf = X["Neighborhood"].map(nbhd_stats["med_sf"])
            mean_q = X["Neighborhood"].map(nbhd_stats["mean_q"])
            med_sf = med_sf.fillna(nbhd_stats["global_med_sf"])
            mean_q = mean_q.fillna(nbhd_stats["global_mean_q"])
            X["NbhdSF_ratio"] = X["TotalSF"] / med_sf.replace(0, np.nan)
            X["NbhdSF_ratio"] = X["NbhdSF_ratio"].fillna(1.0)
            X["NbhdSF_diff"] = X["TotalSF"] - med_sf
            X["QualNbhdDiff"] = X["OverallQual"] - mean_q
        if te_map is not None:
            X["NeighborhoodTE"] = X["Neighborhood"].map(te_map).fillna(global_te)
        return X

    def _ordinal_map(self, X):
        X = X.copy()
        for col, mp in C.ORDINAL_MAPS.items():
            if col in X.columns:
                X[col] = X[col].map(mp)
                X[col] = X[col].fillna(0).astype(float)
        return X

    # ---------- fit ----------
    def fit(self, X, y=None):
        X = self._basic_clean(X)
        # modes for every object column (from raw, pre-fill)
        self.modes_ = {}
        for col in X.columns:
            if X[col].dtype == object:
                m = X[col].mode(dropna=True)
                self.modes_[col] = m.iloc[0] if len(m) else "None"
        # MSZoning group mode by MSSubClass
        if "MSZoning" in X.columns:
            gm = (X.dropna(subset=["MSZoning"])
                    .groupby("MSSubClass")["MSZoning"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else self.modes_["MSZoning"]))
            self.mszoning_group_mode_ = gm.to_dict()
        else:
            self.mszoning_group_mode_ = {}
        # LotFrontage group medians by Neighborhood
        lf = X.dropna(subset=["LotFrontage"])
        self.lotfrontage_med_ = lf.groupby("Neighborhood")["LotFrontage"].median().to_dict()
        self.lotfrontage_global_ = float(lf["LotFrontage"].median())
        # medians for remaining numeric cols
        self.num_median_ = X.select_dtypes(include=[np.number]).median().to_dict()
        # work through fill+ordinal+engineer (no nbhd/te yet) to get TotalSF etc.
        Xw = self._structural_fill(X, self.modes_)
        Xw = self._fill_rest(Xw)
        Xw = self._ordinal_map(Xw)
        Xw = self._engineer(Xw)
        # neighborhood aggregates (X-only stats, safe to fit on full train fold)
        g = Xw.groupby("Neighborhood")
        self.nbhd_stats_ = {
            "med_sf": g["TotalSF"].median().to_dict(),
            "mean_q": g["OverallQual"].mean().to_dict(),
            "global_med_sf": float(Xw["TotalSF"].median()),
            "global_mean_q": float(Xw["OverallQual"].mean()),
        }
        # target encoding stats (full-fold mapping for transform path)
        # NOTE: y passed in is already log1p(SalePrice)
        if y is not None:
            ylog = np.asarray(y, dtype=float)
            self.global_te_ = float(np.mean(ylog))
            df = pd.DataFrame({"nb": Xw["Neighborhood"].values, "y": ylog})
            agg = df.groupby("nb")["y"].agg(["sum", "count"])
            m = self.te_smoothing
            self.te_map_ = ((agg["sum"] + m * self.global_te_)
                            / (agg["count"] + m)).to_dict()
        else:
            self.global_te_ = 0.0
            self.te_map_ = {}
        # nominal category codes (ordinal branch)
        self.cat_codes_ = {}
        for col in C.NOMINAL_COLS:
            if col in Xw.columns:
                cats = sorted(Xw[col].astype(str).unique().tolist())
                self.cat_codes_[col] = {c: i for i, c in enumerate(cats)}
        self.columns_ = None  # set in _finalize
        self._fitted_X_ = Xw  # engineered, pre-encoding frame (for fit_transform)
        return self

    def _fill_rest(self, X):
        X = X.copy()
        # LotFrontage by neighborhood median
        if "LotFrontage" in X.columns:
            med = X["Neighborhood"].map(self.lotfrontage_med_)
            X["LotFrontage"] = X["LotFrontage"].fillna(med).fillna(self.lotfrontage_global_)
        # MSZoning group mode
        if "MSZoning" in X.columns and X["MSZoning"].isna().any():
            gm = X["MSSubClass"].map(self.mszoning_group_mode_)
            X["MSZoning"] = X["MSZoning"].fillna(gm).fillna(self.modes_.get("MSZoning", "RL"))
        # remaining categorical NaN -> mode; numeric NaN -> median
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype == object:
                    X[col] = X[col].fillna(self.modes_.get(col, "None"))
                else:
                    X[col] = X[col].fillna(self.num_median_.get(col, 0))
        return X

    def _finalize(self, Xw):
        """Apply nbhd/te + group selection + nominal encoding."""
        X = self._engineer(Xw, nbhd_stats=self.nbhd_stats_,
                           te_map=self.te_map_, global_te=self.global_te_)
        keep_groups = self.groups if self.groups is not None else list(C.FEATURE_GROUPS)
        drop_eng = [c for g, cols in C.FEATURE_GROUPS.items()
                    if g not in keep_groups for c in cols]
        X = X.drop(columns=[c for c in drop_eng if c in X.columns])
        if self.cat_encoding == "ordinal":
            for col, mp in self.cat_codes_.items():
                if col in X.columns:
                    X[col] = X[col].astype(str).map(mp).fillna(-1).astype(float)
        # 'onehot'/'native': leave strings
        return X

    def fit_transform(self, X, y=None, **kw):
        self.fit(X, y)
        Xw = self._fitted_X_
        out = self._finalize(Xw)
        # OOF target encoding for the training rows (avoid self-leakage)
        if y is not None and "NeighborhoodTE" in out.columns:
            ylog = np.asarray(y, dtype=float)  # already log1p target
            nb = Xw["Neighborhood"].values
            oof = np.full(len(out), self.global_te_)
            kf = KFold(5, shuffle=True, random_state=C.SEED)
            idx = np.arange(len(out))
            for tr_i, va_i in kf.split(idx):
                df = pd.DataFrame({"nb": nb[tr_i], "y": ylog[tr_i]})
                agg = df.groupby("nb")["y"].agg(["sum", "count"])
                m = self.te_smoothing
                mp = ((agg["sum"] + m * self.global_te_)
                      / (agg["count"] + m)).to_dict()
                oof[va_i] = pd.Series(nb[va_i]).map(mp).fillna(self.global_te_).values
            out["NeighborhoodTE"] = oof
        self.columns_ = list(out.columns)
        self.cat_cols_ = [c for c in C.NOMINAL_COLS if c in out.columns
                          and out[c].dtype == object]
        self.num_cols_ = [c for c in out.columns if c not in self.cat_cols_]
        return out

    def transform(self, X):
        X = self._basic_clean(X)
        X = self._structural_fill(X, self.modes_)
        X = self._fill_rest(X)
        X = self._ordinal_map(X)
        X = self._engineer(X)
        out = self._finalize(X)
        # align columns exactly with fit_transform output
        for c in self.columns_:
            if c not in out.columns:
                out[c] = 0
        return out[self.columns_]
