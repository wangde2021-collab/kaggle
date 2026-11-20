"""Model builders. Three branches:
- linear: HousePreprocessor(onehot) -> Yeo-Johnson + RobustScaler (numeric),
          OneHot(min_frequency=10) (cats) -> linear/SVR model
- tree:   GBDT wrappers around HousePreprocessor(ordinal) with internal
          early stopping on a 12% carve-out of the training fold
          (or fixed iterations when fixed_iters is set, for full-train fits)
- cat:    CatBoostWrapper with HousePreprocessor(native), same early-stop idea
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR

import config as C
from features import HousePreprocessor


def _num_selector(df):
    return [c for c in df.columns if df[c].dtype != object]


def _cat_selector(df):
    return [c for c in df.columns if df[c].dtype == object]


def linear_pipeline(model, groups=None):
    num = Pipeline([
        ("yj", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("sc", RobustScaler()),
    ])
    enc = ColumnTransformer([
        ("num", num, _num_selector),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10),
         _cat_selector),
    ])
    return Pipeline([
        ("prep", HousePreprocessor(cat_encoding="onehot", garage_yr="yearbuilt",
                                   groups=groups)),
        ("enc", enc),
        ("model", model),
    ])


def _eval_split(X, y, frac=0.12, seed=C.SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_ev = max(30, int(len(X) * frac))
    ev, tr = idx[:n_ev], idx[n_ev:]
    return X.iloc[tr], y[tr], X.iloc[ev], y[ev]


class _GBDTBase(BaseEstimator, RegressorMixin):
    def __init__(self, groups=None, fixed_iters=None, **params):
        self.groups = groups
        self.fixed_iters = fixed_iters
        self.params = params

    def get_params(self, deep=True):
        p = {"groups": self.groups, "fixed_iters": self.fixed_iters}
        p.update(self.params)
        return p

    def set_params(self, **kw):
        for k in ("groups", "fixed_iters"):
            if k in kw:
                setattr(self, k, kw.pop(k))
        self.params.update(kw)
        return self

    def _prep(self, X, y):
        self.prep_ = HousePreprocessor(cat_encoding="ordinal", garage_yr="zero",
                                       groups=self.groups)
        return self.prep_.fit_transform(X, y)


class LGBMWrapper(_GBDTBase):
    def fit(self, X, y):
        from lightgbm import LGBMRegressor, early_stopping
        Xt = self._prep(X, y)
        p = dict(learning_rate=0.03, num_leaves=15, max_depth=4,
                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=5.0, random_state=C.SEED,
                 verbose=-1, n_jobs=4)
        p.update(self.params)
        if self.fixed_iters:
            self.model_ = LGBMRegressor(n_estimators=self.fixed_iters, **p)
            self.model_.fit(Xt, y)
            self.best_iter_ = self.fixed_iters
        else:
            Xtr, ytr, Xev, yev = _eval_split(Xt, np.asarray(y))
            self.model_ = LGBMRegressor(n_estimators=3000, **p)
            self.model_.fit(Xtr, ytr, eval_set=[(Xev, yev)],
                            callbacks=[early_stopping(100, verbose=False)])
            self.best_iter_ = self.model_.best_iteration_
        return self

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))


class XGBWrapper(_GBDTBase):
    def fit(self, X, y):
        from xgboost import XGBRegressor
        Xt = self._prep(X, y)
        p = dict(learning_rate=0.03, max_depth=3, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=3, reg_lambda=5.0,
                 random_state=C.SEED, n_jobs=4)
        p.update(self.params)
        if self.fixed_iters:
            self.model_ = XGBRegressor(n_estimators=self.fixed_iters, **p)
            self.model_.fit(Xt, y)
            self.best_iter_ = self.fixed_iters
        else:
            Xtr, ytr, Xev, yev = _eval_split(Xt, np.asarray(y))
            self.model_ = XGBRegressor(n_estimators=3000,
                                       early_stopping_rounds=100, **p)
            self.model_.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
            self.best_iter_ = self.model_.best_iteration
        return self

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))


class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, groups=None, fixed_iters=None, **cb_params):
        self.groups = groups
        self.fixed_iters = fixed_iters
        self.cb_params = cb_params

    def get_params(self, deep=True):
        p = {"groups": self.groups, "fixed_iters": self.fixed_iters}
        p.update(self.cb_params)
        return p

    def set_params(self, **kw):
        for k in ("groups", "fixed_iters"):
            if k in kw:
                setattr(self, k, kw.pop(k))
        self.cb_params.update(kw)
        return self

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.prep_ = HousePreprocessor(cat_encoding="native", garage_yr="zero",
                                       groups=self.groups)
        Xt = self.prep_.fit_transform(X, y)
        params = dict(loss_function="RMSE", random_seed=C.SEED, verbose=0,
                      allow_writing_files=False)
        params.update(self.cb_params)
        if self.fixed_iters:
            params["iterations"] = self.fixed_iters
            self.model_ = CatBoostRegressor(**params)
            self.model_.fit(Xt, y, cat_features=self.prep_.cat_cols_)
            self.best_iter_ = self.fixed_iters
        else:
            params.setdefault("iterations", 3000)
            params["od_type"] = "Iter"
            params["od_wait"] = 100
            Xtr, ytr, Xev, yev = _eval_split(Xt, np.asarray(y))
            self.model_ = CatBoostRegressor(**params)
            self.model_.fit(Xtr, ytr, eval_set=(Xev, yev), use_best_model=True,
                            cat_features=self.prep_.cat_cols_)
            self.best_iter_ = self.model_.get_best_iteration()
        return self

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))


# ---- factories (params=None -> documented defaults) ----

def make_ridge(params=None, groups=None):
    p = dict(alpha=1.0)
    if params: p.update(params)
    return linear_pipeline(Ridge(**p), groups)


def make_elasticnet(params=None, groups=None):
    p = dict(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=C.SEED)
    if params: p.update(params)
    return linear_pipeline(ElasticNet(**p), groups)


def make_huber(params=None, groups=None):
    p = dict(alpha=0.01, epsilon=1.35, max_iter=500)
    if params: p.update(params)
    return linear_pipeline(HuberRegressor(**p), groups)


def make_svr(params=None, groups=None):
    p = dict(C=10.0, gamma="scale", epsilon=0.05)
    if params: p.update(params)
    return linear_pipeline(SVR(**p), groups)


def make_lgbm(params=None, groups=None, fixed_iters=None):
    p = dict(learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)
    if params: p.update(params)
    return LGBMWrapper(groups=groups, fixed_iters=fixed_iters, **p)


def make_xgb(params=None, groups=None, fixed_iters=None):
    p = dict(learning_rate=0.03, max_depth=3, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=3, reg_lambda=5.0)
    if params: p.update(params)
    return XGBWrapper(groups=groups, fixed_iters=fixed_iters, **p)


def make_cat(params=None, groups=None, fixed_iters=None):
    p = dict(learning_rate=0.03, depth=5, l2_leaf_reg=5.0)
    if params: p.update(params)
    return CatBoostWrapper(groups=groups, fixed_iters=fixed_iters, **p)
