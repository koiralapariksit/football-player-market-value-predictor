"""
Football Ensemble Pipeline Class.
Encapsulates preprocessor and tree models in a single scikit-learn compatible estimator.
"""

from sklearn.base import BaseEstimator, RegressorMixin


class FootballEnsemblePipeline(BaseEstimator, RegressorMixin):
    """
    Weighted Super-Ensemble combining LightGBM, XGBoost, and HistGradientBoosting.
    """
    def __init__(self, preprocessor=None, m_lgb=None, m_xgb=None, m_hgb=None, w_lgb=0.40, w_xgb=0.35, w_hgb=0.25):
        self.preprocessor = preprocessor
        self.m_lgb = m_lgb
        self.m_xgb = m_xgb
        self.m_hgb = m_hgb
        self.w_lgb = w_lgb
        self.w_xgb = w_xgb
        self.w_hgb = w_hgb

    def fit(self, X, y):
        X_trans = self.preprocessor.fit_transform(X)
        self.m_lgb.fit(X_trans, y)
        self.m_xgb.fit(X_trans, y)
        self.m_hgb.fit(X_trans, y)
        return self

    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        p1 = self.m_lgb.predict(X_trans)
        p2 = self.m_xgb.predict(X_trans)
        p3 = self.m_hgb.predict(X_trans)
        return self.w_lgb * p1 + self.w_xgb * p2 + self.w_hgb * p3
