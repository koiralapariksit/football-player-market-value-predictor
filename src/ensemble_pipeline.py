"""
Football Ensemble Pipeline Class.
Encapsulates preprocessor and tree models in a single scikit-learn compatible estimator.
"""

from sklearn.base import BaseEstimator, RegressorMixin


class FootballEnsemblePipeline(BaseEstimator, RegressorMixin):
    """
    Weighted Super-Ensemble combining LightGBM (40%), XGBoost (35%), and HistGradientBoosting (25%).
    """
    def __init__(self, preprocessor=None, m_lgb=None, m_xgb=None, m_hgb=None):
        self.preprocessor = preprocessor
        self.m_lgb = m_lgb
        self.m_xgb = m_xgb
        self.m_hgb = m_hgb

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
        return 0.40 * p1 + 0.35 * p2 + 0.25 * p3
