import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib

class SMFIFModel:
    def __init__(self):
        self.selected_features = ["Cr", "GGT", "UA", "TP", "apoA1", "ALT", "TP", "ADA",
                                  "Glu", "DBIL", "Cl-", "GLo", "ALP", "eGFR", "HDL", "C4"]
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        rf = RandomForestClassifier(n_estimators=600, max_depth=100,
                                    min_samples_split=30, min_samples_leaf=25,
                                    max_features='sqrt')
        lgbm = lgb.LGBMClassifier(random_state=0, num_leaves=12, learning_rate=0.05,
                                  n_estimators=85, max_depth=-1)
        svc = SVC(C=10, kernel='rbf', gamma='auto', class_weight='balanced',
                  probability=True, random_state=0)
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                            learning_rate=0.05, n_estimators=61, max_depth=5,
                            subsample=0.5, colsample_bytree=0.5)
        knn = KNeighborsClassifier(n_neighbors=11, weights='uniform',
                                   metric='euclidean', algorithm='kd_tree', p=1)
        estimators = [('rf', rf), ('lgbm', lgbm), ('svm', svc), ('xgb', xgb), ('knn', knn)]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    def fit(self, df):
        X = df[self.selected_features]
        y = df['lable']
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, df):
        X = df[self.selected_features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df):
        X = df[self.selected_features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path='smfif_model.pkl'):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path='smfif_model.pkl'):
        bundle = joblib.load(path)
        self.model = bundle['model']
        self.scaler = bundle['scaler']