import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin


"""
 Classification Algorithms
"""

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns a Logistic Regression model."""
    model = LogisticRegression(C=0.8, solver='liblinear', max_iter=500, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Logistic Regression CV Accuracy: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


def train_RandomForest_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_split=10, min_samples_leaf=4, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Random Forest CV Accuracy: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


def train_XGBoost_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns an XGBoost Classifier."""
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, min_child_weight=3, 
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.1, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"XGBoost CV Accuracy: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


"""
 Regression Algorithms
"""

def train_Linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns a Linear Regression model."""
    model = LinearRegression()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"Linear Regression CV R² Score: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


def train_RandomForest_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_split=8, min_samples_leaf=4, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"Random Forest CV R² Score: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


def train_XGBoost_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns an XGBoost Regressor."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, min_child_weight=3, 
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=0.1, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"XGBoost CV R² Score: {np.mean(scores):.4f}")
    model.fit(X_train, y_train)
    return model


"""
 Clustering Algorithms
"""

def train_KMeans(X_scaled: pd.DataFrame, num_clusters: int) -> ClusterMixin:
    """Trains and returns a K-Means clustering model."""
    model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, random_state=42)
    model.fit(X_scaled)
    return model


def train_dbscan(X_scaled: pd.DataFrame) -> ClusterMixin:
    """Trains and returns a DBSCAN clustering model."""
    model = DBSCAN(eps=0.7, min_samples=4)
    model.fit(X_scaled)
    return model


def train_heirarchical(X_scaled: pd.DataFrame, num_clusters: int) -> ClusterMixin:
    """Trains and returns a Hierarchical Clustering model."""
    model = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete')
    model.fit(X_scaled)
    return model
