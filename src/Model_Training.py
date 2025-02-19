import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin


"""
 Classification Algorithms
"""

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns a Logistic Regression model."""
    model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_RandomForest_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_XGBoost_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Trains and returns an XGBoost Classifier."""
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


"""
 Regression Algorithms
"""

def train_Linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_RandomForest_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_XGBoost_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Trains and returns an XGBoost Regressor."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


"""
 Clustering Algorithms
"""

def train_KMeans(X_scaled: pd.DataFrame, num_clusters: int) -> ClusterMixin:
    """Trains and returns a K-Means clustering model."""
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X_scaled)
    return model


def train_dbscan(X_scaled: pd.DataFrame) -> ClusterMixin:
    """Trains and returns a DBSCAN clustering model."""
    model = DBSCAN(eps=0.5, min_samples=5)
    model.fit(X_scaled)
    return model


def train_heirarchical(X_scaled: pd.DataFrame, num_clusters: int) -> ClusterMixin:
    """Trains and returns a Hierarchical Clustering model."""
    model = AgglomerativeClustering(n_clusters=num_clusters)
    model.fit(X_scaled)
    return model
