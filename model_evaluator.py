import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

def train_and_evaluate_models(df, target_column, selected_models):
    """Train selected ML models, return accuracies, training times, and performance metrics."""

    # Split dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Available models
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    models = {name: model_dict[name] for name in selected_models}

    accuracies = {}
    training_times = {}
    metrics = {}

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_times[name] = round(time.time() - start_time, 2)  # Record training time

        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

        # Store performance metrics
        metrics[name] = {
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)  # Store matrix separately
        }

    best_model = max(accuracies, key=accuracies.get)
    return accuracies, best_model, training_times, metrics


def get_feature_importance(df, target_column):
    """Get feature importance from Random Forest model."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Detect whether it's classification or regression
    if y.nunique() < 10:  # Assuming classification if unique values < 10
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)

    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return sorted_features
