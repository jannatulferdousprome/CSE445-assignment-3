# ml_tools.py
"""
Tool registry for the Autonomous Local LLM ML Agent (CSE445 Assignment #3).

Task 1 (baseline):
    - load_dataset_summary
    - train_sklearn_model
    - train_pytorch_mlp

Task 2 (advanced tool expansion):
    - tune_hyperparameters            (GridSearchCV / RandomizedSearchCV for SVC & Decision Tree)
    - reduce_dimensionality           (PCA)
    - select_features                 (Sequential Feature Selection)
    - train_deep_pytorch_classifier   (Dropout + BatchNorm + LR scheduler)

Every tool returns a JSON string so the ReAct controller can safely embed the
result as an "Observation" and so the LLM can parse it back into structured
data if it needs to reason about it further.
"""

import json
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Shared dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def _get_dataset(name: str):
    name = name.lower().strip()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}")
    return DATASETS[name]()


# ---------------------------------------------------------------------------
# Task 1: Baseline tools
# ---------------------------------------------------------------------------
def load_dataset_summary(dataset_name: str) -> str:
    """Loads a standard benchmark dataset and returns summary statistics."""
    try:
        data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    summary = {
        "dataset": dataset_name.lower().strip(),
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Trains a Scikit-Learn model (decision_tree, logistic_regression, random_forest)."""
    try:
        data = _get_dataset(dataset_name)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    # --- self-healing friendly parameter validation ---
    if not (0.05 <= test_size <= 0.5):
        raise ValueError(
            f"Invalid test_size={test_size}. Must be between 0.05 and 0.5 "
            f"(shape mismatch risk if too large/small)."
        )

    model_type = model_type.lower().strip()

    # Logistic Regression benefits from (and often needs) standardized features
    # to converge cleanly within a reasonable number of iterations.
    features = StandardScaler().fit_transform(data.data) if model_type == "logistic_regression" \
        else data.data

    X_train, X_test, y_train, y_test = train_test_split(
        features, data.target, test_size=test_size, random_state=42, stratify=data.target
    )

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        raise ValueError(f"Unsupported model '{model_type}'. Options: decision_tree, "
                          f"logistic_regression, random_forest.")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(clf, features, data.target, cv=5)

    return json.dumps({
        "model": model_type,
        "dataset": dataset_name.lower().strip(),
        "test_accuracy": round(acc, 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    })


def train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50,
                       lr: float = 0.01) -> str:
    """Trains a simple PyTorch Multilayer Perceptron on the selected dataset."""
    data = _get_dataset(dataset_name)

    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}.")
    if epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs}.")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}.")

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    if final_loss is not None and np.isnan(final_loss):
        raise FloatingPointError(
            f"Training diverged: loss became NaN with lr={lr}, hidden_dim={hidden_dim}. "
            f"Try a smaller learning rate."
        )

    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch",
        "dataset": dataset_name.lower().strip(),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(acc, 4),
    })


# ---------------------------------------------------------------------------
# Task 2.1: Hyperparameter Tuning Tool (GridSearchCV / RandomizedSearchCV)
# ---------------------------------------------------------------------------
_PARAM_GRIDS = {
    "svc": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1],
        "kernel": ["rbf", "linear"],
    },
    "decision_tree": {
        "max_depth": [2, 3, 4, 5, 8, None],
        "min_samples_split": [2, 4, 6, 10],
        "criterion": ["gini", "entropy"],
    },
}


def tune_hyperparameters(dataset_name: str, model_type: str, search_type: str = "grid",
                          cv: int = 5, n_iter: int = 10) -> str:
    """
    Runs GridSearchCV or RandomizedSearchCV for a Support Vector Classifier
    (model_type='svc') or a Decision Tree (model_type='decision_tree').
    Returns the best hyperparameters and best cross-validated score.
    """
    data = _get_dataset(dataset_name)
    model_type = model_type.lower().strip()
    search_type = search_type.lower().strip()

    if model_type not in _PARAM_GRIDS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Options: "
                          f"{list(_PARAM_GRIDS.keys())}.")
    if search_type not in ("grid", "random"):
        raise ValueError(f"Unsupported search_type '{search_type}'. Options: grid, random.")
    if cv < 2:
        raise ValueError(f"cv must be >= 2, got {cv}.")

    base_model = SVC() if model_type == "svc" else DecisionTreeClassifier(random_state=42)
    param_grid = _PARAM_GRIDS[model_type]

    # Standardize features -- important for SVC, harmless for trees.
    X = StandardScaler().fit_transform(data.data) if model_type == "svc" else data.data
    y = data.target

    if search_type == "grid":
        search = GridSearchCV(base_model, param_grid, cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(base_model, param_grid, cv=cv, n_iter=n_iter,
                                     random_state=42, n_jobs=-1)

    search.fit(X, y)

    return json.dumps({
        "model_type": model_type,
        "search_type": search_type,
        "dataset": dataset_name.lower().strip(),
        "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                         for k, v in search.best_params_.items()},
        "best_cv_score": round(float(search.best_score_), 4),
        "cv_folds": cv,
    })


# ---------------------------------------------------------------------------
# Task 2.2: Feature Selection & Dimensionality Reduction Tool
# ---------------------------------------------------------------------------
def reduce_dimensionality(dataset_name: str, n_components: int = 2) -> str:
    """Applies PCA to the dataset and reports explained variance ratios."""
    data = _get_dataset(dataset_name)
    max_components = min(data.data.shape[0], data.data.shape[1])

    if not (1 <= n_components <= max_components):
        raise ValueError(
            f"n_components={n_components} is invalid for this dataset "
            f"(must be between 1 and {max_components})."
        )

    X_scaled = StandardScaler().fit_transform(data.data)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    return json.dumps({
        "dataset": dataset_name.lower().strip(),
        "n_components": n_components,
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "total_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
    })


def select_features(dataset_name: str, n_features_to_select: int = 3,
                     direction: str = "forward") -> str:
    """
    Runs Sequential Feature Selection with a Logistic Regression estimator
    and reports which original features were chosen, plus the resulting
    cross-validated accuracy.
    """
    data = _get_dataset(dataset_name)
    n_total_features = data.data.shape[1]
    direction = direction.lower().strip()

    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got '{direction}'.")
    if not (1 <= n_features_to_select < n_total_features):
        raise ValueError(
            f"n_features_to_select={n_features_to_select} is invalid "
            f"(dataset has {n_total_features} features)."
        )

    X_scaled = StandardScaler().fit_transform(data.data)
    y = data.target

    estimator = LogisticRegression(max_iter=1000, random_state=42)
    sfs = SequentialFeatureSelector(
        estimator, n_features_to_select=n_features_to_select,
        direction=direction, cv=5, n_jobs=-1,
    )
    sfs.fit(X_scaled, y)

    selected_mask = sfs.get_support()
    selected_names = [f for f, keep in zip(data.feature_names, selected_mask) if keep]

    X_selected = X_scaled[:, selected_mask]
    cv_scores = cross_val_score(estimator, X_selected, y, cv=5)

    return json.dumps({
        "dataset": dataset_name.lower().strip(),
        "direction": direction,
        "n_features_to_select": n_features_to_select,
        "selected_features": selected_names,
        "cv_mean_accuracy_with_selected_features": round(float(cv_scores.mean()), 4),
    })


# ---------------------------------------------------------------------------
# Task 2.3: Deep PyTorch Classifier with Regularization
# ---------------------------------------------------------------------------
class DeepMLP(nn.Module):
    """Configurable MLP with optional BatchNorm and Dropout on each hidden layer."""

    def __init__(self, in_features, hidden_dims, num_classes, dropout=0.3, use_batchnorm=True):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_deep_pytorch_classifier(dataset_name: str, hidden_dims=None, dropout: float = 0.3,
                                   use_batchnorm: bool = True, epochs: int = 100,
                                   lr: float = 0.01, scheduler_type: str = "step") -> str:
    """
    Trains a configurable deep PyTorch MLP with Dropout, BatchNorm, and a
    learning-rate scheduler (scheduler_type: 'step' or 'cosine').
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    if isinstance(hidden_dims, str):
        hidden_dims = json.loads(hidden_dims)

    data = _get_dataset(dataset_name)

    if not hidden_dims or any(h <= 0 for h in hidden_dims):
        raise ValueError(f"hidden_dims must be a list of positive integers, got {hidden_dims}.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}.")
    scheduler_type = scheduler_type.lower().strip()
    if scheduler_type not in ("step", "cosine"):
        raise ValueError(f"scheduler_type must be 'step' or 'cosine', got '{scheduler_type}'.")

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)

    model = DeepMLP(num_features, hidden_dims, num_classes, dropout=dropout,
                     use_batchnorm=use_batchnorm)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

    final_loss = loss_history[-1]
    if np.isnan(final_loss):
        raise FloatingPointError(
            f"Training diverged: loss became NaN with lr={lr}, hidden_dims={hidden_dims}. "
            f"Try a smaller learning rate or add more regularization."
        )

    model.eval()
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch-DeepMLP",
        "dataset": dataset_name.lower().strip(),
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "use_batchnorm": use_batchnorm,
        "scheduler_type": scheduler_type,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(acc, 4),
    })


# ---------------------------------------------------------------------------
# Registry mapping tool names -> callables (used by the ReAct controller)
# ---------------------------------------------------------------------------
AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_dimensionality": reduce_dimensionality,
    "select_features": select_features,
    "train_deep_pytorch_classifier": train_deep_pytorch_classifier,
}
