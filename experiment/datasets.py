from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer as sk_load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
    fetch_openml,
    make_classification,
    make_regression,
)


@dataclass
class Dataset:
    name: str
    task_type: str
    X: np.ndarray
    y: np.ndarray
    tags: List[str] = field(default_factory=list)
    minority_ratio: Optional[float] = None


def minority_ratio(y):
    counts = np.bincount(y.astype(int))
    return float(counts.min() / counts.sum())


def load_breast_cancer():
    data = sk_load_breast_cancer()
    return data.data, data.target


def load_credit_g():
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    X = pd.get_dummies(data.data, drop_first=True).to_numpy(dtype=float)
    y = np.where(data.target == "bad", 1, 0)
    return X, y


def load_pima_diabetes():
    data = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
    y = np.where(data.target == "tested_positive", 1, 0)
    return data.data, y


def load_synthetic_classification(
    imbalance_ratio=0.1,
    n_samples=2000,
    n_features=20,
    seed=0,
):
    weights = [1 - imbalance_ratio, imbalance_ratio]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        weights=weights,
        random_state=seed,
    )
    return X, y


def load_diabetes_regression():
    data = load_diabetes()
    return data.data, data.target


def load_california_housing():
    data = fetch_california_housing()
    return data.data, data.target


def load_synthetic_regression(
    n_samples=2000,
    n_features=20,
    noise=10.0,
    seed=0,
):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=seed,
    )
    return X, y


REGISTRY = {
    "breast_cancer": (
        "classification",
        load_breast_cancer,
        ["medical", "balanced"],
    ),
    "credit_g": (
        "classification",
        load_credit_g,
        ["finance", "imbalanced"],
    ),
    "pima_diabetes": (
        "classification",
        load_pima_diabetes,
        ["medical", "imbalanced"],
    ),
    "synthetic_classification": (
        "classification",
        load_synthetic_classification,
        ["synthetic", "tunable"],
    ),
    "diabetes_regression": (
        "regression",
        load_diabetes_regression,
        ["medical", "small"],
    ),
    "california_housing": (
        "regression",
        load_california_housing,
        ["real_estate", "large"],
    ),
    "synthetic_regression": (
        "regression",
        load_synthetic_regression,
        ["synthetic", "tunable"],
    ),
}


def load_dataset(name: str, **kwargs) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(
            f"unknown dataset {name}, choices: {list(REGISTRY)}"
        )

    task_type, loader, tags = REGISTRY[name]

    X, y = loader(**kwargs)
    X, y = np.asarray(X), np.asarray(y)

    minority_ratio_value = (
        minority_ratio(y) if task_type == "classification" else None
    )

    return Dataset(
        name=name,
        task_type=task_type,
        X=X,
        y=y,
        tags=tags,
        minority_ratio=minority_ratio_value,
    )


CLASSIFICATION_PORTFOLIO = [
    "breast_cancer",
    "credit_g",
    "pima_diabetes",
]

REGRESSION_PORTFOLIO = [
    "diabetes_regression",
    "california_housing",
]


def default_portfolio(task_type: str) -> List[str]:
    if task_type == "classification":
        return list(CLASSIFICATION_PORTFOLIO)

    if task_type == "regression":
        return list(REGRESSION_PORTFOLIO)

    raise ValueError(f"unknown task_type {task_type}")