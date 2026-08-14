from dataclasses import dataclass, field
from typing import Dict, List, Optional
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


CUSTOM_DATASETS: Dict[str, Dataset] = {}


def register_csv_dataset(
    path,
    target_column,
    task_type,
    positive_label=None,
    name="user_data",
):
    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got '{task_type}'"
        )

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in CSV columns: {list(df.columns)}"
        )

    if df[target_column].isna().any():
        raise ValueError(
            f"target column '{target_column}' has missing values - clean the CSV first"
        )

    feature_df = df.drop(columns=[target_column])

    if feature_df.isna().any().any():
        bad_columns = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(
            f"columns {bad_columns} have missing values - "
            "clean or impute them in the CSV before using it"
        )

    encoded = pd.get_dummies(feature_df, drop_first=True)
    non_numeric = encoded.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"could not convert columns to numeric after one-hot encoding: {non_numeric}. "
            "Drop or reformat these columns in the CSV before using it."
        )

    X = encoded.to_numpy(dtype=float)
    y_raw = df[target_column]

    if task_type == "classification":
        uniques = list(y_raw.unique())

        if len(uniques) != 2:
            raise ValueError(
                f"binary classification requires exactly 2 classes in '{target_column}', "
                f"found {len(uniques)}: {uniques}"
            )

        if positive_label is None:
            positive_label = y_raw.value_counts().idxmin()
        elif str(positive_label) not in [str(u) for u in uniques]:
            raise ValueError(
                f"positive_label '{positive_label}' not among classes {uniques}"
            )

        y = np.where(y_raw.astype(str) == str(positive_label), 1, 0)
    else:
        y = y_raw.to_numpy(dtype=float)

    dataset = Dataset(
        name=name,
        task_type=task_type,
        X=X,
        y=y,
        tags=["user_provided"],
        minority_ratio=minority_ratio(y) if task_type == "classification" else None,
    )

    CUSTOM_DATASETS[name] = dataset
    return dataset


def list_datasets():
    entries = []

    for reg_name, (task_type, _, tags) in REGISTRY.items():
        entries.append({
            "name": reg_name,
            "task_type": task_type,
            "tags": tags,
            "user_provided": False,
        })

    for custom_name, dataset in CUSTOM_DATASETS.items():
        entries.append({
            "name": custom_name,
            "task_type": dataset.task_type,
            "tags": dataset.tags,
            "user_provided": True,
            "n_samples": dataset.X.shape[0],
            "n_features": dataset.X.shape[1],
            "minority_ratio": dataset.minority_ratio,
        })

    return entries


def load_dataset(name: str, **kwargs) -> Dataset:
    if name in CUSTOM_DATASETS:
        return CUSTOM_DATASETS[name]

    if name not in REGISTRY:
        raise ValueError(
            f"unknown dataset {name}, choices: {list(REGISTRY) + list(CUSTOM_DATASETS)}"
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