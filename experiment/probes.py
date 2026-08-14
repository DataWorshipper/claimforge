import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SMOTE

from experiment.spec import ExperimentSpec, ExperimentResult, DatasetResult, Probe, Variant
from experiment.datasets import load_dataset


CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "catboost": CatBoostClassifier,
}

REGRESSORS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
    "catboost": CatBoostRegressor,
}

MODEL_KWARGS = {
    "logistic_regression": lambda seed: {
        "max_iter": 1000,
        "random_state": seed,
    },
    "linear_regression": lambda seed: {},
    "random_forest": lambda seed: {
        "random_state": seed,
    },
    "gradient_boosting": lambda seed: {
        "random_state": seed,
    },
    "xgboost": lambda seed: {
        "random_state": seed,
    },
    "catboost": lambda seed: {
        "random_state": seed,
        "verbose": False,
    },
}

SUPPORTS_CLASS_WEIGHT = {
    "logistic_regression",
    "random_forest",
    "catboost",
}

TECHNIQUES = {
    "smote": lambda seed: [
        ("resample", SMOTE(random_state=seed))
    ],
    "adasyn": lambda seed: [
        ("resample", ADASYN(random_state=seed))
    ],
}

CLASSIFICATION_SCORERS = {
    "f1": make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0,
    ),
    "precision": make_scorer(
        precision_score,
        pos_label=1,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        pos_label=1,
        zero_division=0,
    ),
    "auc": make_scorer(
        roc_auc_score,
        response_method="predict_proba",
    ),
}

REGRESSION_SCORERS = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

DEFAULT_METRICS = {
    "classification": [
        "f1",
        "precision",
        "recall",
        "auc",
    ],
    "regression": [
        "rmse",
        "mae",
        "r2",
    ],
}


def model_registry(task_type):
    return (
        CLASSIFIERS
        if task_type == "classification"
        else REGRESSORS
    )


def scorer_registry(task_type):
    return (
        CLASSIFICATION_SCORERS
        if task_type == "classification"
        else REGRESSION_SCORERS
    )


def build_model(variant: Variant, task_type, seed):
    registry = model_registry(task_type)

    if variant.model not in registry:
        raise ValueError(
            f"{variant.model} is not a valid {task_type} model"
        )

    kwargs = dict(
        MODEL_KWARGS.get(
            variant.model,
            lambda s: {},
        )(seed)
    )

    if variant.class_weight:
        if (
            task_type != "classification"
            or variant.model not in SUPPORTS_CLASS_WEIGHT
        ):
            raise ValueError(
                f"{variant.model} ({task_type}) does not support class_weight"
            )

        kwargs["class_weight"] = variant.class_weight

    return registry[variant.model](**kwargs)


def build_pipeline(variant: Variant, task_type, seed):
    steps = [("scale", StandardScaler())]

    if variant.technique:
        if task_type != "classification":
            raise ValueError(
                f"technique '{variant.technique}' only applies to classification"
            )

        steps.extend(
            TECHNIQUES[variant.technique](seed)
        )

    steps.append(
        ("clf", build_model(variant, task_type, seed))
    )

    return Pipeline(steps)


def run_cv(
    pipeline,
    X,
    y,
    task_type,
    metrics,
    cv_folds,
    seeds,
):
    scorer_registry_instance = scorer_registry(task_type)

    scoring = {
        m: scorer_registry_instance[m]
        for m in metrics
    }

    scores = {
        m: []
        for m in metrics
    }

    for seed in range(seeds):
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=seed,
            )
        else:
            cv = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=seed,
            )

        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
        )

        for m in metrics:
            scores[m].extend(
                result[f"test_{m}"]
            )

    return {
        m: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }
        for m, v in scores.items()
    }


def resolve_metrics(
    spec: ExperimentSpec,
    task_type,
):
    metrics = (
        spec.metrics
        or DEFAULT_METRICS[task_type]
    )

    if spec.primary_metric not in metrics:
        metrics = metrics + [spec.primary_metric]

    return metrics


def measure(
    variant_a,
    variant_b,
    dataset,
    spec,
):
    metrics = resolve_metrics(
        spec,
        dataset.task_type,
    )

    pipeline_a = build_pipeline(
        variant_a,
        dataset.task_type,
        seed=0,
    )

    pipeline_b = build_pipeline(
        variant_b,
        dataset.task_type,
        seed=0,
    )

    metrics_a = run_cv(
        pipeline_a,
        dataset.X,
        dataset.y,
        dataset.task_type,
        metrics,
        spec.cv_folds,
        spec.seeds,
    )

    metrics_b = run_cv(
        pipeline_b,
        dataset.X,
        dataset.y,
        dataset.task_type,
        metrics,
        spec.cv_folds,
        spec.seeds,
    )

    delta = {
        m: metrics_b[m]["mean"] - metrics_a[m]["mean"]
        for m in metrics
    }

    return metrics_a, metrics_b, delta


def classify_tradeoff(
    delta,
    positive_metrics=("precision", "recall"),
):
    present = [
        m
        for m in positive_metrics
        if m in delta
    ]

    signs = [
        delta[m] > 0
        for m in present
    ]

    if not signs:
        return "unknown"

    if all(signs):
        return "pareto_improvement"

    if not any(signs):
        return "pareto_regression"

    return "tradeoff"


def run_ab_probe(
    spec: ExperimentSpec,
) -> ExperimentResult:
    per_dataset = []

    for name in spec.datasets:
        dataset = load_dataset(name)

        metrics_a, metrics_b, delta = measure(
            spec.variant_a,
            spec.variant_b,
            dataset,
            spec,
        )

        supported = (
            delta[spec.primary_metric]
            > spec.min_delta
        )

        per_dataset.append(
            DatasetResult(
                dataset=name,
                task_type=dataset.task_type,
                metrics_a=metrics_a,
                metrics_b=metrics_b,
                delta=delta,
                supported=supported,
            )
        )

    support_count = sum(
        1
        for r in per_dataset
        if r.supported
    )

    notes = (
        f"A={spec.variant_a}, "
        f"B={spec.variant_b}, "
        f"primary_metric={spec.primary_metric}"
    )

    return ExperimentResult(
        spec=spec,
        per_dataset=per_dataset,
        support_count=support_count,
        total_count=len(per_dataset),
        notes=notes,
    )


def run_leakage_check(
    spec: ExperimentSpec,
) -> ExperimentResult:
    if not spec.variant_b.technique:
        raise ValueError(
            "leakage_check requires variant_b.technique to be set"
        )

    per_dataset = []

    for name in spec.datasets:
        dataset = load_dataset(name)

        metrics = resolve_metrics(
            spec,
            dataset.task_type,
        )

        clean_pipeline = build_pipeline(
            spec.variant_b,
            dataset.task_type,
            seed=0,
        )

        clean_metrics = run_cv(
            clean_pipeline,
            dataset.X,
            dataset.y,
            dataset.task_type,
            metrics,
            spec.cv_folds,
            spec.seeds,
        )

        resampler = TECHNIQUES[
            spec.variant_b.technique
        ](0)[0][1]

        X_scaled = StandardScaler().fit_transform(dataset.X)

        X_leaked, y_leaked = resampler.fit_resample(
            X_scaled,
            dataset.y,
        )

        leaky_variant = Variant(
            model=spec.variant_b.model,
            class_weight=spec.variant_b.class_weight,
        )

        leaky_pipeline = build_pipeline(
            leaky_variant,
            dataset.task_type,
            seed=0,
        )

        leaky_metrics = run_cv(
            leaky_pipeline,
            X_leaked,
            y_leaked,
            dataset.task_type,
            metrics,
            spec.cv_folds,
            spec.seeds,
        )

        delta = {
            m: leaky_metrics[m]["mean"]
            - clean_metrics[m]["mean"]
            for m in metrics
        }

        leaked = (
            delta[spec.primary_metric]
            > spec.min_delta
        )

        per_dataset.append(
            DatasetResult(
                dataset=name,
                task_type=dataset.task_type,
                metrics_a=clean_metrics,
                metrics_b=leaky_metrics,
                delta=delta,
                supported=leaked,
            )
        )

    support_count = sum(
        1
        for r in per_dataset
        if r.supported
    )

    notes = (
        f"technique={spec.variant_b.technique}; "
        "A=in-fold(clean), "
        "B=global-resample-before-split(leaky)"
    )

    return ExperimentResult(
        spec=spec,
        per_dataset=per_dataset,
        support_count=support_count,
        total_count=len(per_dataset),
        notes=notes,
    )


def run_seed_variance_probe(
    spec: ExperimentSpec,
) -> ExperimentResult:
    per_dataset = []

    for name in spec.datasets:
        dataset = load_dataset(name)

        metrics_a, metrics_b, delta = measure(
            spec.variant_a,
            spec.variant_b,
            dataset,
            spec,
        )

        pooled_std = (
            metrics_a[spec.primary_metric]["std"]
            + metrics_b[spec.primary_metric]["std"]
        ) / 2

        effect_size = (
            delta[spec.primary_metric] / pooled_std
            if pooled_std > 1e-9
            else float("inf")
        )

        delta["effect_size"] = effect_size

        supported = (
            delta[spec.primary_metric] > spec.min_delta
            and abs(effect_size) >= spec.z_threshold
        )

        per_dataset.append(
            DatasetResult(
                dataset=name,
                task_type=dataset.task_type,
                metrics_a=metrics_a,
                metrics_b=metrics_b,
                delta=delta,
                supported=supported,
            )
        )

    support_count = sum(
        1
        for r in per_dataset
        if r.supported
    )

    notes = (
        f"z_threshold={spec.z_threshold}; "
        f"delta must exceed {spec.z_threshold}x pooled std "
        "to count as signal, not noise"
    )

    return ExperimentResult(
        spec=spec,
        per_dataset=per_dataset,
        support_count=support_count,
        total_count=len(per_dataset),
        notes=notes,
    )


def run_boundary_sweep(
    spec: ExperimentSpec,
) -> ExperimentResult:
    if (
        spec.sweep_param is None
        or spec.sweep_values is None
    ):
        raise ValueError(
            "boundary_sweep requires sweep_param and sweep_values"
        )

    if len(spec.datasets) != 1:
        raise ValueError(
            "boundary_sweep operates on exactly one parametrized dataset"
        )

    dataset_name = spec.datasets[0]
    per_dataset = []

    for value in spec.sweep_values:
        dataset = load_dataset(
            dataset_name,
            **{spec.sweep_param: value},
        )

        metrics_a, metrics_b, delta = measure(
            spec.variant_a,
            spec.variant_b,
            dataset,
            spec,
        )

        supported = (
            delta[spec.primary_metric]
            > spec.min_delta
        )

        per_dataset.append(
            DatasetResult(
                dataset=f"{dataset_name}[{spec.sweep_param}={value}]",
                task_type=dataset.task_type,
                metrics_a=metrics_a,
                metrics_b=metrics_b,
                delta=delta,
                supported=supported,
            )
        )

    support_count = sum(
        1
        for r in per_dataset
        if r.supported
    )

    notes = (
        f"swept {spec.sweep_param} "
        f"over {spec.sweep_values}"
    )

    return ExperimentResult(
        spec=spec,
        per_dataset=per_dataset,
        support_count=support_count,
        total_count=len(per_dataset),
        notes=notes,
    )


RUNNERS = {
    Probe.DIRECT_AB: run_ab_probe,
    Probe.STRENGTHEN_BASELINE: run_ab_probe,
    Probe.METRIC_DECOMPOSE: run_ab_probe,
    Probe.LEAKAGE_CHECK: run_leakage_check,
    Probe.SEED_VARIANCE: run_seed_variance_probe,
    Probe.BOUNDARY_SWEEP: run_boundary_sweep,
}


def run_probe(
    spec: ExperimentSpec,
) -> ExperimentResult:
    if spec.probe not in RUNNERS:
        raise ValueError(
            f"probe {spec.probe} not implemented yet"
        )

    return RUNNERS[spec.probe](spec)