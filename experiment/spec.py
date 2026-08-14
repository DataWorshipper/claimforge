from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Probe(str, Enum):
    DIRECT_AB = "direct_ab"
    STRENGTHEN_BASELINE = "strengthen_baseline"
    LEAKAGE_CHECK = "leakage_check"
    METRIC_DECOMPOSE = "metric_decompose"
    SEED_VARIANCE = "seed_variance"
    BOUNDARY_SWEEP = "boundary_sweep"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    SCOPE_LIMITED = "scope_limited"


@dataclass
class Variant:
    model: str
    class_weight: Optional[str] = None
    technique: Optional[str] = None


@dataclass
class ExperimentSpec:
    probe: Probe
    datasets: List[str]
    variant_a: Variant
    variant_b: Variant
    primary_metric: str
    metrics: Optional[List[str]] = None
    cv_folds: int = 5
    seeds: int = 10
    min_delta: float = 0.0
    z_threshold: float = 1.0
    sweep_param: Optional[str] = None
    sweep_values: Optional[list] = None


@dataclass
class DatasetResult:
    dataset: str
    task_type: str
    metrics_a: dict
    metrics_b: dict
    delta: dict
    supported: bool
    n_samples: int
    n_features: int
    minority_ratio: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    sweep_value: Optional[float] = None


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    per_dataset: List[DatasetResult]
    support_count: int
    total_count: int
    notes: str = ""