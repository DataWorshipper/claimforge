# Calibration check

Three claims with known ground truth, used to sanity-check that the system's verdicts
actually track reality instead of being noise. If claims 1 and 2 land anywhere other than
their expected verdict, that's a signal something in the pipeline is broken, not that the
claim was surprising.

| # | Claim | Ground truth | Expected verdict | Actual verdict | Notes |
|---|-------|---------------|-------------------|-----------------|-------|
| 1 | Does XGBoost achieve a higher F1 score than plain logistic regression on tabular binary classification data? | Well-established consensus: yes, generally | `supported` (possibly with a small-N exception) | Proposer: `supported` (literature). Skeptic: `scope_limited` (experiments) | Default (untuned) XGBoost lost to logistic regression on breast_cancer (-0.014) and pima_diabetes (-0.029), ~tied on credit_g (+0.004). n_features sweep on synthetic data: helps at 10-20 features (+0.10 to +0.20), reverses at 100 (-0.181) - not monotonic. Reads as correct, not a calibration failure: no probe tunes hyperparameters, and breast_cancer is a near-ceiling-effect dataset where any model does well. The tool caught more nuance than the "obvious" consensus claim assumed. |
| 2 | Does applying SMOTE to the whole dataset before splitting into train and test give a trustworthy, leakage-free estimate of its performance improvement? | No - textbook data leakage mistake | `refuted` | *(not yet run)* | |
| 3 | Does SMOTE improve F1 more than class-weighting on imbalanced binary classification data? | Genuinely disputed in the literature | `contested` or `scope_limited` | Proposer: `scope_limited`. Skeptic: `refuted` | direct_ab on user_data/credit_g/pima_diabetes: flat-to-negative everywhere. boundary_sweep across imbalance_ratio 0.05-0.40: consistently not supported, no crossover found. Matches expectation - genuine disagreement between agents, no clean universal answer. |

Fill in claim 2 after that run finishes.
