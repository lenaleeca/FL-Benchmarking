from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


def compute_binary_prob_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    prefix: str = "",
    n_boot: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 1111,
    sparse_n_threshold: int = 20,
) -> Dict[str, float]:
    """
    Compute binary classification metrics from predicted probabilities.

    Reporting rules:
        1. If both classes are present:
           - estimate AUC, AUPRC, and 95% CIs.
           - if n < sparse_n_threshold, flag as sparse.

        2. If only one outcome class is present:
           - AUC and AUPRC are not estimable.
           - return NaN for AUC/AUPRC and their CIs.

    Brier score and log loss are calculated in all non-empty datasets.
    Log loss uses labels=[0, 1] so sklearn does not crash in one-class strata.
    """
    yt = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(y_prob).astype(float).reshape(-1)

    if len(yt) != len(p):
        raise ValueError("y_true and y_prob must have the same length")

    if len(yt) == 0:
        raise ValueError("y_true and y_prob must not be empty")

    def k(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    metrics: Dict[str, float] = {}

    p_clip = np.clip(p, 1e-7, 1 - 1e-7)

    n = int(len(yt))
    n_positive = int(np.sum(yt == 1))
    n_negative = int(np.sum(yt == 0))

    has_both_classes = n_positive > 0 and n_negative > 0
    sparse_subgroup = n < sparse_n_threshold
    can_estimate_auc = has_both_classes

    metrics[k("n")] = n
    metrics[k("n_positive")] = n_positive
    metrics[k("n_negative")] = n_negative
    metrics[k("one_class_outcome")] = not has_both_classes
    metrics[k("sparse_subgroup")] = sparse_subgroup
    metrics[k("sparse_n_threshold")] = int(sparse_n_threshold)

    if can_estimate_auc:
        metrics[k("auc")] = float(roc_auc_score(yt, p))
        metrics[k("auprc")] = float(average_precision_score(yt, p))

        if sparse_subgroup:
            metrics[k("metric_note")] = (
                f"AUC/AUPRC estimated but flagged as sparse because subgroup "
                f"n={n} is below {sparse_n_threshold}."
            )
        else:
            metrics[k("metric_note")] = ""
    else:
        metrics[k("auc")] = float("nan")
        metrics[k("auprc")] = float("nan")
        metrics[k("metric_note")] = (
            "AUC/AUPRC not estimable because y_true contains only one "
            f"outcome class (n={n}, positives={n_positive}, negatives={n_negative})."
        )

    metrics[k("brier")] = float(brier_score_loss(yt, p))

    metrics[k("logloss")] = float(
        log_loss(
            yt,
            p_clip,
            labels=[0, 1],
        )
    )

    alpha = 1 - ci_level
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)

    rng = np.random.default_rng(random_state)

    boot_auc = []
    boot_auprc = []
    boot_brier = []
    boot_logloss = []

    if can_estimate_auc:
        pos_idx = np.where(yt == 1)[0]
        neg_idx = np.where(yt == 0)[0]

        for _ in range(n_boot):
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

            yt_b = yt[boot_idx]
            p_b = p[boot_idx]
            p_b_clip = np.clip(p_b, 1e-7, 1 - 1e-7)

            boot_auc.append(roc_auc_score(yt_b, p_b))
            boot_auprc.append(average_precision_score(yt_b, p_b))
            boot_brier.append(brier_score_loss(yt_b, p_b))
            boot_logloss.append(
                log_loss(
                    yt_b,
                    p_b_clip,
                    labels=[0, 1],
                )
            )
    else:
        boot_idx_base = np.arange(n)

        for _ in range(n_boot):
            boot_idx = rng.choice(boot_idx_base, size=n, replace=True)

            yt_b = yt[boot_idx]
            p_b = p[boot_idx]
            p_b_clip = np.clip(p_b, 1e-7, 1 - 1e-7)

            boot_brier.append(brier_score_loss(yt_b, p_b))
            boot_logloss.append(
                log_loss(
                    yt_b,
                    p_b_clip,
                    labels=[0, 1],
                )
            )

    metrics[k("auc_ci_lower")] = (
        float(np.percentile(boot_auc, lower_q)) if boot_auc else float("nan")
    )
    metrics[k("auc_ci_upper")] = (
        float(np.percentile(boot_auc, upper_q)) if boot_auc else float("nan")
    )

    metrics[k("auprc_ci_lower")] = (
        float(np.percentile(boot_auprc, lower_q)) if boot_auprc else float("nan")
    )
    metrics[k("auprc_ci_upper")] = (
        float(np.percentile(boot_auprc, upper_q)) if boot_auprc else float("nan")
    )

    metrics[k("brier_ci_lower")] = (
        float(np.percentile(boot_brier, lower_q)) if boot_brier else float("nan")
    )
    metrics[k("brier_ci_upper")] = (
        float(np.percentile(boot_brier, upper_q)) if boot_brier else float("nan")
    )

    metrics[k("logloss_ci_lower")] = (
        float(np.percentile(boot_logloss, lower_q)) if boot_logloss else float("nan")
    )
    metrics[k("logloss_ci_upper")] = (
        float(np.percentile(boot_logloss, upper_q)) if boot_logloss else float("nan")
    )

    return metrics