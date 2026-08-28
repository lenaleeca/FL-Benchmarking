# xgb_posthoc_primary/evaluation.py

from pathlib import Path

import numpy as np

from functions.xgb.prepare_input import prepare_xgb_input
from functions.common.metrics import compute_binary_prob_metrics


def xgb_model_predict(
    *,
    booster,
    df_raw,
    dmat,
    y_true,
    threshold=0.5,
):
    """
    Apply an XGBoost Booster to an already-prepared DMatrix.

    Returns
    -------
    metrics : dict
        Binary probability metrics from compute_binary_prob_metrics().

    pred_df : pandas.DataFrame
        Copy of df_raw with:
            pred_prob
            pred_label
    """
    y_prob = np.asarray(booster.predict(dmat)).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_binary_prob_metrics(y_true, y_prob)

    pred_df = df_raw.copy()
    pred_df["pred_prob"] = y_prob
    pred_df["pred_label"] = y_pred

    return metrics, pred_df

def evaluate_model_on_csv(
    *,
    booster,
    model_path,
    csv_path,
    site_id,
    model_type,
    dataset_name,
    output_dir,
    best_round=None,
    scale=False,
    drop_cols=True,
    verbose=True,
):
    """
    Apply one XGBoost model to one CSV file and save predictions.

    This function is used by both:
        1. regular held-out evaluation
        2. stratified held-out evaluation

    Parameters
    ----------
    booster:
        Loaded xgboost.Booster object.

    model_path:
        Path to the model file. Used as metadata in the output metrics.

    csv_path:
        Path to the held-out CSV file.

    site_id:
        Site identifier, for example 1, 2, 3, 4.

    model_type:
        Model label, for example:
            - "local"
            - "central"
            - "fl_site"

    dataset_name:
        Dataset label, for example:
            - "regular"
            - "male"
            - "female"
            - "age_18_44"
            - "age_45_64"
            - "age_65_79"
            - "age_ge_80"

    output_dir:
        Directory where prediction CSVs will be saved.

    best_round:
        Common selected FL best round.
        Use None for central and independently trained local models.

    scale:
        Whether to scale predictors inside prepare_xgb_input().
        Usually False because the held-out CSVs are already scaled.

    drop_cols:
        Columns to drop before making the XGBoost DMatrix.
        True means drop ["row_id"] if present.

    verbose:
        Print progress messages.

    Returns
    -------
    dict
        Metrics dictionary with metadata columns added.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    df_raw, dmat, y_true = prepare_xgb_input(
        csv_path=csv_path,
        model=booster,
        scale=scale,
        drop_cols=drop_cols,
    )

    metrics, pred_df = xgb_model_predict(
        booster=booster,
        df_raw=df_raw,
        dmat=dmat,
        y_true=y_true,
    )

    metrics["site_id"] = site_id
    metrics["model_type"] = model_type
    metrics["dataset"] = dataset_name
    metrics["best_round"] = best_round
    metrics["model_path"] = str(model_path)
    metrics["heldout_csv"] = str(csv_path)

    pred_out_path = build_prediction_output_path(
        output_dir=output_dir,
        site_id=site_id,
        model_type=model_type,
        dataset_name=dataset_name,
        best_round=best_round,
    )

    pred_df.to_csv(pred_out_path, index=False)

    metrics["prediction_csv"] = str(pred_out_path)

    if verbose:
        print(f"  model: {model_path}")
        print(f"  data : {csv_path}")
        print(f"  pred : {pred_out_path}")

    return metrics


def build_prediction_output_path(
    *,
    output_dir,
    site_id,
    model_type,
    dataset_name,
    best_round=None,
):
    """
    Create a standardized prediction CSV path.

    Examples
    --------
    site_1_local_predictions.csv
    site_1_central_predictions.csv
    site_1_fl_site_round_57_predictions.csv
    site_1_local_male_predictions.csv
    site_1_fl_site_age_65_79_round_57_predictions.csv
    """
    output_dir = Path(output_dir)

    pred_name = f"site_{site_id}_{model_type}_{dataset_name}"

    if best_round is not None:
        pred_name += f"_round_{best_round}"

    pred_name += "_predictions.csv"

    return output_dir / pred_name