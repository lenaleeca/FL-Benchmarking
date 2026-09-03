from pathlib import Path

import torch

from functions.mlp.prepare_input import prepare_mlp_input
from functions.common.metrics import compute_binary_prob_metrics


@torch.no_grad()

def build_prediction_output_path(*, output_dir, site_id, model_type, dataset_name, best_round=None):
    output_dir = Path(output_dir)
    pred_name = f"site_{site_id}_{model_type}_{dataset_name}"
    if best_round is not None:
        pred_name += f"_round_{best_round}"
    pred_name += "_predictions.csv"
    return output_dir / pred_name


def mlp_model_predict(*, model, df_raw, X, y_true, device="cpu", threshold=0.5):
    """
    Apply a PyTorch MLP model to a processed numpy feature matrix.
    """
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(Xt)

    y_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_binary_prob_metrics(y_true, y_prob)

    pred_df = df_raw.copy()
    pred_df["pred_prob"] = y_prob
    pred_df["pred_label"] = y_pred

    return metrics, pred_df


def evaluate_model_on_csv(
    *,
    model_info,
    csv_path,
    site_id,
    model_type,
    dataset_name,
    output_dir,
    best_round=None,
    label_col=None,
    scale=False,
    drop_cols=True,
    threshold=0.5,
    verbose=True,
):
    """
    Apply one loaded MLP model to one CSV file and save predictions.

    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    df_raw, X, y_true = prepare_mlp_input(
        csv_path=csv_path,
        expected_n_features=model_info.get("in_dim"),
        label_col=label_col,
        scale=scale,
        drop_cols=drop_cols,
    )

    metrics, pred_df = mlp_model_predict(
        model=model_info["model"],
        df_raw=df_raw,
        X=X,
        y_true=y_true,
        device=model_info.get("device", "cpu"),
        threshold=threshold,
    )

    metrics["site_id"] = site_id
    metrics["model_type"] = model_type
    metrics["dataset"] = dataset_name
    metrics["best_round"] = best_round
    metrics["model_path"] = str(model_info["model_path"])
    metrics["heldout_csv"] = str(csv_path)
    metrics["model_family"] = "mlp"
    metrics["in_dim"] = model_info.get("in_dim")
    metrics["hidden"] = model_info.get("hidden")
    metrics["out_dim"] = model_info.get("out_dim")

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
        print(f"  model: {model_info['model_path']}")
        print(f"  data : {csv_path}")
        print(f"  pred : {pred_out_path}")

    return metrics
