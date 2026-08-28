from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _is_test_partition_file(path):
    return path.is_file() and path.name.lower() == "test_partition_scaled.csv"


def find_test_partition_file_in_site_result_dir(site_dir):
    """
    Find one primary held-out test partition file under an explicit site result folder.

    Supports both:
        run_folder/partitions/test_partition_scaled.csv
        run_folder/results/partitions/test_partition_scaled.csv
    """
    site_dir = Path(site_dir)
    if not site_dir.is_dir():
        raise ValueError(f"Not a folder: {site_dir}")

    partition_files = sorted(
        p for p in site_dir.rglob("test_partition_scaled.csv")
        if _is_test_partition_file(p)
    )

    if not partition_files:
        raise FileNotFoundError(f"No test_partition_scaled.csv found under: {site_dir}")

    if len(partition_files) > 1:
        msg = [
            f"Multiple test_partition_scaled.csv files found under explicit site result folder: {site_dir}",
            "The path is ambiguous.",
            "Please provide the exact run folder containing one test_partition_scaled.csv file, or remove the ambiguity.",
            "Found:",
        ]
        msg.extend(f" - {p}" for p in partition_files)
        raise FileNotFoundError("\n".join(msg))

    return partition_files[0]


def get_test_partition_csv(site_dir):
    return find_test_partition_file_in_site_result_dir(site_dir)


def load_input_data(csv_path, drop_cols=None):
    df = pd.read_csv(csv_path)

    if drop_cols is True:
        drop_cols = ["row_id"]
    elif drop_cols is False or drop_cols is None:
        drop_cols = []

    drop_cols = [c for c in drop_cols if c in df.columns]

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def prepare_mlp_input(csv_path, *, expected_n_features=None, label_col=None, scale=False, drop_cols=True):
    """
    Prepare held-out CSV for PyTorch MLP inference.

    If label_col is provided, y=df[label_col] and X=remaining columns.
    If label_col is None, the first column after drop_cols is the outcome.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df_raw = load_input_data(csv_path, drop_cols=drop_cols)

    if df_raw.shape[1] < 2:
        raise ValueError(
            "Input CSV must have at least 2 columns after dropping columns: "
            "one outcome column and at least one predictor column."
        )

    if label_col is not None:
        if label_col not in df_raw.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. Available columns: {list(df_raw.columns)}"
            )
        y_true = pd.to_numeric(df_raw[label_col], errors="raise").astype(int).to_numpy()
        X = df_raw.drop(columns=[label_col]).to_numpy(dtype="float32")
    else:
        y_true = pd.to_numeric(df_raw.iloc[:, 0], errors="raise").astype(int).to_numpy()
        X = df_raw.iloc[:, 1:].to_numpy(dtype="float32")

    if expected_n_features is not None and X.shape[1] != int(expected_n_features):
        raise ValueError(
            f"Feature count mismatch: input has {X.shape[1]} predictors, "
            f"but model expects {expected_n_features}."
        )

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    return df_raw, X, y_true
