from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def _is_test_partition_file(path):
    name = path.name.lower()
    return (
        path.is_file()
        and name == "test_partition_scaled.csv"
    )


def _is_stratified_partition_file(path, stratum):
    name = path.name.lower()
    expected = f"test_partition_scaled_{stratum}.csv".lower()
    return (
        path.is_file()
        and name == expected
    )


def find_test_partition_file_in_site_result_dir(site_dir):
    """
    Find the single primary held-out test partition file in one explicit
    site result folder.

    """
    site_dir = Path(site_dir)

    if not site_dir.is_dir():
        raise ValueError(f"Not a folder: {site_dir}")

    partition_files = sorted(
        p for p in site_dir.rglob("test_partition_scaled.csv")
        if _is_test_partition_file(p)
    )

    if not partition_files:
        raise FileNotFoundError(
            f"No test_partition_scaled.csv found under: {site_dir}"
        )

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


def find_stratified_partition_file_in_site_result_dir(site_dir, stratum):
    """
    Find the single stratified held-out partition file for one stratum.
    """
    site_dir = Path(site_dir)

    if not site_dir.is_dir():
        raise ValueError(f"Not a folder: {site_dir}")

    pattern = f"test_partition_scaled_{stratum}.csv"

    partition_files = sorted(
        p for p in site_dir.rglob(pattern)
        if _is_stratified_partition_file(p, stratum)
    )

    if not partition_files:
        raise FileNotFoundError(
            f"No {pattern} found under: {site_dir}"
        )

    if len(partition_files) > 1:
        msg = [
            f"Multiple {pattern} files found under explicit site result folder: {site_dir}",
            "The path is ambiguous.",
            "Please provide the exact run folder containing one matching stratified partition file, or remove the ambiguity.",
            "Found:",
        ]
        msg.extend(f" - {p}" for p in partition_files)
        raise FileNotFoundError("\n".join(msg))

    return partition_files[0]


def get_partitions_dir(site_dir):
    """
    Return the folder containing test_partition_scaled.csv.

    This is useful for writing stratified partition files next to the original
    test partition.
    """
    return find_test_partition_file_in_site_result_dir(site_dir).parent


def get_test_partition_csv(site_dir):
    return find_test_partition_file_in_site_result_dir(site_dir)


def get_test_partition_stratified_csv(site_dir, stratum):
    return find_stratified_partition_file_in_site_result_dir(site_dir, stratum)


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


def prepare_xgb_input(csv_path, model, scale=False, drop_cols=True):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df_raw = load_input_data(
        csv_path,
        drop_cols=drop_cols,
    )

    if df_raw.shape[1] < 2:
        raise ValueError(
            "Input CSV must have at least 2 columns after dropping columns: "
            "first column = outcome, remaining columns = predictors."
        )

    y_true = pd.to_numeric(df_raw.iloc[:, 0], errors="raise").astype(int).to_numpy()
    X = df_raw.iloc[:, 1:].to_numpy(dtype="float32")

    if X.shape[1] != model.num_features():
        raise ValueError(
            f"Feature count mismatch: input has {X.shape[1]} predictors, "
            f"but model expects {model.num_features()}."
        )

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    dmat = xgb.DMatrix(X, label=y_true)
    return df_raw, dmat, y_true