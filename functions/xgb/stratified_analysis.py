# functions/xgb/stratified_analysis.py

from pathlib import Path
import pandas as pd

from functions.xgb.prepare_input import (
    get_test_partition_csv,
    get_test_partition_stratified_csv,
    get_partitions_dir,
)
from functions.xgb.evaluation import evaluate_model_on_csv
from functions.common.path_config import site_dir_from_root_or_map


STRATA = [
    "male",
    "female",
    "age_18_44",
    "age_45_64",
    "age_65_79",
    "age_ge_80",
]


def get_test_partition_raw_csv(site_dir):
    """
    Find the raw test partition for one site.

    This is used only to define strata when raw_csv_by_site is not provided.
    """
    site_dir = Path(site_dir)
    partitions_dir = get_partitions_dir(site_dir)

    candidates = [
        partitions_dir / "test_partition_raw.csv",
        partitions_dir / "test_partition_unscaled.csv",
        partitions_dir / "test_partition_original.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        p for p in partitions_dir.glob("*test*raw*.csv")
        if p.is_file()
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileExistsError(
            "Multiple raw test partition CSVs found. Please make the raw test "
            "partition name unambiguous or pass raw_csv_by_site explicitly:\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        f"No raw test partition CSV found in: {partitions_dir}. "
        "Expected test_partition_raw.csv, test_partition_unscaled.csv, "
        "test_partition_original.csv, or a file matching *test*raw*.csv."
    )


def prepare_stratified_heldout_csvs(
    *,
    scaled_heldout_csv,
    strata_source_csv,
    site_dir,
    age_col,
    sex_col,
    row_id_col="row_id",
):
    """
    Create stratified scaled held-out CSVs.

    scaled_heldout_csv:
        Model-ready/scaled test partition used for prediction.

    strata_source_csv:
        Either:
            - external raw site CSV, if raw_csv_by_site is provided, or
            - raw test partition from partitions folder, if raw_csv_by_site is None

        This file is used only to identify row_id values for each age/sex stratum.

    Saved stratified CSVs keep only the columns from scaled_heldout_csv.
    """
    scaled_heldout_csv = Path(scaled_heldout_csv)
    strata_source_csv = Path(strata_source_csv)
    site_dir = Path(site_dir)

    if not scaled_heldout_csv.exists():
        raise FileNotFoundError(f"Scaled held-out CSV not found: {scaled_heldout_csv}")

    if not strata_source_csv.exists():
        raise FileNotFoundError(f"Strata source CSV not found: {strata_source_csv}")

    out_dir = get_partitions_dir(site_dir) / "stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_scaled = pd.read_csv(scaled_heldout_csv)
    df_strata = pd.read_csv(strata_source_csv)

    if row_id_col not in df_scaled.columns:
        raise ValueError(f"{row_id_col} not found in scaled held-out data.")

    needed_strata_cols = [row_id_col, age_col, sex_col]
    missing_strata_cols = [
        c for c in needed_strata_cols
        if c not in df_strata.columns
    ]

    if missing_strata_cols:
        raise ValueError(
            f"Missing columns in strata source data: {missing_strata_cols}"
        )

    if df_scaled[row_id_col].duplicated().any():
        raise ValueError(
            f"Duplicate {row_id_col} values found in scaled held-out data."
        )

    if df_strata[row_id_col].duplicated().any():
        raise ValueError(
            f"Duplicate {row_id_col} values found in strata source data."
        )

    scaled_ids = set(df_scaled[row_id_col])
    strata_source_ids = set(df_strata[row_id_col])

    missing_from_strata_source = scaled_ids - strata_source_ids

    if missing_from_strata_source:
        raise ValueError(
            f"{len(missing_from_strata_source)} scaled held-out rows are missing "
            "from the strata source data by row_id."
        )

    # Keep only rows corresponding to the scaled held-out test partition.
    df_strata = df_strata[df_strata[row_id_col].isin(scaled_ids)].copy()

    strata_row_ids = {
        "male": set(df_strata.loc[df_strata[sex_col] == 0, row_id_col]),
        "female": set(df_strata.loc[df_strata[sex_col] == 1, row_id_col]),
        "age_18_44": set(
            df_strata.loc[
                (df_strata[age_col] >= 18) & (df_strata[age_col] <= 44),
                row_id_col,
            ]
        ),
        "age_45_64": set(
            df_strata.loc[
                (df_strata[age_col] >= 45) & (df_strata[age_col] <= 64),
                row_id_col,
            ]
        ),
        "age_65_79": set(
            df_strata.loc[
                (df_strata[age_col] >= 65) & (df_strata[age_col] <= 79),
                row_id_col,
            ]
        ),
        "age_ge_80": set(
            df_strata.loc[df_strata[age_col] >= 80, row_id_col]
        ),
    }

    out_paths = {}

    for stratum_name, row_ids in strata_row_ids.items():
        sub_scaled = df_scaled[df_scaled[row_id_col].isin(row_ids)].copy()

        out_path = out_dir / f"test_partition_scaled_{stratum_name}.csv"
        sub_scaled.to_csv(out_path, index=False)
        out_paths[stratum_name] = out_path

    return out_paths


def get_stratified_site_datasets(site_dir):
    """
    Return paths to all stratified scaled held-out CSVs for one site.
    """
    datasets = {}

    for stratum in STRATA:
        datasets[stratum] = get_test_partition_stratified_csv(
            site_dir,
            stratum,
        )

    return datasets


def csv_has_enough_rows_for_evaluation(csv_path, min_rows=1):
    """
    Basic guard against evaluating an empty stratum. This checks row count only. 
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return False

    return len(df) >= min_rows


def make_sex_auc_difference_table(results_df, auc_col="auc"):
    """
    Calculate absolute male-female AUC difference within each site and model.

    Delta AUC = abs(AUC_male - AUC_female)
    """
    rows = []

    if results_df is None or results_df.empty:
        return pd.DataFrame()

    required_cols = {"site_id", "model_type", "dataset", auc_col}
    missing_cols = required_cols - set(results_df.columns)

    if missing_cols:
        raise ValueError(
            f"Cannot calculate sex AUC difference. Missing columns: {missing_cols}"
        )

    sex_df = results_df[
        results_df["dataset"].isin(["male", "female"])
    ].copy()

    for (site_id, model_type), g in sex_df.groupby(["site_id", "model_type"]):
        male_auc = g.loc[g["dataset"] == "male", auc_col]
        female_auc = g.loc[g["dataset"] == "female", auc_col]

        if male_auc.empty or female_auc.empty:
            continue

        male_auc = male_auc.iloc[0]
        female_auc = female_auc.iloc[0]

        if pd.isna(male_auc) or pd.isna(female_auc):
            delta_auc_abs = pd.NA
        else:
            male_auc = float(male_auc)
            female_auc = float(female_auc)
            delta_auc_abs = abs(male_auc - female_auc)

        rows.append(
            {
                "site_id": site_id,
                "model_type": model_type,
                "comparison": "male_female_absolute_difference",
                "male_auc": male_auc,
                "female_auc": female_auc,
                "delta_auc_abs": delta_auc_abs,
            }
        )

    return pd.DataFrame(rows)


def make_age_auc_summary_table(results_df, auc_col="auc"):
    """
    Create age-stratified AUC summary table.

    Keeps AUC and its 95% CI columns if available.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    age_datasets = [
        "age_18_44",
        "age_45_64",
        "age_65_79",
        "age_ge_80",
    ]

    age_df = results_df[
        results_df["dataset"].isin(age_datasets)
    ].copy()

    keep_cols = [
        "site_id",
        "model_type",
        "dataset",
        auc_col,
        f"{auc_col}_ci_lower",
        f"{auc_col}_ci_upper",
    ]

    keep_cols = [c for c in keep_cols if c in age_df.columns]

    if not keep_cols:
        return pd.DataFrame()

    return age_df[keep_cols].sort_values(
        ["site_id", "model_type", "dataset"]
    )


def build_stratified_models_for_site(
    *,
    site_id,
    best_round,
    central_booster,
    central_model_path,
    local_models,
    fl_booster,
    fl_model_path,
):
    """
    Build model dictionary for one site.

    """
    local_key = f"site_{site_id}"

    if local_key not in local_models:
        raise KeyError(f"{local_key} not found in local_models.")

    local_info = local_models[local_key]

    models = {
        "local": {
            "booster": local_info["booster"],
            "model_path": local_info["model_path"],
            "best_round": local_info.get("best_round"),
            "model_round": local_info.get("model_round"),
            "model_kind": local_info.get("model_kind"),
            "loaded_from": local_info.get("loaded_from", pd.NA),
        },
        "central": {
            "booster": central_booster,
            "model_path": central_model_path,
            "best_round": None,
            "model_round": None,
            "model_kind": "LOCAL",
            "loaded_from": "central_results_dir",
        },
        "fl_site": {
            "booster": fl_booster,
            "model_path": fl_model_path,
            "best_round": best_round,
            "model_round": best_round,
            "model_kind": "GLOBAL",
            "loaded_from": "server_results_dir",
        },
    }

    return models


def run_stratified_analysis(
    *,
    results_root_dir=None,
    fl_site_results_dirs=None,
    site_ids,
    best_round,
    central_booster,
    central_model_path,
    local_models,
    fl_booster,
    fl_model_path,
    output_dir,
    age_col,
    sex_col,
    raw_csv_by_site=None,
    verbose=True,
):
    """
    Run full stratified held-out analysis.

    If raw_csv_by_site is provided:
        use the provided external raw CSV for that site as the strata source.

    If raw_csv_by_site is not provided:
        use the raw test partition from the site's partitions folder as the
        strata source.

    In both cases, row_id membership is created from the strata source, and
    those row_ids are used to subset the scaled test partition. The evaluated
    stratified CSVs therefore remain scaled/model-ready.
    """
    if results_root_dir is not None:
        results_root_dir = Path(results_root_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stratified_metrics_rows = []

    for site_id in site_ids:
        site_dir = site_dir_from_root_or_map(
            results_root_dir,
            site_id,
            path_map=fl_site_results_dirs,
        )

        scaled_heldout_csv = get_test_partition_csv(site_dir)

        if raw_csv_by_site is not None:
            strata_source_csv = raw_csv_by_site.get(str(site_id))

            if strata_source_csv is None:
                strata_source_csv = raw_csv_by_site.get(site_id)

            if strata_source_csv is None:
                raise KeyError(
                    f"No raw CSV path provided for site {site_id}. "
                    "Either provide one raw CSV path per site, or set "
                    "raw_csv_by_site=None to use the raw test partition."
                )
        else:
            strata_source_csv = get_test_partition_raw_csv(site_dir)

        if verbose:
            print(f"\nPreparing stratified held-out CSVs for site {site_id}")
            print(f"  site folder     : {site_dir}")
            print(f"  scaled test CSV : {scaled_heldout_csv}")
            print(f"  strata source   : {strata_source_csv}")

        prepare_stratified_heldout_csvs(
            scaled_heldout_csv=scaled_heldout_csv,
            strata_source_csv=strata_source_csv,
            site_dir=site_dir,
            age_col=age_col,
            sex_col=sex_col,
        )

        if verbose:
            print(f"\nRunning stratified evaluation for site {site_id}")

        datasets = get_stratified_site_datasets(site_dir)

        models = build_stratified_models_for_site(
            site_id=site_id,
            best_round=best_round,
            central_booster=central_booster,
            central_model_path=central_model_path,
            local_models=local_models,
            fl_booster=fl_booster,
            fl_model_path=fl_model_path,
        )

        for dataset_name, csv_path in datasets.items():
            if not csv_has_enough_rows_for_evaluation(csv_path):
                if verbose:
                    print(
                        f"\nSkipping site {site_id}, stratum {dataset_name}: "
                        "no rows available."
                    )
                continue

            for model_type, model_info in models.items():
                if verbose:
                    print(
                        f"\nProcessing stratified dataset {dataset_name}: "
                        f"{model_type} model, site {site_id}"
                    )

                metrics = evaluate_model_on_csv(
                    booster=model_info["booster"],
                    model_path=model_info["model_path"],
                    csv_path=csv_path,
                    site_id=site_id,
                    model_type=model_type,
                    dataset_name=dataset_name,
                    output_dir=output_dir,
                    best_round=model_info["best_round"],
                    verbose=verbose,
                )

                metrics["model_round"] = model_info.get("model_round")
                metrics["model_kind"] = model_info.get("model_kind")
                metrics["loaded_from"] = model_info.get("loaded_from", pd.NA)

                stratified_metrics_rows.append(metrics)

    stratified_results_df = pd.DataFrame(stratified_metrics_rows)

    stratified_metrics_out = output_dir / "heldout_stratified_all_model_metrics.csv"
    stratified_results_df.to_csv(stratified_metrics_out, index=False)

    sex_auc_difference_df = make_sex_auc_difference_table(
        stratified_results_df,
        auc_col="auc",
    )

    age_auc_summary_df = make_age_auc_summary_table(
        stratified_results_df,
        auc_col="auc",
    )

    sex_auc_difference_out = output_dir / "heldout_sex_auc_differences.csv"
    age_auc_summary_out = output_dir / "heldout_age_auc_summary.csv"

    sex_auc_difference_df.to_csv(sex_auc_difference_out, index=False)
    age_auc_summary_df.to_csv(age_auc_summary_out, index=False)

    if verbose:
        print(f"\nStratified metrics saved to:\n{stratified_metrics_out}")
        print(f"\nSex AUC difference table saved to:\n{sex_auc_difference_out}")
        print(f"\nAge AUC summary table saved to:\n{age_auc_summary_out}")

    return {
        "stratified_results_df": stratified_results_df,
        "sex_auc_difference_df": sex_auc_difference_df,
        "age_auc_summary_df": age_auc_summary_df,
        "stratified_metrics_out": stratified_metrics_out,
        "sex_auc_difference_out": sex_auc_difference_out,
        "age_auc_summary_out": age_auc_summary_out,
    }