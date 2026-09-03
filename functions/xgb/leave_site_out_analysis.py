from pathlib import Path
import pandas as pd

from functions.common.get_common_best_round import get_common_best_round
from functions.xgb.load_models import (
    load_central_xgb,
    load_fl_xgb,
)
from functions.xgb.prepare_input import get_test_partition_csv
from functions.xgb.evaluation import evaluate_model_on_csv
from functions.common.path_config import (
    normalize_site_path_map,
    site_dir_from_root_or_map,
    site_sort_key,
)


def _normalize_leave_site_out_runs(leave_site_out_runs):
    """
    Normalize excluded_site_id -> leave-site-out config.

    Supports excluded site IDs such as:
        1, "1", "A", "B"

    Expected direct-path format:
        leave_site_out_runs = {
            "A": {
                "server_results_dir": "...",
                "central_results_dir": "...",
                "best_round": 51,  # optional if fl_site_results_dirs provided
                "fl_site_results_dirs": {
                    "B": "...included site B run folder...",
                    "C": "...included site C run folder...",
                },
            },
        }
    """
    if leave_site_out_runs is None:
        raise ValueError("leave_site_out_runs is required for leave-site-out analysis.")

    normalized = {}

    for excluded_site_id, cfg in leave_site_out_runs.items():
        site_text = str(excluded_site_id)
        key = int(site_text) if site_text.isdigit() else excluded_site_id

        cfg = dict(cfg)

        if "server_results_dir" not in cfg:
            raise KeyError(
                f"leave_site_out_runs[{excluded_site_id}] is missing "
                "'server_results_dir'."
            )

        if "central_results_dir" not in cfg:
            raise KeyError(
                f"leave_site_out_runs[{excluded_site_id}] is missing "
                "'central_results_dir'."
            )

        if cfg.get("best_round") is None and cfg.get("fl_site_results_dirs") is None:
            raise KeyError(
                f"leave_site_out_runs[{excluded_site_id}] must provide either "
                "'best_round' or 'fl_site_results_dirs' so the leave-site-out "
                "best round can be determined."
            )

        if cfg.get("fl_site_results_dirs") is not None:
            cfg["fl_site_results_dirs"] = normalize_site_path_map(
                cfg["fl_site_results_dirs"],
                name=f"leave_site_out_runs[{excluded_site_id}]['fl_site_results_dirs']",
            )

        cfg["server_results_dir"] = Path(cfg["server_results_dir"])
        cfg["central_results_dir"] = Path(cfg["central_results_dir"])
        cfg["run_dir"] = Path(cfg.get("run_dir") or cfg["server_results_dir"].parent)

        normalized[key] = cfg

    return normalized


def get_excluded_site_heldout_csv(
    *,
    primary_results_root_dir=None,
    primary_fl_site_results_dirs=None,
    excluded_site_id,
):
    excluded_site_dir = site_dir_from_root_or_map(
        primary_results_root_dir,
        excluded_site_id,
        path_map=primary_fl_site_results_dirs,
    )

    return get_test_partition_csv(excluded_site_dir)


def get_primary_auc(
    *,
    primary_results_df,
    site_id,
    model_type,
    auc_col="auc",
):
    if primary_results_df is None or primary_results_df.empty:
        return pd.NA

    required_cols = {"site_id", "model_type", auc_col}
    missing_cols = required_cols - set(primary_results_df.columns)

    if missing_cols:
        raise ValueError(
            f"primary_results_df is missing required columns: {missing_cols}"
        )

    sub = primary_results_df[
        (primary_results_df["site_id"].astype(str) == str(site_id))
        & (primary_results_df["model_type"].astype(str) == str(model_type))
    ]

    if sub.empty:
        return pd.NA

    return sub.iloc[0][auc_col]


def get_leave_site_out_auc(
    *,
    leave_site_out_results_df,
    excluded_site_id,
    model_type,
    auc_col="auc",
):
    if leave_site_out_results_df is None or leave_site_out_results_df.empty:
        return pd.NA

    required_cols = {"excluded_site_id", "model_type", auc_col}
    missing_cols = required_cols - set(leave_site_out_results_df.columns)

    if missing_cols:
        raise ValueError(
            "leave_site_out_results_df is missing required columns: "
            f"{missing_cols}"
        )

    sub = leave_site_out_results_df[
        (leave_site_out_results_df["excluded_site_id"].astype(str) == str(excluded_site_id))
        & (leave_site_out_results_df["model_type"].astype(str) == str(model_type))
    ]

    if sub.empty:
        return pd.NA

    return sub.iloc[0][auc_col]


def make_leave_site_out_summary_table(
    *,
    primary_results_df,
    leave_site_out_results_df,
    auc_col="auc",
):
    if leave_site_out_results_df is None or leave_site_out_results_df.empty:
        return pd.DataFrame()

    rows = []

    excluded_site_ids = sorted(
        leave_site_out_results_df["excluded_site_id"].dropna().unique(),
        key=site_sort_key,
    )

    for excluded_site_id in excluded_site_ids:
        rows.append(
            {
                "excluded_site_id": excluded_site_id,
                "excluded_site": f"Site {excluded_site_id} excluded",
                "local_excluded_site": get_primary_auc(
                    primary_results_df=primary_results_df,
                    site_id=excluded_site_id,
                    model_type="local",
                    auc_col=auc_col,
                ),
                "central_within_sample": get_primary_auc(
                    primary_results_df=primary_results_df,
                    site_id=excluded_site_id,
                    model_type="central",
                    auc_col=auc_col,
                ),
                "central_leave_site_out": get_leave_site_out_auc(
                    leave_site_out_results_df=leave_site_out_results_df,
                    excluded_site_id=excluded_site_id,
                    model_type="central_leave_site_out",
                    auc_col=auc_col,
                ),
                "fl_within_sample": get_primary_auc(
                    primary_results_df=primary_results_df,
                    site_id=excluded_site_id,
                    model_type="fl_site",
                    auc_col=auc_col,
                ),
                "fl_leave_site_out": get_leave_site_out_auc(
                    leave_site_out_results_df=leave_site_out_results_df,
                    excluded_site_id=excluded_site_id,
                    model_type="fl_site_leave_site_out",
                    auc_col=auc_col,
                ),
            }
        )

    return pd.DataFrame(rows)


def _get_lso_best_round(cfg, *, excluded_site_id, verbose=True):
    best_round = cfg.get("best_round")

    if best_round is not None:
        return int(best_round), pd.DataFrame(), []

    best_round, round_summary_df, metrics_files = get_common_best_round(
        fl_site_results_dirs=cfg["fl_site_results_dirs"],
        verbose=verbose,
    )

    if verbose:
        print(
            f"Leave-site-out run for excluded site {excluded_site_id}: "
            f"derived best_round={best_round}"
        )

    return best_round, round_summary_df, metrics_files


def run_leave_site_out_analysis(
    *,
    primary_results_root_dir=None,
    primary_fl_site_results_dirs=None,
    leave_site_out_runs,
    primary_results_df,
    output_dir,
    verbose=True,
):

    primary_fl_site_results_dirs = normalize_site_path_map(
        primary_fl_site_results_dirs,
        name="primary_fl_site_results_dirs",
    )

    leave_site_out_runs = _normalize_leave_site_out_runs(leave_site_out_runs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_site_ids = sorted(
        leave_site_out_runs.keys(),
        key=site_sort_key,
    )

    leave_site_out_rows = []

    if verbose:
        print("\nRunning leave-site-out analysis")
        print(f"Detected excluded sites: {excluded_site_ids}")

    for excluded_site_id in excluded_site_ids:
        cfg = leave_site_out_runs[excluded_site_id]

        leave_run_dir = cfg["run_dir"]
        central_results_dir = cfg["central_results_dir"]
        server_results_dir = cfg["server_results_dir"]

        if verbose:
            print("\n" + "-" * 60)
            print(f"Leave-site-out run: Site {excluded_site_id} excluded")
            print(f"Run folder: {leave_run_dir}")
            print(f"Central dir: {central_results_dir}")
            print(f"Server results dir: {server_results_dir}")

        heldout_csv = get_excluded_site_heldout_csv(
            primary_results_root_dir=primary_results_root_dir,
            primary_fl_site_results_dirs=primary_fl_site_results_dirs,
            excluded_site_id=excluded_site_id,
        )

        lso_best_round, _, _ = _get_lso_best_round(
            cfg,
            excluded_site_id=excluded_site_id,
            verbose=verbose,
        )

        if verbose:
            print(f"Excluded site held-out CSV: {heldout_csv}")
            print(f"Leave-site-out selected best round: {lso_best_round}")

        lso_central_model = load_central_xgb(central_results_dir)

        lso_fl_model = load_fl_xgb(
            server_results_dir=server_results_dir,
            best_round=lso_best_round,
        )

        central_metrics = evaluate_model_on_csv(
            booster=lso_central_model["booster"],
            model_path=lso_central_model["model_path"],
            csv_path=heldout_csv,
            site_id=excluded_site_id,
            model_type="central_leave_site_out",
            dataset_name="leave_site_out",
            output_dir=output_dir,
            best_round=None,
            verbose=verbose,
        )

        central_metrics["excluded_site_id"] = excluded_site_id
        central_metrics["leave_site_out_run_dir"] = str(leave_run_dir)
        central_metrics["leave_site_out_best_round"] = lso_best_round
        central_metrics["model_round"] = lso_central_model.get("model_round")
        central_metrics["model_kind"] = lso_central_model.get("model_kind")
        central_metrics["loaded_from"] = lso_central_model.get(
            "loaded_from",
            "central_results_dir",
        )

        leave_site_out_rows.append(central_metrics)

        fl_metrics = evaluate_model_on_csv(
            booster=lso_fl_model["booster"],
            model_path=lso_fl_model["model_path"],
            csv_path=heldout_csv,
            site_id=excluded_site_id,
            model_type="fl_site_leave_site_out",
            dataset_name="leave_site_out",
            output_dir=output_dir,
            best_round=lso_best_round,
            verbose=verbose,
        )

        fl_metrics["excluded_site_id"] = excluded_site_id
        fl_metrics["leave_site_out_run_dir"] = str(leave_run_dir)
        fl_metrics["leave_site_out_best_round"] = lso_best_round
        fl_metrics["model_round"] = lso_fl_model.get("model_round")
        fl_metrics["model_kind"] = lso_fl_model.get("model_kind")
        fl_metrics["loaded_from"] = lso_fl_model.get(
            "loaded_from",
            "server_results_dir",
        )

        leave_site_out_rows.append(fl_metrics)

    leave_site_out_results_df = pd.DataFrame(leave_site_out_rows)

    leave_site_out_metrics_out = (
        output_dir / "heldout_leave_site_out_all_model_metrics.csv"
    )
    leave_site_out_results_df.to_csv(leave_site_out_metrics_out, index=False)

    leave_site_out_summary_df = make_leave_site_out_summary_table(
        primary_results_df=primary_results_df,
        leave_site_out_results_df=leave_site_out_results_df,
        auc_col="auc",
    )

    leave_site_out_summary_out = output_dir / "heldout_leave_site_out_summary.csv"
    leave_site_out_summary_df.to_csv(leave_site_out_summary_out, index=False)

    if verbose:
        print("\nLeave-site-out analysis complete.")
        print(f"Leave-site-out metrics saved to:\n{leave_site_out_metrics_out}")
        print(f"Leave-site-out summary saved to:\n{leave_site_out_summary_out}")

    return {
        "leave_site_out_results_df": leave_site_out_results_df,
        "leave_site_out_summary_df": leave_site_out_summary_df,
        "leave_site_out_metrics_out": leave_site_out_metrics_out,
        "leave_site_out_summary_out": leave_site_out_summary_out,
    }