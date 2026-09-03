"""
Select the common best FL round using the 
highest mean validation area under the curve (AUC) across sites, reflecting one deployable 
global FL model.

Input:
    One explicit FedEMRai FL result directory per site. Each result folder must contain
    a ``*_metrics.csv`` file with columns:

        round, dataset, metric, value

Output:
    The selected best round, a round-level validation AUC summary, and the
    metric-file entries used in the calculation.
"""

from pathlib import Path
import pandas as pd


EXCLUDED_METRIC_FILE = (
    "shareable_metrics",
    "timing_metrics",
    "port_transfer",
    "transfer_metrics",
    "report",
)


def find_metric_file(site_dir, pattern="*_metrics.csv"):
    
    site_dir = Path(site_dir)

    if not site_dir.is_dir():
        raise ValueError(f"Not a folder: {site_dir}")

    metric_files = sorted(
        path
        for path in site_dir.rglob(pattern)
        if path.is_file()
        and not any(
            excluded in path.name.lower()
            for excluded in EXCLUDED_METRIC_FILE
        )
    )

    if not metric_files:
        raise ValueError(
            f"No primary metrics file matching '{pattern}' found under: {site_dir}"
        )

    if len(metric_files) > 1:
        msg = [
            f"Multiple primary metrics files found under result folder: {site_dir}",
            "FedEMRai result folder should contain only one *_metrics.csv file.",
            "Please provide the exact FedEMRai result folder containing one *_metrics.csv file, or remove the ambiguity.",
            "Found:",
        ]
        msg.extend(f" - {p}" for p in metric_files)
        raise ValueError("\n".join(msg))

    return metric_files[0]


def get_best_round(metric_files, verbose=True):
    
    rows = []

    for site_id, metric_file in metric_files.items():
        df = pd.read_csv(metric_file)

        required = {"round", "dataset", "metric", "value"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{metric_file} is missing required columns: {sorted(missing)}"
            )

        site_auc = df[
            df["dataset"].astype(str).str.lower().eq("val")
            & df["metric"].astype(str).str.lower().eq("auc")
        ][["round", "value"]].copy()

        site_auc["round"] = pd.to_numeric(
            site_auc["round"],
            errors="coerce",
        )
        site_auc["value"] = pd.to_numeric(
            site_auc["value"],
            errors="coerce",
        )

        site_auc = site_auc.dropna()

        if site_auc.empty:
            raise ValueError(
                f"No valid validation AUC rows found in: {metric_file}"
            )

        site_auc["round"] = site_auc["round"].astype(int)
        site_auc["site_id"] = site_id

        rows.append(site_auc)

    combined = pd.concat(rows, ignore_index=True)

    summary = (
        combined.groupby("round", as_index=False)
        .agg(
            mean_val_auc=("value", "mean"),
            n_sites=("site_id", "nunique"),
        )
        .sort_values("round")
    )

    n_sites = len(metric_files)
    summary = summary[summary["n_sites"] == n_sites].copy()

    if summary.empty:
        raise ValueError(
            "No FL rounds were present in all site metric files."
        )

    best_row = summary.loc[summary["mean_val_auc"].idxmax()]
    best_round = int(best_row["round"])

    if verbose:
        print("\nMean validation AUC by round:")
        print(summary.to_string(index=False))

        print(
            f"\nBest round: {best_round} "
            f"(mean validation AUC = {best_row['mean_val_auc']:.6f})"
        )
        
        print(f"Number of sites contributing = {int(best_row['n_sites'])}")

    return best_round, summary


def get_common_best_round(
    fl_site_results_dirs,
    pattern="*_metrics.csv",
    verbose=True,
):
    """
    Find site metric files and select the common best FL round.

    Returns
    -------
    best_round
        FL round with the highest mean validation AUC across sites.
    summary
        Round-level mean validation AUC summary.
    metric_files
        Dictionary mapping site IDs to metric CSV paths.
    """
    if not fl_site_results_dirs:
        raise ValueError("fl_site_results_dirs is required.")

    metric_files = {
        site_id: find_metric_file(site_dir, pattern)
        for site_id, site_dir in fl_site_results_dirs.items()
    }

    if verbose:
        for site_id, metric_file in metric_files.items():
            print(f"Metrics for site {site_id}: {metric_file}")

    best_round, summary = get_best_round(
        metric_files,
        verbose=verbose,
    )

    return best_round, summary, metric_files
