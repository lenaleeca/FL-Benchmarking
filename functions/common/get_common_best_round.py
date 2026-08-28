"""Common best-round selection helpers.

Best-round rule:
    select the FL round with the highest mean validation AUC across sites.

Expected metric CSV long format:
    round, dataset, metric, value

Direct-path mode:
    The caller provides exact site result folders. Each folder must contain
    exactly one usable *_metrics.csv file.
"""

from pathlib import Path
import pandas as pd


EXCLUDED_METRIC_FILE_SUBSTRINGS = (
    "shareable_metrics",
    "timing_metrics",
    "port_transfer",
    "transfer_metrics",
    "report",
)


def _is_primary_metrics_file(path):
    name = path.name.lower()
    return (
        path.is_file()
        and name.endswith("_metrics.csv")
        and not any(bad in name for bad in EXCLUDED_METRIC_FILE_SUBSTRINGS)
    )


def _metric_entry(site_id, path):
    return {
        "site_id": site_id,
        "path": Path(path),
    }


def get_metric_entry_path(entry):
    """
    Return Path from either:
        {"site_id": 1, "path": "..."}
    or a raw file path.

    This keeps backward compatibility with older code that passed only paths.
    """
    if isinstance(entry, dict):
        return Path(entry["path"])
    return Path(entry)


def get_metric_entry_site_id(entry):
    """
    Return site_id from a metric entry if available.
    """
    if isinstance(entry, dict):
        return entry.get("site_id")
    return None


def find_metric_file_in_site_result_dir(site_dir, pattern="*_metrics.csv"):
    """
    Find the single primary metrics file in one explicit site result folder.

    This does not choose the latest file. If multiple primary metrics files are
    found, the folder is ambiguous and an error is raised.
    """
    site_dir = Path(site_dir)

    if not site_dir.is_dir():
        raise ValueError(f"Not a folder: {site_dir}")

    metric_files = sorted(
        p for p in site_dir.rglob(pattern)
        if _is_primary_metrics_file(p)
    )

    if not metric_files:
        raise ValueError(
            f"No primary metrics file matching '{pattern}' found under: {site_dir}"
        )

    if len(metric_files) > 1:
        msg = [
            f"Multiple primary metrics files found under explicit site result folder: {site_dir}",
            "Direct-path mode does not select the latest file automatically.",
            "Please provide the exact run folder containing one *_metrics.csv file, or remove the ambiguity.",
            "Found:",
        ]
        msg.extend(f" - {p}" for p in metric_files)
        raise ValueError("\n".join(msg))

    return metric_files[0]


def find_metric_entries_from_site_result_dirs(
    fl_site_results_dirs,
    pattern="*_metrics.csv",
    verbose=True,
):
    """
    Gather one metrics entry from each explicitly provided site result folder.

    Returns
    -------
    metric_entries : list[dict]
        Each entry contains:
            site_id : key from fl_site_results_dirs
            path    : Path to the site's *_metrics.csv
    """
    if fl_site_results_dirs is None:
        raise ValueError("fl_site_results_dirs is required.")

    metric_entries = []

    for site_id, site_dir in fl_site_results_dirs.items():
        metric_file = find_metric_file_in_site_result_dir(
            site_dir,
            pattern=pattern,
        )

        metric_entries.append(_metric_entry(site_id, metric_file))

        if verbose:
            print(f"Metrics for site {site_id}: {metric_file}")

    return metric_entries


# Backward-compatible alias used by older code.
def find_metric_files_from_site_result_dirs(
    fl_site_results_dirs,
    pattern="*_metrics.csv",
    verbose=True,
):
    return find_metric_entries_from_site_result_dirs(
        fl_site_results_dirs=fl_site_results_dirs,
        pattern=pattern,
        verbose=verbose,
    )


def get_best_round_from_metric_files(metric_files, verbose=True):
    """
    Select common best round from metric files/entries.

    metric_files can be either:
        - list of paths
        - list of {"site_id": ..., "path": ...} dictionaries
    """
    rows = []

    for entry in metric_files:
        file = get_metric_entry_path(entry)
        site_id = get_metric_entry_site_id(entry)

        if not file.exists():
            print(f"Skipping missing file: {file}")
            continue

        df = pd.read_csv(file)

        required_cols = {"round", "dataset", "metric", "value"}

        if not required_cols.issubset(df.columns):
            print(f"Skipping {file.name}: missing required columns")
            continue

        sub = df[
            df["dataset"].astype(str).str.lower().eq("val")
            & df["metric"].astype(str).str.lower().eq("auc")
        ][["round", "value"]].copy()

        sub["round"] = pd.to_numeric(sub["round"], errors="coerce")
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna()
        sub["round"] = sub["round"].astype(int)

        if sub.empty:
            print(f"Skipping {file.name}: no valid val auc rows")
            continue

        # Use site_id when available so duplicate filenames from different
        # folders do not collapse into one contributor.
        sub["source_file"] = str(file)
        sub["site_id"] = site_id if site_id is not None else file.stem

        rows.append(sub)

    if not rows:
        raise ValueError("No valid validation AUC data found.")

    combined = pd.concat(rows, ignore_index=True)

    summary = (
        combined.groupby("round", as_index=False)
        .agg(
            mean_val_auc=("value", "mean"),
            n_sites=("site_id", "nunique"),
        )
        .sort_values("round")
    )

    summary["round"] = summary["round"].astype(int)

    valid_site_count = combined["site_id"].nunique()
    summary = summary[summary["n_sites"] == valid_site_count]

    if summary.empty:
        raise ValueError("No rounds were present in all valid input files.")

    best_row = summary.loc[summary["mean_val_auc"].idxmax()]
    best_round = int(best_row["round"])

    if verbose:
        print("\nMean validation AUC by round:")
        print(summary.to_string(index=False))

        print("\nBest round:")
        print(f"Round = {best_round}")
        print(f"Mean val AUC = {best_row['mean_val_auc']:.6f}")
        print(f"Number of sites contributing = {int(best_row['n_sites'])}")

    return best_round, summary, metric_files


def get_common_best_round(
    fl_site_results_dirs,
    pattern="*_metrics.csv",
    verbose=True,
):
    metric_entries = find_metric_entries_from_site_result_dirs(
        fl_site_results_dirs=fl_site_results_dirs,
        pattern=pattern,
        verbose=verbose,
    )

    return get_best_round_from_metric_files(
        metric_entries,
        verbose=verbose,
    )
