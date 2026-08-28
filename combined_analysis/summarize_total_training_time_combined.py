"""
Summarise total model-training time across clinical projects.

This script reads a run-tracking CSV and calculates total training time by
clinical project, base model, training strategy, and federated-learning
algorithm.

Only runs marked as valid are included. Leave-site-out / site-withheld runs
are excluded from the main training-time summary because they are considered
secondary or sensitivity analyses.

Input format
------------
The input CSV must contain the following columns:

    Name
        Descriptive run name used to identify the model and training strategy.
        Expected naming conventions include terms such as:
            XGB, MLP, LR
            Local, Central
            FL, FedAvg, FedAvgM, FedProx
            XGB Hist, XGB Tree

        Examples:
            XGB Hist FL (all sites connected)
            MLP FedProx FL
            MLP Local Site 1
            LR Central

    Project
        Clinical task or project name, for example:
            Sepsis
            AMI
            Diabetes

    Total Run Time (from FL Console)
        Training duration. Accepted formats are:
            - numeric values interpreted as minutes; or
            - duration strings such as HH:MM:SS or D days HH:MM:SS.

    Valid to include in analysis
        Boolean-like value indicating whether the run should be included.
        Accepted true values include:
            TRUE, true, yes, y, 1

Other columns may be present in the CSV but are ignored.

Output
------
A CSV containing total training time grouped by:

    Project
    Base model
    Training strategy
    FL algorithm

The output also reports the number of runs contributing to each total and
the summed training time in hours.
"""
from pathlib import Path
import re
import pandas as pd


INPUT_CSV = Path(r"C:\path\to\run_tracker.csv")
OUTPUT_CSV = Path(r"C:\path\to\total_training_time_summary_hours.csv")

VALID_COL = "Valid to include in analysis"
NAME_COL = "Name"
PROJECT_COL = "Project"
RUNTIME_COL = "Total Run Time (from FL Console)"

# Exclude leave-site-out / leave-site sensitivity runs from the main timing table.
# These rows are valid runs, but they are not part of the primary training-time calculation.
EXCLUDE_NAME_PATTERNS = [
    r"leave[- ]?site",
    r"leave[- ]?site[- ]?out",
    r"site[- ]?out",
]


def to_bool(x):
    """Robust TRUE/FALSE conversion for CSV columns."""
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "t", "yes", "y", "1"}


def parse_runtime_to_hours(x):
    """
    Convert Total Run Time to hours.

    Handles both formats:
      1) numeric values interpreted as minutes, as in the tracker sheet
      2) duration strings such as '0:40:26.707672' interpreted as hh:mm:ss
    """
    if pd.isna(x) or str(x).strip() == "":
        return 0.0

    text = str(x).strip()

    # If the tracker column is exported as a plain number, treat it as minutes.
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return float(text) / 60.0

    # If the tracker column is exported as hh:mm:ss or d days hh:mm:ss.
    td = pd.to_timedelta(text, errors="coerce")
    if pd.notna(td):
        return td.total_seconds() / 3600.0

    raise ValueError(f"Could not parse run time value: {x!r}")


def classify_run(name):
    """
    Return base_model, training_strategy, fl_algorithm from the tracker Name.
    Local site rows are intentionally grouped together as one Local total.
    """
    name = str(name).strip()
    low = name.lower()

    if low.startswith("xgb"):
        base_model = "XGBoost"
    elif low.startswith("mlp"):
        base_model = "MLP"
    elif low.startswith("lr"):
        base_model = "LR"
    else:
        base_model = "Other"

    if "local" in low:
        training_strategy = "Local"
    elif "central" in low:
        training_strategy = "Centralised"
    elif " fl" in low or "fedavg" in low or "fedprox" in low:
        training_strategy = "FL"
    else:
        training_strategy = "Other"

    fl_algorithm = ""
    if training_strategy == "FL":
        if "xgb hist" in low:
            fl_algorithm = "HistAgg"
        elif "xgb tree" in low:
            fl_algorithm = "TreeShare"
        elif "fedavgm" in low:
            fl_algorithm = "FedAvgM"
        elif "fedprox" in low:
            fl_algorithm = "FedProx"
        elif "fedavg" in low or "fedavgfl" in low or "fedavg fl" in low or "fedavg" in low.replace(" ", ""):
            fl_algorithm = "FedAvg"
        else:
            fl_algorithm = "Unknown FL"

    return pd.Series({
        "Base model": base_model,
        "Training strategy": training_strategy,
        "FL algorithm": fl_algorithm,
    })


def main():
    df = pd.read_csv(INPUT_CSV)

    # 1. Keep only valid runs.
    df = df[df[VALID_COL].apply(to_bool)].copy()

    # 1b. Exclude leave-site / leave-site-out runs from calculations.
    #     These are secondary/sensitivity runs and should not contribute to
    #     the main total training time table.
    exclude_regex = "|".join(EXCLUDE_NAME_PATTERNS)
    leave_site_mask = df[NAME_COL].astype(str).str.contains(
        exclude_regex, case=False, regex=True, na=False
    )
    excluded_leave_site_n = int(leave_site_mask.sum())
    df = df[~leave_site_mask].copy()

    # 2. Convert run time to hours.
    df["training_time_h"] = df[RUNTIME_COL].apply(parse_runtime_to_hours)

    # 3. Classify each row.
    class_cols = df[NAME_COL].apply(classify_run)
    df = pd.concat([df, class_cols], axis=1)

    # 4. Sum by project + model + training strategy + FL algorithm.
    #    This combines all Local Site 1/2/3/... rows into one Local total.
    summary = (
        df.groupby(
            [PROJECT_COL, "Base model", "Training strategy", "FL algorithm"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{
                "Number of runs included": (NAME_COL, "size"),
                "Total training time (h)": ("training_time_h", "sum"),
            }
        )
    )

    summary["Total training time (h)"] = summary["Total training time (h)"].round(3)

    strategy_order = {"Local": 1, "Centralised": 2, "FL": 3, "Other": 4}
    model_order = {"XGBoost": 1, "MLP": 2, "LR": 3, "Other": 4}
    summary["_model_order"] = summary["Base model"].map(model_order).fillna(99)
    summary["_strategy_order"] = summary["Training strategy"].map(strategy_order).fillna(99)
    summary = summary.sort_values(
        [PROJECT_COL, "_model_order", "_strategy_order", "FL algorithm"]
    ).drop(columns=["_model_order", "_strategy_order"])

    summary.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")
    print(summary.to_string(index=False))
    print(f"\nExcluded leave-site rows: {excluded_leave_site_n}")
    print(f"Saved: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
