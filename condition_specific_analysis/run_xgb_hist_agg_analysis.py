"""
Run the Sepsis XGBoost HistAgg analysis using the shared FL analysis executor.

This condition-specific script supplies Sepsis paths and analysis settings to
``functions.executor.run_analysis()``. With ``model_family="xgb"`` and
``algorithm_name="xgb_hist_agg"``, the executor uses the XGBoost-specific
loading, evaluation, stratified, and leave-site-out modules together with
shared reporting and comparison functions.

Inputs
------
The pipeline requires paths to:
- FL site result directories (FedEMRai output).
- The FL server result directory (FedEMRai output).
- Centralised model results (FedEMRai output).
- Independently trained local model results (FedEMRai output).
- Optional raw site-level CSV files for stratified analysis.
- Optional leave-site-out FL and centralised result directories (FedEMRai output).
- Optional pooled central-model performance values (FedEMRai output).

Outputs
-------
A timestamped analysis folder is created under ``output_root_dir`` containing:
- Primary held-out performance metrics and prediction CSV files.
- Secondary analysis outputs, when enabled.
- Paired within-site differences in AUC and Brier score with 95% confidence
  intervals.
- Leave-site-out and fidelity / transportability results, when enabled.
- Stratified analysis outputs, when enabled.
- FL training-time and data-transfer summaries, when enabled.
- Validation-AUC figures in PNG/PDF format.
- A Word report containing the main analysis tables and figures.

Run
---
Navigate to the .venv directory of your project, then run in the terminal by:
    python "path to this script"
    Example:python "C:\\path\\to\\run_xgb_hist_agg_analysis.py"
"""

import sys
import pandas as pd
from pathlib import Path

# Directory configuration

# Add the FL-Benchmarking repository to the Python import path.
FUNCTIONS_DIR = Path(r"C:\path\to\FL-Benchmarking")

FL_RESULTS_ROOT = Path(
    r"C:\path\to\federated_learning_result"
)

RAW_DATA_ROOT = Path(
    r"C:\path\to\baseline_characteristics"
)

OUTPUT_DIR = Path(r"C:\path\to\analysis_outputs")


sys.path.insert(0, str(FUNCTIONS_DIR))


from functions.executor import run_analysis

# Provide the necessary paths and parameters to run the analysis 
out = run_analysis(
    model_family="xgb",

    # Use exactly xgb_hist_agg so timing/data extraction uses the XGB Hist Agg logic.
    algorithm_name="xgb_hist_agg",

    # Where to place the outputs of this analysis.
    output_root_dir=OUTPUT_DIR,

    site_sample_sizes={
        "1": 6837,
        "2": 2337,
        "3": 3112,
        "4": 993,
    },

    # Paths to the federated learning site result folders (output of FedEMRai). 
    # The keys should be the site IDs (integers), and the values should be the paths to the respective site result folders. 
    fl_site_results_dirs={
        "1": FL_RESULTS_ROOT / "xgb_hist_agg" / "site-aa68-01_xgb_hist_agg_20260610_004719",
        "2": FL_RESULTS_ROOT / "xgb_hist_agg" / "site-aa68-02_xgb_hist_agg_20260610_004719",
        "3": FL_RESULTS_ROOT / "xgb_hist_agg" / "site-aa68-03_xgb_hist_agg_20260610_004719",
        "4": FL_RESULTS_ROOT / "xgb_hist_agg" / "site-aa68-04_xgb_hist_agg_20260610_004718",
    },

    # Path to the federated learning server results folder.
    server_results_dir=FL_RESULTS_ROOT / "xgb_hist_agg" / "server_run_xgb_hist_agg_20260610_004718",

    # Centralised result folder.
    central_results_dir=FL_RESULTS_ROOT / "xgb_central" / "site-aa68-07_xgb_hist_agg_20260610_042133",

    # Local result folders.
    local_results_dirs={
        "1": FL_RESULTS_ROOT / "xgb_local" / "site_1" / "site-aa68-07_xgb_hist_agg_20260610_151014",
        "2": FL_RESULTS_ROOT / "xgb_local" / "site_2" / "site-aa68-07_xgb_hist_agg_20260610_152612",
        "3": FL_RESULTS_ROOT / "xgb_local" / "site_3" / "site-aa68-07_xgb_hist_agg_20260610_154135",
        "4": FL_RESULTS_ROOT / "xgb_local" / "site_4" / "site-aa68-07_xgb_hist_agg_20260610_160033",
    },
    
    # Central model result on pooled held out data
    central_pooled_results_df = pd.DataFrame([{
        "auc": 0.9770,
        "brier": 0.0462,
        "auprc": 0.8908,
    }]),

    # If None, common best round is selected by mean validation AUC across FL sites.
    best_round=None,

    # Timing/data-transfer table
    include_timing_data_summary=True,
    strict_timing_data_summary=False,

    # Secondary analysis
    secondary_analysis=True,

    # Stratified analysis. Implemented for model_family="xgb".
    stratified_analysis=True,
    age_col="age",
    sex_col="sexfemale01",
    raw_csv_by_site={
        "1": RAW_DATA_ROOT / "site_1.csv",

        "2": RAW_DATA_ROOT / "site_2.csv",

        "3": RAW_DATA_ROOT / "site_3.csv",

        "4": RAW_DATA_ROOT / "site_4.csv",
    },
    
    
    # Validation AUC figure settings
    val_auc_plot_kwargs={
        "figsize": (7, 2.5),

        "x_label": "FL communication round",
        "y_label": "Site-specific validation AUC",

        "y_axis_min": 0.90,
        "y_axis_max": 1.00,
        "x_tick_interval": 50,

        "line_width": 2.2,
        "marker_size": 80,

        "axis_label_fontsize": 7,
        "tick_label_fontsize": 8,
        "legend_fontsize": 6.8,

        "legend_ncol": None,
        "max_legend_rows": 2,
        "legend_bottom_y": 0.015,

        "bottom_margin": 0.28,
        "left_margin": 0.10,
        "right_margin": 0.98,
        "top_margin": 0.95,

        "width_inches": 6.5,

        "dpi": 300,
        "formats": ("png", "pdf"),
    },
    
    paired_comparison_analysis=True,
    
    # Leave-one-site-out analysis. Implemented for model_family="xgb".
    leave_site_out_analysis=True,
    leave_site_out_runs={
        "1": {
            # FL run trained with sites 2, 3, 4 only.
            "fl_site_results_dirs": {
                "2": FL_RESULTS_ROOT / "leave_site_1" / "site-aa68-02_xgb_hist_agg_20260610_020432",
                "3": FL_RESULTS_ROOT / "leave_site_1" / "site-aa68-03_xgb_hist_agg_20260610_020431",
                "4": FL_RESULTS_ROOT / "leave_site_1" / "site-aa68-04_xgb_hist_agg_20260610_020432",
            },
            "server_results_dir": FL_RESULTS_ROOT / "leave_site_1" / "server_run_xgb_hist_agg_20260610_020431",
            "central_results_dir": FL_RESULTS_ROOT / "leave_site_1" / "site-aa68-07_xgb_hist_agg_20260610_044337",
            
            
        },
        "2": {
            # FL run trained with sites 1, 3, 4 only.
            "fl_site_results_dirs": {
                "1": FL_RESULTS_ROOT / "leave_site_2" / "site-aa68-01_xgb_hist_agg_20260610_024102",
                "3": FL_RESULTS_ROOT / "leave_site_2" / "site-aa68-03_xgb_hist_agg_20260610_024102",
                "4": FL_RESULTS_ROOT / "leave_site_2" / "site-aa68-04_xgb_hist_agg_20260610_024101",
            },
            "server_results_dir": FL_RESULTS_ROOT / "leave_site_2" / "server_run_xgb_hist_agg_20260610_024059",
            "central_results_dir": FL_RESULTS_ROOT / "leave_site_2" / "site-aa68-07_xgb_hist_agg_20260610_050452",
            
        },
        "3": {
            # FL run trained with sites 1, 2, 4 only.
            "fl_site_results_dirs": {
                "1": FL_RESULTS_ROOT / "leave_site_3" / "site-aa68-01_xgb_hist_agg_20260610_031440",
                "2": FL_RESULTS_ROOT / "leave_site_3" / "site-aa68-02_xgb_hist_agg_20260610_031441",
                "4": FL_RESULTS_ROOT / "leave_site_3" / "site-aa68-04_xgb_hist_agg_20260610_031440",
            },
            "server_results_dir": FL_RESULTS_ROOT / "leave_site_3" / "server_run_xgb_hist_agg_20260610_031438",
            "central_results_dir": FL_RESULTS_ROOT / "leave_site_3" / "site-aa68-07_xgb_hist_agg_20260610_052609",
            
        },
        "4": {
            # FL run trained with sites 1, 2, 3 only.
            "fl_site_results_dirs": {
                "1": FL_RESULTS_ROOT / "leave_site_4" / "site-aa68-01_xgb_hist_agg_20260610_034713",
                "2": FL_RESULTS_ROOT / "leave_site_4" / "site-aa68-02_xgb_hist_agg_20260610_034713",
                "3": FL_RESULTS_ROOT / "leave_site_4" / "site-aa68-03_xgb_hist_agg_20260610_034713",
            },
            "server_results_dir": FL_RESULTS_ROOT / "leave_site_4" / "server_run_xgb_hist_agg_20260610_034712",
            "central_results_dir": FL_RESULTS_ROOT / "leave_site_4" / "site-aa68-07_xgb_hist_agg_20260610_144908",

        },
    },

    verbose=True,
)

print("\nDone.")