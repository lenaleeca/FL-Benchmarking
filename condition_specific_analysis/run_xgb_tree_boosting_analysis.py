"""
Run the Sepsis XGBoost TreeBoosting analysis using the shared FL analysis executor.

This condition-specific script supplies Sepsis paths and analysis settings to
``functions.executor.run_analysis()``. With ``model_family="xgb"`` and
``algorithm_name="xgb_tree_boosting"``, the executor uses the XGBoost-specific
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
    Example:python "C:\\path\\to\\run_xgb_tree_boosting_analysis.py"
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

OUTPUT_DIR = Path(r"C:\path\to\analysis_outputs")

sys.path.insert(0, str(FUNCTIONS_DIR),)


from functions.executor import run_analysis

out = run_analysis(
    model_family="xgb",
    
    algorithm_name="xgb_tree_boosting",

    # Where to place the outputs of this posthoc analysis (metrics, word report, etc.)
    output_root_dir=OUTPUT_DIR,


    # Provide site sample sizes for the plots
    site_sample_sizes={
        "1": 6837,
        "2": 2337,
        "3": 3112,
        "4": 993,
    },

    # Paths to the federated learning site result folders (output of FedEMRai). 
    # The keys should be the site IDs (integers), and the values should be the paths to the respective site result folders. 
    fl_site_results_dirs={
        "1": FL_RESULTS_ROOT / "xgb_tree_boosting" / "site-aa68-01_xgb_tree_boosting_20260610_013732",
        "2": FL_RESULTS_ROOT / "xgb_tree_boosting" / "site-aa68-02_xgb_tree_boosting_20260610_013731",
        "3": FL_RESULTS_ROOT / "xgb_tree_boosting" / "site-aa68-03_xgb_tree_boosting_20260610_013731",
        "4": FL_RESULTS_ROOT / "xgb_tree_boosting" / "site-aa68-04_xgb_tree_boosting_20260610_013731",
    },

    # Path to the federated learning server results folder.
    server_results_dir=FL_RESULTS_ROOT / "xgb_tree_boosting" / "server_run_xgb_tree_boosting_20260610_013730",

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

    # Timing/data-transfer table
    include_timing_data_summary=True,
    strict_timing_data_summary=False,

    # Secondary analysis
    secondary_analysis=True,
    
    # Validation AUC figure settings
        val_auc_plot_kwargs={
            "figsize": (7, 2.5),
    
            "x_label": "FL communication round",
            "y_label": "Site-specific validation AUC",
    
            "y_axis_min": 0.50,
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


    verbose=True,
)

print("\nDone.")