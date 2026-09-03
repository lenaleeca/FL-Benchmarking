"""
Run the complete federated learning analysis pipeline for a selected model
family and algorithm.

The script identifies the common best FL communication round (primary analysis), loads the FL,
centralised, and local models, evaluates them on each site's held-out data,
and optionally performs secondary site-specific-round analysis, stratified
analysis, leave-site-out analysis, paired within-site comparisons, fidelity /
transportability comparisons, and training-time/data-transfer summaries.

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
"""

from pathlib import Path
from datetime import datetime
import importlib
import pandas as pd
from sklearn import metrics

from functions.common.get_common_best_round import (
    get_common_best_round,
    find_metric_file,
)
from functions.common.paired_comparisons import run_paired_comparison_analysis
from functions.common.path_config import (
    normalize_site_path_map,
    sorted_site_ids_from_map,
    site_dir_from_root_or_map,
)
from functions.common.report_to_word import save_analysis_report_to_word
from functions.common.timing_data_summary import collect_timing_data_summary


SUPPORTED_MODEL_FAMILIES = {"xgb", "mlp", "lr"}


def infer_model_family(algorithm_name):
    text = str(algorithm_name or "").strip().lower()

    if text == "xgb_hist_agg" or text.startswith("xgb"):
        return "xgb"

    if text.startswith("mlp") or text.startswith("tabnn"):
        return "mlp"

    if text.startswith("lr") or text.startswith("glm") or "logistic" in text:
        return "lr"

    raise ValueError(
        "Could not infer model_family from algorithm_name. "
        "Please pass model_family='xgb', model_family='mlp', "
        "or model_family='lr'."
    )


def _normalise_model_family(model_family, algorithm_name):
    if model_family is None:
        model_family = infer_model_family(algorithm_name)

    model_family = str(model_family).strip().lower()

    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"model_family must be one of "
            f"{sorted(SUPPORTED_MODEL_FAMILIES)}, "
            f"got {model_family!r}."
        )

    return model_family


def _load_family_modules(model_family):
    """
    Load model-family-specific modules used by the shared analysis pipeline.
    """

    if model_family == "xgb":
        return {
            "load_models": importlib.import_module(
                "functions.xgb.load_models"
            ),
            "prepare_input": importlib.import_module(
                "functions.xgb.prepare_input"
            ),
            "evaluation": importlib.import_module(
                "functions.xgb.evaluation"
            ),
        }

    if model_family == "mlp":
        return {
            "load_models": importlib.import_module(
                "functions.mlp.load_models"
            ),
            "prepare_input": importlib.import_module(
                "functions.mlp.prepare_input"
            ),
            "evaluation": importlib.import_module(
                "functions.mlp.evaluation"
            ),
        }

    if model_family == "lr":
        return {
            "load_models": importlib.import_module(
                "functions.lr.load_models"
            ),
            "prepare_input": importlib.import_module(
                "functions.lr.prepare_input"
            ),
            "evaluation": importlib.import_module(
                "functions.lr.evaluation"
            ),
        }

    raise ValueError(
        f"Unsupported model_family: {model_family}"
    )


def _load_central_model(model_family, loaders, central_results_dir, *, dropout=0.0, device="cpu"):
    if model_family == "xgb":
        return loaders.load_central_xgb(central_results_dir)
    if model_family == "lr":
        return loaders.load_central_lr(central_results_dir, device=device)
    return loaders.load_central_mlp(central_results_dir, dropout=dropout, device=device)


def _load_local_models(model_family, loaders, local_results_dirs, site_ids, *, dropout=0.0, device="cpu"):
    if model_family == "xgb":
        return loaders.load_local_xgb_by_site(local_results_dirs=local_results_dirs, site_ids=site_ids)
    if model_family == "lr":
        return loaders.load_local_lr_by_site(
            local_results_dirs=local_results_dirs,
            site_ids=site_ids,
            device=device,
        )
    return loaders.load_local_mlp_by_site(
        local_results_dirs=local_results_dirs,
        site_ids=site_ids,
        dropout=dropout,
        device=device,
    )


def _load_primary_fl_model(model_family, loaders, server_results_dir, best_round, *, dropout=0.0, device="cpu"):
    if model_family == "xgb":
        return loaders.load_fl_xgb(server_results_dir=server_results_dir, best_round=best_round)
    if model_family == "lr":
        return loaders.load_fl_lr(
            server_results_dir=server_results_dir,
            best_round=best_round,
            device=device,
        )
    return loaders.load_fl_mlp(
        server_results_dir=server_results_dir,
        best_round=best_round,
        dropout=dropout,
        device=device,
    )


def _load_secondary_fl_models(model_family, loaders, fl_site_results_dirs, site_ids, *, dropout=0.0, device="cpu"):
    if model_family == "xgb":
        return loaders.load_fl_local_xgb_by_site_best_rounds(
            fl_site_results_dirs=fl_site_results_dirs,
            site_ids=site_ids,
        )
    if model_family == "lr":
        return loaders.load_fl_local_lr_by_site_best_rounds(
            fl_site_results_dirs=fl_site_results_dirs,
            site_ids=site_ids,
            device=device,
        )
    return loaders.load_fl_local_mlp_by_site_best_rounds(
        fl_site_results_dirs=fl_site_results_dirs,
        site_ids=site_ids,
        dropout=dropout,
        device=device,
    )


def build_models_for_site(
    *,
    site_id,
    central_model,
    local_models,
    fl_model=None,
    fl_models_by_site=None,
):
    local_key = f"site_{site_id}"
    if local_key not in local_models:
        raise KeyError(f"{local_key} not found in local_models.")

    if fl_models_by_site is not None:
        if local_key not in fl_models_by_site:
            raise KeyError(f"{local_key} not found in fl_models_by_site.")
        fl_info = fl_models_by_site[local_key]
    elif fl_model is not None:
        fl_info = fl_model
    else:
        raise ValueError("Either fl_model or fl_models_by_site must be provided.")

    return {
        "local": local_models[local_key],
        "central": central_model,
        "fl_site": fl_info,
    }


def _evaluate_one_model(
    *,
    model_family,
    evaluator,
    model_info,
    csv_path,
    site_id,
    model_type,
    dataset_name,
    output_dir,
    best_round=None,
    label_col=None,
    verbose=True,
):
    if model_family == "xgb":
        metrics = evaluator.evaluate_model_on_csv(
            booster=model_info["booster"],
            model_path=model_info["model_path"],
            csv_path=csv_path,
            site_id=site_id,
            model_type=model_type,
            dataset_name=dataset_name,
            output_dir=output_dir,
            best_round=best_round,
            verbose=verbose,
        )
        metrics["model_family"] = "xgb"
        return metrics

    metrics = evaluator.evaluate_model_on_csv(
        model_info=model_info,
        csv_path=csv_path,
        site_id=site_id,
        model_type=model_type,
        dataset_name=dataset_name,
        output_dir=output_dir,
        best_round=best_round,
        label_col=label_col,
        verbose=verbose,
    )
    metrics["model_family"] = model_family
    return metrics


def evaluate_heldout_all_sites(
    *,
    model_family,
    prepare_input_module,
    evaluation_module,
    results_root_dir=None,
    fl_site_results_dirs,
    site_ids,
    central_model,
    local_models,
    output_dir,
    dataset_name,
    fl_model=None,
    fl_models_by_site=None,
    label_col=None,
    verbose=True,
):
    rows = []

    for site_id in site_ids:
        fl_site_dir = site_dir_from_root_or_map(
            results_root_dir,
            site_id,
            path_map=fl_site_results_dirs,
        )
        heldout_csv = prepare_input_module.get_test_partition_csv(fl_site_dir)

        if verbose:
            print(f"\nEvaluating {dataset_name} data for site {site_id}")
            print(f"Held-out site folder: {fl_site_dir}")
            print(f"Held-out CSV: {heldout_csv}")

        models = build_models_for_site(
            site_id=site_id,
            central_model=central_model,
            local_models=local_models,
            fl_model=fl_model,
            fl_models_by_site=fl_models_by_site,
        )

        for model_type, model_info in models.items():
            if verbose:
                print(f"\nProcessing {model_type} model on site {site_id}, dataset={dataset_name}")
                print(f"  model round: {model_info.get('model_round')}")
                print(f"  best round : {model_info.get('best_round')}")
                print(f"  best epoch : {model_info.get('best_epoch')}")
                print(f"  best iteration : {model_info.get('best_iteration')}")

            metrics = _evaluate_one_model(
                model_family=model_family,
                evaluator=evaluation_module,
                model_info=model_info,
                csv_path=heldout_csv,
                site_id=site_id,
                model_type=model_type,
                dataset_name=dataset_name,
                output_dir=output_dir,
                best_round=model_info.get("best_round"),
                label_col=label_col,
                verbose=verbose,
            )
            metrics["model_round"] = model_info.get("model_round")
            metrics["best_round"] = model_info.get("best_round", metrics.get("best_round", pd.NA))
            metrics["best_epoch"] = model_info.get("best_epoch", pd.NA)
            metrics["best_iteration"] = model_info.get("best_iteration", pd.NA)
            metrics["model_kind"] = model_info.get("model_kind")
            metrics["loaded_from"] = model_info.get("loaded_from", pd.NA)
            metrics["loaded_best_model"] = model_info.get("loaded_best_model", False)
            rows.append(metrics)

    return pd.DataFrame(rows)


def split_results_by_model_type(results_df):
    return {
        "fl_site": results_df[results_df["model_type"] == "fl_site"].copy(),
        "central": results_df[results_df["model_type"] == "central"].copy(),
        "local": results_df[results_df["model_type"] == "local"].copy(),
    }


def save_primary_heldout_outputs(*, output_dir, primary_results_df, best_round):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "primary_metrics_out": output_dir / "heldout_primary_all_model_metrics.csv",
        "combined_metrics_out": output_dir / "heldout_all_model_metrics.csv",
        "fl_site_metrics_out": output_dir / f"heldout_primary_fl_site_common_global_metrics_round_{best_round}.csv",
        "central_metrics_out": output_dir / "heldout_primary_central_metrics.csv",
        "local_metrics_out": output_dir / "heldout_primary_local_metrics.csv",
    }
    primary_results_df.to_csv(paths["primary_metrics_out"], index=False)
    primary_results_df.to_csv(paths["combined_metrics_out"], index=False)
    split = split_results_by_model_type(primary_results_df)
    split["fl_site"].to_csv(paths["fl_site_metrics_out"], index=False)
    split["central"].to_csv(paths["central_metrics_out"], index=False)
    split["local"].to_csv(paths["local_metrics_out"], index=False)
    return paths, split


def save_secondary_heldout_outputs(*, output_dir, secondary_results_df, secondary_best_round_by_site):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "secondary_metrics_out": output_dir / "heldout_secondary_all_model_metrics.csv",
        "fl_site_metrics_out": output_dir / "heldout_secondary_fl_site_local_best_metrics.csv",
        "central_metrics_out": output_dir / "heldout_secondary_central_metrics.csv",
        "local_metrics_out": output_dir / "heldout_secondary_local_metrics.csv",
        "secondary_rounds_out": output_dir / "heldout_secondary_fl_site_best_rounds.csv",
    }
    secondary_results_df.to_csv(paths["secondary_metrics_out"], index=False)
    split = split_results_by_model_type(secondary_results_df)
    split["fl_site"].to_csv(paths["fl_site_metrics_out"], index=False)
    split["central"].to_csv(paths["central_metrics_out"], index=False)
    split["local"].to_csv(paths["local_metrics_out"], index=False)

    secondary_rounds_df = pd.DataFrame([
        {"site_id": site_id, "site_specific_fl_best_round": int(best_round)}
        for site_id, best_round in secondary_best_round_by_site.items()
    ])
    if not secondary_rounds_df.empty:
        secondary_rounds_df["site_num"] = pd.to_numeric(secondary_rounds_df["site_id"], errors="coerce")
        secondary_rounds_df = secondary_rounds_df.sort_values(["site_num", "site_id"]).drop(columns=["site_num"])
    secondary_rounds_df.to_csv(paths["secondary_rounds_out"], index=False)
    return paths, split, secondary_rounds_df


def combine_primary_secondary_results(*, output_dir, primary_results_df, secondary_results_df):
    output_dir = Path(output_dir)
    primary = primary_results_df.copy()
    primary["analysis"] = "primary"
    secondary = secondary_results_df.copy()
    secondary["analysis"] = "secondary"
    combined = pd.concat([primary, secondary], ignore_index=True)
    combined_out = output_dir / "heldout_primary_secondary_all_model_metrics.csv"
    combined.to_csv(combined_out, index=False)
    return combined, combined_out


def run_analysis(
    *,
    output_root_dir,
    algorithm_name,
    model_family=None,
    site_sample_sizes=None,
    age_col="age",
    sex_col="sex",
    raw_csv_by_site=None,
    stratified_analysis=False,
    leave_site_out_analysis=False,
    leave_site_out_runs=None,
    fl_site_results_dirs,
    server_results_dir,
    central_results_dir,
    central_pooled_results_df=None,
    local_results_dirs,
    best_round=None,
    results_root_dir=None,
    label_col=None,
    device="cpu",
    dropout=0.0,
    secondary_analysis=True,
    paired_comparison_analysis=True,
    paired_comparison_n_boot=1000,
    paired_comparison_ci_level=0.95,
    paired_comparison_random_state=1111,
    include_timing_data_summary=True,
    strict_timing_data_summary=False,
    val_auc_plot_kwargs=None,
    verbose=True,
):
    """
    Run the complete federated learning analysis pipeline.

    The pipeline selects the common FL round (primary analysis), evaluates FL, centralised, and
    local models on held-out data, optionally performs secondary, stratified,
    leave-site-out, and timing analyses, and generates CSV, figure, and Word
    report outputs.
    """
    model_family = _normalise_model_family(model_family, algorithm_name)
    modules = _load_family_modules(model_family)
    loaders = modules["load_models"]
    prepare_input_module = modules["prepare_input"]
    evaluation_module = modules["evaluation"]

    if model_family != "xgb" and (stratified_analysis or leave_site_out_analysis):
        raise NotImplementedError(
            "stratified_analysis and leave_site_out_analysis are currently implemented for model_family='xgb' only."
        )

    output_root_dir = Path(output_root_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = (
    output_root_dir
    / f"{algorithm_name}_analysis_{timestamp}"
    )
    output_dir = run_output_dir / "heldout_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    word_out_path = (
    run_output_dir
    / f"{algorithm_name}_analysis_report_{timestamp}.docx"
    )

    fl_site_results_dirs = normalize_site_path_map(fl_site_results_dirs, name="fl_site_results_dirs")
    local_results_dirs = normalize_site_path_map(local_results_dirs, name="local_results_dirs")
    site_ids = sorted_site_ids_from_map(fl_site_results_dirs)

    if verbose:
        print(f"\nAlgorithm: {algorithm_name}")
        print(f"Model family: {model_family}")
        print(f"Run secondary analysis: {secondary_analysis}")
        print(f"Run stratified analysis: {stratified_analysis}")
        print(f"Run leave-site-out analysis: {leave_site_out_analysis}")
        print(f"Include timing/data summary: {include_timing_data_summary}")
        print(f"Output root: {output_root_dir}")
        print(f"Run output dir: {run_output_dir}")
        print(f"Server results dir: {server_results_dir}")
        print(f"Central results dir: {central_results_dir}")
        print(f"Detected/effective site IDs: {site_ids}")

    if best_round is None:
        best_round, round_summary_df, metrics_files = get_common_best_round(
            fl_site_results_dirs=fl_site_results_dirs,
            verbose=verbose,
        )
    else:
        best_round = int(best_round)
        round_summary_df = pd.DataFrame()
        metrics_files = find_metric_file(
            fl_site_results_dirs=fl_site_results_dirs,
            verbose=verbose,
        )

    central_model = _load_central_model(
        model_family,
        loaders,
        central_results_dir,
        dropout=dropout,
        device=device,
    )
    local_models = _load_local_models(
        model_family,
        loaders,
        local_results_dirs,
        site_ids,
        dropout=dropout,
        device=device,
    )
    primary_fl_model = _load_primary_fl_model(
        model_family,
        loaders,
        server_results_dir,
        best_round,
        dropout=dropout,
        device=device,
    )

    if verbose:
        print(f"\nLoaded central {model_family.upper()} model:")
        print(f"  path={central_model['model_path']}")
        print(f"  kind={central_model.get('model_kind')}")
        print(f"  round={central_model.get('model_round')}")
        print(f"  best_epoch={central_model.get('best_epoch')}")
        print(f"  best_iteration={central_model.get('best_iteration')}")
        print(f"\nLoaded primary common server GLOBAL {model_family.upper()} model:")
        print(f"  path={primary_fl_model['model_path']}")
        print(f"  kind={primary_fl_model.get('model_kind')}")
        print(f"  round={primary_fl_model.get('model_round')}")
        print(f"\nLoaded independently trained local {model_family.upper()} models:")
        for site_name, model_info in local_models.items():
            print(
                f"  {site_name}: {model_info['model_path']} "
                f"(round={model_info.get('model_round')}, "
                f"best_epoch={model_info.get('best_epoch')}, "
                f"best_iteration={model_info.get('best_iteration')})"
            )
    primary_results_df = evaluate_heldout_all_sites(
        model_family=model_family,
        prepare_input_module=prepare_input_module,
        evaluation_module=evaluation_module,
        results_root_dir=results_root_dir,
        fl_site_results_dirs=fl_site_results_dirs,
        site_ids=site_ids,
        central_model=central_model,
        local_models=local_models,
        fl_model=primary_fl_model,
        output_dir=output_dir,
        dataset_name="primary_heldout",
        label_col=label_col,
        verbose=verbose,
    )

    primary_output_paths, primary_split = save_primary_heldout_outputs(
        output_dir=output_dir,
        primary_results_df=primary_results_df,
        best_round=best_round,
    )
    fl_site_results_df = primary_split["fl_site"]
    central_results_df = primary_split["central"]
    local_results_df = primary_split["local"]

    if secondary_analysis:
        secondary_fl_models_by_site = _load_secondary_fl_models(
            model_family,
            loaders,
            fl_site_results_dirs,
            site_ids,
            dropout=dropout,
            device=device,
        )
        secondary_best_round_by_site = {
            site_id: int(secondary_fl_models_by_site[f"site_{site_id}"]["best_round"])
            for site_id in site_ids
        }
        if verbose:
            print(f"\nLoaded secondary site-specific FL LOCAL {model_family.upper()} models:")
            for site_name, model_info in secondary_fl_models_by_site.items():
                print(
                    f"  {site_name}: {model_info['model_path']} "
                    f"(round={model_info.get('model_round')}, best_round={model_info.get('best_round')})"
                )
        secondary_results_df = evaluate_heldout_all_sites(
            model_family=model_family,
            prepare_input_module=prepare_input_module,
            evaluation_module=evaluation_module,
            results_root_dir=results_root_dir,
            fl_site_results_dirs=fl_site_results_dirs,
            site_ids=site_ids,
            central_model=central_model,
            local_models=local_models,
            fl_models_by_site=secondary_fl_models_by_site,
            output_dir=output_dir,
            dataset_name="secondary_heldout",
            label_col=label_col,
            verbose=verbose,
        )
        secondary_output_paths, secondary_split, secondary_rounds_df = save_secondary_heldout_outputs(
            output_dir=output_dir,
            secondary_results_df=secondary_results_df,
            secondary_best_round_by_site=secondary_best_round_by_site,
        )
        secondary_fl_site_results_df = secondary_split["fl_site"]
        all_results_df, combined_all_out = combine_primary_secondary_results(
            output_dir=output_dir,
            primary_results_df=primary_results_df,
            secondary_results_df=secondary_results_df,
        )
    else:
        secondary_best_round_by_site = {}
        secondary_results_df = pd.DataFrame()
        secondary_output_paths = {}
        secondary_rounds_df = pd.DataFrame()
        secondary_fl_site_results_df = pd.DataFrame()
        all_results_df = primary_results_df.copy()
        combined_all_out = primary_output_paths["combined_metrics_out"]

    if stratified_analysis:
        stratified_module = importlib.import_module("functions.xgb.stratified_analysis")
        stratified_outputs = stratified_module.run_stratified_analysis(
            results_root_dir=results_root_dir,
            fl_site_results_dirs=fl_site_results_dirs,
            site_ids=site_ids,
            best_round=best_round,
            central_booster=central_model["booster"],
            central_model_path=central_model["model_path"],
            local_models=local_models,
            fl_booster=primary_fl_model["booster"],
            fl_model_path=primary_fl_model["model_path"],
            output_dir=output_dir,
            age_col=age_col,
            sex_col=sex_col,
            raw_csv_by_site=raw_csv_by_site,
            verbose=verbose,
        )
        stratified_results_df = stratified_outputs["stratified_results_df"]
        sex_auc_difference_df = stratified_outputs["sex_auc_difference_df"]
        age_auc_summary_df = stratified_outputs["age_auc_summary_df"]
    else:
        stratified_outputs = None
        stratified_results_df = pd.DataFrame()
        sex_auc_difference_df = pd.DataFrame()
        age_auc_summary_df = pd.DataFrame()

    
    if leave_site_out_analysis:
        lso_module = importlib.import_module("functions.xgb.leave_site_out_analysis")
        leave_site_out_outputs = lso_module.run_leave_site_out_analysis(
            primary_results_root_dir=results_root_dir,
            primary_fl_site_results_dirs=fl_site_results_dirs,
            leave_site_out_runs=leave_site_out_runs,
            primary_results_df=primary_results_df,
            output_dir=output_dir,
            verbose=verbose,
        )
        leave_site_out_summary_df = leave_site_out_outputs["leave_site_out_summary_df"]
    else:
        leave_site_out_outputs = None
        leave_site_out_summary_df = pd.DataFrame()
        
    # Paired within-site comparisons
    
    if paired_comparison_analysis:

        leave_site_out_metrics_csv = None

        if leave_site_out_outputs is not None:
            leave_site_out_metrics_csv = (
                leave_site_out_outputs.get(
                    "leave_site_out_metrics_out"
                )
            )

        paired_comparison_outputs = (
            run_paired_comparison_analysis(
                primary_metrics_csv=(
                    primary_output_paths[
                        "primary_metrics_out"
                    ]
                ),
                leave_site_out_metrics_csv=(
                    leave_site_out_metrics_csv
                ),
                output_dir=output_dir,
                n_boot=paired_comparison_n_boot,
                ci_level=paired_comparison_ci_level,
                random_state=(
                    paired_comparison_random_state
                ),
                verbose=verbose,
            )
        )

        paired_within_site_df = (
            paired_comparison_outputs[
                "paired_within_site_df"
            ]
        )

        fidelity_recovery_df = (
            paired_comparison_outputs[
                "fidelity_recovery_df"
            ]
        )

    else:
        paired_comparison_outputs = None
        paired_within_site_df = (
            pd.DataFrame()
        )
        fidelity_recovery_df = (
            pd.DataFrame()
        )

    if include_timing_data_summary:
        timing_data_summary = collect_timing_data_summary(
            algorithm_name=algorithm_name,
            server_results_dir=server_results_dir,
            fl_site_results_dirs=fl_site_results_dirs,
            site_ids=site_ids,
            output_dir=output_dir,
            strict=strict_timing_data_summary,
            verbose=verbose,
        )
        timing_data_output_paths = timing_data_summary.get("output_paths", {})
        fl_training_report_table_df = timing_data_summary.get("fl_training_report_table_df", pd.DataFrame())
    else:
        timing_data_output_paths = {}
        fl_training_report_table_df = pd.DataFrame()

    report_kwargs = dict(
        out_docx_path=word_out_path,
        algorithm_name=algorithm_name,
        analysis_label="Analysis Report",
        best_round=best_round,
        round_summary_df=round_summary_df,
        metrics_files=metrics_files,
        site_sample_sizes=site_sample_sizes,
        fl_heldout_results_df=fl_site_results_df,
        central_heldout_results_df=central_results_df,
        central_pooled_results_df=central_pooled_results_df,
        local_heldout_results_df=local_results_df,
        secondary_analysis=secondary_analysis,
        secondary_best_round_by_site=secondary_best_round_by_site,
        secondary_metrics_files=metrics_files,
        secondary_results_df=secondary_results_df,
        secondary_fl_heldout_results_df=secondary_fl_site_results_df,
        fl_training_report_table_df=fl_training_report_table_df,
        stratified_results_df=stratified_results_df,
        sex_auc_difference_df=sex_auc_difference_df,
        age_auc_summary_df=age_auc_summary_df,
        paired_within_site_df=paired_within_site_df,
        fidelity_recovery_df=fidelity_recovery_df,
        leave_site_out_summary_df=leave_site_out_summary_df,
        val_auc_plot_kwargs=val_auc_plot_kwargs,
    )
    save_analysis_report_to_word(**report_kwargs)

    if verbose:
        print(f"\nPrimary held-out metrics saved to:\n{primary_output_paths['primary_metrics_out']}")
        print(f"\nCombined primary held-out metrics saved to:\n{primary_output_paths['combined_metrics_out']}")
        if secondary_analysis:
            print(f"\nSecondary held-out metrics saved to:\n{secondary_output_paths['secondary_metrics_out']}")
            print(f"\nCombined primary + secondary metrics saved to:\n{combined_all_out}")
        if stratified_outputs is not None:
            print(f"\nStratified held-out metrics saved to:\n{stratified_outputs['stratified_metrics_out']}")
        if leave_site_out_outputs is not None:
            print(f"\nLeave-site-out metrics saved to:\n{leave_site_out_outputs['leave_site_out_metrics_out']}")
        if include_timing_data_summary:
            print(f"\nFL training report table saved to:\n{timing_data_output_paths.get('fl_training_report_table_out', '')}")
        print(f"\nWord report saved to:\n{word_out_path}")

    return {
    "algorithm_name": algorithm_name,
    "model_family": model_family,

    "best_round": best_round,
    "secondary_best_round_by_site": secondary_best_round_by_site,

    "run_output_dir": run_output_dir,
    "word_out_path": word_out_path,

    "primary_results_df": primary_results_df,
    "secondary_results_df": secondary_results_df,

    "paired_within_site_df": paired_within_site_df,
    "fidelity_recovery_df": fidelity_recovery_df,

    "leave_site_out_outputs": leave_site_out_outputs,
}
