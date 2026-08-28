from pathlib import Path
import json
import re
import xgboost as xgb


def _as_results_dir(path):
    path = Path(path)
    return path / "results" if (path / "results").is_dir() else path


def _load_xgb_model(model_path):
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


# ==================================================
# General round-specific model loader
# Used for FL GLOBAL/LOCAL round snapshots
# ==================================================

def _find_xgb_model(folder, *, kind, round_num=None):
    """
    Find one XGBoost round-specific model.

    kind:
        "LOCAL" or "GLOBAL"

    If round_num is None:
        load the max available round.

    If round_num is provided:
        load that exact round.

    Expected examples:
        site_xxx_LOCAL_round_57.json
        server_xxx_GLOBAL_round_57.json
    """
    folder = _as_results_dir(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    models = []

    for p in folder.rglob(f"*_{kind}_round_*.json"):
        name = p.name.lower()

        if any(skip in name for skip in ["metrics", "report", "best_round"]):
            continue

        m = re.search(rf"_{kind}_round_(\d+)\.json$", p.name)
        if m:
            models.append((int(m.group(1)), p))

    if not models:
        raise FileNotFoundError(f"No *_{kind}_round_*.json models found in: {folder}")

    if round_num is None:
        model_round, model_path = max(models, key=lambda x: x[0])
    else:
        round_num = int(round_num)
        matches = [(r, p) for r, p in models if r == round_num]

        if not matches:
            raise FileNotFoundError(
                f"No *_{kind}_round_{round_num}.json model found in: {folder}"
            )

        if len(matches) > 1:
            print(f"Warning: multiple {kind} round {round_num} models found, using first:")
            for _, p in matches:
                print(f" - {p}")

        model_round, model_path = sorted(matches, key=lambda x: str(x[1]))[0]

    return model_round, model_path


def _load_model_info(folder, *, kind, round_num=None, loaded_from=None):
    """
    Load a regular round-specific XGB model.

    Used for:
        - primary FL GLOBAL model at common best round
        - secondary FL LOCAL model at site-specific best round
    """
    model_round, model_path = _find_xgb_model(
        folder,
        kind=kind,
        round_num=round_num,
    )

    return {
        "booster": _load_xgb_model(model_path),
        "model_path": model_path,
        "model_round": model_round,
        "best_round": int(round_num) if round_num is not None else None,
        "model_kind": kind,
        "loaded_from": loaded_from,
        "loaded_best_model": False,
    }


# ==================================================
# BEST model loader
# Used for central and independently trained local XGB models
# ==================================================

def _find_xgb_best_model(folder):
    """
    Find one saved XGB BEST model file.

    This is used for central and independently trained local XGB models.

    Expected examples:
        site-aa68-01_xgb_hist_agg_BEST_round.json
        site-aa68-01_xgb_hist_agg_BEST_round_57.json
        something_LOCAL_BEST_round.json

    The important part is:
        *_BEST_round*.json
    """
    folder = _as_results_dir(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    matches = sorted(
        p for p in folder.rglob("*_BEST_round*.json")
        if p.is_file()
        and "metrics" not in p.name.lower()
        and "report" not in p.name.lower()
        and p.name.lower() != "best_round.json"
    )

    if not matches:
        raise FileNotFoundError(
            f"No *_BEST_round*.json model found in: {folder}"
        )

    if len(matches) > 1:
        print("Warning: multiple *_BEST_round*.json models found, using first:")
        for p in matches:
            print(f" - {p}")

    return matches[0]


def read_best_round(site_dir):
    """
    Read best_round.json from one result folder.

    Expected format:
        {"best_round": 8, "dataset": ...}

    Also accepts:
        8
    """
    folder = _as_results_dir(site_dir)
    json_path = folder / "best_round.json"

    if not json_path.is_file():
        matches = sorted(p for p in folder.rglob("best_round.json") if p.is_file())

        if len(matches) == 1:
            json_path = matches[0]
        elif len(matches) == 0:
            raise FileNotFoundError(f"No best_round.json found under: {folder}")
        else:
            raise FileNotFoundError(
                f"Multiple best_round.json files found under {folder}. "
                f"Please provide the exact result folder."
            )

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)

    if isinstance(payload, dict) and "best_round" in payload:
        return int(payload["best_round"])

    raise ValueError(
        f"Invalid best_round.json format in {json_path}. "
        f"Expected integer or dict with key 'best_round'."
    )


def read_best_round_if_available(site_dir):
    """
    Read best_round.json if available.

    Returns None if no best_round.json exists.

    This is useful for central/local BEST models because the model file itself
    may be named like:
        site-aa68-01_xgb_hist_agg_BEST_round.json

    and may not include the round number in the filename.
    """
    folder = _as_results_dir(site_dir)
    json_path = folder / "best_round.json"

    if not json_path.is_file():
        matches = sorted(p for p in folder.rglob("best_round.json") if p.is_file())

        if len(matches) == 1:
            json_path = matches[0]
        elif len(matches) == 0:
            return None
        else:
            raise FileNotFoundError(
                f"Multiple best_round.json files found under {folder}. "
                f"Please provide the exact result folder."
            )

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)

    if isinstance(payload, dict) and "best_round" in payload:
        return int(payload["best_round"])

    return None


def _load_best_model_info(folder, *, kind="BEST", loaded_from=None):
    """
    Load the saved *_BEST_round*.json XGB model.

    Used for:
        - central XGB model
        - independently trained local XGB models

    best_round/model_round are read from best_round.json when available.
    """
    model_path = _find_xgb_best_model(folder)
    best_round = read_best_round_if_available(folder)

    return {
        "booster": _load_xgb_model(model_path),
        "model_path": model_path,
        "model_round": best_round,
        "best_round": best_round,
        "model_kind": kind,
        "loaded_from": loaded_from,
        "loaded_best_model": True,
    }


# ==================================================
# Mapping helper
# ==================================================

def _get_mapping_value(mapping, site_id, mapping_name):
    for key in (site_id, str(site_id)):
        if key in mapping:
            return mapping[key]

    try:
        int_key = int(site_id)
        if int_key in mapping:
            return mapping[int_key]
    except (TypeError, ValueError):
        pass

    raise KeyError(f"No entry for site {site_id} in {mapping_name}.")


# ==================================================
# Public loaders
# ==================================================

def load_central_xgb(central_results_dir):
    """
    Central XGB model.

    Load the saved *_BEST_round*.json model, for example:
        site-aa68-01_xgb_hist_agg_BEST_round.json

    If best_round.json exists, its best_round value is stored in:
        model_round
        best_round
    """
    return _load_best_model_info(
        central_results_dir,
        kind="BEST",
        loaded_from="central_results_dir",
    )


def load_local_xgb_by_site(local_results_dirs, site_ids=None):
    """
    Independently trained local XGB models.

    For each site, load the saved *_BEST_round*.json model, for example:
        site-aa68-01_xgb_hist_agg_BEST_round.json
    """
    if local_results_dirs is None:
        raise ValueError("local_results_dirs is required.")

    if site_ids is None:
        site_ids = sorted(
            local_results_dirs.keys(),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )

    local_models = {}

    for site_id in site_ids:
        site_dir = _get_mapping_value(
            local_results_dirs,
            site_id,
            "local_results_dirs",
        )

        local_models[f"site_{site_id}"] = _load_best_model_info(
            site_dir,
            kind="BEST",
            loaded_from="local_results_dirs",
        )

    return local_models


def load_fl_xgb(server_results_dir, best_round):
    """
    Primary FL analysis.

    Load GLOBAL model at the common selected best round from the FL server
    result folder.

    Expected example:
        something_GLOBAL_round_57.json
    """
    return _load_model_info(
        server_results_dir,
        kind="GLOBAL",
        round_num=best_round,
        loaded_from="server_results_dir",
    )


def load_fl_local_xgb_by_site_best_rounds(
    fl_site_results_dirs,
    site_ids=None,
):
    """
    Secondary FL analysis.

    For each site:
        1. read that site's best_round.json
        2. load that site's FL LOCAL model at that round

    Expected example:
        something_LOCAL_round_57.json

    This is separate from *_BEST_round*.json because FL secondary analysis
    evaluates each site's communication-round LOCAL snapshot.
    """
    if fl_site_results_dirs is None:
        raise ValueError("fl_site_results_dirs is required.")

    if site_ids is None:
        site_ids = sorted(
            fl_site_results_dirs.keys(),
            key=lambda x: int(x) if str(x).isdigit() else str(x),
        )

    fl_models = {}

    for site_id in site_ids:
        site_dir = _get_mapping_value(
            fl_site_results_dirs,
            site_id,
            "fl_site_results_dirs",
        )

        site_best_round = read_best_round(site_dir)

        fl_models[f"site_{site_id}"] = _load_model_info(
            site_dir,
            kind="LOCAL",
            round_num=site_best_round,
            loaded_from="fl_site_results_dirs",
        )

    return fl_models