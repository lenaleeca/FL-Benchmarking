from pathlib import Path
import json
import re

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def _as_results_dir(path):
    path = Path(path)
    return path / "results" if (path / "results").is_dir() else path


def _torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _state_dict_from_checkpoint(path, device="cpu"):
    checkpoint = _torch_load(path, device=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint did not contain a state_dict-like dictionary: {path}")

    cleaned = {}
    for k, v in state_dict.items():
        new_k = str(k)
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        cleaned[new_k] = v
    return cleaned


def _infer_lr_dims_from_state_dict(path, device="cpu"):
    state_dict = _state_dict_from_checkpoint(path, device=device)
    if "linear.weight" not in state_dict:
        raise KeyError(
            f"Could not infer LR dimensions from {path}. Missing key 'linear.weight'. "
            f"Available keys include: {list(state_dict.keys())[:20]}"
        )
    w = state_dict["linear.weight"]
    out_dim = int(w.shape[0])
    in_dim = int(w.shape[1])
    return in_dim, out_dim, state_dict


def _load_lr_model(model_path, *, model_name=None, device="cpu"):
    model_path = Path(model_path)
    in_dim, out_dim, state_dict = _infer_lr_dims_from_state_dict(model_path, device=device)
    model = LogisticRegression(in_dim=in_dim, out_dim=out_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return {
        "model_family": "lr",
        "name": model_name or model_path.stem,
        "model": model,
        "state_dict": state_dict,
        "model_path": model_path,
        "path": model_path,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "device": device,
    }


def read_best_iteration(results_dir):
    folder = _as_results_dir(results_dir)
    json_path = folder / "best_iteration"
    if not json_path.is_file():
        json_path = folder / "best_iteration.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"No best_iteration or best_iteration.json found in: {folder}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)
    if isinstance(payload, str) and payload.strip().isdigit():
        return int(payload.strip())
    if isinstance(payload, dict):
        for key in ("best_iteration", "iteration", "best_iter", "selected_iteration"):
            if key in payload and payload[key] is not None:
                return int(payload[key])

    raise ValueError(
        f"Invalid best_iteration format in {json_path}. "
        "Expected integer or dict with key 'best_iteration' or 'iteration'."
    )


def read_best_round(site_dir):
    folder = _as_results_dir(site_dir)
    json_path = folder / "best_round.json"
    if not json_path.is_file():
        json_path = folder / "best_round"
    if not json_path.is_file():
        raise FileNotFoundError(f"No best_round.json or best_round found in: {folder}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)
    if isinstance(payload, dict) and "best_round" in payload:
        return int(payload["best_round"])
    raise ValueError(f"Invalid best_round file format in {json_path}.")


def _parse_round_from_name(path):
    m = re.search(r"(?:^|[_-])round[_-](\d+)(?:[_\.-]|$)", Path(path).name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_iteration_from_name(path):
    m = re.search(r"(?:^|[_-])iteration[_-](\d+)(?:[_\.-]|$)", Path(path).name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_lr_iteration_model(folder, best_iteration):
    folder = _as_results_dir(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    best_iteration = int(best_iteration)
    matches = []
    for ext in ("pt", "pth"):
        matches.extend(p for p in folder.glob(f"*iteration_{best_iteration}.{ext}") if p.is_file())

    if not matches:
        raise FileNotFoundError(
            f"No LR checkpoint matching '*iteration_{best_iteration}.pt/.pth' found in: {folder}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple iteration_{best_iteration} checkpoints found in {folder}. "
            "Please keep only one matching checkpoint."
        )
    return matches[0]


def _find_lr_round_model(folder, *, kind, round_num=None):
    folder = _as_results_dir(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    kind_text = str(kind).lower()
    candidates = []
    for ext in ("pt", "pth"):
        for p in folder.rglob(f"*.{ext}"):
            name = p.name.lower()
            if kind_text not in name or any(skip in name for skip in ("metrics", "report")):
                continue
            r = _parse_round_from_name(p)
            if r is None:
                continue
            if round_num is not None and r != int(round_num):
                continue
            it = _parse_iteration_from_name(p)
            score = 1000 + r
            if f"_{kind_text}_" in name:
                score += 100
            if it is None:
                score += 50
            else:
                score += min(it, 100000) / 1000000
            candidates.append((score, r, it if it is not None else -1, str(p), p))

    if not candidates:
        msg_round = f" round_{round_num}" if round_num is not None else ""
        raise FileNotFoundError(f"No LR {kind}{msg_round} checkpoint found under: {folder}")

    candidates.sort(key=lambda x: (-x[0], x[3]))
    score, model_round, iteration, _, path = candidates[0]
    return model_round, path


def _load_round_model_info(folder, *, kind, round_num=None, loaded_from=None, device="cpu"):
    model_round, model_path = _find_lr_round_model(folder, kind=kind, round_num=round_num)
    model_info = _load_lr_model(
        model_path,
        model_name=f"{loaded_from}_{kind}_round_{model_round}",
        device=device,
    )
    model_info.update({
        "model_round": model_round,
        "best_round": int(round_num) if round_num is not None else None,
        "best_iteration": _parse_iteration_from_name(model_path),
        "model_kind": kind,
        "loaded_from": loaded_from,
        "loaded_best_model": round_num is not None,
    })
    return model_info


def _load_best_iteration_model_info(folder, *, kind, loaded_from=None, device="cpu"):
    best_iteration = read_best_iteration(folder)
    model_path = _find_lr_iteration_model(folder, best_iteration)
    model_info = _load_lr_model(
        model_path,
        model_name=f"{loaded_from}_{kind}_iteration_{best_iteration}",
        device=device,
    )
    model_info.update({
        "model_round": _parse_round_from_name(model_path),
        "best_round": None,
        "best_iteration": int(best_iteration),
        "model_kind": kind,
        "loaded_from": loaded_from,
        "loaded_best_model": True,
    })
    return model_info


def _get_mapping_value(mapping, site_id, mapping_name):
    for key in (site_id, str(site_id)):
        if key in mapping:
            return mapping[key]
    text = str(site_id)
    if text.isdigit():
        int_key = int(text)
        if int_key in mapping:
            return mapping[int_key]
    raise KeyError(f"No entry for site {site_id} in {mapping_name}.")


def load_central_lr(central_results_dir, *, device="cpu", **kwargs):
    return _load_best_iteration_model_info(
        central_results_dir,
        kind="LOCAL",
        loaded_from="central_results_dir",
        device=device,
    )


def load_local_lr_by_site(local_results_dirs, site_ids=None, *, device="cpu", **kwargs):
    if local_results_dirs is None:
        raise ValueError("local_results_dirs is required.")
    if site_ids is None:
        site_ids = sorted(local_results_dirs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    local_models = {}
    for site_id in site_ids:
        site_dir = _get_mapping_value(local_results_dirs, site_id, "local_results_dirs")
        local_models[f"site_{site_id}"] = _load_best_iteration_model_info(
            site_dir,
            kind="LOCAL",
            loaded_from="local_results_dirs",
            device=device,
        )
    return local_models


def load_fl_lr(server_results_dir, best_round, *, device="cpu", **kwargs):
    return _load_round_model_info(
        server_results_dir,
        kind="GLOBAL",
        round_num=best_round,
        loaded_from="server_results_dir",
        device=device,
    )


def load_fl_local_lr_by_site_best_rounds(fl_site_results_dirs, site_ids=None, *, device="cpu", **kwargs):
    if fl_site_results_dirs is None:
        raise ValueError("fl_site_results_dirs is required.")
    if site_ids is None:
        site_ids = sorted(fl_site_results_dirs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    fl_models = {}
    for site_id in site_ids:
        site_dir = _get_mapping_value(fl_site_results_dirs, site_id, "fl_site_results_dirs")
        site_best_round = read_best_round(site_dir)
        fl_models[f"site_{site_id}"] = _load_round_model_info(
            site_dir,
            kind="LOCAL",
            round_num=site_best_round,
            loaded_from="fl_site_results_dirs",
            device=device,
        )
    return fl_models
