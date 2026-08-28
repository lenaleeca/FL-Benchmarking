from pathlib import Path
import json
import re

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _as_results_dir(path):
    path = Path(path)
    return path / "results" if (path / "results").is_dir() else path

def read_best_epoch(results_dir):
    folder = _as_results_dir(results_dir)
    json_path = folder / "best_epoch.json"

    if not json_path.is_file():
        raise FileNotFoundError(f"No best_epoch.json found in: {folder}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)

    if isinstance(payload, dict):
        if "best_epoch" in payload:
            return int(payload["best_epoch"])
        if "epoch" in payload:
            return int(payload["epoch"])

    raise ValueError(
        f"Invalid best_epoch.json format in {json_path}. "
        "Expected integer or dict with key 'best_epoch' or 'epoch'."
    )


def _find_mlp_epoch_model(folder, best_epoch):
    folder = _as_results_dir(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    best_epoch = int(best_epoch)
    matches = []

    for ext in ("pt", "pth"):
        matches.extend(
            p for p in folder.glob(f"*epoch_{best_epoch}.{ext}")
            if p.is_file()
        )

    if not matches:
        raise FileNotFoundError(
            f"No model matching '*epoch_{best_epoch}.pt/.pth' found in: {folder}"
        )

    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple epoch_{best_epoch} models found in {folder}. "
            "Please keep only one matching checkpoint."
        )

    return matches[0]

def _torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_state_dict_from_checkpoint(path, device="cpu"):
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
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        cleaned[new_k] = v

    return cleaned


def infer_mlp_dims_from_state_dict(path, device="cpu"):
    state_dict = get_state_dict_from_checkpoint(path, device=device)
    required_keys = ["net.0.weight", "net.6.weight"]
    missing = [k for k in required_keys if k not in state_dict]

    if missing:
        raise KeyError(
            f"Could not infer MLP dimensions from {path}. Missing keys: {missing}. "
            f"Available keys include: {list(state_dict.keys())[:20]}"
        )

    w0 = state_dict["net.0.weight"]
    w_last = state_dict["net.6.weight"]

    hidden = int(w0.shape[0])
    in_dim = int(w0.shape[1])
    out_dim = int(w_last.shape[0])

    return in_dim, hidden, out_dim, state_dict


def _load_mlp_model(model_path, *, model_name=None, dropout=0.0, device="cpu"):
    model_path = Path(model_path)
    in_dim, hidden, out_dim, state_dict = infer_mlp_dims_from_state_dict(model_path, device=device)

    # Dropout value does not affect inference because model.eval() disables dropout.
    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if model_name is None:
        model_name = model_path.stem

    return {
        "model_family": "mlp",
        "name": model_name,
        "model": model,
        "state_dict": state_dict,
        "model_path": model_path,
        "path": model_path,
        "in_dim": in_dim,
        "hidden": hidden,
        "out_dim": out_dim,
        "device": device,
    }


def _find_mlp_model(folder, *, kind, round_num=None):
    folder = _as_results_dir(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {folder}")

    models = []
    for ext in ("pt", "pth"):
        for p in folder.rglob(f"*_{kind}_round_*.{ext}"):
            name = p.name.lower()
            if any(skip in name for skip in ["metrics", "report", "best_round"]):
                continue
            m = re.search(rf"_{kind}_round_(\d+)\.(?:pt|pth)$", p.name)
            if m:
                models.append((int(m.group(1)), p))

    if not models:
        raise FileNotFoundError(f"No *_{kind}_round_*.pt/.pth models found in: {folder}")

    if round_num is None:
        model_round, model_path = max(models, key=lambda x: x[0])
    else:
        round_num = int(round_num)
        matches = [(r, p) for r, p in models if r == round_num]
        if not matches:
            raise FileNotFoundError(f"No *_{kind}_round_{round_num}.pt/.pth model found in: {folder}")
        if len(matches) > 1:
            print(f"Warning: multiple {kind} round {round_num} models found, using first:")
            for _, p in matches:
                print(f" - {p}")
        model_round, model_path = sorted(matches, key=lambda x: str(x[1]))[0]

    return model_round, model_path


def _load_model_info(folder, *, kind, round_num=None, loaded_from=None, dropout=0.0, device="cpu"):
    model_round, model_path = _find_mlp_model(folder, kind=kind, round_num=round_num)
    model_info = _load_mlp_model(
        model_path,
        model_name=f"{loaded_from}_{kind}_round_{model_round}",
        dropout=dropout,
        device=device,
    )
    model_info.update({
        "model_round": model_round,
        "best_round": int(round_num) if round_num is not None else None,
        "model_kind": kind,
        "loaded_from": loaded_from,
        "dropout": dropout,
    })
    return model_info


def read_best_round(site_dir):
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
                f"Multiple best_round.json files found under {folder}. Please provide the exact site result folder."
            )

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, int):
        return int(payload)
    if isinstance(payload, dict) and "best_round" in payload:
        return int(payload["best_round"])

    raise ValueError(
        f"Invalid best_round.json format in {json_path}. Expected integer or dict with key 'best_round'."
    )


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

def _load_best_epoch_model_info(folder, *, kind, loaded_from=None, dropout=0.0, device="cpu"):
    best_epoch = read_best_epoch(folder)
    model_path = _find_mlp_epoch_model(folder, best_epoch)

    model_info = _load_mlp_model(
        model_path,
        model_name=f"{loaded_from}_{kind}_epoch_{best_epoch}",
        dropout=dropout,
        device=device,
    )

    model_info.update({
        "model_round": None,
        "best_round": None,
        "best_epoch": int(best_epoch),
        "model_kind": kind,
        "loaded_from": loaded_from,
        "dropout": dropout,
    })

    return model_info

def load_central_mlp(central_results_dir, *, dropout=0.0, device="cpu"):
    return _load_best_epoch_model_info(
        central_results_dir,
        kind="LOCAL",
        loaded_from="central_results_dir",
        dropout=dropout,
        device=device,
    )


def load_local_mlp_by_site(local_results_dirs, site_ids=None, *, dropout=0.0, device="cpu"):
    if local_results_dirs is None:
        raise ValueError("local_results_dirs is required.")
    if site_ids is None:
        site_ids = sorted(local_results_dirs.keys(), key=lambda x: int(x))

    local_models = {}

    for site_id in site_ids:
        site_dir = _get_mapping_value(local_results_dirs, site_id, "local_results_dirs")

        local_models[f"site_{site_id}"] = _load_best_epoch_model_info(
            site_dir,
            kind="LOCAL",
            loaded_from="local_results_dirs",
            dropout=dropout,
            device=device,
        )

    return local_models


def load_fl_mlp(server_results_dir, best_round, *, dropout=0.0, device="cpu"):
    return _load_model_info(
        server_results_dir,
        kind="GLOBAL",
        round_num=best_round,
        loaded_from="server_results_dir",
        dropout=dropout,
        device=device,
    )


def load_fl_local_mlp_by_site_best_rounds(fl_site_results_dirs, site_ids=None, *, dropout=0.0, device="cpu"):
    if fl_site_results_dirs is None:
        raise ValueError("fl_site_results_dirs is required.")
    if site_ids is None:
        site_ids = sorted(fl_site_results_dirs.keys(), key=lambda x: int(x))

    fl_models = {}
    for site_id in site_ids:
        site_dir = _get_mapping_value(fl_site_results_dirs, site_id, "fl_site_results_dirs")
        site_best_round = read_best_round(site_dir)
        fl_models[f"site_{site_id}"] = _load_model_info(
            site_dir,
            kind="LOCAL",
            round_num=site_best_round,
            loaded_from="fl_site_results_dirs",
            dropout=dropout,
            device=device,
        )
    return fl_models
