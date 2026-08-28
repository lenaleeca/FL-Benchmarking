"""Timing and data-transfer summary helpers for FL post-hoc reports.

Branching rule
--------------
- algorithm_name == "xgb_hist_agg" uses XGB Hist Agg extraction.
- any other algorithm_name uses non-XGB-Hist-Agg extraction.

Outputs
-------
- <algorithm_name>_timing_data_summary_long.csv for audit/debugging
- <algorithm_name>_fl_training_report_table.csv for the manuscript/report table
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


XGB_HIST_AGG_ALGORITHM_NAME = "xgb_hist_agg"

# Non-XGB Hist Agg server files
SERVER_TRANSFER_FILES = (
    "all_sites_transfer_metrics.csv",
    "all_sites_tree_xgb_transfer_metrics.csv",
)

SERVER_AGGREGATION_FILE = ("server_aggregation_metrics.csv", "server_aggregation_timing_metrics.csv")
SERVER_ROUND_TIMING_FILE = "server_round_timing_metrics.csv"

# XGB Hist Agg server files
XGB_PORT_TRANSFER_FILE = "all_sites_xgb_port_transfer_metrics.csv"
XGB_SERVER_TIMING_FILE = "all_sites_xgb_timing_metrics.csv"

# Fallback generic timing filename used by some runs/site folders
SERVER_TIMING_FILE = "timing_metrics.csv"

TRANSFER_VALUE_COL = "bidirectional_shareable_mb"
TRANSFER_DIRECTION_VALUE_COLS = {
    "Bidirectional": "bidirectional_shareable_mb",
}
SERVER_TIME_VALUE_COL = "value_sec"
XGB_PORT_TRANSFER_VALUE_COL = "mb_total"

ROUND_COL = "round"
METRIC_COL = "metric"
SCOPE_COL = "scope"
ALL_ROUNDS_VALUE = "all_rounds"

SERVER_AGGREGATION_TOTAL_METRIC_VALUE = "aggregation_total_time_sec"

# Non-XGB client timing metric names
CLIENT_LOCAL_TRAINING_METRIC_VALUE = "local_train_total_time_sec"
CLIENT_TOTAL_ROUND_METRIC_VALUE = "client_round_total_elapsed_sec"

# XGB Hist Agg client timing metric names
XGB_CLIENT_LOCAL_TRAINING_METRIC_VALUE = "federated_xgb_train_time_sec"
XGB_CLIENT_TOTAL_TIME_METRIC_VALUE = "client_elapsed_time_sec"


def _normalise_algorithm_name(algorithm_name):
    return str(algorithm_name or "").strip().lower()


def is_xgb_histagg_algorithm(algorithm_name):
    return _normalise_algorithm_name(algorithm_name) == XGB_HIST_AGG_ALGORITHM_NAME


def _as_results_dir(path):
    path = Path(path)
    return path / "results" if (path / "results").is_dir() else path


def _site_sort_key(site_id):
    text = str(site_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


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


def _safe_read_csv(path, *, strict=False):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if strict:
            raise
        return pd.DataFrame({"_read_error": [str(exc)]})


def _find_exact_file(root, file_name, *, label, strict=False):
    root = _as_results_dir(root)
    if not root.is_dir():
        msg = f"Results folder not found for {label}: {root}"
        if strict:
            raise FileNotFoundError(msg)
        return None, msg

    matches = sorted(p for p in root.rglob(file_name) if p.is_file() and p.name == file_name)
    if not matches:
        msg = f"Missing exact {label} file '{file_name}' under {root}"
        if strict:
            raise FileNotFoundError(msg)
        return None, msg
    if len(matches) > 1:
        msg = f"Multiple exact {label} files named '{file_name}' found; used {matches[-1]}"
        if strict:
            found = "\n".join(f" - {p}" for p in matches)
            raise FileNotFoundError(f"Multiple exact {label} files named '{file_name}' under {root}:\n{found}")
        return matches[-1], msg
    return matches[0], ""


def _find_first_existing_file(root, file_names, *, label, strict=False):
    notes = []
    for file_name in file_names:
        path, note = _find_exact_file(root, file_name, label=label, strict=False)
        if path is not None:
            return path, "; ".join(x for x in [note] if x)
        notes.append(note)
    msg = "; ".join(notes) if notes else f"No {label} file found."
    if strict:
        raise FileNotFoundError(msg)
    return None, msg


def _exact_filter(df, *, column, value):
    if value is None:
        return df
    if column not in df.columns:
        return pd.DataFrame()
    mask = df[column].astype(str).str.strip().str.lower().eq(str(value).strip().lower())
    return df[mask].copy()


def _last_numeric_value(
    df,
    *,
    value_col,
    strict=False,
    round_value=None,
    metric_value=None,
    scope_value=None,
) -> Tuple[Optional[float], str]:
    if df is None or df.empty:
        return None, "No rows available."
    if "_read_error" in df.columns:
        msg = f"Could not read CSV: {df['_read_error'].iloc[0]}"
        if strict:
            raise ValueError(msg)
        return None, msg
    if value_col not in df.columns:
        msg = f"Column '{value_col}' not found."
        if strict:
            raise ValueError(msg)
        return None, msg

    sub = df.copy()
    notes = []

    if scope_value is not None:
        sub = _exact_filter(sub, column=SCOPE_COL, value=scope_value)
        if sub.empty:
            msg = f"No rows found where {SCOPE_COL} == '{scope_value}'."
            if strict:
                raise ValueError(msg)
            return None, msg
        notes.append(f"used last row where {SCOPE_COL} == '{scope_value}'")

    if round_value is not None:
        sub = _exact_filter(sub, column=ROUND_COL, value=round_value)
        if sub.empty:
            msg = f"No rows found where {ROUND_COL} == '{round_value}'."
            if strict:
                raise ValueError(msg)
            return None, msg
        notes.append(f"used last row where {ROUND_COL} == '{round_value}'")

    if metric_value is not None:
        sub = _exact_filter(sub, column=METRIC_COL, value=metric_value)
        if sub.empty:
            msg = f"No rows found where {METRIC_COL} == '{metric_value}'."
            if strict:
                raise ValueError(msg)
            return None, msg
        notes.append(f"used last row where {METRIC_COL} == '{metric_value}'")

    vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
    if vals.empty:
        msg = f"No numeric values found in column '{value_col}' after exact filtering."
        if strict:
            raise ValueError(msg)
        return None, msg
    return float(vals.iloc[-1]), "; ".join(notes)


def _infer_numeric_round_count(df):
    """Infer number of communication rounds, excluding summary rows like round=-1."""
    if df is None or df.empty or ROUND_COL not in df.columns:
        return None
    rounds = pd.to_numeric(df[ROUND_COL], errors="coerce").dropna()
    rounds = rounds[rounds >= 0]
    if rounds.empty:
        return None
    return int(rounds.nunique())


def _make_row(
    *,
    algorithm_name,
    scope,
    site_id,
    metric_name,
    value,
    unit,
    source_file,
    source_column,
    description,
    note="",
):
    return {
        "algorithm_name": algorithm_name,
        "scope": scope,
        "site_id": site_id,
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "source_file": str(source_file) if source_file is not None else "",
        "source_column": source_column,
        "description": description,
        "note": note,
    }



def _calc_mean_sd(values):
    """Return raw mean and sample SD for numeric values.

    The raw values are kept for audit output. Display formatting is handled
    separately so that very small non-zero SDs are not shown as 0.000 in the
    Word/report table.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vals.empty:
        return None, None, vals
    mean_val = float(vals.mean())
    sd_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return mean_val, sd_val, vals


def _format_small_nonzero_sd(sd_val, *, digits=3):
    """Format SD for manuscript table display.

    If the raw SD is non-zero but would round to 0.000, report it as
    <0.001 instead of 0.000. This keeps the table consistent with a trend
    arrow when tiny but real changes occur across rounds.
    """
    if sd_val is None or pd.isna(sd_val):
        return ""
    sd_val = float(sd_val)
    if sd_val == 0.0:
        return f"{0.0:.{digits}f}"
    smallest_display_unit = 10 ** (-digits)
    if abs(sd_val) < smallest_display_unit:
        return f"<{smallest_display_unit:.{digits}f}"
    return f"{sd_val:.{digits}f}"


def _format_mean_sd(values, *, digits=3):
    mean_val, sd_val, vals = _calc_mean_sd(values)
    if vals.empty:
        return ""
    sd_text = _format_small_nonzero_sd(sd_val, digits=digits)
    return f"{mean_val:.{digits}f} ({sd_text})"


def _format_trend_arrow(round_nums, vals, *, atol=1e-9):
    """Return an arrow for overall per-round transfer trend.

    Uses a simple least-squares slope of transfer volume versus round number.
    Returns an empty string when values are effectively unchanged.
    """
    x = pd.to_numeric(round_nums, errors="coerce")
    y = pd.to_numeric(vals, errors="coerce")
    trend_df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(trend_df) < 2:
        return ""

    y_range = float(trend_df["y"].max() - trend_df["y"].min())
    if y_range <= atol:
        return ""

    x_centered = trend_df["x"] - trend_df["x"].mean()
    denom = float((x_centered ** 2).sum())
    if denom <= 0:
        return ""

    slope = float(((x_centered) * (trend_df["y"] - trend_df["y"].mean())).sum() / denom)
    # Scale tolerance to the observed magnitude so tiny floating-point noise does not create arrows.
    scale = max(abs(float(trend_df["y"].mean())), y_range, 1.0)
    slope_tol = atol * scale
    if slope > slope_tol:
        return " ↑"
    if slope < -slope_tol:
        return " ↓"
    return ""


def _read_numeric_transfer_round_rows(path, *, strict=False):
    """Read numeric per-round transfer rows from mixed transfer CSVs.

    The non-XGB transfer files may contain repeated embedded header rows and
    cumulative all_rounds rows. This function keeps only true numeric round rows
    from the main per-round table and converts transfer columns to numeric.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        msg = f"Could not read transfer CSV: {exc}"
        if strict:
            raise ValueError(msg)
        return pd.DataFrame(), msg

    if ROUND_COL not in df.columns:
        msg = f"Column '{ROUND_COL}' not found in transfer CSV."
        if strict:
            raise ValueError(msg)
        return pd.DataFrame(), msg

    out = df.copy()
    out["_round_num"] = pd.to_numeric(out[ROUND_COL], errors="coerce")
    out = out[out["_round_num"].notna() & (out["_round_num"] >= 0)].copy()
    if out.empty:
        msg = "No numeric per-round transfer rows found."
        if strict:
            raise ValueError(msg)
        return out, msg

    for col in TRANSFER_DIRECTION_VALUE_COLS.values():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["_round_num"] = out["_round_num"].astype(int)
    return out, f"used {out['_round_num'].nunique()} numeric per-round transfer rows"


def _summarise_transfer_per_round_by_direction(path, *, strict=False):
    rows, note = _read_numeric_transfer_round_rows(path, strict=strict)
    if rows.empty:
        return None, None, note, {}

    parts = []
    missing = []
    total_from_rounds = None
    raw_stats = {}
    for label, col in TRANSFER_DIRECTION_VALUE_COLS.items():
        if col not in rows.columns:
            missing.append(col)
            continue
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        if vals.empty:
            missing.append(col)
            continue

        mean_val, sd_val, clean_vals = _calc_mean_sd(vals)
        key_prefix = label.strip().lower().replace(" ", "_")
        raw_stats[f"{key_prefix}_mean_raw"] = mean_val
        raw_stats[f"{key_prefix}_sd_raw"] = sd_val
        raw_stats[f"{key_prefix}_n_rounds"] = int(clean_vals.shape[0])

        arrow = ""
        if label == "Bidirectional":
            arrow = _format_trend_arrow(rows["_round_num"], vals)
        parts.append(f"{_format_mean_sd(vals, digits=3)}{arrow}")
        if label == "Bidirectional":
            total_from_rounds = float(vals.sum())

    summary = "; ".join(parts) if parts else None
    if missing:
        note = "; ".join(x for x in [note, f"missing/non-numeric direction columns: {', '.join(missing)}"] if x)
    return summary, total_from_rounds, note, raw_stats


def _infer_server_round_count_from_timing(df):
    if df is None or df.empty or ROUND_COL not in df.columns:
        return None
    sub = df.copy()
    if METRIC_COL in sub.columns:
        metric_mask = sub[METRIC_COL].astype(str).str.strip().str.lower().eq("server_round_elapsed_sec")
        if metric_mask.any():
            sub = sub[metric_mask].copy()
    rounds = pd.to_numeric(sub[ROUND_COL], errors="coerce").dropna()
    rounds = rounds[rounds >= 0]
    if rounds.empty:
        return None
    return int(rounds.nunique())

def _read_last_transfer_summary_block_value(
    path,
    *,
    value_col=TRANSFER_VALUE_COL,
    strict=False,
) -> Tuple[Optional[float], str]:
    """Read final all_rounds value from all_sites_transfer_metrics.csv.

    This handles non-XGB transfer files that may contain repeated summary blocks
    with their own headers.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    except Exception as exc:
        msg = f"Could not read transfer CSV as text: {exc}"
        if strict:
            raise ValueError(msg)
        return None, msg

    header_indices = []
    for i, line in enumerate(raw_lines):
        parts = [x.strip() for x in line.split(",")]
        if parts and parts[0].lower() == ROUND_COL and value_col in parts:
            header_indices.append(i)
    if not header_indices:
        msg = f"No transfer header found with first column '{ROUND_COL}' and value column '{value_col}'."
        if strict:
            raise ValueError(msg)
        return None, msg

    last_error = ""
    for header_idx in reversed(header_indices):
        header = [x.strip() for x in raw_lines[header_idx].split(",")]
        if value_col not in header:
            continue
        value_pos = header.index(value_col)
        next_header_idx = len(raw_lines)
        for j in range(header_idx + 1, len(raw_lines)):
            first_cell = raw_lines[j].split(",", 1)[0].strip().lower()
            if first_cell == ROUND_COL:
                next_header_idx = j
                break
        candidate_lines = raw_lines[header_idx + 1:next_header_idx]
        all_rounds_lines = [
            line for line in candidate_lines
            if line.split(",", 1)[0].strip().lower() == ALL_ROUNDS_VALUE
        ]
        if not all_rounds_lines:
            last_error = f"No '{ALL_ROUNDS_VALUE}' row found after transfer header at line {header_idx + 1}."
            continue
        data = [x.strip() for x in all_rounds_lines[-1].split(",")]
        if value_pos >= len(data):
            last_error = f"Column '{value_col}' position {value_pos} is outside the all_rounds row length {len(data)}."
            continue
        raw_value = data[value_pos]
        try:
            return float(raw_value), (
                f"used final repeated transfer block where {ROUND_COL} == "
                f"'{ALL_ROUNDS_VALUE}' and column == '{value_col}'"
            )
        except Exception:
            last_error = f"Could not convert '{raw_value}' to float for column '{value_col}'."
            continue

    msg = last_error or f"Could not extract '{value_col}' from final transfer summary block."
    if strict:
        raise ValueError(msg)
    return None, msg


def extract_non_xgb_histagg_server_summary(server_results_dir, *, algorithm_name, strict=False):
    rows = []

    transfer_path, transfer_file_note = _find_first_existing_file(
        server_results_dir,
        SERVER_TRANSFER_FILES,
        label="all-sites transfer metrics",
        strict=strict,
    )
    transfer_value = None
    transfer_value_note = ""
    per_round_transfer_summary = None
    per_round_transfer_total = None
    per_round_transfer_note = ""
    per_round_transfer_raw_stats = {}
    if transfer_path is not None:
        transfer_value, transfer_value_note = _read_last_transfer_summary_block_value(
            transfer_path,
            value_col=TRANSFER_VALUE_COL,
            strict=strict,
        )
        per_round_transfer_summary, per_round_transfer_total, per_round_transfer_note, per_round_transfer_raw_stats = _summarise_transfer_per_round_by_direction(
            transfer_path,
            strict=strict,
        )
        if transfer_value is None and per_round_transfer_total is not None:
            transfer_value = per_round_transfer_total
            transfer_value_note = "final all_rounds value unavailable; used sum of numeric per-round bidirectional shareable MiB"
        elif transfer_value is not None and per_round_transfer_total is not None:
            diff = abs(float(transfer_value) - float(per_round_transfer_total))
            if diff > 1e-6:
                transfer_value_note = "; ".join(x for x in [
                    transfer_value_note,
                    f"check: final all_rounds differs from numeric-round sum by {diff:.6f} MiB",
                ] if x)
            else:
                transfer_value_note = "; ".join(x for x in [
                    transfer_value_note,
                    "verified against numeric-round sum",
                ] if x)
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_mb",
        value=transfer_value,
        unit="MiB",
        source_file=transfer_path,
        source_column=TRANSFER_VALUE_COL,
        description="Total bidirectional shareable MiB from the final all_rounds transfer summary block, verified against numeric per-round rows when possible.",
        note="; ".join(x for x in [transfer_file_note, transfer_value_note] if x),
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_mib_mean_sd_direction",
        value=per_round_transfer_summary,
        unit="MiB per round",
        source_file=transfer_path,
        source_column=TRANSFER_VALUE_COL,
        description="Mean (SD) of per-round bidirectional shareable MiB, calculated from numeric round rows only. Very small non-zero SDs are displayed as <0.001 in the report table.",
        note="; ".join(x for x in [transfer_file_note, per_round_transfer_note] if x),
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_mib_mean_raw",
        value=per_round_transfer_raw_stats.get("bidirectional_mean_raw"),
        unit="MiB per round",
        source_file=transfer_path,
        source_column=TRANSFER_VALUE_COL,
        description="Raw unrounded mean of numeric-round bidirectional_shareable_mb values, saved for audit/debugging.",
        note="not displayed in the Word table",
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_mib_sd_raw",
        value=per_round_transfer_raw_stats.get("bidirectional_sd_raw"),
        unit="MiB per round",
        source_file=transfer_path,
        source_column=TRANSFER_VALUE_COL,
        description="Raw unrounded sample SD of numeric-round bidirectional_shareable_mb values, saved for audit/debugging.",
        note="not displayed in the Word table; report table shows <0.001 when this raw SD is non-zero but rounds to 0.000",
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_n_rounds",
        value=per_round_transfer_raw_stats.get("bidirectional_n_rounds"),
        unit="count",
        source_file=transfer_path,
        source_column=ROUND_COL,
        description="Number of numeric transfer round rows used for per-round bidirectional mean and SD.",
        note="not displayed in the Word table",
    ))

    aggregation_path, aggregation_file_note = _find_first_existing_file(
        server_results_dir,
        SERVER_AGGREGATION_FILE,
        label="server aggregation metrics",
        strict=strict,
    )
    aggregation_value = None
    aggregation_value_note = ""
    if aggregation_path is not None:
        df = _safe_read_csv(aggregation_path, strict=strict)
        aggregation_value, aggregation_value_note = _last_numeric_value(
            df,
            value_col=SERVER_TIME_VALUE_COL,
            strict=strict,
            round_value=ALL_ROUNDS_VALUE,
            metric_value=SERVER_AGGREGATION_TOTAL_METRIC_VALUE,
        )
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="cumulative_server_aggregation_time_sec",
        value=aggregation_value,
        unit="sec",
        source_file=aggregation_path,
        source_column=SERVER_TIME_VALUE_COL,
        description="Cumulative server aggregation time from the final all_rounds aggregation_total_time_sec row.",
        note="; ".join(x for x in [aggregation_file_note, aggregation_value_note] if x),
    ))

    round_timing_path, round_timing_file_note = _find_exact_file(
        server_results_dir,
        SERVER_ROUND_TIMING_FILE,
        label="server round timing metrics",
        strict=strict,
    )
    round_timing_value = None
    round_timing_value_note = ""
    server_round_count = None
    server_round_count_note = ""
    if round_timing_path is not None:
        df = _safe_read_csv(round_timing_path, strict=strict)
        round_timing_value, round_timing_value_note = _last_numeric_value(
            df,
            value_col=SERVER_TIME_VALUE_COL,
            strict=strict,
            round_value=ALL_ROUNDS_VALUE,
        )
        server_round_count = _infer_server_round_count_from_timing(df)
        server_round_count_note = f"inferred {server_round_count} numeric server rounds" if server_round_count else "could not infer numeric server round count"
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_server_round_duration_sec",
        value=round_timing_value,
        unit="sec",
        source_file=round_timing_path,
        source_column=SERVER_TIME_VALUE_COL,
        description="Total server run duration from the final all_rounds row.",
        note="; ".join(x for x in [round_timing_file_note, round_timing_value_note] if x),
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="server_numeric_round_count",
        value=server_round_count,
        unit="count",
        source_file=round_timing_path,
        source_column=ROUND_COL,
        description="Number of numeric server rounds used to calculate mean wall-clock time per round.",
        note="; ".join(x for x in [round_timing_file_note, server_round_count_note] if x),
    ))

    return pd.DataFrame(rows)


def extract_xgb_histagg_server_summary(server_results_dir, *, algorithm_name, strict=False):
    rows = []

    port_path, port_file_note = _find_exact_file(
        server_results_dir,
        XGB_PORT_TRANSFER_FILE,
        label="XGB port transfer metrics",
        strict=strict,
    )
    port_value = None
    port_value_note = ""
    if port_path is not None:
        df = _safe_read_csv(port_path, strict=strict)
        port_value, port_value_note = _last_numeric_value(
            df,
            value_col=XGB_PORT_TRANSFER_VALUE_COL,
            strict=strict,
            scope_value=ALL_ROUNDS_VALUE if SCOPE_COL in df.columns else None,
        )
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_mb",
        value=port_value,
        unit="MiB",
        source_file=port_path,
        source_column=XGB_PORT_TRANSFER_VALUE_COL,
        description="Total network traffic on the XGBoost communication port from the all_rounds row, mb_total column.",
        note="; ".join(x for x in [port_file_note, port_value_note] if x),
    ))
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_mib_mean_sd_direction",
        value=None,
        unit="MiB per round",
        source_file=port_path,
        source_column=XGB_PORT_TRANSFER_VALUE_COL,
        description="Per-round direction-specific transfer is not available from XGB Hist Agg port totals.",
        note="not separately tracked by direction in XGB Hist Agg port metrics",
    ))

    timing_path, timing_file_note = _find_first_existing_file(
        server_results_dir,
        [XGB_SERVER_TIMING_FILE, SERVER_TIMING_FILE],
        label="XGB server timing metrics",
        strict=strict,
    )
    timing_value = None
    timing_value_note = ""
    if timing_path is not None:
        df = _safe_read_csv(timing_path, strict=strict)
        timing_value, timing_value_note = _last_numeric_value(
            df,
            value_col=SERVER_TIME_VALUE_COL,
            strict=strict,
            scope_value=ALL_ROUNDS_VALUE if SCOPE_COL in df.columns else None,
        )
    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="total_server_round_duration_sec",
        value=timing_value,
        unit="sec",
        source_file=timing_path,
        source_column=SERVER_TIME_VALUE_COL,
        description="Whole XGB Hist Agg server run elapsed time from all_sites_xgb_timing_metrics.csv value_sec.",
        note="; ".join(x for x in [timing_file_note, timing_value_note] if x),
    ))

    rows.append(_make_row(
        algorithm_name=algorithm_name,
        scope="server",
        site_id="all_sites",
        metric_name="cumulative_server_aggregation_time_sec",
        value=None,
        unit="sec",
        source_file=None,
        source_column="",
        description="Not separately tracked for XGB Hist Agg.",
        note="not separately tracked",
    ))

    return pd.DataFrame(rows)


def _find_client_timing_file(site_results_dir, *, algorithm_name, strict=False):
    site_results_dir = _as_results_dir(site_results_dir)
    if not site_results_dir.is_dir():
        msg = f"Client/site results folder not found: {site_results_dir}"
        if strict:
            raise FileNotFoundError(msg)
        return None, msg

    exact_matches = sorted(
        p for p in site_results_dir.rglob(SERVER_TIMING_FILE)
        if p.is_file() and p.name == SERVER_TIMING_FILE
    )
    if len(exact_matches) == 1:
        return exact_matches[0], ""
    if len(exact_matches) > 1:
        msg = f"Multiple client timing files named '{SERVER_TIMING_FILE}' found; used {exact_matches[-1]}"
        if strict:
            found = "\n".join(f" - {p}" for p in exact_matches)
            raise FileNotFoundError(f"Multiple client timing files named '{SERVER_TIMING_FILE}' under {site_results_dir}:\n{found}")
        return exact_matches[-1], msg

    algorithm_name = _normalise_algorithm_name(algorithm_name)
    pattern = f"*_{algorithm_name}_timing_metrics.csv"
    pattern_matches = sorted(p for p in site_results_dir.rglob(pattern) if p.is_file())
    if len(pattern_matches) == 1:
        return pattern_matches[0], f"used site-specific timing file pattern '{pattern}'"
    if len(pattern_matches) > 1:
        msg = f"Multiple client timing files matching '{pattern}' found; used {pattern_matches[-1]}"
        if strict:
            found = "\n".join(f" - {p}" for p in pattern_matches)
            raise FileNotFoundError(f"Multiple client timing files matching '{pattern}' under {site_results_dir}:\n{found}")
        return pattern_matches[-1], msg

    msg = f"Missing client timing file. Expected either '{SERVER_TIMING_FILE}' or pattern '{pattern}' under {site_results_dir}"
    if strict:
        raise FileNotFoundError(msg)
    return None, msg


def _extract_non_xgb_client_values(df, *, strict=False):
    train_value, train_note = _last_numeric_value(
        df,
        value_col=SERVER_TIME_VALUE_COL,
        strict=strict,
        round_value=ALL_ROUNDS_VALUE,
        metric_value=CLIENT_LOCAL_TRAINING_METRIC_VALUE,
    )
    total_value, total_note = _last_numeric_value(
        df,
        value_col=SERVER_TIME_VALUE_COL,
        strict=strict,
        round_value=ALL_ROUNDS_VALUE,
        metric_value=CLIENT_TOTAL_ROUND_METRIC_VALUE,
    )
    round_count = _infer_numeric_round_count(df)
    if train_value is not None and round_count and round_count > 0:
        mean_train_per_round = train_value / round_count
        mean_note = f"mean per-round local training time = {CLIENT_LOCAL_TRAINING_METRIC_VALUE} / {round_count} numeric rounds"
    else:
        mean_train_per_round = None
        mean_note = f"could not calculate mean per-round local training time because {CLIENT_LOCAL_TRAINING_METRIC_VALUE} or round count was missing"

    values = {
        "local_training_time_sec": train_value,
        "mean_per_round_local_training_time_sec": mean_train_per_round,
        "total_client_round_time_sec": total_value,
        "client_numeric_round_count": round_count,
    }
    note = "; ".join(x for x in [train_note, total_note, mean_note] if x)
    return values, note


def _extract_xgb_client_values(df, *, strict=False):
    train_value, train_note = _last_numeric_value(
        df,
        value_col=SERVER_TIME_VALUE_COL,
        strict=strict,
        scope_value=ALL_ROUNDS_VALUE if SCOPE_COL in df.columns else None,
        metric_value=XGB_CLIENT_LOCAL_TRAINING_METRIC_VALUE,
    )
    total_value, total_note = _last_numeric_value(
        df,
        value_col=SERVER_TIME_VALUE_COL,
        strict=strict,
        scope_value=ALL_ROUNDS_VALUE if SCOPE_COL in df.columns else None,
        metric_value=XGB_CLIENT_TOTAL_TIME_METRIC_VALUE,
    )
    round_count = _infer_numeric_round_count(df)
    if train_value is not None and round_count and round_count > 0:
        mean_train_per_round = train_value / round_count
        mean_note = f"mean per-round local training time = {XGB_CLIENT_LOCAL_TRAINING_METRIC_VALUE} / {round_count} numeric rounds"
    else:
        mean_train_per_round = None
        mean_note = f"could not calculate mean per-round local training time because {XGB_CLIENT_LOCAL_TRAINING_METRIC_VALUE} or round count was missing"

    values = {
        "local_training_time_sec": train_value,
        "mean_per_round_local_training_time_sec": mean_train_per_round,
        "total_client_round_time_sec": total_value,
        "client_numeric_round_count": round_count,
    }
    note = "; ".join(x for x in [train_note, total_note, mean_note] if x)
    return values, note


def extract_client_timing_data_summary(fl_site_results_dirs, site_ids=None, *, algorithm_name, strict=False):
    if fl_site_results_dirs is None:
        raise ValueError("fl_site_results_dirs is required for client timing extraction.")
    if site_ids is None:
        site_ids = sorted(fl_site_results_dirs.keys(), key=_site_sort_key)

    is_xgb = is_xgb_histagg_algorithm(algorithm_name)
    rows = []
    for site_id in site_ids:
        site_dir = _get_mapping_value(fl_site_results_dirs, site_id, "fl_site_results_dirs")
        timing_file, file_note = _find_client_timing_file(site_dir, algorithm_name=algorithm_name, strict=strict)

        if is_xgb:
            values = {
                "local_training_time_sec": None,
                "mean_per_round_local_training_time_sec": None,
                "total_client_round_time_sec": None,
                "client_numeric_round_count": None,
            }
            metric_descriptions = {
                "local_training_time_sec": "Client XGB Hist Agg local training time from the final all_rounds federated_xgb_train_time_sec row.",
                "mean_per_round_local_training_time_sec": "Client XGB Hist Agg local training time divided by inferred numeric round count.",
                "total_client_round_time_sec": "Client XGB Hist Agg total elapsed time from the final all_rounds client_elapsed_time_sec row.",
                "client_numeric_round_count": "Number of unique numeric rounds inferred from client timing_metrics.csv, excluding round -1 summary rows.",
            }
        else:
            values = {
                "local_training_time_sec": None,
                "mean_per_round_local_training_time_sec": None,
                "total_client_round_time_sec": None,
                "client_numeric_round_count": None,
            }
            metric_descriptions = {
                "local_training_time_sec": "Client cumulative local training time from the final all_rounds local_train_total_time_sec row.",
                "mean_per_round_local_training_time_sec": "Client cumulative local training time divided by inferred numeric round count.",
                "total_client_round_time_sec": "Client cumulative total round time from the final all_rounds client_round_total_elapsed_sec row.",
                "client_numeric_round_count": "Number of unique numeric rounds inferred from client timing_metrics.csv.",
            }

        value_note = ""
        if timing_file is not None:
            df = _safe_read_csv(timing_file, strict=strict)
            if is_xgb:
                values, value_note = _extract_xgb_client_values(df, strict=strict)
            else:
                values, value_note = _extract_non_xgb_client_values(df, strict=strict)

        for metric_name, value in values.items():
            unit = "count" if metric_name == "client_numeric_round_count" else "sec"
            rows.append(_make_row(
                algorithm_name=algorithm_name,
                scope="client",
                site_id=site_id,
                metric_name=metric_name,
                value=value,
                unit=unit,
                source_file=timing_file,
                source_column=SERVER_TIME_VALUE_COL,
                description=metric_descriptions[metric_name],
                note="; ".join(x for x in [file_note, value_note] if x),
            ))
    return pd.DataFrame(rows)


def _get_metric_value(summary_long_df, *, scope, site_id, metric_name):
    if summary_long_df is None or summary_long_df.empty:
        return None
    sub = summary_long_df[
        summary_long_df["scope"].astype(str).eq(str(scope))
        & summary_long_df["site_id"].astype(str).eq(str(site_id))
        & summary_long_df["metric_name"].astype(str).eq(str(metric_name))
    ]
    if sub.empty:
        return None
    vals = pd.to_numeric(sub["value"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def _get_metric_raw_value(summary_long_df, *, scope, site_id, metric_name):
    if summary_long_df is None or summary_long_df.empty:
        return None
    sub = summary_long_df[
        summary_long_df["scope"].astype(str).eq(str(scope))
        & summary_long_df["site_id"].astype(str).eq(str(site_id))
        & summary_long_df["metric_name"].astype(str).eq(str(metric_name))
    ]
    if sub.empty:
        return None
    val = sub["value"].iloc[-1]
    if pd.isna(val):
        return None
    return val


def make_fl_training_report_table(summary_long_df):
    columns = [
        "Cumulative server aggregation time (s)",
        "Mean wall-clock time for 500 rounds (s)",
        "Total data transmitted (MiB)",
        "Total data transmitted per round (MiB; mean (SD); direction)",
    ]
    if summary_long_df is None or summary_long_df.empty:
        return pd.DataFrame(columns=columns)

    cumulative_server_aggregation_time = _get_metric_value(
        summary_long_df,
        scope="server",
        site_id="all_sites",
        metric_name="cumulative_server_aggregation_time_sec",
    )
    total_server_round_duration_sec = _get_metric_value(
        summary_long_df,
        scope="server",
        site_id="all_sites",
        metric_name="total_server_round_duration_sec",
    )
    if total_server_round_duration_sec is None:
        total_server_round_duration_sec = _get_metric_value(
            summary_long_df,
            scope="server",
            site_id="all_sites",
            metric_name="total_server_run_duration_sec",
        )

    server_round_count = _get_metric_value(
        summary_long_df,
        scope="server",
        site_id="all_sites",
        metric_name="server_numeric_round_count",
    )
    # This report is for 500-round FL training. If the round count cannot be
    # inferred from the timing CSV, use 500 as the intended denominator.
    denominator = server_round_count if server_round_count and server_round_count > 0 else 500.0
    mean_wall_clock_sec = (
        None if total_server_round_duration_sec is None else float(total_server_round_duration_sec) / float(denominator)
    )

    total_data_transmitted_mib = _get_metric_value(
        summary_long_df,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_mb",
    )
    per_round_direction_summary = _get_metric_raw_value(
        summary_long_df,
        scope="server",
        site_id="all_sites",
        metric_name="total_data_transmitted_per_round_mib_mean_sd_direction",
    )

    return pd.DataFrame([{
        "Cumulative server aggregation time (s)": cumulative_server_aggregation_time,
        "Mean wall-clock time for 500 rounds (s)": mean_wall_clock_sec,
        "Total data transmitted (MiB)": total_data_transmitted_mib,
        "Total data transmitted per round (MiB; mean (SD); direction)": per_round_direction_summary,
    }])


def collect_timing_data_summary(
    *,
    algorithm_name,
    server_results_dir,
    fl_site_results_dirs,
    site_ids=None,
    output_dir=None,
    strict=False,
    verbose=True,
):
    algorithm_name = _normalise_algorithm_name(algorithm_name)

    if is_xgb_histagg_algorithm(algorithm_name):
        server_df = extract_xgb_histagg_server_summary(
            server_results_dir=server_results_dir,
            algorithm_name=algorithm_name,
            strict=strict,
        )
    else:
        server_df = extract_non_xgb_histagg_server_summary(
            server_results_dir=server_results_dir,
            algorithm_name=algorithm_name,
            strict=strict,
        )

    client_df = extract_client_timing_data_summary(
        fl_site_results_dirs=fl_site_results_dirs,
        site_ids=site_ids,
        algorithm_name=algorithm_name,
        strict=strict,
    )
    summary_long_df = pd.concat([server_df, client_df], ignore_index=True)
    fl_training_report_table_df = make_fl_training_report_table(summary_long_df)

    output_paths = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        long_out = output_dir / f"{algorithm_name}_timing_data_summary_long.csv"
        report_table_out = output_dir / f"{algorithm_name}_fl_training_report_table.csv"
        summary_long_df.to_csv(long_out, index=False)
        fl_training_report_table_df.to_csv(report_table_out, index=False)
        output_paths = {
            "timing_data_summary_long_out": long_out,
            "fl_training_report_table_out": report_table_out,
        }

    if verbose:
        label = "XGB Hist Agg" if is_xgb_histagg_algorithm(algorithm_name) else "Non-XGB Hist Agg"
        print(f"\n{label} timing/data summary for algorithm_name='{algorithm_name}':")
        print("\nDetailed timing/data extraction table:")
        if summary_long_df.empty:
            print("  No timing/data summary values were extracted.")
        else:
            cols = ["algorithm_name", "scope", "site_id", "metric_name", "value", "unit", "source_file", "note"]
            print(summary_long_df[cols].to_string(index=False))
        print("\nManuscript-style FL training table:")
        if fl_training_report_table_df.empty:
            print("  No manuscript-style training table was created.")
        else:
            print(fl_training_report_table_df.to_string(index=False))
        for name, path in output_paths.items():
            print(f"  {name}: {path}")

    return {
        "algorithm_name": algorithm_name,
        "summary_long_df": summary_long_df,
        "fl_training_report_table_df": fl_training_report_table_df,
        "output_paths": output_paths,
    }