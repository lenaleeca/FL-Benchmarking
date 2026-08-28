"""
Create combined fidelity and transportability forest plots across clinical tasks.

This script reads ``fidelity_recovery_differences.csv`` from the timestamped
analysis output produced by ``functions.executor.run_analysis()`` for each
clinical task.

Expected input:
    <analysis_run_dir>/heldout_eval/fidelity_recovery_differences.csv

Outputs:
    Combined AUC and Brier-score forest plots saved as PNG and PDF files.
"""

from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEPSIS_ANALYSIS_DIR = Path(r"C:\path\to\sepsis_analysis_output")
AMI_ANALYSIS_DIR = Path(r"C:\path\to\ami_analysis_output")
DIABETES_ANALYSIS_DIR = Path(r"C:\path\to\diabetes_analysis_output")
OUTPUT_DIR = Path(r"C:\path\to\figure_analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_RUNS = [
    ("Sepsis", SEPSIS_ANALYSIS_DIR, {1: "Site 1", 2: "Site 2", 3: "Site 3", 4: "Site 4"}),
    ("AMI", AMI_ANALYSIS_DIR, {1: "Site 1", 2: "Site 2", 3: "Site 3", 4: "Site 4", 5: "Site 5", 6: "Site 6"}),
    ("Diabetes", DIABETES_ANALYSIS_DIR, {1: "Site A", 2: "Site B", 3: "Site C"}),
]

order = [
    ("Sepsis", ["Site 1", "Site 2", "Site 3", "Site 4"]),
    ("AMI", ["Site 1", "Site 2", "Site 3", "Site 4", "Site 5", "Site 6"]),
    ("Diabetes", ["Site A", "Site B", "Site C"]),
]

FLc = "#6A61C6"
CENTc = "#C07D2A"


def load_fidelity_results(task_runs):
    D = {}

    for task, analysis_dir, site_labels in task_runs:
        csv_path = analysis_dir / "heldout_eval" / "fidelity_recovery_differences.csv"

        if not csv_path.is_file():
            print(f"Skipping {task}: analysis output not found")
            continue

        df = pd.read_csv(csv_path)
        task_results = {}
        task_complete = True

        for site_id, site_label in site_labels.items():
            site_df = df[pd.to_numeric(df["site_id"], errors="coerce") == site_id]
            site_result = {"AUC": {}, "BRIER": {}}

            for short, setting in {"WN": "Within-network", "LSO": "Leave-site-out"}.items():
                setting_df = site_df[site_df["setting"] == setting]
                fl = setting_df[setting_df["comparison"] == "FL - local"]
                ce = setting_df[setting_df["comparison"] == "Centralised - local"]

                if len(fl) != 1 or len(ce) != 1:
                    print(f"Skipping {task}: incomplete results at {site_label}, {setting}")
                    task_complete = False
                    break

                fl, ce = fl.iloc[0], ce.iloc[0]

                site_result["AUC"][short] = (
                    (fl["auc_difference"], fl["auc_ci_lower"], fl["auc_ci_upper"]),
                    (ce["auc_difference"], ce["auc_ci_lower"], ce["auc_ci_upper"]),
                )

                site_result["BRIER"][short] = (
                    (fl["brier_difference"], fl["brier_ci_lower"], fl["brier_ci_upper"]),
                    (ce["brier_difference"], ce["brier_ci_lower"], ce["brier_ci_upper"]),
                )

            if not task_complete:
                break

            task_results[(task, site_label)] = site_result

        if task_complete:
            D.update(task_results)

    if not D:
        raise ValueError("No complete clinical-task results were found.")

    return D


D = load_fidelity_results(TASK_RUNS)


def add_direction_arrows(ax, better_side, y=-0.1, color="#C0392B", fontsize=9, gap=0.012, max_len=0.2):
    xlo, xhi = ax.get_xlim()
    f0 = min(max((0 - xlo) / (xhi - xlo), 0), 1)
    Ll = max(0, min(max_len, f0 - gap - 0.02))
    Lr = max(0, min(max_len, (1 - f0) - gap - 0.02))
    tr = ax.transAxes
    ap = dict(arrowstyle="-|>", color=color, lw=2)

    if Ll > 0.02:
        ax.annotate("", xy=(f0-gap-Ll, y), xytext=(f0-gap, y), xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)
    if Lr > 0.02:
        ax.annotate("", xy=(f0+gap+Lr, y), xytext=(f0+gap, y), xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)

    left = "Shared better       " if better_side == "left" else "Shared worse"
    right = "Shared worse" if better_side == "left" else "Shared better"

    if Ll > 0.02:
        ax.text(f0-gap-Ll/2, y-0.02, left, transform=tr, ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)
    if Lr > 0.02:
        ax.text(f0+gap+Lr/2, y-0.02, right, transform=tr, ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)


def build(metric, xlabel, xlim, note, fname, better_side):
    rows, ypos, ylab, spans, y = [], [], [], {}, 0.0

    for task, sites in order:
        sites = [site for site in sites if (task, site) in D]

        if not sites:
            continue

        start = y

        for site in sites:
            rows.append((task, site))
            ypos.append(y)
            ylab.append(site)
            y += 1
        spans[task] = (start, y - 1)
        y += 0.8

    ymax = y
    ypos = [ymax - 1 - p for p in ypos]
    spans = {t: (ymax - 1 - b, ymax - 1 - a) for t, (a, b) in spans.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.4), sharey=True)
    off = 0.16

    for ax, setting in zip(axes, ["WN", "LSO"]):
        for i, (task, site) in enumerate(rows):
            fl, ce = D[(task, site)][metric][setting]
            yy = ypos[i]

            ax.errorbar(fl[0], yy+off, xerr=[[fl[0]-fl[1]], [fl[2]-fl[0]]],
                        fmt="o", color=FLc, ms=5, capsize=2.5, lw=1.3, ecolor=FLc, zorder=3)
            ax.errorbar(ce[0], yy-off, xerr=[[ce[0]-ce[1]], [ce[2]-ce[0]]],
                        fmt="s", color=CENTc, ms=5, capsize=2.5, lw=1.3, ecolor=CENTc, zorder=3)

        ax.axvline(0, color="#444", lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axvspan(-0.01, 0.01, color="#EAF1F8", alpha=0.6, zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, ymax - 0.4)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylab, fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        for task, (lo, hi) in spans.items():
            ax.text(xlim[0] + 0.012*(xlim[1]-xlim[0]), (lo+hi)/2, task,
                    rotation=90, va="center", ha="center", fontsize=10.5,
                    fontweight="bold", color="#333")

        add_direction_arrows(ax, better_side)

    axes[0].set_title("Fidelity recovery  (site-contributing)", fontsize=12, fontweight="bold", pad=8)
    axes[1].set_title("Fidelity transportability  (site-withheld)", fontsize=12, fontweight="bold", pad=8)
    axes[0].annotate(note, xy=(0, ymax-0.5), xytext=(0.012*(xlim[1]-xlim[0]), ymax-0.15),
                     fontsize=8, color="#444", va="top")

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FLc, markersize=8, label="FL (HistAgg) \u2212 local"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CENTc, markersize=8, label="Centralised \u2212 local"),
    ]
    fig.legend(handles=leg, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.945])
    plt.savefig(OUTPUT_DIR / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close()

    print(f"saved {OUTPUT_DIR / fname}")


build("AUC", "AUC difference  (shared \u2212 local)", (-0.29, 0.08), " ", "fig3_auc_fidelity_recovery_transportability_combined", better_side="right")
build("BRIER", "Brier difference (shared \u2212 local)", (-0.06, 0.30), " ", "fig3_brier_fidelity_recovery_transportability_combined", better_side="left")