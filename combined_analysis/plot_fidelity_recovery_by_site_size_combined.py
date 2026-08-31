"""
Plot within-network FL vs local Brier differences against site sample size
across clinical tasks.

This script reads ``fidelity_recovery_differences.csv`` from the timestamped
analysis output produced by ``functions.executor.run_analysis()`` for each
clinical task and retrieves Brier differences and 95% CIs. Site sample sizes are specified directly in this script.

Expected input:
    <analysis_run_dir>/heldout_eval/fidelity_recovery_differences.csv

Outputs:
    Combined Brier-score difference versus site sample-size plots saved as PNG and PDF files.
"""
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

SEPSIS_ANALYSIS_DIR = Path(r"C:\path\to\sepsis_analysis_output")
AMI_ANALYSIS_DIR = Path(r"C:\path\to\ami_analysis_output")
DIABETES_ANALYSIS_DIR = Path(r"C:\path\to\diabetes_analysis_output")
OUTPUT_DIR = Path(r"C:\path\to\combined_figure_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TASK_RUNS = [
    ("Sepsis", SEPSIS_ANALYSIS_DIR, {
        1: ("Site 1", 6837), # specify sample sizes for each site
        2: ("Site 2", 2337),
        3: ("Site 3", 3112),
        4: ("Site 4", 993),
    }),
    ("AMI", AMI_ANALYSIS_DIR, {
        1: ("Site 1", 4248),
        2: ("Site 2", 1180),
        3: ("Site 3", 442),
        4: ("Site 4", 650),
        5: ("Site 5", 1152),
        6: ("Site 6", 778),
    }),
    ("Diabetes", DIABETES_ANALYSIS_DIR, {
        1: ("Site A", 8018),
        2: ("Site B", 6228),
        3: ("Site C", 4635),
    }),
]

style = {
    "Sepsis": ("#2C6FA6", "o"),
    "AMI": ("#B5651D", "X"),
    "Diabetes": ("#2E8B6B", "D"),
}

def load_results(task_runs):
    sites = []

    for task, analysis_dir, site_info in task_runs:
        csv_path = analysis_dir / "heldout_eval" / "fidelity_recovery_differences.csv"

        if not csv_path.is_file():
            print(f"Skipping {task}: analysis output not found")
            continue

        df = pd.read_csv(csv_path)
        task_rows = []

        for site_id, (site_label, sample_size) in site_info.items():
            row = df[
                (pd.to_numeric(df["site_id"], errors="coerce") == site_id)
                & (df["setting"] == "Within-network")
                & (df["comparison"] == "FL - local")
            ]

            if len(row) != 1:
                print(f"Skipping {task}: incomplete result at {site_label}")
                task_rows = []
                break

            row = row.iloc[0]
            task_rows.append((
                task,
                site_label,
                sample_size,
                float(row["brier_difference"]),
                float(row["brier_ci_lower"]),
                float(row["brier_ci_upper"]),
            ))

        sites.extend(task_rows)

    if not sites:
        raise ValueError("No complete clinical-task results were found.")

    return sites

def add_direction_arrows(ax, better_side, y=-0.1, color="#C0392B",
                         fontsize=9, gap=0.012, max_len=0.3):
    xlo, xhi = ax.get_xlim()
    f0 = (0.0 - xlo) / (xhi - xlo)
    f0 = min(max(f0, 0.0), 1.0)

    Ll = max(0.0, min(max_len, f0 - gap - 0.02))
    Lr = max(0.0, min(max_len, (1 - f0) - gap - 0.02))

    tr = ax.transAxes
    ap = dict(arrowstyle="-|>", color=color, lw=2)

    if Ll > 0.02: # left-pointing arrow
        ax.annotate("", xy=(f0-gap-Ll, y), xytext=(f0-gap, y),
                    xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)

    if Lr > 0.02: # right-pointing arrow
        ax.annotate("", xy=(f0+gap+Lr, y), xytext=(f0+gap, y),
                    xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)

    left_label = "FL better" if better_side == "left" else "FL worse"
    right_label = "FL worse" if better_side == "left" else "FL better"

    if Ll > 0.02:
        ax.text(f0-gap-Ll/2, y-0.02, left_label, transform=tr,
                ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)

    if Lr > 0.02:
        ax.text(f0+gap+Lr/2, y-0.02, right_label, transform=tr,
                ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)


sites = load_results(TASK_RUNS)

fig, ax = plt.subplots(figsize=(6, 7))
ax.set_yscale("log")
ax.axhspan(300, 1500, color="#F4F4F0", alpha=0.7, zorder=0)
ax.axvspan(-0.01, 0.01, color="#EAF1F8", alpha=0.7, zorder=0)
ax.axvline(0, color="#333", ls=(0, (5, 3)), lw=1, zorder=1)

for task, site, n, d, lo, hi in sites:
    c, m = style[task]
    ax.errorbar(
        d, n,
        xerr=[[d-lo], [hi-d]],
        fmt=m,
        color=c,
        mfc=c,
        mec=c,
        ms=8,
        capsize=3,
        lw=1.2,
        elinewidth=1.2,
        zorder=4,
    )

add_direction_arrows(ax, "left")

ax.set_xlim(-0.045, 0.025)
ax.set_ylim(360, 9500)
ax.yaxis.set_major_locator(FixedLocator([500, 1000, 2000, 4000, 8000]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FixedFormatter(["500", "1000", "2000", "4000", "8000"]))

ax.set_xlabel("Brier difference  (FL − local)", fontsize=10)
ax.set_ylabel("Site sample size (n, log scale)", fontsize=10)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax.text(-0.043, 1400, "sites with n < 1,500",
        fontsize=9, color="#666", ha="left", va="center")

available_tasks = [
    task for task in ["Sepsis", "AMI", "Diabetes"]
    if any(row[0] == task for row in sites)
]

handles = [
    Line2D(
        [0], [0],
        marker=style[task][1],
        color="w",
        markerfacecolor=style[task][0],
        markeredgecolor=style[task][0],
        markersize=8,
        label=task,
    )
    for task in available_tasks
]

ax.legend(
    handles=handles,
    loc="upper left",
    fontsize=9,
    frameon=True,
    title="Clinical task",
    title_fontsize=9,
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_fidelity_recovery_by_site_size_brier.png", dpi=200, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig4_fidelity_recovery_by_site_size_brier.pdf", bbox_inches="tight")
plt.close()

print(f"saved {OUTPUT_DIR / 'fig4_fidelity_recovery_by_site_size_brier.pdf'}")