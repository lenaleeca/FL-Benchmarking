"""
Build site-specific baseline characteristic tables from preprocessed CSV files.

Input:
    Path to the folder containing site-level CSV files. Each CSV file contains the outcome variable and
    baseline variables. The variables included in the table are specified
    in VARS_FOR_TABLE1.

Output:
    One Word document (.docx) per site containing the baseline characteristic
    table, summarized overall and by outcome.
"""

from pathlib import Path
import sys
import pandas as pd


FUNCTIONS_DIR = Path(r"C:\path\to\FL-Benchmarking")

INPUT_DIR = Path(
    r"C:\path\to\data"
)

OUTPUT_DIR = Path(
    r"C:\path\to\reports"
)

sys.path.insert(0, str(FUNCTIONS_DIR))

from functions.common.build_characteristic_table import build_table1, save_table1_to_word

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


csv_files = sorted(INPUT_DIR.glob("site_*.csv"))

print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(" -", f.name)

# -----------------------------
# put the variables for Table 1 here
# outcome variable must be first
# -----------------------------
vars_for_table1 = [
    "ase01",   # outcome column
    "age",
    "sexmale01",
    "iculos_day",
    "bloodcx01",
    "antimic_any01",
    "imv_any01",
    "vasop_any01",
    "death",
    "labs_blood_lactate_max"
]

for csv_file in csv_files:
    print("\n----------------------------------")
    print(f"Processing: {csv_file.name}")

    df = pd.read_csv(csv_file)
    print("Shape:", df.shape)

    df_use = df[vars_for_table1].copy()

    table1 = build_table1(
        df_use,
        label_col=0
    )

    site_name = csv_file.stem
    out_docx = OUTPUT_DIR / f"{site_name}_table1.docx"
    
    save_table1_to_word(
        table1,
        str(out_docx),
        title="Table 1. Baseline Characteristics",
        site_id=site_name,
        n_rows=len(df_use)
    )

    print(f"Saved Word: {out_docx.name}")
    