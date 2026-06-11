import argparse
import os
import re
from typing import Dict, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_mutation_code(desc: str, chain: str = "P") -> Optional[str]:
    if not isinstance(desc, str):
        return None
    m = re.search(r"([A-Z])(\d+)([A-Z])", desc)
    if m:
        return f"{m.group(1)}{chain}{m.group(2)}{m.group(3)}"
    m = re.search(r"([A-Za-z]{3})(\d+)([A-Za-z]{3})", desc)
    if m:
        wt1 = THREE_TO_ONE.get(m.group(1).upper())
        mut1 = THREE_TO_ONE.get(m.group(3).upper())
        if wt1 and mut1:
            return f"{wt1}{chain}{m.group(2)}{mut1}"
    return None


def prepare_condition(df: pd.DataFrame, label: str, ddg_map: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        print(f"{label}: no mutation data loaded")
        return pd.DataFrame()
    subset = df[(df["gene"] == "NFE2L2") & (df["summary"] == "missense")].copy()
    subset["mutation_code"] = subset["protein_desc"].apply(parse_mutation_code)
    subset = subset.dropna(subset=["mutation_code"])
    merged = subset.merge(ddg_map, on="mutation_code", how="inner")
    return merged


def plot_condition(merged: pd.DataFrame, label: str, out_dir: str) -> None:
    if merged.empty:
        print(f"{label}: no overlapping mutations with FoldX ddG")
        return
    merged = merged.drop_duplicates(subset=["mutation_code"]).copy()
    merged["mutation_label"] = merged["mutation_code"].str.replace(
        r"^([A-Z])[A-Z](\d+)([A-Z])$",
        r"\1\2\3",
        regex=True,
    )
    order = merged.sort_values("ddG", ascending=False)["mutation_label"]
    plt.figure(figsize=(max(8, 0.35 * len(order)), 4))
    sns.barplot(data=merged, x="mutation_label", y="ddG", order=order
                , color="#4C72B0", linewidth = 1, edgecolor = "black")
    plt.title(f"{label} mutations")
    plt.ylabel("FoldX ΔΔG of Binding (kcal/mol)")
    plt.xlabel("")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.axhline(0, color = "gray", linestyle="--", linewidth=0.8)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}_ddg_barplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate qualitative FoldX ddG barplots for NFE2L2 mutations by condition."
    )
    parser.add_argument("--foldx-ddg", default="./Foldx_binding/2FLU/chainP/ddg_results.csv")
    parser.add_argument("--cross", default="../CROSS_mutations.csv")
    parser.add_argument("--flot", default="../FLOT_mutations.csv")
    parser.add_argument("--nt", default="../NT_mutations.csv")
    parser.add_argument("--ecx", default="../ECX_mutations.csv")
    parser.add_argument("--eox", default="../EOX_mutations.csv")
    parser.add_argument("--out-dir", default="./plots")
    args = parser.parse_args()

    ddg = pd.read_csv(args.foldx_ddg)
    ddg_map = ddg[["mutation_code", "ddG"]].drop_duplicates()

    condition_paths: Dict[str, str] = {
        "CROSS": args.cross,
        "FLOT": args.flot,
        "NT": args.nt,
        "ECX": args.ecx,
        "EOX": args.eox,
    }

    for label, path in condition_paths.items():
        if not os.path.exists(path):
            print(f"{label}: missing file {path}")
            continue
        df = pd.read_csv(path)
        merged = prepare_condition(df, label, ddg_map)
        plot_condition(merged, label, args.out_dir)


if __name__ == "__main__":
    main()
