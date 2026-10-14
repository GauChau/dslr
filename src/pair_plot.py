#!/usr/bin/env python3
"""
Pair plot visualization for DSLR.
Answers the question: From this visualization, what features are you going to use for your logistic regression?
Saves the resulting plot to plots/pair_plot.png and displays the figure.
"""

import os
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is on sys.path
SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


HOUSE_PALETTE = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}


def print_feature_selection_summary() -> None:
    print("=" * 74)
    print("  Hogwarts Courses Pair Plot & Feature Selection Analysis")
    print("=" * 74)
    print("  Question: From this visualization, what features are you going to use")
    print("            for your logistic regression?")
    print("  " + "-" * 70)
    print("  1. Features to EXCLUDE (Noisy / Homogeneous):")
    print("     - 'Care of Magical Creatures': Four houses overlap completely (no separation).")
    print("     - 'Arithmancy': Four houses overlap completely (no separation).")
    print("")
    print("  2. Features to EXCLUDE (Redundant / Multi-collinear):")
    print("     - 'Defense Against the Dark Arts': Perfectly collinear with 'Astronomy'")
    print("       (r = -1.000000); keeping both adds redundant parameters.")
    print("")
    print("  3. Features SELECTED for Logistic Regression (10 features):")
    print("     - 'Astronomy'            (Separates Slytherin & Ravenclaw)")
    print("     - 'Herbology'            (Separates Gryffindor & Hufflepuff)")
    print("     - 'Divination'           (Clear multimodal cluster separation)")
    print("     - 'Muggle Studies'       (Distinct house clusters)")
    print("     - 'Ancient Runes'        (Strong house distinction)")
    print("     - 'History of Magic'     (Strong discriminative boundaries)")
    print("     - 'Transfiguration'      (Strong discriminative boundaries)")
    print("     - 'Potions'              (Moderate separation)")
    print("     - 'Charms'               (Distinct multi-house distribution)")
    print("     - 'Flying'               (Strong separating diagonal)")
    print("=" * 74)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py <dataset_train.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    dataset = pd.read_csv(csv_path)

    # Filter to Hogwarts House and all numeric courses (excluding Index)
    numeric_courses = [c for c in dataset.select_dtypes(include="number").columns if c != "Index"]
    cols_to_plot = ["Hogwarts House"] + numeric_courses
    plot_df = dataset[cols_to_plot].dropna()

    # 1. Print feature selection analysis
    print_feature_selection_summary()

    # 2. Render Pair Plot using seaborn
    print("\nRendering pair plot (this may take a few seconds)...")
    g = sns.pairplot(
        plot_df,
        hue="Hogwarts House",
        palette=HOUSE_PALETTE,
        corner=True,
        height=1.2,
        plot_kws={"s": 8, "alpha": 0.45},
        diag_kws={"alpha": 0.5},
    )
    g.fig.suptitle(
        "Hogwarts Courses Pair Plot Matrix (Feature Selection Visualization)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # 3. Save plot to plots/
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/pair_plot.png"
    g.savefig(out_path, dpi=90, bbox_inches="tight")
    print(f"[OUTPUT] Saved pair plot visualization to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
