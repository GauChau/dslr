#!/usr/bin/env python3
"""
Scatter plot visualization for DSLR.
Answers the question: What are the two features that are similar?
Saves the resulting plot to plots/scatter_plot.png and displays the figure.
"""

import os
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src/ is on sys.path
SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import utils.stats as stats


HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Hufflepuff": "#ecb939",
    "Ravenclaw": "#222f5b",
    "Slytherin": "#2a623d",
}


def find_most_similar_features(df: pd.DataFrame, courses: list[str]) -> list[tuple[str, str, float]]:
    """
    Computes pairwise Pearson correlation across all numerical courses using stats.pearson_corr.
    Returns pairs sorted by absolute correlation in descending order.
    """
    pairs = []
    n = len(courses)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = courses[i], courses[j]
            r = stats.pearson_corr(df[c1], df[c2])
            pairs.append((c1, c2, r))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scatter_plot.py <dataset_train.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    dataset = pd.read_csv(csv_path)
    numeric_cols = [c for c in dataset.select_dtypes(include="number").columns if c != "Index"]

    # 1. Compute pairwise correlations
    ranked_pairs = find_most_similar_features(dataset, numeric_cols)
    feat1, feat2, best_r = ranked_pairs[0]

    print("=" * 74)
    print("  Hogwarts Courses Feature Similarity Analysis (Pearson Correlation)")
    print("=" * 74)
    print("  Rank | Feature 1                 | Feature 2                 | Pearson r   | |r|")
    print("  " + "-" * 70)
    for idx, (f1, f2, r) in enumerate(ranked_pairs[:5], start=1):
        print(f"   {idx:<3} | {f1:<25} | {f2:<25} | {r:<11.6f} | {abs(r):.6f}")

    print("  " + "-" * 70)
    print(f"  Conclusion: '{feat1}' and '{feat2}' are the two most similar features")
    print(f"  (perfect negative linear correlation: r = {best_r:.6f}).")
    print("=" * 74)

    # 2. Plot scatter plot colored by house
    fig, ax = plt.subplots(figsize=(10, 7))

    for house, color in HOUSE_COLORS.items():
        house_mask = dataset["Hogwarts House"] == house
        x_vals = dataset.loc[house_mask, feat1]
        y_vals = dataset.loc[house_mask, feat2]
        ax.scatter(x_vals, y_vals, label=house, color=color, alpha=0.6, s=28, edgecolors="none")

    ax.set_title(
        f"Similar Features Scatter Plot: {feat1} vs. {feat2}\n"
        f"Pearson Correlation r = {best_r:.6f} (Strict Linear Relationship)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(feat1, fontsize=11, fontweight="bold")
    ax.set_ylabel(feat2, fontsize=11, fontweight="bold")
    ax.legend(title="Hogwarts House", fontsize=10, title_fontsize=11, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    # 3. Save plot to plots/
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/scatter_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OUTPUT] Saved scatter plot visualization to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
