#!/usr/bin/env python3
"""
Histogram visualization for DSLR.
Answers the question: Which Hogwarts course has a homogeneous score distribution between all four houses?
Saves the resulting plot to plots/histogram.png and displays the figure.
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
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


def compute_homogeneity(df: pd.DataFrame, courses: list[str]) -> list[tuple[str, float]]:
    """
    Computes between-house variance of means normalized by pooled variance for each course.
    Lower score indicates more homogeneous score distribution across houses.
    """
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    scores = []
    for course in courses:
        means = []
        stds = []
        for house in houses:
            series = df.loc[df["Hogwarts House"] == house, course].dropna()
            if len(series) > 1:
                vals = series.to_numpy()
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1)))
            else:
                means.append(0.0)
                stds.append(1.0)
        pooled_std = float(np.mean(stds)) if stds else 1.0
        norm_var = float(np.var(means)) / (pooled_std ** 2) if pooled_std > 0 else 0.0
        scores.append((course, norm_var))

    scores.sort(key=lambda x: x[1])
    return scores


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python histogram.py <dataset_train.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: file '{csv_path}' not found.")
        sys.exit(1)

    dataset = pd.read_csv(csv_path)
    numeric_cols = [c for c in dataset.select_dtypes(include="number").columns if c != "Index"]

    # 1. Compute homogeneity across houses
    homogeneity_scores = compute_homogeneity(dataset, numeric_cols)

    print("=" * 74)
    print("  Hogwarts Courses Score Distribution Homogeneity Analysis")
    print("=" * 74)
    print("  Rank | Course Name                    | Between-House Variance")
    print("  " + "-" * 60)
    for idx, (course, score) in enumerate(homogeneity_scores[:5], start=1):
        print(f"   {idx:<3} | {course:<30} | {score:<20.6f}")

    most_homo = homogeneity_scores[0][0]
    second_homo = homogeneity_scores[1][0]
    print("  " + "-" * 60)
    print(f"  Conclusion: '{most_homo}' and '{second_homo}' have the most homogeneous")
    print(f"  score distributions between all four houses (lowest between-house variance).")
    print("=" * 74)

    # 2. Plot multi-course histogram matrix with translucent overlaps
    nb_cols = 3
    nb_rows = (len(numeric_cols) + nb_cols - 1) // nb_cols
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(16, 3.8 * nb_rows))
    fig.suptitle(
        "Hogwarts Courses Score Distributions by House\n"
        f"Answer: '{most_homo}' & '{second_homo}' are homogeneous across all 4 houses",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for ax, course in zip(axes.flat, numeric_cols):
        for house, color in HOUSE_COLORS.items():
            vals = dataset.loc[dataset["Hogwarts House"] == house, course].dropna()
            ax.hist(vals, bins=25, alpha=0.45, density=True, color=color, label=house)

        # Highlight homogeneous features with special styling
        if course in (most_homo, second_homo):
            ax.set_title(f"★ {course} (Homogeneous)", fontsize=10, fontweight="bold", color="#8b0000")
            for spine in ax.spines.values():
                spine.set_edgecolor("#8b0000")
                spine.set_linewidth(1.8)
        else:
            ax.set_title(course, fontsize=10, fontweight="bold")

        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2, linestyle="--")

    # Hide any unused subplots
    for ax in list(axes.flat)[len(numeric_cols):]:
        ax.set_visible(False)

    fig.tight_layout()

    # 3. Save plot to plots/
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/histogram.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OUTPUT] Saved histogram visualization to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()