#!/usr/bin/env python3
"""
DSLR Learning Curves Visualization Module
==========================================
Generates 5-optimizer comparative learning curve visualizations:
  - 4 House subplots (Loss vs. Epochs) with Early Stopping annotations
  - 1 Overall mean loss trajectory comparison
  - 1 Performance summary horizontal bar chart
Exports figure to learning_curves.png.

Comprehensive Ablation & Benchmark Dashboard:
  - 4-panel dashboard visualizing Optimizers dynamics, L2 weight decay,
    feature compression, and Early Stopping computational savings.
Exports figures to plots/learning_curves.png and plots/visualization.png.
"""

from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.early_stopping import TrainingHistory


def plot_learning_curves(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    save_path: str = "plots/learning_curves.png",
) -> dict[str, TrainingHistory]:
    """
    Train all 5 optimizers, plot Loss vs. Epochs convergence comparisons, and save figure.

    Arguments:
        hold_df: Standardized training DataFrame containing feature values and one-hot house targets.
        weights: DataFrame with feature indices and house columns.
        bias: Series with house biases.
        l2_lambda: Non-negative L2 regularization parameter.
        early_stopping: Flag to enable early stopping during benchmarking.
        patience: Epochs to wait for loss improvement before early termination.
        min_delta: Minimum loss delta considered as improvement.
        save_path: Destination path for the generated plot image.

    Returns:
        dict[str, TrainingHistory]: Dictionary mapping optimizer names to their training loss histories.
    """
    # Import optimize locally to avoid circular dependencies
    from utils.optimizers import optimize

    maisons = list(weights.columns)
    optimizers_meta = [
        ("bgd", "Batch GD (BGD)", "#1f77b4", "-"),
        ("sgd", "Stochastic GD (SGD)", "#ff7f0e", "--"),
        ("mini", "Mini-batch GD (B=32)", "#2ca02c", "-."),
        ("momentum", "Momentum GD", "#9467bd", ":"),
        ("adam", "Adam", "#d62728", "-"),
    ]

    all_histories: dict[str, TrainingHistory] = {}
    final_losses_summary: dict[str, float] = {}
    epochs_ran_summary: dict[str, int] = {}

    print("\n[Plot] Benchmarking all 5 optimizers for learning curve visualization...")
    for opt_name, label, _, _ in optimizers_meta:
        init_weights = pd.DataFrame(0.0, index=weights.index, columns=maisons)
        init_bias = pd.Series(0.0, index=maisons)
        # Give a consistent 100 epochs horizon for comparison
        _, _, hist = optimize(
            hold_df,
            init_weights,
            init_bias,
            opt=opt_name,
            epochs=100,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=True,
        )
        all_histories[opt_name] = hist
        mean_final_loss = float(np.mean([hist[m][-1] for m in maisons]))
        max_epochs_ran = max(len(hist[m]) for m in maisons)
        final_losses_summary[opt_name] = mean_final_loss
        epochs_ran_summary[opt_name] = max_epochs_ran
        status_suffix = ""
        if early_stopping:
            stopped_houses = [f"{m}@{hist.early_stopped[m]}" for m in maisons if hist.early_stopped.get(m)]
            if stopped_houses:
                status_suffix = f" (Early stopped: {', '.join(stopped_houses)})"
        print(f"  • {label:24s}: Final Mean Log Loss = {mean_final_loss:.5f}, Epochs = {max_epochs_ran}{status_suffix}")

    # Create 2x3 visualization grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Grid positions for the 4 houses
    house_positions = {
        "Gryffindor": axes[0, 0],
        "Hufflepuff": axes[0, 1],
        "Ravenclaw": axes[0, 2],
        "Slytherin": axes[1, 0],
    }

    # 1. Plot individual house learning curves (4 subplots)
    for house, ax in house_positions.items():
        for opt_name, label, color, ls in optimizers_meta:
            hist = all_histories[opt_name]
            losses = hist[house]
            ep_axis = list(range(1, len(losses) + 1))
            stop_ep = hist.early_stopped.get(house)

            lbl = f"{label} ({losses[-1]:.4f})"
            ax.plot(ep_axis, losses, label=lbl, color=color, linestyle=ls, linewidth=2.0, alpha=0.9)

            if stop_ep is not None and stop_ep <= len(losses):
                ax.scatter(
                    [stop_ep],
                    [losses[stop_ep - 1]],
                    color=color,
                    s=120,
                    marker="*",
                    zorder=6,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_title(f"House: {house}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel("Log Loss", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=9, loc="upper right")

    # 2. Subplot (1, 1): Overall Mean Loss Across All 4 Houses
    ax_mean = axes[1, 1]
    for opt_name, label, color, ls in optimizers_meta:
        hist = all_histories[opt_name]
        max_len = max(len(hist[m]) for m in maisons)
        mean_curve = []
        for ep in range(max_len):
            vals = [hist[m][ep] if ep < len(hist[m]) else hist[m][-1] for m in maisons]
            mean_curve.append(float(np.mean(vals)))

        ep_axis = list(range(1, max_len + 1))
        ax_mean.plot(
            ep_axis,
            mean_curve,
            label=f"{label} (Final: {mean_curve[-1]:.4f})",
            color=color,
            linestyle=ls,
            linewidth=2.2,
        )

    ax_mean.set_title("Overall Mean Loss (All 4 Houses Combined)", fontsize=13, fontweight="bold")
    ax_mean.set_xlabel("Epochs", fontsize=11)
    ax_mean.set_ylabel("Average Log Loss", fontsize=11)
    ax_mean.grid(True, linestyle="--", alpha=0.45)
    ax_mean.legend(fontsize=9, loc="upper right")

    # 3. Subplot (1, 2): Performance & Convergence Summary Bar Chart
    ax_bar = axes[1, 2]
    opt_labels = [m[1] for m in optimizers_meta]
    opt_colors = [m[2] for m in optimizers_meta]
    final_loss_vals = [final_losses_summary[m[0]] for m in optimizers_meta]
    epochs_vals = [epochs_ran_summary[m[0]] for m in optimizers_meta]

    bars = ax_bar.barh(opt_labels, final_loss_vals, color=opt_colors, alpha=0.85, edgecolor="black", height=0.55)
    ax_bar.set_xlabel("Final Mean Log Loss", fontsize=11)
    ax_bar.set_title("Optimizer Performance Summary", fontsize=13, fontweight="bold")
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.45)

    # Annotate bars with loss and epochs
    for bar, loss_val, ep_val in zip(bars, final_loss_vals, epochs_vals):
        width = bar.get_width()
        ep_text = f"Loss: {loss_val:.4f} ({ep_val} eps)"
        ax_bar.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            ep_text,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
        )

    # Set x-limit to leave room for text
    max_w = max(final_loss_vals)
    ax_bar.set_xlim(0, max_w * 1.35)

    # Global title
    mode_text = "Early Stopping: Enabled" if early_stopping else "Fixed Horizon: 100 Epochs"
    fig.suptitle(
        f"DSLR Learning Curves: 5 Gradient Descent Optimizers Convergence Dynamics\n(L2 lambda = {l2_lambda} | {mode_text})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Comparison chart successfully saved to: {save_path}\n")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

    return all_histories


def visualize_benchmark(
    hold_df: pd.DataFrame,
    verification_df: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    bias: pd.Series | None = None,
    save_path: str = "plots/visualization.png",
    show_plot: bool = True,
) -> None:
    """
    Generate a 4-panel comprehensive ablation & benchmark dashboard:
      Panel 1: Optimizer Convergence Dynamics (Loss vs. Epochs for 5 Optimizers)
      Panel 2: L2 Regularization Dynamics (Lambda vs ||w||_2 weight norm and validation accuracy)
      Panel 3: Feature Weight Shrinkage Comparison (lambda=0 vs lambda=50 across 13 courses)
      Panel 4: Early Stopping Computational Savings (Fixed 100 epochs vs Early Stopped epochs)

    Arguments:
        hold_df: Standardized training DataFrame containing feature values and one-hot house targets.
        verification_df: Optional standardized validation DataFrame. If None, an 80/20 holdout is formed.
        weights: Optional initial weights DataFrame.
        bias: Optional initial bias Series.
        save_path: Output file path for the benchmark dashboard image (default: plots/visualization.png).
        show_plot: Flag to call plt.show() if running in an interactive graphical environment.
    """
    # Import optimize and sigmoid locally to avoid circular dependencies
    from utils.optimizers import optimize, sigmoid

    maisons = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    available_houses = [m for m in maisons if m in hold_df.columns]
    if len(available_houses) != 4:
        raise ValueError("hold_df must contain one-hot columns for all 4 Hogwarts houses.")

    matieres = [c for c in hold_df.columns if c not in maisons]
    if weights is None:
        weights = pd.DataFrame(0.0, index=matieres, columns=maisons)
    if bias is None:
        bias = pd.Series(0.0, index=maisons)

    if verification_df is None:
        val_df = hold_df.sample(frac=0.2, random_state=42)
        train_df = hold_df.drop(index=val_df.index)
    else:
        train_df = hold_df
        val_df = verification_df

    print("\n[Benchmark] Generating Comprehensive 4-Panel Benchmark Dashboard...")

    # -------------------------------------------------------------
    # Panel 1 Data: 5 Optimizers Convergence Trajectory
    # -------------------------------------------------------------
    print("  [1/4] Evaluating 5-Optimizer Convergence Dynamics...")
    optimizers_meta = [
        ("bgd", "Batch GD (BGD)", "#1f77b4", "-"),
        ("sgd", "Stochastic GD (SGD)", "#ff7f0e", "--"),
        ("mini", "Mini-batch GD (B=32)", "#2ca02c", "-."),
        ("momentum", "Momentum GD", "#9467bd", ":"),
        ("adam", "Adam", "#d62728", "-"),
    ]

    opt_histories: dict[str, TrainingHistory] = {}
    for opt_name, _, _, _ in optimizers_meta:
        init_w = pd.DataFrame(0.0, index=matieres, columns=maisons)
        init_b = pd.Series(0.0, index=maisons)
        _, _, hist = optimize(
            train_df,
            init_w,
            init_b,
            opt=opt_name,
            epochs=100,
            l2_lambda=0.0,
            early_stopping=False,
            return_history=True,
        )
        opt_histories[opt_name] = hist

    # -------------------------------------------------------------
    # Panel 2 Data: L2 Regularization Lambdas (Weight Norm & Accuracy)
    # -------------------------------------------------------------
    print("  [2/4] Scanning L2 Regularization Lambdas (Weight Norm & Accuracy)...")
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0]
    w_norms = []
    val_accuracies = []
    w_lam0: pd.DataFrame | None = None
    w_lam50: pd.DataFrame | None = None

    for lam in lambdas:
        init_w = pd.DataFrame(0.0, index=matieres, columns=maisons)
        init_b = pd.Series(0.0, index=maisons)
        w_res, b_res = optimize(
            train_df,
            init_w,
            init_b,
            opt="bgd",
            epochs=100,
            l2_lambda=lam,
            early_stopping=False,
        )
        norm_val = float(np.linalg.norm(w_res.to_numpy()))
        w_norms.append(norm_val)

        if lam == 0.0:
            w_lam0 = w_res.copy()
        elif lam == 50.0:
            w_lam50 = w_res.copy()

        # Multi-class accuracy on validation set
        val_probs = pd.DataFrame(index=val_df.index, columns=maisons)
        for m in maisons:
            z = val_df[matieres].dot(w_res[m]) + b_res[m]
            val_probs[m] = sigmoid(z)
        pred_house = val_probs.idxmax(axis=1)
        true_house = val_df[maisons].idxmax(axis=1)
        acc = float((pred_house == true_house).mean() * 100.0)
        val_accuracies.append(acc)

    # -------------------------------------------------------------
    # Panel 3 Data: Feature Weight Shrinkage (lambda=0 vs. lambda=50)
    # -------------------------------------------------------------
    print("  [3/4] Computing Feature Weight Shrinkage (lambda=0 vs. lambda=50)...")
    if w_lam0 is None or w_lam50 is None:
        init_w = pd.DataFrame(0.0, index=matieres, columns=maisons)
        init_b = pd.Series(0.0, index=maisons)
        w_lam0, _ = optimize(train_df, init_w, init_b, opt="bgd", epochs=100, l2_lambda=0.0)
        init_w = pd.DataFrame(0.0, index=matieres, columns=maisons)
        init_b = pd.Series(0.0, index=maisons)
        w_lam50, _ = optimize(train_df, init_w, init_b, opt="bgd", epochs=100, l2_lambda=50.0)

    mean_abs_0 = w_lam0.abs().mean(axis=1)
    mean_abs_50 = w_lam50.abs().mean(axis=1)
    sorted_features = mean_abs_0.sort_values(ascending=True).index.tolist()
    vals_0 = [mean_abs_0[f] for f in sorted_features]
    vals_50 = [mean_abs_50[f] for f in sorted_features]

    # -------------------------------------------------------------
    # Panel 4 Data: Early Stopping Efficiency Across 5 Optimizers
    # -------------------------------------------------------------
    print("  [4/4] Evaluating Early Stopping Savings Across 5 Optimizers...")
    es_epochs: dict[str, float] = {}
    for opt_name, _, _, _ in optimizers_meta:
        init_w = pd.DataFrame(0.0, index=matieres, columns=maisons)
        init_b = pd.Series(0.0, index=maisons)
        _, _, hist_es = optimize(
            train_df,
            init_w,
            init_b,
            opt=opt_name,
            epochs=100,
            l2_lambda=0.0,
            early_stopping=True,
            patience=5,
            min_delta=1e-4,
            return_history=True,
        )
        avg_ep = float(np.mean([len(hist_es[m]) for m in maisons]))
        es_epochs[opt_name] = avg_ep

    # -------------------------------------------------------------
    # 2x2 Dashboard Figure Plotting
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- Subplot (0, 0): Optimizer Convergence Dynamics ---
    ax1 = axes[0, 0]
    for opt_name, label, color, ls in optimizers_meta:
        hist = opt_histories[opt_name]
        mean_losses = [float(np.mean([hist[m][ep] for m in maisons])) for ep in range(100)]
        ax1.plot(
            range(1, 101),
            mean_losses,
            label=f"{label} (Final: {mean_losses[-1]:.4f})",
            color=color,
            linestyle=ls,
            linewidth=2.0,
            alpha=0.9,
        )
    ax1.set_title("1. Optimizer Convergence Dynamics (Loss vs. Epochs)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Mean Log Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(fontsize=9, loc="upper right")

    # --- Subplot (0, 1): L2 Regularization Dynamics ---
    ax2 = axes[0, 1]
    lam_indices = list(range(len(lambdas)))
    lam_labels = [str(l) for l in lambdas]
    color_w = "#1f77b4"
    ax2.plot(lam_indices, w_norms, color=color_w, marker="o", linewidth=2.2, label=r"L2 Weight Norm $\|w\|_2$")
    ax2.set_xlabel(r"Regularization Strength $\lambda$", fontsize=11)
    ax2.set_ylabel(r"L2 Weight Norm $\|w\|_2$", color=color_w, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_w)
    ax2.set_xticks(lam_indices)
    ax2.set_xticklabels(lam_labels)
    ax2.grid(True, linestyle="--", alpha=0.45)

    ax2_twin = ax2.twinx()
    color_acc = "#2ca02c"
    ax2_twin.plot(
        lam_indices,
        val_accuracies,
        color=color_acc,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label="Validation Accuracy (%)",
    )
    ax2_twin.set_ylabel("Validation Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_acc)
    min_acc = min(val_accuracies) - 2.0
    max_acc = max(val_accuracies) + 2.0
    ax2_twin.set_ylim(max(0.0, min_acc), min(100.0, max_acc))
    ax2.set_title("2. L2 Regularization: Weight Decay vs. Accuracy Trade-off", fontsize=13, fontweight="bold")

    # --- Subplot (1, 0): L2 Feature Shrinkage Bar Chart ---
    ax3 = axes[1, 0]
    y_pos = np.arange(len(sorted_features))
    height = 0.38
    ax3.barh(
        y_pos + height / 2,
        vals_0,
        height,
        label=r"Unregularized ($\lambda=0$)",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax3.barh(
        y_pos - height / 2,
        vals_50,
        height,
        label=r"Regularized ($\lambda=50$)",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
    )
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sorted_features, fontsize=9.5)
    ax3.set_xlabel(r"Mean Absolute Weight $|w|$ (Across 4 Houses)", fontsize=11)
    ax3.set_title(r"3. L2 Feature Weight Shrinkage ($\lambda=0$ vs. $\lambda=50$)", fontsize=13, fontweight="bold")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.45)
    ax3.legend(fontsize=9.5, loc="lower right")

    # --- Subplot (1, 1): Early Stopping Efficiency Bar Chart ---
    ax4 = axes[1, 1]
    opt_names = [m[0] for m in optimizers_meta]
    opt_labels = [m[1] for m in optimizers_meta]
    opt_colors = [m[2] for m in optimizers_meta]
    actual_eps = [es_epochs[name] for name in opt_names]
    y_pos4 = np.arange(len(opt_labels))

    ax4.barh(
        y_pos4,
        [100] * len(opt_labels),
        color="#e0e0e0",
        height=0.55,
        edgecolor="gray",
        linestyle="--",
        label="Fixed Horizon (100 Epochs)",
    )
    bars_es = ax4.barh(
        y_pos4,
        actual_eps,
        color=opt_colors,
        height=0.55,
        alpha=0.85,
        edgecolor="black",
        label="Early Stopping Horizon",
    )
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(opt_labels, fontsize=10)
    ax4.set_xlabel("Epochs Executed", fontsize=11)
    ax4.set_xlim(0, 145)
    ax4.set_title("4. Early Stopping Efficiency & Compute Savings (Patience = 5)", fontsize=13, fontweight="bold")
    ax4.grid(True, axis="x", linestyle="--", alpha=0.45)
    ax4.legend(fontsize=9, loc="lower right", bbox_to_anchor=(0.98, 0.23))

    for bar, ep_val in zip(bars_es, actual_eps):
        saved = ((100.0 - ep_val) / 100.0) * 100.0
        txt = f"{ep_val:.0f} eps (Saved {saved:.1f}%)" if saved > 0 else f"{ep_val:.0f} eps (Full)"
        ax4.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    # Global title & saving
    fig.suptitle(
        "DSLR Comprehensive Benchmark: Optimizers, L2 Regularization & Early Stopping",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Benchmark] Dashboard successfully generated and saved to: {save_path}\n")

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# Aliases for flexible API compatibility
visualization = visualize_benchmark
visualize = visualize_benchmark
plot_comprehensive_benchmark = visualize_benchmark
plot_ablation_benchmark = visualize_benchmark

__all__ = [
    "plot_learning_curves",
    "visualize_benchmark",
    "visualization",
    "visualize",
    "plot_comprehensive_benchmark",
    "plot_ablation_benchmark",
]
