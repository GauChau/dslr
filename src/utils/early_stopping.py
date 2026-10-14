#!/usr/bin/env python3
"""
DSLR Training Lifecycle & Early Stopping Module
================================================
Encapsulates:
  1. TrainingHistory: Tracks per-house log loss trajectories across epochs.
  2. EarlyStopping: Monitors loss improvements and halts optimization when loss plateaus,
     automatically rolling back to the historical best parameter weights.
"""

from __future__ import annotations
import numpy as np


class TrainingHistory(dict):
    """Container tracking loss trajectory and early stopping epoch per house."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopped: dict[str, int | None] = {}


class EarlyStopping:
    """
    Monitor loss convergence and trigger early stopping if improvement plateaus.

    Mathematical Formulation:
      delta_loss = best_loss - current_loss
      - If delta_loss > min_delta: improvement is significant, reset patience counter.
      - If delta_loss <= min_delta: increment patience counter.
      - If patience_counter >= patience: trigger early termination.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_w: np.ndarray | None = None
        self.best_b: float | None = None
        self.counter = 0
        self.stopped_epoch: int | None = None

    def update(self, loss: float, w: np.ndarray, b: float, epoch: int) -> bool:
        """
        Record current epoch's loss and parameter state.

        Returns:
            bool: True if early stopping criteria is met and training should halt.
        """
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.best_w = w.copy()
            self.best_b = b
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def restore_best(self, current_w: np.ndarray, current_b: float) -> tuple[np.ndarray, float]:
        """
        Restore the historical best weights and bias.

        Safeguards against retaining degraded or overfitted parameters from the final epoch.
        """
        if self.best_w is not None and self.best_b is not None:
            return self.best_w, self.best_b
        return current_w, current_b

