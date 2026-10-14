#!/usr/bin/env python3
"""
DSLR Optimization Algorithms Module
====================================
Encapsulates:
  1. Shared Sigmoid activation and L2 regularization penalty
  2. Five Gradient Descent optimizers:
     - Batch Gradient Descent (BGD) [Mandatory default]
     - Stochastic Gradient Descent (SGD) [Bonus 2]
     - Mini-batch Gradient Descent (Mini) [Bonus 2]
     - Momentum Gradient Descent [Bonus 2]
     - Adam (Adaptive Moment Estimation) [Bonus 2]
  3. Unified optimize() dispatcher supporting L2 regularization and Early Stopping.

Note:
  EarlyStopping and TrainingHistory are located in utils.early_stopping.
  plot_learning_curves is located in utils.learning_curves.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from utils.early_stopping import EarlyStopping, TrainingHistory


def sigmoid(z: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Compute the sigmoid function 1 / (1 + exp(-z)) with overflow protection."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
# Clip at 500 cuz exp(500) (natural log) is about 1.4e217, which is the largest float64 before inf.
#     Since the sigmoid bounces between 0 and 1, the extreme cases wouldn't make much difference.


def l2_penalty(w: np.ndarray, l2_lambda: float, m_samples: int) -> float:
    """[Bonus 3] Compute the L2 (Ridge) regularization cost term: (lambda / (2 * m)) * sum(w_j^2).

    Note: The bias term (theta_0) is stored separately and is strictly excluded from regularization.
    """
    if l2_lambda <= 0.0 or m_samples <= 0:
        return 0.0
    return float((l2_lambda / (2.0 * m_samples)) * float(w @ w))


def compute_log_loss(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    l2_lambda: float = 0.0,
    m_samples: int | None = None,
) -> float:
    """Compute log loss with optional L2 regularization penalty."""
    if m_samples is None or m_samples <= 0:
        m_samples = len(X)
    z = X @ w + b
    p = np.clip(sigmoid(z), 1e-15, 1.0 - 1e-15)
    loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    if l2_lambda > 0.0 and m_samples > 0:
        loss += float((l2_lambda / (2.0 * m_samples)) * float(w @ w))
    return loss


# =====================================================================
# 1. Batch Gradient Descent (BGD) - 批次梯度下降
#    每個 Epoch 使用「全部訓練樣本 (m 筆)」計算一次平均梯度，才更新一次權重。
#    優點：梯度方向最穩定、直接朝全局最優解前進。
#    缺點：當資料量極大時，每走一步都要算完整張表，運算成本高。
# =====================================================================
def batch_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    tol: float = 1e-4,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """Train One-vs-All logistic regression weights using Batch Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns  # 學院→特徵欄

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    history = TrainingHistory({m: [] for m in maisons})

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)         # 分類模型的真實標籤，通常是 0 或 1。
        w = weights[m].to_numpy(dtype=float).copy()  # 權重向量
        b = float(bias.loc[m])                       # 系統誤差
        history.early_stopped[m] = None
        stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

        for epoch in range(epochs):
            z = X @ w + b   # 矩陣乘法：每筆樣本的加權特徵總和。
            p = sigmoid(z)  # 轉換成機率
            e = p - y       # 計算誤差：預測機率 - 真實標籤

            # L2 正則化僅懲罰特徵權重 w，Bias 項 b (theta_0) 不參與正則化
            # Bonus 3-2.-步驟二
            grads = (X.T @ e) / m_samples + (l2_lambda / m_samples) * w
            grad_bias = float(sum(e)) / m_samples

            w -= learning_rate * grads
            b -= learning_rate * grad_bias

            curr_loss = compute_log_loss(X, y, w, b, l2_lambda, m_samples)
            history[m].append(curr_loss)

            if stopper and stopper.update(curr_loss, w, b, epoch + 1):
                w, b = stopper.restore_best(w, b)
                history.early_stopped[m] = epoch + 1
                print(f"[BGD] Early stopping reached at epoch {epoch + 1} ({m}): loss = {stopper.best_loss:.5f}")
                break

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[BGD] Convergence reached ({m})", _, gradient_max)
                break
            joe =_
        print(joe)
        weights[m] = w
        bias.loc[m] = b

    if return_history:
        return weights, bias, history
    return weights, bias
# X.T @ e:
#     先將矩陣 X 轉置，再乘上誤差向量 e。


# =====================================================================
# 2. Stochastic Gradient Descent (SGD) - 隨機梯度下降 [Bonus]
#    每個 Epoch 先將資料隨機洗牌 (Shuffle)，接著「每看 1 筆學生資料」就立刻算梯度並更新權重。
#    優點：更新頻率極高（1 個 Epoch 更新 m 次），收斂初期極快。
#    缺點：單筆樣本雜訊大，損失曲線會呈現上下震盪，不會完全靜止。
# =====================================================================
def stochastic_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.01,
    epochs: int = 20,
    tol: float = 1e-4,
    l2_lambda: float = 0.0,
    random_state: int = 50,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """[Bonus] Train One-vs-All logistic regression weights using Stochastic Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    rng = np.random.default_rng(random_state)
    history = TrainingHistory({m: [] for m in maisons})

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])
        history.early_stopped[m] = None
        stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

        for epoch in range(epochs):
            # 每個 Epoch 隨機打亂樣本順序，避免模型記住固定順序
            indices = rng.permutation(m_samples)
            for idx in indices:
                xi = X[idx]
                yi = y[idx]

                z = float(xi @ w + b)
                p = float(sigmoid(np.array([z]))[0])
                e = p - yi

                grads = xi * e + (l2_lambda / m_samples) * w
                grad_bias = e

                w -= learning_rate * grads
                b -= learning_rate * grad_bias

            curr_loss = compute_log_loss(X, y, w, b, l2_lambda, m_samples)
            history[m].append(curr_loss)

            if stopper and stopper.update(curr_loss, w, b, epoch + 1):
                w, b = stopper.restore_best(w, b)
                history.early_stopped[m] = epoch + 1
                print(f"[SGD] Early stopping reached at epoch {epoch + 1} ({m}): loss = {stopper.best_loss:.5f}")
                break

            # Epoch 結束後檢查整體梯度是否已收斂
            full_e = sigmoid(X @ w + b) - y
            full_grads = (X.T @ full_e) / m_samples + (l2_lambda / m_samples) * w
            full_grad_bias = float(sum(full_e)) / m_samples
            gradient_max = max(float(max(abs(g) for g in full_grads)), abs(full_grad_bias))
            if gradient_max < tol:
                print(f"[SGD] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    if return_history:
        return weights, bias, history
    return weights, bias


# =====================================================================
# 3. Mini-batch Gradient Descent (Mini) - 小批次梯度下降 [Bonus]
#    結合 BGD 與 SGD 的優點：每個 Epoch 洗牌後，每次抽取一小批樣本（如 batch_size = 32）
#    計算該批次的平均梯度並更新權重。
#    優點：既能利用矩陣向量化加速，又比 BGD 更新更頻繁、比 SGD 更穩定。
# =====================================================================
def mini_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.05,
    epochs: int = 50,
    batch_size: int = 32,
    tol: float = 1e-4,
    l2_lambda: float = 0.0,
    random_state: int = 50,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """[Bonus] Train One-vs-All logistic regression weights using Mini-batch Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    rng = np.random.default_rng(random_state)
    history = TrainingHistory({m: [] for m in maisons})

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])
        history.early_stopped[m] = None
        stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

        for epoch in range(epochs):
            indices = rng.permutation(m_samples)
            for start in range(0, m_samples, batch_size):
                # 將資料切成每32個一組，分批計算。
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                b_len = len(X_batch)

                z = X_batch @ w + b
                p = sigmoid(z)
                e = p - y_batch

                grads = (X_batch.T @ e) / b_len + (l2_lambda / m_samples) * w
                grad_bias = float(sum(e)) / b_len

                w -= learning_rate * grads
                b -= learning_rate * grad_bias

            curr_loss = compute_log_loss(X, y, w, b, l2_lambda, m_samples)
            history[m].append(curr_loss)

            if stopper and stopper.update(curr_loss, w, b, epoch + 1):
                w, b = stopper.restore_best(w, b)
                history.early_stopped[m] = epoch + 1
                print(f"[Mini-batch GD] Early stopping reached at epoch {epoch + 1} ({m}): loss = {stopper.best_loss:.5f}")
                break

            full_e = sigmoid(X @ w + b) - y
            full_grads = (X.T @ full_e) / m_samples + (l2_lambda / m_samples) * w
            full_grad_bias = float(sum(full_e)) / m_samples
            gradient_max = max(float(max(abs(g) for g in full_grads)), abs(full_grad_bias))
            if gradient_max < tol:
                print(f"[Mini-batch GD] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    if return_history:
        return weights, bias, history
    return weights, bias


# =====================================================================
# 4. Gradient Descent with Momentum - 動量梯度下降法 [Bonus]
#    模擬物理學中「鐵球滾下山坡」的動量慣性 (Velocity v_t)：
#    v_t = beta * v_{t-1} + (1 - beta) * grad
#    theta = theta - lr * v_t
#    優點：能累積同方向的加速度衝過平坦高原區，並抵消狹窄山谷的左右震盪。
# =====================================================================
def momentum_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.1,
    epochs: int = 100,
    beta: float = 0.9,
    tol: float = 1e-4,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """[Bonus] Train One-vs-All logistic regression weights using Gradient Descent with Momentum."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    history = TrainingHistory({m: [] for m in maisons})

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])
        history.early_stopped[m] = None
        stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

        # 初始化動量向量 (Velocity) 為 0
        v_w = np.zeros_like(w)
        v_b = 0.0

        for epoch in range(epochs):
            z = X @ w + b
            p = sigmoid(z)
            e = p - y

            grads = (X.T @ e) / m_samples + (l2_lambda / m_samples) * w
            grad_bias = float(sum(e)) / m_samples

            # 針對梯度累積動量
            v_w = beta * v_w + (1.0 - beta) * grads
            v_b = beta * v_b + (1.0 - beta) * grad_bias

            w -= learning_rate * v_w
            b -= learning_rate * v_b

            curr_loss = compute_log_loss(X, y, w, b, l2_lambda, m_samples)
            history[m].append(curr_loss)

            if stopper and stopper.update(curr_loss, w, b, epoch + 1):
                w, b = stopper.restore_best(w, b)
                history.early_stopped[m] = epoch + 1
                print(f"[Momentum] Early stopping reached at epoch {epoch + 1} ({m}): loss = {stopper.best_loss:.5f}")
                break

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[Momentum] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    if return_history:
        return weights, bias, history
    return weights, bias


# =====================================================================
# 5. Adam (Adaptive Moment Estimation) - 自適應矩估計優化器 [Bonus]
#    結合「一階動差 (Momentum 方向慣性 m_t)」與「二階動差 (RMSProp 自適應步長 v_t)」，
#    並進行偏差校正 (Bias Correction)，為現代機器學習與深度學習最主流的優化演算法。
#    優點：自動為每個特徵量身調整學習率，收斂速度極快且極為穩健。
# =====================================================================
def adam_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.05,
    epochs: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-4,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """[Bonus] Train One-vs-All logistic regression weights using the Adam optimizer."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    history = TrainingHistory({m: [] for m in maisons})

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])
        history.early_stopped[m] = None
        stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None

        # 初始化一階動差 (m) 與二階動差 (v)
        m_w = np.zeros_like(w)
        v_w = np.zeros_like(w)
        m_b = 0.0
        v_b = 0.0

        for t in range(1, epochs + 1):
            z = X @ w + b
            p = sigmoid(z)
            e = p - y

            grads = (X.T @ e) / m_samples + (l2_lambda / m_samples) * w
            grad_bias = float(sum(e)) / m_samples

            # 更新一階動差（梯度平均方向）與二階動差（梯度平方平均幅度）
            m_w = beta1 * m_w + (1.0 - beta1) * grads
            v_w = beta2 * v_w + (1.0 - beta2) * (grads ** 2)
            m_b = beta1 * m_b + (1.0 - beta1) * grad_bias
            v_b = beta2 * v_b + (1.0 - beta2) * (grad_bias ** 2)

            # 偏差校正 (Bias correction)，修正初期偏向 0 的問題
            m_w_hat = m_w / (1.0 - beta1 ** t)
            v_w_hat = v_w / (1.0 - beta2 ** t)
            m_b_hat = m_b / (1.0 - beta1 ** t)
            v_b_hat = v_b / (1.0 - beta2 ** t)

            w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            b -= learning_rate * m_b_hat / ((v_b_hat ** 0.5) + eps)

            curr_loss = compute_log_loss(X, y, w, b, l2_lambda, m_samples)
            history[m].append(curr_loss)

            if stopper and stopper.update(curr_loss, w, b, t):
                w, b = stopper.restore_best(w, b)
                history.early_stopped[m] = t
                print(f"[Adam] Early stopping reached at epoch {t} ({m}): loss = {stopper.best_loss:.5f}")
                break

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[Adam] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    if return_history:
        return weights, bias, history
    return weights, bias


# =====================================================================
# 統一接口函式 (Unified Dispatcher)
# 讓 logreg_train.py 只需呼叫 optimize(..., opt="bgd" | "sgd" | "mini" | "momentum" | "adam")
# =====================================================================
def optimize(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    opt: str = "bgd",
    learning_rate: float | None = None,
    epochs: int | None = None,
    batch_size: int = 32,
    l2_lambda: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    return_history: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, TrainingHistory]:
    """Dispatch training to the selected optimization algorithm."""
    opt_name = opt.lower().strip()
    if opt_name in ("bgd", "batch"):
        lr = 0.07 if learning_rate is None else learning_rate
        ep = 100 if epochs is None else epochs
        return batch_gd(
            hold_df,
            weights,
            bias,
            learning_rate=lr,
            epochs=ep,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=return_history,
        )
    if opt_name == "sgd":
        lr = 0.01 if learning_rate is None else learning_rate
        ep = 20 if epochs is None else epochs
        return stochastic_gd(
            hold_df,
            weights,
            bias,
            learning_rate=lr,
            epochs=ep,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=return_history,
        )
    if opt_name == "mini":
        lr = 0.05 if learning_rate is None else learning_rate
        ep = 50 if epochs is None else epochs
        return mini_gd(
            hold_df,
            weights,
            bias,
            learning_rate=lr,
            epochs=ep,
            batch_size=batch_size,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=return_history,
        )
    if opt_name == "momentum":
        lr = 0.1 if learning_rate is None else learning_rate
        ep = 100 if epochs is None else epochs
        return momentum_gd(
            hold_df,
            weights,
            bias,
            learning_rate=lr,
            epochs=ep,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=return_history,
        )
    if opt_name == "adam":
        lr = 0.05 if learning_rate is None else learning_rate
        ep = 100 if epochs is None else epochs
        return adam_gd(
            hold_df,
            weights,
            bias,
            learning_rate=lr,
            epochs=ep,
            l2_lambda=l2_lambda,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            return_history=return_history,
        )
    raise ValueError(
        f"Unknown optimizer '{opt}'. Choose from: 'bgd', 'sgd', 'mini', 'momentum', 'adam'."
    )
