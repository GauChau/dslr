import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Compute the sigmoid function 1 / (1 + exp(-z)) with overflow protection."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


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
    learning_rate: float = 0.07,
    epochs: int = 100,
    tol: float = 1e-4,
) -> tuple[pd.DataFrame, pd.Series]:
    """Train One-vs-All logistic regression weights using Batch Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])

        for _ in range(epochs):
            z = X @ w + b
            p = sigmoid(z)
            e = p - y

            grads = (X.T @ e) / m_samples
            grad_bias = float(sum(e)) / m_samples

            w -= learning_rate * grads
            b -= learning_rate * grad_bias

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[BGD] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    return weights, bias


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
    random_state: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """[Bonus] Train One-vs-All logistic regression weights using Stochastic Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    rng = np.random.default_rng(random_state)

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])

        for _ in range(epochs):
            # 每個 Epoch 隨機打亂樣本順序，避免模型記住固定順序
            indices = rng.permutation(m_samples)
            for idx in indices:
                xi = X[idx]
                yi = y[idx]

                z = float(xi @ w + b)
                p = float(sigmoid(np.array([z]))[0])
                e = p - yi

                grads = xi * e
                grad_bias = e

                w -= learning_rate * grads
                b -= learning_rate * grad_bias

            # Epoch 結束後檢查整體梯度是否已收斂
            full_e = sigmoid(X @ w + b) - y
            full_grads = (X.T @ full_e) / m_samples
            full_grad_bias = float(sum(full_e)) / m_samples
            gradient_max = max(float(max(abs(g) for g in full_grads)), abs(full_grad_bias))
            if gradient_max < tol:
                print(f"[SGD] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    return weights, bias


# =====================================================================
# 3. Mini-batch Gradient Descent (MBGD) - 小批次梯度下降 [Bonus]
#    結合 BGD 與 SGD 的優點：每個 Epoch 洗牌後，每次抽取一小批樣本（如 batch_size = 32）
#    計算該批次的平均梯度並更新權重。
#    優點：既能利用矩陣向量化加速，又比 BGD 更新更頻繁、比 SGD 更穩定。
# =====================================================================
def minibatch_gd(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    learning_rate: float = 0.05,
    epochs: int = 50,
    batch_size: int = 32,
    tol: float = 1e-4,
    random_state: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """[Bonus] Train One-vs-All logistic regression weights using Mini-batch Gradient Descent."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)
    rng = np.random.default_rng(random_state)

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])

        for _ in range(epochs):
            indices = rng.permutation(m_samples)
            for start in range(0, m_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                b_len = len(X_batch)

                z = X_batch @ w + b
                p = sigmoid(z)
                e = p - y_batch

                grads = (X_batch.T @ e) / b_len
                grad_bias = float(sum(e)) / b_len

                w -= learning_rate * grads
                b -= learning_rate * grad_bias

            full_e = sigmoid(X @ w + b) - y
            full_grads = (X.T @ full_e) / m_samples
            full_grad_bias = float(sum(full_e)) / m_samples
            gradient_max = max(float(max(abs(g) for g in full_grads)), abs(full_grad_bias))
            if gradient_max < tol:
                print(f"[Mini-batch GD] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

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
) -> tuple[pd.DataFrame, pd.Series]:
    """[Bonus] Train One-vs-All logistic regression weights using Gradient Descent with Momentum."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])

        # 初始化動量向量 (Velocity) 為 0
        v_w = np.zeros_like(w)
        v_b = 0.0

        for _ in range(epochs):
            z = X @ w + b
            p = sigmoid(z)
            e = p - y

            grads = (X.T @ e) / m_samples
            grad_bias = float(sum(e)) / m_samples

            # 指數移動平均累積梯度動量
            v_w = beta * v_w + (1.0 - beta) * grads
            v_b = beta * v_b + (1.0 - beta) * grad_bias

            w -= learning_rate * v_w
            b -= learning_rate * v_b

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[Momentum] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

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
) -> tuple[pd.DataFrame, pd.Series]:
    """[Bonus] Train One-vs-All logistic regression weights using the Adam optimizer."""
    weights = weights.copy()
    bias = bias.copy()
    matieres = weights.index
    maisons = weights.columns

    X = hold_df.loc[:, matieres].to_numpy(dtype=float)
    m_samples = len(X)

    for m in maisons:
        y = hold_df[m].to_numpy(dtype=float)
        w = weights[m].to_numpy(dtype=float).copy()
        b = float(bias.loc[m])

        # 初始化一階動差 (m) 與二階動差 (v)
        m_w = np.zeros_like(w)
        v_w = np.zeros_like(w)
        m_b = 0.0
        v_b = 0.0

        for t in range(1, epochs + 1):
            z = X @ w + b
            p = sigmoid(z)
            e = p - y

            grads = (X.T @ e) / m_samples
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

            gradient_max = max(float(max(abs(g) for g in grads)), abs(grad_bias))
            if gradient_max < tol:
                print(f"[Adam] Convergence reached ({m})")
                break

        weights[m] = w
        bias.loc[m] = b

    return weights, bias


# =====================================================================
# 統一入口函式 (Unified Dispatcher)
# 讓 logreg_train.py 只需呼叫 optimize(..., opt="bgd" | "sgd" | "minibatch" | "momentum" | "adam")
# =====================================================================
def optimize(
    hold_df: pd.DataFrame,
    weights: pd.DataFrame,
    bias: pd.Series,
    opt: str = "bgd",
    learning_rate: float = 0.07,
    epochs: int = 100,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatch training to the selected optimization algorithm."""
    opt_name = opt.lower().strip()
    if opt_name in ("bgd", "batch"):
        return batch_gd(hold_df, weights, bias, learning_rate=learning_rate, epochs=epochs)
    if opt_name == "sgd":
        return stochastic_gd(hold_df, weights, bias, learning_rate=learning_rate, epochs=epochs)
    if opt_name in ("minibatch", "mb", "mbgd"):
        return minibatch_gd(
            hold_df, weights, bias, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size
        )
    if opt_name == "momentum":
        return momentum_gd(hold_df, weights, bias, learning_rate=learning_rate, epochs=epochs)
    if opt_name == "adam":
        return adam_gd(hold_df, weights, bias, learning_rate=learning_rate, epochs=epochs)
    raise ValueError(
        f"Unknown optimizer '{opt}'. Choose from: 'bgd', 'sgd', 'minibatch', 'momentum', 'adam'."
    )
