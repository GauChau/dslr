# DSLR - Data Science × Logistic Regression <a id="dslr-ds-lr"></a>

<p align="right">
  <a href="#dslr-資料科學與邏輯回歸">
    <img src="https://img.shields.io/badge/中文-sienna?style=for-the-badge" />
  </a>
</p>

> **Harry Potter and the Data Scientist** — Recreate the Hogwarts Sorting Hat!

---

## 📖 Table of Contents
- [1. Project Overview](#1-project-overview)
  - [1.1 Story Background](#11-story-background)
  - [1.2 Project Objectives](#12-project-objectives)
- [2. What There Is to Learn](#2-what-there-is-to-learn)
  - [2.1 Exploratory Data Analysis & Feature Engineering](#21-exploratory-data-analysis--feature-engineering)
  - [2.2 Descriptive Statistics from Scratch](#22-descriptive-statistics-from-scratch)
  - [2.3 Logistic Regression & Optimization](#23-logistic-regression--optimization)
  - [2.4 One-vs-All Multi-Class Classification](#24-one-vs-all-multi-class-classification)
- [3. Rules & Constraints](#3-rules--constraints)
- [4. How We Implement It](#4-how-we-implement-it)
  - [4.1 Project Directory Structure](#41-project-directory-structure)
  - [4.2 Module Responsibilities](#42-module-responsibilities)
- [5. Step-by-Step Implementation Guide](#5-step-by-step-implementation-guide)
  - [Part 1: Descriptive Statistics (`describe.py`)](#part-1-descriptive-statistics-describepy)
  - [Part 2: Data Visualization (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)](#part-2-data-visualization-histogrampy-scatter_plotpy-pair_plotpy)
  - [Part 3: Model Training & Prediction (`logreg_train.py`, `logreg_predict.py`)](#part-3-model-training--prediction-logreg_trainpy-logreg_predictpy)
- [6. Mathematical Foundations](#6-mathematical-foundations)
  - [6.1 Sigmoid Function](#61-sigmoid-function)
  - [6.2 Cost Function (Binary Cross-Entropy)](#62-cost-function-binary-cross-entropy)
  - [6.3 Gradient Derivation & Weight Update (Gradient Descent)](#63-gradient-derivation--weight-update-gradient-descent)
  - [6.4 Feature Standardization (Z-score)](#64-feature-standardization-z-score)
- [7. Bonus Part](#7-bonus-part)
- [8. Getting Started](#8-getting-started)

---

## 1. Project Overview

### 1.1 Story Background
The famous **Sorting Hat** at Hogwarts School of Witchcraft and Wizardry has been bewitched by dark forces and no longer responds! With the new academic year approaching, Professor McGonagall and the Headmaster have called upon you—a Muggle "data scientist"—to recreate a magical Sorting Hat using a "computer." With a quick *"Digitalis!"* spell, an ancient, dusty spellbook of student records and course grades is transformed into CSV datasets on a USB stick.

### 1.2 Project Objectives
The goal of this project is to build a complete data analysis and classification pipeline **from scratch without relying on high-level machine learning models or built-in statistical aggregators** (such as `sklearn.linear_model.LogisticRegression` or `pandas.DataFrame.describe()`):
1. **Descriptive Statistics Tool**: Compute and display formatted summary statistics for all numerical features.
2. **Data Visualization**: Plot histograms, scatter plots, and pair plots to answer analytical questions and select the most informative features.
3. **Logistic Regression Classifier**: Implement a **One-vs-All (One-vs-Rest)** multi-class classifier trained via gradient descent to sort students into the four Hogwarts houses (`Gryffindor`, `Hufflepuff`, `Ravenclaw`, `Slytherin`), achieving **$\ge 98\%$ accuracy** on the test set.

---

## 2. What There Is to Learn

By completing this project, you will master the foundational pillars of modern machine learning:

### 2.1 Exploratory Data Analysis & Feature Engineering
- **Exploratory Data Analysis (EDA)**: Inspect data types, value ranges, missing value (`NaN`) distributions, and outliers.
- **Feature Selection**: Use statistical distributions and visual plots to identify uninformative homogeneous features (courses where all four houses have identical score distributions) and redundant collinear features (pairs of courses with strong linear correlation), removing noise to improve generalization.
- **Data Preprocessing**: Master missing value imputation strategies (such as median imputation) and feature standardization (Z-score normalization), understanding why scaling is essential for gradient descent convergence.

### 2.2 Descriptive Statistics from Scratch
- Look inside the black box of high-level libraries by writing manual algorithms for key statistical metrics:
  - Valid sample count ($Count$)
  - Arithmetic mean ($Mean$)
  - Unbiased sample standard deviation ($Std$, using degrees of freedom $N - 1$)
  - Extrema ($Min$, $Max$)
  - Quartiles ($25\%$, $50\%$, $75\%$): Understand quantile interpolation algorithms (such as linear interpolation).

### 2.3 Logistic Regression & Optimization
- Understand Generalized Linear Models (GLMs) and how the **Sigmoid function** maps linear combinations into $[0, 1]$ probability estimates.
- Understand the mathematical nature and convexity of the **Binary Cross-Entropy (Log Loss)** cost function.
- Derive the gradient vector and implement **Batch Gradient Descent (BGD)** from scratch, tuning the learning rate ($\alpha$) and training epochs.

### 2.4 One-vs-All Multi-Class Classification
- Learn how binary classifiers extend to multi-class classification problems.
- Train 4 independent binary logistic regression classifiers—one for each Hogwarts house (e.g., *Gryffindor vs. Not Gryffindor*).
- During inference, evaluate the probability output of all 4 classifiers and pick the class with the highest probability ($\arg\max$).

---

## 3. Rules & Constraints

> **No Heavy Lifting Allowed**
> - In `describe.py` and across the project, **it is strictly forbidden** to use built-in functions that do the statistical heavy lifting for you (e.g., `df.describe()`, `np.mean()`, `np.std()`, `np.min()`, `np.max()`, `np.percentile()`, etc.).
> - In model training and prediction, **it is forbidden** to use high-level estimators like `sklearn.linear_model.LogisticRegression`.
> - Your final predictions on `dataset_test.csv` must reach an accuracy score of **$\ge 98\%$** (evaluated using `sklearn.metrics.accuracy_score`).

---

## 4. How We Implement It

### 4.1 Project Directory Structure
```bash
dslr/
├── docs/
│   ├── en.subject.pdf          # Official 42 Subject specification
│   └── TODO.md                 # Project audit & Bonus implementation roadmap
├── data/
│   ├── dataset_train.csv       # Training dataset (with Hogwarts House labels)
│   └── dataset_test.csv        # Test dataset (for prediction evaluation)
├── src/
│   ├── describe.py             # Mandatory: From-scratch summary statistics CLI
│   ├── histogram.py            # Mandatory: Histogram answering course homogeneity
│   ├── scatter_plot.py         # Mandatory: Scatter plot answering feature similarity
│   ├── pair_plot.py            # Mandatory: Scatter plot matrix for feature selection
│   ├── logreg_train.py         # Mandatory: Train One-vs-All model & save weights
│   ├── logreg_predict.py       # Mandatory: Load weights & generate houses.csv
│   └── utils/
│       ├── __init__.py
│       ├── stats.py            # From-scratch statistical functions (mean, std, quartiles, etc.)
│       ├── preprocessor.py     # Median imputation & Z-score standardization
│       └── model.py            # Logistic Regression core (Sigmoid, Loss, BGD, SGD, MBGD)
├── weights.csv                 # Trained model weights & scaling parameters
├── houses.csv                  # Final generated house predictions
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

### 4.2 Module Responsibilities
- **`src/utils/stats.py`**: Implements pure from-scratch statistical functions on numerical sequences, shared by `describe.py` and the preprocessing pipeline.
- **`src/utils/preprocessor.py`**: Stores training set statistics (medians, means, standard deviations) and applies the exact same transformation to the test set to prevent **Data Leakage**.
- **`src/utils/model.py`**: Encapsulates the One-vs-All Logistic Regression classifier, supporting Batch Gradient Descent (and Bonus optimizers: SGD, Mini-batch GD, $L_2$ regularization) and loss history tracking.

---

## 5. Step-by-Step Implementation Guide

### Part 1: Descriptive Statistics (`describe.py`)
- **Goal**: Read a CSV dataset, filter numerical columns, compute summary statistics from scratch, and display a neatly aligned table.
- **Metrics Computed**:
  - `Count`: Number of valid non-`NaN` entries.
  - `Mean`: $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$.
  - `Std`: $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2}$ (unbiased sample standard deviation).
  - `Min` / `Max`: Single-pass $O(N)$ traversal to find extrema.
  - `25%` / `50%` / `75%`: Sort valid values in ascending order and apply linear interpolation:
    $$\text{index} = (N - 1) \times p$$
    $$Q_p = x_{\lfloor \text{index} \rfloor} + (\text{index} - \lfloor \text{index} \rfloor) \times (x_{\lceil \text{index} \rceil} - x_{\lfloor \text{index} \rfloor})$$
- **Formatting**: Adaptive column widths with all values formatted to 6 decimal places (`%.6f`).

### Part 2: Data Visualization (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)

#### 1. `histogram.py`
- **Core Question**: *Which Hogwarts course has a homogeneous score distribution between all four houses?*
- **Approach**: Plot overlapping translucent histograms (`alpha=0.5`) of student grades grouped by the 4 houses across all courses.
- **Conclusion**: Identify courses where the 4 house distributions overlap almost completely (specifically **`Care of Magical Creatures`** and **`Arithmancy`**). Because these courses provide no discriminative power between houses, they should be excluded from training.

#### 2. `scatter_plot.py`
- **Core Question**: *What are the two features that are similar?*
- **Approach**: Compute the pairwise Pearson correlation coefficient from scratch across all courses, find the pair with the strongest linear correlation ($|r| \to 1$), and plot their 2D scatter plot colored by house.
- **Conclusion**: **`Astronomy`** and **`Defense Against the Dark Arts`** exhibit a perfect negative linear correlation ($r \approx -1.0$). Highly collinear features carry duplicate information; keeping only one reduces dimensionality and avoids multicollinearity.

#### 3. `pair_plot.py`
- **Core Question**: *From this visualization, which features are you going to use for your logistic regression?*
- **Approach**: Render a full Scatter Plot Matrix (Pair Plot) with house-colored histograms on the diagonal and 2D scatter plots off-diagonal.
- **Conclusion**:
  - Select courses where house clusters show clear separation.
  - Drop homogeneous noise features (`Care of Magical Creatures`, `Arithmancy`) and redundant collinear features (`Defense Against the Dark Arts`).

### Part 3: Model Training & Prediction (`logreg_train.py`, `logreg_predict.py`)

#### 1. `logreg_train.py`
- **Input**: `dataset_train.csv`.
- **Pipeline**:
  1. **Data Cleaning**: Drop non-numerical metadata (`First Name`, `Last Name`, `Birthday`, `Best Hand`) and excluded courses.
  2. **Missing Value Imputation**: Compute the **median** ($50\%$ quantile) of each selected feature on the training set and fill missing (`NaN`) values.
  3. **Feature Standardization**: Compute each feature's $\mu$ and $\sigma$, transforming values into standard normal Z-scores.
  4. **One-vs-All Label Encoding**: Convert `Hogwarts House` into 4 binary target vectors ($y \in \{0, 1\}$).
  5. **Gradient Descent Optimization**: Prepend a bias column of ones ($x_0 = 1$) and iteratively update the weight matrix $\theta$ using Batch Gradient Descent.
  6. **Save Parameters**: Export the trained weights, biases, and scaling parameters ($\text{median}, \mu, \sigma$) to `weights.csv`.

> #### 💡 Deep Dive: Why Impute Missing Values with the **Median**?
> Common strategies for handling `NaN` grades include dropping rows, mean imputation, mode imputation, and median imputation. Choosing the **median** is optimal for four key reasons:
> 1. **Robustness to Outliers**:
>    Course grades may contain extreme scores or data anomalies. While the arithmetic mean is easily skewed by extreme values, the median (the 50th percentile) remains highly stable and accurately reflects the central tendency.
> 2. **Handling Skewed Distributions**:
>    Real-world grade distributions are rarely perfectly symmetric Gaussians. Under left- or right-skewed distributions, the mean is pulled toward the long tail, whereas the median better represents a typical student's performance.
> 3. **Preserving Sample Size**:
>    If we drop every row containing at least one `NaN` across 13 courses (each missing 2%–5% of entries), we lose 30%–50% of our training samples—severely degrading generalization and risking accuracy dropping below the 98% requirement. Also, `dataset_test.csv` contains `NaN` values that *must* be predicted and cannot be dropped.
> 4. **Inapplicability of Mode for Continuous Data**:
>    Course scores are continuous floating-point numbers with very few exact duplicates, making the mode statistically meaningless.

#### 2. `logreg_predict.py`
- **Input**: `dataset_test.csv` and `weights.csv`.
- **Pipeline**:
  1. Load the test dataset, select the active features, impute `NaN` values using the saved training medians, and standardize using the saved training $\mu$ and $\sigma$.
  2. Compute the predicted probability across all 4 house classifiers for each student:
     $$P(y = k \mid x) = \sigma(\theta_k^T x), \quad k \in \{\text{Gryffindor}, \text{Hufflepuff}, \text{Ravenclaw}, \text{Slytherin}\}$$
  3. Assign each student to the house with the highest probability:
     $$\hat{y} = \arg\max_{k} P(y = k \mid x)$$
  4. Write the predictions to `houses.csv` matching the exact required format:
     ```csv
     Index,Hogwarts House
     0,Gryffindor
     1,Hufflepuff
     ...
     ```

---

## 6. Mathematical Foundations

### 6.1 Sigmoid Function
Logistic regression applies the logistic (Sigmoid) function to map any real-valued linear combination into a probability in $(0, 1)$:
$$g(z) = \frac{1}{1 + e^{-z}}$$

Model hypothesis:
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Its first derivative satisfies the elegant identity:
$$g'(z) = g(z)(1 - g(z))$$

### 6.2 Cost Function (Binary Cross-Entropy)
Given $m$ training samples, the cost function for binary logistic regression is the Binary Cross-Entropy (Log Loss):
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \Big]$$

- When the true label is $y=1$ and $h_\theta(x) \to 0$, the loss approaches $\infty$.
- When the true label is $y=0$ and $h_\theta(x) \to 1$, the loss approaches $\infty$.
- This cost function is strictly **convex** in parameter space, guaranteeing no local minima so gradient descent converges to the global optimum.

> #### 💡 Deep Dive: Why Binary Cross-Entropy Instead of Mean Squared Error (MSE)?
> Linear regression uses Mean Squared Error (MSE), but logistic regression requires Binary Cross-Entropy for three fundamental mathematical reasons:
> 1. **Guaranteed Convexity**:
>    - Plugging the non-linear Sigmoid hypothesis $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$ into MSE produces a **non-convex** loss surface riddled with local minima and flat plateaus where gradient descent gets trapped.
>    - In contrast, Binary Cross-Entropy paired with Sigmoid yields a positive semi-definite Hessian matrix, proving it is strictly **convex** (bowl-shaped) so any local minimum is the global minimum.
> 2. **Maximum Likelihood Estimation (MLE) Derivation**:
>    - Logistic regression models conditional Bernoulli probabilities: $P(y \mid x) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$.
>    - Assuming independent and identically distributed (i.i.d.) samples, taking the Negative Log-Likelihood (NLL) of the joint likelihood $L(\theta) = \prod_{i=1}^m P(y^{(i)} \mid x^{(i)})$ divided by $m$ directly yields the Binary Cross-Entropy formula. **Minimizing cross-entropy is mathematically equivalent to maximizing the likelihood of the observed data.**
> 3. **Preventing Gradient Vanishing**:
>    - Under MSE, when a prediction is confidently wrong (e.g., $z = \theta^T x \gg 0$ while $y=0$), the Sigmoid derivative $\sigma'(z) \approx 0$ causes **gradient vanishing**, stalling weight updates.
>    - With Cross-Entropy, the logarithm cancels the exponential derivative of Sigmoid, yielding a clean linear gradient proportional to the prediction error $(h_\theta(x) - y)$: the larger the error, the stronger the corrective gradient!

### 6.3 Gradient Derivation & Weight Update (Gradient Descent)
Partial derivative of the cost function with respect to $\theta_j$:
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

Vectorized matrix form:
$$\nabla_\theta J(\theta) = \frac{1}{m} X^T \left( g(X\theta) - y \right)$$

Simultaneous weight update rule with learning rate $\alpha$:
$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$

### 6.4 Feature Standardization (Z-score)
Because Hogwarts courses have vastly different score scales (e.g., `Arithmancy` in the tens of thousands vs. `Herbology` between $-10$ and $10$), unscaled features create elongated elliptical loss contours that cause gradient descent to oscillate or diverge.
Standardization formula:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
where $\mu$ is the feature mean and $\sigma$ is the sample standard deviation.
> **Note**: The test set must be standardized using the **training set's** $\mu$ and $\sigma$ to prevent Data Leakage.

---

## 7. Bonus Part

Once the Mandatory Part is 100% compliant and achieving $\ge 98\%$ accuracy, the following Bonus features are supported:

1. **Extended Statistical Metrics in `describe.py`**:
   - Adds sample Variance ($s^2$), Interquartile Range ($IQR$), Range ($Max - Min$), Skewness ($g_1$), Excess Kurtosis ($g_2$), Missing value percentage ($Missing\%$), and Unique count ($Unique$).
2. **Multiple Optimization Algorithms**:
   - **Stochastic Gradient Descent (SGD)**: Updates weights using 1 randomly shuffled sample per step for rapid online updates.
   - **Mini-Batch Gradient Descent (MBGD)**: Updates weights on small shuffled batches (e.g., `batch_size=32`), balancing matrix vectorization speed with convergence stability.
3. **$L_2$ Regularization (Ridge)**:
   - Penalizes large weights to prevent overfitting:
     $$J_{\text{reg}}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$
4. **Learning Curve Visualization & Early Stopping**:
   - Stops training early when loss plateaus and plots **Cost vs. Epochs** comparing BGD, Mini-batch GD, and SGD convergence.

---

## 8. Getting Started

### 8.1 Environment Setup
```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 8.2 Execution Commands

#### 1. Descriptive Statistics
```bash
python3 src/describe.py data/dataset_train.csv
```

#### 2. Data Visualization
```bash
# Inspect which course has a homogeneous distribution across all four houses
python3 src/histogram.py data/dataset_train.csv

# Inspect which two features are strongly correlated
python3 src/scatter_plot.py data/dataset_train.csv

# Plot the scatter plot matrix for feature selection
python3 src/pair_plot.py data/dataset_train.csv
```

#### 3. Model Training & Prediction
```bash
# Train the One-vs-All logistic regression model and output weights.csv
python3 src/logreg_train.py data/dataset_train.csv

# Predict houses on dataset_test.csv and generate houses.csv
python3 src/logreg_predict.py data/dataset_test.csv weights.csv
```

#### 4. Accuracy Verification
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

---

## 📜 Evaluation Criteria Summary
- **Rule Compliance**: Zero forbidden built-in statistical or ML functions used for heavy lifting.
- **Visualizations & Feature Justification**: Clear plots and sound reasoning for feature selection.
- **Classification Accuracy**: $\ge 98\%$ accuracy on `dataset_test.csv`.
- **Peer Defense**: Ability to explain the gradient derivation, Sigmoid properties, Z-score standardization, and One-vs-All classification.

---

<br/>

---

<br/>

<p align="right">
  <a href="#dslr-ds-lr">
    <img src="https://img.shields.io/badge/-TOP-sienna?style=for-the-badge" />
  </a>
</p>

# DSLR 資料科學與邏輯回歸

> **Harry Potter and the Data Scientist** — 編寫霍格華茲的分類帽！

---

## 📖 目錄 (Table of Contents)
- [1. 專案簡介 (Project Overview)](#1-專案簡介-project-overview)
  - [1.1 故事背景](#11-故事背景)
  - [1.2 任務目標](#12-任務目標)
- [2. 學習目標與核心概念 (What There Is to Learn)](#2-學習目標與核心概念-what-there-is-to-learn)
  - [2.1 資料探索與特徵工程 (EDA & Feature Engineering)](#21-資料探索與特徵工程-eda--feature-engineering)
  - [2.2 手刻基礎統計指標 (Descriptive Statistics from Scratch)](#22-手刻基礎統計指標-descriptive-statistics-from-scratch)
  - [2.3 邏輯回歸與梯度下降 (Logistic Regression & Optimization)](#23-邏輯回歸與梯度下降-logistic-regression--optimization)
  - [2.4 多類別分類 (One-vs-All Classification)](#24-多類別分類-one-vs-all-classification)
- [3. 規範與限制 (Rules & Constraints)](#3-規範與限制-rules--constraints)
- [4. 實作架構設計 (How We Implement It)](#4-實作架構設計-how-we-implement-it)
  - [4.1 專案目錄結構](#41-專案目錄結構)
  - [4.2 模組職責劃分](#42-模組職責劃分)
- [5. 各階段詳細實作指南 (Step-by-Step Implementation)](#5-各階段詳細實作指南-step-by-step-implementation)
  - [Part 1: 描述性統計 `describe.py`](#part-1-描述性統計-describepy)
  - [Part 2: 資料視覺化分析 (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)](#part-2-資料視覺化分析-histogrampy-scatter_plotpy-pair_plotpy)
  - [Part 3: 訓練與預測 (`logreg_train.py`, `logreg_predict.py`)](#part-3-訓練與預測-logreg_trainpy-logreg_predictpy)
- [6. 數學原理與公式推導 (Mathematical Foundations)](#6-數學原理與公式推導-mathematical-foundations)
  - [6.1 Sigmoid 函數](#61-sigmoid-函數)
  - [6.2 成本函數 (Cost Function / Binary Cross-Entropy)](#62-成本函數-cost-function--binary-cross-entropy)
  - [6.3 梯度推導與權重更新 (Gradient Descent)](#63-梯度推導與權重更新-梯度下降-gradient-descent)
  - [6.4 特徵標準化 (Feature Standardization / Z-score)](#64-特徵標準化-feature-standardization--z-score)
- [7. 加分項目 (Bonus Part)](#7-加分項目-bonus-part)
- [8. 安裝與執行指南 (Getting Started)](#8-安裝與執行指南-getting-started)

---

## 1. 專案簡介 (Project Overview)

### 1.1 故事背景
霍格華茲魔法學校的**分類帽（Sorting Hat）**正面臨重大危機！校長鄧不利多向麻瓜資料科學家求助，希望能透過麻瓜的「電腦」重新創造出魔法分類帽的功能。麥米奈娃教授提供了一本記載學生資料與魔法課程成績的古老魔法書（經過「數位化」轉為 CSV 檔案）。

### 1.2 任務目標
本專案目標是**完全不依賴機器學習現成高階函式庫（如 Scikit-Learn 的模型訓練、Pandas 的 `describe()` 等）**，從零手刻：
1. **資料統計工具**：計算並格式化輸出各特徵的統計數據。
2. **資料視覺化**：繪製直方圖、散布圖與配對圖矩陣，回答資料特徵問題並篩選有效特徵。
3. **邏輯回歸分類器**：實作 One-vs-All 多類別分類演算法與梯度下降，將學生準確分配至四個學院（Gryffindor、Hufflepuff、Ravenclaw、Slytherin），在測試集上達到 **98% 以上的準確率（Accuracy）**。

---

## 2. 學習目標與核心概念 (What There Is to Learn)

透過完成本專案，你將掌握現代機器學習的核心基礎：

### 2.1 資料探索與特徵工程 (EDA & Feature Engineering)
- **探索性資料分析 (Exploratory Data Analysis)**：學會判讀資料型態、數值範圍、缺失值（NaN）分布與異常值。
- **特徵篩選 (Feature Selection)**：透過統計分佈與視覺化圖表，識別無區分度的均勻特徵（如各學院分數分佈完全重疊的課程）與冗餘特徵（高度線性相關的共線性特徵），從而剔除雜訊、提升模型泛化能力。
- **資料預處理 (Data Preprocessing)**：掌握缺失值補齊策略（如以特徵中位數或均值填補）與特徵標準化（Z-score Normalization），了解為何標準化是梯度下降收斂的關鍵前提。

### 2.2 手刻基礎統計指標 (Descriptive Statistics from Scratch)
- 脫離高階工具的黑盒子，親手撰寫演算法計算各項數值統計量：
  - 樣本計數 ($Count$)
  - 算術平均數 ($Mean$)
  - 樣本標準差 ($Sample\ Standard\ Deviation$, 使用自由度 $N - 1$)
  - 極值 ($Min$, $Max$)
  - 四分位數 ($25\%$, $50\%$, $75\%$)：深入理解分位數插值演算法（如線性插值 Linear Interpolation）。

### 2.3 邏輯回歸與梯度下降 (Logistic Regression & Optimization)
- 理解廣義線性模型（Generalized Linear Models），以及如何透過 **Sigmoid 函數** 將線性輸出對應至 $[0, 1]$ 區間作為機率估計。
- 理解對數近似損失（Binary Cross-Entropy Loss）的數學本質與凸優化特性。
- 掌握梯度向量的推導過程，手刻批次梯度下降法（Batch Gradient Descent, BGD），手動調校學習率（Learning Rate $\alpha$）與迭代次數（Epochs）。

### 2.4 多類別分類 (One-vs-All Classification)
- 理解二元分類器如何擴展至多元分類問題。
- 針對 4 個學院分別訓練 4 個獨立的二元邏輯回歸模型（例如：葛來分多 vs 非葛來分多）。
- 預測階段計算各分類器的輸出機率，並取機率最高者（$\arg\max$）作為預測類別。

---

## 3. 規範與限制 (Rules & Constraints)

> **禁止調用現成函數（No Heavy Lifting）**
> - 在 `describe` 中，**禁止**使用任何現成統計函數（例如：`df.describe()`, `np.mean()`, `np.std()`, `np.min()`, `np.max()`, `np.percentile()` 等）。
> - 在模型訓練與預測中，**禁止**調用 `sklearn.linear_model.LogisticRegression` 等高階封裝模型。
> - 最終預測結果在測試集上的分類準確率必須達到 **$\ge 98\%$**（評估標準使用 `sklearn.metrics.accuracy_score`）。

---

## 4. 實作架構設計 (How We Implement It)

### 4.1 專案目錄結構
```bash
dslr/
├── docs/
│   ├── en.subject.pdf          # 題目規格說明書
│   └── TODO.md                 # 專案審查與改善待辦清單
├── data/
│   ├── dataset_train.csv       # 訓練資料集
│   └── dataset_test.csv        # 測試資料集
├── src/
│   ├── describe.py             # Mandatory: 自行計算數值特徵統計資訊
│   ├── histogram.py            # Mandatory: 回答均勻分佈課程問題
│   ├── scatter_plot.py         # Mandatory: 回答特徵相似性問題
│   ├── pair_plot.py            # Mandatory: 繪製特徵散布圖矩陣以進行特徵選擇
│   ├── logreg_train.py         # Mandatory: 訓練 One-vs-All 模型並儲存權重
│   ├── logreg_predict.py       # Mandatory: 讀取權重進行預測並輸出 houses.csv
│   └── utils/
│       ├── __init__.py
│       ├── stats.py            # 手刻統計演算法 (mean, std, quartiles, etc.)
│       ├── preprocessor.py     # 缺失值填補 (Imputation) 與特徵標準化 (Scaler)
│       └── model.py            # 邏輯回歸核心類別 (Sigmoid, Loss, BGD, SGD)
├── weights.csv                 # 訓練後輸出的權重參數
├── houses.csv                  # 最終生成的預測輸出檔
├── requirements.txt            # 相依套件清單 (numpy, pandas, matplotlib, seaborn)
└── README.md                   # 專案說明文件
```

### 4.2 模組職責劃分
- **`src/utils/stats.py`**：實現數值型序列的純手刻統計指標，提供給 `describe.py` 及標準化模組使用。
- **`src/utils/preprocessor.py`**：保存訓練集統計特性（均值、標準差、中位數），確保測試集使用與訓練集相同的尺度進行轉換（防止 Data Leakage）。
- **`src/utils/model.py`**：封裝邏輯回歸類別，支援梯度下降（可擴充支援 SGD 與 Mini-batch GD）、Sigmoid 計算與損失紀錄。

---

## 5. 各階段詳細實作指南 (Step-by-Step Implementation)

### Part 1: 描述性統計 `describe.py`
- **目的**：讀取 CSV 檔案，過濾數值型欄位，計算統計函數並整齊排版輸出。
- **輸出指標**：
  - `Count`：有效非空值數量。
  - `Mean`：$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$。
  - `Std`：$s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2}$（無偏樣本標準差）。
  - `Min` / `Max`：走訪陣列獲取極值。
  - `25%` / `50%` / `75%`：對陣列進行升序排序後，採用百分位數插值法（如線性插值）：
    $$\text{index} = (N - 1) \times p$$
    $$Q_p = x_{\lfloor \text{index} \rfloor} + (\text{index} - \lfloor \text{index} \rfloor) \times (x_{\lceil \text{index} \rceil} - x_{\lfloor \text{index} \rfloor})$$
- **輸出格式**：欄寬自適應對齊，數值精確格式化至小數點後 6 位。

### Part 2: 資料視覺化分析 (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)

#### 1. `histogram.py`
- **核心問題**：*Which Hogwarts course has a homogeneous score distribution between all four houses?*
- **作法**：對所有課程特徵繪製 4 個學院的直方圖重疊圖或子圖矩陣。
- **結論分析**：尋找在 4 個學院中分佈曲線幾乎重合、常態分佈無差異的課程（例如 `Care of Magical Creatures` 或 `Arithmancy`），這代表該課程無法提供學院分類的鑑別度，訓練時應予排除。

#### 2. `scatter_plot.py`
- **核心問題**：*What are the two features that are similar?*
- **作法**：計算特徵兩兩之間的相關係數（Pearson Correlation），找出相關係數絕對值最高（接近 1 或 -1）的兩門課程，並繪製 2D 散布圖展示其強線性關係（例如 `Astronomy` 與 `Defense Against the Dark Arts`）。
- **結論分析**：高度共線性的特徵攜帶高度重複的資訊，保留其中一個即可，避免增加模型維度與多重共線性問題。

#### 3. `pair_plot.py`
- **核心問題**：*From this visualization, which features are you going to use for your logistic regression?*
- **作法**：繪製 Pair Plot（散布圖矩陣，對角線為直方圖/KDE，非對角線為雙特徵散布圖），並以學院顏色（紅、黃、藍、綠）進行區分。
- **結論分析**：
  - 挑選在不同學院間群聚分離明顯的特徵（分界清晰）。
  - 捨棄分佈重疊嚴重（無鑑別度）或高度共線（冗餘）的特徵。

### Part 3: 訓練與預測 (`logreg_train.py`, `logreg_predict.py`)

#### 1. `logreg_train.py`
- **輸入**：`dataset_train.csv`。
- **流程**：
  1. **資料清洗**：過濾非數值欄位（姓名、生日等），剔除無效特徵。
  2. **缺失值處理（Imputation）**：計算各特徵之**中位數（Median）**並補齊遺失值。
  3. **特徵標準化**：計算各特徵之 $\mu$ 與 $\sigma$，轉換為標準常態分佈。
  4. **One-vs-All 標籤構建**：將 4 個學院標籤轉換為 4 組二元標籤矩陣（$y \in \{0, 1\}$）。
  5. **模型訓練**：增加 Bias 偏置項（插入全為 1 的欄位），使用批次梯度下降（Batch Gradient Descent）迭代更新權重向量 $\theta$。
  6. **儲存權重**：將 4 個分類器的權重與標準化參數（Median/Mean/Std）儲存至檔案（如 `weights.csv`）。

> #### 💡 深入剖析：為什麼缺失值處理採用「中位數（Median）」填補？
> 處理成績數據中的遺失值（`NaN`）時，常見策略包括「整行刪除（Drop Rows）」、「補平均值（Mean）」、「補眾數（Mode）」與「補中位數（Median）」。選擇中位數的核心考量如下：
> 1. **對極端值與異常值具備強韌性（Robustness to Outliers）**：
>    霍格華茲課程成績可能包含極端魔法分數或異常值。平均數（Mean）極易受極端值拉扯而大幅失真；中位數僅取排序後的第 50 百分位數，數值極為穩定，能客觀呈現特徵的中心趨勢（Central Tendency）。
> 2. **適應偏態分佈（Handling Skewed Distributions）**：
>    真實特徵分佈通常並非完美的對稱常態分佈，常呈現左偏或右偏。偏態分佈下平均數會被長尾拉扯，而中位數更能代表半數以上學生的常態表現水準。
> 3. **保留充足樣本數，避免資料浪費（Preserving Sample Size）**：
>    若採取直接刪除包含 `NaN` 的資料列，只要多門課程各自遺失 2%~5% 成績，全集聯集刪除後將損失 30%~50% 的珍貴訓練樣本。資料量銳減會直接劣化模型泛化能力，導致準確率無法達到 98% 門檻。
> 4. **連續數值型資料不適用眾數（Mode）**：
>    課程成績皆為連續型浮點數（Continuous Floats），各分數重複率低，眾數缺乏統計代表性（眾數通常僅適用於離散類別型特徵 Categorical Features）。

#### 2. `logreg_predict.py`
- **輸入**：`dataset_test.csv` 與權重檔。
- **流程**：
  1. 載入測試集，並套用訓練階段儲存的特徵選取與標準化參數（Median、Mean 與 Std）。
  2. 對每位學生計算 4 個二元分類器的預測機率：
     $$P(y = k \mid x) = \sigma(\theta_k^T x), \quad k \in \{\text{Gryffindor}, \text{Hufflepuff}, \text{Ravenclaw}, \text{Slytherin}\}$$
  3. 選擇機率最高者作為最終預測結果：
     $$\hat{y} = \arg\max_{k} P(y = k \mid x)$$
  4. 生成符合規格要求的 `houses.csv`：
     ```csv
     Index,Hogwarts House
     0,Gryffindor
     1,Hufflepuff
     ...
     ```

---

## 6. 數學原理與公式推導 (Mathematical Foundations)

### 6.1 Sigmoid 函數
邏輯回歸利用 Sigmoid 函數將任意實數線性組合映射為 $(0, 1)$ 之間的機率值：
$$g(z) = \frac{1}{1 + e^{-z}}$$

模型假設（Hypothesis）：
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

其一階導數具備優良的數學性質：
$$g'(z) = g(z)(1 - g(z))$$

### 6.2 成本函數 (Cost Function / Binary Cross-Entropy)
給定 $m$ 筆訓練樣本，邏輯回歸的損失函數為對數損失（交叉熵損失）：
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \Big]$$

- 當真實標籤 $y=1$ 時，若 $h_\theta(x) \to 0$，懲罰項趨近於無窮大 $\infty$。
- 當真實標籤 $y=0$ 時，若 $h_\theta(x) \to 1$，懲罰項趨近於無窮大 $\infty$。
- 此損失函數在參數空間內是凸函數（Convex），不存在局部極小值（Local Minima），能保證梯度下降收斂至全局最優解。

> #### 💡 深入剖析：為什麼邏輯回歸使用此特定的交叉熵成本函數？（Why this Cost Function?）
> 線性回歸通常使用均方誤差（MSE, Mean Squared Error），但邏輯回歸必須使用二元交叉熵損失（Binary Cross-Entropy Loss / Log Loss），核心原因包括：
> 1. **凸優化保證（Convexity vs. Non-Convexity）**：
>    - 若將線性回歸的 MSE 直接套用在邏輯回歸上，由於複合了非線性的 Sigmoid 函數 $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$，損失曲面將變成**非凸函數（Non-Convex）**。此時曲面充滿大量局部極小值（Local Minima）與梯度趨近零的鞍點／平原區（Plateaus），導致梯度下降容易卡在局部劣解。
>    - 相反地，二元交叉熵損失與 Sigmoid 結合後的 Hessian 矩陣為半正定矩陣，在數學上證明為**凸函數（Convex Function）**。凸函數形如完美碗狀，「任何局部極小值即為全局最優解（Global Minimum）」，從根本上保證了梯度下降演算法一定能找到全局最優參數。
> 2. **數學理論源頭：最大近似估計（Maximum Likelihood Estimation, MLE）**：
>    - 邏輯回歸本質是在估計條件機率：$P(y=1 \mid x) = h_\theta(x)$，而 $P(y=0 \mid x) = 1 - h_\theta(x)$。
>    - 樣本標籤服從**伯努利分佈（Bernoulli Distribution）**：$P(y \mid x) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$。
>    - 假設樣本獨立同分布（i.i.d.），整體資料集的聯合近似函數為 $L(\theta) = \prod_{i=1}^m P(y^{(i)} \mid x^{(i)})$。對近似函數取負對數（Negative Log-Likelihood）除以 $m$，便自然推導出二元交叉熵成本函數。**最小化交叉熵損失，在數學本質上完全等價於最大化觀測樣本的近似機率**。
> 3. **懲罰機制與避免梯度消失（Preventing Gradient Vanishing）**：
>    - **嚴厲懲罰自信的錯誤**：當真實值 $y=1$ 但預測 $h_\theta(x) \to 0$ 時，損失 $-\log(h_\theta(x)) \to \infty$；反之 $y=0$ 預測 $h_\theta(x) \to 1$ 亦趨向無窮大，能對嚴重的預測偏差施加巨大的梯度回饋。
>    - **優雅求導與穩定梯度**：若使用 MSE，在預測嚴重錯誤處（$z = \theta^T x$ 很大但 $y=0$），Sigmoid 的導數 $\sigma'(z) \approx 0$，會造成**梯度消失（Gradient Vanishing）**使模型無法更新。而交叉熵損失求偏導時，對數與指數完美抵消，得到極度優雅的梯度公式：
>      $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
>      **誤差 $(h_\theta(x) - y)$ 越懸殊，梯度就越大，模型校正速度就越快**，確保了訓練的穩定與高效收斂。

### 6.3 梯度推導與權重更新 (梯度下降 Gradient Descent)
對參數 $\theta_j$ 求偏導：
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

向量化形式（Vectorized Form）：
$$\nabla_\theta J(\theta) = \frac{1}{m} X^T \left( g(X\theta) - y \right)$$

權重更新規則（梯度下降，學習率為 $\alpha$）：
$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$

### 6.4 特徵標準化 (Feature Standardization / Z-score)
各課程成績的分數尺度與變異程度可能不同，若不進行標準化，損失函數的等高線將呈現扁平橢圓形，導致梯度下降震盪且難以收斂。
標準化公式：
$$x_{norm} = \frac{x - \mu}{\sigma}$$
其中 $\mu$ 為該特徵平均值，$\sigma$ 為樣本標準差。
> **注意**：測試集必須使用訓練集所計算出的 $\mu$ 與 $\sigma$ 進行標準化，避免資料外洩（Data Leakage）。

---

## 7. 加分項目 (Bonus Part)

在完美達成所有 Mandatory 規範的前提下，可實作以下加分功能：

1. **豐富化 `describe.py` 統計資訊**：
   - 增加變異數 ($Variance$)、四分位距 ($IQR$)、全距 ($Range$)、偏態 ($Skewness$)、峰態 ($Kurtosis$) 以及缺失值比例 ($Missing\%$)。
2. **多種優化演算法支援**：
   - **隨機梯度下降 (Stochastic Gradient Descent, SGD)**：每次迭代隨機選取 1 筆樣本更新梯度，計算速度極快。
   - **小批次梯度下降 (Mini-batch Gradient Descent)**：兼顧向量化矩陣運算效率與收斂穩定性。
3. **正則化項 (Regularization)**：
   - 實作 $L_2$ 正則化（Ridge）避免模型過度擬合（Overfitting）：
     $$J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$
4. **學習率自適應排程與早停機制 (Early Stopping)**：
   - 當連續若干輪迭代損失無顯著改善時提早終止訓練，並繪製 Cost vs. Epochs 學習曲線。

---

## 8. 安裝與執行指南 (Getting Started)

### 8.1 環境安裝
```bash
# 建立虛擬環境 (推薦)
python3 -m venv venv
source venv/bin/activate

# 安裝基本套件
pip install -r requirements.txt
```

### 8.2 執行指令範例

#### 1. 描述性統計
```bash
python3 src/describe.py data/dataset_train.csv
```

#### 2. 資料視覺化
```bash
# 查看哪門課程四學院成績分佈最均勻
python3 src/histogram.py data/dataset_train.csv

# 查看哪兩門特徵最為相似 (強相關)
python3 src/scatter_plot.py data/dataset_train.csv

# 繪製特徵散布圖矩陣以進行特徵選擇
python3 src/pair_plot.py data/dataset_train.csv
```

#### 3. 模型訓練與預測
```bash
# 訓練模型並輸出權重檔 weights.csv
python3 src/logreg_train.py data/dataset_train.csv

# 讀取測試資料與權重檔，產生預測結果 houses.csv
python3 src/logreg_predict.py data/dataset_test.csv weights.csv
```

#### 4. 驗證準確率 (自我評估)
```python
# 可於驗證腳本中呼叫 sklearn 評估準確率 (需 >= 98%)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

---

## 📜 專案評分標準與總結
- **程式碼合規性**：未違規調用現成函式計算統計量或進行模型擬合。
- **回答邏輯與視覺化圖表**：能清晰闡述特徵選擇與資料分析理由。
- **最終準確率**：在 `dataset_test.csv` 上達到 $\ge 98\%$。
- **口試答辯 (Defense)**：能詳細解說梯度下降數學推導、Sigmoid 函數意義、特徵標準化的重要性以及 One-vs-All 分類機制。
