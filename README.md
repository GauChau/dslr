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
  - [4.3 Implementation Walkthrough & Mathematical Foundations (`docs/walkthrough.md`)](./docs/walkthrough.md)
- [5. Getting Started](#5-getting-started)
  - [5.1 Environment Setup](#51-environment-setup)
  - [5.2 Execution Commands](#52-execution-commands)
  - [5.3 Automated Verification & Mathematical Meaning of Tests (`tester/tester.py`)](#53-automated-verification--mathematical-meaning-of-tests-testertesterpy)

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
- Understand the mathematical nature and convexity of the **Log Loss** function.
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
│   ├── walkthrough.md          # Step-by-step implementation, math derivations & Bonus guide
│   └── TODO.md                 # Project audit & actionable checklist
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
│       └── optimizers.py       # Sigmoid & 5 Gradient Descent optimizers (BGD, SGD, Mini, Momentum, Adam)
├── tester/
│   └── tester.py               # Automated test suite (describe vs. Pandas, AST audit, L2 & accuracy >= 98%)
├── modele.csv                  # Trained model weights & scaling parameters
├── houses.csv                  # Final generated house predictions
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

### 4.2 Module Responsibilities
- **`src/utils/stats.py`**: Implements pure from-scratch statistical functions on numerical sequences (plus vectorized Bonus metrics), shared by `describe.py`, `scatter_plot.py`, and the preprocessing pipeline.
- **`src/utils/optimizers.py`**: Encapsulates the Sigmoid activation and 5 gradient descent optimization algorithms (Batch GD, Stochastic GD, Mini-batch GD, Momentum, and Adam) via a unified `optimize()` interface.
- **`src/logreg_train.py` & `src/logreg_predict.py`**: Manage data cleaning, training set parameter storage (medians, means, standard deviations) to prevent **Data Leakage**, One-vs-All training, and final house inference.
- **`tester/tester.py`**: Standalone automated test and oral defense suite verifying numerical equivalence against Pandas, edge-case stability, AST static rule compliance, $L_2$ regularization weight decay mathematics, and end-to-end multi-class classification accuracy ($\ge 98\%$).

### 4.3 Detailed Walkthrough & Mathematical Derivations
For in-depth implementation steps, EDA feature selection justifications, logistic regression mathematical proofs, and Bonus Part formulations, please refer to:
👉 **[`docs/walkthrough.md` — Walkthrough & Mathematical Foundations](./docs/walkthrough.md#dslr-implementation-walkthrough-and-mathematical-foundations)**
- **Section 1**: Guide (`describe.py`, EDA Plots, `logreg_train.py` & `logreg_predict.py`, and why Median Imputation is used)
- **Section 2**: Mathematical Foundations (Sigmoid derivative, Log Loss function vs. MSE convexity/MLE proof, Gradient Descent vectorization, Z-score standardization)
- **Section 3**: Bonus Part (9 extended statistical metrics & 4 advanced gradient descent optimizers)

---

## 5. Getting Started

### 5.1 Environment Setup
```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 5.2 Execution Commands

All CLI scripts in `src/` include the `#!/usr/bin/env python3` shebang and executable permissions (`chmod +x src/*.py`), so you can run them either directly (`./src/<script>.py`) or via `python3`:

#### 1. Descriptive Statistics
```bash
./src/describe.py data/dataset_train.csv
# or: python3 src/describe.py data/dataset_train.csv
```

#### 2. Data Visualization
```bash
# Inspect which course has a homogeneous distribution across all four houses
./src/histogram.py data/dataset_train.csv

# Inspect which two features are strongly correlated
./src/scatter_plot.py data/dataset_train.csv

# Plot the scatter plot matrix for feature selection
./src/pair_plot.py data/dataset_train.csv
```

#### 3. Model Training & Prediction
```bash
# Mandatory: Train the One-vs-All logistic regression model and output modele.csv
./src/logreg_train.py data/dataset_train.csv

# Bonus: Train with custom optimizer and/or L2 regularization
./src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01

# Predict houses on dataset_test.csv and generate houses.csv
./src/logreg_predict.py data/dataset_test.csv modele.csv
```

#### 4. Automated Test & Defense Verification Suite (`tester/tester.py`)
```bash
# Run all 5 test suites (describe vs. pandas, Bonus stats, AST audit, L2 weight decay, and >= 98% accuracy)
./tester/tester.py

# Or run a specific test group only:
./tester/tester.py --only [describe|stats|ast|l2|e2e]
```

### 5.3 Automated Verification & Mathematical Meaning of Tests (`tester/tester.py`)

Provides 5 standalone verification modules (`./tester/tester.py --only [describe|stats|ast|l2|e2e]`):
1. **`describe` vs. Official Pandas `df.describe()`**: Verifies all 8 mandatory metrics within IEEE-754 `float64` machine precision (relative error $\le 1.88 \times 10^{-15}$).
2. **Bonus 1 Extended Metrics & Edge-Case Robustness**: Validates all 7 extended metrics + `pearson_corr` (`Astronomy` vs. `DADA`, $r = -1.000000$) and stresses edge cases (all-`NaN`, $N=1$, constant column $s=0$).
3. **Static AST Compliance Audit**: Uses `ast.parse` to guarantee zero forbidden statistical calls in `src/describe.py` and `src/utils/stats.py`.
4. **Sigmoid Stability & $L_2$ Weight Decay Verification**: Verifies extreme logit stability ($z = \pm 1000$), confirms bias $\theta_0$ is unpenalized, and proves $\|w\|_2$ shrinks under $L_2$ regularization.
5. **End-to-End Multi-Class Optimizer Benchmark & Confusion Matrix**: Benchmarks all 5 optimizers (`bgd`, `sgd`, `mini`, `momentum`, `adam`) + `--lambda 0.01` ($\ge 98.0\%$ accuracy, reaching **98.94%**), prints Precision / Recall / F1-Score and the Confusion Matrix, and validates `houses.csv` on `dataset_test.csv`.

👉 For the detailed explanation of how each test confirms **code integrity** and its **mathematical meaning**, see **[`docs/walkthrough.md` — Bonus 5](./docs/walkthrough.md#bonus-5-automated-test--defense-verification-suite-testertesterpy)**.

---

## 📜 Evaluation Criteria Summary
- **Rule Compliance**: Zero forbidden built-in statistical or ML functions used for heavy lifting (verified via AST in `TEST 3`).
- **Visualizations & Feature Justification**: Clear plots and sound reasoning for feature selection.
- **Classification Accuracy**: $\ge 98\%$ accuracy on `dataset_test.csv` (verified at $98.94\%$ across all 5 optimizers in `TEST 5`).
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
  - [2.2 製作基礎統計指標 (Descriptive Statistics from Scratch)](#22-製作基礎統計指標-descriptive-statistics-from-scratch)
  - [2.3 邏輯回歸與梯度下降 (Logistic Regression & Optimization)](#23-邏輯回歸與梯度下降-logistic-regression--optimization)
  - [2.4 多類別分類 (One-vs-All Classification)](#24-多類別分類-one-vs-all-classification)
- [3. 規範與限制 (Rules & Constraints)](#3-規範與限制-rules--constraints)
- [4. 實作架構設計 (How We Implement It)](#4-實作架構設計-how-we-implement-it)
  - [4.1 專案目錄結構](#41-專案目錄結構)
  - [4.2 模組職責劃分](#42-模組職責劃分)
  - [4.3 細節與數學推導 (`docs/walkthrough.md`)](./docs/walkthrough.md)
- [5. 安裝與執行指南 (Getting Started)](#5-安裝與執行指南-getting-started)
  - [5.1 環境安裝](#51-環境安裝)
  - [5.2 執行指令範例](#52-執行指令範例)
  - [5.3 自動化測試解析：如何確認程式碼完整性與其數學意義 (`tester/tester.py`)](#53-自動化測試解析如何確認程式碼完整性與其數學意義-testertesterpy)

---

## 1. 專案簡介 (Project Overview)

### 1.1 故事背景
霍格華茲魔法學校的**分類帽（Sorting Hat）**正面臨重大危機！校長鄧不利多向麻瓜資料科學家求助，希望能透過麻瓜的「電腦」重新創造出魔法分類帽的功能。麥米奈娃教授提供了一本記載學生資料與魔法課程成績的古老魔法書（經過「數位化」轉為 CSV 檔案）。

### 1.2 任務目標
本專案目標是**完全不依賴機器學習現成高階函式庫（如 Scikit-Learn 的模型訓練、Pandas 的 `describe()` 等）**，從零製作：
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

### 2.2 製作基礎統計指標 (Descriptive Statistics from Scratch)
- 脫離高階工具的黑盒子，親手撰寫演算法計算各項數值統計量：
  - 樣本計數 ($Count$)
  - 算術平均數 ($Mean$)
  - 樣本標準差 ($Sample\ Standard\ Deviation$, 使用自由度 $N - 1$)
  - 極值 ($Min$, $Max$)
  - 四分位數 ($25\%$, $50\%$, $75\%$)：深入理解分位數插值演算法（如線性插值 Linear Interpolation）。

### 2.3 邏輯回歸與梯度下降 (Logistic Regression & Optimization)
- 理解廣義線性模型（Generalized Linear Models），以及如何透過 **Sigmoid 函數** 將線性輸出對應至 $[0, 1]$ 區間作為機率估計。
- Log Loss的數學本質與凸優化特性。
- 掌握梯度向量的推導過程，製作批次梯度下降法（Batch Gradient Descent, BGD），手動調校學習率（Learning Rate $\alpha$）與迭代次數（Epochs）。

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
│   ├── walkthrough.md          # 各階段實作詳解、數學公式推導與 Bonus 指南
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
│       ├── stats.py            # 統計指標 (mean, std, quartiles, etc.)
│       └── optimizers.py       # Sigmoid 與五大梯度下降優化器 (BGD, SGD, Mini, Momentum, Adam)
├── tester/
│   └── tester.py               # 自動化測試與口試驗證腳本 (Pandas 誤差比對、AST 檢查、L2 與準確率驗證)
├── modele.csv                  # 訓練後輸出的權重與標準化參數
├── houses.csv                  # 最終生成的預測輸出檔
├── requirements.txt            # 相依套件清單 (numpy, pandas, matplotlib, seaborn, scikit-learn)
└── README.md                   # 專案說明文件
```

### 4.2 模組職責劃分
- **`src/utils/stats.py`**：實現數值型序列的純製作統計指標（含向量化 Bonus 指標），提供給 `describe.py`、`scatter_plot.py` 及標準化流程使用。
- **`src/utils/optimizers.py`**：封裝 Sigmoid 激活函數與五大梯度下降優化演算法（BGD、SGD、Mini-batch GD、Momentum、Adam），並提供統一接口 `optimize()`。
- **`src/logreg_train.py` 與 `src/logreg_predict.py`**：負責資料清洗、保存訓練集統計參數（中位數、均值、標準差，防止 Data Leakage）、One-vs-All 模型訓練與最終學院預測輸出。
- **`tester/tester.py`**：獨立的自動化測試與口試答辯驗證套件，涵蓋與官方 Pandas 數值精確度比對、極端邊界容錯測試、AST 語法樹合規掃描、$L_2$ 權重衰減數學驗證，以及端到端多類別分類準確率（$\ge 98\%$）與混淆矩陣評估。

### 4.3 細節與數學推導
關於特徵選擇推論、邏輯回歸完整數學推導與加分項目（Bonus Part）定義，請參閱：
👉 **[`docs/walkthrough.md` — 細節與數學推導](./docs/walkthrough.md)**
- **第 1 節**：各階段指南（`describe.py`、三大視覺化分析、`logreg_train.py` 與 `logreg_predict.py`，以及為何採用中位數填補缺失值）
- **第 2 節**：數學原理與公式推導（Sigmoid 導數、對數損失 vs. MSE 凸優化與最大概似估計證明、梯度下降向量化、Z-score 標準化）
- **第 3 節**：Bonus 加分項目（9 項進階統計指標與 4 種進階梯度下降優化器）

---

## 5. 安裝與執行指南 (Getting Started)

### 5.1 環境安裝
```bash
# 建立虛擬環境 (推薦)
python3 -m venv venv
source venv/bin/activate

# 安裝基本套件
pip install -r requirements.txt
```

### 5.2 執行指令範例

`src/` 下的所有腳本皆已加上 `#!/usr/bin/env python3` 與可執行權限（`chmod +x src/*.py`），可直接以 `./src/<script>.py` 或 `python3 src/<script>.py` 執行：

#### 1. 描述性統計
```bash
./src/describe.py data/dataset_train.csv
# 或：python3 src/describe.py data/dataset_train.csv
```

#### 2. 資料視覺化
```bash
# 查看哪門課程四學院成績分佈最均勻
./src/histogram.py data/dataset_train.csv

# 查看哪兩門特徵最為相似 (強相關)
./src/scatter_plot.py data/dataset_train.csv

# 繪製特徵散布圖矩陣以進行特徵選擇
./src/pair_plot.py data/dataset_train.csv
```

#### 3. 模型訓練與預測
```bash
# Mandatory: 訓練模型並輸出權重檔 modele.csv
./src/logreg_train.py data/dataset_train.csv

# Bonus: 指定進階優化器與 L2 正則化係數訓練
./src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01

# 讀取測試資料與權重檔，產生預測結果 houses.csv
./src/logreg_predict.py data/dataset_test.csv modele.csv
```

#### 4. 自動化測試與口試驗證腳本 (`tester/tester.py`)
```bash
# 一鍵執行完整 5 大測試套件 (describe vs. Pandas 誤差、Bonus 統計指標、AST 合規掃描、L2 正則化與 >= 98% 準確率驗證)
./tester/tester.py

# 或僅執行特定測試項目：
./tester/tester.py --only [describe|stats|ast|l2|e2e]
```

### 5.3 自動化測試解析：如何確認程式碼完整性與其數學意義 (`tester/tester.py`)

提供 5 大獨立驗證模組（支援 `./tester/tester.py --only [describe|stats|ast|l2|e2e]`）：
1. **`describe` vs. 官方 Pandas `df.describe()` 數值比對**：驗證 8 項 Mandatory 指標在雙精度浮點數機器極限（相對誤差 $\le 1.88 \times 10^{-15}$）下完全一致。
2. **Bonus 1 進階統計指標與極端退化邊界測試**：驗證 7 項進階指標與皮爾森相關係數（`Astronomy` vs. `DADA`, $r = -1.000000$），並測試全 `NaN`、$N=1$、常數零變異欄（$s=0$）不發生除零崩潰。
3. **AST 抽象語法樹靜態合規掃描**：透過 `ast.parse` 證明 `src/describe.py` 與 `src/utils/stats.py` 零調用任何禁用內建統計函式。
4. **Sigmoid 極端值穩定度與 $L_2$ Weight Decay 驗證**：驗證 $z = \pm 1000$ 無溢位、截距項 $\theta_0$ 不受 $L_2$ 懲罰，以及特徵權重範數 $\|w\|_2$ 嚴格收縮。
5. **端到端五大優化器基準測試與混淆矩陣**：無損備份工作區檔案，驗證 `bgd`, `sgd`, `mini`, `momentum`, `adam` 與 `--lambda 0.01` 全部超越 $\ge 98.0\%$ 門檻（實測 **98.94%**），並輸出 Precision / Recall / F1-Score、混淆矩陣（Confusion Matrix）與 `dataset_test.csv` 預測格式檢查。

👉 關於每項測試**如何確認程式碼完整性（Code Integrity）**及其**背後的數學意義（Mathematical Meaning）**之完整解析，請參閱：**[`docs/walkthrough.md` — Bonus 5](./docs/walkthrough.md#bonus-5-自動化測試與口試答辯驗證套件-testertesterpy)**。

---

## 📜 專案評分標準與總結
- **程式碼合規性**：未違規調用現成函式計算統計量或進行模型擬合（由 `TEST 3` AST 語法樹靜態驗證）。
- **回答邏輯與視覺化圖表**：能清晰闡述特徵選擇與資料分析理由。
- **最終準確率**：在 `dataset_test.csv` 上達到 $\ge 98\%$（由 `TEST 5` 驗證五大優化器皆達 $98.94\%$）。
- **口試答辯 (Defense)**：能詳細解說梯度下降數學推導、Sigmoid 函數意義、特徵標準化的重要性以及 One-vs-All 分類機制。
