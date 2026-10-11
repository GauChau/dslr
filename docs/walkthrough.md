<p align="right">
  <a href="#dslr-implementation-walkthrough-and-mathematical-foundations">
    <img src="https://img.shields.io/badge/English-sienna?style=for-the-badge" />
  </a>
</p>

# DSLR 實作詳解與數學推導指南
## 📖 目錄 (Table of Contents)
- [1. 基礎項目](#1-各階段詳細實作指南-step-by-step-implementation)
  - [Part 1: 描述性統計 (`describe.py`)](#part-1-描述性統計-describepy)
  - [Part 2: 資料視覺化分析 (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)](#part-2-資料視覺化分析-histogrampy-scatter_plotpy-pair_plotpy)
  - [Part 3: 訓練與預測 (`logreg_train.py`, `logreg_predict.py`)](#part-3-訓練與預測-logreg_trainpy-logreg_predictpy)
- [2. 數學原理與公式推導](#2-數學原理與公式推導)
  - [2.0 什麼是「對數機率迴歸」(logistic regression)](#20-什麼是對數機率迴歸logistic-regression)
  - [2.1 Sigmoid 函數](#21-sigmoid-函數)
  - [2.2 成本函數 (Cost Function / Binary Cross-Entropy)](#22-成本函數-cost-function--binary-cross-entropy)
  - [2.3 梯度推導與權重更新 (Gradient Descent)](#23-梯度推導與權重更新-梯度下降-gradient-descent)
  - [2.4 特徵標準化 (Feature Standardization / Z-score)](#24-特徵標準化-feature-standardization--z-score)
- [3. 加分項目](#3-加分項目)
  - [Bonus 1: `src/utils/stats.py` 擴充統計指標與嚴謹數學定義](#bonus-1-srcutilsstatspy-擴充統計指標與嚴謹數學定義)
  - [Bonus 2: `src/utils/optimizers.py` 各類梯度下降優化演算法與其優點](#bonus-2-srcutilsoptimizerspy-各類梯度下降優化演算法與其優點)
  - [Bonus 3: 正則化機制 ($L_2$ Regularization / Ridge)](#bonus-3-正則化機制--regularization--ridge)
  - [Bonus 4: 學習曲線視覺化與 Early Stopping](#bonus-4-學習曲線視覺化與-early-stopping)
  - [Bonus 5: 自我驗證與口試答辯評估工具 (`src/evaluate.py`)](#bonus-5-自我驗證與口試答辯評估工具-srcevaluatepy)

---

## 1. 基礎項目

### Part 1: 描述性統計 `describe.py`
- **目的**：讀取 CSV 檔案，過濾數值型欄位，計算統計函數並整齊排版輸出。
- **基礎輸出指標**：
  - `Count`：有效非空值數量。
  - `Mean`：$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$。
  - `Std`：$s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2}$（無偏樣本標準差）。
  - `Min` / `Max`：單次 $O(N)$ 走訪陣列獲取極值。
  - `25%` / `50%` / `75%`：對陣列進行升序排序後，採用百分位數線性插值法（Linear Interpolation）：
    $$\text{index} = (N - 1) \times p$$
    $$Q_p = x_{\lfloor \text{index} \rfloor} + (\text{index} - \lfloor \text{index} \rfloor) \times (x_{\lceil \text{index} \rceil} - x_{\lfloor \text{index} \rfloor})$$
- **輸出格式**：欄寬自適應對齊，數值精確格式化至小數點後 6 位（`%.6f`）。

### Part 2: 資料視覺化分析 (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)

#### 1. `histogram.py`
- **核心問題**：*Which Hogwarts course has a homogeneous score distribution between all four houses?*
- **作法**：對所有課程特徵繪製 4 個學院的半透明重疊直方圖（`alpha=0.5`）子圖矩陣。
- **結論分析**：尋找在 4 個學院中分佈曲線幾乎重合、常態分佈無差異的課程（具體為 **`Care of Magical Creatures`** 與 **`Arithmancy`**），這代表該課程無法提供學院分類的鑑別度，訓練時應予排除。

#### 2. `scatter_plot.py`
- **核心問題**：*What are the two features that are similar?*
- **作法**：手刻計算特徵兩兩之間的皮爾森相關係數（Pearson Correlation），找出相關係數絕對值最高（接近 $1$ 或 $-1$）的兩門課程，並依學院顏色繪製 2D 散布圖展示其強線性關係。
- **結論分析**：**`Astronomy`** 與 **`Defense Against the Dark Arts`** 呈現完美的負線性相關（$r \approx -1.0$）。高度共線性的特徵攜帶高度重複的資訊，保留其中一個即可，避免增加模型維度與多重共線性問題。

#### 3. `pair_plot.py`
- **核心問題**：*From this visualization, which features are you going to use for your logistic regression?*
- **作法**：繪製 Pair Plot（散布圖矩陣，對角線為直方圖/KDE，非對角線為雙特徵散布圖），並以學院顏色（紅、黃、藍、綠）進行區分。
- **結論分析**：
  - 挑選在不同學院間群聚分離明顯的特徵（分界清晰）。
  - 捨棄分佈重疊嚴重（無鑑別度：`Care of Magical Creatures`、`Arithmancy`）或高度共線（冗餘：`Defense Against the Dark Arts`）的特徵。

### Part 3: 訓練與預測 (`logreg_train.py`, `logreg_predict.py`)

#### 1. `logreg_train.py`
- **輸入**：`dataset_train.csv`。
- **流程**：
  1. **資料清洗**：過濾非數值欄位（姓名 `First Name`, `Last Name`、生日 `Birthday`、慣用手 `Best Hand` 等），並剔除無效特徵。
  2. **缺失值處理（Imputation）**：計算訓練集中各選取特徵之**中位數（Median，$50\%$ 分位數）**並補齊遺失值（`NaN`）。
  3. **特徵標準化**：計算各特徵之 $\mu$ 與 $\sigma$，轉換為標準常態分佈 Z-score。
  4. **One-vs-All 標籤構建**：將 4 個學院標籤轉換為 4 組二元標籤向量（$y \in \{0, 1\}$）。
  5. **模型訓練**：增加 Bias 偏置項（插入全為 $1$ 的欄位 $x_0 = 1$），使用梯度下降演算法迭代更新權重矩陣 $\theta$。
  6. **儲存權重**：將 4 個分類器的權重、偏置項與標準化參數（Median / Mean / Std）儲存至權重檔。

> #### 💡 深入剖析：為什麼缺失值處理採用「中位數（Median）」填補？
> 處理成績數據中的遺失值（`NaN`）時，常見策略包括「整行刪除（Drop Rows）」、「補平均值（Mean）」、「補眾數（Mode）」與「補中位數（Median）」。選擇中位數的核心考量如下：
> 1. **對極端值與異常值具備強韌性（Robustness to Outliers）**：
>    霍格華茲課程成績可能包含極端魔法分數或異常值。平均數（Mean）極易受極端值拉扯而大幅失真；中位數僅取排序後的第 50 百分位數，數值極為穩定，能客觀呈現特徵的中心趨勢（Central Tendency）。
> 2. **適應偏態分佈（Handling Skewed Distributions）**：
>    真實特徵分佈通常並非完美的對稱常態分佈，常呈現左偏或右偏。偏態分佈下平均數會被長尾拉扯，而中位數更能代表半數以上學生的常態表現水準。
> 3. **保留充足樣本數，避免資料浪費（Preserving Sample Size）**：
>    若採取直接刪除包含 `NaN` 的資料列，只要多門課程各自遺失 2%~5% 成績，全集聯集刪除後將損失 30%~50% 的珍貴訓練樣本。資料量銳減會直接劣化模型泛化能力，導致準確率無法達到 98% 門檻；此外 `dataset_test.csv` 中亦含有 `NaN`，測試樣本不可丟棄，必須填補後才能預測。
> 4. **連續數值型資料不適用眾數（Mode）**：
>    課程成績皆為連續型浮點數（Continuous Floats），各分數重複率極低，眾數缺乏統計代表性（眾數通常僅適用於離散類別型特徵 Categorical Features）。

#### 2. `logreg_predict.py`
- **輸入**：`dataset_test.csv` 與訓練產出之權重檔。
- **流程**：
  1. 載入測試集，選取相同特徵，並套用訓練階段儲存的標準化參數（以訓練集 Median 填補缺失值，再以訓練集 $\mu$ 與 $\sigma$ 標準化）。
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

## 2. 數學原理與公式推導
### 2.0 什麼是「對數機率迴歸」(logistic regression)
對數機率迴歸（Logistic Regression、邏輯迴歸）是一種用於分類問題的監督式學習模型。它先計算輸入特徵的線性組合，再透過 Sigmoid 函數將結果轉換為 0 到 1 之間的值，作為二元分類中正類別的預測機率。模型通常使用對數損失（Log Loss）進行訓練，而不是線性迴歸常用的平方誤差（Squared Error），並透過梯度下降等最佳化方法調整模型參數。

### 2.1 Sigmoid 函數
邏輯回歸利用 Sigmoid（Logistic）函數將任意實數線性組合映射為 $(0, 1)$ 之間的機率值：
$$g(z) = \frac{1}{1 + e^{-z}}$$

模型假設（Hypothesis）：
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

其一階導數具備優良的數學性質：
$$g'(z) = g(z)(1 - g(z))$$

### 2.2 成本函數 (Cost Function / Binary Cross-Entropy)
給定 $m$ 筆訓練樣本，邏輯回歸的損失函數為二元交叉熵損失（Binary Cross-Entropy / Log Loss）：
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \Big]$$

- 當真實標籤 $y=1$ 時，若 $h_\theta(x) \to 0$，懲罰項趨近於無窮大 $\infty$。
- 當真實標籤 $y=0$ 時，若 $h_\theta(x) \to 1$，懲罰項趨近於無窮大 $\infty$。
- 此損失函數在參數空間內是嚴謹**凸函數（Convex）**，不存在局部極小值（Local Minima），能保證梯度下降收斂至全局最優解。

> #### 💡 深入剖析：為什麼邏輯回歸使用此特定的交叉熵成本函數？（Why this Cost Function?）
> 線性回歸通常使用均方誤差（MSE, Mean Squared Error），但邏輯回歸必須使用二元交叉熵損失（Binary Cross-Entropy Loss / Log Loss），核心原因包括：
> 1. **凸優化保證（Convexity vs. Non-Convexity）**：
>    - 若將線性回歸的 MSE 直接套用在邏輯回歸上，由於複合了非線性的 Sigmoid 函數 $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$，損失曲面將變成**非凸函數（Non-Convex）**。此時曲面充滿大量局部極小值（Local Minima）與梯度趨近零的鞍點／平原區（Plateaus），導致梯度下降容易卡在局部劣解。
>    - 相反地，二元交叉熵損失與 Sigmoid 結合後的 Hessian 矩陣為半正定矩陣，在數學上證明為**凸函數（Convex Function）**。凸函數形如完美碗狀，「任何局部極小值即為全局最優解（Global Minimum）」，從根本上保證了梯度下降演算法一定能找到全局最優參數。
> 2. **數學理論源頭：最大概似估計（Maximum Likelihood Estimation, MLE）**：
>    - 邏輯回歸本質是在估計條件機率：$P(y=1 \mid x) = h_\theta(x)$，而 $P(y=0 \mid x) = 1 - h_\theta(x)$。
>    - 樣本標籤服從**伯努利分佈（Bernoulli Distribution）**：$P(y \mid x) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$。
>    - 假設樣本獨立同分布（i.i.d.），整體資料集的聯合概似函數為 $L(\theta) = \prod_{i=1}^m P(y^{(i)} \mid x^{(i)})$。對概似函數取負對數（Negative Log-Likelihood, NLL）並除以 $m$，便自然推導出二元交叉熵成本函數。**最小化交叉熵損失，在數學本質上完全等價於最大化觀測樣本的概似機率**。
> 3. **懲罰機制與避免梯度消失（Preventing Gradient Vanishing）**：
>    - **嚴厲懲罰自信的錯誤**：當真實值 $y=1$ 但預測 $h_\theta(x) \to 0$ 時，損失 $-\log(h_\theta(x)) \to \infty$；反之 $y=0$ 預測 $h_\theta(x) \to 1$ 亦趨向無窮大，能對嚴重的預測偏差施加巨大的梯度回饋。
>    - **優雅求導與穩定梯度**：若使用 MSE，在預測嚴重錯誤處（$z = \theta^T x \gg 0$ 但 $y=0$），Sigmoid 的導數 $\sigma'(z) \approx 0$，會造成**梯度消失（Gradient Vanishing）**使模型停滯不前。而交叉熵損失求偏導時，對數與指數完美抵消，得到極度優雅的線性梯度公式：
>      $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
>      **誤差 $(h_\theta(x) - y)$ 越懸殊，梯度就越大，模型校正速度就越快**，確保了訓練的穩定與高效收斂。

### 2.3 梯度推導與權重更新 (梯度下降)
對參數 $\theta_j$ 求偏導：
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

向量化矩陣形式（Vectorized Form）：
$$\nabla_\theta J(\theta) = \frac{1}{m} X^T \left( g(X\theta) - y \right)$$

同步權重更新規則（學習率為 $\alpha$）：
$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$

### 2.4 特徵標準化 (Feature Standardization / Z-score)
各魔法課程成績的分數尺度與變異程度差異極大（例如 `Arithmancy` 分數高達數萬，而 `Herbology` 介於 $-10$ 至 $10$ 之間）。若不進行標準化，損失函數的等高線將呈現狹長扁平橢圓形，導致梯度下降劇烈震盪甚至發散。
標準化公式：
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
其中 $\mu$ 為該特徵平均值，$\sigma$ 為樣本標準差。
> **注意**：測試集必須使用**訓練集**所計算出的 $\mu$ 與 $\sigma$ 進行標準化，防止資料外洩（Data Leakage）。

---

## 3. 加分項目

本專案設計並支援五大加分模組：
1. **擴充 `describe.py` 與 `stats.py` 統計指標**：新增變異數 ($s^2$)、四分位距 ($IQR$)、全距 ($Range$)、偏態 ($Skewness$)、超額峰態 ($Kurtosis$)、缺失值比例 ($Missing\%$) 與不重複值計數 ($Unique$)。
2. **各類梯度下降優化演算法 (`optimizers.py`)**：支援 BGD、SGD、Mini-batch GD、Momentum 與 Adam。
3. **$L_2$ 正則化 (Ridge Regularization)**：懲罰過大權重以防止過度擬合。
4. **學習曲線視覺化與早停機制 (Learning Curve & Early Stopping)**。
5. **自我驗證與口試答辯評估工具 (`src/evaluate.py`)**。

---

### Bonus 1: `src/utils/stats.py` 擴充統計指標與嚴謹數學定義

我們已在 `src/utils/stats.py` 與 `src/describe.py` 中利用 **「複用手刻基礎函式 + 底層 NumPy/Pandas 陣列向量化（Vectorization）」** 實作了以下 9 項進階統計與相關性指標，完全避開禁用函式並大幅提升運算速度。以下為各項指標之數學定義與口試答辯解析：

#### 1. 無偏樣本變異數 (Unbiased Sample Variance — `dfvar`)
- **定義**：
  設有效非空樣本為 $x_1, x_2, \dots, x_n$，算術平均數為 $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$：
  $$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
- **統計意義（Bessel's Correction 貝索校正）**：
  衡量資料偏離平均數的平方平均距離。分母除以 $n - 1$（自由度 $\text{ddof} = 1$）而非 $n$，是因為計算離差 $(x_i - \bar{x})$ 時已受限於 $\sum (x_i - \bar{x}) = 0$ 這一個線性約束，喪失了 1 個自由度；除以 $n - 1$ 才能確保樣本變異數 $s^2$ 是母體變異數 $\sigma^2$ 的**無偏估計量（Unbiased Estimator，$\mathbb{E}[s^2] = \sigma^2$）**。

#### 2. 無偏樣本標準差 (Unbiased Sample Standard Deviation — `dfstd`)
- **定義**：
  $$s = \sqrt{s^2} = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
- **統計意義**：
  將變異數開平方根，使單位（量綱）還原至與原始特徵一致，並作為後續 Z-score 標準化 $z = \frac{x - \bar{x}}{s}$ 的縮放基準。在 `stats.py` 中直接以單行 Series 向量化 `return dfvar(dataframe) ** 0.5` 實作。

#### 3. 全距 (Range — `dfrange`)
- **定義**：
  $$\text{Range} = x_{\max} - x_{\min} = \max_{1 \le i \le n} x_i - \min_{1 \le i \le n} x_i$$
- **統計意義**：
  衡量資料分佈的總跨度（極差）。對極端值（Outliers）極為敏感，可用於一眼看出不同魔法課程的分數尺度差異（例如 `Arithmancy` 全距超過 $1.29 \times 10^5$，而 `Herbology` 僅約 $21.9$），直接證明為何梯度下降前必須進行特徵標準化。

#### 4. 四分位距 (Interquartile Range — `dfiqr`)
- **定義**：
  $$\text{IQR} = Q_{75\%} - Q_{25\%}$$
  其中 $Q_{p}$ 為透過線性插值法求得之第 $p$ 百分位數。
- **統計意義**：
  衡量排序後中間 $50\%$ 核心樣本的分散程度。相較於全距與標準差，IQR 完全不受首尾各 $25\%$ 極端離群值的干擾，是**強健統計學（Robust Statistics）**中最重要的離散度指標（亦常用於定義箱型圖異常值邊界 $[Q_{25\%} - 1.5\,\text{IQR},\ Q_{75\%} + 1.5\,\text{IQR}]$）。

#### 5. 偏態係數 (Fisher-Pearson Coefficient of Skewness — `dfskew`)
- **定義**：
  定義第 $k$ 階樣本中心動差（$k$-th Central Moment）為 $m_k = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^k$，則偏態係數 $g_1$ 為經標準化後的三階中心動差：
  $$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2\right]^{3/2}}$$
- **統計意義**：
  - 衡量資料機率分佈相對於平均數的**左右不對稱程度（Asymmetry）**。
  - 分子的三次方 $(x_i - \bar{x})^3$ 保留了正負號，且會強烈放大遠離均值的長尾離群值；分母 $m_2^{3/2}$ 則消除資料尺度的影響（無因次化）。
  - **判讀基準**：
    - $g_1 \approx 0$：分佈大致左右對稱（注意：$g_1 = 0$ 是對稱的必要非充分條件，正負方向的三次偏差亦可能剛好抵消）。
    - $g_1 > 0$：**右偏／正偏態（Right-skewed）**，右側具長尾（少數極高分拉高平均數，通常 $\text{Mean} > \text{Median}$）。
    - $g_1 < 0$：**左偏／負偏態（Left-skewed）**，左側具長尾（少數極低分拉低平均數，通常 $\text{Mean} < \text{Median}$）。

#### 6. 超額峰態係數 (Excess Kurtosis, Fisher's Definition — `dfkurt`)
- **定義**：
  以四階中心動差 $m_4$ 與二階中心動差平方 $m_2^2$ 之比值減去常態分佈基準值 $3$：
  $$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^4}{\left[\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2\right]^2} - 3$$
- **統計意義**：
  - 衡量資料分佈的**尾端厚度（Tail Heaviness）與峰部尖銳度**，反映極端離群值（Outliers）出現的機率高低。
  - 公式中的 4 次方 $(x_i - \bar{x})^4$ 會極端放大遠離平均數的異常值貢獻。
  - **為什麼要減 3？** 標準常態分佈 $\mathcal{N}(\mu, \sigma^2)$ 的原始峰度 $\frac{m_4}{m_2^2}$ 恆等於 $3$。減去 $3$ 後得到的**超額峰度（Excess Kurtosis）**以 $0$ 為常態分佈基準：
    - $g_2 \approx 0$：**常態峰（Mesokurtic）**，尾部厚度與常態分佈一致。
    - $g_2 > 0$：**高狹峰／厚尾分佈（Leptokurtic）**，尖峰且肥尾，極端值出現機率高於常態分佈。
    - $g_2 < 0$：**低闊峰／薄尾分佈（Platykurtic）**，分佈較平緩（例如連續均勻分佈的 $g_2 = -1.2$），極端值少於常態分佈。

#### 7. 缺失值百分比 (Missing Value Percentage — `dfmissing_pct`)
- **定義**：
  設資料集總樣本列數為 $N_{\text{total}}$，該欄位非空有效樣本數為 $n = \text{Count}(x)$：
  $$\text{Missing\%} = \frac{N_{\text{total}} - n}{N_{\text{total}}} \times 100\%$$
- **統計意義**：
  衡量各特徵的資料遺失率（本專案各課程約有 $1.8\% \sim 2.5\%$ 的缺失率），證明若直接整列刪除（Drop Rows）將損失大量樣本，因此必須採用中位數填補（Median Imputation）。

#### 8. 不重複有效值數量 (Unique Non-NaN Count — `dfunique`)
- **定義**：
  $$\text{Unique}(x) = \left| \left\{ x_i \;\middle|\; x_i \text{ is not NaN},\ 1 \le i \le N_{\text{total}} \right\} \right|$$
- **統計意義**：
  利用集合（Set）互異性計算有效觀測值之基數（Cardinality），用以確認特徵屬於高基數之連續型浮點數（不適用眾數 Mode 填補）或離散型數值。

#### 9. 皮爾森積動差相關係數 (Pearson Correlation Coefficient — `pearson_corr`)
- **定義**：
  給定兩特徵欄位共同非空之配對觀測集 $\{(x_i, y_i)\}_{i=1}^{n}$，其樣本均值分別為 $\bar{x} = \frac{1}{n}\sum x_i$ 與 $\bar{y} = \frac{1}{n}\sum y_i$：
  $$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \in [-1, 1]$$
- **統計意義**：
  即標準化後的共變異數（$\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$），衡量兩變數之間的**線性相關強度與方向**。當 $|r_{xy}| \to 1$ 時代表兩特徵高度共線性（如 `Astronomy` 與 `Defense Against the Dark Arts` 的 $r_{xy} = -1.0$），為 `scatter_plot.py` 提供完全合規的手刻計算核心。

---

### Bonus 2: `src/utils/optimizers.py` 各類梯度下降優化演算法與其優點

我們已在 `src/utils/optimizers.py` 中實作了 5 種梯度下降優化演算法，並透過統一接口 `optimize()` 與 `src/logreg_train.py` 的 `--opt` 參數串接（支援 `--opt [bgd|sgd|mbgd|momentum|adam]`）。以下為各演算法之定義與核心優點：

#### 1. 批次梯度下降 (Batch Gradient Descent, BGD — `--opt bgd`)
- **定義**：
  每個訓練週期使用**全部 $m$ 筆訓練樣本**計算真實平均梯度 $g_t$：
  $$g_t = \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{m} X^T (h_\theta(X) - y)$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **優點與特性**：
  - 梯度估計無雜訊，每一步都精確朝向全局最優解（Global Minimum）下降，收斂軌跡最平滑穩定。
  - 缺點是每更新一次權重就必須遍歷整張資料表，在超大型資料集上單步運算成本高。

#### 2. 隨機梯度下降 (Stochastic Gradient Descent, SGD — `--opt sgd`)
- **定義**：
  每個訓練週期先將樣本索引隨機洗牌（Shuffle），接著**每觀測 1 筆樣本 $(x^{(i)}, y^{(i)})$ 就立刻更新一次權重**：
  $$g_t = \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **優點與特性**：
  - **更新頻率極高**：在 $m = 1280$ 筆訓練集上，跑 1 個訓練週期就更新了 1280 次權重，初期下降速度極快！
  - 單筆梯度帶有的隨機雜訊（Stochastic Noise）在非凸問題中有助於跳脫局部極小值或鞍點，但在最優解附近會持續微幅震盪。

#### 3. 小批次梯度下降 (Mini-batch Gradient Descent, MBGD — `--opt mini/mbgd/mb`)
- **定義**：
  每個訓練週期洗牌後，將資料切分為大小為 $B$（預設 $B = 32$）的小批次 $\mathcal{B}_k$，對每個小批次計算平均梯度並更新權重：
  $$g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{B} X_{\mathcal{B}_k}^T (h_\theta(X_{\mathcal{B}_k}) - y_{\mathcal{B}_k})$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **優點與特性**：
  - **完美折衷 BGD 與 SGD**：既能利用 NumPy 矩陣向量化（Vectorization）硬體加速，又比 BGD 更新頻繁（每個訓練週期更新 $\lceil m/B \rceil = 40$ 次），且大幅降低了 SGD 的單筆震盪雜訊。

#### 4. 動量梯度下降法 (Gradient Descent with Momentum — `--opt momentum`)
- **定義**：
  引入物理學中的**「動量／速度向量（Velocity $v_t$）」**，對過去的梯度進行指數移動平均（Exponentially Weighted Moving Average，動量衰減係數 $\beta = 0.9$）：
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
  $$\theta_{t+1} = \theta_t - \alpha v_t$$
- **核心原理與優點**：
  - **平滑梯度變化**：當梯度在某個方向上反覆改變時，歷史梯度與目前梯度可能互相抵消，降低參數更新的劇烈波動。
  - **累積一致方向的梯度**：如果多個連續步驟的梯度大致指向相同方向，移動平均會保留這個方向的訊號，使參數更新不會只依賴當前單一步驟的梯度。
  - **改善收斂效率**：在梯度方向持續變化或不同方向的曲率差異較大的情況下，Momentum 通常能減少更新方向的震盪，並加快沿著穩定方向的前進速度。
  > Momentum 並**不**保證每次更新都更快，也不保證一定收斂。實際效果取決於學習率、動量係數、損失函數的形狀，以及梯度的變化情況。

#### 5. 自適應矩估計優化器 (Adam: Adaptive Moment Estimation — `--opt adam`)
- **定義**：
  同時維護梯度的**一階動差估計 $m_t$（平均值／方向慣性）**與**二階原始動差估計 $v_t$（未中心化變異數／梯度平方幅度）**，並加上初期偏差校正（Bias Correction），預設超參數為 $\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$：
  1. **更新一階與二階動差（指數移動平均）**：
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  2. **偏差校正（Bias Correction，修正 $t$ 較小時因初始化為 $0$ 導致的低估偏差）**：
     $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  3. **自適應參數更新**：
     $$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
- **優點與核心原理**：
  - **結合 Momentum 與 RMSProp 的現代王牌優化器**：
    - 分子 $\hat{m}_t$ 提供 **Momentum 的方向慣性**，消除震盪並加速前進；
    - 分母 $\sqrt{\hat{v}_t}$ 提供 **RMSProp 的逐參數自適應學習率（Per-parameter Adaptive Learning Rate）**：對於梯度經常很大、坡度陡峭的特徵，$\sqrt{\hat{v}_t}$ 較大，會自動縮小有效步長防止發散；對於梯度微小、稀疏或平緩的特徵，$\sqrt{\hat{v}_t}$ 較小，會自動放大有效步長加速收斂！
  - 對初始學習率 $\alpha$ 的敏感度極低，通常在極少迭代輪數內即可達到極高準確率。

#### 執行指令範例：
```bash
# 預設 Mandatory 梯度下降
python3 src/logreg_train.py data/dataset_train.csv

# 使用 Bonus 優化器
python3 src/logreg_train.py data/dataset_train.csv --opt bgd
python3 src/logreg_train.py data/dataset_train.csv --opt sgd
python3 src/logreg_train.py data/dataset_train.csv --opt mbgd
python3 src/logreg_train.py data/dataset_train.csv --opt momentum
python3 src/logreg_train.py data/dataset_train.csv --opt adam
```

---

### Bonus 3: 正則化機制 ($L_2$ Regularization / Ridge)

#### 1. 為什麼需要 $L_2$ 正則化？
為防止模型權重過大造成過度擬合（Overfitting），在損失函數中加入 $L_2$ 懲罰項。模型將學生分到某學院的預測機率為 $h_{w,b}(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$。若沒有正則化，會面臨三個問題：
- **權重無限膨脹（Weight Explosion）**：為了讓訓練集上的交叉熵損失（Log Loss）趨近於 $0$，模型會試圖做出「百分之百自信」的預測（讓 $h_{w,b}(x) \to 1$ 或 $0$）。由於 Sigmoid 只有在輸入趨近 $\pm\infty$ 時才會到達 $1$ 或 $0$，梯度下降會驅使權重 $w_j$ 越長越大。
- **過度敏感與死記雜訊（Overfitting）**：當少數課程的權重 $w_j$ 變得極大時，只要測試集學生的某科成績有微小波動或離群值（Outlier），算出來的 $w^T x + b$ 就會劇烈震盪，導致分類錯誤。
- **共線性權重失控（Multicollinearity）**：當兩門課高度相關（如 `Astronomy` 與 `Defense Against the Dark Arts`）時，無正則化模型可能賦予極端權重；而 $L_2$ 偏好「小而分散」的權重（因為 $250^2 + 250^2 \ll 1000^2 + 0^2$），能穩定分配共線特徵的權重。

#### 2. 定義與「權重衰減（Weight Decay）」推導
- **步驟一：在損失函數加上「權重平方和罰款」**（對應 `src/utils/optimizers.py` 的 `l2_penalty()`）：
  $$J_{\text{reg}}(w, b) = \underbrace{J(w, b)}_{\text{原始交叉熵損失}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 懲罰項}}$$
  - **$\lambda \ge 0$（即 CLI 的 `--lambda` 參數）**：控制懲罰力道。當 $\lambda = 0$ 時退化為原始邏輯回歸；$\lambda$ 越大，對大權重的懲罰越重（分母除以 $2$ 是為了微分時與次方 $2$ 剛好約分抵消）。
- **步驟二：在梯度更新時產生「權重衰減（Weight Decay）」**：
  對特徵權重 $w_j\ (j \ge 1)$ 求偏導數後，梯度多了一項線性懲罰 $\frac{\lambda}{m} w_j$：
  $$\nabla_w J_{\text{reg}}(w, b) = \underbrace{\frac{1}{m} X^T (p - y)}_{\text{原始梯度}} + \underbrace{\frac{\lambda}{m} w}_{\text{L2 懲罰梯度}}, \qquad \frac{\partial J_{\text{reg}}(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (p^{(i)} - y^{(i)})$$
  將正則化梯度代入權重更新公式 `w := w - alpha * grads` 並移項整理：
  $$w := w - \alpha \left( \nabla_w J(w, b) + \frac{\lambda}{m} w \right) \implies w := \underbrace{\left(1 - \frac{\alpha \lambda}{m}\right)}_{\text{小於 1 的收縮比例}} w - \alpha \nabla_w J(w, b)$$
  - 因為 $\left(1 - \frac{\alpha \lambda}{m}\right) < 1$，**每走一步梯度下降，所有特徵權重 $w$ 都會先自動往 $0$ 的方向縮小一點點（Weight Decay）**。若某課程對分類無幫助（原始梯度 $\approx 0$），其權重便會逐步衰減趨近於 $0$；只有真正對分類有持續貢獻的特徵，才能抵抗收縮力道並維持權重。

> #### 💡 口試答辯必考細節（Peer Defense Q&A）
> 1. **為什麼偏置項 Bias $\theta_0$（程式碼中的 `b`）不參與正則化？**
>    - 偏置項 $b$ 沒有乘上任何輸入特徵 $x_j$，它只決定決策超平面在空間中的**整體平移門檻（基準機率）**，不會因為 $b$ 較大而放大輸入資料的雜訊（Variance）。若連 $b$ 一起懲罰壓向 $0$，反而會強迫決策邊界通過原點附近，造成不必要的偏差（Underfitting）。因此在 `optimizers.py` 中，只有 `grads` 加上 `(l2_lambda / m_samples) * w`，而 `grad_bias` 完全不加懲罰項。
> 2. **為什麼使用 $L_2$ 正則化前必須先做 Z-score 特徵標準化？**
>    - 因為 $L_2$ 使用同一個係數 $\lambda$ 平等懲罰所有權重平方和 $\sum w_j^2$。若未標準化，分數高達數萬分的科目（如 `Arithmancy`）其權重 $w_j$ 天生極小，而分數僅在 $-10 \sim 10$ 的科目（如 `Herbology`）其權重 $w_j$ 天生較大，$L_2$ 就會不公平地重罰數值尺度小的科目。先做 Z-score 標準化讓所有特徵同尺度，$\lambda$ 的懲罰才具備公平性。

#### 3. 支援範圍與 CLI 指令範例
五種優化器（`batch_gd`, `stochastic_gd`, `minibatch_gd`, `momentum_gd`, `adam_gd`）與統一接口 `optimize()` 皆支援 `l2_lambda` 參數（預設 `0.0`）：
```bash
# 使用預設 BGD 搭配 L2 正則化 (lambda = 0.01)
python3 src/logreg_train.py data/dataset_train.csv --lambda 0.01

# 結合進階優化器 (如 Adam 或 Mini-batch) 與 L2 正則化
python3 src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01
```

---

### Bonus 4: 學習曲線視覺化與 Early Stopping

- **實作設計**：
  1. **記錄訓練與驗證損失**：在每個訓練週期（epoch）結束時記錄 Binary Cross-Entropy 損失與當前準確率。
  2. **學習曲線繪圖 (`--plot` 開關)**：繪製 $\text{Loss vs. Epochs}$ 曲線，直觀比較 BGD、Mini-batch GD、SGD、Momentum 與 Adam 的收斂動態。
  3. **早停機制 (Early Stopping)**：當損失在連續 $P$ 個訓練週期改善幅度小於閾值 $\text{tol}$ 時自動提早終止訓練，節省算力並避免過度擬合。

---

### Bonus 5: 自我驗證與口試答辯評估工具 (`src/evaluate.py`)

- **實作設計**：
  提供 5-Fold 交叉驗證（Cross-Validation）、混淆矩陣（Confusion Matrix）、Precision / Recall / F1-Score 以及 `sklearn.metrics.accuracy_score` 驗證輸出，確保模型在各折驗證上穩定超越 $\ge 98.0\%$ 門檻。

---

<br/>

---

<br/>

<p align="right">
  <a href="#dslr-實作詳解與數學推導指南">
    <img src="https://img.shields.io/badge/-TOP-sienna?style=for-the-badge" />
  </a>
</p>

# DSLR Implementation Walkthrough and Mathematical Foundations <a id="dslr-implementation-walkthrough-and-mathematical-foundations"></a>

## 📖 Table of Contents
- [1. Mandatory Part](#1-mandatory-part)
  - [Part 1: Descriptive Statistics (`describe.py`)](#part-1-descriptive-statistics-describepy)
  - [Part 2: Data Visualization (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)](#part-2-data-visualization-histogrampy-scatter_plotpy-pair_plotpy)
  - [Part 3: Model Training & Prediction (`logreg_train.py`, `logreg_predict.py`)](#part-3-model-training--prediction-logreg_trainpy-logreg_predictpy)
- [2. Mathematical Foundations](#2-mathematical-foundations)
  - [2.0 Logistic Regression](#20-logistic-regression)
  - [2.1 Sigmoid Function](#21-sigmoid-function)
  - [2.2 Cost Function (Binary Cross-Entropy)](#22-cost-function-binary-cross-entropy)
  - [2.3 Gradient Derivation & Weight Update (Gradient Descent)](#23-gradient-derivation--weight-update-gradient-descent)
  - [2.4 Feature Standardization (Z-score)](#24-feature-standardization-z-score)
- [3. Bonus Part](#3-bonus-part)
  - [Bonus 1: Extended Metrics in `src/utils/stats.py` & Mathematical Definitions](#bonus-1-extended-metrics-in-srcutilsstatspy--mathematical-definitions--implemented)
  - [Bonus 2: Five Optimization Algorithms in `src/utils/optimizers.py` — Math & Advantages](#bonus-2-five-optimization-algorithms-in-srcutilsoptimizerspy--math--advantages--implemented)
  - [Bonus 3: $L_2$ Regularization (Ridge)](#bonus-3--regularization-ridge)
  - [Bonus 4: Learning Curve Visualization & Early Stopping](#bonus-4-learning-curve-visualization--early-stopping)
  - [Bonus 5: Self-Evaluation & Defense Script (`src/evaluate.py`)](#bonus-5-self-evaluation--defense-script-srcevaluatepy)

---

## 1. Mandatory Part

### Part 1: Descriptive Statistics (`describe.py`)
- **Goal**: Read a CSV dataset, filter numerical columns, compute summary statistics from scratch, and display a neatly aligned table.
- **Core Metrics Computed**:
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
  5. **Gradient Descent Optimization**: Prepend a bias column of ones ($x_0 = 1$) and iteratively update the weight matrix $\theta$ using Gradient Descent.
  6. **Save Parameters**: Export the trained weights, biases, and scaling parameters ($\text{median}, \mu, \sigma$) to the weight file.

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
- **Input**: `dataset_test.csv` and trained weight file.
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

## 2. Mathematical Foundations

### 2.0 Logistic Regression

Logistic regression is a supervised learning model used for classification tasks. It first computes a linear combination of the input features, then applies the sigmoid function to transform the result into a value between 0 and 1, representing the predicted probability of the positive class in binary classification. The model is typically trained using the log loss function rather than the squared error commonly used in linear regression, with its parameters adjusted through optimization methods such as gradient descent.

### 2.1 Sigmoid Function
Logistic regression applies the logistic (Sigmoid) function to map any real-valued linear combination into a probability in $(0, 1)$:
$$g(z) = \frac{1}{1 + e^{-z}}$$

Model hypothesis:
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Its first derivative satisfies the elegant identity:
$$g'(z) = g(z)(1 - g(z))$$

### 2.2 Cost Function (Binary Cross-Entropy)
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

### 2.3 Gradient Derivation & Weight Update (Gradient Descent)
Partial derivative of the cost function with respect to $\theta_j$:
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

Vectorized matrix form:
$$\nabla_\theta J(\theta) = \frac{1}{m} X^T \left( g(X\theta) - y \right)$$

Simultaneous weight update rule with learning rate $\alpha$:
$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$

### 2.4 Feature Standardization (Z-score)
Because Hogwarts courses have vastly different score scales (e.g., `Arithmancy` in the tens of thousands vs. `Herbology` between $-10$ and $10$), unscaled features create elongated elliptical loss contours that cause gradient descent to oscillate or diverge.
Standardization formula:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
where $\mu$ is the feature mean and $\sigma$ is the sample standard deviation.
> **Note**: The test set must be standardized using the **training set's** $\mu$ and $\sigma$ to prevent Data Leakage.

---

## 3. Bonus Part

Once the Mandatory Part is 100% compliant and achieving $\ge 98\%$ accuracy, the project supports 5 Bonus modules:
1. **Extended Statistical Metrics in `describe.py` & `stats.py`**: Adds sample Variance ($s^2$), Interquartile Range ($IQR$), Range ($Max - Min$), Skewness ($g_1$), Excess Kurtosis ($g_2$), Missing value percentage ($Missing\%$), and Unique count ($Unique$).
2. **Five Optimization Algorithms (`optimizers.py`)**: Supports BGD, SGD, Mini-Batch GD, Momentum, and Adam.
3. **$L_2$ Regularization (Ridge)**: Penalizes large weights to prevent overfitting.
4. **Learning Curve Visualization & Early Stopping**.
5. **Self-Evaluation & Defense Script (`src/evaluate.py`)**.

---

### Bonus 1: Extended Metrics in `src/utils/stats.py` & Mathematical Definitions

We have implemented 9 bonus statistical and correlation functions in `src/utils/stats.py` and `src/describe.py` by combining **from-scratch base functions with C-backed array vectorization**, strictly avoiding forbidden built-ins while achieving high performance. Below are the exact mathematical definitions and theoretical justifications for peer defense:

#### 1. Unbiased Sample Variance (`dfvar`)
- **Definition**:
  Given valid non-`NaN` observations $x_1, x_2, \dots, x_n$ with arithmetic mean $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$:
  $$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
- **Statistical Meaning (Bessel's Correction)**:
  Measures the average squared deviation from the sample mean. Dividing by $n - 1$ ($\text{ddof} = 1$) rather than $n$ corrects for the loss of 1 degree of freedom caused by estimating the population mean $\mu$ with the sample mean $\bar{x}$ (which imposes the linear constraint $\sum_{i=1}^n (x_i - \bar{x}) = 0$). This makes $s^2$ an **unbiased estimator** of the true population variance $\sigma^2$ ($\mathbb{E}[s^2] = \sigma^2$).

#### 2. Unbiased Sample Standard Deviation (`dfstd`)
- **Definition**:
  $$s = \sqrt{s^2} = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
- **Statistical Meaning**:
  Restores the dispersion measure to the original unit of measurement of the feature and serves as the scaling denominator for Z-score standardization $z = \frac{x - \bar{x}}{s}$. Implemented in `stats.py` as a 1-line vectorized Series operation: `return dfvar(dataframe) ** 0.5`.

#### 3. Range (`dfrange`)
- **Definition
  $$\text{Range} = x_{\max} - x_{\min} = \max_{1 \le i \le n} x_i - \min_{1 \le i \le n} x_i$$
- **Statistical Meaning**:
  Measures the total spread between extrema. Highly sensitive to outliers, it immediately highlights the drastic scale differences across Hogwarts courses (e.g., `Arithmancy` spans $> 1.29 \times 10^5$ whereas `Herbology` spans $\approx 21.9$), demonstrating why feature standardization is mandatory before running gradient descent.

#### 4. Interquartile Range (`dfiqr`)
- **Definition**:
  $$\text{IQR} = Q_{75\%} - Q_{25\%}$$
  where $Q_p$ denotes the $p$-th percentile computed via linear interpolation.
- **Statistical Meaning**:
  Captures the spread of the middle $50\%$ of sorted data. Unlike Range or Standard Deviation, the IQR is completely immune to extreme outliers in the top and bottom $25\%$ tails, making it the cornerstone dispersion metric in **Robust Statistics** (and the basis for Tukey's outlier fence $[Q_{25\%} - 1.5\,\text{IQR},\ Q_{75\%} + 1.5\,\text{IQR}]$).

#### 5. Fisher-Pearson Coefficient of Skewness (`dfskew`)
- **Definition
  Let the $k$-th sample central moment be $m_k = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^k$. The sample skewness $g_1$ is the standardized third central moment:
  $$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2\right]^{3/2}}$$
- **Statistical Meaning**:
  - Quantifies the **asymmetry** of a probability distribution around its mean.
  - Raising deviations $(x_i - \bar{x})$ to the 3rd power preserves their sign ($+$ or $-$) while heavily amplifying distant tail values; dividing by $m_2^{3/2}$ makes the metric dimensionless (scale-invariant).
  - **Interpretation**:
    - $g_1 \approx 0$: Roughly symmetric distribution (note: $g_1 = 0$ does not strictly guarantee symmetry if asymmetric positive and negative cubic deviations cancel out).
    - $g_1 > 0$: **Right-skewed (Positive skew)** — longer right tail; extreme high scores pull the mean above the median ($\text{Mean} > \text{Median}$).
    - $g_1 < 0$: **Left-skewed (Negative skew)** — longer left tail; extreme low scores pull the mean below the median ($\text{Mean} < \text{Median}$).

#### 6. Excess Kurtosis — Fisher's Definition (`dfkurt`)
- **Definition**:
  Defined as the standardized fourth central moment $\frac{m_4}{m_2^2}$ minus the normal baseline of $3$:
  $$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^4}{\left[\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2\right]^2} - 3$$
- **Statistical Meaning**:
  - Measures **tail heaviness (outlier propensity) and peak sharpness** relative to a normal distribution.
  - Raising deviations to the 4th power makes $m_4$ overwhelmingly sensitive to extreme outliers far from the mean.
  - **Why subtract 3?** For any Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, Pearson's raw kurtosis $\frac{m_4}{m_2^2}$ is fixed at $3$. Subtracting $3$ (Fisher's definition) centers **Excess Kurtosis** at $0$:
    - $g_2 \approx 0$: **Mesokurtic** — tail weight matches a normal distribution.
    - $g_2 > 0$: **Leptokurtic (Heavy-tailed / Sharp peak)** — higher probability of extreme outliers than a normal distribution.
    - $g_2 < 0$: **Platykurtic (Light-tailed / Flat peak)** — fewer extreme outliers than a normal distribution (e.g., a uniform distribution has $g_2 = -1.2$).

#### 7. Missing Value Percentage (`dfmissing_pct`)
- **Definition**:
  Given total dataset rows $N_{\text{total}}$ and valid non-`NaN` count $n = \text{Count}(x)$:
  $$\text{Missing\%} = \frac{N_{\text{total}} - n}{N_{\text{total}}} \times 100\%$$
- **Statistical Meaning**:
  Quantifies data incompleteness per column ($\approx 1.8\%\text{–}2.5\%$ per course in `dataset_train.csv`), justifying median imputation over row deletion.

#### 8. Unique Non-NaN Count (`dfunique`)
- **Definition**:
  $$\text{Unique}(x) = \left| \left\{ x_i \;\middle|\; x_i \text{ is not NaN},\ 1 \le i \le N_{\text{total}} \right\} \right|$$
- **Statistical Meaning**:
  Computes the cardinality $|\mathcal{S}|$ of the set of distinct valid observations, distinguishing continuous floating-point features from discrete categorical features.

#### 9. Pearson Correlation Coefficient (`pearson_corr`)
- **Definition**:
  Given $n$ paired non-`NaN` observations $\{(x_i, y_i)\}_{i=1}^{n}$ with sample means $\bar{x} = \frac{1}{n}\sum x_i$ and $\bar{y} = \frac{1}{n}\sum y_i$:
  $$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} \in [-1, 1]$$
- **Statistical Meaning**:
  Normalized covariance ($\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$) measuring the **strength and direction of linear association** between two features. Powers `src/scatter_plot.py` to detect collinear features ($r_{xy} = -1.0$ between `Astronomy` and `Defense Against the Dark Arts`) without calling `df.corr()`.

---

### Bonus 2: Five Optimization Algorithms in `src/utils/optimizers.py` — Math & Advantages

We have implemented 5 gradient descent optimization algorithms in `src/utils/optimizers.py` and wired them to `src/logreg_train.py` via the unified dispatcher `optimize()` and the `--opt [bgd|sgd|mbgd|momentum|adam]` CLI flag. Below are their mathematical definitions and key advantages for peer defense:

#### 1. Batch Gradient Descent (BGD — `--opt bgd`)
- **Definition**:
  Computes the exact average gradient $g_t$ across **all $m$ training samples** at each epoch:
  $$g_t = \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{m} X^T (h_\theta(X) - y)$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **Advantages & Characteristics**:
  - Noise-free gradient direction; takes deterministic steps directly toward the global minimum on convex loss surfaces.
  - Requires scanning the entire dataset for a single weight update, making per-step computation expensive on massive datasets.

#### 2. Stochastic Gradient Descent (SGD — `--opt sgd`)
- **Definition**:
  Randomly shuffles sample indices at each epoch and **updates weights immediately after evaluating each individual sample $(x^{(i)}, y^{(i)})$**:
  $$g_t = \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **Advantages & Characteristics**:
  - **High Update Frequency**: Performs $m$ weight updates per epoch ($1280\times$ more updates per epoch than BGD), yielding rapid initial loss reduction.
  - High variance in single-sample gradients introduces stochastic oscillations around the minimum, which can help escape local minima/saddle points in non-convex landscapes.

#### 3. Mini-batch Gradient Descent (MBGD — `--opt mini/mbgd/mb`)
- **Definition**:
  Shuffles the dataset and partitions it into small batches $\mathcal{B}_k$ of size $B$ (default $B = 32$), updating weights using each mini-batch's average gradient:
  $$g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{B} X_{\mathcal{B}_k}^T (h_\theta(X_{\mathcal{B}_k}) - y_{\mathcal{B}_k})$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **Advantages & Characteristics**:
  - **Best of Both Worlds**: Combines the hardware-accelerated matrix vectorization of BGD with the frequent updates ($\lceil m/B \rceil$ updates per epoch) of SGD, while significantly reducing gradient variance.

#### 4. Gradient Descent with Momentum (`--opt momentum`)
- **Definition**:
  Introduces a physical **velocity vector $v_t$** that accumulates an exponentially weighted moving average of past gradients (momentum decay factor $\beta = 0.9$):
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
  $$\theta_{t+1} = \theta_t - \alpha v_t$$
- **Advantages & Physical Intuition**:
  - **Heavy Ball Rolling Downhill**: Standard gradient descent has no memory of previous steps—it stalls on flat plateaus and zig-zags wildly across steep, narrow ravines where surface curvature differs across dimensions.
  - **Dampens Oscillations & Accelerates Convergence**: By averaging historical gradients into $v_t$, opposing oscillations perpendicular to the valley cancel out, while consistent components pointing toward the minimum accumulate momentum and accelerate convergence.

#### 5. Adam: Adaptive Moment Estimation (`--opt adam`)
- **Definition**:
  Maintains exponentially decaying averages of both the **first moment $m_t$ (mean / directional momentum)** and the **second raw moment $v_t$ (uncentered variance / squared gradient magnitude)**, coupled with bias correction ($\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$):
  1. **Update biased first and second moment estimates**:
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  2. **Compute bias-corrected moment estimates** (counteracting initialization at $0$ during early steps $t$):
     $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  3. **Update parameters adaptively**:
     $$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
- **Advantages & Core Mechanism**:
  - **Combines Momentum + RMSProp**:
    - The numerator $\hat{m}_t$ provides **directional momentum** to smooth out noisy gradients;
    - The denominator $\sqrt{\hat{v}_t}$ provides **per-parameter adaptive learning rates**: features with consistently large/steep gradients receive a smaller effective step size (preventing divergence), while features with small/flat gradients receive a larger effective step size (accelerating learning).
  - Converges dramatically faster than standard gradient descent and is highly robust to hyperparameter choices.

#### CLI Usage Examples:
```bash
# Default Mandatory Gradient Descent
python3 src/logreg_train.py data/dataset_train.csv

# Bonus Optimizers
python3 src/logreg_train.py data/dataset_train.csv --opt bgd
python3 src/logreg_train.py data/dataset_train.csv --opt sgd
python3 src/logreg_train.py data/dataset_train.csv --opt mbgd
python3 src/logreg_train.py data/dataset_train.csv --opt momentum
python3 src/logreg_train.py data/dataset_train.csv --opt adam
```

---

### Bonus 3: $L_2$ Regularization (Ridge)

#### 1. Why Do We Need $L_2$ Regularization?
To penalize large feature weights to prevent overfitting. The predicted probability is $h_{w,b}(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$. Without regularization, three issues arise:
- **Weight Explosion**: To drive Binary Cross-Entropy loss toward $0$ on the training set, the model tries to make $100\%$ confident predictions ($h_{w,b}(x) \to 1$ or $0$). Because Sigmoid only reaches $1$ or $0$ as its input approaches $\pm\infty$, gradient descent pushes weights $w_j$ to grow indefinitely.
- **Overfitting & Sensitivity to Noise**: When certain course weights $w_j$ become excessively large, tiny fluctuations or outliers in a test student's grades cause massive swings in $w^T x + b$, degrading test accuracy.
- **Multicollinearity Instability**: When two features are strongly correlated (e.g., `Astronomy` and `Defense Against the Dark Arts`), unregularized models can assign extreme opposing weights; $L_2$ prefers smaller, evenly distributed weights ($250^2 + 250^2 \ll 1000^2 + 0^2$), stabilizing training.

#### 2. Mathematical Formulation & "Weight Decay" Derivation
- **Step 1: Add the Squared Weight Penalty to the Cost Function** (implemented via `l2_penalty()` in `src/utils/optimizers.py`):
  $$J_{\text{reg}}(w, b) = \underbrace{J(w, b)}_{\text{Original Cross-Entropy}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 Penalty}}$$
  - **$\lambda \ge 0$ (`--lambda` CLI flag)**: Controls regularization strength. When $\lambda = 0$, it reduces to standard logistic regression; larger $\lambda$ penalizes large weights more heavily (dividing by $2$ cancels the exponent $2$ upon differentiation).
- **Step 2: Induce "Weight Decay" During Gradient Updates**:
  Differentiating with respect to feature weights $w_j\ (j \ge 1)$ adds a linear penalty term $\frac{\lambda}{m} w_j$:
  $$\nabla_w J_{\text{reg}}(w, b) = \underbrace{\frac{1}{m} X^T (p - y)}_{\text{Original Gradient}} + \underbrace{\frac{\lambda}{m} w}_{\text{L2 Penalty Gradient}}, \qquad \frac{\partial J_{\text{reg}}(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (p^{(i)} - y^{(i)})$$
  Substituting into the gradient descent update `w := w - alpha * grads` and rearranging yields:
  $$w := w - \alpha \left( \nabla_w J(w, b) + \frac{\lambda}{m} w \right) \implies w := \underbrace{\left(1 - \frac{\alpha \lambda}{m}\right)}_{\text{Shrinkage Factor } < 1} w - \alpha \nabla_w J(w, b)$$
  - Because $\left(1 - \frac{\alpha \lambda}{m}\right) < 1$, **at every gradient step, all feature weights $w$ automatically decay slightly toward $0$ (Weight Decay)**. Uninformative features whose error gradient is near $0$ decay toward $0$, while genuinely predictive features maintain their weights.

> #### 💡 Key Peer Defense Questions (Q&A)
> 1. **Why is the Bias term $\theta_0$ (`b` in code) excluded from regularization?**
>    - The bias $b$ is not multiplied by any input feature $x_j$; it only shifts the decision boundary's baseline threshold in space and does not amplify input noise (variance). Penalizing $b$ toward $0$ would artificially force the decision hyperplane toward the origin, introducing underfitting bias. Thus, in `optimizers.py`, `(l2_lambda / m_samples) * w` is added only to `grads` and never to `grad_bias`.
> 2. **Why is Z-score feature standardization mandatory before applying $L_2$ regularization?**
>    - $L_2$ applies a single uniform penalty $\lambda$ across $\sum w_j^2$. Without standardization, large-scale features (`Arithmancy` in the tens of thousands) naturally have tiny weights $w_j$, whereas small-scale features (`Herbology` in $[-10, 10]$) require larger weights $w_j$—causing $L_2$ to unfairly penalize small-scale features. Z-score standardization puts all features on the same scale so $\lambda$ penalizes them fairly.

#### 3. CLI Usage Examples
All 5 optimizers (`batch_gd`, `stochastic_gd`, `minibatch_gd`, `momentum_gd`, `adam_gd`) and `optimize()` accept `l2_lambda` (default `0.0`):
```bash
# Default BGD with L2 regularization (lambda = 0.01)
python3 src/logreg_train.py data/dataset_train.csv --lambda 0.01

# Combine an advanced optimizer (e.g., Adam) with L2 regularization
python3 src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01
```

---

### Bonus 4: Learning Curve Visualization & Early Stopping

- **Design**:
  - Tracks training/validation Binary Cross-Entropy loss across epochs.
  - Supports `--plot` to visualize **Cost vs. Epochs** comparing BGD, Mini-batch GD, SGD, Momentum, and Adam convergence dynamics.
  - Supports `--early-stopping` to terminate training automatically once loss improvement plateaus for $P$ consecutive epochs.

---

### Bonus 5: Self-Evaluation & Defense Script (`src/evaluate.py`)

- **Design**:
  - Runs 5-Fold Cross-Validation and prints the Confusion Matrix, Precision, Recall, F1-Score, and `sklearn.metrics.accuracy_score` ($\ge 98.0\%$).
