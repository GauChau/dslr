<p align="right">
  <a href="#dslr-implementation-walkthrough-and-mathematical-foundations">
    <img src="https://img.shields.io/badge/English-sienna?style=for-the-badge" />
  </a>
</p>

# DSLR 製作細節與數學推導
## 📖 目錄 (Table of Contents)
- [1. 基礎項目](#1-基礎項目)
  - [Part 1: 描述性統計 (`describe.py`)](#part-1-描述性統計-describepy)
  - [Part 2: 資料視覺化分析 (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)](#part-2-資料視覺化分析-histogrampy-scatter_plotpy-pair_plotpy)
  - [Part 3: 訓練與預測 (`logreg_train.py`, `logreg_predict.py`)](#part-3-訓練與預測-logreg_trainpy-logreg_predictpy)
- [2. 數學原理與公式推導](#2-數學原理與公式推導)
  - [2.0 什麼是「對數機率迴歸」(logistic regression)](#20-什麼是對數機率迴歸logistic-regression)
  - [2.1 Sigmoid 函數](#21-sigmoid-函數)
  - [2.2 成本函數 (Cost Function)](#22-成本函數-cost-function)
  - [2.3 梯度下降：批次梯度下降 (Batch Gradient Descent, BGD — `--opt bgd [default]`)](#23-梯度下降批次梯度下降-batch-gradient-descent-bgd----opt-bgd-default)
  - [2.4 特徵標準化 (Feature Standardization)](#24-特徵標準化-feature-standardization)
- [3. 加分項目](#3-加分項目)
  - [Bonus 1: `src/utils/stats.py` 擴充統計指標](#bonus-1-srcutilsstatspy-擴充統計指標)
  - [Bonus 2: `src/utils/optimizers.py` 各類梯度下降優化演算法與其優點](#bonus-2-srcutilsoptimizerspy-各類梯度下降優化演算法與其優點)
  - [Bonus 3: 正則化機制 ($L_2$ Regularization / Ridge)](#bonus-3-正則化機制-l_2-regularization--ridge)
  - [Bonus 4: 學習曲線視覺化與 Early Stopping](#bonus-4-學習曲線視覺化與-early-stopping)
  - [Bonus 5: 自動化測試與口試答辯驗證套件 (`tester/tester.py`)](#bonus-5-自動化測試與口試答辯驗證套件-testertesterpy)

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
- **作法**：製作計算特徵兩兩之間的皮爾森相關係數（Pearson Correlation），找出相關係數絕對值最高（接近 $1$ 或 $-1$）的兩門課程，並依學院顏色繪製 2D 散布圖展示其強線性關係。
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
  3. **特徵標準化**：計算各特徵之 $\mu$ 與 $\sigma$，轉換為標準常態分佈。
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

### 2.2 成本函數 (Cost Function)
給定 $m$ 筆訓練樣本，邏輯回歸的損失函數為對數損失（Log Loss）：
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \Big]$$

- 當真實標籤 $y=1$ 時，若 $h_\theta(x) \to 0$，懲罰項趨近於無窮大 $\infty$。
- 當真實標籤 $y=0$ 時，若 $h_\theta(x) \to 1$，懲罰項趨近於無窮大 $\infty$。
- 此損失函數在參數空間內是嚴謹**凸函數（Convex）**，不存在局部極小值（Local Minima），能保證梯度下降收斂至全局最優解。

> #### 💡 深入剖析：為什麼邏輯回歸使用此特定的對數成本函數？（Why this Cost Function?）
> 線性回歸通常使用均方誤差（MSE, Mean Squared Error），但邏輯回歸必須使用對數損失（Log Loss），核心原因包括：
> 1. **凸優化保證（Convexity vs. Non-Convexity）**：
>    - 若將線性回歸的 MSE 直接套用在邏輯回歸上，由於複合了非線性的 Sigmoid 函數 $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$，損失曲面將變成**非凸函數（Non-Convex）**。此時曲面充滿大量局部極小值（Local Minima）與梯度趨近零的鞍點／平原區（Plateaus），導致梯度下降容易卡在局部劣解。
>    - 相反地，對數損失與 Sigmoid 結合後的 Hessian 矩陣為半正定矩陣，在數學上證明為**凸函數（Convex Function）**。凸函數形如完美碗狀，「任何局部極小值即為全局最優解（Global Minimum）」，從根本上保證了梯度下降演算法一定能找到全局最優參數。
> 2. **數學理論源頭：最大概似估計（Maximum Likelihood Estimation, MLE）**：
>    - 邏輯回歸本質是在估計條件機率：$P(y=1 \mid x) = h_\theta(x)$，而 $P(y=0 \mid x) = 1 - h_\theta(x)$。
>    - 樣本標籤服從**伯努利分佈（Bernoulli Distribution）**：$P(y \mid x) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$。
>    - 假設樣本獨立同分布（i.i.d.），整體資料集的聯合概似函數為 $L(\theta) = \prod_{i=1}^m P(y^{(i)} \mid x^{(i)})$。對概似函數取負對數（Negative Log-Likelihood, NLL）並除以 $m$，便自然推導出對數成本函數。**最小化對數損失，在數學本質上完全等價於最大化觀測樣本的概似機率**。
> 3. **懲罰機制與避免梯度消失（Preventing Gradient Vanishing）**：
>    - **嚴厲懲罰自信的錯誤**：當真實值 $y=1$ 但預測 $h_\theta(x) \to 0$ 時，損失 $-\log(h_\theta(x)) \to \infty$；反之 $y=0$ 預測 $h_\theta(x) \to 1$ 亦趨向無窮大，能對嚴重的預測偏差施加巨大的梯度回饋。
>    - **優雅求導與穩定梯度**：若使用 MSE，在預測嚴重錯誤處（$z = \theta^T x \gg 0$ 但 $y=0$），Sigmoid 的導數 $\sigma'(z) \approx 0$，會造成**梯度消失（Gradient Vanishing）**使模型停滯不前。而對數損失求偏導時，對數與指數完美抵消，得到極度優雅的線性梯度公式：
>      $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$
>      **誤差 $(h_\theta(x) - y)$ 越懸殊，梯度就越大，模型校正速度就越快**，確保了訓練的穩定與高效收斂。

### 2.3 梯度下降：批次梯度下降 (Batch Gradient Descent, BGD — `--opt bgd [default]`)
- **定義**：
  每個訓練週期使用**全部 $m$ 筆訓練樣本**計算真實平均梯度 $g_t$：
  $$g_t = \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{m} X^T (h_\theta(X) - y)$$
  同步權重更新規則（學習率為 $\alpha$）：
$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$
- **優點與特性**：
  - 梯度估計無雜訊，每一步都精確朝向全局最優解（Global Minimum）下降，收斂軌跡最平滑穩定。
  - 缺點是每更新一次權重就必須遍歷整張資料表，在超大型資料集上單步運算成本高。

### 2.4 特徵標準化 (Feature Standardization)
各魔法課程成績的分數尺度與變異程度差異極大（例如 `Arithmancy` 分數高達數萬，而 `Herbology` 介於 $-10$ 至 $10$ 之間）。若不進行標準化，損失函數的等高線將呈現狹長扁平橢圓形，導致梯度下降劇烈震盪甚至發散。
標準化公式：
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
其中 $\mu$ 為該特徵平均值，$\sigma$ 為樣本標準差。
> **注意**：將資料拆分成 **80%** 訓練集與 **20%** 驗證集，是為了評估模型對未參與訓練的資料的泛化能力。為避免資料外洩，標準化時應只使用訓練集計算的 $\mu$ 與 $\sigma$，再將這些參數套用到訓練集與驗證集。資料外洩（Data Leakage）是指模型在訓練或模型選擇過程中，取得了原本不應該使用的資訊，導致評估結果無法真實反映模型在新資料上的表現。

---

## 3. 加分項目

本專案設計並支援五大加分模組：
1. **擴充 `describe.py` 與 `stats.py` 統計指標**：新增變異數 ($s^2$)、四分位距 ($IQR$)、全距 ($Range$)、偏態 ($Skewness$)、超額峰態 ($Kurtosis$)、缺失值比例 ($Missing\%$) 與不重複值計數 ($Unique$)。
2. **各類梯度下降優化演算法 (`optimizers.py`)**：支援 BGD、SGD、Mini-batch GD、Momentum 與 Adam。
3. **$L_2$ 正則化 (Ridge Regularization)**：懲罰過大權重以防止過度擬合。
4. **學習曲線視覺化與早停機制 (Learning Curve & Early Stopping)**。
5. **自我驗證與口試答辯評估工具 (`src/evaluate.py`)**。

---

### Bonus 1: `src/utils/stats.py` 擴充統計指標

我們已在 `src/utils/stats.py` 與 `src/describe.py` 中利用 **「複用製作基礎函式 + 底層 NumPy/Pandas 陣列向量化（Vectorization）」** 製作了以下 9 項進階統計與相關性指標，完全避開禁用函式並大幅提升運算速度。以下為各項指標之數學定義與口試答辯解析：

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
  將變異數開平方根，使單位（量綱）還原至與原始特徵一致，並作為後續標準化 $z = \frac{x - \bar{x}}{s}$ 的縮放基準。在 `stats.py` 中直接以單行 Series 向量化 `return dfvar(dataframe) ** 0.5`。

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
  即標準化後的共變異數（$\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$），衡量兩變數之間的**線性相關強度與方向**。當 $|r_{xy}| \to 1$ 時代表兩特徵高度共線性（如 `Astronomy` 與 `Defense Against the Dark Arts` 的 $r_{xy} = -1.0$），為 `scatter_plot.py` 提供完全合規的製作計算核心。

---

### Bonus 2: `src/utils/optimizers.py` 各類梯度下降優化演算法與其優點

除了第 2.3 節預設的批次梯度下降（`--opt bgd`）外，我們在 `src/utils/optimizers.py` 中額外設計了 4 種進階梯度下降優化演算法，並透過統一接口 `optimize()` 與 `src/logreg_train.py` 的 `--opt` 參數串接（支援 `--opt [bgd|sgd|mini|momentum|adam]`）。以下為各演算法之定義與核心優點：

#### 1. 隨機梯度下降 (Stochastic Gradient Descent, SGD — `--opt sgd`)
- **定義**：
  每個訓練週期先將樣本索引隨機洗牌（Shuffle），接著**每觀測 1 筆樣本 $(x^{(i)}, y^{(i)})$ 就立刻更新一次權重**：
  $$g_t = \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **優點與特性**：
  - **更新頻率極高**：在 $m = 1280$ 筆訓練集上，跑 1 個訓練週期就更新了 1280 次權重，初期下降速度極快！
  - 單筆梯度帶有的隨機雜訊（Stochastic Noise）在非凸問題中有助於跳脫局部極小值或鞍點，但在最優解附近會持續微幅震盪。

#### 2. 小批次梯度下降 (Mini-batch Gradient Descent, Mini — `--opt mini`)
- **定義**：
  每個訓練週期洗牌後，將資料切分為大小為 $B$（預設 $B = 32$）的小批次 $\mathcal{B}_k$，對每個小批次計算平均梯度並更新權重：
  $$g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{B} X_{\mathcal{B}_k}^T (h_\theta(X_{\mathcal{B}_k}) - y_{\mathcal{B}_k})$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **優點與特性**：
  - **完美折衷 BGD 與 SGD**：既能利用 NumPy 矩陣向量化（Vectorization）硬體加速，又比 BGD 更新頻繁（每個訓練週期更新 $\lceil m/B \rceil = 40$ 次），且大幅降低了 SGD 的單筆震盪雜訊。

#### 3. 動量梯度下降法 (Gradient Descent with Momentum — `--opt momentum`)
- **定義**：
  引入物理學中的 **「動量／速度向量（Velocity $v_t$）」**，對過去的**梯度**進行指數移動平均（Exponentially Weighted Moving Average，動量衰減係數 $\beta = 0.9$）：
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
  $$\theta_{t+1} = \theta_t - \alpha v_t$$
- **核心原理與優點**：
  - **平滑梯度變化**：當梯度在某個方向上反覆改變時，歷史梯度與目前梯度可能互相抵消，降低參數更新的劇烈波動。
  - **累積一致方向的梯度**：如果多個連續步驟的梯度大致指向相同方向，移動平均會保留這個方向的訊號，使參數更新不會只依賴當前單一步驟的梯度。
  - **改善收斂效率**：在梯度方向持續變化或不同方向的曲率差異較大的情況下，Momentum 通常能減少更新方向的震盪，並加快沿著穩定方向的前進速度。
  > Momentum 並**不**保證每次更新都更快，也不保證一定收斂。實際效果取決於學習率、動量係數、損失函數的形狀，以及梯度的變化情況。

#### 4. 自適應矩估計優化器 (Adam: Adaptive Moment Estimation — `--opt adam`)
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
# 預設 Mandatory 批次梯度下降 (--opt bgd 為預設值)
./src/logreg_train.py data/dataset_train.csv

# 使用 Bonus 優化器
./src/logreg_train.py data/dataset_train.csv --opt sgd
./src/logreg_train.py data/dataset_train.csv --opt mini
./src/logreg_train.py data/dataset_train.csv --opt momentum
./src/logreg_train.py data/dataset_train.csv --opt adam
```

---

### Bonus 3: 正則化機制 ($L_2$ Regularization / Ridge)

#### 0. 什麼是 $L_2$？
- **$L$（Lebesgue）**：源自泛函分析中的 **$L^p$ 勒貝格空間（Lebesgue Space）**，以法國數學家昂利·勒貝格（Henri Lebesgue）命名。在數學中，$L$ 代表基於勒貝格積分定義的測度與範數體系。
- **$2$（二階次數 / 平方和）**：表示使用平方來衡量向量大小，加總後再取平方根（即 $p = 2$）。
- **$L_2$ 範數（$L_2$ Norm / Euclidean Norm）**：在機器學習的有限維實數參數向量 $w \in \mathbb{R}^n$ 中，它就是我們最熟悉的**歐幾里得範數（長度）**：
  $$\|w\|_2 = \sqrt{\sum_{j=1}^n w_j^2} = \sqrt{w_1^2 + w_2^2 + \dots + w_n^2}$$
  在 $L_2$ 正則化中，懲罰項採用的是其平方 $\|w\|_2^2 = \sum_{j=1}^n w_j^2$，因而處處可微（Differentiable）且計算平滑。

#### 1. 為什麼需要 $L_2$ 正則化？
為防止模型權重過大造成過度擬合（Overfitting），在損失函數中加入 $L_2$ 懲罰項。模型將學生分到某學院的預測機率為 $h_{w,b}(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$。若沒有正則化，會面臨三個問題：
- **權重無限膨脹（Weight Explosion）**：為了讓訓練集上的對數損失（Log Loss）趨近於 $0$，模型會試圖做出「百分之百自信」的預測（讓 $h_{w,b}(x) \to 1$ 或 $0$）。由於 Sigmoid 只有在輸入趨近 $\pm\infty$ 時才會到達 $1$ 或 $0$，梯度下降會驅使權重 $w_j$ 越長越大。
- **過度敏感與死記雜訊（Overfitting）**：當少數課程的權重 $w_j$ 變得極大時，只要測試集學生的某科成績有微小波動或離群值（Outlier），算出來的 $w^T x + b$ 就會劇烈震盪，導致分類錯誤。
- **共線性權重失控（Multicollinearity）**：當兩門課高度相關（如 `Astronomy` 與 `Defense Against the Dark Arts`）時，無正則化模型可能賦予極端權重；而 $L_2$ 偏好「小而分散」的權重（因為 $250^2 + 250^2 \ll 1000^2 + 0^2$），能穩定分配共線特徵的權重。

#### 2. 定義與「權重衰減（Weight Decay）」推導
- **步驟一：在損失函數加上「權重平方和罰款」**（對應 `src/utils/optimizers.py` 的 `l2_penalty()`）：
  $$J_{\text{reg}}(w, b) = \underbrace{J(w, b)}_{\text{原始對數損失}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 懲罰項}}$$
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
> 2. **為什麼使用 $L_2$ 正則化前必須先將特徵標準化？**
>    - 因為 $L_2$ 使用同一個係數 $\lambda$ 平等懲罰所有權重平方和 $\sum w_j^2$。若未標準化，分數高達數萬分的科目（如 `Arithmancy`）其權重 $w_j$ 天生極小，而分數僅在 $-10 \sim 10$ 的科目（如 `Herbology`）其權重 $w_j$ 天生較大，$L_2$ 就會不公平地重罰數值尺度小的科目。先標準化讓所有特徵同尺度，$\lambda$ 的懲罰才具備公平性。
> 3. **為什麼在消融實驗（`--visual`）中，$\lambda$ 的大小與驗證集準確率幾乎沒有顯著關聯？**
>    - **「0-1 損失（準確率）」與「對數損失（Log Loss）」的本質差異**：準確率是階梯函數（Step Function），只要樣本沒有橫跨那條 $0.5$ 邊界，準確率就不會動。而 $L_2$ 正則化保護的是平滑的機率校準與模型泛化能力（防止權重過大對離群值過度敏感）。
>    - **資料集的高分離度與幾何邊界（Large Margin）**：霍格華茲資料集在核心科目上（如 `Herbology`, `Astronomy`, `Ancient Runes`, `Flying`）具備強大的幾何邊界與分離度。在多類別分類中，預測依據是取最大者 $\arg\max_k (w_k^T x + b_k)$。$L_2$ 正則化懲罰縮短了權重長度，但沒有大幅改變權重在特徵空間中的朝向（方向）。因此雖然 Sigmoid 機率從原本極端的 $0.999$ 被平滑為 $0.85$，但樣本的相對大小排名（$\arg\max$）完全不變，因而準確率保持穩定。
>    - **極限欠擬合實驗（Underfitting Threshold）**：實測將 $\lambda$ 放大至極端數值：
>      - $\lambda = 0 \sim 2000$：準確率穩定在 **$97.19\%$**（$\|w\|_2$ 從 $2.83$ 縮至 $0.62$）。
>      - $\lambda = 5000$：$\|w\|_2 = 0.30$，準確率開始跌至 **$91.88\%$**。
>      - $\lambda = 10000$：$\|w\|_2 = 0.16$，模型嚴重欠擬合，準確率暴跌至 **$59.06\%$**。
>      這證明了在合理的正則化範圍（$\lambda \le 200$）內，模型兼具了防止過擬合的抗噪能力與堅固的分類精確度。
> 4. **既然準確率看不出明顯變化，消融實驗如何證明 $L_2$ 正則化在此有效發揮了作用？**
>    - **雜訊特徵被壓縮趨近於 0**：在 EDA 直方圖中四學院完全重疊、毫無區分度的課程（如 `Care of Magical Creatures` 與 `Arithmancy`），在 $\lambda=50$ 下的權重被壓到只剩微不足道的 $0.018$；而真正有區分度的特徵（如 `Ancient Runes`）則頂住懲罰，維持主導地位。這證明 $L_2$ 正確扮演了「平滑特徵降維與抗噪」的物理角色。
>    - **模型機率更誠實（Well-Calibrated）**：無正則化時，模型可能會盲目自大地給出 $99.99\%$ 的極端機率；加上 $L_2$ 正則化後，模型機率更為平緩（如 $85\%$），這在真實世界面對未知的離群樣本時，模型的容錯率與穩健性（Robustness）是遠高於無正則化模型的！

#### 3. 支援範圍與 CLI 指令範例
五種優化器（`batch_gd`, `stochastic_gd`, `mini_gd`, `momentum_gd`, `adam_gd`）與統一接口 `optimize()` 皆支援 `l2_lambda` 參數（預設 `0.0`）：
```bash
# 使用預設 BGD 搭配 L2 正則化 (lambda = 0.01)
./src/logreg_train.py data/dataset_train.csv --lambda 0.01

# 結合進階優化器 (如 Adam 或 Mini-batch) 與 L2 正則化
./src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01
```

---

### Bonus 4: 學習曲線視覺化與 Early Stopping (`src/utils/early_stopping.py` & `src/utils/learning_curves.py`)

#### 1. 學習曲線視覺化 (`src/utils/learning_curves.py` — `--visual` / `-v`)
- **設計理念與圖表架構**：
  在機器學習與深度學習中，「損失隨訓練週期變化（Loss vs. Epochs）」的學習曲線是診斷優化器收斂速度、數值穩定性、步長適應性與過度擬合最直觀且強大的視覺化工具。
  當使用者在執行 `logreg_train.py` 時傳入 `-v` 或 `--visual` 參數，系統將自動於根目錄下建立 `plots/` 目錄，並同時呼叫 `plot_learning_curves()` 與 `visualize_benchmark()`。
  其中，`plot_learning_curves()` 以統一標準基準（相同資料分割、標準化特徵與初始權重 $\theta = 0$）對 5 大梯度下降演算法（BGD、SGD、Mini-batch、Momentum、Adam）進行收斂對比評測，並繪製成 $2 \times 3$ 的高解析度圖表矩陣，輸出為 `plots/learning_curves.png`：
  - **子圖 (0,0) 至 (1,0) [共 4 張]**：分別展示 4 個學院（Gryffindor, Hufflepuff, Ravenclaw, Slytherin）在 5 大優化器下的二元對數損失（Log Loss）收斂軌跡。若觸發早停機制，圖中將以星號（$\star$）精確標註早停發生的 Epoch 與對應損失值。
  - **子圖 (1,1)**：4 個學院整體平均對數損失（Overall Mean Loss）收斂曲線，清晰展現各優化器的宏觀收斂速度與極限損失表現。
  - **子圖 (1,2)**：橫向長條圖（Bar Chart），橫向對比 5 大優化器的最終平均損失數值與實際耗費 Epoch 數，提供一目了然的效能評比。

> **關於學習曲線中「星號（$\star$）」的具體意義**：
> 當啟用 `--early-stopping`（`-e`）時，各學院的獨立 One-vs-All 分類器若在訓練中達到早停條件（即連續 `--patience` 輪改善量 $< 10^{-4}$），曲線便會停止延伸，並在**提前終止的該輪（Epoch）與當前損失值坐標上打上一顆顯著的五角星號（$\star$）**。這代表：
> 1. **訓練在此輪正式凍結**，防止在損失平原上無效空轉；
> 2. **參數在此刻自動回滾**至歷史最優權重 $\theta^*$；
> 3. 若某條曲線尾端**沒有星號**，則代表該優化器在整個 100 輪內每輪皆能維持實質進展（改善量 $> 10^{-4}$），因此順暢跑滿設定的 Horizon。

- **五大優化器收斂特徵分析（口試答辯重點）**：
  1. **Batch GD (BGD)**：曲線呈現最平滑的單調遞減軌跡，因為每一步梯度皆在全體樣本上精確計算，零雜訊，但收斂速度較慢。
  2. **Stochastic GD (SGD)**：曲線呈現輕微震盪但以極陡峭的速度快速下降，僅需數十輪即可抵達低損失區，震盪源於單一樣本梯度的方差，能協助跳脫平坦區域。
  3. **Mini-batch GD (B=32)**：在平滑度與收斂速度之間達到最佳平衡，兼具向量化計算效率與適度隨機性。
  4. **Momentum GD**：初期具備強大慣性衝力，能迅速穿越平緩坡度，收斂斜率顯著大於標準 BGD。
  5. **Adam**：結合一階動差與二階動差自適應步長調整，初期快速下降並在最少輪次內達到極致平穩收斂。

#### 2. 早停機制 (`src/utils/early_stopping.py` — `--early-stopping` / `-e`, `--patience`)
- **數學定義與演算法邏輯**：
  在迭代優化過程中，持續訓練至固定週期上限可能導致模型在損失平原上空轉或對訓練集雜訊過度擬合（Overfitting）。早停機制透過動態追蹤歷史最優對數損失 $J^*_{\text{best}}$，監控模型是否停止實質改善：
  $$\Delta J_t = J^*_{\text{best}} - J_t$$
  若在當前 Epoch $t$ 中，損失改善幅度大於設定閾值 $\epsilon_{\text{min\_delta}}$（預設 $10^{-4}$）：
  $$J^*_{\text{best}} := J_t, \quad \theta^* := \theta_t, \quad \text{patience\_counter} := 0$$
  若改善幅度未達閾值（$\Delta J_t \le \epsilon_{\text{min\_delta}}$），則耐受度計數器累加：
  $$\text{patience\_counter} := \text{patience\_counter} + 1$$
  當累計連續未實質改善次數達到設定之耐受度閾值 $P$（`--patience`，預設 10 輪）時：
  $$\text{patience\_counter} \ge P \implies \text{終止訓練 (Halt Optimization)}$$
  **關鍵保護**：提前終止時，系統自動將模型權重與偏置項回滾至歷史最佳狀態 $\theta^*$，而非保留可能已因震盪劣化或過度擬合的最終週期權重。

- **CLI 指令範例**：
  ```bash
  # 啟用早停機制（預設 patience=10）
  ./src/logreg_train.py data/dataset_train.csv -e

  # 指定優化器並設定自訂耐受度 (patience=5)
  ./src/logreg_train.py data/dataset_train.csv --opt adam -e --patience 5

  # 生成視覺化基準圖表（於 plots/ 同時輸出 learning_curves.png 與 visualization.png）
  ./src/logreg_train.py data/dataset_train.csv --visual

  # 結合早停機制與視覺化輸出：
  ./src/logreg_train.py data/dataset_train.csv -e -v
  ```

#### 4. 全維度消融與基準儀表板視覺化 (`src/utils/learning_curves.py` — `--visual` / `-v`)
- **設計理念與四大分析維度**：
  若要全方位一圖看懂優化器動力學、$L_2$ 正則化與 Early Stopping 的實質差異，系統提供 `visualize_benchmark()`，並透過 `--visual` 一鍵於 `plots/` 生成 $2 \times 2$ 高解析度儀表板 `plots/visualization.png`：
  1. **優化器收斂動力學**（Loss vs. Epochs）：對比 BGD、SGD、Mini-batch、Momentum 與 Adam 在 100 輪內的收斂速率與軌跡平滑度。
  2. **$L_2$ 權重衰減效應**（$\lambda$ vs. $\|w\|_2$ & 驗證準確率）：掃描 $\lambda \in [0, 200]$，展示雙軸曲線（權重歐氏範數嚴格單調遞減，驗證準確率在適當正則化區間維持 $\ge 98\%$）。
  3. **$L_2$ 特徵權重壓縮長條圖**（$\lambda=0$ vs. $\lambda=50$）：並排對比 13 門課程特徵權重，直觀證明無區分度的雜訊特徵（如 `Care of Magical Creatures`）被強力壓縮趨近於 0。
  4. **Early Stopping 運算節省效益**（Fixed 100 輪 vs. 早停輪數）：量化展示早停機制省下的梯度更新次數與計算百分比（如 SGD 節省逾 60% 算力且維持相同高準確率）。

> #### 💡 口試答辯必考細節（Peer Defense Q&A）
> 1. **為什麼在消融儀表板第四張圖中，SGD 提早觸發早停，而 Adam、Momentum、BGD 卻跑滿 100 輪？難道 SGD 比 Adam 優秀？**
>    - **「Epoch（訓練輪次）」$\neq$「Gradient Step（參數更新次數）」**：
>      - **1 個 Epoch 的定義**：模型完整看過整份訓練集（$m = 1280$ 筆樣本）一次。
>      - **Full-Batch 優化器（BGD / Momentum / Adam）**：每看完整批 $1280$ 筆資料，計算全量無偏梯度，**只更新 1 次參數**。跑滿 100 輪實質上總共只更新了 **$100$ 次參數**！
>      - **SGD（隨機梯度下降）**：每看 1 筆樣本就立即更新 1 次參數，**1 個 Epoch 內就更新了 $1280$ 次參數**！當 SGD 跑到第 39 輪觸發早停時，背後已經累積了 $39 \times 1280 = \mathbf{49,920\ 次}$ 參數更新！
>      - 因此在 Epoch 尺度下，SGD 在前 5 輪就更新了 6,400 次，迅速抵達谷底，但隨後因單樣本高方差雜訊在谷底「隨機漫步」，連續 5 輪改善幅度低於閾值（$\Delta J < 10^{-4}$），因而滿足早停條件提前結束。
>    - **為什麼 Adam 跑滿 100 輪？「沒停」代表它比較差嗎？**
>      - **恰恰相反，這證明 Adam 仍在穩健而有效地持續進步！**
>      - 早停機制的物理意義是「模型學不動了、邊際效益遞減（$\Delta J \le 10^{-4}$）」。實測 Adam 在第 91 到 100 輪每輪的損失改善量：
>        - `Epoch 91 | Delta = 0.000202 (> 1e-4) -> 持續有效收斂`
>        - `Epoch 95 | Delta = 0.000195 (> 1e-4) -> 持續有效收斂`
>        - `Epoch 100| Delta = 0.000186 (> 1e-4) -> 持續有效收斂`
>      - 因為 Adam 每一步都是全量樣本的精確無偏梯度，且結合自適應學習率與動量校正，每一輪都在以大於閾值的速度朝全域最優解精準下降。既然每一輪都還在持續進步，早停計數器自然不斷重置，因此它有充分的理由跑滿 100 輪。
>      - 若將輪次上限放寬至 300 輪，Adam 會在第 **145～274 輪**之間，當 Loss 精準下降到極小值（約 $0.039$）後才優雅早停。
>    - **資料集規模（$N=1600$）與凸優化特性（Convexity）的交互作用**：
>      - 邏輯回歸是嚴格凸問題，不存在局部極小值。在小資料集上，Full-Batch Adam 像精密手術刀一樣步步逼近極限理論最優點；而在工業界大數據（百萬樣本或深度神經網路）中，記憶體無法容納全量樣本，Adam 通常與 Mini-batch 結合（如 Batch Size = 32），此時 Adam 也會在數個 Epoch 內完成數百次更新並迅速早停。

---

### Bonus 5: 自動化測試與口試答辯驗證套件 (`tester/tester.py`)

為了驗證**軟體工程完整性（Code Integrity）**與**底層數學正確性（Mathematical Meaning）**，`tester/tester.py` 設計了 5 大核心模組與 `--visual` 儀表板測試（支援 `./tester/tester.py --only [describe|stats|ast|l2|e2e|visual]` 或 `./tester/tester.py --visual`，驗證 `plots/` 兩大圖表）：

#### TEST 1 (`--only describe`)：Mandatory 8 大描述性統計指標 vs. 官方 `pandas.DataFrame.describe()`
- **如何確認程式碼完整性 (Code Integrity)**：
  動態呼叫 `describe(df, include_bonus=False)` 計算 `dataset_train.csv` 全部 14 個數值特徵的 8 項基本指標（`Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max`），並與官方 `pandas.DataFrame.describe()` 進行矩陣逐元素比對（`np.allclose(rtol=1e-9, atol=1e-9)`），同時輸出**最大絕對誤差（Max Absolute Error）**與**最大相對誤差（Max Relative Error）**。
- **背後的數學意義 (Mathematical Meaning)**：
  - **貝索校正（Bessel's Correction, 自由度 $N - 1$）**：驗證樣本標準差採用無偏估計量（Unbiased Estimator）$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}$，而非母體標準差的分母 $n$。當樣本取自未知母體時，樣本均值 $\bar{x}$ 已經消耗了 1 個自由度，除以 $n - 1$ 才能使樣本變異數的期望值 $\mathbb{E}[s^2] = \sigma^2$。
  - **Hyndman–Fan Type 7 分位數線性插值法**：驗證對於排序後的有效樣本 $x_{(0)} \le \dots \le x_{(n-1)}$ 與分位點 $p \in \{0.25, 0.50, 0.75\}$，虛擬索引 $h = p(n - 1)$ 拆解為整數部 $i = \lfloor h \rfloor$ 與小數部 $f = h - i$，並滿足線性插值公式 $Q(p) = x_{(i)} + f \cdot (x_{(i+1)} - x_{(i)})$。
  - **IEEE-754 雙精度浮點數機器псилон（Machine Epsilon, $\epsilon_{\text{mach}} = 2^{-53} \approx 1.11 \times 10^{-16}$）**：在 64 位元浮點數運算中，累加順序的差異會在尾數（Mantissa）最後一位產生 $O(\epsilon_{\text{mach}})$ 的捨入差異。當特徵數量級極大（如 `Arithmancy` 的平均數 $\bar{x} \approx 4.96 \times 10^4$）時，1 ULP（Unit in the Last Place，最小有效位單位）會隨數量級放大為 $\bar{x} \cdot \epsilon_{\text{mach}} \approx 10^{-11}$。透過同時檢驗**相對誤差**（$\frac{|x_{\text{ours}} - x_{\text{ref}}|}{|x_{\text{ref}}|} \le 1.88 \times 10^{-15}$），可嚴格證明手寫統計迴圈與 Pandas 底層 C 語言在硬體浮點數精度極限下完全等價。

#### TEST 2 (`--only stats`)：Bonus 1 進階統計指標、皮爾森相關係數與退化邊界測試
- **如何確認程式碼完整性 (Code Integrity)**：
  1. 將 `src/utils/stats.py` 製作的 7 項進階指標（`var`, `range`, `iqr`, `skew`, `kurt`, `missing%`, `unique`）及 `pearson_corr()` 與 Pandas/NumPy 官方公式逐欄比對。
  2. 構造極端退化資料集（Edge Cases）：**全空值欄（All-`NaN`, $n = 0$）**、**單一元素欄（$n = 1$）**、與**常數零變異欄（$[42, 42, 42, 42], s = 0$）**，逐一測試所有統計函數，確保無任何 `ZeroDivisionError`、`IndexError` 或崩潰。
- **背後的數學意義 (Mathematical Meaning)**：
  - **樣本變異數（$s^2$）的平方尺度效應**：變異數將測量單位平方，使得 `Arithmancy` 的變異數高達 $s^2 \approx 2.78 \times 10^8$。在此數量級下，雙精度浮點數的 $1\text{ ULP} = 2.78 \times 10^8 \times 2^{-53} \approx 5.96 \times 10^{-8}$。因此絕對誤差 `5.96e-08` 換算為**相對誤差僅為 $2.15 \times 10^{-16}$**，代表手寫雙迴圈與向量化在 53 bits 尾數上僅差最後 1 個 bit。
  - **高階標準化中心動差（Central Moments）與自由度門檻**：
    - **二階動差（變異數 $s^2$ / 標準差 $s$）**：分母含 $n - 1$，故樣本數必須滿足 $n \ge 2$。
    - **三階動差（Fisher–Pearson 修正樣本偏度 $g_1$）**：
      $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$$
      分母含 $(n-1)(n-2)$ 以校正小樣本偏誤，故數學上嚴格要求 $n \ge 3$ 且 $s > 0$。
    - **四階動差（樣本超額峰度 Excess Kurtosis $g_2$）**：
      $$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$
      扣除標準常態分佈 $\mathcal{N}(0,1)$ 的基準峰度 $3$，且無偏校正項分母含 $(n-2)(n-3)$，故嚴格要求 $n \ge 4$ 且 $s > 0$。
  - **零變異數奇異點（Singularity when $s = 0$）**：當隨機變數為常數 $X = c$ 時，所有中心動差 $\mu_k = \mathbb{E}[(X - \mu)^k] = 0$，此時標準化動差 $\frac{\mu_k}{\sigma^k} = \frac{0}{0}$ 以及皮爾森相關係數 $r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{0}{0}$ 在數學上皆為未定義（Indeterminate Form）。測試確認程式能優雅攔截 $s = 0$ 並回傳 `NaN`。
  - **完全共線性證明（`pearson_corr`）**：驗證 `Astronomy` 與 `Defense Against the Dark Arts` 的皮爾森相關係數 $r = -1.000000$（相對誤差 $2.22 \times 10^{-16}$），從線性代數角度證明兩特徵構成秩為 1（Rank-1）的完全負仿射變換（$Y = -aX + b$），資訊完全重疊。

#### TEST 3 (`--only ast`)：AST 抽象語法樹靜態合規掃描（No Heavy Lifting 驗證）
- **如何確認程式碼完整性 (Code Integrity)**：
  使用 Python 內建的 `ast` 模組將 `src/describe.py` 與 `src/utils/stats.py` 解析為**抽象語法樹（Abstract Syntax Tree, `ast.parse`）**，並遍歷所有語法節點（`ast.walk`）。針對每一個函式呼叫節點（`ast.Call` 的 `func.id` 與 `func.attr`），驗證完全沒有調用任何禁用內建統計函式（`describe`, `mean`, `std`, `var`, `min`, `max`, `percentile`, `quantile`, `median`, `skew`, `kurt`, `corr`, `nunique`）。
- **背後的數學意義 (Mathematical Meaning)**：
  相較於單純的文字搜尋（`grep` 容易被註解誤導或被變數別名繞過），AST 語法樹直接檢驗程式語言的**上下文無關文法（Context-Free Grammar）推導結構**。通過 AST 靜態分析證明所有統計估計量 100% 是由最基礎的四則運算（$\sum, +, -, \times, \div$、次方與比較排序）從零建構而成。

#### TEST 4 (`--only l2`)：Sigmoid 極端值數值穩定度與 $L_2$ 正則化（Weight Decay）數學驗證
- **如何確認程式碼完整性 (Code Integrity)**：
  1. 將極端對數勝算比（Logits）$z = [-1000.0, 0.0, 1000.0]$ 輸入 `sigmoid()`，確認不產生 `OverflowError` 或 `NaN`，且精確回傳 $[0.0, 0.5, 1.0]$。
  2. 檢驗 `l2_penalty(w, lambda_=0.5, m=10)` 與解析解 $\frac{\lambda}{2m}\|w\|_2^2 = 0.725$ 完全一致。
  3. 在固定合成資料上執行 1 步批次梯度下降（1-Step BGD），對比無正則化（$\lambda = 0.0$）與強正則化（$\lambda = 10.0$），嚴格斷言兩項數學不變量（Invariants）：
     - 截距項（Bias $\theta_0$ / `b`）的更新量**完全相同**（$\Delta b = 0.000\text{e+}00$）。
     - 特徵權重向量的歐幾里得範數嚴格收縮（$\|w_{\lambda=10}\|_2 < \|w_{\lambda=0}\|_2$）。
- **背後的數學意義 (Mathematical Meaning)**：
  - **指數溢位防護（Exponential Overflow Prevention）**：在 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 中，當 $z = -1000$ 時，$e^{-z} = e^{1000}$ 會遠超 IEEE-754 雙精度浮點數上限（$\approx 1.798 \times 10^{308} \approx e^{709.78}$）。透過預先將 $z$ 截斷（Clip）於 $[-500, 500]$，使 $e^{-z} \le e^{500} \approx 1.40 \times 10^{217}$ 保持在有限浮點數範圍內，同時 $\frac{1}{1 + e^{500}}$ 在雙精度下精確收斂至漸近極限 $0.0$。
  - **為何截距項 $\theta_0$ 不懲罰，以及 $L_2$ 如何導出權重衰減（Weight Decay）**：
    加入 $L_2$ 正則化後的目標函數為 $J(\theta) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_\theta(x^{(i)}) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$。對特徵權重 $\theta_j$（$j \ge 1$）求偏導並展開梯度更新式：
    $$\theta_j^{(t+1)} = \theta_j^{(t)} - \alpha \left( \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j^{(t)} \right) = \underbrace{\left(1 - \frac{\alpha\lambda}{m}\right)}_{\text{Weight Decay 衰減因子} < 1} \theta_j^{(t)} - \frac{\alpha}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
    可見每一步迭代都會先將特徵權重向量 $w$ 向原點等比例壓縮 $(1 - \frac{\alpha\lambda}{m})$ 倍，限制決策超平面的 Lipschitz 常數（平滑度），防止共線特徵權重無限膨脹導致過度擬合（Overfitting）。反之，截距項 $\theta_0$（$j = 0$）僅負責決策超平面 $w^T x + \theta_0 = 0$ 的整體平移（決定各學院先驗基準機率），不影響對輸入特徵微小擾動的敏感度；若對 $\theta_0$ 進行懲罰，反而會引入系統性偏差（Systematic Bias）。

#### TEST 5 (`--only e2e`)：端到端多類別分類流水線、五大優化器收斂與混淆矩陣評估
- **如何確認程式碼完整性 (Code Integrity)**：
  自動備份並於測試後無損還原工作區檔案（`modele.csv`, `model_weights.csv`, `houses.csv`），透過子程序完整執行 `src/logreg_train.py` $\rightarrow$ `src/logreg_predict.py` 流水線，遍歷全部 5 種優化器（`bgd`, `sgd`, `mini`, `momentum`, `adam`）與 `--lambda 0.01`，驗證所有組合皆超越 **98.0% 準確率門檻**（實測皆達 **98.94%**），列印多類別分類報告（Classification Report）與混淆矩陣（Confusion Matrix），最後驗證針對未知測試集 `dataset_test.csv` 產生的 `houses.csv` 格式（標頭為 `Index,Hogwarts House`、共 400 筆預測、0 個 `NaN` 空值）。
- **背後的數學意義 (Mathematical Meaning)**：
  - **對數損失的全局凸性（Global Convexity）與跨優化器收斂一致性**：二元邏輯回歸損失函數的海森矩陣（Hessian Matrix）為 $H = \frac{1}{m}X^T D X$，其中對角矩陣元素 $D_{ii} = h_\theta(x^{(i)})(1 - h_\theta(x^{(i)})) > 0$。對任意非零向量 $v$，二次型 $v^T H v = \frac{1}{m}\|D^{1/2}Xv\|_2^2 \ge 0$ 恆為半正定（加入 $L_2$ 後為嚴格正定 $H + \frac{\lambda}{m}I \succ 0$）。這保證了損失曲面不存在局部極小值陷阱——無論採用全批次確定性梯度（`bgd`）、單樣本隨機梯度（`sgd`）、小批次梯度（`mini`）、指數加權動量（`momentum`）或自適應一階/二階動差預優化（`adam`），五種軌跡最終皆收斂至同一個全局最優盆地（訓練集準確率 **98.94%**，1600 名學生中精確分類 1583 人）。
  - **多類別混淆矩陣（Confusion Matrix）與 Precision / Recall / $F_1$-Score**：
    設 $4 \times 4$ 混淆矩陣為 $C$，其中 $C_{i,j}$ 代表真實學院為 $i$、經 One-vs-All 最大後驗機率決策規則 $\hat{y} = \arg\max_{k \in \{1..4\}} \sigma(x^T w^{(k)} + b^{(k)})$ 預測為學院 $j$ 的學生人數：
    $$\text{Accuracy} = \frac{\sum_{k=1}^4 C_{k,k}}{\sum_{i=1}^4 \sum_{j=1}^4 C_{i,j}}, \quad \text{Precision}_k = \frac{C_{k,k}}{\sum_{i=1}^4 C_{i,k}}, \quad \text{Recall}_k = \frac{C_{k,k}}{\sum_{j=1}^4 C_{k,j}}, \quad F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
    單看整體 Accuracy 可能掩蓋類別不平衡問題（例如 `Hufflepuff` 有 529 人，而 `Slytherin` 僅 303 人）；透過驗證四個學院各自的 $\text{Precision}_k$、$\text{Recall}_k$ 與調和平均數 $F_{1,k}$ 皆高達 $0.98 \sim 0.99$，證明分類器在四個類別上皆具備極高且均衡的鑑別力。

---

<br/>

---

<br/>

<p align="right">
  <a href="#dslr-製作細節與數學推導">
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
  - [2.2 Cost Function](#22-cost-function)
  - [2.3 Gradient Descent: Batch Gradient Descent (BGD — `--opt bgd [default]`)](#23-gradient-descent-batch-gradient-descent-bgd----opt-bgd-default)
  - [2.4 Feature Standardization](#24-feature-standardization)
- [3. Bonus Part](#3-bonus-part)
  - [Bonus 1: Extended Metrics in `src/utils/stats.py`](#bonus-1-extended-metrics-in-srcutilsstatspy)
  - [Bonus 2: Other Optimization Algorithms in `src/utils/optimizers.py` — Math & Advantages](#bonus-2-other-optimization-algorithms-in-srcutilsoptimizerspy--math--advantages)
  - [Bonus 3: $L_2$ Regularization (Ridge)](#bonus-3-l_2-regularization-ridge)
  - [Bonus 4: Learning Curve Visualization & Early Stopping](#bonus-4-learning-curve-visualization--early-stopping)
  - [Bonus 5: Automated Test & Defense Verification Suite (`tester/tester.py`)](#bonus-5-automated-test--defense-verification-suite-testertesterpy)

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
  3. **Feature Standardization**: Compute each feature's $\mu$ and $\sigma$, transforming values into a standard normal distribution.
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

### 2.2 Cost Function
Given $m$ training samples, the cost function for binary logistic regression is the Log Loss function:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \Big]$$

- When the true label is $y=1$ and $h_\theta(x) \to 0$, the loss approaches $\infty$.
- When the true label is $y=0$ and $h_\theta(x) \to 1$, the loss approaches $\infty$.
- This cost function is strictly **convex** in parameter space, guaranteeing no local minima so gradient descent converges to the global optimum.

> #### 💡 Deep Dive: Why Log Loss function Instead of Mean Squared Error (MSE)?
> Linear regression uses Mean Squared Error (MSE), but logistic regression requires log loss for three fundamental mathematical reasons:
> 1. **Guaranteed Convexity**:
>    - Plugging the non-linear Sigmoid hypothesis $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$ into MSE produces a **non-convex** loss surface riddled with local minima and flat plateaus where gradient descent gets trapped.
>    - In contrast, log loss paired with Sigmoid yields a positive semi-definite Hessian matrix, proving it is strictly **convex** (bowl-shaped) so any local minimum is the global minimum.
> 2. **Maximum Likelihood Estimation (MLE) Derivation**:
>    - Logistic regression models conditional Bernoulli probabilities: $P(y \mid x) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$.
>    - Assuming independent and identically distributed (i.i.d.) samples, taking the Negative Log-Likelihood (NLL) of the joint likelihood $L(\theta) = \prod_{i=1}^m P(y^{(i)} \mid x^{(i)})$ divided by $m$ directly yields the log loss formula. **Minimizing log loss is mathematically equivalent to maximizing the likelihood of the observed data.**
> 3. **Preventing Gradient Vanishing**:
>    - Under MSE, when a prediction is confidently wrong (e.g., $z = \theta^T x \gg 0$ while $y=0$), the Sigmoid derivative $\sigma'(z) \approx 0$ causes **gradient vanishing**, stalling weight updates.
>    - With Log Loss, the logarithm cancels the exponential derivative of Sigmoid, yielding a clean linear gradient proportional to the prediction error $(h_\theta(x) - y)$: the larger the error, the stronger the corrective gradient!

### 2.3 Gradient Descent: Batch Gradient Descent (BGD — `--opt bgd [default]`)
- **Definition**:
  Computes the exact average gradient $g_t$ across **all $m$ training samples** at each training epoch:
  $$g_t = \nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{m} X^T (h_\theta(X) - y)$$
  Simultaneous weight update rule (with learning rate $\alpha$):
  $$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \frac{\alpha}{m} X^T \left( g(X\theta) - y \right)$$
- **Advantages & Characteristics**:
  - Noise-free gradient estimation; every step moves deterministically toward the global minimum, yielding the smoothest and most stable convergence trajectory.
  - Requires scanning the entire dataset for each weight update, which carries a higher per-step computational cost on very large datasets.

### 2.4 Feature Standardization
Because Hogwarts courses have vastly different score scales (e.g., `Arithmancy` in the tens of thousands vs. `Herbology` between $-10$ and $10$), unscaled features create elongated elliptical loss contours that cause gradient descent to oscillate or diverge.
Standardization formula:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
where $\mu$ is the feature mean and $\sigma$ is the sample standard deviation.
> **Note**: Splitting the data into **80%** training data and **20%** validation data allows us to evaluate the model’s ability to generalize to unseen data. To prevent data leakage, standardization should use only the mean $\mu$ and standard deviation $\sigma$ calculated from the training set, and these same parameters should then be applied to both the training and validation sets. Data leakage refers to a situation in which a model gains access to information that should not be available during training or model selection, resulting in an evaluation that does not accurately reflect its performance on new, unseen data.

---

## 3. Bonus Part

Once the Mandatory Part is 100% compliant and achieving $\ge 98\%$ accuracy, the project supports 5 Bonus modules:
1. **Extended Statistical Metrics in `describe.py` & `stats.py`**: Adds sample Variance ($s^2$), Interquartile Range ($IQR$), Range ($Max - Min$), Skewness ($g_1$), Excess Kurtosis ($g_2$), Missing value percentage ($Missing\%$), and Unique count ($Unique$).
2. **Other Optimization Algorithms (`optimizers.py`)**: Supports BGD, SGD, Mini-Batch GD, Momentum, and Adam.
3. **$L_2$ Regularization (Ridge)**: Penalizes large weights to prevent overfitting.
4. **Learning Curve Visualization & Early Stopping**.
5. **Self-Evaluation & Defense Script (`src/evaluate.py`)**.

---

### Bonus 1: Extended Metrics in `src/utils/stats.py`

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
  Restores the dispersion measure to the original unit of measurement of the feature and serves as the scaling denominator for standardization $z = \frac{x - \bar{x}}{s}$. Implemented in `stats.py` as a 1-line vectorized Series operation: `return dfvar(dataframe) ** 0.5`.

#### 3. Range (`dfrange`)
- **Definition**:
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
- **Definition**:
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

### Bonus 2: Other Optimization Algorithms in `src/utils/optimizers.py` — Math & Advantages

In addition to the default Batch Gradient Descent (`--opt bgd`) covered in Section 2.3, we have implemented 4 advanced gradient descent optimization algorithms in `src/utils/optimizers.py` and wired them to `src/logreg_train.py` via the unified dispatcher `optimize()` and the `--opt [bgd|sgd|mini|momentum|adam]` CLI flag. Below are their mathematical definitions and key advantages:

#### 1. Stochastic Gradient Descent (SGD — `--opt sgd`)
- **Definition**:
  Randomly shuffles sample indices at each epoch and **updates weights immediately after evaluating each individual sample $(x^{(i)}, y^{(i)})$**:
  $$g_t = \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **Advantages & Characteristics**:
  - **High Update Frequency**: Performs $m$ weight updates per epoch ($1280\times$ more updates per epoch than BGD), yielding rapid initial loss reduction.
  - High variance in single-sample gradients introduces stochastic oscillations around the minimum, which can help escape local minima/saddle points in non-convex landscapes.

#### 2. Mini-batch Gradient Descent (Mini — `--opt mini`)
- **Definition**:
  Shuffles the dataset and partitions it into small batches $\mathcal{B}_k$ of size $B$ (default $B = 32$), updating weights using each mini-batch's average gradient:
  $$g_t = \frac{1}{B} \sum_{i \in \mathcal{B}_k} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} = \frac{1}{B} X_{\mathcal{B}_k}^T (h_\theta(X_{\mathcal{B}_k}) - y_{\mathcal{B}_k})$$
  $$\theta_{t+1} = \theta_t - \alpha g_t$$
- **Advantages & Characteristics**:
  - **Best of Both Worlds**: Combines the hardware-accelerated matrix vectorization of BGD with the frequent updates ($\lceil m/B \rceil = 40$ updates per epoch) of SGD, while significantly reducing gradient variance.

#### 3. Gradient Descent with Momentum (`--opt momentum`)
- **Definition**:
  Introduces a physical **velocity vector $v_t$** that computes an exponentially weighted moving average of **past gradients** (momentum decay factor $\beta = 0.9$):
  $$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
  $$\theta_{t+1} = \theta_t - \alpha v_t$$
- **Core Mechanism & Advantages**:
  - **Smooths Gradient Fluctuations**: When gradients repeatedly flip direction along a certain dimension, historical gradients and the current gradient partially cancel out, reducing violent oscillations in parameter updates.
  - **Accumulates Consistent Gradient Directions**: When gradients across consecutive steps point in a similar direction, the moving average preserves this directional signal so updates do not rely solely on a single step's gradient.
  - **Improves Convergence Efficiency**: When gradient directions fluctuate or curvature varies significantly across dimensions, Momentum dampens oscillatory updates and accelerates progress along stable directions.
  > Momentum does **not** guarantee faster updates at every step, nor does it unconditionally guarantee convergence. Its actual effectiveness depends on the learning rate, momentum coefficient, loss surface geometry, and gradient dynamics.

#### 4. Adam: Adaptive Moment Estimation (`--opt adam`)
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
# Default Mandatory Batch Gradient Descent (--opt bgd is the default)
./src/logreg_train.py data/dataset_train.csv

# Bonus Optimizers
./src/logreg_train.py data/dataset_train.csv --opt sgd
./src/logreg_train.py data/dataset_train.csv --opt mini
./src/logreg_train.py data/dataset_train.csv --opt momentum
./src/logreg_train.py data/dataset_train.csv --opt adam
```

---

### Bonus 3: $L_2$ Regularization (Ridge)

#### 0. What Is $L_2$?
- **$L$ (Lebesgue)**: Originates from the **$L^p$ Lebesgue space** in functional analysis, named after French mathematician Henri Lebesgue. In mathematics, $L$ represents function spaces defined via Lebesgue integration and norm metrics.
- **$2$ (Order $p=2$ / Sum of Squares)**: Denotes that vector components are squared, summed, and then square-rooted ($p=2$).
- **$L_2$ Norm (Euclidean Norm)**: In the context of finite-dimensional real parameter vectors $w \in \mathbb{R}^n$ in machine learning, the $L_2$ norm is precisely the standard **Euclidean length**:
  $$\|w\|_2 = \sqrt{\sum_{j=1}^n w_j^2} = \sqrt{w_1^2 + w_2^2 + \dots + w_n^2}$$
  In $L_2$ regularization (Ridge regression), we penalize the squared $L_2$ norm $\|w\|_2^2 = \sum_{j=1}^n w_j^2$, which is strictly smooth, convex, everywhere differentiable, and yields an elegant linear gradient penalty.

#### 1. Why Do We Need $L_2$ Regularization?
To penalize large feature weights to prevent overfitting. The predicted probability is $h_{w,b}(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$. Without regularization, three issues arise:
- **Weight Explosion**: To drive log loss toward $0$ on the training set, the model tries to make $100\%$ confident predictions ($h_{w,b}(x) \to 1$ or $0$). Because Sigmoid only reaches $1$ or $0$ as its input approaches $\pm\infty$, gradient descent pushes weights $w_j$ to grow indefinitely.
- **Overfitting & Sensitivity to Noise**: When certain course weights $w_j$ become excessively large, tiny fluctuations or outliers in a test student's grades cause massive swings in $w^T x + b$, degrading test accuracy.
- **Multicollinearity Instability**: When two features are strongly correlated (e.g., `Astronomy` and `Defense Against the Dark Arts`), unregularized models can assign extreme opposing weights; $L_2$ prefers smaller, evenly distributed weights ($250^2 + 250^2 \ll 1000^2 + 0^2$), stabilizing training.

#### 2. Mathematical Formulation & "Weight Decay" Derivation
- **Step 1: Add the Squared Weight Penalty to the Cost Function** (implemented via `l2_penalty()` in `src/utils/optimizers.py`):
  $$J_{\text{reg}}(w, b) = \underbrace{J(w, b)}_{\text{Original Log Loss}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 Penalty}}$$
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
> 2. **Why is feature standardization mandatory before applying $L_2$ regularization?**
>    - $L_2$ applies a single uniform penalty $\lambda$ across $\sum w_j^2$. Without standardization, large-scale features (`Arithmancy` in the tens of thousands) naturally have tiny weights $w_j$, whereas small-scale features (`Herbology` in $[-10, 10]$) require larger weights $w_j$—causing $L_2$ to unfairly penalize small-scale features. Standardization puts all features on the same scale so $\lambda$ penalizes them fairly.
> 3. **Why does $\lambda$ show virtually no correlation with Validation Accuracy in the benchmark ablation study (`--visual`)?**
>    - **Fundamental Dichotomy: Step Function (Accuracy) vs. Smooth Probability Calibration (Log Loss)**: Classification accuracy is a 0-1 step loss that only changes when an observation crosses the decision boundary. $L_2$ regularization penalizes large weights to calibrate probabilities and guard against test variance, but does not alter hard labels unless the penalty is overwhelmingly massive.
>    - **High Signal-to-Noise Ratio & Geometric Separation**: The Hogwarts dataset contains highly separable discriminant features (`Herbology`, `Astronomy`, `Ancient Runes`, `Flying`). Because multi-class predictions rely on $\arg\max_k (w_k^T x + b_k)$, $L_2$ regularization scales down the norm $\|w\|_2$ without altering the relative angular orientation of the hyperplanes. The probabilities soften from extreme confidence ($0.999 \to 0.85$), but the relative rank ordering remains identical.
>    - **Empirical Underfitting Limits**:
>      - $\lambda = 0 \sim 2000$: Validation accuracy holds firmly at **$97.19\%$** (while $\|w\|_2$ contracts from $2.83$ to $0.62$).
>      - $\lambda = 5000$: $\|w\|_2 = 0.30$, accuracy begins to drop to **$91.88\%$**.
>      - $\lambda = 10000$: $\|w\|_2 = 0.16$, severe underfitting plunges accuracy to **$59.06\%$**.
> 4. **Since accuracy remains virtually unchanged, how does the ablation study prove that $L_2$ regularization is actually effective?**
>    - **Noise Feature Compression to ~0**: In the EDA score histograms, non-discriminative courses where all four houses share identical overlapping distributions (such as `Care of Magical Creatures` and `Arithmancy`) have their learned weights compressed under $\lambda=50$ down to a negligible $0.018$. In contrast, truly predictive courses (`Ancient Runes`, `Herbology`, `Astronomy`) resist the shrinkage penalty and retain dominant weights. This confirms that $L_2$ regularization performs continuous feature shrinkage and effective noise suppression.
>    - **Well-Calibrated Probabilities**: Without regularization, unconstrained optimization can yield overly extreme, overconfident probabilities ($99.99\%$). Under $L_2$ regularization, predicted probabilities soften to well-calibrated confidence intervals (e.g. $85\%$), dramatically boosting model robustness against real-world test outliers and measurement variance.

#### 3. CLI Usage Examples
All 5 optimizers (`batch_gd`, `stochastic_gd`, `mini_gd`, `momentum_gd`, `adam_gd`) and `optimize()` accept `l2_lambda` (default `0.0`):
```bash
# Default BGD with L2 regularization (lambda = 0.01)
./src/logreg_train.py data/dataset_train.csv --lambda 0.01

# Combine an advanced optimizer (e.g., Adam) with L2 regularization
./src/logreg_train.py data/dataset_train.csv --opt adam --lambda 0.01
```

---

### Bonus 4: Learning Curve Visualization & Early Stopping (`src/utils/early_stopping.py` & `src/utils/learning_curves.py`)

#### 1. Learning Curve Visualization (`src/utils/learning_curves.py` — `--visual` / `-v`)
- **Architectural Design**:
  When passing `--visual` (or `-v`) to `src/logreg_train.py`, the system automatically creates a `plots/` directory under the project root and benchmarks all 5 gradient descent algorithms (`BGD`, `SGD`, `Mini-batch`, `Momentum`, `Adam`) on the identical standardized feature set starting from uniform initial parameters $\theta = 0$.
  It renders a comprehensive $2 \times 3$ visualization grid saved to `plots/learning_curves.png`:
  - **Subplots (0,0) to (1,0) [4 House Panels]**: Loss vs. Epochs trajectories for Gryffindor, Hufflepuff, Ravenclaw, and Slytherin. Star markers ($\star$) indicate the exact epoch where early stopping was triggered.
  - **Subplot (1,1) [Overall Mean Loss Panel]**: Average log loss across all four houses, clearly illustrating convergence speed and asymptotic loss floors across optimizers.
  - **Subplot (1,2) [Performance Summary Panel]**: Horizontal bar chart summarizing final log loss and total epochs executed per optimizer.

> **Interpretation of Star Markers ($\star$) in Learning Curves**:
> When `--early-stopping` (`-e`) is enabled, if an individual One-vs-All house classifier satisfies the early termination criterion (no loss improvement $> 10^{-4}$ for `--patience` consecutive epochs), training ceases immediately, and a prominent star marker ($\star$) is plotted at the final epoch and loss coordinate. This indicates:
> 1. **Training was frozen** at that exact epoch, preventing idle stagnation on loss plateaus;
> 2. **Parameters were automatically rolled back** to the historical best weights $\theta^*$;
> 3. If a curve **lacks a star marker**, that optimizer made consistent progress ($> 10^{-4}$ loss delta per epoch) throughout the run, smoothly completing the full 100-epoch horizon.

- **Optimizer Convergence Characteristics (Peer-Defense Focus)**:
  1. **Batch GD (BGD)**: Monotonically decreasing, smooth loss trajectory driven by deterministic full-dataset gradients; slower to converge per unit of computational time.
  2. **Stochastic GD (SGD)**: Characterized by stochastic loss fluctuations but aggressive descent rate, breaking out of shallow plateaus in fewer total epochs.
  3. **Mini-batch GD (B=32)**: Optimal trade-off between vectorized parallel execution and stochastic gradient variance.
  4. **Momentum GD**: Accelerated convergence through directional velocity accumulation, cutting through ravines and flat loss plateaus.
  5. **Adam**: Combining first and second moment adaptations, achieving near-instant descent and steady asymptotic stabilization.

#### 2. Early Stopping Mechanism (`src/utils/early_stopping.py` — `--early-stopping` / `-e`, `--patience`)
- **Mathematical Formulation**:
  Early stopping continuously monitors the improvement in log loss relative to the best recorded loss $J^*_{\text{best}}$:
  $$\Delta J_t = J^*_{\text{best}} - J_t$$
  - If $\Delta J_t > \epsilon_{\text{min\_delta}}$ (default $10^{-4}$):
    $$J^*_{\text{best}} := J_t, \quad \theta^* := \theta_t, \quad \text{patience\_counter} := 0$$
  - If $\Delta J_t \le \epsilon_{\text{min\_delta}}$:
    $$\text{patience\_counter} := \text{patience\_counter} + 1$$
  - When $\text{patience\_counter} \ge P$ (`--patience`, default 10):
    Optimization is halted immediately, and the parameter vector is reverted to the historical best $\theta^*$ to safeguard against noisy oscillations or overfitting on loss plateaus.

#### 3. CLI Usage Examples
```bash
# Enable early stopping with default patience (10 epochs)
./src/logreg_train.py data/dataset_train.csv -e

# Combine an advanced optimizer with custom patience
./src/logreg_train.py data/dataset_train.csv --opt adam --early-stopping --patience 5

# Generate both visual benchmarks in plots/ (learning_curves.png & visualization.png)
./src/logreg_train.py data/dataset_train.csv --visual

# Combine early stopping with visual benchmarks
./src/logreg_train.py data/dataset_train.csv -e -v
```

#### 4. Comprehensive Benchmark & Ablation Dashboard (`src/utils/learning_curves.py` — `--visual` / `-v`)
- **Architectural Scope & 4 Visual Perspectives**:
  To synthesize the multidimensional interplay between Optimizers, $L_2$ regularization, and Early Stopping into a single publication-ready figure, `visualize_benchmark()` generates `plots/visualization.png` ($2 \times 2$ grid):
  1. **Optimizer Convergence Dynamics** (Loss vs. Epochs): Compares descent speeds and stability curves across BGD, SGD, Mini-batch, Momentum, and Adam.
  2. **$L_2$ Regularization Weight Decay vs. Validation Accuracy**: Twin-axis curve mapping $\lambda \in [0, 200]$ to Euclidean weight norm $\|w\|_2$ decay and multi-class accuracy.
  3. **$L_2$ Feature Weight Compression** ($\lambda=0$ vs. $\lambda=50$): Grouped horizontal bar chart showing how weights across all 13 subjects compress, shrinking non-informative noise features (such as `Care of Magical Creatures`) to near zero.
  4. **Early Stopping Efficiency & Compute Savings**: Grouped bar chart comparing fixed 100 epochs against early stopped epochs, reporting the exact percentage of compute time saved.

> #### 💡 Key Peer Defense Questions (Q&A)
> 1. **Why does SGD trigger Early Stopping early while Adam, Momentum, and BGD run all 100 epochs? Does that mean SGD is superior?**
>    - **"Epoch" $\neq$ "Gradient Update Step"**:
>      - **Epoch**: One complete pass through all $m = 1280$ training samples.
>      - **Full-Batch Optimizers (BGD, Momentum, Adam)**: Execute exactly **1 gradient update per epoch** (100 total parameter updates across 100 epochs).
>      - **SGD**: Updates parameters after every single sample (**1,280 updates per epoch**). By epoch 39, SGD has executed $39 \times 1280 = \mathbf{49,920\ parameter\ updates}$!
>      - In terms of epochs, SGD rapidly dives into the loss basin within the first 5 epochs (6,400 updates), but subsequently wanders randomly around the minimum due to high sample variance. When consecutive epoch improvements drop below $\Delta J < 10^{-4}$ for 5 epochs, early stopping triggers.
>    - **Why does Adam run all 100 epochs? Does continuing mean it is slower?**
>      - **Quite the contrary: Adam is actively and steadily making meaningful improvements!**
>      - Early stopping halts only when marginal gain plateaus ($\Delta J \le 10^{-4}$). Empirical logging reveals Adam's progress in late epochs:
>        - `Epoch 91 | Delta = 0.000202 (> 1e-4) -> Continuous convergence`
>        - `Epoch 95 | Delta = 0.000195 (> 1e-4) -> Continuous convergence`
>        - `Epoch 100| Delta = 0.000186 (> 1e-4) -> Continuous convergence`
>      - Because each update leverages clean, unbiased full-dataset gradients with adaptive second moments, Adam continues descending with step deltas $> 10^{-4}$, resetting the patience counter each time.
>      - If given a 300-epoch horizon, Adam halts gracefully between epochs **145 and 274** once loss reaches its true global minimum ($\approx 0.039$).
>    - **Dataset Scale ($N=1600$) and Convexity Dynamics**:
>      - Logistic regression is strictly convex (no local minima). On small-to-medium datasets, Full-Batch Adam acts like a precision scalpel. In large-scale deep learning where full-batching is memory-prohibitive, Adam is coupled with mini-batches, where it also completes hundreds of steps and triggers early stopping within a few epochs.

---

### Bonus 5: Automated Test & Defense Verification Suite (`tester/tester.py`)

To rigorously validate both **software engineering integrity** and **mathematical correctness**, `tester/tester.py` is structured into 5 core test groups plus `--visual` dashboard generation (`./tester/tester.py --only [describe|stats|ast|l2|e2e|visual]` or `./tester/tester.py --visual` verifying both charts in `plots/`):

#### TEST 1 (`--only describe`): Mandatory Descriptive Statistics vs. Official `pandas.DataFrame.describe()`
- **How It Confirms Code Integrity**:
  Dynamically calls `describe(df, include_bonus=False)` on all 14 numeric columns of `dataset_train.csv` and compares each of the 8 mandatory metrics (`Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max`) against `pandas.DataFrame.describe()` using `np.allclose(rtol=1e-9, atol=1e-9)`. It prints both the **Maximum Absolute Error** ($\max |x_{\text{ours}} - x_{\text{ref}}|$) and **Maximum Relative Error** ($\max \frac{|x_{\text{ours}} - x_{\text{ref}}|}{|x_{\text{ref}}|}$).
- **Mathematical Meaning**:
  - **Bessel's Correction ($N - 1$)**: Confirms that sample standard deviation uses the unbiased estimator $s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}$ rather than the biased population denominator $n$.
  - **Hyndman–Fan Type 7 Linear Quantile Interpolation**: Confirms that for any quantile $p \in \{0.25, 0.50, 0.75\}$ on sorted valid observations $x_{(0)} \le \dots \le x_{(n-1)}$, the virtual index $h = p(n - 1)$ is decomposed into integer part $i = \lfloor h \rfloor$ and fractional part $f = h - i$, evaluating $Q(p) = x_{(i)} + f \cdot (x_{(i+1)} - x_{(i)})$.
  - **IEEE-754 `float64` Machine Epsilon ($\epsilon_{\text{mach}} = 2^{-53} \approx 1.11 \times 10^{-16}$)**: In 64-bit floating-point arithmetic, summation order affects the least significant bits by $O(\epsilon_{\text{mach}})$. For large-magnitude features like `Arithmancy` ($\bar{x} \approx 4.96 \times 10^4$), 1 ULP (Unit in the Last Place) scales with the magnitude ($\bar{x} \cdot \epsilon_{\text{mach}} \approx 10^{-11}$). Measuring **Relative Error** ($\le 1.88 \times 10^{-15}$) proves that our from-scratch loops match C-level Pandas routines down to hardware floating-point limits.

#### TEST 2 (`--only stats`): Bonus 1 Extended Statistical Metrics, Pearson's $r$ & Degenerate Edge Cases
- **How It Confirms Code Integrity**:
  1. Compares all 7 Bonus 1 metrics (`var`, `range`, `iqr`, `skew`, `kurt`, `missing%`, `unique`) and `pearson_corr()` against Pandas/NumPy reference values across all 14 features.
  2. Feeds degenerate inputs into every function in `src/utils/stats.py`—an **all-`NaN` column** ($n = 0$), a **single observation** ($n = 1$), and a **constant column** ($[42, 42, 42, 42]$ where $s = 0$)—confirming zero `ZeroDivisionError`, `IndexError`, or `ValueError` crashes.
- **Mathematical Meaning**:
  - **Scale of Sample Variance ($s^2$)**: Because variance squares the units of measurement, `Arithmancy` has $s^2 \approx 2.78 \times 10^8$. At this scale, $1\text{ ULP} = 2.78 \times 10^8 \times 2^{-53} \approx 5.96 \times 10^{-8}$. An absolute discrepancy of $5.96 \times 10^{-8}$ corresponds to a **relative error of $2.15 \times 10^{-16}$**, which is literally a 1-bit difference in the 53-bit IEEE-754 mantissa.
  - **Higher-Order Standardized Central Moments & Degrees-of-Freedom Thresholds**:
    - **Sample Variance / Std ($k=2$)**: Requires $n \ge 2$ so that $n - 1 > 0$.
    - **Adjusted Fisher–Pearson Sample Skewness ($g_1, k=3$)**:
      $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$$
      Requires $n \ge 3$ (since $n - 2$ is in the denominator) and $s > 0$.
    - **Sample Excess Kurtosis ($g_2, k=4$)**:
      $$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$
      Subtracts the normal baseline kurtosis of $3$ and requires $n \ge 4$ (since $n - 3$ is in the denominator) and $s > 0$.
  - **Zero-Variance Singularity ($s = 0$)**: For a constant random variable $X = c$, all central moments $\mu_k = \mathbb{E}[(X - \mu)^k] = 0$. Consequently, standardized moments $\frac{\mu_k}{\sigma^k} = \frac{0}{0}$ and Pearson correlation $r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{0}{0}$ are mathematically undefined (`NaN`), which our edge-case guards explicitly enforce.
  - **Perfect Collinearity Detection**: Verifies $r(\text{Astronomy}, \text{Defense Against the Dark Arts}) = -1.000000$, proving that the two features lie on an exact negative affine line ($Y = -aX + b$) with rank-1 redundancy.

#### TEST 3 (`--only ast`): Static AST Compliance Audit ("No Heavy Lifting")
- **How It Confirms Code Integrity**:
  Parses the source files `src/describe.py` and `src/utils/stats.py` into Python's **Abstract Syntax Tree (`ast.parse`)** and walks every syntax node (`ast.walk`). It inspects all function call nodes (`ast.Call` via `func.id` and `func.attr`) to verify that zero forbidden aggregators (`describe`, `mean`, `std`, `var`, `min`, `max`, `percentile`, `quantile`, `median`, `skew`, `kurt`, `corr`, `nunique`) exist anywhere in executable code.
- **Mathematical Meaning**:
  Unlike regex or text search (which can be bypassed by variable aliasing or triggered falsely by comments), an AST represents the formal context-free grammar derivation tree of the source code. Passing the AST audit proves that every statistical quantity in our codebase is constructed strictly from primitive algebraic operations ($+, -, \times, \div$, power, and comparison sorting).

#### TEST 4 (`--only l2`): Sigmoid Numerical Stability & $L_2$ Regularization (Weight Decay) Proof
- **How It Confirms Code Integrity**:
  1. Evaluates `sigmoid(np.array([-1000.0, 0.0, 1000.0]))` to ensure no runtime `OverflowError` or `NaN` occurs and verifies exact outputs $[0.0, 0.5, 1.0]$.
  2. Verifies `l2_penalty(w, lambda_=0.5, m=10)` against the closed-form value $\frac{\lambda}{2m}\|w\|_2^2 = 0.725$.
  3. Runs a controlled 1-step Batch Gradient Descent update comparing $\lambda = 0.0$ against $\lambda = 10.0$ on identical synthetic data, asserting that:
     - The bias parameter $\theta_0$ (`b`) is **strictly identical** ($\Delta b = 0.000\text{e+}00$).
     - The Euclidean norm of the feature weights strictly shrinks ($\|w_{\lambda=10}\|_2 < \|w_{\lambda=0}\|_2$).
- **Mathematical Meaning**:
  - **Asymptotic Clipping of Sigmoid**: For $z = -1000$, computing $e^{-z} = e^{1000}$ directly overflows IEEE-754 `float64` ($\max \approx 1.798 \times 10^{308} \approx e^{709.78}$). By clipping $z \in [-500, 500]$ before exponentiation, $e^{-z} \le e^{500} \approx 1.40 \times 10^{217}$ stays safely finite while $\frac{1}{1 + e^{500}}$ rounds to $0.0$ within `float64` precision.
  - **Why Bias $\theta_0$ Is Excluded & How $L_2$ Induces Weight Decay**:
    With the $L_2$-regularized objective $J(\theta) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_\theta(x^{(i)}) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$, taking the partial derivative with respect to feature weight $\theta_j$ ($j \ge 1$) yields:
    $$\theta_j^{(t+1)} = \theta_j^{(t)} - \alpha \left( \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j^{(t)} \right) = \underbrace{\left(1 - \frac{\alpha\lambda}{m}\right)}_{\text{Weight Decay Factor}} \theta_j^{(t)} - \frac{\alpha}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
    Each gradient step contracts the weight vector $w$ toward the origin by $(1 - \frac{\alpha\lambda}{m}) < 1$, shrinking its $L_2$ norm $\|w\|_2$ to prevent overfitting and collinear weight explosion. Conversely, the intercept $\theta_0$ ($j = 0$) only translates the decision boundary hyperplane $w^T x + \theta_0 = 0$ without affecting its orientation or sensitivity to input perturbations; keeping $\frac{\partial}{\partial \theta_0}\left(\frac{\lambda}{2m}\|w\|_2^2\right) = 0$ prevents artificial bias in the baseline log-odds.

#### TEST 5 (`--only e2e`): End-to-End Multi-Class Pipeline, Optimizer Convergence & Confusion Matrix
- **How It Confirms Code Integrity**:
  Safely backs up and restores existing workspace files (`modele.csv`, `model_weights.csv`, `houses.csv`), executes `src/logreg_train.py` and `src/logreg_predict.py` via subprocess across all 5 optimizers (`bgd`, `sgd`, `mini`, `momentum`, `adam`) plus `--lambda 0.01`, checks that every configuration exceeds the **98.0% accuracy threshold** (reaching **98.94%** on `dataset_train.csv`), prints the multi-class Classification Report and Confusion Matrix, and validates `houses.csv` generated from `dataset_test.csv` (exact `Index,Hogwarts House` header, $400$ rows, $0$ `NaN`s, all predictions valid house names).
- **Mathematical Meaning**:
  - **Global Convexity of Log Loss Across Optimizers**: The Hessian matrix of log loss is $H = \frac{1}{m}X^T D X$ where $D = \text{diag}(h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))) \succ 0$. Because $v^T H v = \frac{1}{m}\|D^{1/2}Xv\|_2^2 \ge 0$ for all $v$, the loss surface is **convex** (and strictly convex under $L_2$ regularization). Thus, whether we follow full-batch deterministic gradients (`bgd`), noisy single-sample gradients (`sgd`), mini-batch estimates (`mini`), exponentially weighted velocity (`momentum`), or adaptive first/second moment preconditioning (`adam`), all trajectories converge to the same global optimum basin ($\approx 98.94\%$ accuracy).
  - **Multi-Class Evaluation Metrics (Precision, Recall, $F_1$ & Confusion Matrix)**:
    Given the $4 \times 4$ Confusion Matrix $C$ where $C_{i,j}$ is the number of students from true house $i$ classified into house $j$ via $\hat{y} = \arg\max_{k \in \{1..4\}} \sigma(x^T w^{(k)} + b^{(k)})$:
    $$\text{Accuracy} = \frac{\sum_{k=1}^4 C_{k,k}}{\sum_{i=1}^4 \sum_{j=1}^4 C_{i,j}}, \quad \text{Precision}_k = \frac{C_{k,k}}{\sum_{i=1}^4 C_{i,k}}, \quad \text{Recall}_k = \frac{C_{k,k}}{\sum_{j=1}^4 C_{k,j}}, \quad F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
    Inspecting $\text{Precision}_k$, $\text{Recall}_k$, and $F_{1,k} \approx 0.99$ across all 4 houses confirms that the classifier achieves balanced discriminative power across both majority (`Hufflepuff`, $n=529$) and minority (`Slytherin`, $n=303$) classes.
