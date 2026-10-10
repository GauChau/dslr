import pandas as pd


def dfcount(dataframe: pd.DataFrame) -> pd.Series:
    """Count the number of non-NaN values."""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        total = 0
        for value in dataframe[col]:
            if pd.notna(value):
                total += 1
        resultats.loc[col] = float(total)
    return resultats


def dfmean(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the arithmetic mean."""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        total = 0.0
        nbval = 0
        for value in dataframe[col]:
            if pd.notna(value):
                total += value
                nbval += 1
        resultats.loc[col] = total / nbval if nbval > 0 else float("nan")
    return resultats


def dfvar(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the unbiased sample variance (ddof = 1)."""
    res = pd.Series(dtype="float64")
    mean = dfmean(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        s = dataframe[col].dropna()
        n = len(s)
        if n < 2:
            res.loc[col] = float("nan")
            continue

        diff = s - mean[col]
        res.loc[col] = sum(diff ** 2) / (n - 1)
    return res


def dfstd(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the unbiased sample standard deviation (ddof = 1)."""
    return dfvar(dataframe) ** 0.5


def dfmin(dataframe: pd.DataFrame) -> pd.Series:
    """Find the minimum value for each numerical column in O(N) time."""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        min_val = None
        for value in dataframe[col]:
            if pd.notna(value):
                if min_val is None or value < min_val:
                    min_val = value
        resultats.loc[col] = float(min_val) if min_val is not None else float("nan")
    return resultats


def dfmax(dataframe: pd.DataFrame) -> pd.Series:
    """Find the maximum value for each numerical column in O(N) time."""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        max_val = None
        for value in dataframe[col]:
            if pd.notna(value):
                if max_val is None or value > max_val:
                    max_val = value
        resultats.loc[col] = float(max_val) if max_val is not None else float("nan")
    return resultats


def dfpercentile(dataframe: pd.DataFrame, prct: float) -> pd.Series:
    """Calculate the prct-th percentile (0 <= prct <= 100) using linear interpolation."""
    count = dfcount(dataframe)
    resultats = pd.Series(dtype="float64")

    for col in dataframe.select_dtypes(include="number").columns:
        n = int(count[col])
        if n == 0:
            resultats.loc[col] = float("nan")
            continue

        sorted_vals = dataframe[col].dropna().sort_values(ascending=True)
        k = (prct * (n - 1)) / 100.0
        kentier = int(k)
        fraction = k - kentier

        if fraction == 0.0 or kentier + 1 >= n:
            resultats.loc[col] = float(sorted_vals.iloc[kentier])
        else:
            lower = sorted_vals.iloc[kentier]
            upper = sorted_vals.iloc[kentier + 1]
            resultats.loc[col] = float(lower + fraction * (upper - lower))

    return resultats


# =====================================================================
# Bonus Statistical Metrics & Helper Functions
# =====================================================================

def dfrange(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the range (Max - Min)."""
    return dfmax(dataframe) - dfmin(dataframe)


def dfiqr(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the interquartile range (IQR = Q75 - Q25)."""
    return dfpercentile(dataframe, 75) - dfpercentile(dataframe, 25)


# 分布偏右(>0)或偏左(<0)
def dfskew(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the Fisher-Pearson coefficient of skewness."""
    res = pd.Series(dtype="float64")
    mean = dfmean(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        s = dataframe[col].dropna()
        n = len(s)
        if n < 3:  # For statistical meaningfulness
            res.loc[col] = float("nan")
            continue

        diff = s - mean[col]
        m2 = sum(diff ** 2) / n  # Second central moment, for standardization
        m3 = sum(diff ** 3) / n  # Third central moment
        res.loc[col] = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    return res
# m3 Third central moment:
#     立方保留正負號，而且會放大絕對值大的偏差，因此對分布兩側的不平衡特別敏感。
#     (但偏態係數不代表分布一定對稱，可能正負方向的三次偏差剛好抵消)
# Pandas 的向量化運算:  ie. diff = s - mean[col], s = dataframe[col].dropna()
#     在 Python 中，如果用 for 迴圈逐行處理資料（像是 for value in dataframe[col]:），
#     Python 必須一邊執行迴圈、一邊檢查型態、一邊做加減乘除，速度會非常慢。
#     向量化運算則是利用底層由 C 語言或 Fortran 編寫的高效程式碼（透過 NumPy），
#     一次性地對整個陣列（Array）或欄位進行平行計算。


# 超額峰度：相比於常態分佈，資料分佈的「尖銳程度」的差異。
# → 比常態分佈極端值出現的機率更高/低。
# 峰度（kurtosis）是用來衡量資料分佈的「尾端厚度（tails）」與「尖銳程度」的指標，
#   它描述的是資料極端值（outliers）出現的機率高低。
# 標準的常態分佈（normal distribution）其峰度固定等於 3。
#   為了方便比較，統計學家將峰度減去 3，這個差值就稱為超額峰度：
#       excess kurtosis = kurtosis - 3
# 高峰分布：極端值出現的機率比常態分佈高。
# 平坦分布：極端值出現的機率比常態分佈低。
# 公式中使用 4 次方，使其對距離平均數很遠的「極端值」非常敏感。
def dfkurt(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the excess kurtosis (Fisher's definition)."""
    res = pd.Series(dtype="float64")
    mean = dfmean(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        s = dataframe[col].dropna()
        n = len(s)
        if n < 4:
            res.loc[col] = float("nan")
            continue

        diff = s - mean[col]
        m2 = sum(diff ** 2) / n
        m4 = sum(diff ** 4) / n
        res.loc[col] = (m4 / (m2 ** 2)) - 3.0 if m2 > 0 else 0.0
    return res


def dfmissing_pct(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the percentage of missing (NaN) values."""
    total_rows = len(dataframe)
    if total_rows == 0:
        return pd.Series(
            0.0,
            index=dataframe.select_dtypes(include="number").columns
        )  # 索引是所有數值欄位名稱，數值全部填 0.0 的 pd.Series
    return ((total_rows - dfcount(dataframe)) / total_rows) * 100.0


def dfunique(dataframe: pd.DataFrame) -> pd.Series:
    """Count the number of distinct non-NaN values."""
    res = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        # Set本身不允許重複元素，所以重複出現的數值會自動被過濾。
        res.loc[col] = float(len(set(dataframe[col].dropna())))
    return res


# 衡量兩組連續變數之間線性相關程度
def pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    """Calculate the Pearson correlation coefficient between two numerical Series."""
    mask = s1.notna() & s2.notna()
    x = s1[mask]
    y = s2[mask]
    n = len(x)
    if n < 2:
        return 0.0

    d1 = x - (sum(x) / n)
    d2 = y - (sum(y) / n)

    denom = (sum(d1 ** 2) * sum(d2 ** 2)) ** 0.5
    return float(sum(d1 * d2) / denom) if denom > 0 else 0.0
