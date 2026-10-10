import pandas as pd


def dfcount(dataframe: pd.DataFrame) -> pd.Series:
    """Count the number of non-NaN values"""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        total = 0
        for value in dataframe[col]:
            if pd.notna(value):
                total += 1
        resultats.loc[col] = float(total)
    return resultats


def dfmean(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the arithmetic mean"""
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
    """Calculate the unbiased sample variance (ddof = 1)"""
    resultats = pd.Series(dtype="float64")
    mean = dfmean(dataframe)
    count = dfcount(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        if count[col] < 2:
            resultats.loc[col] = float("nan")
            continue

        total = 0.0
        for value in dataframe[col]:
            if pd.notna(value):
                total += (value - mean[col]) ** 2
        resultats.loc[col] = total / (count[col] - 1)
    return resultats


def dfstd(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the unbiased sample standard deviation (ddof = 1)"""
    resultats = pd.Series(dtype="float64")
    var = dfvar(dataframe)
    for col in var.index:
        resultats.loc[col] = var[col] ** 0.5 if pd.notna(var[col]) else float("nan")
    return resultats


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
    """Calculate the range (Max - Min)"""
    return dfmax(dataframe) - dfmin(dataframe)


def dfiqr(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the interquartile range (IQR = Q75 - Q25)"""
    return dfpercentile(dataframe, 75) - dfpercentile(dataframe, 25)


def dfskew(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the Fisher-Pearson coefficient of skewness"""
    resultats = pd.Series(dtype="float64")
    mean = dfmean(dataframe)
    count = dfcount(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        n = count[col]
        if n < 3:
            resultats.loc[col] = float("nan")
            continue
        m2 = 0.0
        m3 = 0.0
        for value in dataframe[col]:
            if pd.notna(value):
                diff = value - mean[col]
                m2 += diff ** 2
                m3 += diff ** 3
        m2 /= n
        m3 /= n
        resultats.loc[col] = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    return resultats


def dfkurt(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the excess kurtosis (Fisher's definition)"""
    resultats = pd.Series(dtype="float64")
    mean = dfmean(dataframe)
    count = dfcount(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        n = count[col]
        if n < 4:
            resultats.loc[col] = float("nan")
            continue
        m2 = 0.0
        m4 = 0.0
        for value in dataframe[col]:
            if pd.notna(value):
                diff = value - mean[col]
                m2 += diff ** 2
                m4 += diff ** 4
        m2 /= n
        m4 /= n
        resultats.loc[col] = (m4 / (m2 ** 2)) - 3.0 if m2 > 0 else 0.0
    return resultats


def dfmissing_pct(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the percentage of missing (NaN) values"""
    resultats = pd.Series(dtype="float64")
    total_rows = len(dataframe)
    count = dfcount(dataframe)

    for col in dataframe.select_dtypes(include="number").columns:
        if total_rows == 0:
            resultats.loc[col] = 0.0
        else:
            missing = total_rows - count[col]
            resultats.loc[col] = (missing / total_rows) * 100.0
    return resultats


def dfunique(dataframe: pd.DataFrame) -> pd.Series:
    """Count the number of distinct non-NaN values"""
    resultats = pd.Series(dtype="float64")
    for col in dataframe.select_dtypes(include="number").columns:
        seen = set()
        for value in dataframe[col]:
            if pd.notna(value):
                seen.add(value)
        resultats.loc[col] = float(len(seen))
    return resultats


def pearson_corr(s1: pd.Series, s2: pd.Series) -> float:
    """Calculate the Pearson correlation coefficient between two numerical Series"""
    valid_pairs = [
        (float(v1), float(v2))
        for v1, v2 in zip(s1, s2)
        if pd.notna(v1) and pd.notna(v2)
    ]
    n = len(valid_pairs)
    if n < 2:
        return 0.0

    mean1 = sum(p[0] for p in valid_pairs) / n
    mean2 = sum(p[1] for p in valid_pairs) / n

    num = 0.0
    den1 = 0.0
    den2 = 0.0
    for v1, v2 in valid_pairs:
        d1 = v1 - mean1
        d2 = v2 - mean2
        num += d1 * d2
        den1 += d1 ** 2
        den2 += d2 ** 2

    denom = (den1 * den2) ** 0.5
    return num / denom if denom > 0 else 0.0