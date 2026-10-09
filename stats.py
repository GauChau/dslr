import pandas as pd
import sys

def dfcount(dataframe: pd.DataFrame)-> pd.Series :
	resultats = pd.Series(dtype="int64")
	for col in dataframe.columns:
		total =0
		for value in dataframe[col]:
			if pd.notna(value):
				total += 1
		resultats.loc[col] = total

	return resultats


def dfmean(dataframe: pd.DataFrame)-> pd.Series:
	resultats = pd.Series(dtype="int64")
	for col in dataframe.select_dtypes(include="number").columns:
		total=0
		nbval=0
		for value in dataframe[col]:
			if pd.notna(value):
				total += value
				nbval+= 1
		resultats.loc[col] = total / nbval if nbval > 0 else float("nan")
	return resultats


def dfstd(dataframe: pd.DataFrame) -> pd.Series:
	resultats = pd.Series(dtype="int64")
	mean = dfmean(dataframe)
	count = dfcount(dataframe)

	for col in dataframe.select_dtypes(include="number").columns:
		total=0
		if count[col] < 2:
			resultats.loc[col] = float("nan")
			continue

		for value in dataframe[col]:
			if pd.notna(value):
				total += (value - mean[col]) ** 2
		resultats.loc[col] = ((total / (count[col]-1))) **0.5 #if count[col] > 0 else float("nan")


	return resultats

def dfmin(dataframe: pd.DataFrame) -> pd.Series:
	resultats = pd.Series(dtype="int64")
	for col in dataframe.select_dtypes(include="number").columns:		
		resultats.loc[col] = dataframe[col].sort_values(ascending=True).iloc[0]
	return resultats

def dfpercentile(dataframe: pd.DataFrame, prct: int) ->pd.Series:
	#k = (P * (N + 1)) / 100
	count = dfcount(dataframe)


	resultats = pd.Series(dtype="int64")
	for col in dataframe.select_dtypes(include="number").columns:
		if count[col] == 0:
			resultats.loc[col] = float("nan")
			continue
		k = (prct * (count[col] - 1)) / 100
		sorted = dataframe[col].dropna().sort_values(ascending=True)
		kentier= int(k)
		# kentier = k

		if k- kentier == 0:
			resultats.loc[col] = sorted.iloc[kentier]

		fraction = k - kentier
		resultats.loc[col] = sorted.iloc[kentier] + fraction * (sorted.iloc[kentier+1]-sorted.iloc[kentier] )
		
	return resultats


def dfmax(dataframe: pd.DataFrame) -> pd.Series:
	resultats = pd.Series(dtype="int64")
	for col in dataframe.select_dtypes(include="number").columns:		
		resultats.loc[col] = dataframe[col].sort_values(ascending=False).iloc[0]
	return resultats