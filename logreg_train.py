import pandas as pd
import numpy as np
import describe as dc
import stats as st
import matplotlib.pyplot as plt
import sys
import math

def model_weight_applicator(mats_weight: pd.Series, std_values: pd.DataFrame, bias: float)->pd.Series:

	result= std_values.copy()
	for i in result.index:
		for col in mats_weight.index:
			result.loc[i, col] *= mats_weight.loc[col]
	
	return result.loc[:, mats_weight.index].sum(axis=1) + bias

def sigmoid_calculator(z: pd.Series)->pd.Series:
	p=z.copy()
	# print(z)
	for i in z.index:
		p.loc[i] = 1 / (1 + np.exp(-z.loc[i]))
	return p

def _verifier_entrees(reality: pd.Series,data: pd.DataFrame,sigmoid: pd.Series) -> None:
	if not reality.index.equals(data.index):
		raise ValueError("reality doit avoir le même index que data.")
	if not sigmoid.index.equals(data.index):
		raise ValueError("sigmoid doit avoir le même index que data.")
	if not reality.isin([0, 1]).all():
		raise ValueError("reality doit contenir uniquement des 0 et des 1.")
	if not sigmoid.between(0, 1).all():
		raise ValueError("sigmoid doit contenir des probabilités entre 0 et 1.")

def logloss(reality: pd.Series, data: pd.DataFrame, sigmoid: pd.Series) -> pd.Series:
	"""Perte logarithmique binaire pour chaque ligne."""
	_verifier_entrees(reality, data, sigmoid)

	y = reality.astype(float)
	# Évite log(0) lorsque la sigmoïde donne exactement 0 ou 1.
	p = sigmoid.astype(float).clip(1e-15, 1 - 1e-15)

	return (-(y * np.log(p) + (1 - y) * np.log(1 - p))).rename("L")

def ecart(reality: pd.Series,data: pd.DataFrame,sigmoid: pd.Series) -> pd.Series:
	"""Écart signé p - y, utilisé pour calculer les gradients."""
	_verifier_entrees(reality, data, sigmoid)

	return (sigmoid.astype(float) - reality.astype(float)).rename("e")

def gradient_calculator(dataset: pd.DataFrame, mats_weight: pd.Series, ecart:pd.Series) -> pd.Series:
	grads = mats_weight.copy()

	for mats in mats_weight.index:
		grads.loc[mats] = (dataset[mats] * ecart).sum() / len(dataset)
	return grads



if len(sys.argv) < 2:
	print("Usage : python script.py fichier.csv")
	sys.exit(1)
nom_fichier = sys.argv[1]
dataset = pd.read_csv(nom_fichier, sep=",")

weights = pd.DataFrame()
maison = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")
matieres = dataset.select_dtypes(include="number").drop(columns="Index").columns
bias = 0.0
learning_rate = 0.01


described_dataset = dc.describe(dataset.select_dtypes(include="number").drop(columns="Index"))
standardized_dataset = dataset.select_dtypes(include="number").drop(columns="Index")

for m in maison:
	for mats in matieres:
		weights.loc[mats, m] = 0.0

for mats in matieres:
	for i, notes in standardized_dataset.iterrows():
		standardized_dataset.loc[i,mats] = (standardized_dataset.loc[i,mats] - described_dataset.loc["mean"].loc[mats])/described_dataset.loc["std"].loc[mats]

standardized_dataset = standardized_dataset.fillna(0)

reality = pd.DataFrame()
for m in maison:
	reality[m] = (dataset["Hogwarts House"] == m).astype(int)
print(reality)

# reality = (dataset["Hogwarts House"] == "Gryffindor").astype(int)
calcs = pd.DataFrame()
calcs["z"] = model_weight_applicator(weights["Gryffindor"], standardized_dataset,bias)
calcs["p"] = sigmoid_calculator(calcs["z"])
calcs["L"] = logloss(reality["Gryffindor"], standardized_dataset, calcs["p"])
calcs["e"] = ecart(reality["Gryffindor"], standardized_dataset, calcs["p"])

gradients = weights.copy()
grads = gradient_calculator(standardized_dataset, weights["Gryffindor"], calcs["e"])
grad_bias = calcs["e"].sum() / len(standardized_dataset)

weights["Gryffindor"] -= learning_rate * grads
bias -= learning_rate * grad_bias


print( "===calcs===\n",calcs, "\n======\n")
# print( "====std_data====\n",standardized_dataset, "========\n")
print(grads)
# joe= gradient_calculator(standardized_dataset,grads)

# print(weights["Gryffindor"])
# gradients(standardized_dataset,weights.iloc[:,:])















# def logloss(maison: str, data:pd.DataFrame)->pd.Series:

# 	result = data["p"].copy()
# 	for i in data.index:
# 		if data.iloc[i]["Hogwarts House"] == maison:
# 			y = 1
# 		else:
# 			y = 0
# 		p = result.iloc[i]
# 		result.iloc[i] = -(y * math.log(p) + (1 - y) * math.log(1-p))
# 	return result

# def logloss_clean(reality: pd.Series, data:pd.DataFrame, sigmoid: pd.Series)-> pd.Series:
# 	result = sigmoid.copy()
# 	for i in reality.index:
# 		y = reality.iloc[i]
# 		p = result.iloc[i]
# 		result.iloc[i] = -(y * math.log(p) + (1 - y) * math.log(1-p))
# 	return result

# def ecart(maison:str, data:pd.DataFrame)->pd.Series:
# 	result = data["p"].copy()
# 	for i in data.index:
# 			if data.iloc[i]["Hogwarts House"] == maison:
# 				y = 1
# 			else:
# 				y = 0
# 			e = result.iloc[i]
# 			result.iloc[i] = e - y
# 	return result
