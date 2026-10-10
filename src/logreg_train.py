import argparse
import pandas as pd
import numpy as np
import describe as dc
import utils.stats as st
import matplotlib.pyplot as plt
import utils.optimizers as opt_utils
from utils.optimizers import sigmoid
import sys
import math

def model_weight_applicator(mats_weight: pd.Series, std_values: pd.DataFrame, bias: float)->pd.Series:

	# result= std_values.copy()
	# for i in result.index:
	# 	for col in mats_weight.index:
	# 		result.loc[i, col] *= mats_weight.loc[col]

	# return result.loc[:, mats_weight.index].sum(axis=1) + bias
	notes = std_values.loc[:, mats_weight.index]

	# Multiplie chaque colonne par son poids,
	# puis additionne les matières de chaque élève.
	return notes.mul(mats_weight, axis="columns").sum(axis=1) + bias

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


def holdOut(dataset: pd.DataFrame, reality: pd.DataFrame)-> pd.DataFrame:


	result = reality.join(dataset)
	train = []
	for m in reality.columns:
		group = result[result[m] == 1]
		# print(len(group),"\n")
		selection=group.sample(frac=0.8,random_state=50)
		# selection=group.sample(frac=0.8)
		train.append(selection)

	finish = pd.concat(train)
	# verification = result.drop(index=finish.index)
	# print("\n=====verif====\n",verification,len(verification),"\n=========\n")
	# print("\n====result=====\n",finish,"\n=========\n")
	return finish


def main():
	parser = argparse.ArgumentParser(description="Train logistic regression model.")
	parser.add_argument("dataset", help="Path to dataset_train.csv")
	parser.add_argument(
		"--opt",
		choices=["bgd", "sgd", "minibatch", "momentum", "adam"],
		default=None,
		help="Bonus optimization algorithm ('bgd', 'sgd', 'minibatch', 'momentum', or 'adam')",
	)
	args = parser.parse_args()

	nom_fichier = args.dataset
	dataset = pd.read_csv(nom_fichier, sep=",")

	maison = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")
	matieres = dataset.select_dtypes(include="number").drop(columns="Index").columns
	weights = pd.DataFrame(0.0, index=matieres,columns=maison)
	reality = pd.DataFrame({m: (dataset["Hogwarts House"] == m).astype(int) for m in maison})

	learning_rate = 0.07
	bias = pd.Series(0.0, index=maison)

	hold_df = holdOut(dataset.select_dtypes(include="number").drop(columns="Index"),reality).sort_index()
	verification_df = reality.join(dataset.select_dtypes(include="number").drop(columns="Index")).drop(index=hold_df.index)

	described_dataset = dc.describe(hold_df)
	# print(described_dataset)

	# standardization of 80% to training
	for mats in matieres:
		hold_df[mats] = ((hold_df[mats] - described_dataset.loc["mean",mats])/described_dataset.loc["std", mats]).fillna(0)

	# standardization of the 20% verifiers
	for mats in matieres:
		verification_df[mats] = ((verification_df[mats] - described_dataset.loc["mean", mats])/ described_dataset.loc["std", mats]).fillna(0)

	# print(hold_df)

	if args.opt is not None:
		weights, bias = opt_utils.optimize(
			hold_df, weights, bias, opt=args.opt, learning_rate=learning_rate
		)
	else:
		for m in maison:
			for i in range(100):
				calcs = pd.DataFrame()
				calcs["z"] = model_weight_applicator(weights[m], hold_df,bias.loc[m])
				calcs["p"] = sigmoid(calcs["z"])
				calcs["L"] = logloss(hold_df[m], hold_df, calcs["p"])
				calcs["e"] = ecart(hold_df[m], hold_df, calcs["p"])
				gradients = weights.copy()
				grads = gradient_calculator(hold_df, weights[m], calcs["e"])
				grad_bias = calcs["e"].sum() / len(hold_df)

				weights[m] -= learning_rate * grads
				bias.loc[m] -= learning_rate * grad_bias

				gradient_max = max(grads.abs().max(), abs(grad_bias))

				if gradient_max < 1e-4:
					print("Convergence atteinte")
					break

	# print(weights)


	for m in maison:
		# Prédictions avec les poids et le biais appris
		z = model_weight_applicator(weights[m], verification_df, bias.loc[m])
		p = sigmoid(z)

		# 1 = Gryffindor, 0 = pas Gryffindor
		predictions = (p >= 0.5).astype(int)
		attendu = verification_df[m]

		accuracy = (predictions == attendu).mean()
		perte = logloss(attendu, verification_df, p).mean()

		# print(f"Prédictions correctes : {accuracy:.1%}")
		# print(f"Logloss moyenne : {perte:.4f}")

		# Référence : répondre toujours « pas Gryffindor »
		reference = (attendu == 0).mean()
		# print(f"Score sans distinction des élèves : {reference:.1%}")


	weights.loc["bias"] = bias

	weights.to_csv("model_weights.csv")


	statistiques = described_dataset.loc[["mean", "std"], matieres].T

	parametres = weights.join(statistiques)
	print(parametres)
	parametres.to_csv("modele.csv")


if __name__ == "__main__":
	main()
