import pandas as pd
import numpy as np
import describe as dc
import utils.stats as st
import matplotlib.pyplot as plt
import sys
import math
import logreg_train as lt

if len(sys.argv) < 3:
	print("Usage : python script.py fichier.csv")
	sys.exit(1)
dataset_file = sys.argv[1]
dataset = pd.read_csv(dataset_file, sep=",")
weights_file = sys.argv[2]
weights = pd.read_csv(weights_file, sep=",", index_col=0)

# matieres = dataset.select_dtypes(include="number").drop(columns="Index").columns
matieres = weights.index.drop("bias")
dataset = dataset.loc[:, ["Index", *matieres]].copy()
maison = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")

# print(weights)
# print(dataset)
for mats in matieres:
	dataset[mats] = ((dataset[mats] - weights.loc[mats,"mean"]) / weights.loc[mats,"std"]).fillna(0)


probabilites = pd.DataFrame(index=dataset.index)

for m in maison:
	z = (dataset[matieres].mul(weights.loc[matieres, m], axis="columns")
		.sum(axis=1)
		+ weights.loc["bias", m])
	probabilites[m] = 1 / (1 + np.exp(-z))

resultat = pd.DataFrame({
	"Index": dataset["Index"],
	"Hogwarts House": probabilites.idxmax(axis=1)})

resultat.to_csv("houses.csv", index=False)
print(resultat)