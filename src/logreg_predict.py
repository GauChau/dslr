#!/usr/bin/env python3
import pandas as pd
import numpy as np
import describe as dc
import utils.stats as st
import matplotlib.pyplot as plt
import sys
import math
import logreg_train as lt

if len(sys.argv) < 3:
	print("Usage : script.py dataset.csv modele.csv >> houses.csv")
	sys.exit(1)
dataset_file = sys.argv[1]
# dataset = pd.read_csv(dataset_file, sep=",")
try:
    dataset = pd.read_csv(dataset_file, sep=",")
except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
    sys.exit(f"Erreur de lecture du CSV : {e}")

weights_file = sys.argv[2]
try:
	weights = pd.read_csv(weights_file, sep=",", index_col=0)
except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
    sys.exit(f"Erreur de lecture du CSV : {e}")


try:
	matieres = weights.index.drop("bias")
	dataset = dataset.loc[:, ["Index", *matieres]].copy()
	maison = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")

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

except:
	print("Corrupted .csv")