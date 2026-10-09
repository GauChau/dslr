import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import stats as st


if len(sys.argv) < 2:
    print("Usage : python script.py fichier.csv")
    sys.exit(1)
nom_fichier = sys.argv[1]
dataset = pd.read_csv(nom_fichier, sep=",")


#vire les pas chiffres et la colonne index
matieres = dataset.select_dtypes(include="number").drop(columns="Index")
# correlation de pearson, donne des chiffres de -1 a 1, on les meilleurs sont proches de 1 ou -1
# donc on abs pour trouver les meilleurs
correl = matieres.corr(method="pearson").abs()
# on pass en nan la diagonale qui est tjrs a 1 vu que c;est les matiere correl avec elles meme
correl = correl.mask(np.eye(len(correl), dtype=bool))

# print(st.dfmax(correl).sort_values(ascending=False))
# reshape de la df pour que chaque mat ait toutes ses correl avec les autres
# id max trouve le plus grand couple d'indice, garde le 1st en cas de multiples
mat1, mat2 = correl.stack().idxmax()
# plot & show
dataset.plot.scatter(x=mat1, y=mat2)
plt.show()
