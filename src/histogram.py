import pandas as pd
import matplotlib.pyplot as plt
import sys
import utils.stats as stats
import describe


# def histo(dataframe: pd.dataframe, col: str) :
# 	bruh = dataframe.hist(column=col)
# 	plt.show()
# 	return bruh



if len(sys.argv) < 2:
    print("Usage : python script.py fichier.csv")
    sys.exit(1)
nom_fichier = sys.argv[1]
dataset = pd.read_csv(nom_fichier, sep=",")

# histo(describe.describe(dataset,"std"))
# dscbed = describe.describe(dataset).T
# # stded = dscbed.loc["std"]
# # dscbed.loc["std"].hist()
# # print(dscbed.T)
# hist = dscbed.hist(column = "std")
# plt.show()
# print(histo(dataset))


# dscbed = describe.describe(dataset)
gryfindor = dataset[dataset["Hogwarts House"] == "Gryffindor"].copy()
hufflepuff = dataset[dataset["Hogwarts House"] == "Hufflepuff"].copy()
slytherin = dataset[dataset["Hogwarts House"] == "Slytherin"].copy()
ravenclaw = dataset[dataset["Hogwarts House"] == "Ravenclaw"].copy()

notes = [
    gryfindor["Muggle Studies"].dropna(),
    hufflepuff["Muggle Studies"].dropna(),
    slytherin["Muggle Studies"].dropna(),
    ravenclaw["Muggle Studies"].dropna(),
]

# plt.hist(notes)
# plt.hist(
#     notes,
#     bins=20,
#     # density=True,
#     # histtype="step",
#     color=["red", "gold", "blue", "green"],
#     label=["Maison 1", "Maison 2", "Maison 3", "Maison 4"],
# )
# dscbed.hist()
# print(describe.describe(gryfindor))
# print(describe.describe(hufflepuff))
# print(describe.describe(slytherin))
# print(describe.describe(ravenclaw))

# ax = dscbed.plot.bar(y="mean", legend=False, figsize=(12, 5))
# ax.set_xlabel("Matières")
# ax.set_ylabel("Écart type")

# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()


matieres = [
    col for col in dataset.select_dtypes(include="number").columns
    if col != "Index"
]

maisons = [
    ("Gryffindor", gryfindor, "red"),
    ("Hufflepuff", hufflepuff, "gold"),
    ("Slytherin", slytherin, "green"),
    ("Ravenclaw", ravenclaw, "blue"),
]

nb_colonnes = 3
nb_lignes = (len(matieres) + nb_colonnes - 1) // nb_colonnes
fig, axes = plt.subplots(nb_lignes, nb_colonnes, figsize=(15, 4 * nb_lignes))

for ax, matiere in zip(axes.flat, matieres):
    notes = [df[matiere].dropna() for _, df, _ in maisons]
    ax.hist(
        notes,
        bins=20,
        density=True,
        # histtype="step",
        color=[couleur for _, _, couleur in maisons],
        label=[nom for nom, _, _ in maisons],
    )
    ax.set_title(matiere)
    ax.legend(fontsize=7)

for ax in list(axes.flat)[len(matieres):]:
    ax.set_visible(False)

fig.tight_layout()
plt.show()