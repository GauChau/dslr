import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

if len(sys.argv) < 2:
    print("Usage : python script.py fichier.csv")
    sys.exit(1)
nom_fichier = sys.argv[1]
dataset = pd.read_csv(nom_fichier, sep=",")

sns.pairplot(
    dataset.drop(columns="Index"),
    hue="Hogwarts House",
    corner=True,
    height=1.2,
    plot_kws={"s":8, "alpha":0.5},
)

plt.show()
