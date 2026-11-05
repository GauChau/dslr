
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


if len(sys.argv) < 3:
	print("Usage : accuracy_score.py dataset.csv model_predictions.csv")
	sys.exit(1)
dataset_file = sys.argv[1]
try:
	dataset = pd.read_csv(dataset_file, sep=",")
except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
	sys.exit(f"Erreur de lecture du CSV : {e}")

model_result_file = sys.argv[2]
try:
	model_result = pd.read_csv(model_result_file, sep=",")
except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
	sys.exit(f"Erreur de lecture du CSV : {e}")

a, b = dataset["Hogwarts House"].align(model_result["Hogwarts House"], join="inner")
pourcentage = accuracy_score(a, b)
print("accuracy: ",pourcentage,"\n")