import pandas as pd
import sys
import utils.stats as stats


# nom_fichier = sys.argv[1]
# dataset = pd.read_csv(nom_fichier, sep=",")
def describe(dataset: pd.DataFrame) -> pd.DataFrame:
	numerique = dataset.select_dtypes(include="number")

	statistiques = {
		"count": stats.dfcount(numerique),
		"mean": stats.dfmean(numerique),
		"std": stats.dfstd(numerique),
		"var": stats.dfvar(numerique),
		"min": stats.dfmin(numerique),
		"25%": stats.dfpercentile(numerique, 25),
		"50%": stats.dfpercentile(numerique, 50),
		"75%": stats.dfpercentile(numerique, 75),
		"max": stats.dfmax(numerique),
		"range": stats.dfrange(numerique),
		"iqr": stats.dfiqr(numerique),
		"skew": stats.dfskew(numerique),
		"kurt": stats.dfkurt(numerique),
		"missing%": stats.dfmissing_pct(numerique),
		"unique": stats.dfunique(numerique),
	}

	return pd.DataFrame(statistiques).T

def main():
    if len(sys.argv) < 2:
        print("Usage : python script.py fichier.csv")
        sys.exit(1)
    nom_fichier = sys.argv[1]
    dataset = pd.read_csv(nom_fichier, sep=",")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.float_format", "{:.6f}".format,
    ):
        print(describe(dataset))


if __name__ == "__main__":
    main()
