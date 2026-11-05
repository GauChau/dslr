# Review du code — regard de correcteur 42

5 novembre 2026 — `6a25d00` et modifications locales examinées.
Périmètre : `src/`, hors testeurs locaux. Référence : [sujet version 5](en.subject.pdf).
Uniquement les constats ouverts ; identifiants conservés pour le suivi.
P2 : défaut à corriger. P3 : amélioration de conception. Aucun P1 démontré sur
le parcours obligatoire avec les CSV fournis. Ce classement n’est pas un barème officiel.

## Standards

### S1 — [P2, règle locale] Moyennes prédéfinies

**Code :** [optimizers.py:56](../src/utils/optimizers.py#L56),
[logreg_train.py:194](../src/logreg_train.py#L194), lignes 195 et 200,
et `utils/learning_curves.py`.

`np.mean()` et `.mean()` contredisent l’interdiction générale du
[README:79](../README.md#L79). Le sujet est moins explicite pour les moyennes
calculées hors de l’analyse descriptive : ce constat n’établit pas à lui seul une tricherie.

**À faire :** utiliser une somme divisée par l’effectif.

### S4 — [P2, documentation] Médiane annoncée, moyenne utilisée

**Code :** [logreg_train.py:146](../src/logreg_train.py#L146),
[logreg_predict.py:33](../src/logreg_predict.py#L33).
**Documentation :** [walkthrough.md:575](walkthrough.md#L575).

La documentation décrit une imputation par médiane et sa sauvegarde. Le code
centre les notes puis remplace les valeurs manquantes par zéro : pour un écart
type non nul, cela revient à imputer par moyenne. Aucune médiane n’est sauvegardée.

**À faire :** aligner la documentation avec la méthode retenue. La moyenne est
permise ; le problème est de défendre une méthode différente de celle du code.

### S6 — [P3, conception] Entraînement et inférence dupliqués

**Code :** [optimizers.py:68](../src/utils/optimizers.py#L68),
[logreg_train.py:145](../src/logreg_train.py#L145),
[logreg_predict.py:32](../src/logreg_predict.py#L32).

Les cinq optimiseurs répètent le cycle par maison, l’historique, l’arrêt et la
restitution. Normalisation et inférence sont aussi répétées. Le prédicteur
réimplémente notamment la sigmoïde sans la protection contre l’overflow du helper.

**À faire :** partager transformation et inférence, puis isoler le cycle commun
sans masquer les formules propres aux optimiseurs. Amélioration de maintenabilité,
pas violation formelle du sujet.

## Spec

### F1 — [P2, bonus] `--visual` peut échouer après une convergence normale

**Code :** [learning_curves.py:384](../src/utils/learning_curves.py#L384),
[logreg_train.py:177](../src/logreg_train.py#L177).

Le graphique lit exactement 100 points par maison, mais la tolérance de gradient
peut arrêter un optimiseur plus tôt, même sans early stopping : `IndexError`.
Le graphique précède l’export, donc son échec empêche aussi la création du modèle.

**Reproduction :** 40 lignes, 10 par maison, une matière alternant −1/+1 avec
la même distribution dans chaque maison, et quatre cibles one-hot.
`visualize_benchmark(df, verification_df=df)` échoue lorsqu’Adam converge avant 100 epochs.
Ce plantage n’a pas été établi sur les CSV fournis.

**À faire :** utiliser les longueurs réelles des historiques ou prolonger les
trajectoires terminées ; exporter le modèle avant les graphiques facultatifs.

### F2 — [P2, cas limite] Conclusion d’homogénéité erronée

**Code :** [histogram.py:54](../src/histogram.py#L54), notamment ligne 57.

Si les écarts types par maison sont nuls, le score vaut zéro, même si leurs
moyennes diffèrent. Trois notes identiques par maison, respectivement 0, 10, 20
et 30, donnent ainsi le meilleur score malgré une séparation complète.
Plus généralement, comparer les moyennes ne suffit pas à comparer les distributions.

**Impact :** classement, conclusions et titres ; les barres utilisent toujours les
notes originales. Aucun plantage ni mauvais choix sur les CSV fournis n’est démontré.

**À faire :** traiter ce cas limite et limiter la portée du score, ou supprimer
la détection automatique et ses annotations. Le sujet exige un histogramme
permettant de répondre à la question : une lecture visuelle justifiée suffit.

## Vérifications et limites

Vérifications antérieures : cinq optimiseurs entraînés puis utilisés pour prédire,
sorties valides de 400 lignes ; cas synthétiques F1 et F2 reproduits directement.
Les 98 % officiels restent non vérifiables sans les étiquettes du jeu de test.
Aucun nouveau test pour cette condensation ; aucun code source modifié.

**Bilan : Standards — 3 constats (S1, S4, S6), priorité maximale P2 ;
Spec — 2 constats conditionnels (F1, F2), priorité maximale P2.**
