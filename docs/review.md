# Review de conformité au sujet DSLR

5 novembre 2026 — `6a25d00` et modifications locales examinées.
Périmètre : `src/`, hors testeurs locaux. Référence : [sujet version 5](en.subject.pdf).
Ce rapport conserve uniquement les points ouverts.

## Standards

### [P2, règle locale] Moyennes prédéfinies

**Code :** [optimizers.py:56](../src/utils/optimizers.py#L56),
[logreg_train.py:194](../src/logreg_train.py#L194), lignes 195 et 200,
et `utils/learning_curves.py`.

`np.mean()` et `.mean()` contredisent le [README, section 3](../README.md#3-rules--constraints),
qui les interdit dans tout le projet.

**À faire :** remplacer ces moyennes par une somme divisée par l’effectif.

**Nuance :** la liste détaillée du sujet vise l’analyse descriptive (PDF, page 8).
L’interdiction d’une moyenne de perte est moins explicite ; ce seul appel ne
prouve donc pas une tricherie au regard du PDF.

## Conformité au sujet

Aucun défaut majeur démontré sur le parcours obligatoire avec les CSV fournis.
Les défauts conditionnels de visualisation et les écarts de documentation sont
présentés dans [la review détaillée](review-prof-42.md).

**Limite :** les 98 % officiels restent non vérifiables sans les véritables
étiquettes de `dataset_test.csv`.

**Vérifications déjà réalisées :** statistiques obligatoires comparées à une
référence ; cinq optimiseurs entraînés puis utilisés pour prédire, avec sorties
valides de 400 lignes. Aucun nouveau test pour cette condensation documentaire.

**Bilan : 1 constat Standards (P2) ; 0 défaut majeur de conformité confirmé sur les
CSV fournis.**
