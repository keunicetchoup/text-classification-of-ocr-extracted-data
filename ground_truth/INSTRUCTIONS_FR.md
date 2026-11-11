# Correction de la vérité terrain (30 pages)

C'est l'étape la plus importante du projet : sans transcription de référence, le CER et le WER n'ont aucune valeur.

## Méthode

1. Ouvrir côte à côte `images/<page>.jpg` et `to_correct/<page>.txt` (même nom).
2. Corriger le texte pour qu'il reproduise **exactement** la page imprimée.
3. Enregistrer le fichier corrigé, **même nom**, dans `verified/` (encodage UTF-8).
4. Compter environ 15 minutes par page. Faire 5 à 8 pages par jour.

## Règles (identiques pour toutes les pages)

- Recopier l'orthographe, les accents, les majuscules et la ponctuation **tels qu'imprimés**, y compris les fautes éventuelles de l'original.
- **Une ligne imprimée = une ligne de texte.** Pas de ligne vide entre les lignes d'un même paragraphe ; une ligne vide entre deux blocs (titre, exercice, paragraphe).
- Colonnes : lire la colonne de gauche entièrement, puis celle de droite.
- Tableaux : une ligne du tableau par ligne de texte, cellules séparées par ` | `.
- Formules mathématiques : écriture linéaire simple (`x^2 + 3x - 4 = 0`, `f(x) = (2x+1)/(x-3)`). Pour une formule illisible ou très complexe, écrire `[FORMULE]`.
- Figures, schémas, logos : ignorer. Texte placé dans une figure : ne le recopier que s'il est lisible.
- Mot illisible sur l'image : écrire `[?]`.
- Ne rien ajouter qui ne figure pas sur la page (pas de commentaire, pas de numéro de page inventé).

## Vérification

Une fois au moins 10 pages dans `verified/`, lancer :

```
python scripts/05_evaluate_ocr.py
```

Le script ignore automatiquement les pages pas encore corrigées.
