# Benchmark des modèles de reconnaissance faciale

Ce benchmark a été produit dans le projet d'entraînement ML-Identification (dossier `entrainement/`, commande `python -m app.benchmark_reco`), qui n'est pas nécessaire pour utiliser le package ; seuls les résultats sont donnés ici.

## Protocole

- **Jeu :** LFW (version « funneled »), protocole standard de vérification : 10 plis x (300 paires de la même personne + 300 de personnes différentes) = 6 000 paires, 7 701 images.
- **Équité :** tous les modèles reçoivent exactement les mêmes visages (détection YuNet, 5 repères, alignement 112x112 du package). YuNet a trouvé un visage sur les 7 701 images.
- **Décision :** similarité cosinus entre deux empreintes. Exactitude en validation croisée : le seuil est choisi sur 9 plis et mesuré sur le 10e.
- **Mesures de sécurité :** EER (faux accepts = faux rejets) et taux de bonnes acceptations quand on tolère 1 % ou 0,1 % de faux accepts (3 000 paires différentes : à 0,1 %, cela ne représente que 3 paires, donc c'est grossier).
- **Vitesse :** CPU, un visage à la fois, mesure bruitée (la machine faisait autre chose : MobileFaceNet est mesuré entre 4,3 et 9,2 ms selon l'essai).

## Résultats

| Modèle | Exactitude | EER | Acceptés à 1 % de faux accepts | Acceptés à 0,1 % | ms/visage | Taille | Licence des poids |
|---|---|---|---|---|---|---|---|
| SFace | 0,991 ± 0,004 | 0,011 | 0,989 | 0,981 | ~10 | 38,7 Mo | Apache 2.0 |
| **ArcFace MobileFaceNet** | **0,993 ± 0,004** | **0,009** | **0,991** | **0,988** | **~5-9** | **13,6 Mo** | InsightFace, non commercial |
| ArcFace ResNet50 | 0,996 ± 0,003 | 0,007 | 0,993 | 0,993 | ~51 | 174,4 Mo | InsightFace, non commercial |

## Décision : ArcFace MobileFaceNet

- Le ResNet50 est le plus précis, mais l'écart avec MobileFaceNet (0,3 point d'exactitude) est du même ordre que l'écart-type entre plis (0,3 à 0,4 point) : on ne peut pas affirmer qu'il est vraiment meilleur. Il est en revanche ~5 à 10 fois plus lent et 13 fois plus lourd.
- MobileFaceNet est meilleur que SFace sur toutes les mesures de précision, plus petit, et au moins aussi rapide.
- **Seuil d'acceptation par défaut : 0,23** de similarité cosinus (= 0,1 % de faux accepts sur LFW). 0,165 correspond à 1 %.

## Limites (à citer dans le dossier)

- LFW est un jeu facile et presque saturé : photos de personnalités, souvent de face, bien éclairées. Les scores y sont plus hauts que dans la vraie vie. Il ne mesure ni les jumeaux, ni les photos présentées devant la caméra (usurpation), ni votre webcam dans votre pièce.
- Le seuil 0,23 vient de LFW. **Il doit être recalibré avec des photos de vos vrais utilisateurs et d'inconnus à votre webcam.**
- Licence : les poids InsightFace sont limités à la recherche non commerciale (acceptable en projet scolaire). SFace (Apache 2.0) est la solution de repli à 0,2 point près (0,991 contre 0,993).
- 3 modèles seulement, pas d'optimisation fine ; un seul alignement (YuNet).
