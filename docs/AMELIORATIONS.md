# Améliorations possibles (hors livraison actuelle)

Décision : livrer au plus vite un modèle capable de reconnaître les visages ; ce qui suit est volontairement mis de côté (et hors du package).

| Amélioration | Pourquoi | Effort estimé |
|---|---|---|
| **Anti-usurpation (détection de vivacité)** | Aujourd'hui une photo ou une vidéo de la personne devant la caméra peut être acceptée. Piste : challenge aléatoire (clignement, tourner la tête) ou modèle dédié (MiniFASNet). | moyen à élevé |
| **Recalibrer le seuil** avec de vraies photos | Le seuil 0,23 vient du jeu public LFW (facile) ; sur votre webcam et vos utilisateurs il peut être trop bas ou trop haut. Mesurer faux accepts / faux rejets. | faible (quelques photos + un script) |
| **Tests Docker et Linux** | Seul Windows + Python 3.13 a été testé ; Docker n'était pas démarré. | faible |
| **Licence des poids** | Les poids ArcFace (InsightFace) sont non commerciaux. Repli : SFace (Apache 2.0), ~0,2 point de précision en moins. | faible (changer de modèle + benchmark déjà prêt) |
| **Plusieurs images par tentative** | Réduit les faux rejets (mauvaise image isolée). | faible |
| **Mesure de charge** (plusieurs workers) | Débit réel non mesuré. | faible |
| **Détecteur YOLOv5n visage** | Entraîné et évalué (mAP50 0,79), non livré : YuNet détecte aussi bien. Servirait pour le dossier ou si on veut un détecteur maison (il faudrait écrire le décodage de sa sortie ONNX). | moyen |
| **Reconnaissance sur le PC hôte** (agent natif) | Si la capture ne peut pas se faire dans le navigateur. | moyen |
| **Firmware ESP8266 + LED** | Demandé par la consigne, pas commencé ; indépendant du modèle. | moyen |
