# Licences des modèles et dépendances

Ce package embarque deux modèles ONNX dans `src/visage_id/modeles/` :

| Fichier | Rôle | Origine | Licence |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | trouver le visage | OpenCV Zoo (YuNet) | MIT |
| `arcface_mobilefacenet_w600k.onnx` | reconnaître le visage | InsightFace (`w600k_mbf`, pack `buffalo_sc` v0.7) | **recherche non commerciale uniquement** |

**Attention :** les poids InsightFace sont réservés à la recherche non commerciale. Acceptable pour un projet scolaire ; pour un usage
commercial, remplacer par un modèle dont la licence l'autorise (par exemple SFace, OpenCV Zoo, Apache 2.0, ~0,2 point de précision en
moins sur LFW : voir `docs/BENCHMARK_RECO.md`) et recalculer les gabarits. Ne rendez pas ce dépôt public sans avoir vérifié ce point.

Dépendances : numpy (BSD), opencv-python-headless (Apache 2.0), pillow (HPND).
