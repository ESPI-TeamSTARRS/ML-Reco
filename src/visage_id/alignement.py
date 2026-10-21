"""Alignement d'un visage sur le gabarit 112x112 attendu par les modèles de reconnaissance (type ArcFace)."""
import cv2
import numpy as np

from visage_id.detecteur import Visage

TAILLE = 112
# Position standard des 5 repères dans une image 112x112 (œil droit, œil gauche, nez, bouche droite, bouche gauche),
# gabarit ArcFace/InsightFace, aussi utilisé par SFace.
GABARIT = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def aligner(image_bgr: np.ndarray, visage: Visage) -> np.ndarray:
    """Retourne le visage redressé et recadré en 112x112 BGR, à partir des 5 repères détectés."""
    if len(visage.reperes) != 5:
        raise ValueError("Le visage doit porter 5 repères.")
    source = np.array(visage.reperes, dtype=np.float32)
    matrice, _ = cv2.estimateAffinePartial2D(source, GABARIT, method=cv2.LMEDS)
    if matrice is None:
        raise ValueError("Alignement impossible : repères dégénérés.")
    return cv2.warpAffine(image_bgr, matrice, (TAILLE, TAILLE), borderValue=0)
