"""Détection de visages (YuNet, ONNX, CPU) : répond à « y a-t-il un visage dans cette image ? »."""
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visage_id.erreurs import ImageInvalide

MODELE_PAR_DEFAUT = Path(__file__).parent / "modeles" / "face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class Visage:
    x: int
    y: int
    largeur: int
    hauteur: int
    score: float
    # 5 repères (œil droit, œil gauche, nez, coin droit et gauche de la bouche), en pixels : ((x, y), ...)
    reperes: tuple[tuple[float, float], ...] = ()


class DetecteurVisage:
    """Détecteur chargé une fois, réutilisable pour chaque image et depuis plusieurs threads (un verrou sérialise les appels)."""

    def __init__(self, chemin_modele: Path | str = MODELE_PAR_DEFAUT, seuil: float = 0.7) -> None:
        chemin = Path(chemin_modele)
        if not chemin.is_file():
            raise FileNotFoundError(f"Modèle introuvable : {chemin}")
        self._verrou = threading.Lock()
        self._detecteur = cv2.FaceDetectorYN.create(str(chemin), "", (320, 320), score_threshold=seuil)

    def detecter(self, image_bgr: np.ndarray) -> list[Visage]:
        """Retourne les visages trouvés, du plus sûr au moins sûr. `image_bgr` : tableau OpenCV (H, W, 3)."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ImageInvalide("L'image doit être un tableau uint8 de forme (hauteur, largeur, 3) en BGR.")
        hauteur, largeur = image_bgr.shape[:2]
        with self._verrou:
            self._detecteur.setInputSize((largeur, hauteur))
            _, resultats = self._detecteur.detect(image_bgr)
        if resultats is None:
            return []
        visages = [
            Visage(
                int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[14]),
                tuple((float(r[4 + 2 * i]), float(r[5 + 2 * i])) for i in range(5)),
            )
            for r in resultats
        ]
        return sorted(visages, key=lambda v: v.score, reverse=True)
