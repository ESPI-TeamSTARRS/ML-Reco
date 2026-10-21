"""Reconnaissance d'un visage : « est-ce bien la personne enregistrée ? » (vérification 1:1).

Chaîne : détection YuNet -> alignement 112x112 -> empreinte de 512 nombres (ArcFace MobileFaceNet, ONNX, CPU)
-> similarité cosinus avec le gabarit de la personne attendue.
"""
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visage_id.alignement import aligner
from visage_id.detecteur import DetecteurVisage, Visage
from visage_id.erreurs import AucunVisage
from visage_id.gabarit import valider_gabarit

MODELE_PAR_DEFAUT = Path(__file__).parent / "modeles" / "arcface_mobilefacenet_w600k.onnx"

# Similarité minimale pour accepter. 0,23 = seuil qui laisse passer 0,1 % de faux accepts sur LFW
# (benchmark, voir docs/BENCHMARK_RECO.md). À recalibrer avec des photos de vos vrais utilisateurs.
SEUIL_PAR_DEFAUT = 0.23

# Conversion similarité -> probabilité « même personne » : régression logistique ajustée sur les 6 000 paires LFW
# (python -m app.benchmark_reco). Indicative : valable à peu près dans les conditions de LFW, avec autant de cas
# « même personne » que « personnes différentes ». Ce n'est pas une garantie de sécurité : la décision reste le seuil.
CONFIANCE_A = 28.5392
CONFIANCE_B = -6.1162


@dataclass(frozen=True)
class Verification:
    reconnu: bool
    score: float  # similarité cosinus, de -1 à 1 ; 0.0 si aucun visage
    visage_trouve: bool


@dataclass(frozen=True)
class Identification:
    """Résultat pour UN visage de l'image."""
    visage: Visage
    nom: str | None  # None = inconnu (aucun gabarit assez proche)
    score: float  # similarité avec le gabarit le plus proche (même si inconnu)

    @property
    def confiance(self) -> float:
        """Probabilité estimée (0 à 1) que ce visage soit bien celui du gabarit le plus proche."""
        return confiance(self.score)


def similarite(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux empreintes."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def confiance(score: float) -> float:
    """Convertit une similarité en probabilité estimée de « même personne » (voir CONFIANCE_A/B)."""
    x = CONFIANCE_A * score + CONFIANCE_B
    return float(1 / (1 + np.exp(-np.clip(x, -50, 50))))


class Reconnaisseur:
    """À créer UNE fois au démarrage du backend, puis à réutiliser pour toutes les requêtes.

    Utilisable depuis plusieurs threads (un verrou sérialise les calculs : ~30 ms par vérification sur CPU).
    Pour traiter plus de requêtes en parallèle, lancez plusieurs processus (workers), pas plusieurs threads.
    """

    def __init__(
        self,
        detecteur: DetecteurVisage | None = None,
        chemin_modele: Path | str = MODELE_PAR_DEFAUT,
        seuil: float = SEUIL_PAR_DEFAUT,
    ) -> None:
        chemin = Path(chemin_modele)
        if not chemin.is_file():
            raise FileNotFoundError(f"Modèle introuvable : {chemin}")
        if not -1.0 <= seuil <= 1.0:
            raise ValueError("Le seuil est une similarité cosinus, entre -1 et 1.")
        self._reseau = cv2.dnn.readNetFromONNX(str(chemin))
        self._verrou = threading.Lock()
        self._detecteur = detecteur or DetecteurVisage()
        self.seuil = seuil

    def empreinte(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Empreinte (512 float32, norme 1) du plus grand visage de l'image, ou None s'il n'y en a pas."""
        visages = self._detecteur.detecter(image_bgr)
        if not visages:
            return None
        return self._empreinte_du_visage(image_bgr, max(visages, key=lambda v: v.largeur * v.hauteur))

    def _empreinte_du_visage(self, image_bgr: np.ndarray, visage: Visage) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            aligner(image_bgr, visage), 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        with self._verrou:
            self._reseau.setInput(blob)
            brute = self._reseau.forward()[0].copy()
        return (brute / np.linalg.norm(brute)).astype(np.float32)

    def creer_gabarit(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Gabarit d'une personne = moyenne des empreintes de plusieurs photos d'elle.

        Pour le stocker : `gabarit_en_octets(gabarit)`. Lève AucunVisage si aucune photo ne contient de visage.
        Conseil : 5 à 10 photos variées (angle, lumière, expression) valent mieux que 30 quasi identiques.
        """
        empreintes = [e for e in (self.empreinte(i) for i in images_bgr) if e is not None]
        if not empreintes:
            raise AucunVisage("Aucun visage détecté dans les images fournies.")
        moyenne = np.mean(empreintes, axis=0)
        return (moyenne / np.linalg.norm(moyenne)).astype(np.float32)

    def verifier(self, image_bgr: np.ndarray, gabarit: np.ndarray) -> Verification:
        """Compare le visage de l'image au gabarit attendu (1:1, pour une connexion avec identifiant)."""
        gabarit = valider_gabarit(gabarit)
        empreinte = self.empreinte(image_bgr)
        if empreinte is None:
            return Verification(reconnu=False, score=0.0, visage_trouve=False)
        score = similarite(empreinte, gabarit)
        return Verification(reconnu=score >= self.seuil, score=score, visage_trouve=True)

    def verifier_photo(self, image_bgr: np.ndarray, photo_reference_bgr: np.ndarray) -> Verification:
        """Variante sans gabarit : « le visage de `image_bgr` est-il celui de la photo de référence ? ».

        Le backend donne directement la photo enregistrée. Plus simple, mais la photo de référence est retraitée à chaque appel
        (un peu plus lent) et une seule photo est moins fiable qu'un gabarit moyenné sur 5 à 10 photos (`creer_gabarit`).
        Lève AucunVisage si la photo de référence ne contient pas de visage.
        """
        reference = self.empreinte(photo_reference_bgr)
        if reference is None:
            raise AucunVisage("Aucun visage détecté dans la photo de référence.")
        return self.verifier(image_bgr, reference)

    def identifier(self, image_bgr: np.ndarray, gabarits: dict[str, np.ndarray]) -> list[Identification]:
        """Pour chaque visage de l'image : le nom du gabarit le plus proche s'il dépasse le seuil, sinon `nom=None`.

        Mode « qui est-ce ? » (1:N), utile pour une démo. Pour une connexion avec identifiant, préférer `verifier` (1:1),
        plus sûr : plus il y a de personnes enregistrées, plus le risque de confondre augmente.
        """
        gabarits = {nom: valider_gabarit(g) for nom, g in gabarits.items()}
        resultats = []
        for visage in self._detecteur.detecter(image_bgr):
            empreinte = self._empreinte_du_visage(image_bgr, visage)
            scores = {nom: similarite(empreinte, g) for nom, g in gabarits.items()}
            nom, score = max(scores.items(), key=lambda kv: kv[1]) if scores else (None, 0.0)
            resultats.append(Identification(visage, nom if score >= self.seuil else None, score))
        return resultats
