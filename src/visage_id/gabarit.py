"""Sérialisation d'un gabarit pour le stocker en base (colonne binaire / bytea).

Format : MAGIC (4) | version du format (1) | empreinte du modèle (16) | 512 float32 little-endian (2048) = 2069 octets.
L'empreinte du modèle lie le gabarit au modèle qui l'a calculé : si le modèle change un jour, les anciens gabarits sont
REFUSÉS (GabaritInvalide) au lieu de donner des comparaisons fausses en silence. Il faudra alors réenrôler les utilisateurs.
"""
import hashlib

import numpy as np

from visage_id.erreurs import GabaritInvalide

MODELE_ID = "arcface_mobilefacenet_w600k_v1"
DIMENSION = 512

_MAGIC = b"VIDG"
_VERSION_FORMAT = 1
_EMPREINTE_MODELE = hashlib.sha256(MODELE_ID.encode()).digest()[:16]
_TAILLE = len(_MAGIC) + 1 + len(_EMPREINTE_MODELE) + DIMENSION * 4


def gabarit_en_octets(gabarit: np.ndarray) -> bytes:
    """Gabarit (512 float32, norme 1) -> octets à stocker."""
    verifie = valider_gabarit(gabarit)
    return _MAGIC + bytes([_VERSION_FORMAT]) + _EMPREINTE_MODELE + verifie.astype("<f4").tobytes()


def gabarit_depuis_octets(octets: bytes) -> np.ndarray:
    """Octets lus en base -> gabarit utilisable. Lève GabaritInvalide si quoi que ce soit cloche."""
    if not isinstance(octets, (bytes, bytearray, memoryview)) or len(octets) != _TAILLE:
        raise GabaritInvalide("Gabarit de taille inattendue.")
    donnees = bytes(octets)
    if donnees[:4] != _MAGIC:
        raise GabaritInvalide("Ce n'est pas un gabarit visage_id.")
    if donnees[4] != _VERSION_FORMAT:
        raise GabaritInvalide("Version de format de gabarit non prise en charge.")
    if donnees[5:21] != _EMPREINTE_MODELE:
        raise GabaritInvalide("Gabarit calculé avec un autre modèle : réenrôlement nécessaire.")
    return valider_gabarit(np.frombuffer(donnees[21:], dtype="<f4").astype(np.float32))


def valider_gabarit(vecteur: np.ndarray) -> np.ndarray:
    """Vérifie forme, valeurs finies et norme ; retourne le gabarit renormalisé. Lève GabaritInvalide."""
    vecteur = np.asarray(vecteur, dtype=np.float32)
    if vecteur.shape != (DIMENSION,):
        raise GabaritInvalide(f"Un gabarit a {DIMENSION} valeurs.")
    if not np.all(np.isfinite(vecteur)):
        raise GabaritInvalide("Gabarit contenant des valeurs invalides.")
    norme = float(np.linalg.norm(vecteur))
    if not 0.99 <= norme <= 1.01:
        raise GabaritInvalide("Gabarit non normalisé.")
    return vecteur / norme
