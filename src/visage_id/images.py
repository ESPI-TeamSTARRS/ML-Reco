"""Décodage sûr d'une image reçue du réseau (upload, capture envoyée par le navigateur...)."""
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from visage_id.erreurs import ImageInvalide

TAILLE_MAX_OCTETS = 10 * 1024 * 1024  # fichier reçu
COTE_MAX_PX = 4096  # largeur ou hauteur ; un visage de webcam n'en demande pas plus
FORMATS_ACCEPTES = {"JPEG", "PNG"}  # on refuse tout le reste : moins de décodeurs exposés


def decoder_image(octets: bytes, taille_max_octets: int = TAILLE_MAX_OCTETS, cote_max_px: int = COTE_MAX_PX) -> np.ndarray:
    """Octets d'un fichier JPEG/PNG -> tableau BGR uint8 (hauteur, largeur, 3).

    Les dimensions sont lues dans l'en-tête AVANT le décodage complet : une image minuscule en octets mais énorme en pixels
    (« bombe de décompression ») est refusée sans allouer sa mémoire.
    """
    if not isinstance(octets, (bytes, bytearray, memoryview)) or len(octets) == 0:
        raise ImageInvalide("Image vide.")
    if len(octets) > taille_max_octets:
        raise ImageInvalide(f"Image trop lourde (maximum {taille_max_octets} octets).")
    try:
        with Image.open(io.BytesIO(octets)) as entete:
            format_, (largeur, hauteur) = entete.format, entete.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as erreur:
        raise ImageInvalide("Fichier image illisible.") from erreur
    if format_ not in FORMATS_ACCEPTES:
        raise ImageInvalide("Format d'image non accepté (JPEG ou PNG uniquement).")
    if largeur > cote_max_px or hauteur > cote_max_px:
        raise ImageInvalide(f"Image trop grande (maximum {cote_max_px} px de côté).")

    image = cv2.imdecode(np.frombuffer(octets, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageInvalide("Fichier image illisible.")
    return image
