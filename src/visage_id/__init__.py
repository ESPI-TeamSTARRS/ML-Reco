"""Package d'inférence visage à intégrer dans le back-office."""
from importlib.metadata import PackageNotFoundError, version

from visage_id.detecteur import DetecteurVisage, Visage
from visage_id.erreurs import AucunVisage, GabaritInvalide, ImageInvalide
from visage_id.gabarit import MODELE_ID, gabarit_depuis_octets, gabarit_en_octets, valider_gabarit
from visage_id.images import decoder_image
from visage_id.reconnaissance import (
    SEUIL_PAR_DEFAUT,
    Identification,
    Reconnaisseur,
    Verification,
    confiance,
    similarite,
)

try:
    __version__ = version("visage-id")
except PackageNotFoundError:  # utilisé sans installation (copie du dossier source)
    __version__ = "0+inconnue"

__all__ = [
    "MODELE_ID", "SEUIL_PAR_DEFAUT", "AucunVisage", "DetecteurVisage", "GabaritInvalide", "Identification",
    "ImageInvalide", "Reconnaisseur", "Verification", "Visage", "confiance", "decoder_image",
    "gabarit_depuis_octets", "gabarit_en_octets", "similarite", "valider_gabarit",
]
