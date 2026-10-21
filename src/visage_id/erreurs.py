"""Erreurs du package. Toutes héritent de ValueError : le backend peut les attraper d'un bloc ou une par une.

Aucun message ne contient de donnée biométrique ; ils sont sûrs à journaliser.
"""


class ImageInvalide(ValueError):
    """Image absente, illisible, d'un format non accepté (JPEG/PNG seulement) ou trop grande."""


class AucunVisage(ValueError):
    """Aucun visage exploitable dans la ou les images fournies."""


class GabaritInvalide(ValueError):
    """Gabarit corrompu, d'une autre version du package, ou calculé avec un autre modèle."""
