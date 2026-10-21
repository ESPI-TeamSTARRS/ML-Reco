import numpy as np
import pytest

from visage_id import Identification, Reconnaisseur, Visage, confiance


def test_confiance_croissante_et_bornee():
    valeurs = [confiance(s) for s in (-1.0, 0.0, 0.1, 0.23, 0.3, 0.5, 1.0)]
    assert valeurs == sorted(valeurs)
    assert 0.0 <= valeurs[0] < 0.01 and 0.99 < valeurs[-1] <= 1.0
    assert 0.5 < confiance(0.23) < 0.7  # au seuil par défaut : un peu plus d'une chance sur deux


def test_identification_expose_la_confiance():
    ident = Identification(Visage(0, 0, 10, 10, 0.9), "sam", 0.4)
    assert ident.confiance == pytest.approx(confiance(0.4))


def test_identifier_sans_visage_donne_une_liste_vide():
    assert Reconnaisseur().identifier(np.zeros((480, 640, 3), dtype=np.uint8), {"sam": (np.ones(512) / np.sqrt(512)).astype(np.float32)}) == []
