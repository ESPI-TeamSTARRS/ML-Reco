import numpy as np
import pytest

from visage_id import DetecteurVisage


@pytest.fixture(scope="module")
def detecteur() -> DetecteurVisage:
    return DetecteurVisage()


def test_image_sans_visage_donne_liste_vide(detecteur):
    assert detecteur.detecter(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_image_de_mauvais_format_est_refusee(detecteur):
    with pytest.raises(ValueError):
        detecteur.detecter(np.zeros((480, 640), dtype=np.uint8))


def test_modele_absent_leve_une_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError):
        DetecteurVisage(tmp_path / "absent.onnx")
