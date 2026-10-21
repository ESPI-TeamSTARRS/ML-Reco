import numpy as np
import pytest

from visage_id import AucunVisage, Reconnaisseur, similarite


@pytest.fixture(scope="module")
def reconnaisseur() -> Reconnaisseur:
    return Reconnaisseur()


def test_similarite_cosinus():
    a = np.array([1.0, 0.0])
    assert similarite(a, a) == pytest.approx(1.0)
    assert similarite(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert similarite(a, -a) == pytest.approx(-1.0)


def test_image_sans_visage_n_a_pas_d_empreinte(reconnaisseur):
    assert reconnaisseur.empreinte(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_verification_sans_visage_est_refusee(reconnaisseur):
    resultat = reconnaisseur.verifier(np.zeros((480, 640, 3), dtype=np.uint8), (np.ones(512) / np.sqrt(512)).astype(np.float32))
    assert (resultat.reconnu, resultat.visage_trouve, resultat.score) == (False, False, 0.0)


def test_gabarit_sans_aucun_visage_leve_une_erreur(reconnaisseur):
    with pytest.raises(AucunVisage):
        reconnaisseur.creer_gabarit([np.zeros((480, 640, 3), dtype=np.uint8)])


def test_modele_absent_leve_une_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError):
        Reconnaisseur(chemin_modele=tmp_path / "absent.onnx")


def test_verifier_photo_sans_visage_dans_la_reference_leve_une_erreur(reconnaisseur):
    vide = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(AucunVisage):
        reconnaisseur.verifier_photo(vide, vide)
