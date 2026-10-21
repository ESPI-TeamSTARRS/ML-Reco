"""Garde-fous pour l'usage en backend : images reçues du réseau, stockage des gabarits, accès concurrents."""
import io
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from visage_id import (
    GabaritInvalide,
    ImageInvalide,
    Reconnaisseur,
    Visage,
    decoder_image,
    gabarit_depuis_octets,
    gabarit_en_octets,
    valider_gabarit,
)


def _jpeg(largeur=200, hauteur=100) -> bytes:
    return cv2.imencode(".jpg", np.full((hauteur, largeur, 3), 120, np.uint8))[1].tobytes()


def _gabarit(graine=0) -> np.ndarray:
    v = np.random.default_rng(graine).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


# --- images reçues ---------------------------------------------------------------------------------------------------

def test_jpeg_et_png_sont_decodes_en_bgr():
    for extension in (".jpg", ".png"):
        octets = cv2.imencode(extension, np.zeros((50, 80, 3), np.uint8))[1].tobytes()
        assert decoder_image(octets).shape == (50, 80, 3)


@pytest.mark.parametrize("octets", [b"", b"pas une image", b"\xff\xd8\xff" + b"\x00" * 50])
def test_fichiers_illisibles_refuses(octets):
    with pytest.raises(ImageInvalide):
        decoder_image(octets)


def test_format_non_accepte_refuse():
    tampon = io.BytesIO()
    Image.new("RGB", (20, 20)).save(tampon, format="BMP")
    with pytest.raises(ImageInvalide, match="Format"):
        decoder_image(tampon.getvalue())


def test_image_trop_lourde_refusee_sans_decodage():
    with pytest.raises(ImageInvalide, match="lourde"):
        decoder_image(_jpeg(), taille_max_octets=100)


def test_image_trop_grande_en_pixels_refusee_avant_decodage():
    """5000x5 : minuscule en octets mais dépasse la limite de côté ; refusée sur l'en-tête."""
    tampon = io.BytesIO()
    Image.new("L", (5000, 5)).save(tampon, format="PNG")
    with pytest.raises(ImageInvalide, match="grande"):
        decoder_image(tampon.getvalue())


def test_une_image_envoyee_n_est_pas_un_tableau_numpy_mal_forme():
    reco = Reconnaisseur()
    with pytest.raises(ImageInvalide):
        reco.empreinte(np.zeros((10, 10), np.uint8))


# --- gabarits stockés ----------------------------------------------------------------------------------------------------

def test_gabarit_aller_retour_binaire():
    g = _gabarit()
    octets = gabarit_en_octets(g)
    assert len(octets) == 2069
    assert np.allclose(gabarit_depuis_octets(octets), g, atol=1e-6)


def test_gabarit_accepte_un_memoryview_comme_en_donne_la_base():
    g = _gabarit()
    assert np.allclose(gabarit_depuis_octets(memoryview(gabarit_en_octets(g))), g, atol=1e-6)


@pytest.mark.parametrize("abime", [
    lambda b: b[:-1],                       # tronqué
    lambda b: b"XXXX" + b[4:],              # mauvais en-tête
    lambda b: b[:4] + b"\x09" + b[5:],      # version de format inconnue
    lambda b: b[:5] + b"\x00" * 16 + b[21:],  # autre modèle
    lambda b: b[:21] + np.full(512, np.nan, "<f4").tobytes(),  # valeurs invalides
    lambda b: b[:21] + (np.ones(512, "<f4") * 5).tobytes(),    # non normalisé
])
def test_gabarit_abime_refuse(abime):
    with pytest.raises(GabaritInvalide):
        gabarit_depuis_octets(abime(gabarit_en_octets(_gabarit())))


def test_valider_gabarit_refuse_une_mauvaise_forme():
    with pytest.raises(GabaritInvalide):
        valider_gabarit(np.ones(128, np.float32))


def test_verifier_refuse_un_gabarit_invalide_avant_tout_calcul():
    with pytest.raises(GabaritInvalide):
        Reconnaisseur().verifier(np.zeros((100, 100, 3), np.uint8), np.ones(10, np.float32))


def test_seuil_hors_bornes_refuse():
    with pytest.raises(ValueError):
        Reconnaisseur(seuil=2.0)


# --- accès concurrents ------------------------------------------------------------------------------------------------------

def test_appels_concurrents_donnent_les_memes_resultats_que_sequentiels():
    reco = Reconnaisseur()
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (200, 200, 3), dtype=np.uint8) for _ in range(3)]
    visage = Visage(40, 40, 120, 120, 0.9, ((80.0, 80.0), (120.0, 80.0), (100.0, 100.0), (85.0, 125.0), (115.0, 125.0)))
    attendu = [reco._empreinte_du_visage(i, visage) for i in images]

    erreurs, resultats = [], []

    def travail():
        try:
            for _ in range(15):
                for image, ref in zip(images, attendu):
                    resultats.append(np.array_equal(reco._empreinte_du_visage(image, visage), ref))
        except Exception as e:  # noqa: BLE001 - le test doit tout remonter
            erreurs.append(e)

    fils = [threading.Thread(target=travail) for _ in range(8)]
    [f.start() for f in fils]
    [f.join() for f in fils]
    assert not erreurs
    assert len(resultats) == 8 * 15 * 3 and all(resultats)
